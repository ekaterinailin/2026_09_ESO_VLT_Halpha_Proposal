#!/usr/bin/env python3
"""visibility_astroplan

Minimal description

Inputs
- targets: list of `astroplan.FixedTarget` or a catalogue with `ra_deg`/`dec_deg`
    (2MASS designations are parsed when present).
- observer/site: choose from `SITES` (default ``vlt``) providing lat/lon/height
- exptime_s: exposure time in seconds
- fixed_overhead_s: additive overhead per exposure in seconds (readout,
    acquisition, instrument, telescope). The required contiguous block per night
    is ``exptime_s + fixed_overhead_s``.
- constraints: `airmass` limit, `twilight` type, optional `moon_sep` (deg) and
    `moon_illum` (0-1).
- year (int) and step_min (sampling resolution in minutes).

Assumptions
- Site geometry and nominal airmass limit are taken from the `SITES` dict
    (default: VLT / UVES, Cerro Paranal).
- Overheads are additive per exposure and must be available as a single fixed
    number applied to each exposure (no multiplicative factor used).
- A usable night requires one uninterrupted contiguous block at least as long
    as ``exptime_s + fixed_overhead_s``; fragmented windows are not combined.
- Targets must have valid coordinates or parseable 2MASS names; partial/edge
    nights at the start/end of the year are ignored.

This module uses `astroplan` constraints (AirmassConstraint, AtNightConstraint,
MoonSeparationConstraint, MoonIlluminationConstraint) to build per-night masks
and computes contiguous baselines from those masks.
"""
from astropy.utils import iers
iers.conf.auto_download = False        # keep runs offline and reproducible
iers.conf.auto_max_age = None
import warnings                                                  # noqa: E402
import re
import numpy as np
import sys
import argparse

warnings.filterwarnings("ignore", message=".*IERS.*")
warnings.filterwarnings("ignore", message=".*polar motion.*")
# astroplan's Moon separation triggers an astropy note about the direction of
# the angular separation; it is informational and does not affect the result.
warnings.filterwarnings("ignore", message=".*Angular separation can depend.*")

import astropy.units as u                                        # noqa: E402
from astropy.coordinates import EarthLocation, SkyCoord          # noqa: E402
from astropy.time import Time                                    # noqa: E402
from astroplan import (AirmassConstraint, AtNightConstraint,     # noqa: E402
                       FixedTarget, MoonIlluminationConstraint,
                       MoonSeparationConstraint, Observer)

# ---------------------------------------------------------------------------
SITES = {
    "vlt": dict(
        name="VLT / UVES, Cerro Paranal (UT2 Kueyen)",
        lat=-24.6272, lon=-70.4045, height=2635.0,
        airmass_limit=2.0,
    ),
}
DEFAULT_SITE = "vlt"

TWILIGHT = {
    "astronomical": AtNightConstraint.twilight_astronomical,   # Sun < -18 deg
    "nautical": AtNightConstraint.twilight_nautical,           # Sun < -12 deg
    "civil": AtNightConstraint.twilight_civil,                 # Sun <  -6 deg
}


def parse_angle(value, is_ra: bool) -> float:
    """Degrees from a float or sexagesimal string. RA in hours, Dec in degrees."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    try:
        return float(s)
    except ValueError:
        pass
    parts = [p for p in re.split(r"[:\s hdms'\"]+", s) if p]
    sign = -1.0 if parts[0].startswith("-") else 1.0
    nums = [abs(float(p)) for p in parts[:3]] + [0.0, 0.0]
    val = nums[0] + nums[1] / 60.0 + nums[2] / 3600.0
    return sign * val * (15.0 if is_ra else 1.0)


def twomass_to_radec(designation):
    """RA, Dec in degrees from a 2MASS designation, or NaN."""
    s = re.sub(r"\s+", "", str(designation)).upper()
    s = re.sub(r"^2MASS", "", s)
    s = re.sub(r"^J", "", s)
    m = re.match(r"^(\d{2})(\d{2})(\d{2})(\d{2})([+-])(\d{2})(\d{2})(\d{2})(\d?)", s)
    if not m:
        return float("nan"), float("nan")
    hh, mm, ss, ff, sign, dd, dm, ds, df = m.groups()
    ra = (int(hh) + int(mm) / 60 + (int(ss) + int(ff) / 100) / 3600) * 15.0
    dec = int(dd) + int(dm) / 60 + (int(ds) + int(df or 0) / 10) / 3600
    return ra, (-dec if sign == "-" else dec)


# ---------------------------------------------------------------------------
def longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of True in a 1-D boolean array."""
    if not mask.any():
        return 0
    # Differences of the cumulative sum of False values group the True runs.
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return int((idx[1::2] - idx[0::2]).max())


def build_constraints(airmass: float, twilight: str,
                      moon_sep: float | None,
                      moon_illum: float | None) -> list:
    cons = [AirmassConstraint(max=airmass, boolean_constraint=True),
            TWILIGHT[twilight]()]
    if moon_sep is not None:
        cons.append(MoonSeparationConstraint(min=moon_sep * u.deg))
    if moon_illum is not None:
        cons.append(MoonIlluminationConstraint(max=moon_illum))
    return cons


def annual_observability(targets, observer: Observer, constraints: list,
                         exptime_s: float, fixed_overhead_s: float, year: int,
                         step_min: float, lon_deg: float,
                         chunk: int = 25) -> dict:
    """Per-night contiguous baseline and totals for a list of FixedTargets.

    ``fixed_overhead_s`` is an additive overhead in seconds applied to each
    exposure (readout, acquisition, instrument and telescope overheads).
    The required contiguous block per night is therefore
    (exptime_s + fixed_overhead_s) seconds converted to hours.
    """
    start = Time(f"{year}-01-01T00:00:00", scale="utc")
    n_steps = int(round(365 * 24 * 60 / step_min))
    times = start + np.arange(n_steps) * step_min * u.min

    # Label samples by local night, running local noon to local noon.
    night_id = np.floor(times.mjd + lon_deg / 360.0 + 0.5).astype(int)
    ids = np.unique(night_id)
    keep = np.ones(len(ids), bool)
    keep[0] = keep[-1] = False           # partial nights at the grid ends
    slices = [np.flatnonzero(night_id == nid) for nid in ids]

    step_h = step_min / 60.0
    need_h = (exptime_s + fixed_overhead_s) / 3600.0

    n_t = len(targets)
    baseline = np.zeros((n_t, len(ids)))
    window = np.zeros((n_t, len(ids)))

    # Evaluate constraints in target chunks to bound peak memory.
    for lo in range(0, n_t, chunk):
        sub = targets[lo:lo + chunk]
        masks = [c(observer, sub, times=times, grid_times_targets=True)
                 for c in constraints]
        ok = np.all(np.asarray(masks, dtype=bool), axis=0)     # (n_sub, n_time)
        for j, sl in enumerate(slices):
            block = ok[:, sl]
            window[lo:lo + len(sub), j] = block.sum(axis=1) * step_h
            for i in range(block.shape[0]):
                baseline[lo + i, j] = longest_run(block[i]) * step_h

    baseline, window = baseline[:, keep], window[:, keep]
    usable = baseline >= need_h

    # Cadence, which matters for a variability programme: how the usable
    # nights are distributed through the year, not just how many there are.
    first, last, gap = [], [], []
    for row in usable:
        idx = np.flatnonzero(row)
        if len(idx) == 0:
            first.append(np.nan); last.append(np.nan); gap.append(np.nan)
        else:
            first.append(int(idx[0])); last.append(int(idx[-1]))
            gap.append(int(np.diff(idx).max()) if len(idx) > 1 else np.nan)

    return dict(
        need_h=need_h, n_nights=int(keep.sum()),
        baseline=baseline, window=window, usable=usable,
        max_baseline_h=baseline.max(axis=1),
        median_baseline_h=np.array([np.median(r[r > 0]) if (r > 0).any() else 0.0
                                    for r in baseline]),
        nights_any=usable.shape[1] - (baseline == 0).sum(axis=1),
        nights_usable=usable.sum(axis=1),
        total_h=usable.sum(axis=1) * need_h,
        total_window_h=window.sum(axis=1),
        split_h=(window - baseline).sum(axis=1),
        first_night=np.array(first, dtype=float),
        last_night=np.array(last, dtype=float),
        max_gap_nights=np.array(gap, dtype=float),
    )


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Annual observability and usable-night count via astroplan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ra", default=None)
    p.add_argument("--dec", default=None)
    p.add_argument("--catalogue", default=None)
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--exptime", type=float, default=3600.0, help="seconds")
    p.add_argument("--overhead", type=float, default=1.2)
    p.add_argument("--site", default=DEFAULT_SITE, choices=sorted(SITES))
    p.add_argument("--airmass", type=float, default=None)
    p.add_argument("--twilight", default="astronomical", choices=sorted(TWILIGHT))
    p.add_argument("--moon-sep", type=float, default=None,
                   help="minimum Moon separation, degrees")
    p.add_argument("--moon-illum", type=float, default=None,
                   help="maximum Moon illuminated fraction, 0-1")
    # additive overhead components (seconds)
    p.add_argument('--tel-overhead', type=float, default=0.0,
                   help='standard telescope overhead in seconds')
    p.add_argument('--instr-overhead', type=float, default=0.0,
                   help='instrument setup overhead in seconds')
    p.add_argument('--acq-overhead', type=float, default=0.0,
                   help='acquisition overhead in seconds')
    p.add_argument('--readout-per-spec', type=float, default=0.0,
                   help='readout time per spectrum in seconds')
    p.add_argument('--n-spec', type=int, default=0,
                   help='number of spectra/readouts (default 0)')
    p.add_argument("--year", type=int, default=2027)
    p.add_argument("--step", type=float, default=30.0, help="sampling, minutes")
    args = p.parse_args(argv)

    site = SITES[args.site]
    airmass = args.airmass if args.airmass is not None else site["airmass_limit"]
    loc = EarthLocation(lat=site["lat"] * u.deg, lon=site["lon"] * u.deg,
                        height=site["height"] * u.m)
    observer = Observer(location=loc, name=site["name"])
    constraints = build_constraints(airmass, args.twilight,
                                    args.moon_sep, args.moon_illum)

    # --- assemble the target list ---
    names, ras, decs, cat = [], [], [], None
    if args.catalogue:
        import pandas as pd
        cat = pd.read_csv(args.catalogue)
        racol = next((c for c in ("ra_deg", "ra", "RA") if c in cat.columns), None)
        deccol = next((c for c in ("dec_deg", "dec", "DEC") if c in cat.columns), None)
        if racol is None or deccol is None:
            dcol = next((c for c in ("designation", "2MASS", "name")
                         if c in cat.columns), None)
            coords = [twomass_to_radec(v) for v in cat[dcol]] if dcol else []
            if not coords or all(not np.isfinite(c[0]) for c in coords):
                raise SystemExit(f"no RA/Dec and no parseable designation in "
                                 f"{args.catalogue}; columns: {list(cat.columns)}")
            cat["ra_deg"] = [c[0] for c in coords]
            cat["dec_deg"] = [c[1] for c in coords]
            racol, deccol = "ra_deg", "dec_deg"
            print(f"  coordinates parsed from '{dcol}'")
        ncol = next((c for c in ("name", "designation") if c in cat.columns), None)
        names = [str(v) for v in (cat[ncol] if ncol else range(len(cat)))]
        ras, decs = list(cat[racol]), list(cat[deccol])
    else:
        if args.ra is None or args.dec is None:
            p.error("give --ra and --dec, or --catalogue")
        names = ["target"]
        ras = [parse_angle(args.ra, True)]
        decs = [parse_angle(args.dec, False)]

    targets = [FixedTarget(coord=SkyCoord(r * u.deg, d * u.deg), name=n)
               for n, r, d in zip(names, ras, decs)]

    print("=" * 72)
    print("OBSERVABILITY (astroplan)")
    print("=" * 72)
    print(f"  site        : {site['name']}")
    print(f"  constraints : airmass < {airmass}, {args.twilight} twilight"
          + (f", Moon > {args.moon_sep} deg" if args.moon_sep else "")
          + (f", illumination < {args.moon_illum}" if args.moon_illum else ""))
    print(f"  year {args.year}, {args.step:.0f}-minute sampling, "
          f"{len(targets)} target(s)")
    fixed_overhead_s = (args.tel_overhead + args.instr_overhead +
                        args.acq_overhead + args.readout_per_spec * args.n_spec)
    effective_sec = args.exptime + fixed_overhead_s
    print(f"  requirement : {args.exptime:.0f} s + {fixed_overhead_s:.0f} s = "
          f"{effective_sec / 3600:.3f} h contiguous per night")

    r = annual_observability(targets, observer, constraints, args.exptime,
                             fixed_overhead_s, args.year, args.step,
                             site["lon"])

    if len(targets) == 1:
        i = 0
        print(f"\n  longest contiguous block : {r['max_baseline_h'][i]:.2f} h "
              f"on the best night")
        print(f"  median when observable   : {r['median_baseline_h'][i]:.2f} h")
        print(f"  nights target observable : {r['nights_any'][i]} of {r['n_nights']}")
        print(f"  nights with the baseline : {r['nights_usable'][i]}")
        if np.isfinite(r["max_gap_nights"][i]):
            # The observing season often straddles the calendar year, so the
            # first and last usable night indices are not informative on their
            # own. The largest gap between usable nights is: it is the length
            # of the out-of-season period.
            print(f"  longest gap between them : "
                  f"{r['max_gap_nights'][i]:.0f} night(s) out of season")
        print(f"\n  ANNUAL TOTAL             : {r['total_h'][i]:.1f} h")
        print(f"    full windows summed    : {r['total_window_h'][i]:.1f} h")
        if r["split_h"][i] > 0.05:
            print(f"    lost to split windows  : {r['split_h'][i]:.1f} h "
                  f"(observable but not contiguous)")
    else:
        import pandas as pd
        out = cat.reset_index(drop=True).copy()
        out["baseline_max_h"] = r["max_baseline_h"]
        out["baseline_median_h"] = r["median_baseline_h"]
        out["nights_observable"] = r["nights_any"]
        out["nights_with_baseline"] = r["nights_usable"]
        out["total_h"] = r["total_h"]
        out["total_window_h"] = r["total_window_h"]
        out["split_loss_h"] = r["split_h"]
        out["first_usable_night"] = r["first_night"]
        out["last_usable_night"] = r["last_night"]
        out["max_gap_nights"] = r["max_gap_nights"]
        dest = args.output or "visibility_astroplan.csv"
        out.to_csv(dest, index=False)

        n = out["nights_with_baseline"]
        print(f"\n  nights with the baseline : {n.min()} to {n.max()}, "
              f"median {n.median():.0f}")
        print(f"  never usable             : {(n == 0).sum()} target(s)")
        print(f"  annual total per target  : {out['total_h'].min():.1f} to "
              f"{out['total_h'].max():.1f} h")
        loss = out["split_loss_h"].sum()
        print(f"  lost to split windows    : {loss:.1f} h across the sample"
              + ("  (zero, as expected without a Moon constraint)"
                 if loss < 0.05 else ""))
        print(f"\n  written to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
