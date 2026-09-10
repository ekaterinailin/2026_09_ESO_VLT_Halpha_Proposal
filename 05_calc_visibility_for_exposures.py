#!/usr/bin/env python3
"""
05_calc_visibility_for_exposures.py
====================================

Given `out.csv` (collector output) compute whether each target can
achieve N exposures per visit, M visits per year (default N=1, M=18
visits of 3x6 exposures means N=18 total exposures spread as 3 per
visit across 6 visits — but the user requested 3 x 6 = 18 exposures
in a year; this script treats that as 18 exposures of the given DIT
and asks whether the annual usable-night count supports them).

The script uses the `visibility_astroplan.py` logic by calling it as a
library to compute per-target `usable_nights`, using each row's own
`etc_exptime_s` (written by collect_etc_snr) plus the given overhead --
there is no separate exposure time to pass on the command line, and a row
with no `etc_exptime_s` (no ETC result for that target) is left without a
usable_nights verdict rather than guessed at. It then checks if
usable_nights >= required_visits and reports pass/fail.

Output: CSV with columns from input plus `required_visits`, `usable_nights`,
`can_schedule` (bool) and `notes`.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from visibility_astroplan import annual_observability, build_constraints, SITES, DEFAULT_SITE
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroplan import Observer, FixedTarget
from astropy.time import Time


def compute_for_catalogue(catfile: str, outcsv: str, site: str,
                          overhead: float,
                          visits: int, exposures_per_visit: int,
                          year: int, step_min: float,
                          moon_sep: float | None, moon_illum: float | None,
                          twilight: str,
                          fixed_overhead_s: float = 0.0):
    cat = pd.read_csv(catfile)

    # total required exposures and visits
    total_exposures = visits * exposures_per_visit
    required_visits = visits

    # build observer and constraints
    site_key = site.lower()
    if site_key not in SITES:
        raise SystemExit(f"unknown site '{site}'; choices: {sorted(SITES)}")
    s = SITES[site_key]
    # Construct an EarthLocation and Observer
    from astropy.coordinates import EarthLocation
    loc = EarthLocation(lat=s['lat'] * u.deg, lon=s['lon'] * u.deg, height=s['height'] * u.m)
    observer = Observer(location=loc, name=s['name'])
    constraints = build_constraints(s.get('airmass_limit', 2.0), twilight, moon_sep, moon_illum)

    # assemble FixedTarget list from the ra_deg/dec_deg columns written by
    # collect_etc_snr (the 2MASS designation is resolved to coordinates
    # exactly once, upstream of this script; nothing here re-derives them).
    if 'ra_deg' not in cat.columns or 'dec_deg' not in cat.columns:
        raise SystemExit(
            f"'{catfile}' has no ra_deg/dec_deg columns; run collect_etc_snr "
            f"first so coordinates are resolved before visibility is computed")
    if 'name' not in cat.columns:
        raise SystemExit(f"'{catfile}' has no 'name' column")

    # collect_etc_snr now writes one row per (target, exposure time) probed,
    # so a target can appear several times with a different etc_exptime_s
    # each. The expensive part -- which nights a target clears the airmass/
    # twilight/Moon constraints at all -- depends only on its RA/Dec, not on
    # exposure time, so that (`baseline`, per target per night) is computed
    # once per unique target. Exposure time only sets the per-night
    # contiguous-block threshold applied on top of that, which is cheap, so
    # it is reapplied per row with that row's own exposure time.
    uniq = cat.drop_duplicates(subset='name')[['name', 'ra_deg', 'dec_deg']]
    coords = [SkyCoord(r * u.deg, d * u.deg)
             for r, d in zip(uniq['ra_deg'], uniq['dec_deg'])]
    targets = [FixedTarget(coord=c, name=n)
              for c, n in zip(coords, uniq['name'])]
    # The exptime_s argument here only affects fields of `r` this function
    # doesn't use (nights_usable and friends, computed from a single global
    # threshold); `baseline` -- the only field read below -- depends on
    # RA/Dec alone, so a placeholder is passed and each row's own exposure
    # time is applied afterward instead.
    r = annual_observability(targets, observer, constraints, 0.0,
                             float(fixed_overhead_s), year, step_min, s['lon'])
    baseline_by_name = dict(zip(uniq['name'], r['baseline']))

    # Exposure time is only known per row, from collect_etc_snr's own
    # etc_exptime_s (there is no global fallback: a row with no ETC result
    # has no exposure time to check visibility for).
    if 'etc_exptime_s' not in cat.columns:
        raise SystemExit(
            f"'{catfile}' has no etc_exptime_s column; run collect_etc_snr "
            f"first so each row carries the exposure time it was measured at")
    row_exptime = cat['etc_exptime_s'].astype(float)
    have_exptime = row_exptime.notna()
    if not have_exptime.any():
        raise SystemExit(f"'{catfile}' has no rows with an exposure time "
                         f"(etc_exptime_s is empty)")

    need_h = (row_exptime + fixed_overhead_s) / 3600.0
    usable_nights = pd.Series(np.nan, index=cat.index)
    usable_nights.loc[have_exptime] = [
        int((baseline_by_name[n] >= h).sum())
        for n, h in zip(cat.loc[have_exptime, 'name'], need_h[have_exptime])
    ]
    can_schedule = pd.Series(pd.NA, index=cat.index, dtype='boolean')
    can_schedule.loc[have_exptime] = usable_nights[have_exptime] >= required_visits

    notes = pd.Series('', index=cat.index)
    notes.loc[~have_exptime] = 'no exposure time (no ETC result for this target)'
    notes.loc[have_exptime & (usable_nights < required_visits)] = \
        'insufficient usable nights'

    cat['required_visits'] = required_visits
    cat['exposures_per_visit'] = exposures_per_visit
    cat['total_exposures_required'] = total_exposures
    cat['usable_nights'] = usable_nights
    cat['can_schedule'] = can_schedule
    cat['notes'] = notes
    cat['fixed_overhead_s'] = float(fixed_overhead_s)
    cat['effective_exptime_s'] = need_h * 3600.0

    cat.to_csv(outcsv, index=False)


def cli(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('input', help='collector output CSV (out.csv)')
    p.add_argument('--out', default=None,
                   help='where to write `input` plus the visibility columns '
                        '(default: update `input` in place)')
    p.add_argument('--site', default=DEFAULT_SITE)
    p.add_argument('--overhead', type=float, default=1.2)
    p.add_argument('--tel-overhead', type=float, default=600.0,
                   help='standard telescope overhead in seconds')
    p.add_argument('--instr-overhead', type=float, default=60.0,
                   help='instrument setup overhead in seconds')
    p.add_argument('--acq-overhead', type=float, default=120.0,
                   help='acquisition overhead in seconds')
    p.add_argument('--readout-per-spec', type=float, default=52.0,
                   help='readout time per spectrum in seconds')
    p.add_argument('--n-spec', type=int, default=1,
                   help='number of spectra/readouts (default 1)')
    p.add_argument('--visits', type=int, default=6, help='number of visits per year')
    p.add_argument('--per-visit', type=int, default=3, help='exposures per visit')
    p.add_argument('--year', type=int, default=2027)
    p.add_argument('--step', type=float, default=10.0)
    p.add_argument('--moon-sep', type=float, default=None)
    p.add_argument('--moon-illum', type=float, default=None)
    p.add_argument('--twilight', default='astronomical')
    args = p.parse_args(argv)

    fixed_overhead_s = (args.tel_overhead + args.instr_overhead +
                        args.acq_overhead + args.readout_per_spec * args.n_spec)

    compute_for_catalogue(args.input, args.out or args.input, args.site,
                          args.overhead, args.visits, args.per_visit,
                          args.year, args.step, args.moon_sep, args.moon_illum,
                          args.twilight, fixed_overhead_s=fixed_overhead_s)


if __name__ == '__main__':
    cli()
