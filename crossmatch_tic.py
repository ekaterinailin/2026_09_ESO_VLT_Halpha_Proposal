#!/usr/bin/env python3
"""
crossmatch_tic.py
=================

Cross-match a catalogue keyed by 2MASS designation (Reiners & Basri 2008)
against a table keyed by TIC ID (the Petrucci TESS results), using astroquery.

    python3 crossmatch_tic.py reiners_basri_2008_table4.csv petrucci_results.csv \\
            -o crossmatch.csv --tic-cache tic_cache.csv

WHY NOT A CONE SEARCH
---------------------
The obvious approach, turning both sides into coordinates and matching within a
few arcseconds, is the wrong default here for two reasons.

First, it is unnecessary.  The TIC already carries the 2MASS cross-identification
in its TWOMASS column, so the match is an exact identifier join once the TIC rows
have been retrieved.  That is both cheaper and unambiguous.

Second, done naively it is wrong.  These are nearby ultracool dwarfs with large
proper motions -- several in the Reiners & Basri sample exceed 1 arcsec/yr.  A
2MASS designation encodes the position at the 2MASS observation epoch, around
1999, while TIC positions derive from Gaia.  Sixteen years at 1 arcsec/yr is a
16 arcsec offset, so a 3 arcsec cone search would miss exactly the nearby, fast
objects the sample is made of.  The positional route here therefore propagates
TIC positions with their own proper motions before matching, and reports the
separation both with and without that correction so the epoch assumption can be
checked against the data rather than trusted.

ROUTES
------
  id        exact join on the TIC TWOMASS column.  Primary.
  position  proper-motion-propagated cone match.  Independent check, and a
            fallback for TIC rows whose TWOMASS field is empty.
  simbad    optional, resolves each 2MASS name through SIMBAD and reads any TIC
            identifier it carries.  Catches objects the TIC did not link.

The routes are run independently and compared, so a disagreement shows up
instead of one route silently overriding the other.

NETWORK
-------
The MAST query is cached to CSV.  Run once with --tic-cache to fetch, then
re-run with --no-query to work offline.  If astroquery cannot reach MAST the
script says so and exits rather than falling back to an untrustworthy match.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

# 2MASS observations of this sample fall between 1997 and 2001; the designation
# encodes the position at the observation epoch, not at J2000.
EPOCH_2MASS_DEFAULT = 1999.3
# TIC v8 reports ra/dec in ICRS at epoch 2000.0, with pmRA/pmDEC alongside.
EPOCH_TIC_DEFAULT = 2000.0

TIC_COLUMNS = ["ID", "ra", "dec", "pmRA", "pmDEC", "plx", "Tmag", "Jmag",
               "Vmag", "TWOMASS", "GAIA", "objType", "typeSrc"]


# ---------------------------------------------------------------------------
# 2MASS designations
# ---------------------------------------------------------------------------
def norm_2mass(text) -> str:
    """Reduce a 2MASS designation to the bare 'HHMMSSss+DDMMSSs' form.

    Handles '2MASS J03140344+1603056', 'J03140344+1603056', '03140344+1603056'
    and any spacing.  The TIC stores the designation without the '2MASS J'
    prefix, so both sides have to be reduced to the same form before joining.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = re.sub(r"\s+", "", str(text)).upper()
    s = re.sub(r"^2MASS", "", s)
    s = re.sub(r"^J", "", s)
    m = re.match(r"^(\d{8}[+-]\d{7})$", s)
    return m.group(1) if m else ""


def parse_2mass_coords(designation):
    """RA, Dec in degrees from a 2MASS designation, at the 2MASS epoch."""
    s = norm_2mass(designation)
    if not s:
        return np.nan, np.nan
    ra = (int(s[0:2]) + int(s[2:4]) / 60 + float(s[4:6] + "." + s[6:8]) / 3600) * 15.0
    sign = -1.0 if s[8] == "-" else 1.0
    dec = sign * (int(s[9:11]) + int(s[11:13]) / 60
                  + float(s[13:15] + "." + s[15:16]) / 3600)
    return ra, dec


def spt_numeric(spt) -> float:
    """M0 -> 0, M9 -> 9, L0 -> 10, L3.5 -> 13.5.  NaN if unparseable."""
    m = re.match(r"^\s*([MLT])\s*([0-9]*\.?[0-9]*)", str(spt).upper())
    if not m:
        return np.nan
    base = {"M": 0.0, "L": 10.0, "T": 20.0}[m.group(1)]
    try:
        return base + float(m.group(2) or 0)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# TIC retrieval
# ---------------------------------------------------------------------------
def fetch_tic(ids, cache: str | None, allow_query: bool) -> pd.DataFrame:
    """TIC rows for a list of IDs, from cache when available.

    A cached file is used as-is if it covers every requested ID; otherwise the
    missing ones are queried and the cache is extended.
    """
    ids = [int(i) for i in ids]
    have = pd.DataFrame()
    if cache and os.path.isfile(cache):
        have = pd.read_csv(cache)
        have["ID"] = have["ID"].astype("int64")
        print(f"  TIC cache            : {cache} ({len(have)} row(s))")

    missing = sorted(set(ids) - set(have["ID"]) if len(have) else set(ids))
    if missing and not allow_query:
        raise SystemExit(
            f"{len(missing)} TIC ID(s) are not in the cache and --no-query was "
            f"given.\n  Re-run without --no-query to fetch them from MAST.")

    if missing:
        try:
            from astroquery.mast import Catalogs
        except ImportError:
            raise SystemExit("astroquery is not installed:  pip install astroquery")
        print(f"  querying MAST for    : {len(missing)} TIC ID(s)")
        try:
            tab = Catalogs.query_criteria(catalog="Tic", ID=missing)
        except Exception as exc:
            raise SystemExit(
                f"the MAST query failed: {type(exc).__name__}: {exc}\n"
                f"  Without TIC rows there is no reliable way to match the two "
                f"catalogues, so this stops here rather than guessing.\n"
                f"  If this machine has no route to mast.stsci.edu, run the "
                f"query elsewhere and pass the result with --tic-cache.")
        got = tab.to_pandas()
        keep = [c for c in TIC_COLUMNS if c in got.columns]
        got = got[keep].copy()
        got["ID"] = got["ID"].astype("int64")
        have = pd.concat([have, got], ignore_index=True) if len(have) else got
        if cache:
            have.to_csv(cache, index=False)
            print(f"  cache written        : {cache}")

    out = have[have["ID"].isin(ids)].drop_duplicates(subset="ID")
    if len(out) < len(set(ids)):
        lost = sorted(set(ids) - set(out["ID"]))
        print(f"  WARNING: {len(lost)} TIC ID(s) returned no row: "
              f"{', '.join(map(str, lost[:6]))}"
              + (" ..." if len(lost) > 6 else ""))
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def match_by_id(rb: pd.DataFrame, tic: pd.DataFrame) -> pd.DataFrame:
    """Exact join on the normalised 2MASS designation."""
    if "TWOMASS" not in tic.columns:
        print("  note: the TIC rows carry no TWOMASS column, so the identifier "
              "route is unavailable")
        return pd.DataFrame(columns=["rb_index", "ID", "route"])
    left = rb[["_2mass_key"]].reset_index().rename(columns={"index": "rb_index"})
    right = tic.assign(_2mass_key=tic["TWOMASS"].map(norm_2mass))
    right = right[right["_2mass_key"] != ""]
    m = left.merge(right[["_2mass_key", "ID"]], on="_2mass_key", how="inner")
    m["route"] = "id"
    return m[["rb_index", "ID", "route"]]


def match_by_position(rb: pd.DataFrame, tic: pd.DataFrame, radius_arcsec: float,
                      epoch_2mass: float, epoch_tic: float):
    """Cone match after propagating TIC positions to the 2MASS epoch.

    Returns the matches plus a diagnostic frame comparing separations with and
    without the proper-motion correction, so the epoch assumption is testable.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    ok = rb["_ra_2mass"].notna() & rb["_dec_2mass"].notna()
    if not ok.any() or not len(tic):
        return pd.DataFrame(columns=["rb_index", "ID", "route", "sep_arcsec"]), \
            pd.DataFrame()

    src = rb[ok]
    c_rb = SkyCoord(src["_ra_2mass"].values * u.deg,
                    src["_dec_2mass"].values * u.deg)

    dt = epoch_2mass - epoch_tic                    # years, usually negative
    pmra = pd.to_numeric(tic.get("pmRA"), errors="coerce").fillna(0.0).values
    pmdec = pd.to_numeric(tic.get("pmDEC"), errors="coerce").fillna(0.0).values
    dec0 = tic["dec"].values
    # TIC proper motions are in mas/yr, so convert to degrees before applying:
    # mas/yr -> deg/yr is a factor 1/(1000 * 3600).  pmRA is mu_alpha*, i.e. it
    # already includes cos(dec), so it is divided by cos(dec) to become a
    # coordinate increment in RA.
    MAS_TO_DEG = 1.0 / (1000.0 * 3600.0)
    ra_prop = tic["ra"].values + (pmra * dt * MAS_TO_DEG) / np.cos(np.radians(dec0))
    dec_prop = dec0 + pmdec * dt * MAS_TO_DEG

    c_raw = SkyCoord(tic["ra"].values * u.deg, tic["dec"].values * u.deg)
    c_prop = SkyCoord(ra_prop * u.deg, dec_prop * u.deg)

    rows, diag = [], []
    for label, cat in (("with_pm", c_prop), ("no_pm", c_raw)):
        idx, sep, _ = c_rb.match_to_catalog_sky(cat)
        for i, (j, d) in enumerate(zip(idx, sep.arcsec)):
            diag.append(dict(rb_index=src.index[i], ID=int(tic["ID"].iloc[j]),
                             mode=label, sep_arcsec=float(d)))
            if label == "with_pm" and d <= radius_arcsec:
                rows.append(dict(rb_index=src.index[i],
                                 ID=int(tic["ID"].iloc[j]), route="position",
                                 sep_arcsec=float(d)))
    return pd.DataFrame(rows), pd.DataFrame(diag)


def match_by_simbad(rb: pd.DataFrame):
    """Resolve each 2MASS name through SIMBAD and read any TIC identifier."""
    try:
        from astroquery.simbad import Simbad
    except ImportError:
        raise SystemExit("astroquery is not installed:  pip install astroquery")
    sim = Simbad()
    rows = []
    for i, name in zip(rb.index, rb["designation"]):
        try:
            ids = sim.query_objectids(str(name))
        except Exception:
            ids = None
        if ids is None:
            continue
        col = ids.colnames[0]
        for val in ids[col]:
            m = re.match(r"^\s*TIC\s+(\d+)", str(val))
            if m:
                rows.append(dict(rb_index=i, ID=int(m.group(1)),
                                 route="simbad"))
                break
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Cross-match a 2MASS-keyed catalogue against a TIC-keyed table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("rb_catalogue", help="CSV with a 2MASS designation column")
    p.add_argument("tess_table", help="CSV with a ticid column")
    p.add_argument("-o", "--output", default="crossmatch.csv")
    p.add_argument("--tic-cache", default="tic_cache.csv")
    p.add_argument("--no-query", action="store_true",
                   help="use the cache only, never contact MAST")
    p.add_argument("--radius", type=float, default=5.0,
                   help="positional match radius, arcsec")
    p.add_argument("--epoch-2mass", type=float, default=EPOCH_2MASS_DEFAULT)
    p.add_argument("--epoch-tic", type=float, default=EPOCH_TIC_DEFAULT)
    p.add_argument("--use-simbad", action="store_true")
    p.add_argument("--designation-column", default=None)
    args = p.parse_args(argv)

    rb = pd.read_csv(args.rb_catalogue)
    tess = pd.read_csv(args.tess_table)

    dcol = args.designation_column
    if dcol is None:
        for c in ("designation", "2MASS", "twomass", "name"):
            if c in rb.columns:
                dcol = c
                break
    if dcol is None:
        raise SystemExit(f"no designation column found in {args.rb_catalogue}; "
                         f"columns: {list(rb.columns)}")
    if "ticid" not in tess.columns:
        raise SystemExit(f"no 'ticid' column in {args.tess_table}; "
                         f"columns: {list(tess.columns)}")

    rb["_2mass_key"] = rb[dcol].map(norm_2mass)
    coords = rb[dcol].map(parse_2mass_coords)
    rb["_ra_2mass"] = [c[0] for c in coords]
    rb["_dec_2mass"] = [c[1] for c in coords]

    bad = (rb["_2mass_key"] == "").sum()
    print("=" * 74)
    print("CROSS-MATCH")
    print("=" * 74)
    print(f"  {os.path.basename(args.rb_catalogue):<28s} {len(rb)} row(s), "
          f"designation column '{dcol}'"
          + (f", {bad} unparseable" if bad else ""))
    print(f"  {os.path.basename(args.tess_table):<28s} {len(tess)} row(s), "
          f"{tess['ticid'].nunique()} unique TIC ID(s)")

    # --- spectral types, before spending any network calls ----------------
    scol = "spt" if "spt" in rb.columns else None
    if scol and "SpT" in tess.columns:
        a = rb[scol].map(spt_numeric).dropna()
        b = tess["SpT"].map(spt_numeric).dropna()
        def lab(v):
            return f"M{v:.1f}" if v < 10 else f"L{v - 10:.1f}"
        print(f"\n  spectral types        : "
              f"{lab(a.min())}-{lab(a.max())} vs {lab(b.min())}-{lab(b.max())}")
        if b.max() < a.min() or a.max() < b.min():
            print(f"  NOTE: the two samples do not overlap in spectral type at "
                  f"all. Any match therefore")
            print(f"        implies the two papers classified the same object "
                  f"differently, which is")
            print(f"        common right at the M/L boundary but should be "
                  f"checked rather than assumed.")

    tic = fetch_tic(tess["ticid"].unique(), args.tic_cache, not args.no_query)
    print(f"  TIC rows available    : {len(tic)}")

    # --- routes ------------------------------------------------------------
    m_id = match_by_id(rb, tic)
    m_pos, diag = match_by_position(rb, tic, args.radius,
                                    args.epoch_2mass, args.epoch_tic)
    parts = [m_id, m_pos]
    if args.use_simbad:
        parts.append(match_by_simbad(rb))

    print(f"\n  matches by route:")
    for part in parts:
        if len(part):
            r = part["route"].iloc[0]
            print(f"    {r:<10s} {len(part)}")
        elif part is m_id:
            print(f"    {'id':<10s} 0")
    if not len(m_pos):
        print(f"    {'position':<10s} 0  (within {args.radius}\")")

    # Do the routes agree?
    if len(m_id) and len(m_pos):
        a = set(zip(m_id["rb_index"], m_id["ID"]))
        b = set(zip(m_pos["rb_index"], m_pos["ID"]))
        print(f"    both routes agree on {len(a & b)} pair(s); "
              f"{len(a - b)} id-only, {len(b - a)} position-only")

    # --- how much does proper motion matter here? -------------------------
    if len(diag):
        w = diag[diag["mode"] == "with_pm"].set_index("rb_index")["sep_arcsec"]
        n = diag[diag["mode"] == "no_pm"].set_index("rb_index")["sep_arcsec"]
        common = w.index.intersection(n.index)
        near = (w[common] < 60)
        if near.any():
            print(f"\n  nearest-neighbour separations under 60\": "
                  f"{int(near.sum())} object(s)")
            print(f"    median with proper motion : "
                  f"{w[common][near].median():.2f}\"")
            print(f"    median without            : "
                  f"{n[common][near].median():.2f}\"")

    matches = pd.concat([p for p in parts if len(p)], ignore_index=True) \
        if any(len(p) for p in parts) else pd.DataFrame(
            columns=["rb_index", "ID", "route"])

    if len(matches):
        matches = (matches.sort_values("route")
                   .groupby(["rb_index", "ID"], as_index=False)
                   .agg(routes=("route", lambda v: "+".join(sorted(set(v)))),
                        sep_arcsec=("sep_arcsec", "min")
                        if "sep_arcsec" in matches.columns else ("route", "size")))
        out = (rb.reset_index().rename(columns={"index": "rb_index"})
               .merge(matches, on="rb_index", how="inner")
               .merge(tess, left_on="ID", right_on="ticid", how="left",
                      suffixes=("", "_tess")))
    else:
        out = pd.DataFrame(columns=list(rb.columns) + ["ID", "routes"])

    out = out.drop(columns=[c for c in ("_2mass_key", "_ra_2mass", "_dec_2mass")
                            if c in out.columns])
    out.to_csv(args.output, index=False)
    print(f"\n  {len(out)} matched pair(s) written to {args.output}")
    if not len(out):
        print(f"  An empty result is a real answer here, not a failure: see the "
              f"spectral-type note above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
