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
library to compute per-target `nights_with_baseline` (usable nights)
for the given `exptime` and overhead. It then checks if nights_with_baseline
>= required_visits and reports pass/fail.

Output: CSV with columns from input plus `required_visits`, `usable_nights`,
`can_schedule` (bool) and `notes`.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from visibility_astroplan import annual_observability, build_constraints, SITES, DEFAULT_SITE
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroplan import Observer, FixedTarget
from astropy.time import Time


def compute_for_catalogue(catfile: str, outcsv: str, site: str,
                          exptime: float, overhead: float,
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
    names = list(cat['name']) if 'name' in cat.columns else list(cat.get('designation', []))
    coords = [SkyCoord(r * u.deg, d * u.deg) for r, d in zip(cat['ra_deg'], cat['dec_deg'])]
    targets = [FixedTarget(coord=c, name=n) for c, n in zip(coords, names)]

    # compute annual observability (this returns arrays aligned with targets)
    # Pass exptime and additive fixed_overhead_s (seconds) to annual_observability
    effective_exptime = float(exptime) + float(fixed_overhead_s)
    r = annual_observability(targets, observer, constraints, float(exptime), float(fixed_overhead_s), year, step_min, s['lon'])

    # load the by-time table and merge/add columns keyed on `name`
    bytimefile = outcsv
    if not os.path.exists(bytimefile):
        raise SystemExit(f"by-time file '{bytimefile}' not found")
    by = pd.read_csv(bytimefile)

    # Build a small summary DataFrame from `cat` aligned with `r`
    summary = pd.DataFrame({
        'name': names,
        'required_visits': required_visits,
        'exposures_per_visit': exposures_per_visit,
        'total_exposures_required': total_exposures,
        'usable_nights': r['nights_usable'],
        'can_schedule': r['nights_usable'] >= required_visits,
        'notes': ['' if x else 'insufficient usable nights' for x in (r['nights_usable'] >= required_visits)],
        'fixed_overhead_s': float(fixed_overhead_s),
        'effective_exptime_s': float(effective_exptime),
    })

    # merge: there may be multiple by-time rows per target; use left join on name
    merged = by.merge(summary, on='name', how='left')

    # overwrite the by-time file
    merged.to_csv(bytimefile, index=False)


def cli(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('input', help='collector output CSV (out.csv)')
    p.add_argument('--out', default='out_bytime.csv',
                   help='by-time CSV to update (default: out_bytime.csv)')
    p.add_argument('--site', default=DEFAULT_SITE)
    p.add_argument('--exptime', type=float, default=3600.0, help='DIT seconds')
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

    compute_for_catalogue(args.input, args.out, args.site, args.exptime,
                          args.overhead, args.visits, args.per_visit,
                          args.year, args.step, args.moon_sep, args.moon_illum,
                          args.twilight, fixed_overhead_s=fixed_overhead_s)


if __name__ == '__main__':
    cli()
