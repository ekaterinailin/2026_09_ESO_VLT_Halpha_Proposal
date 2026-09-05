#!/usr/bin/env python3
"""
make_etc_jobs.py
================

Take the catalogue produced by `add_halpha_flux` and stamp out one ESO
UVES ETC 2.0 input JSON per target, by patching a template downloaded from
the ETC web form.

    python3 make_etc_jobs.py rb08_with_fluxes.csv uves_etc_inputform.json \
            --outdir etc_jobs --track ratio --exptime 3600

What gets patched, and nothing else
-----------------------------------
    target.sed.emissionline.params.lambda   <- 656.28 nm (Halpha, air)
    target.sed.emissionline.params.fwhm     <- fwhm_nm from the catalogue
    target.brightness.flux                  <- F_line in W/m^2
    sky.airmass                             <- per target, or a fixed value
    sky.almanac.targetName / ra / dec       <- parsed from the 2MASS designation
    timesnr.DET2.WIN1.UIT1                  <- exposure time in seconds
    timesnr.SEQ.NEXPO                       <- number of exposures

Everything else in the template, including the order lists, readout mode and
slit width, is passed through untouched, so the jobs stay consistent with
whatever was configured in the web form.

Units, confirmed against the template
-------------------------------------
`lambda` is 656.3 in the template, so wavelengths are nm and `fwhm` is nm too.
`brightnesstype` is "flux" with a value of 1e-19, which is the integrated line
flux in W/m^2. Both match the catalogue columns directly. The script range-checks
both anyway and refuses values that look like they are in the wrong unit.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

LAMBDA_HA_NM = 656.280           # air
PARANAL_LAT = -24.6270           # degrees

# Sanity envelopes. These are not physics, they are unit tripwires.
FWHM_NM_RANGE = (1e-3, 5.0)      # 0.5 to 2300 km/s at Halpha
FLUX_WM2_RANGE = (1e-24, 1e-12)


# ---------------------------------------------------------------------------
def parse_2mass(designation: str):
    """RA and Dec in degrees from a 2MASS designation.

    '2MASS J03140344+1603056' -> (48.51433, 16.05156)

    The designation truncates rather than rounds, so positions are good to
    about 0.1 arcsec. That is irrelevant for the ETC, which uses the position
    only for the almanac, and only when almanacMode is enabled.
    """
    m = re.search(r"J?(\d{2})(\d{2})(\d{2})(\d{2})([+-])(\d{2})(\d{2})(\d{2})(\d?)",
                  str(designation).replace(" ", ""))
    if not m:
        raise ValueError(f"cannot parse coordinates from {designation!r}")
    hh, mm, ss, ff, sign, dd, dm, ds, df = m.groups()
    ra = (int(hh) + int(mm) / 60 + (int(ss) + int(ff) / 100) / 3600) * 15.0
    dec = int(dd) + int(dm) / 60 + (int(ds) + int(df or 0) / 10) / 3600
    if sign == "-":
        dec = -dec
    return ra, dec


def airmass_at_transit(dec_deg: float, lat_deg: float = PARANAL_LAT) -> float:
    """Minimum airmass reachable from a site, i.e. sec(z) at transit.

    Returns inf for targets that never rise above the horizon.
    """
    z = abs(dec_deg - lat_deg)
    if z >= 90.0:
        return float("inf")
    return 1.0 / math.cos(math.radians(z))


# ---------------------------------------------------------------------------
def patch_template(template: dict, *, flux_wm2: float, fwhm_nm: float,
                   airmass: float, exptime: float, nexpo: int,
                   name: str, ra: float | None, dec: float | None,
                   lambda_nm: float = LAMBDA_HA_NM,
                   target_snr: float | None = None,
                   snr_window: tuple[float, float] | None = None) -> dict:
    """Return a deep copy of the template with the per-target values set.

    Raises rather than writing a job whose SED type does not match what we are
    setting, because the ETC would otherwise silently ignore the emission-line
    parameters and return the S/N of whatever SED the template actually holds.
    """
    job = copy.deepcopy(template)

    sed = job["target"]["sed"]
    if sed.get("sedtype") != "emissionline":
        raise ValueError(
            f"template SED type is {sed.get('sedtype')!r}, not 'emissionline'. "
            f"Re-download the form with an emission-line SED selected, or the "
            f"line parameters set here would have no effect on the result.")
    bt = job["target"]["brightness"].get("brightnesstype")
    if bt != "flux":
        raise ValueError(
            f"template brightness type is {bt!r}, not 'flux'. The catalogue "
            f"supplies an integrated line flux, which only maps onto the "
            f"'flux' brightness type.")

    if not (FWHM_NM_RANGE[0] <= fwhm_nm <= FWHM_NM_RANGE[1]):
        raise ValueError(
            f"fwhm {fwhm_nm} is outside {FWHM_NM_RANGE} nm; check the units "
            f"(the template expects nm, not Angstrom or km/s)")
    if not (FLUX_WM2_RANGE[0] <= flux_wm2 <= FLUX_WM2_RANGE[1]):
        raise ValueError(
            f"flux {flux_wm2} is outside {FLUX_WM2_RANGE} W/m^2; check the "
            f"units (the template expects W/m^2, not erg/s/cm^2)")

    sed["emissionline"]["params"]["lambda"] = round(lambda_nm, 4)
    sed["emissionline"]["params"]["fwhm"] = round(float(fwhm_nm), 6)
    job["target"]["brightness"]["flux"] = float(flux_wm2)

    job["sky"]["airmass"] = round(float(airmass), 3)
    alm = job["sky"].get("almanac")
    if alm is not None:
        alm["targetName"] = str(name)
        if ra is not None:
            alm["ra"] = ra
        if dec is not None:
            alm["dec"] = dec

    ts = job["timesnr"]
    ts["DET2.WIN1.UIT1"] = float(exptime)

    # Two ETC modes share this block:
    #   fixed-exposure mode  has SEQ.NEXPO and returns S/N
    #   SNR-solve mode       has red_snr and returns the required NEXPO
    # Adding SEQ.NEXPO to a template in SNR-solve mode would over-constrain the
    # calculation, so the key is only written when it is already present.
    if "SEQ.NEXPO" in ts:
        ts["SEQ.NEXPO"] = int(nexpo)
    if target_snr is not None:
        if "red_snr" not in ts:
            raise ValueError(
                "--snr was given but the template has no 'red_snr' field, so "
                "it is not in SNR-solve mode. Re-download the form with a "
                "target S/N requested.")
        ts["red_snr"] = float(target_snr)
    if snr_window is not None:
        lo, hi = snr_window
        ts["red_wavelengthmin"] = float(lo)
        ts["red_wavelengthmax"] = float(hi)
    return job


# ---------------------------------------------------------------------------
def continuum_to_line_ratio(row, fwhm_nm: float) -> float:
    """Continuum flux inside the line band, divided by the line flux.

    The template's SED is an emission line on no continuum, so the ETC will
    not count photon noise from the target's own continuum. That is fine when
    the line dominates its own resolution element and wrong when it does not.
    For a Gaussian line extracted with matched weighting the relevant band is
    1.5056 * FWHM wide, so the ratio is

        continuum / line = 1.5056 * FWHM[A] / EW[A]

    Values above ~0.3 mean the omitted continuum noise is not negligible.
    """
    ew = row.get("ew_ha")
    if ew is None or not np.isfinite(ew) or ew <= 0:
        return float("nan")
    return 1.5056 * (fwhm_nm * 10.0) / ew


def verify_catalogue_units(cat: pd.DataFrame, flux_col: str) -> None:
    """Cross-check the catalogue's units against its own redundant columns.

    Range checks cannot catch a factor-of-ten error, because 0.7 is a legal
    value in both nm and Angstrom. Redundancy can. The catalogue carries the
    line width in two units and the flux ingredients alongside the flux, so
    both can be recomputed independently and compared.
    """
    problems = []

    # 1. fwhm_nm must equal fwhm_kms * lambda / c, exactly.
    if "fwhm_kms" in cat.columns:
        expect = LAMBDA_HA_NM * cat["fwhm_kms"] / 2.99792458e5
        rel = (cat["fwhm_nm"] - expect).abs() / expect
        if rel.max() > 1e-6:
            worst = cat.loc[rel.idxmax()]
            problems.append(
                f"fwhm_nm is inconsistent with fwhm_kms by a factor "
                f"{worst['fwhm_nm'] / (LAMBDA_HA_NM * worst['fwhm_kms'] / 2.99792458e5):.4g} "
                f"(worst: {worst.get('name')})")

    # 2. Recompute the flux from spt, J and the activity measure.
    try:
        from halpha_flux import halpha_line_flux
    except ImportError:
        return
    have = {"spt", "J"} <= set(cat.columns)
    src = "log_lha_lbol" if flux_col.endswith("ratio_W_m2") else "ew_ha"
    if have and src in cat.columns:
        rel_errs = []
        for _, r in cat.iterrows():
            if not (np.isfinite(r[flux_col]) and np.isfinite(r[src])):
                continue
            kw = {"log_ratio": r[src]} if src == "log_lha_lbol" else {"ew": r[src]}
            try:
                f = halpha_line_flux(r["spt"], r["J"], **kw)
            except ValueError:
                continue
            rel_errs.append(abs(f - r[flux_col]) / r[flux_col])
        if rel_errs and max(rel_errs) > 1e-6:
            problems.append(
                f"{flux_col} does not reproduce from spt/J/{src}; worst "
                f"relative difference {max(rel_errs):.3g}. Check the units "
                f"(W/m^2 vs erg/s/cm^2) or the column choice.")
        elif rel_errs:
            print(f"  flux column re-derived from spt/J/{src} for "
                  f"{len(rel_errs)} rows, max difference "
                  f"{max(rel_errs):.1e} -> units confirmed")

    if problems:
        raise SystemExit("catalogue unit check failed:\n  - "
                         + "\n  - ".join(problems))


def build_jobs(cat: pd.DataFrame, template: dict, outdir: str,
               track: str, exptime: float, nexpo: int,
               airmass_mode: str, fixed_airmass: float,
               max_airmass: float, detections_only: bool,
               target_snr: float | None = None,
               snr_window_fwhm: float | None = None) -> pd.DataFrame:
    os.makedirs(outdir, exist_ok=True)

    flux_col = {"ratio": "F_line_ratio_W_m2",
                "ew": "F_line_ew_W_m2"}[track]
    if flux_col not in cat.columns:
        raise SystemExit(f"column {flux_col!r} not in the catalogue. Run "
                         f"add_halpha_flux first, or pick the other --track.")
    if "fwhm_nm" not in cat.columns:
        raise SystemExit("column 'fwhm_nm' not in the catalogue")
    verify_catalogue_units(cat, flux_col)

    rows = []
    for _, r in cat.iterrows():
        name = str(r.get("name", r.get("designation", "target")))
        flux = r[flux_col]
        fwhm = r["fwhm_nm"]

        skip = None
        if not (np.isfinite(flux) and np.isfinite(fwhm)):
            skip = "flux or FWHM not calculated"
        elif detections_only and "is_detection" in cat.columns and not r["is_detection"]:
            skip = "upper limit, not a detection"

        ra = dec = np.nan
        am_min = np.nan
        desig = r.get("designation", name)
        try:
            ra, dec = parse_2mass(desig)
            am_min = airmass_at_transit(dec)
        except ValueError:
            pass

        if skip is None and np.isfinite(am_min) and am_min > max_airmass:
            skip = f"never below airmass {max_airmass} from Paranal (min {am_min:.2f})"
        elif skip is None and not np.isfinite(am_min):
            skip = "never rises at Paranal"

        if airmass_mode == "transit" and np.isfinite(am_min):
            airmass = max(am_min, 1.0)
        else:
            airmass = fixed_airmass

        rec = dict(name=name, designation=desig, ra_deg=ra, dec_deg=dec,
                   airmass_min=am_min, airmass_used=airmass,
                   flux_W_m2=flux, fwhm_nm=fwhm,
                   cont_over_line=continuum_to_line_ratio(r, fwhm)
                   if np.isfinite(fwhm) else np.nan,
                   json_file="", skipped=skip or "")

        if skip is None:
            safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
            path = os.path.join(outdir, f"{safe}.json")
            win = None
            if snr_window_fwhm:
                half = 0.5 * snr_window_fwhm * float(fwhm)
                win = (LAMBDA_HA_NM - half, LAMBDA_HA_NM + half)
                rec["snr_window_nm"] = f"{win[0]:.4f}-{win[1]:.4f}"
            job = patch_template(
                template, flux_wm2=float(flux), fwhm_nm=float(fwhm),
                airmass=airmass, exptime=exptime, nexpo=nexpo,
                name=name, ra=float(ra) if np.isfinite(ra) else None,
                dec=float(dec) if np.isfinite(dec) else None,
                target_snr=target_snr, snr_window=win)
            with open(path, "w") as fh:
                json.dump(job, fh, indent=2)
            rec["json_file"] = path

        rows.append(rec)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def write_driver(manifest: pd.DataFrame, outdir: str, etc_cli: str) -> str:
    """Emit a job list and a runner script.

    The runner is driven by jobs.txt rather than by generated per-file command
    lines, so it stays short whatever the sweep size, and it is restartable:
    a job whose output already exists and parses as a real ETC result is
    skipped.  That matters because a full sweep is hundreds of network calls
    and any of them can drop.
    """
    jobs = manifest[manifest["json_file"] != ""]
    root = os.path.abspath(outdir)

    listing = os.path.join(outdir, "jobs.txt")
    with open(listing, "w") as fh:
        for _, r in jobs.iterrows():
            fh.write(os.path.relpath(os.path.abspath(r["json_file"]), root) + "\n")

    path = os.path.join(outdir, "run_etc.sh")
    with open(path, "w") as fh:
        fh.write(RUNNER_TEMPLATE.replace("@ETC_CLI@", etc_cli))
    os.chmod(path, 0o755)
    return path


RUNNER_TEMPLATE = r"""#!/bin/sh
# Run the generated jobs through the ESO UVES ETC 2.0.
#
#   ./run_etc.sh                 run every job in jobs.txt, skipping finished ones
#   ./run_etc.sh -n              dry run: list what would be done
#   ./run_etc.sh -f              re-run jobs even if the output already exists
#   ./run_etc.sh -j 4            up to 4 concurrent calls
#   ./run_etc.sh -r 5            retry each job up to 5 times
#   ./run_etc.sh t3600s/X.json   run specific jobs only
#
# Results are written beside each input as <name>.out.json, so a sweep keeps
# its per-exposure-time folders and collect_etc_snr.py can be pointed at the
# top of the tree.
#
# Environment: ETC_CLI, SERVER, PYTHON override the defaults below.

set -u
cd "$(dirname "$0")" || exit 1

ETC_CLI="${ETC_CLI:-@ETC_CLI@}"
SERVER="${SERVER:-https://etc.eso.org}"
PYTHON="${PYTHON:-python3}"
RETRIES=3
PARALLEL=1
FORCE=0
DRYRUN=0

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# --- internal single-job mode, re-entered by xargs for parallel runs -------
if [ "${1:-}" = "--one" ]; then
    shift
    job="$1"; out="${job%.json}.out.json"
    n=0
    while [ "$n" -lt "${ETC_RETRIES:-3}" ]; do
        n=$((n + 1))
        if "$PYTHON" "$ETC_CLI" uves "$job" -s "$SERVER" -o "$out" \
                >/dev/null 2>"$out.err"; then
            if "$PYTHON" -c 'import json,sys
d=json.load(open(sys.argv[1]))
sp=d.get("data",{}).get("plots",{}).get("spectra")
sys.exit(0 if isinstance(sp,list) and sp else 1)' "$out" 2>/dev/null; then
                rm -f "$out.err"
                echo "ok   $job"
                exit 0
            fi
            echo "bad output (no spectra) on attempt $n" >>"$out.err"
        fi
        [ "$n" -lt "${ETC_RETRIES:-3}" ] && sleep $((n * 5))
    done
    rm -f "$out"
    echo "FAIL $job"
    exit 1
fi

# --- argument parsing -----------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        -f|--force)    FORCE=1; shift ;;
        -n|--dry-run)  DRYRUN=1; shift ;;
        -j)            PARALLEL="$2"; shift 2 ;;
        -r)            RETRIES="$2"; shift 2 ;;
        -h|--help)     usage 0 ;;
        --)            shift; break ;;
        -*)            echo "unknown option: $1" >&2; usage 1 ;;
        *)             break ;;
    esac
done

if [ ! -f "$ETC_CLI" ] && ! command -v "$ETC_CLI" >/dev/null 2>&1; then
    echo "etc_cli.py not found at '$ETC_CLI'." >&2
    echo "Download it from the Tools tab at https://etc.eso.org/uves," >&2
    echo "or set ETC_CLI=/path/to/etc_cli.py" >&2
    exit 2
fi

# --- build the work list --------------------------------------------------
if [ $# -gt 0 ]; then
    printf '%s\n' "$@" >jobs.run
elif [ -f jobs.txt ]; then
    cp jobs.txt jobs.run
else
    echo "no jobs.txt and no files given" >&2; exit 2
fi

total=$(wc -l <jobs.run | tr -d ' ')

# Decide in one Python call which outputs already exist and are real ETC
# results.  Doing this per job would start an interpreter hundreds of times
# on a full sweep.
: >jobs.todo
# The job list is passed as a path, not on stdin: "python -" already uses
# stdin for the program text, so redirecting the list there as well would
# leave the script reading an empty stdin and silently skipping every job.
"$PYTHON" - "$FORCE" jobs.run >jobs.todo <<'EOF'
import json, os, sys
force = sys.argv[1] == "1"
for line in open(sys.argv[2]).read().splitlines():
    job = line.strip()
    if not job:
        continue
    if not os.path.isfile(job):
        print(f"missing input: {job}", file=sys.stderr)
        continue
    out = job[:-5] + ".out.json" if job.endswith(".json") else job + ".out.json"
    done = False
    if not force and os.path.isfile(out) and os.path.getsize(out) > 0:
        try:
            sp = json.load(open(out)).get("data", {}).get("plots", {}) \
                .get("spectra")
            done = isinstance(sp, list) and bool(sp)
        except Exception:
            done = False
    if not done:
        print(job)
EOF
rm -f jobs.run
skipped=$(( total - $(wc -l <jobs.todo | tr -d ' ') ))

todo=$(wc -l <jobs.todo | tr -d ' ')
echo "jobs: $total total, $skipped already done, $todo to run"
echo "      server $SERVER, $PARALLEL at a time, up to $RETRIES attempts each"

if [ "$DRYRUN" -eq 1 ]; then
    [ "$todo" -gt 0 ] && cat jobs.todo
    rm -f jobs.todo
    exit 0
fi
if [ "$todo" -eq 0 ]; then
    rm -f jobs.todo
    echo "nothing to do"
    exit 0
fi

# --- run ------------------------------------------------------------------
ETC_RETRIES="$RETRIES"; export ETC_RETRIES ETC_CLI SERVER PYTHON
start=$(date +%s)
xargs -P "$PARALLEL" -n 1 -I '{}' "$0" --one '{}' <jobs.todo >jobs.log 2>&1
elapsed=$(( $(date +%s) - start ))

# grep -c prints 0 and exits 1 when there is no match, so "|| echo 0" would
# append a second line and break the numeric tests below.
ok=$(grep -c '^ok   ' jobs.log 2>/dev/null || true); ok=${ok:-0}
bad=$(grep -c '^FAIL ' jobs.log 2>/dev/null || true); bad=${bad:-0}
grep '^FAIL ' jobs.log 2>/dev/null | sed 's/^FAIL //' >failed.txt || true
rm -f jobs.todo

echo "done in ${elapsed}s: $ok succeeded, $bad failed"
if [ "$bad" -gt 0 ]; then
    echo "failed jobs listed in failed.txt; retry with:"
    echo "    ./run_etc.sh \$(tr '\n' ' ' <failed.txt)"
    echo "per-job stderr is in the matching .out.json.err files"
    exit 1
fi
rm -f failed.txt
echo "results are beside the inputs as *.out.json"
echo "next: python3 collect_etc_snr.py . --catalogue <catalogue.csv> --output out.csv"
"""


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Stamp out UVES ETC 2.0 input JSONs from a flux catalogue.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("catalogue", help="CSV from add_halpha_flux")
    p.add_argument("template", help="JSON downloaded from the UVES ETC web form")
    p.add_argument("--outdir", default="etc_jobs")
    p.add_argument("--track", choices=["ratio", "ew"], default="ratio",
                   help="which flux column to use; 'ratio' does not depend on chi")
    p.add_argument("--exptime", type=float, default=3600.0,
                   help="exposure time per job, seconds")
    p.add_argument("--nexpo", type=int, default=1)
    p.add_argument("--probe-times", type=str, default=None,
                   help="comma-separated exposure times; emits one job per "
                        "target per time, for fitting the S/N curve")
    p.add_argument("--airmass", choices=["transit", "fixed"], default="transit",
                   help="'transit' uses the minimum airmass reachable from "
                        "Paranal for each declination")
    p.add_argument("--fixed-airmass", type=float, default=1.5)
    p.add_argument("--max-airmass", type=float, default=2.0,
                   help="targets that never get below this are skipped")
    p.add_argument("--all-rows", action="store_true",
                   help="include upper limits as if they were detections")
    p.add_argument("--snr", type=float, default=None,
                   help="target S/N to solve NEXPO for; requires a template "
                        "downloaded in SNR-solve mode (with a red_snr field)")
    p.add_argument("--snr-window-fwhm", type=float, default=None,
                   help="set the red_snr wavelength window to this many line "
                        "FWHM, centred on Halpha, per target")
    p.add_argument("--etc-cli", default="etc_cli.py")
    args = p.parse_args(argv)

    cat = pd.read_csv(args.catalogue)
    with open(args.template) as fh:
        template = json.load(fh)

    times = ([float(t) for t in args.probe_times.split(",")]
             if args.probe_times else [args.exptime])

    manifests = []
    for t in times:
        sub = os.path.join(args.outdir, f"t{int(t)}s") if len(times) > 1 \
            else args.outdir
        m = build_jobs(cat, template, sub, args.track, t, args.nexpo,
                       args.airmass, args.fixed_airmass, args.max_airmass,
                       not args.all_rows, args.snr, args.snr_window_fwhm)
        m["exptime_s"] = t
        manifests.append(m)
    manifest = pd.concat(manifests, ignore_index=True)

    os.makedirs(args.outdir, exist_ok=True)
    mpath = os.path.join(args.outdir, "manifest.csv")
    manifest.to_csv(mpath, index=False)
    driver = write_driver(manifest, args.outdir, args.etc_cli)

    # ---- report -----------------------------------------------------------
    first = manifest[manifest["exptime_s"] == times[0]]
    made = first[first["json_file"] != ""]
    skipped = first[first["json_file"] == ""]

    slit = template.get("instrument", {}).get("INS.SLIT3.WID")
    read = template.get("instrument", {}).get("DET2.READ.SPEED")
    setting = template.get("instrument", {}).get("INS.REDEXP.MODE")

    print("=" * 74)
    print("TEMPLATE")
    print("=" * 74)
    print(f"  setting {setting}, slit {slit}\" -> R = {38700 / slit:,.0f}, "
          f"instrumental FWHM {LAMBDA_HA_NM / (38700 / slit):.4f} nm")
    print(f"  readout {read}")
    print(f"  SED type {template['target']['sed']['sedtype']}, "
          f"brightness {template['target']['brightness']['brightnesstype']}")
    if template.get("sky", {}).get("almanac") is None:
        print("  NOTE: this template has no sky.almanac block, so the ETC "
              "output will not carry")
        print("        a targetName. Pass --manifest "
              f"{os.path.join(args.outdir, 'manifest.csv')} to "
              "collect_etc_snr.py,")
        print("        or leave the generated filenames alone so targets can "
              "be identified from them.")
    ts = template.get("timesnr", {})
    if "red_snr" in ts:
        wlo, whi = ts.get("red_wavelengthmin"), ts.get("red_wavelengthmax")
        print(f"  mode: SNR-solve. The ETC returns the NEXPO needed to reach "
              f"red_snr = {ts['red_snr']} ({ts.get('red_method')})")
        print(f"        over {wlo}-{whi} nm, at a DIT of "
              f"{ts.get('DET2.WIN1.UIT1')} s.")
        print(f"        --exptime / --probe-times set the DIT, not the total "
              f"time.")
    else:
        print(f"  mode: fixed-exposure. The ETC returns S/N for "
              f"DET2.WIN1.UIT1 x SEQ.NEXPO.")

    print("\n" + "=" * 74)
    print("JOBS")
    print("=" * 74)
    print(f"  catalogue rows        : {len(cat)}")
    print(f"  jobs written          : {len(made)} per exposure time "
          f"x {len(times)} time(s) = {len(manifest[manifest['json_file'] != ''])}")
    print(f"  skipped               : {len(skipped)}")
    if len(skipped):
        counts = skipped["skipped"].str.replace(r"\(min .*\)", "", regex=True) \
            .str.replace(r"airmass [\d.]+ from Paranal", "airmass from Paranal",
                         regex=True).value_counts()
        for reason, n in counts.items():
            print(f"      {n:>3d}  {reason}")

    if len(made):
        print(f"\n  flux range            : {made['flux_W_m2'].min():.2e} to "
              f"{made['flux_W_m2'].max():.2e} W/m^2")
        print(f"  line FWHM range       : {made['fwhm_nm'].min():.4f} to "
              f"{made['fwhm_nm'].max():.4f} nm")
        print(f"  airmass used          : {made['airmass_used'].min():.2f} to "
              f"{made['airmass_used'].max():.2f}")

        bad = made[made["cont_over_line"] > 0.3]
        print(f"\n  targets where the omitted continuum matters "
              f"(cont/line > 0.3): {len(bad)} of {len(made)}")
        if len(bad):
            show = bad.nlargest(min(6, len(bad)), "cont_over_line")
            for _, r in show.iterrows():
                print(f"      {r['name']:<16s} cont/line = "
                      f"{r['cont_over_line']:.2f}")

    print(f"\n  manifest              : {mpath}")
    print(f"  driver                : {driver}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
