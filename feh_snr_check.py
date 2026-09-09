#!/usr/bin/env python3
"""Measure average S/N per pixel in a user-specified wavelength band.

This reuses the ETC-array logic from collect_etc_snr.py, but switches the
integration window to a different spectral regime such as the FeH band.

Example:
    python3 feh_snr_check.py etc_jobs/t8000s/2MASS_1506_13.out.json \
        --wave-lo-A 9895 --wave-hi-A 9980
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import re
import sys

import pandas as pd
import numpy as np


def _import_collect_etc_snr():
    """Load 04_collect_etc_snr.py by path.

    Its filename isn't a valid Python identifier (leading digit), so a plain
    `from collect_etc_snr import ...` can't find it; load it by file path.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "04_collect_etc_snr.py")
    spec = importlib.util.spec_from_file_location("collect_etc_snr_04", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


average_snr_per_pixel_in_band = _import_collect_etc_snr().average_snr_per_pixel_in_band

REDMIT_FACTOR = np.sqrt(2)  # factor to convert literature S/N to redMIT S/N


def find_files(patterns):
    files = []
    for pat in patterns:
        if os.path.isdir(pat):
            files.extend(sorted(glob.glob(os.path.join(pat, "*.out.json"))))
        else:
            files.extend(sorted(glob.glob(pat)))
    return [f for f in files if os.path.isfile(f)]


def slugify_name(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_").lower()


def target_key_from_path(path):
    name = os.path.basename(path)
    name = re.sub(r"\.out\.json$", "", name)
    name = re.sub(r"\.json$", "", name)
    return slugify_name(name)


def load_literature_values(path):
    if not os.path.isfile(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if "snr_feh" not in df.columns:
        return {}

    values = {}
    for _, row in df.iterrows():
        name = row.get("name")
        if pd.notna(name):
            values[slugify_name(name)] = {
                "snr": float(row["snr_feh"]) * REDMIT_FACTOR,
                "provenance": row.get("block_used", "unknown"),
            }
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the average S/N per pixel over a wavelength window using "
            "the same ETC spectrum arrays as the Halpha pipeline."
        )
    )
    parser.add_argument("files", nargs="*", help="ETC output JSON files or globs")
    parser.add_argument("--wave-lo-A", type=float, default=9895.0,
                        help="lower wavelength in Angstrom")
    parser.add_argument("--wave-hi-A", type=float, default=9980.0,
                        help="upper wavelength in Angstrom")
    parser.add_argument("--folder", default="etc_jobs_feh",
                        help="default folder to scan for *.out.json files")
    parser.add_argument("--csv", metavar="PATH",
                        help="write the comparison table to a CSV file; if omitted, print the CSV table to stdout")
    args = parser.parse_args()

    if not args.files:
        args.files = [os.path.join(args.folder, "*.out.json")]

    matches = find_files(args.files)
    if not matches:
        print("No FeH ETC output files matched the requested inputs.", file=sys.stderr)
        return 2

    lo_nm = args.wave_lo_A / 10.0
    hi_nm = args.wave_hi_A / 10.0
    literature = load_literature_values("reiners_basri_2008_joined.csv")

    print(f"FeH band check: {args.wave_lo_A:.0f}–{args.wave_hi_A:.0f} A = {lo_nm:.3f}–{hi_nm:.3f} nm")
    print("-" * 80)

    rows = []
    for path in matches:
        try:
            res = average_snr_per_pixel_in_band(path, lo_nm, hi_nm)
            target_key = target_key_from_path(path)
            lit_info = literature.get(target_key)
            lit_value = None if lit_info is None else lit_info["snr"]
            provenance = None if lit_info is None else lit_info["provenance"]
            delta = None if lit_value is None else res["mean_snr_per_pixel"] - lit_value
            rows.append({
                "name": os.path.basename(path),
                "mean": res["mean_snr_per_pixel"],
                "rms": res["rms_snr_per_pixel"],
                "p16": res["p16_snr_per_pixel"],
                "p84": res["p84_snr_per_pixel"],
                "n_pix": res["n_pix"],
                "literature": lit_value,
                "provenance": provenance,
                "delta": delta,
                "path": path,
            })
        except ValueError:
            pass

    if not rows:
        print("No ETC output in the FeH folder covers the requested FeH window.")
        print("This means the current JSONs are not FeH-band ETC runs or the wavelength window does not overlap the target spectra.")
        return 1

    literature_rows = [row for row in rows if row["literature"] is not None]
    for row in rows:
        lit = row["literature"]
        provenance = row["provenance"]
        if lit is None:
            print(f"{row['name']:32s} mean S/N pix^-1 = {row['mean']:8.3f}   RMS S/N pix^-1 = {row['rms']:8.3f}   n_pix = {row['n_pix']}   literature = N/A")
        else:
            print(f"{row['name']:32s} mean S/N pix^-1 = {row['mean']:8.3f}   RMS S/N pix^-1 = {row['rms']:8.3f}   n_pix = {row['n_pix']}   provenance = {provenance:<14s}   literature(redMIT) = {lit:8.3f}   delta = {row['delta']:8.3f}")

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df[
        ["name", "mean", "rms", "p16", "p84", "n_pix", "literature", "provenance", "delta"]
    ].copy()
    comparison_df = comparison_df.rename(columns={
        "name": "target",
        "mean": "measured_mean_snr_per_pixel",
        "rms": "measured_rms_snr_per_pixel",
        "p16": "measured_snr_p16",
        "p84": "measured_snr_p84",
        "n_pix": "n_pixels",
        "literature": "literature_redmit_snr_per_pixel",
        "provenance": "literature_provenance",
        "delta": "delta_measured_minus_lit",
    })
    comparison_df = comparison_df.sort_values("target").reset_index(drop=True)

    if literature_rows:
        measured = [row["mean"] for row in literature_rows]
        lit_vals = [row["literature"] for row in literature_rows]
        deltas = [row["delta"] for row in literature_rows]
        summary = {
            "n_files": len(rows),
            "n_with_literature": len(literature_rows),
            "mean_measured": sum(measured) / len(measured),
            "mean_literature": sum(lit_vals) / len(lit_vals),
            "mean_delta": sum(deltas) / len(deltas),
        }
        print("-" * 80)
        print("Summary")
        print(f"  processed files            : {summary['n_files']}")
        print(f"  matched literature rows   : {summary['n_with_literature']}")
        print(f"  mean measured S/N pix^-1  : {summary['mean_measured']:.3f}")
        print(f"  mean literature S/N pix^-1: {summary['mean_literature']:.3f}")
        print(f"  mean delta                : {summary['mean_delta']:+.3f}")

    csv_text = comparison_df.to_csv(index=False)
    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write(csv_text)
        print("-" * 80)
        print(f"CSV table written to {args.csv}")
    else:
        print("-" * 80)
        print("CSV comparison table")
        print(csv_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
