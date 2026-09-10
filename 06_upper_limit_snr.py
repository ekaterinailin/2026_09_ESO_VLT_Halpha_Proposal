#!/usr/bin/env python3
"""
06_upper_limit_snr.py
=====================

S/N implied by the Reiners & Basri Halpha upper limits.

The RB catalogue stores non-detections as upper limits in equivalent width
(`ew_ha`) and in log(L_Ha/L_bol).  For those rows this takes the RB limit as
the line signal, computes the flux it implies, and reports the ETC S/N that
04_collect_etc_snr already measured for the same target -- i.e. whether UVES
would detect a line sitting right at the RB limit.

The only input is 04_collect_etc_snr's output (out.csv / out2x2.csv): it was
merged into the flux catalogue, so it already carries the RB columns
(`spt`, `J`, `ew_ha`, `ew_is_limit`, ...) alongside `int_snr`.  No separate
crossmatch file is needed.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import pandas as pd


def _import_halpha_flux():
    """Load 01_halpha_flux.py by path (its name isn't a valid identifier).

    Parameters:
      (none)

    Returns:
      the imported 01_halpha_flux module.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "01_halpha_flux.py")
    spec = importlib.util.spec_from_file_location("halpha_flux_01", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


halpha_line_flux = _import_halpha_flux().halpha_line_flux


def upper_limit_flux(row) -> float:
    """Halpha line flux implied by a row's RB upper limit, in W/m^2.

    Parameters:
      row -- a catalogue row carrying spt, J and an ew_ha or log_lha_lbol limit.

    Returns:
      the flux in W/m^2, from the EW limit if positive, else the ratio limit,
      else NaN (also NaN when the spectral type is outside the flux
      calibration range).
    """
    try:
        if pd.notna(row.get("ew_ha")) and float(row["ew_ha"]) > 0:
            return float(halpha_line_flux(row["spt"], row["J"],
                                          ew=float(row["ew_ha"]), si=True))
        if pd.notna(row.get("log_lha_lbol")):
            return float(halpha_line_flux(row["spt"], row["J"],
                                          log_ratio=float(row["log_lha_lbol"]),
                                          si=True))
    except ValueError:
        return float("nan")
    return float("nan")


def main(argv=None) -> int:
    """Filter the collector output to RB upper limits and add the implied flux.

    Parameters:
      argv -- command-line arguments (defaults to sys.argv).

    Returns:
      0 on success. Side effect: writes the filtered table to --output.
    """
    p = argparse.ArgumentParser(
        description="S/N implied by the Reiners & Basri Halpha upper limits.")
    p.add_argument("input", help="04_collect_etc_snr output CSV (out.csv)")
    p.add_argument("--output", default="upper_limit_snr.csv")
    args = p.parse_args(argv)

    df = pd.read_csv(args.input)
    for col in ("name", "ew_is_limit", "logratio_is_limit", "spt", "J"):
        if col not in df.columns:
            raise SystemExit(f"'{args.input}' has no '{col}' column; run "
                             f"04_collect_etc_snr with --catalogue first")

    # One row per target: the shortest exposure time that has a result. Rows
    # with no ETC result have NaN etc_exptime_s and sort last, so a target
    # that was never observed keeps its single NaN row.
    df = df[df["name"].notna()].copy()
    df["name"] = df["name"].astype(str).str.strip()
    df = (df.sort_values(["name", "etc_exptime_s"])
            .drop_duplicates(subset="name", keep="first")
            .reset_index(drop=True))

    limit = df[df["ew_is_limit"].fillna(False)
               | df["logratio_is_limit"].fillna(False)].copy()
    if limit.empty:
        raise SystemExit(f"no RB upper-limit rows in '{args.input}'")

    limit["upper_limit_flux_W_m2"] = limit.apply(upper_limit_flux, axis=1)

    cols = [c for c in (
        "name", "designation", "spt", "J", "ew_ha", "ew_is_limit",
        "log_lha_lbol", "logratio_is_limit", "upper_limit_flux_W_m2",
        "int_snr", "int_snr_cc", "ew_used_A", "etc_folder", "etc_exptime_s",
    ) if c in limit.columns]
    out = limit[cols]
    out.to_csv(args.output, index=False)

    n_seen = int(out["int_snr"].notna().sum()) if "int_snr" in out.columns else 0
    print(f"  {len(out)} RB upper-limit target(s), {n_seen} with an ETC S/N "
          f"-> {args.output}")
    if "int_snr" in out.columns:
        print(out[["name", "ew_ha", "int_snr"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
