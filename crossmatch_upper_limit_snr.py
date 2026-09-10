#!/usr/bin/env python3
"""Calculate the SNR implied by the Reiners & Basri upper limits for matched targets.

The RB catalogue stores non-detections as upper limits in equivalent width
(`ew_ha`) and in log(L_Ha/L_bol).  For these rows we use the RB limit as the
line signal and read the matching ETC S/N from the per-target output files.
This preserves the exact instrument model used to generate the ETC runs rather
than re-deriving a simplified noise estimate.
"""

from __future__ import annotations

import importlib.util
import os

import pandas as pd


def _import_halpha_flux():
    """Load 01_halpha_flux.py by path (its name isn't a valid identifier)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "01_halpha_flux.py")
    spec = importlib.util.spec_from_file_location("halpha_flux_01", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


halpha_line_flux = _import_halpha_flux().halpha_line_flux


def main() -> None:
    cross = pd.read_csv("crossmatch.csv")
    etc = pd.read_csv("out.csv")

    # collect_etc_snr writes one row per (target, exposure time); keep the
    # shortest exposure time available for each target as its canonical result.
    etc = etc[etc["name"].notna()].copy()
    etc["name"] = etc["name"].astype(str).str.strip()
    etc = etc.sort_values(["name", "etc_exptime_s"], ascending=[True, True])
    etc = etc.drop_duplicates(subset="name", keep="first").reset_index(drop=True)

    limit_rows = cross[(cross["ew_is_limit"].fillna(False)) | (cross["logratio_is_limit"].fillna(False))].copy()
    if limit_rows.empty:
        raise SystemExit("No RB upper-limit rows were found in crossmatch.csv")

    def limit_flux(row):
        if pd.notna(row.get("ew_ha")) and float(row["ew_ha"]) > 0:
            return float(halpha_line_flux(row["spt"], row["J"], ew=float(row["ew_ha"]), si=True))
        if pd.notna(row.get("log_lha_lbol")):
            return float(halpha_line_flux(row["spt"], row["J"], log_ratio=float(row["log_lha_lbol"]), si=True))
        return float("nan")

    limit_rows["upper_limit_flux_W_m2"] = limit_rows.apply(limit_flux, axis=1)
    limit_rows["upper_limit_ew_A"] = limit_rows["ew_ha"]
    limit_rows["upper_limit_log_Lha_Lbol"] = limit_rows["log_lha_lbol"]

    merged = limit_rows.merge(
        etc[["name", "int_snr", "int_snr_cc", "ew_used_A", "etc_folder", "etc_exptime_s"]],
        on="name",
        how="left",
        suffixes=("", "_etc"),
    )

    merged["snr_from_rb_upper_limit"] = merged["int_snr"]
    merged["snr_from_rb_upper_limit_cc"] = merged["int_snr_cc"]

    cols = [
        "name",
        "designation",
        "spt",
        "J",
        "ew_ha",
        "ew_is_limit",
        "log_lha_lbol",
        "logratio_is_limit",
        "upper_limit_flux_W_m2",
        "int_snr",
        "int_snr_cc",
        "ew_used_A",
        "etc_folder",
        "etc_exptime_s",
    ]
    merged = merged[cols]
    merged.to_csv("crossmatch_upper_limit_snr.csv", index=False)

    print(f"Wrote {len(merged)} upper-limit targets to crossmatch_upper_limit_snr.csv")
    print(merged[["name", "ew_ha", "int_snr", "etc_folder"]].to_string(index=False))


if __name__ == "__main__":
    main()
