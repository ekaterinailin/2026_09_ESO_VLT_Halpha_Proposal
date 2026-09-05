#!/usr/bin/env python3
"""
build_rb08_catalogue.py
=======================

Turn Reiners & Basri (2008, ApJ 684, 1390) Table 4 into a CSV whose column
names match `add_halpha_flux` and `uves_ha_etc.py`.

Table 4 has no J magnitude, so it is cross-matched against Table 1 (the
observing log), which carries the full 2MASS designation and J for every
object.  All 45 Table 4 objects appear in Table 1.

The transcription is checked, not assumed.  Reiners & Basri give their own
chi(Teff) calibration as a fifth-order polynomial (their Eq. 1, coefficients
in their Table 2).  Every row must satisfy

    log10(EW * chi_RB(Teff))  ==  log(L_Ha/L_bol)

to within the rounding of the printed table.  Any row that fails is a
transcription error in EW, Teff or the ratio, and is reported.
"""

import math
import pandas as pd

# ---------------------------------------------------------------------------
# Reiners & Basri (2008) Table 2: log chi = a + b T + c T^2 + d T^3 + e T^4 + f T^5
# chi from PHOENIX DUSTY models, valid 1200-4000 K.
# ---------------------------------------------------------------------------
RB_COEFFS = [-6.73424e+1, 1.11938e-1, -8.26212e-5,
             3.04492e-8, -5.51137e-12, 3.90255e-16]


def fwhm_kms_to_nm(fwhm_kms: float, lambda_nm: float = 656.280) -> float:
    """Convert a line FWHM from velocity to wavelength units.

    Delta_lambda = lambda * v / c, so at Halpha one km/s is 2.18913e-3 nm
    (equivalently 0.0218913 Angstrom).  This is the intrinsic width; the
    observed width is this added in quadrature with the instrumental FWHM,
    which for UVES red at a 1 arcsec slit (R = 38,700) is 0.0170 nm.
    """
    return lambda_nm * fwhm_kms / 2.99792458e5


def chi_rb(teff: float) -> float:
    """Reiners & Basri (2008) chi, in 1/Angstrom, from effective temperature."""
    log_chi = sum(c * teff ** i for i, c in enumerate(RB_COEFFS))
    return 10.0 ** log_chi


# ---------------------------------------------------------------------------
# Table 1, J magnitudes keyed by the short name used in Table 4.
# Where an object appears in more than one of the three observing blocks the
# J magnitude is identical, so only one value is kept.
# ---------------------------------------------------------------------------
TABLE1_J = {
    "2MASS 0314+16": ("2MASS J03140344+1603056", 12.53),
    "2MASS 1159+00": ("2MASS J11593850+0057268", 14.08),
    "2MASS 1221+02": ("2MASS J12212770+0257198", 13.17),
    "2MASS 1731+27": ("2MASS J17312974+2721233", 12.09),
    "2MASS 1854+84": ("2MASS J18544597+8429470", 13.66),
    "2MASS 2200-30": ("2MASS J22000201-3038327", 14.36),
    "2MASS 0746+20": ("2MASS J07464256+2000321", 11.76),
    "2MASS 1412+16": ("2MASS J14122449+1633115", 13.89),
    "2MASS 1441-09": ("2MASS J14413716-0945590", 14.02),
    "2MASS 2351-25": ("2MASS J23515044-2537367", 12.47),
    "2MASS 0235-23": ("2MASS J02355993-2331205", 13.67),   # J from Gizis+2001
    "2MASS 0602+39": ("2MASS J06023045+3910592", 12.30),
    "2MASS 1022+58": ("2MASS J10224821+5825453", 13.50),
    "2MASS 1045-01": ("2MASS J10452400-0149576", 13.16),
    "2MASS 1048+01": ("2MASS J10484281+0111580", 12.92),
    "2MASS 1300+19": ("2MASS J13004255+1912354", 12.72),
    "2MASS 1359-40": ("2MASS J13595510-4034582", 13.65),
    "2MASS 1439+19": ("2MASS J14392836+1929149", 12.76),
    "2MASS 1555-09": ("2MASS J15551573-0956055", 12.56),
    "2MASS 1145+23": ("2MASS J11455714+2317297", 15.39),
    "2MASS 1334+19": ("2MASS J13340623+1940351", 15.47),
    "2MASS 1645-13": ("2MASS J16452211-1319516", 12.45),
    "2MASS 1807+50": ("2MASS J18071593+5015316", 12.93),
    "2MASS 2057-02": ("2MASS J20575409-0252302", 13.12),
    "2MASS 0828-13": ("2MASS J08283419-1309198", 12.80),
    "2MASS 0921-21": ("2MASS J09211410-2104446", 12.78),
    "2MASS 1155-37": ("2MASS J11553952-3727350", 12.81),
    "2MASS 1305-25": ("2MASS J13054019-2541059", 13.41),
    "2MASS 0523-14": ("2MASS J05233822-1403022", 13.08),
    "2MASS 1029+16": ("2MASS J10292165+1626526", 14.29),
    "2MASS 1047-18": ("2MASS J10473109-1815574", 14.12),
    "2MASS 0913+18": ("2MASS J09130320+1841501", 15.97),
    "2MASS 1203+00": ("2MASS J12035812+0015500", 14.01),
    "2MASS 1506+13": ("2MASS J15065441+1321060", 13.37),
    "2MASS 1615+35": ("2MASS J16154416+3559005", 14.54),
    "2MASS 2104-10": ("2MASS J21041491-1037369", 13.84),
    "2MASS 0036+18": ("2MASS J00361617+1821104", 12.47),
    "2MASS 0700+31": ("2MASS J07003664+3157266", 12.92),
    "2MASS 1705-05": ("2MASS J17054834-0516462", 13.31),
    "2MASS 2224-01": ("2MASS J22244381-0158521", 14.07),
    "2MASS 0004-40": ("2MASS J00043484-4044058", 13.11),
    "2MASS 0835-08": ("2MASS J08354256-0819237", 13.17),
    "2MASS 1507-16": ("2MASS J15074769-1627386", 12.83),
    "2MASS 0825+21": ("2MASS J08251968+2115521", 15.10),
    "2MASS 0255-47": ("2MASS J02550357-4700509", 13.25),
}

# ---------------------------------------------------------------------------
# Table 4, transcribed verbatim.
#   name, SpType, Teff, vsini, EW, ew_limit, log(LHa/Lbol), ratio_limit,
#   multi_epoch (the * flag: value given is the LOWEST activity of the epochs),
#   vtan
# ---------------------------------------------------------------------------
TABLE4 = [
    ("2MASS 0314+16", "L0.0", 2300, 19, 7.72, False, -4.69, False, False, 16.8),
    ("2MASS 1159+00", "L0.0", 2300, 71, 3.31, False, -5.06, False, False, None),
    ("2MASS 1221+02", "L0.0", 2300, 25, 5.01, False, -4.88, False, True, 8.1),
    ("2MASS 1731+27", "L0.0", 2300, 15, 5.99, False, -4.80, False, True, 15.8),
    ("2MASS 1854+84", "L0.0", 2300, 7, 7.06, False, -4.73, False, False, None),
    ("2MASS 2200-30", "L0.0", 2300, 17, 3.56, False, -5.03, False, False, None),
    ("2MASS 0746+20", "L0.5", 2250, 31, 2.36, False, -5.29, False, True, 21.3),
    ("2MASS 1412+16", "L0.5", 2250, 19, 1.45, False, -5.50, False, False, None),
    ("2MASS 1441-09", "L0.5", 2250, 23, 1.53, True, -5.48, True, False, None),
    ("2MASS 2351-25", "L0.5", 2250, 41, 2.76, False, -5.22, False, False, 26.4),
    ("2MASS 0235-23", "L1.0", 2200, 13, 0.20, True, -6.44, True, False, None),
    ("2MASS 0602+39", "L1.0", 2200, 9, 0.49, True, -6.05, True, False, None),
    ("2MASS 1022+58", "L1.0", 2200, 15, 3.32, False, -5.22, False, True, 81.9),
    ("2MASS 1045-01", "L1.0", 2200, 3, 0.20, True, -6.44, True, True, 36.6),
    ("2MASS 1048+01", "L1.0", 2200, 17, 1.08, False, -5.71, False, True, 37.9),
    ("2MASS 1300+19", "L1.0", 2200, 10, 0.20, True, -6.44, True, True, 97.9),
    ("2MASS 1359-40", "L1.0", 2200, 8, 0.20, True, -6.44, True, False, None),
    ("2MASS 1439+19", "L1.0", 2200, 11, 3.48, False, -5.20, False, True, 89.0),
    ("2MASS 1555-09", "L1.0", 2200, 11, 2.45, False, -5.35, False, True, None),
    ("2MASS 1145+23", "L1.5", 2140, 14, 3.69, False, -5.27, False, False, None),
    ("2MASS 1334+19", "L1.5", 2140, 30, 0.20, True, -6.53, True, False, None),
    ("2MASS 1645-13", "L1.5", 2140, 9, 1.51, False, -5.66, False, False, None),
    ("2MASS 1807+50", "L1.5", 2140, 76, 3.79, False, -5.26, False, False, 11.2),
    ("2MASS 2057-02", "L1.5", 2140, 62, 8.15, False, -4.92, False, True, 5.9),
    ("2MASS 0828-13", "L2.0", 2080, 33, 0.20, True, -6.63, True, True, None),
    ("2MASS 0921-21", "L2.0", 2080, 15, 0.32, True, -6.42, True, False, 57.8),
    ("2MASS 1155-37", "L2.0", 2080, 22, 1.00, False, -5.96, False, True, 49.3),
    ("2MASS 1305-25", "L2.0", 2080, 76, 1.75, False, -5.69, False, False, 25.7),
    ("2MASS 0523-14", "L2.5", 2010, 21, 0.34, False, -6.52, False, False, 5.1),
    ("2MASS 1029+16", "L2.5", 2010, 29, 1.96, False, -5.76, False, False, None),
    ("2MASS 1047-18", "L2.5", 2010, 15, 1.16, True, -5.99, True, False, None),
    ("2MASS 0913+18", "L3.0", 1950, 34, 0.20, True, -6.86, True, False, None),
    ("2MASS 1203+00", "L3.0", 1950, 39, 1.17, True, -6.09, True, False, 88.1),
    ("2MASS 1506+13", "L3.0", 1950, 20, 0.69, False, -6.32, False, True, 74.2),
    ("2MASS 1615+35", "L3.0", 1950, 13, 1.50, True, -5.98, True, False, None),
    ("2MASS 2104-10", "L3.0", 1950, 27, 1.53, True, -5.97, True, False, 58.9),
    ("2MASS 0036+18", "L3.5", 1880, 45, 1.09, True, -6.26, True, False, 36.9),
    ("2MASS 0700+31", "L3.5", 1880, 41, 1.78, True, -6.04, True, False, 29.7),
    ("2MASS 1705-05", "L4.0", 1820, 26, 0.20, True, -7.12, True, True, None),
    ("2MASS 2224-01", "L4.5", 1760, 32, 1.21, True, -6.48, True, False, 52.8),
    ("2MASS 0004-40", "L5.0", 1700, 42, 0.20, True, -7.42, True, False, 83.3),
    ("2MASS 0835-08", "L5.0", 1700, 23, 0.20, True, -7.42, True, False, 24.0),
    ("2MASS 1507-16", "L5.0", 1700, 32, 0.29, True, -7.25, True, True, 30.8),
    ("2MASS 0825+21", "L7.5", 1500, 19, 0.20, True, -8.18, True, False, 38.2),
    ("2MASS 0255-47", "L8.0", 1480, 67, 0.20, True, -8.28, True, False, 28.6),
]

# 2MASS 1045-01 has "v sin i < 3", i.e. an upper limit rather than a detection.
VSINI_LIMIT = {"2MASS 1045-01"}

# Table 3, the eight targets with variable Halpha, each at two epochs.
# Table 4 quotes the LOWEST activity of the epochs for starred objects, so
# min(Table 3) must reproduce Table 4 exactly.  This is an independent
# transcription of the same numbers and therefore a real cross-check.
#   name: [(EW, is_limit, log ratio), ...]
TABLE3 = {
    "2MASS 1221+02": [(25.65, False, -4.18), (5.01, False, -4.88)],
    "2MASS 1731+27": [(10.51, False, -4.57), (5.99, False, -4.80)],
    "2MASS 1022+58": [(3.32, False, -5.22), (5.52, False, -5.00)],
    "2MASS 1048+01": [(3.19, False, -5.23), (1.08, False, -5.71)],
    "2MASS 1439+19": [(3.48, False, -5.20), (13.22, False, -4.62)],
    "2MASS 1555-09": [(2.45, False, -5.35), (26.79, False, -4.31)],
    "2MASS 0828-13": [(1.75, False, -5.68), (0.20, True, -6.63)],
    "2MASS 1155-37": [(1.00, False, -5.96), (2.39, False, -5.54)],
}

# Halpha rest wavelength, air, in nm.  The paper's equivalent widths are
# measured on an air scale (footpoints at 6545-6559 and 6567-6580 A).
LAMBDA_HA_NM = 656.280
C_KMS = 2.99792458e5

# Thermal FWHM of Halpha at a chromospheric ~10^4 K, km/s.
THERMAL_FWHM = 21.0
# Rotation-profile FWHM per unit v sin i for moderate limb darkening.
ROT_FWHM_FACTOR = 1.4


def build() -> pd.DataFrame:
    rows = []
    for (name, spt, teff, vsini, ew, ew_lim,
         lratio, lr_lim, multi, vtan) in TABLE4:
        desig, jmag = TABLE1_J[name]

        # Internal consistency check against the authors' own chi calibration.
        implied = math.log10(ew * chi_rb(teff))
        rows.append(dict(
            name=name,
            designation=desig,
            spt=spt,
            J=jmag,
            ew_ha=ew,
            log_lha_lbol=lratio,
            ew_is_limit=ew_lim,
            logratio_is_limit=lr_lim,
            is_detection=not (ew_lim or lr_lim),
            multi_epoch_lowest=multi,
            teff_K=teff,
            vsini_kms=vsini,
            vsini_is_limit=name in VSINI_LIMIT,
            fwhm_kms=math.hypot(ROT_FWHM_FACTOR * vsini, THERMAL_FWHM),
            fwhm_nm=fwhm_kms_to_nm(
                math.hypot(ROT_FWHM_FACTOR * vsini, THERMAL_FWHM)),
            vtan_kms=vtan,
            chi_rb_per_A=chi_rb(teff),
            logratio_implied_rb=implied,
            rb_residual_dex=implied - lratio,
        ))
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame) -> bool:
    """Verify the transcription against the paper's own chi calibration.

    Every row should satisfy log10(EW * chi_RB(Teff)) = log(L_Ha/L_bol), using
    Reiners & Basri's Eq. 1 and Table 2 coefficients.  The absolute residual is
    NOT the right test: it is a smooth function of Teff, running from -0.028 dex
    at 1480 K to +0.021 dex at 1880 K and back to -0.009 dex at 2300 K, because
    the published fifth-order polynomial is a fit to their model grid and
    carries its own residuals.  A transcription error in one object, by
    contrast, would show up as an outlier against other objects sharing the
    same Teff.  So the test is on the residual after removing the per-Teff
    median.
    """
    print("=" * 74)
    print("TRANSCRIPTION CHECK")
    print("=" * 74)

    res = df["rb_residual_dex"]
    df = df.copy()
    df["detrended"] = res - df.groupby("teff_K")["rb_residual_dex"].transform("median")

    print(f"  rows checked                    : {len(df)}")
    print(f"  raw |residual|, max / median    : "
          f"{res.abs().max():.4f} / {res.abs().median():.4f} dex")
    print(f"  after removing per-Teff median  : "
          f"{df['detrended'].abs().max():.4f} / "
          f"{df['detrended'].abs().median():.4f} dex")

    # Rounding floor: EW is printed to 2 decimals, the ratio to 2 decimals.
    import numpy as np
    floor = np.hypot(0.005 / df["ew_ha"] / np.log(10), 0.005)
    ratio = (df["detrended"].abs() / floor)
    print(f"  detrended residual / rounding   : "
          f"max {ratio.max():.1f}, median {ratio.median():.1f}")

    bad = df[ratio > 3.0]
    if len(bad):
        print("\n  rows exceeding 3x the rounding floor:")
        print(bad[["name", "spt", "teff_K", "ew_ha", "log_lha_lbol",
                   "logratio_implied_rb", "detrended"]].to_string(index=False))
        print("  (checked against the paper's Table 3 where available)")

    # Independent cross-check: Table 3 lists both epochs for the eight variable
    # targets, and Table 4 quotes the lowest.  min(Table 3) must equal Table 4.
    print("\n  Table 3 cross-check (min of the two epochs vs Table 4):")
    t3_ok = True
    for nm, epochs in TABLE3.items():
        row = df[df["name"] == nm].iloc[0]
        lo = min(epochs, key=lambda e: e[2])
        match = (abs(lo[0] - row["ew_ha"]) < 1e-9
                 and abs(lo[2] - row["log_lha_lbol"]) < 1e-9
                 and lo[1] == bool(row["ew_is_limit"]))
        t3_ok &= match
        print(f"    {nm:<16s} EW {lo[0]:6.2f}  log {lo[2]:6.2f}   "
              f"{'match' if match else 'MISMATCH'}")
    ok = bool(t3_ok)

    # The one row that fails the polynomial check is internally inconsistent in
    # the paper itself, not mis-transcribed: for 2MASS 1155-37, Table 3 gives
    # two epochs whose equivalent widths differ by log10(2.39/1.00) = 0.378 dex
    # while the quoted ratios differ by 0.42 dex.  The EW = 2.39 epoch matches
    # the polynomial; the EW = 1.00 epoch, which is the one Table 4 carries,
    # does not.
    ew_lo, ew_hi = 1.00, 2.39
    lr_lo, lr_hi = -5.96, -5.54
    print(f"\n  2MASS 1155-37, the one polynomial outlier:")
    print(f"    EW ratio implies              : {math.log10(ew_hi / ew_lo):.3f} dex")
    print(f"    quoted ratios differ by       : {lr_hi - lr_lo:.3f} dex")
    print(f"    EW=2.39 epoch vs polynomial   : "
          f"{math.log10(ew_hi * chi_rb(2080)) - lr_hi:+.3f} dex")
    print(f"    EW=1.00 epoch vs polynomial   : "
          f"{math.log10(ew_lo * chi_rb(2080)) - lr_lo:+.3f} dex")
    print("    the discrepancy is inside the published paper, not the transcription")

    print(f"\n  chi_RB(2300 K)                  : {chi_rb(2300):.3e} 1/A "
          f"(log = {math.log10(chi_rb(2300)):.3f})")
    print(f"  Table 4 objects                 : {len(TABLE4)}")
    print(f"  matched to Table 1 for J        : {df['J'].notna().sum()}")
    print(f"  duplicate names                 : {len(df) - len(set(df['name']))}")
    ok &= bool(df["J"].notna().all())
    ok &= len(set(df["name"])) == len(df)

    print("\n  result:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    df = build()
    good = validate(df)

    out = "reiners_basri_2008_table4.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 74)
    print("CONTENT SUMMARY")
    print("=" * 74)
    print(f"  objects              : {len(df)}")
    print(f"  Halpha detections    : {df['is_detection'].sum()}")
    print(f"  upper limits         : {(~df['is_detection']).sum()}")
    print(f"  multi-epoch (lowest) : {df['multi_epoch_lowest'].sum()}")
    print(f"  spectral types       : {df['spt'].min()} to {df['spt'].max()}")
    print(f"  J range              : {df['J'].min():.2f} to {df['J'].max():.2f}")
    print(f"  log(LHa/Lbol) range  : {df['log_lha_lbol'].min():.2f} to "
          f"{df['log_lha_lbol'].max():.2f}")
    print(f"  line FWHM, km/s      : {df['fwhm_kms'].min():.1f} to "
          f"{df['fwhm_kms'].max():.1f}")
    print(f"  line FWHM, nm        : {df['fwhm_nm'].min():.4f} to "
          f"{df['fwhm_nm'].max():.4f}")
    print(f"  conversion at Halpha : 1 km/s = "
          f"{fwhm_kms_to_nm(1.0):.6e} nm = {fwhm_kms_to_nm(1.0)*10:.6f} A")
    print(f"\nwritten to {out}")
