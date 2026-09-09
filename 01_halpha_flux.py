#!/usr/bin/env python3
"""
01_halpha_flux.py
=================

Standalone conversion from ultracool-dwarf activity strength to Halpha line
flux in SI units.  No dependencies beyond the standard library.

    halpha_line_flux(spt, jmag, ew=...)         ->  W / m^2
    halpha_line_flux(spt, jmag, log_ratio=...)  ->  W / m^2

Give the activity strength either as an Halpha equivalent width in Angstrom
(`ew`) or as log10(L_Halpha / L_bol) (`log_ratio`).  Supplying both runs a
consistency check and raises if they disagree, since the two are linked
through chi and cannot be set independently for a given spectral type.

CHAIN
-----
    m_bol = J + BC_J(SpT)
    F_bol = 10**(-0.4 * (m_bol + 11.482))          erg / s / cm^2
    F_cont(6563) = chi(SpT) * F_bol                erg / s / cm^2 / Angstrom

    from log_ratio:  F_line = 10**log_ratio * F_bol        (chi not needed)
    from ew:         F_line = ew * chi(SpT) * F_bol

    1 erg / s / cm^2  =  1e-3 W / m^2

Note the asymmetry: converting from log(L_Ha/L_bol) needs only the bolometric
correction, because that ratio is by definition F_line / F_bol.  Converting
from an equivalent width additionally needs chi, and so inherits its ~20 per
cent uncertainty and its sensitivity to surface gravity and metallicity.

CALIBRATIONS
------------
BC_J and chi are the medians of Schmidt et al. (2014), AJ 147, 34, Table 5,
valid for optical spectral types M7 to L7.  chi is defined as the ratio of the
continuum flux density around Halpha to the bolometric flux, so that
L_Halpha / L_bol = chi * EW.  The bolometric zero point is the one used in
that paper: m_bol = -2.5 log10(F_bol[cgs]) - 11.482.
"""

from __future__ import annotations

import math

__all__ = ["add_halpha_flux", "halpha_line_flux", "ew_to_log_ratio",
           "log_ratio_to_ew", "bolometric_flux", "bc_j", "chi", "SPT_CAL"]

# ---------------------------------------------------------------------------
# Schmidt et al. (2014) Table 5 medians.
#   subtype index: M7 = 7 ... L0 = 10 ... L7 = 17
#   BC_J in mag; chi in 1/Angstrom; last column is the tabulated scatter in chi.
# ---------------------------------------------------------------------------
SPT_CAL = {
    #  idx    SpT    BC_J     chi        chi_scatter
    7:  ("M7", 1.97, 10.28e-6, 3.13e-6),
    8:  ("M8", 1.97, 4.26e-6, 1.18e-6),
    9:  ("M9", 1.96, 2.52e-6, 0.58e-6),
    10: ("L0", 1.96, 1.98e-6, 0.27e-6),
    11: ("L1", 1.94, 2.25e-6, 0.11e-6),
    12: ("L2", 1.85, 2.11e-6, 0.36e-6),
    13: ("L3", 1.81, 1.67e-6, 0.22e-6),
    14: ("L4", 1.97, 1.16e-6, None),
    15: ("L5", 1.58, 1.46e-6, 0.28e-6),
    16: ("L6", 1.70, 1.23e-6, None),
    17: ("L7", 1.24, 0.73e-6, None),
}

MBOL_ZP = 11.482          # m_bol = -2.5 log10(F_bol[cgs]) - MBOL_ZP
CGS_TO_SI = 1.0e-3        # erg/s/cm^2  ->  W/m^2


# ---------------------------------------------------------------------------
def _spt_index(spt) -> float:
    """Map a spectral type to a numeric subtype index on the M7 = 7 scale.

    Accepts 'M8', 'M8.5', 'L2V', 'l3', or a number already on that scale.
    Returns a float so that half subtypes interpolate rather than round.
    """
    if isinstance(spt, (int, float)) and not isinstance(spt, bool):
        idx = float(spt)
    else:
        s = str(spt).strip().upper().replace(" ", "")
        while s and s[-1] in "VDE":
            s = s[:-1]
        if not s or s[0] not in "ML":
            raise ValueError(f"cannot parse spectral type {spt!r}")
        try:
            sub = float(s[1:])
        except ValueError:
            raise ValueError(f"cannot parse spectral type {spt!r}")
        idx = sub + (0.0 if s[0] == "M" else 10.0)
    if not (7.0 <= idx <= 17.0):
        raise ValueError(
            f"spectral type {spt!r} is outside the M7-L7 calibration range")
    return idx


def _interp(idx: float, column: int) -> float:
    """Linear interpolation between tabulated subtypes.

    chi is interpolated in the log, because it varies by an order of magnitude
    across the range and interpolating chi itself would bias intermediate
    subtypes high.
    """
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    if lo == hi:
        return SPT_CAL[lo][column]
    frac = idx - lo
    a, b = SPT_CAL[lo][column], SPT_CAL[hi][column]
    if column == 2:                                   # chi: interpolate in log
        return 10.0 ** ((1 - frac) * math.log10(a) + frac * math.log10(b))
    return (1 - frac) * a + frac * b


def bc_j(spt) -> float:
    """Bolometric correction in J, such that m_bol = J + BC_J."""
    return _interp(_spt_index(spt), 1)


def chi(spt) -> float:
    """Halpha continuum flux density over bolometric flux, in 1/Angstrom."""
    return _interp(_spt_index(spt), 2)


# ---------------------------------------------------------------------------
def bolometric_flux(spt, jmag: float, si: bool = True) -> float:
    """Bolometric flux at Earth from spectral type and J magnitude.

    Returns W/m^2 by default, or erg/s/cm^2 with si=False.
    """
    m_bol = float(jmag) + bc_j(spt)
    f_bol_cgs = 10.0 ** (-0.4 * (m_bol + MBOL_ZP))
    return f_bol_cgs * CGS_TO_SI if si else f_bol_cgs


def ew_to_log_ratio(spt, ew: float) -> float:
    """log10(L_Halpha / L_bol) from an Halpha equivalent width in Angstrom."""
    if ew <= 0:
        raise ValueError("equivalent width must be positive for emission")
    return math.log10(chi(spt) * ew)


def log_ratio_to_ew(spt, log_ratio: float) -> float:
    """Halpha equivalent width in Angstrom from log10(L_Halpha / L_bol)."""
    return 10.0 ** log_ratio / chi(spt)


# ---------------------------------------------------------------------------
def halpha_line_flux(spt, jmag: float, ew: float | None = None,
                     log_ratio: float | None = None,
                     si: bool = True, tol: float = 0.05,
                     full_output: bool = False):
    """Halpha line flux at Earth, in W/m^2.

    Parameters
    ----------
    spt : str or float
        Optical spectral type, 'M7' to 'L7'.  Half subtypes are interpolated.
    jmag : float
        2MASS J magnitude.
    ew : float, optional
        Halpha equivalent width in Angstrom, positive for emission.
    log_ratio : float, optional
        log10(L_Halpha / L_bol).  Supply this or `ew`, or both.
    si : bool
        Return W/m^2 (default) or erg/s/cm^2.
    tol : float
        Allowed disagreement in dex when both `ew` and `log_ratio` are given.
    full_output : bool
        If True, return a dict of all intermediate quantities instead.

    Returns
    -------
    float, or dict if full_output.

    Examples
    --------
    >>> f'{halpha_line_flux("M8", 12.73, log_ratio=-4.0):.3e}'
    '3.367e-18'
    >>> f'{halpha_line_flux("L3", 15.0, ew=0.6):.3e}'
    '4.831e-21'
    >>> f'{halpha_line_flux("L3", 15.0, ew=0.6, si=False):.3e}'
    '4.831e-18'
    """
    if ew is None and log_ratio is None:
        raise ValueError("supply either ew or log_ratio")

    f_bol_cgs = bolometric_flux(spt, jmag, si=False)
    chi_val = chi(spt)

    if ew is not None and log_ratio is not None:
        implied = ew_to_log_ratio(spt, ew)
        if abs(implied - log_ratio) > tol:
            raise ValueError(
                f"ew = {ew} A implies log(L_Ha/L_bol) = {implied:.3f} for "
                f"{spt}, but log_ratio = {log_ratio:.3f} was given "
                f"(difference {implied - log_ratio:+.3f} dex, tol = {tol}). "
                f"The two are linked by chi = {chi_val:.3e} 1/A and cannot be "
                f"set independently.")

    if log_ratio is not None:
        # Direct: L_Ha/L_bol is by definition F_line/F_bol, so chi does not
        # enter and the result does not inherit the chi uncertainty.
        f_line_cgs = 10.0 ** log_ratio * f_bol_cgs
        route = "log_ratio"
    else:
        f_line_cgs = ew * chi_val * f_bol_cgs
        route = "ew"

    if not full_output:
        return f_line_cgs * CGS_TO_SI if si else f_line_cgs

    f_cont_cgs = chi_val * f_bol_cgs                     # erg/s/cm2/Angstrom
    return {
        "spt": spt,
        "spt_index": _spt_index(spt),
        "jmag": float(jmag),
        "bc_j": bc_j(spt),
        "m_bol": float(jmag) + bc_j(spt),
        "chi_per_A": chi_val,
        "ew_A": ew if ew is not None else log_ratio_to_ew(spt, log_ratio),
        "log_Lha_Lbol": (log_ratio if log_ratio is not None
                         else ew_to_log_ratio(spt, ew)),
        "F_bol_W_m2": f_bol_cgs * CGS_TO_SI,
        "F_cont_W_m2_per_A": f_cont_cgs * CGS_TO_SI,
        "F_line_W_m2": f_line_cgs * CGS_TO_SI,
        "F_line_erg_s_cm2": f_line_cgs,
        "route": route,
    }



# ===========================================================================
# Table driver
# ===========================================================================
COLUMN_ALIASES = {
    "spt": ["spt", "sptype", "sp_type", "spectral_type", "spectraltype",
            "type", "opt_spt", "spectype"],
    "jmag": ["jmag", "j", "j_mag", "mag_j", "j2mass", "jmag_2mass", "j_2mass"],
    "ew": ["ew", "ew_ha", "ewha", "ew_halpha", "halpha_ew", "ha_ew",
           "ew_a", "ewha_a", "ew_angstrom"],
    "log_ratio": ["log_ratio", "logratio", "log_lha_lbol", "loglha_lbol",
                  "loglhalbol", "log_lha_over_lbol", "log_activity",
                  "logl_ha_lbol", "activity"],
}

#: Names of the two columns appended by :func:`add_halpha_flux`.
OUT_EW = "F_line_ew_W_m2"
OUT_RATIO = "F_line_ratio_W_m2"
OUT_NOTE = "halpha_note"


def _resolve_columns(columns, user_map=None):
    """Match the four expected quantities against the table's column names."""
    lower = {str(c).lower().strip(): c for c in columns}
    found = dict(user_map or {})
    for key, aliases in COLUMN_ALIASES.items():
        if key in found:
            continue
        for a in aliases:
            if a in lower:
                found[key] = lower[a]
                break
    for key in ("spt", "jmag"):
        if key not in found:
            raise KeyError(
                f"required column {key!r} not found. Columns present: "
                f"{list(columns)}. Pass columns={{'{key}': 'your_name'}} to "
                f"map it explicitly.")
    if "ew" not in found and "log_ratio" not in found:
        raise KeyError(
            f"neither an equivalent-width nor a log(L_Ha/L_bol) column was "
            f"found. Columns present: {list(columns)}.")
    return found


def _check_ew_sign(values, convention: str):
    """Guard against the negative-EW-means-emission convention.

    Some catalogues record emission as a negative equivalent width. Feeding
    those in unchanged would produce negative line fluxes, so the default is
    to reject them with an explanatory message rather than propagate the sign.
    """
    import math as _m
    finite = [v for v in values if v is not None and not _m.isnan(float(v))]
    if not finite:
        return
    n_neg = sum(1 for v in finite if v < 0)
    if convention == "positive" and n_neg > 0.5 * len(finite):
        raise ValueError(
            f"{n_neg} of {len(finite)} equivalent widths are negative, which "
            f"suggests this catalogue uses the convention that emission is "
            f"negative. Pass ew_convention='negative' to flip the sign, or "
            f"'abs' to take the magnitude.")


def add_halpha_flux(table, columns: dict | None = None,
                    ew_convention: str = "positive",
                    si: bool = True, diagnostics: bool = False,
                    inplace: bool = False):
    """Add Halpha line fluxes to a table, computed along both routes.

    Runs the equivalent-width track and the log(L_Ha/L_bol) track independently
    and appends one column for each, so the two can be compared rather than one
    being chosen for you.

    Parameters
    ----------
    table : DataFrame, path to CSV, sequence of dicts, or anything pandas can
        turn into a DataFrame. Astropy Tables are accepted via ``.to_pandas()``.
    columns : dict, optional
        Explicit mapping for any column the aliases miss, e.g.
        ``{'jmag': 'Jmag_UKIDSS', 'log_ratio': 'logLHa'}``.
    ew_convention : {'positive', 'negative', 'abs'}
        How emission is signed in the input equivalent widths.
    si : bool
        Output in W/m^2 (default) or erg/s/cm^2. The column names carry the
        unit, so they change with this flag.
    diagnostics : bool
        Also append m_bol, chi, the equivalent width implied by the ratio, the
        ratio implied by the equivalent width, and the difference between the
        two routes in dex.
    inplace : bool
        Modify and return the input DataFrame instead of a copy.

    Returns
    -------
    DataFrame with two extra columns:

    ``F_line_ew_W_m2``
        Line flux from EW * chi * F_bol. NaN where no equivalent width is
        available for that row.
    ``F_line_ratio_W_m2``
        Line flux from 10**log_ratio * F_bol. NaN where no ratio is available.
        This route does not use chi, so it does not carry chi's uncertainty.

    A third column, ``halpha_note``, is appended only if at least one row could
    not be converted; it holds the reason for that row and is empty elsewhere.
    Rows never fail quietly: an unparseable spectral type or a missing J
    magnitude produces NaN in the flux columns and a message in the note.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'spt': ['M8', 'L3'], 'J': [12.0, 15.0],
    ...                    'ew_ha': [23.47, 59.88],
    ...                    'log_lha_lbol': [-4.0, -4.0]})
    >>> out = add_halpha_flux(df)
    >>> [f'{v:.3e}' for v in out['F_line_ew_W_m2']]
    ['6.594e-18', '4.822e-19']
    >>> [f'{v:.3e}' for v in out['F_line_ratio_W_m2']]
    ['6.595e-18', '4.822e-19']
    """
    import pandas as pd

    if isinstance(table, str):
        df = pd.read_csv(table)
    elif hasattr(table, "to_pandas"):          # astropy Table
        df = table.to_pandas()
    elif isinstance(table, pd.DataFrame):
        df = table if inplace else table.copy()
    else:
        df = pd.DataFrame(table)

    if ew_convention not in ("positive", "negative", "abs"):
        raise ValueError("ew_convention must be 'positive', 'negative' or 'abs'")

    cmap = _resolve_columns(df.columns, columns)

    unit_suffix = "W_m2" if si else "erg_s_cm2"
    col_ew = f"F_line_ew_{unit_suffix}"
    col_ratio = f"F_line_ratio_{unit_suffix}"

    if "ew" in cmap:
        _check_ew_sign(df[cmap["ew"]].tolist(), ew_convention)

    flux_ew, flux_ratio, notes = [], [], []
    diag = {k: [] for k in ("m_bol", "chi_per_A", "ew_implied_A",
                            "log_ratio_implied", "delta_dex")}

    for _, row in df.iterrows():
        f_ew = f_ratio = float("nan")
        note = ""
        d = {k: float("nan") for k in diag}
        try:
            spt = row[cmap["spt"]]
            jmag = float(row[cmap["jmag"]])
            if not math.isfinite(jmag):
                raise ValueError("J magnitude is not finite")

            d["m_bol"] = jmag + bc_j(spt)
            d["chi_per_A"] = chi(spt)

            ew = None
            if "ew" in cmap:
                raw = row[cmap["ew"]]
                if raw is not None and math.isfinite(float(raw)):
                    ew = float(raw)
                    if ew_convention == "negative":
                        ew = -ew
                    elif ew_convention == "abs":
                        ew = abs(ew)

            lr = None
            if "log_ratio" in cmap:
                raw = row[cmap["log_ratio"]]
                if raw is not None and math.isfinite(float(raw)):
                    lr = float(raw)

            if ew is not None and ew > 0:
                f_ew = halpha_line_flux(spt, jmag, ew=ew, si=si)
                d["log_ratio_implied"] = ew_to_log_ratio(spt, ew)
            elif ew is not None:
                note = f"equivalent width {ew:g} is not positive; no emission"

            if lr is not None:
                f_ratio = halpha_line_flux(spt, jmag, log_ratio=lr, si=si)
                d["ew_implied_A"] = log_ratio_to_ew(spt, lr)

            if math.isfinite(f_ew) and math.isfinite(f_ratio):
                d["delta_dex"] = math.log10(f_ew / f_ratio)
            elif not math.isfinite(f_ew) and not math.isfinite(f_ratio) and not note:
                note = "neither an equivalent width nor a ratio was available"

        except (ValueError, TypeError, KeyError) as exc:
            note = str(exc)

        flux_ew.append(f_ew)
        flux_ratio.append(f_ratio)
        notes.append(note)
        for k in diag:
            diag[k].append(d[k])

    df[col_ew] = flux_ew
    df[col_ratio] = flux_ratio
    if any(notes):
        df[OUT_NOTE] = notes

    if diagnostics:
        for k, v in diag.items():
            df[k] = v

    return df


# ===========================================================================
# Validation
# ===========================================================================
def _selftest() -> bool:
    import pandas as pd
    ok = True

    # 1. Bolometric zero point against the Sun: m_bol = -26.832 -> 1361 W/m^2.
    f = 10.0 ** (-0.4 * (-26.832 + MBOL_ZP)) * CGS_TO_SI
    print(f"solar constant from zero point  : {f:8.1f} W/m^2  (expect 1361)")
    ok &= abs(f / 1361.0 - 1.0) < 0.02

    # 2. Against a tabulated object, Schmidt et al. (2014) Table 4:
    #    SDSS J081058.6+142038.1, M8, i-J = 3.89 so J = 12.73,
    #    tabulated m_bol = 14.7, log chi = -5.53, F_cont = 9.65e-17 cgs.
    m_bol = 12.73 + bc_j("M8")
    f_cont = 10 ** (-5.53) * bolometric_flux("M8", 12.73, si=False)
    print(f"tabulated object m_bol          : {m_bol:8.2f}        (expect 14.70)")
    print(f"tabulated object F_cont         : {f_cont:.3e} cgs  (expect 9.65e-17)")
    ok &= abs(m_bol - 14.70) < 0.02 and abs(f_cont / 9.65e-17 - 1.0) < 0.05

    # 3. Both tracks must agree when the two inputs are mutually consistent.
    df = pd.DataFrame({"spt": ["M8", "M9", "L0", "L2", "L3"],
                       "J": [12.0, 13.0, 13.5, 14.5, 15.0],
                       "ew_ha": [10.0, 10.0, 10.0, 10.0, 10.0]})
    df["log_lha_lbol"] = [ew_to_log_ratio(s, 10.0) for s in df["spt"]]
    out = add_halpha_flux(df, diagnostics=True)
    worst = out["delta_dex"].abs().max()
    print(f"max |ew route - ratio route|    : {worst:.2e} dex")
    ok &= worst < 1e-12

    # 4. Inconsistent inputs must show up as a non-zero difference, not be
    #    reconciled behind the scenes.
    bad = pd.DataFrame({"spt": ["M8"], "J": [13.0],
                        "ew_ha": [10.0], "log_lha_lbol": [-6.0]})
    ob = add_halpha_flux(bad, diagnostics=True)
    print(f"inconsistent row delta          : {ob['delta_dex'][0]:+.3f} dex "
          f"(both routes reported)")
    ok &= abs(ob["delta_dex"][0]) > 1.0

    # 5. Partial rows: one track available, the other NaN, plus a bad type.
    part = pd.DataFrame({"spt": ["M8", "L1", "T4"], "J": [12.0, 14.0, 13.0],
                         "ew_ha": [5.0, float("nan"), 5.0],
                         "log_lha_lbol": [float("nan"), -5.0, -5.0]})
    op = add_halpha_flux(part)
    ok &= math.isnan(op["F_line_ratio_W_m2"][0])
    ok &= math.isnan(op["F_line_ew_W_m2"][1])
    ok &= math.isnan(op["F_line_ew_W_m2"][2]) and bool(op[OUT_NOTE][2])
    print(f"out-of-range row note           : {op[OUT_NOTE][2][:52]}...")

    # 6. Negative-EW convention must be caught rather than yielding negative flux.
    neg = pd.DataFrame({"spt": ["M8", "M9"], "J": [12.0, 13.0],
                        "ew_ha": [-10.0, -8.0]})
    try:
        add_halpha_flux(neg)
        print("negative EW convention          : NOT CAUGHT")
        ok = False
    except ValueError:
        flipped = add_halpha_flux(neg, ew_convention="negative")
        ok &= bool((flipped["F_line_ew_W_m2"] > 0).all())
        print("negative EW convention          : caught, and flips on request")

    # 7. Column aliasing and explicit mapping.
    odd = pd.DataFrame({"SpType": ["L2"], "Jmag_UKIDSS": [14.5],
                        "logLHa": [-5.0]})
    oo = add_halpha_flux(odd, columns={"jmag": "Jmag_UKIDSS",
                                       "log_ratio": "logLHa"})
    ok &= math.isfinite(oo["F_line_ratio_W_m2"][0])
    print(f"explicit column mapping         : "
          f"{oo['F_line_ratio_W_m2'][0]:.3e} W/m^2")

    # 8. The input must not be mutated unless asked.
    src = pd.DataFrame({"spt": ["M8"], "J": [12.0], "ew_ha": [5.0]})
    n_before = len(src.columns)
    add_halpha_flux(src)
    ok &= len(src.columns) == n_before
    print(f"input left unmodified           : {len(src.columns) == n_before}")

    print("\nresult:", "PASS" if ok else "FAIL")
    return ok


def _cli(argv=None):
    import argparse
    import pandas as pd

    p = argparse.ArgumentParser(
        description="Add Halpha line fluxes to a catalogue along both the "
                    "equivalent-width and the log(L_Ha/L_bol) route.")
    p.add_argument("catalogue", nargs="?", help="input CSV")
    p.add_argument("-o", "--output", default=None, help="output CSV")
    p.add_argument("--cgs", action="store_true",
                   help="report erg/s/cm^2 instead of W/m^2")
    p.add_argument("--diagnostics", action="store_true",
                   help="also write m_bol, chi and the route difference")
    p.add_argument("--ew-convention", choices=["positive", "negative", "abs"],
                   default="positive")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args(argv)

    if args.selftest or not args.catalogue:
        import doctest
        fails, _ = doctest.testmod()
        print(f"doctests: {'PASS' if fails == 0 else str(fails) + ' FAILED'}\n")
        good = _selftest()
        return 0 if (good and fails == 0) else 1

    out = add_halpha_flux(args.catalogue, si=not args.cgs,
                          diagnostics=args.diagnostics,
                          ew_convention=args.ew_convention)
    def _fmt(v):
        if v != v:
            return ""
        return f"{v:.4g}" if 1e-3 < abs(v) < 1e5 or v == 0 else f"{v:.4e}"

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(out.to_string(index=False, float_format=_fmt))
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
