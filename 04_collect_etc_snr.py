#!/usr/bin/env python3
"""
04_collect_etc_snr.py
=====================

Read ESO UVES ETC 2.0 output JSONs, integrate the signal-to-noise over the
whole emission line, and merge the result into the catalogue as `int_snr`.

    python3 collect_etc_snr.py etc_jobs/results/*.out.json \\
            --catalogue rb08_with_fluxes.csv --output out.csv

HOW THE INTEGRATION IS DONE
---------------------------
The ETC returns, per echelle order, arrays of target electrons, sky electrons
and total noise per pixel, with

    noise_i = sqrt(target_i + sky_i + dark_term + ron_term)

where dark_term and ron_term are scalar variances per pixel already summed
over the spatial extraction (ron_term = ron^2 * nspat * nspec * ndit).  This
reconstruction was verified exactly against the returned noise array, apart
from the two edge pixels of each order.

Two integrated quantities are computed, because they answer different
questions and differ by 10-30 per cent:

  int_snr           Matched-filter (optimally weighted) S/N.  For a line whose
                    shape is known, the minimum-variance estimate of the total
                    flux has

                        int_snr = sqrt( sum_i (target_i / noise_i)^2 )

                    i.e. the per-pixel S/N added in quadrature.  This is what
                    an optimal extraction actually delivers, and it converges
                    as the window is widened, because pixels with no line flux
                    contribute nothing.  This is the column requested.

  int_snr_boxcar    Plain sum, sum(target) / sqrt(sum(noise^2)), evaluated at
                    the window half-width that maximises it rather than at a
                    fixed one.  Comparing at a fixed window would flatter the
                    matched filter unfairly: the boxcar peaks near +-1.5 sigma
                    and then collapses as empty pixels are added (for the test
                    case, 40.8 at the optimum, 32.6 at +-4 sigma, and 6.4 if
                    the whole order is summed).  The optimal half-width is
                    reported in boxcar_halfwidth_sigma.

The window is set from the line's own flux distribution (flux-weighted centroid
and second moment), not from the requested wavelength, so it is insensitive to
whether the ETC works in air or vacuum.

CONTINUUM CORRECTION (Option A)
-------------------------------
The ETC SED is an emission line on no continuum, so its noise omits photon
noise from the star's own continuum.  That continuum can be recovered from the
run itself plus the measured equivalent width, with no second ETC call and no
throughput assumption:

    k       = N_total / F_line          electrons per unit flux at 6563 A
    F_cont  = F_line / EW               definition of equivalent width
    C_pix   = k * F_cont * dlambda      =  N_total * dlambda / EW

Both k and F_line cancel, so the continuum level per pixel depends only on the
detected line electrons, the dispersion and the equivalent width.  Nothing
about the throughput, the flux calibration or chi enters.  The corrected S/N is

    int_snr_cc = sqrt( sum_i target_i^2 / (target_i + C_pix + sky_i + const) )

The derivation assumes the throughput is the same at the line and across the
continuum window.  The line sits 0.11 of the half-range from the centre of an
order 32 times wider than the integration window, so blaze curvature over that
span is negligible.

Because the continuum here is F_line/EW, using the ratio-track flux (which was
itself built from EW through the Reiners & Basri chi) reproduces exactly the
Reiners & Basri continuum.  That is reported as chi_implied so the choice is
visible rather than buried; it differs from the Schmidt et al. chi by the
offset seen between the two flux tracks.

If a line falls in more than one echelle order, the orders are independent
measurements and their matched-filter S/N add in quadrature.  That is handled,
though for Halpha in the 580 setting the line lands in a single order (m = 93
on the RedMIT chip).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8


# ---------------------------------------------------------------------------
def classify(path: str):
    """Say whether a JSON file is an ETC output, an ETC input, or neither.

    Inputs and outputs live side by side once the jobs have been run in place,
    and both are .json.  An output carries data.plots.spectra; an input does
    not.  Without this split, pointing the collector at a job folder produces
    one confusing KeyError per input file, which is easy to mistake for the
    outputs having failed.

    Parameters:
      path -- path to a JSON file.

    Returns:
      (kind, doc, reason). kind is "output" | "input" | "other" | "unreadable";
      doc is the parsed dict (None if unreadable); reason is a short string,
      empty unless kind is "other" or "unreadable".
    """
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return "unreadable", None, str(exc)[:60]
    if not isinstance(doc, dict):
        return "other", None, "not a JSON object"
    spectra = doc.get("data", {}).get("plots", {}).get("spectra")
    if isinstance(spectra, list) and spectra:
        return "output", doc, ""
    if "timesnr" in doc and "target" in doc:
        return "input", doc, ""
    return "other", doc, "no data.plots.spectra and not an ETC input"


def safe_name(text: str) -> str:
    """The filename form make_etc_jobs uses for a target name.

    Parameters:
      text -- a target name.

    Returns:
      the name with every run of non-alphanumerics collapsed to a single "_".
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")


def parse_2mass_radec(designation: str) -> tuple[float, float]:
    """RA, Dec in degrees from a 2MASS designation, or NaN on a bad string.

    Parameters:
      designation -- a 2MASS designation, e.g. "2MASS J03140344+1603056".

    Returns:
      (ra_deg, dec_deg), or (nan, nan) if the string does not parse.
    """
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


def build_name_resolver(files, manifests, catalogue_names):
    """Work out which target each ETC output belongs to.

    Three sources, in order of preference:

      1. input.sky.almanac.targetName inside the output.  Absent whenever the
         ETC form was downloaded without an almanac block, which is the case
         for templates saved in SNR-solve mode.  When it is missing every
         record ends up nameless, and since the merge, the equivalent-width
         lookup and the per-target collapse all key on the name, the whole
         pipeline quietly produces nothing.
      2. a json_to_target_dict.csv written by make_etc_jobs, matched on the job path.
      3. the output filename stem, matched against the catalogue names after
         normalising punctuation the same way make_etc_jobs does.

    Parameters:
      files           -- ETC output paths to resolve.
      manifests       -- json_to_target_dict.csv paths to match job paths against.
      catalogue_names -- target names to match filename stems against.

    Returns:
      dict mapping absolute output path -> (target_name, source), where source
      is "manifest" or "filename". Paths that could not be resolved are absent.
    """
    resolved = {}

    # (2) manifests
    by_job = {}
    for mpath in manifests:
        try:
            m = pd.read_csv(mpath)
        except Exception:
            continue
        if "json_file" not in m.columns or "name" not in m.columns:
            continue
        root = os.path.dirname(os.path.abspath(mpath))
        for _, r in m.iterrows():
            jf = r.get("json_file")
            if not isinstance(jf, str) or not jf:
                continue
            cand = jf if os.path.isabs(jf) else os.path.join(root, jf)
            key = os.path.normpath(os.path.abspath(cand))
            by_job[key] = str(r["name"])
            by_job[os.path.splitext(key)[0] + ".out.json"] = str(r["name"])

    # (3) catalogue names, normalised
    by_stem = {safe_name(n): n for n in catalogue_names}

    for f in files:
        ap = os.path.normpath(os.path.abspath(f))
        if ap in by_job:
            resolved[ap] = (by_job[ap], "manifest")
            continue
        stem = os.path.basename(ap)
        for suffix in (".out.json", ".json"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        if stem in by_stem:
            resolved[ap] = (by_stem[stem], "filename")
    return resolved


def _order_arrays(spec: dict):
    """Wavelength in nm plus the target, sky and noise arrays for one order.

    Parameters:
      spec -- one entry of data.plots.spectra.

    Returns:
      (w_nm, target_e, sky_e, noise_e, const_var, noise_info). The first four
      are per-pixel arrays; const_var is the scalar dark + read-noise variance
      per pixel; noise_info is the ETC's noise metadata dict (ron, nspat, ...).
    """
    nc = spec["noise_components"]
    w = np.asarray(spec["wavelength"], dtype=float) * 1e9        # m -> nm
    t = np.asarray(nc["target"], dtype=float)
    s = np.asarray(nc["sky"], dtype=float)
    n = np.asarray(nc["noise"], dtype=float)
    terms = nc["noise_terms"]
    const_var = float(terms["dark_term"]) + float(terms["ron_term"])
    return w, t, s, n, const_var, nc["noise_info"]


def average_snr_per_pixel_in_band(path: str, lo_nm: float, hi_nm: float) -> dict:
    """Mean and RMS per-pixel S/N in a wavelength window, using the same ETC arrays.

    This is the direct analogue of the FeH-band measurement in Reiners & Basri,
    where the reported quantity is the average S/N per pixel over a chosen band.

    Parameters:
      path         -- an ETC output JSON.
      lo_nm, hi_nm -- wavelength window in nm.

    Returns:
      dict with mean / RMS / 16th / 84th-percentile per-pixel S/N, pixel count,
      mean signal and noise, and a per_order breakdown.
      Raises ValueError if no pixels fall in the window.
    """
    with open(path) as fh:
        doc = json.load(fh)

    spectra = doc["data"]["plots"]["spectra"]
    total_pix = 0
    total_mean = 0.0
    total_rms2 = 0.0
    total_signal = 0.0
    total_noise = 0.0
    all_snr = []
    per_order = []

    for sp in spectra:
        w, t, s, n, cv, info = _order_arrays(sp)
        mask = (w >= lo_nm) & (w <= hi_nm)
        if not np.any(mask):
            continue

        t_band = t[mask]
        n_band = n[mask]
        snr = np.divide(t_band, n_band, out=np.zeros_like(t_band), where=n_band > 0)
        total_pix += int(mask.sum())
        total_mean += float(snr.sum())
        total_rms2 += float((snr ** 2).sum())
        total_signal += float(t_band.sum())
        total_noise += float(np.sqrt(np.sum(n_band ** 2)))
        all_snr.extend(snr.tolist())

        per_order.append(dict(
            order=sp.get("order"), detector=sp.get("detector_name"),
            npix=int(mask.sum()),
            mean_snr_per_pixel=float(np.mean(snr)) if snr.size else 0.0,
            rms_snr_per_pixel=float(np.sqrt(np.mean(snr ** 2))) if snr.size else 0.0,
            signal_e=float(t_band.sum()), noise_e=float(np.sqrt(np.sum(n_band ** 2))),
            lo_nm=float(lo_nm), hi_nm=float(hi_nm),
        ))

    if total_pix == 0:
        raise ValueError(f"{path}: no pixels fall in {lo_nm}..{hi_nm} nm")

    snr_arr = np.asarray(all_snr, dtype=float)
    p16 = float(np.percentile(snr_arr, 16)) if snr_arr.size else 0.0
    p84 = float(np.percentile(snr_arr, 84)) if snr_arr.size else 0.0

    return dict(
        file=os.path.basename(path),
        lo_nm=float(lo_nm), hi_nm=float(hi_nm),
        n_pix=total_pix,
        mean_snr_per_pixel=total_mean / total_pix,
        rms_snr_per_pixel=math.sqrt(total_rms2 / total_pix),
        p16_snr_per_pixel=p16,
        p84_snr_per_pixel=p84,
        mean_signal_e=total_signal / total_pix,
        mean_noise_e=total_noise / total_pix,
        per_order=per_order,
    )


def integrate_line(path: str, n_sigma: float = 4.0,
                   ew_angstrom: float | None = None) -> dict:
    """Integrate the S/N over the emission line in one ETC output file.

    Parameters:
      path        -- an ETC output JSON.
      n_sigma     -- integration half-width, in line sigmas.
      ew_angstrom -- Halpha equivalent width in Angstrom; when > 0 it enables
                     the continuum correction, otherwise int_snr_cc is NaN.

    Returns:
      dict of one run's results -- int_snr, int_snr_cc, int_snr_boxcar, the
      noise budget, line diagnostics, and _orders (per-order rates, dropped
      before the CSV is written).
      Raises ValueError if no order carries flux or the window is too narrow.
    """
    with open(path) as fh:
        doc = json.load(fh)

    inp = doc.get("input", {})
    line = inp["target"]["sed"]["emissionline"]["params"]
    lam_req = float(line["lambda"])
    fwhm_req = float(line["fwhm"])
    exptime = float(inp["timesnr"]["DET2.WIN1.UIT1"])
    nexpo = int(inp["timesnr"].get("SEQ.NEXPO", 1))
    flux_in = float(inp["target"]["brightness"]["flux"])
    name = inp.get("sky", {}).get("almanac", {}).get("targetName", "")

    spectra = doc["data"]["plots"]["spectra"]

    # Orders that actually carry line flux.  Selecting on the flux rather than
    # on the requested wavelength avoids any air/vacuum assumption.
    hits = []
    for sp in spectra:
        w, t, s, n, cv, info = _order_arrays(sp)
        if t.sum() <= 0:
            continue
        hits.append((sp, w, t, s, n, cv, info))
    if not hits:
        raise ValueError(f"{path}: no order carries any target flux")

    total_flux_all = sum(h[2].sum() for h in hits)

    # Continuum electrons per pixel, from the run itself plus the EW.
    # N_total is summed over every order, so the split-line case stays correct.
    sum_snr2 = 0.0
    sum_snr2_cc = 0.0
    box_terms = []
    captured = 0.0
    cont_pix = float("nan")
    have_cont = ew_angstrom is not None and np.isfinite(ew_angstrom) \
        and ew_angstrom > 0
    per_order = []
    sat = False
    nonlin = False

    for sp, w, t, s, n, cv, info in hits:
        # Line position and width from the profile itself.
        centre = float((t * w).sum() / t.sum())
        sigma = float(np.sqrt((t * (w - centre) ** 2).sum() / t.sum()))
        lo, hi = centre - n_sigma * sigma, centre + n_sigma * sigma
        m = (w >= lo) & (w <= hi)
        if m.sum() < 3:
            continue

        snr_pix = np.where(n > 0, t / n, 0.0)
        sum_snr2 += float((snr_pix[m] ** 2).sum())
        captured += float(t[m].sum())

        if have_cont:
            disp_A = float(np.median(np.diff(w))) * 10.0
            c_pix = total_flux_all * disp_A / ew_angstrom
            cont_pix = c_pix
            var_cc = t[m] + c_pix + s[m] + cv
            sum_snr2_cc += float(((t[m] ** 2) / var_cc).sum())

        # Boxcar at its own optimum, so the comparison with the matched filter
        # is between each method at its best rather than at a shared window.
        best, best_k = -1.0, np.nan
        for k in np.arange(0.4, n_sigma + 0.01, 0.05):
            mk = np.abs(w - centre) <= k * sigma
            if mk.sum() < 3:
                continue
            v = float((n[mk] ** 2).sum())
            if v > 0:
                val = float(t[mk].sum()) / math.sqrt(v)
                if val > best:
                    best, best_k = val, float(k)
        box_terms.append((best, best_k))

        sat |= bool(sp.get("saturation", {}).get("saturationflag", False))
        nonlin |= bool(sp.get("saturation", {}).get("nonlinearflag", False))

        per_order.append(dict(
            order=sp.get("order"), detector=sp.get("detector_name"),
            centre_nm=centre, fwhm_nm=2.3548 * sigma,
            npix=int(m.sum()),
            target_e=float(t[m].sum()), sky_e=float(s[m].sum()),
            const_var=cv, ron=info.get("ron"), nspat=info.get("nspat"),
            dispersion_nm=float(np.median(np.diff(w))),
            # rates, for extrapolating to other exposure times
            target_rate=t[m] / exptime, sky_rate=s[m] / exptime,
            cont_rate=(cont_pix / exptime) if have_cont else 0.0,
            const_var_per_exp=cv,
        ))

    if not per_order:
        raise ValueError(f"{path}: line found but window too narrow to integrate")

    int_snr = math.sqrt(sum_snr2)
    int_snr_cc = math.sqrt(sum_snr2_cc) if have_cont else float("nan")
    # Orders are independent measurements, so both estimators combine in
    # quadrature across orders.
    int_snr_boxcar = math.sqrt(sum(b ** 2 for b, _ in box_terms if b > 0))
    box_halfwidth = box_terms[0][1] if box_terms else float("nan")

    # Noise budget over the integration window.
    tgt = sum(o["target_e"] for o in per_order)
    sky = sum(o["sky_e"] for o in per_order)
    const = sum(o["const_var"] * o["npix"] for o in per_order)
    cont = (cont_pix * sum(o["npix"] for o in per_order)) if have_cont else 0.0
    total_var = tgt + sky + const + cont
    regime = max((("line photons", tgt), ("sky", sky), ("continuum", cont),
                  ("read noise + dark", const)), key=lambda kv: kv[1])[0]

    return dict(
        file=os.path.basename(path), name=name,
        int_snr=int_snr, int_snr_cc=int_snr_cc,
        int_snr_boxcar=int_snr_boxcar,
        boxcar_halfwidth_sigma=box_halfwidth,
        ew_used_A=ew_angstrom if have_cont else float("nan"),
        cont_e_per_pix=cont_pix,
        cont_over_line=(1.5056 * (per_order[0]["fwhm_nm"] * 10.0)
                        / ew_angstrom) if have_cont else float("nan"),
        snr_penalty=(int_snr / int_snr_cc) if have_cont and int_snr_cc > 0
        else float("nan"),
        exptime_s=exptime, nexpo=nexpo, flux_in_W_m2=flux_in,
        lambda_req_nm=lam_req, fwhm_req_nm=fwhm_req,
        line_centre_nm=per_order[0]["centre_nm"],
        fwhm_out_nm=per_order[0]["fwhm_nm"],
        dispersion_nm_pix=per_order[0]["dispersion_nm"],
        n_orders=len(per_order), orders=";".join(str(o["order"]) for o in per_order),
        npix=sum(o["npix"] for o in per_order),
        flux_fraction=captured / total_flux_all,
        target_e=tgt, sky_e=sky, const_var=const,
        limiting_noise=regime,
        frac_var_line=tgt / total_var, frac_var_sky=sky / total_var,
        frac_var_cont=cont / total_var, frac_var_const=const / total_var,
        saturated=sat, nonlinear=nonlin,
        _orders=per_order,
    )


# ---------------------------------------------------------------------------
def predict_snr(rec: dict, t: float, n_exp: int = 1) -> float:
    """Integrated S/N at exposure time t, extrapolated from one run's rates.

    Signal and sky scale with time; the read-noise and dark variance scale with
    the number of exposures.

    Parameters:
      rec   -- an integrate_line result (needs its _orders rate arrays).
      t     -- exposure time in seconds.
      n_exp -- number of exposures the read noise is spread over.

    Returns:
      the extrapolated integrated (matched-filter) S/N.
    """
    total = 0.0
    for o in rec["_orders"]:
        sig = o["target_rate"] * t
        var = sig + o["sky_rate"] * t + o["const_var_per_exp"] * n_exp
        total += float(((sig ** 2) / var).sum())
    return math.sqrt(total)


def check_extrapolation(recs: list) -> pd.DataFrame:
    """Cross-check predict_snr's linear rate model against the probe-times sweep.

    A single ETC run separates target, sky and read-noise into per-pixel
    rates, from which predict_snr extrapolates the S/N at any other exposure
    time. A --probe-times sweep gives an independent way to check that this
    linear-scaling assumption actually holds: each target's shortest run is
    used to predict the others, and predicted vs measured is compared.

    Parameters:
      recs -- integrate_line results, ideally several exposure times per target.

    Returns:
      DataFrame with one row per (target, exposure time): measured vs predicted
      int_snr and their relative error. Empty if no target has more than one time.
    """
    rows = []
    by_name = {}
    for r in recs:
        by_name.setdefault(r["name"] or r["file"], []).append(r)
    for name, group in by_name.items():
        if len(group) < 2:
            continue
        ref = min(group, key=lambda r: r["exptime_s"])
        for r in group:
            pred = predict_snr(ref, r["exptime_s"], r.get("nexpo", 1))
            rows.append(dict(name=name, ref_t=ref["exptime_s"],
                             t=r["exptime_s"], measured=r["int_snr"],
                             predicted=pred,
                             rel_err=(pred - r["int_snr"]) / r["int_snr"]))
    return pd.DataFrame(rows)


def main(argv=None):
    """Integrate every ETC output and write one row per run to --output.

    Parameters:
      argv -- command-line arguments (defaults to sys.argv).

    Returns:
      0 on success. Side effect: writes the CSV to --output, one row per ETC
      run (target x exposure time), merged into --catalogue if one is given.
    """
    p = argparse.ArgumentParser(
        description="Integrate ETC line S/N and merge into the catalogue.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("outputs", nargs="+",
                   help="ETC output JSON files or directories containing them")
    p.add_argument("--catalogue", default=None,
                   help="CSV to merge into; the int_snr column is added to it")
    p.add_argument("--output", default="out.csv",
                   help="one row per ETC run (target x exposure time)")
    p.add_argument("--n-sigma", type=float, default=4.0,
                   help="half-width of the integration window, in line sigmas")
    p.add_argument("--ew-column", default="ew_ha",
                   help="catalogue column holding the Halpha equivalent width "
                        "in Angstrom; drives the continuum correction")
    p.add_argument("--manifest", default=None,
                   help="json_to_target_dict.csv from make_etc_jobs, used to identify "
                        "targets when the ETC output carries no targetName")
    args = p.parse_args(argv)

    files = []
    for spec in args.outputs:
        if os.path.isdir(spec):
            files += sorted(glob.glob(os.path.join(spec, "**", "*.json"),
                                      recursive=True))
        else:
            files += sorted(glob.glob(spec))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise SystemExit("no ETC output files matched")

    # Equivalent widths, needed for the continuum correction, keyed on the
    # same target name the ETC output carries.
    ew_map = {}
    cat = None
    if args.catalogue:
        cat = pd.read_csv(args.catalogue)
        # make_etc_jobs resolves and caches ra_deg/dec_deg into the catalogue
        # it runs against; only fall back to parsing the designation here if
        # that hasn't happened (e.g. a catalogue used outside the pipeline).
        if "designation" in cat.columns and (
                "ra_deg" not in cat.columns or "dec_deg" not in cat.columns):
            radec = [parse_2mass_radec(d) for d in cat["designation"]]
            cat["ra_deg"] = [r for r, _ in radec]
            cat["dec_deg"] = [d for _, d in radec]
        if "name" in cat.columns and args.ew_column in cat.columns:
            lim = (cat["ew_is_limit"] if "ew_is_limit" in cat.columns
                   else pd.Series(False, index=cat.index))
            for _, r in cat.iterrows():
                ew_map[str(r["name"])] = (r[args.ew_column],
                                          bool(lim.loc[_]) if _ in lim.index
                                          else False)

    # Split inputs from outputs before doing any work.
    outputs, inputs, other = [], [], []
    for f in files:
        kind, doc, why = classify(f)
        if kind == "output":
            outputs.append((f, doc))
        elif kind == "input":
            inputs.append(f)
        else:
            other.append((os.path.basename(f), why))
    if not outputs:
        raise SystemExit(
            f"no ETC output files found among {len(files)} JSON file(s)"
            + (f"; {len(inputs)} look like ETC *inputs*, which means the jobs "
               f"have not been run yet or the results are somewhere else"
               if inputs else ""))

    # Names may not be inside the outputs at all, so build a fallback map.
    manifests = sorted({os.path.join(d, "json_to_target_dict.csv")
                        for d in {os.path.dirname(os.path.abspath(f))
                                  for f, _ in outputs}
                        | {os.path.dirname(os.path.dirname(os.path.abspath(f)))
                           for f, _ in outputs}
                        if os.path.isfile(os.path.join(d, "json_to_target_dict.csv"))})
    if args.manifest:
        manifests = [args.manifest] + manifests
    resolver = build_name_resolver(
        [f for f, _ in outputs], manifests,
        list(cat["name"]) if cat is not None and "name" in cat.columns else [])
    name_sources = {}

    recs, failed = [], []
    for f, doc in outputs:
        try:
            nm = doc.get("input", {}).get("sky", {}) \
                .get("almanac", {}).get("targetName", "") or ""
            src = "almanac" if nm else ""
            if not nm:
                nm, src = resolver.get(os.path.normpath(os.path.abspath(f)),
                                       ("", "unresolved"))
            name_sources[src] = name_sources.get(src, 0) + 1
            ew, ew_lim = ew_map.get(nm, (None, False))
            r = integrate_line(f, n_sigma=args.n_sigma, ew_angstrom=ew)
            # integrate_line reads the name out of the document, which is
            # empty when the template had no almanac block; overwrite it with
            # the resolved one.
            r["name"] = nm
            r["name_source"] = src
            r["ew_is_limit"] = ew_lim
            r["folder"] = os.path.basename(os.path.dirname(os.path.abspath(f)))
        except (KeyError, ValueError) as exc:
            failed.append((os.path.basename(f), str(exc)[:70]))
            continue
        recs.append(r)

    if not recs:
        raise SystemExit("no file could be parsed:\n  " +
                         "\n  ".join(f"{a}: {b}" for a, b in failed))

    res = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                        for r in recs])

    # ---- report ----------------------------------------------------------
    print("=" * 78)
    print("INTEGRATED LINE S/N")
    print("=" * 78)
    print(f"  JSON files seen       : {len(files)}")
    print(f"    ETC outputs         : {len(outputs)} -> {len(recs)} integrated")
    if inputs:
        print(f"    ETC inputs skipped  : {len(inputs)} "
              f"(job files sitting beside the results)")
    if other:
        print(f"    unrecognised        : {len(other)}")
        for a, b in other[:5]:
            print(f"        {a}: {b}")
    if failed:
        print(f"    failed to integrate : {len(failed)}")
        for a, b in failed:
            print(f"        {a}: {b}")

    if len(name_sources) > 1 or "almanac" not in name_sources:
        print(f"    target names from    : "
              + ", ".join(f"{k or 'none'} ({v})"
                          for k, v in sorted(name_sources.items())))
    n_unnamed = sum(1 for r in recs if not r["name"])
    if n_unnamed:
        raise SystemExit(
            f"{n_unnamed} of {len(recs)} results could not be matched to a "
            f"target.\n"
            f"  The ETC output carries no sky.almanac.targetName, which happens "
            f"when the form was downloaded without an almanac block.\n"
            f"  Point --manifest at the json_to_target_dict.csv written by "
            f"make_etc_jobs, or keep the job filenames as generated so they "
            f"can be matched to catalogue names.")

    folders = sorted({r["folder"] for r in recs})
    if len(folders) > 1:
        print(f"\n  results found in {len(folders)} folder(s):")
        for fo in folders:
            sub = [r for r in recs if r["folder"] == fo]
            ts = sorted({r["exptime_s"] for r in sub})
            print(f"    {fo:<12s} {len(sub):>3d} target(s), DIT "
                  + ", ".join(f"{t:.0f}s" for t in ts))

    r0 = recs[0]
    print(f"\n  consistency checks on {r0['name'] or r0['file']}:")
    print(f"    line found in order       : {r0['orders']} "
          f"({r0['n_orders']} order(s))")
    print(f"    centroid vs requested     : {r0['line_centre_nm']:.4f} nm vs "
          f"{r0['lambda_req_nm']:.4f} nm  "
          f"({(r0['line_centre_nm'] - r0['lambda_req_nm']) * 1000:+.2f} pm)")
    inst = math.hypot(r0["fwhm_req_nm"], 0.0)
    delivered = r0["fwhm_out_nm"]
    implied_inst = math.sqrt(max(delivered ** 2 - r0["fwhm_req_nm"] ** 2, 0))
    if implied_inst > 0:
        r_val = f"{r0['line_centre_nm'] / implied_inst:,.0f}"
    else:
        r_val = "N/A"
    print(f"    FWHM in / out             : {r0['fwhm_req_nm'] * 1000:.2f} pm -> "
          f"{delivered * 1000:.2f} pm, implying an LSF of "
          f"{implied_inst * 1000:.2f} pm (R = {r_val})")
    print(f"    dispersion                : "
          f"{r0['dispersion_nm_pix'] * 10:.5f} A/pixel")
    print(f"    line flux captured        : {r0['flux_fraction'] * 100:.3f} % "
          f"in {r0['npix']} pixels (+-{args.n_sigma} sigma)")

    if "cont_e_per_pix" in res.columns and res["cont_e_per_pix"].notna().any():
        r = recs[0]
        print(f"\n  continuum correction (Option A):")
        print(f"    EW used                   : {r['ew_used_A']} A")
        print(f"    C_pix = N_total*dlam/EW   : {r['cont_e_per_pix']:.2f} "
              f"e-/pixel  (no throughput or flux term enters)")
        print(f"    continuum / line in band  : {r['cont_over_line']:.3f}")

    print(f"\n  {'name':<16s} {'t[s]':>6s} {'int_snr':>8s} {'int_snr_cc':>11s} "
          f"{'penalty':>8s} {'boxcar':>7s} {'limiting noise':<18s}")
    for _, r in res.sort_values("int_snr", ascending=False).iterrows():
        def f(v, w, d=2):
            return f"{v:{w}.{d}f}" if v is not None and np.isfinite(v) \
                else f"{'-':>{w}}"
        print(f"  {str(r['name'])[:16]:<16s} {r['exptime_s']:6.0f} "
              f"{f(r['int_snr'], 8)} {f(r.get('int_snr_cc'), 11)} "
              f"{f(r.get('snr_penalty'), 8)} {f(r['int_snr_boxcar'], 7)} "
              f"{r['limiting_noise']:<18s}")

    # ---- probe-times handling -------------------------------------------
    n_times = res["exptime_s"].nunique()
    dup = res["name"].duplicated().any()
    if dup:
        chk = check_extrapolation(recs)
        print(f"\n  probe-times sweep: {res['name'].nunique()} target(s) at "
              f"{n_times} exposure time(s), {len(res)} runs")
        print(f"    exposure times          : "
              f"{', '.join(f'{t:.0f}s' for t in sorted(res['exptime_s'].unique()))}")
        if len(chk):
            worst = chk.loc[chk["rel_err"].abs().idxmax()]
            print(f"    extrapolation check     : predicting each run from the "
                  f"shortest one")
            print(f"      max relative error    : {chk['rel_err'].abs().max():.3%} "
                  f"({worst['name']}, {worst['ref_t']:.0f}s -> {worst['t']:.0f}s: "
                  f"predicted {worst['predicted']:.2f} vs measured "
                  f"{worst['measured']:.2f})")
            print(f"      median                : {chk['rel_err'].abs().median():.3%}")
            if chk["rel_err"].abs().max() > 0.05:
                print(f"      WARNING: above 5%, so the linear rate model "
                      f"does not hold well for this target")

    if res["saturated"].any() or res["nonlinear"].any():
        print(f"\n  WARNING: {int(res['saturated'].sum())} saturated, "
              f"{int(res['nonlinear'].sum())} non-linear")

    # ---- write: one row per ETC run (target x exposure time) -----

    res_out = res.drop(columns=[c for c in res.columns if c.startswith("_")])
    if args.catalogue:
        key = "name" if "name" in cat.columns else None
        if key is None:
            raise SystemExit("catalogue has no 'name' column to merge on")
        cols = ["name", "int_snr", "int_snr_cc", "snr_penalty",
                "int_snr_boxcar", "boxcar_halfwidth_sigma",
                "cont_e_per_pix", "cont_over_line", "ew_used_A",
                "exptime_s", "npix",
                "folder", "flux_fraction", "limiting_noise", "frac_var_line",
                "frac_var_sky", "frac_var_cont", "frac_var_const", "saturated"]
        cols = [c for c in cols if c in res_out.columns]
        add = res_out[cols].rename(columns={
            "exptime_s": "etc_exptime_s", "npix": "etc_npix",
            "folder": "etc_folder",
            "flux_fraction": "etc_flux_fraction",
            "limiting_noise": "etc_limiting_noise",
            "saturated": "etc_saturated"})
        # A catalogue row can now match several ETC runs (one per probed
        # exposure time), so this is one-to-many, not one-to-one; a
        # catalogue row with no ETC result still appears once, with NaNs.
        merged = cat.merge(add, on=key, how="left", validate="one_to_many")
        merged.to_csv(args.output, index=False)
        n_targets = merged.loc[merged["int_snr"].notna(), "name"].nunique()
        print(f"\n  merged into {len(merged)} row(s) "
              f"({merged['name'].nunique()} catalogue target(s), "
              f"{n_targets} with an int_snr)")
        print(f"  written to {args.output}")
    else:
        res_out.to_csv(args.output, index=False)
        print(f"\n  written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
