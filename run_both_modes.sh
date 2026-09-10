#!/bin/bash
set -eu

# Summary: Run ETC jobs for one or both instrument modes and update
# visibility. Defaults to both (etc_jobs, etc_jobs_2x2); pass one or more
# directories to run only those, e.g. `./run_both_modes.sh etc_jobs_2x2`.
# Steps (in order, per selected mode):
# 1) Compute Halpha fluxes for the catalogue and generate
#    per-target ETC job JSONs (02_make_etc_jobs.py).
# 2) For each mode (1x2, 2x2) run the ETC client on all JSONs,
#    producing corresponding .out.json results (ETC client: 03_etc_cli.py).
# 3) Collect SNR/results into a summary CSV for the mode
#    (04_collect_etc_snr.py -> out.csv or out2x2.csv).
# 4) Compute annual visibility/scheduling feasibility and merge the
#    results in-place into that mode CSV (05_calc_visibility_for_exposures.py
#    updates the same out.csv / out2x2.csv file).
#
# Python scripts (execution order):
# - 01_halpha_flux.py            # compute Halpha flux columns for the catalogue
# - 02_make_etc_jobs.py          # generate per-target ETC JSONs (run for 1x2 and 2x2);
#                                 # also resolves ra_deg/dec_deg from the 2MASS
#                                 # designation once and caches them into
#                                 # FLUXED_CATALOGUE for every later step to reuse
# - 03_etc_cli.py (ETC_CLI)      # run the ETC on each JSON, producing .out.json
# - 04_collect_etc_snr.py        # collect per-target SNR and merge into out*.csv
# - 05_calc_visibility_for_exposures.py  # compute visibility & merge into out*.csv
#
# Environment knobs: ETC_CLI, SERVER, PYTHON, CATALOGUE, EXPTIME,
# MAKE_JOBS, FORCE_MAKE, SKIP_EXISTING_OUT and visibility-specific vars
# (TEL_OVERHEAD, INSTR_OVERHEAD, ACQ_OVERHEAD, READOUT_PER_SPEC, N_SPEC,
# VISITS_PER_YEAR, EXPOSURES_PER_VISIT, VIS_YEAR, VIS_STEP, VIS_SITE).

ETC_CLI="${ETC_CLI:-03_etc_cli.py}"
SERVER="${SERVER:-https://etc.eso.org}"
PYTHON="${PYTHON:-python3}"
CATALOGUE="${CATALOGUE:-reiners_basri_2008_joined.csv}"
MAKE_JOBS="${MAKE_JOBS:-1}"
FORCE_MAKE="${FORCE_MAKE:-0}"
EXPTIME="${EXPTIME:-3600}"
SKIP_EXISTING_OUT="${SKIP_EXISTING_OUT:-0}"
FLUXED_CATALOGUE="${FLUXED_CATALOGUE:-${CATALOGUE%.csv}_with_fluxes.csv}"

# Visibility check parameters (can be overridden in env)
TEL_OVERHEAD="${TEL_OVERHEAD:-600}"
INSTR_OVERHEAD="${INSTR_OVERHEAD:-60}"
ACQ_OVERHEAD="${ACQ_OVERHEAD:-120}"
READOUT_PER_SPEC="${READOUT_PER_SPEC:-52}"
N_SPEC="${N_SPEC:-1}"
VISITS_PER_YEAR="${VISITS_PER_YEAR:-6}"
EXPOSURES_PER_VISIT="${EXPOSURES_PER_VISIT:-3}"
VIS_YEAR="${VIS_YEAR:-2027}"
VIS_STEP="${VIS_STEP:-30}"

VIS_SITE="${VIS_SITE:-vlt}"

# True if $1 is an existing, non-empty ETC output with real spectra (not a
# truncated download or an error response), so it is safe to reuse instead of
# calling the ETC again. Set SKIP_EXISTING_OUT=1 to enable this reuse.
is_valid_etc_output() {
    [ -s "$1" ] && "$PYTHON" -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
sp = d.get("data", {}).get("plots", {}).get("spectra")
sys.exit(0 if isinstance(sp, list) and sp else 1)
' "$1" 2>/dev/null
}

if [ ! -f "$ETC_CLI" ] && ! command -v "$ETC_CLI" >/dev/null 2>&1; then
    echo "03_etc_cli.py not found at '$ETC_CLI'." >&2
    echo "Set ETC_CLI=/path/to/03_etc_cli.py or copy the CLI into this folder." >&2
    exit 2
fi

# Default comparison pair: 1x2 mode vs 2x2 mode. Pass one or more directories
# to run just those modes instead, e.g. `./run_both_modes.sh etc_jobs_2x2`
# runs only the 2x2 mode (job generation included) and everything after it.
if [ "$#" -ge 1 ]; then
    MODE_DIRS=("$@")
else
    MODE_DIRS=(etc_jobs etc_jobs_2x2)
fi

in_mode_dirs() {
    local d
    for d in "${MODE_DIRS[@]}"; do
        [ "$d" = "$1" ] && return 0
    done
    return 1
}

# Optionally create jobs from the catalogue using the bundled templates.
# Set MAKE_JOBS=0 to skip; set FORCE_MAKE=1 to overwrite existing dirs.
# Only the mode(s) selected via MODE_DIRS above are (re)generated.
if [ "$MAKE_JOBS" -ne 0 ]; then
    echo "Generating ETC job JSONs from catalogue using 02_make_etc_jobs.py"
    if [ ! -f 02_make_etc_jobs.py ]; then
        echo "02_make_etc_jobs.py not found in this folder" >&2
        exit 5
    fi
    # Ensure the catalogue has Halpha flux columns by running 01_halpha_flux.py
    echo "Adding Halpha flux columns to $CATALOGUE -> $FLUXED_CATALOGUE"
    if [ ! -f 01_halpha_flux.py ]; then
        echo "01_halpha_flux.py not found in this folder" >&2
        exit 6
    fi
    "$PYTHON" 01_halpha_flux.py "$CATALOGUE" -o "$FLUXED_CATALOGUE"
    # 1x2 mode
    if ! in_mode_dirs "etc_jobs"; then
        : # not requested this run
    elif [ -d "etc_jobs" ] && [ "$FORCE_MAKE" -eq 0 ]; then
        echo "etc_jobs exists, skipping (set FORCE_MAKE=1 to overwrite)"
    else
        echo "Creating etc_jobs from template uves_etc_inputform.json"
        "$PYTHON" 02_make_etc_jobs.py "$FLUXED_CATALOGUE" uves_etc_inputform.json \
            --outdir etc_jobs --exptime "$EXPTIME"
    fi
    # 2x2 mode
    if ! in_mode_dirs "etc_jobs_2x2"; then
        : # not requested this run
    elif [ -d "etc_jobs_2x2" ] && [ "$FORCE_MAKE" -eq 0 ]; then
        echo "etc_jobs_2x2 exists, skipping (set FORCE_MAKE=1 to overwrite)"
    else
        echo "Creating etc_jobs_2x2 from template uves_etc_inputform_2x2.json"
        "$PYTHON" 02_make_etc_jobs.py "$FLUXED_CATALOGUE" uves_etc_inputform_2x2.json \
            --outdir etc_jobs_2x2 --exptime "$EXPTIME"
    fi
    echo
fi

for dir in "${MODE_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "Missing directory: $dir" >&2
        exit 3
    fi

    mapfile -t jsons < <(find "$dir" -type f -name '*.json' ! -name '*.out.json' | sort)

    if [ ${#jsons[@]} -eq 0 ]; then
        echo "No input .json files found in $dir" >&2
        exit 4
    fi

    echo "Running ${#jsons[@]} ETC jobs in $dir"
    n_skipped=0
    for json in "${jsons[@]}"; do
        out="${json%.json}.out.json"
        if [ "$SKIP_EXISTING_OUT" -ne 0 ] && is_valid_etc_output "$out"; then
            n_skipped=$((n_skipped + 1))
            continue
        fi
        echo "  -> $json"
        "$PYTHON" "$ETC_CLI" uves "$json" -s "$SERVER" -o "$out"
    done
    if [ "$n_skipped" -gt 0 ]; then
        echo "  skipped $n_skipped job(s) with an existing valid .out.json " \
             "(SKIP_EXISTING_OUT=1)"
    fi
    echo
    echo "Finished $dir -> outputs written beside the input JSON files"
    echo "(each input produced its matching .out.json file)"
    echo
        # Collect SNR/results for this mode into a summary CSV. Uses the fluxed
        # catalogue, since make_etc_jobs has already resolved and cached
        # ra_deg/dec_deg into it (see FLUXED_CATALOGUE / CATALOGUE env vars).
        base=$(basename "$dir")
        if [ "$base" = "etc_jobs_2x2" ] || printf '%s' "$base" | grep -q "2x2"; then
            outcsv="out2x2.csv"
        else
            outcsv="out.csv"
        fi
        echo "Collecting SNR results for $dir -> $outcsv"
        "$PYTHON" 04_collect_etc_snr.py "$dir" --catalogue "$FLUXED_CATALOGUE" --output "$outcsv"
        echo
        # Update by-time table with visibility scheduling info
        if [ -f 05_calc_visibility_for_exposures.py ]; then
            echo "Running visibility check and merging results into $outcsv"
            "$PYTHON" 05_calc_visibility_for_exposures.py "$outcsv" \
                --out "$outcsv" \
                --tel-overhead "$TEL_OVERHEAD" \
                --instr-overhead "$INSTR_OVERHEAD" \
                --acq-overhead "$ACQ_OVERHEAD" \
                --readout-per-spec "$READOUT_PER_SPEC" \
                --n-spec "$N_SPEC" \
                --visits "$VISITS_PER_YEAR" \
                --per-visit "$EXPOSURES_PER_VISIT" \
                --year "$VIS_YEAR" \
                --step "$VIS_STEP" \
                --site "$VIS_SITE"
            echo
        else
            echo "05_calc_visibility_for_exposures.py not found; skipping visibility update" >&2
        fi
done

echo "Done: ${MODE_DIRS[*]}"
