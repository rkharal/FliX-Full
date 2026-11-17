#!/usr/bin/env bash
# file: ICDE_Revision/run_unsorted_probes_vary_probe_fixed_build25.sh
#------------------------------------------------------------
# ICDE revision: "unsorted probes" experiment
#
#   - Uses benchmark_updates_unsorted() from benchmarks_updates_unsorted.cuh for ALL variants
#     (selected via -DUNSORTED_PROBE_BENCH in main.cu; no mixed/concurrent batches)
#   - Baselines (WARPCORE, SLAB, GPU_BTREE, LSM_TREE) receive UNSORTED probes
#   - FliX receives SORTED probes; its GPU sort (and scatter-back) cost is recorded
#       probe_time_ms            = FliX lookup only
#       sort_time_ms             = FliX probe sort
#       scatter_time_ms          = FliX scatter back to original order
#       probe_plus_sort_time_ms  = lookup + sort            (FliX query cost incl. sorting)
#       probe_total_time_ms      = lookup + sort + scatter
#     (for baselines sort/scatter are 0, so probe_plus_sort == probe_time)
#   - Initial build size fixed at 2^25
#   - Probe size swept 2^8 .. 2^28
#   - 6 rounds, growth 200% (6 insert rounds followed by 6 delete rounds)
#   - Range queries / successor queries / rebuild are disabled in the benchmark
#   - Saves per-run .txt + updates.csv, then merges and plots
#
# Output: build/ICDE/UnsortedExperiments/
#
# Usage:
#   ./run_unsorted_probes_vary_probe_fixed_build25.sh              # run + plot
#   ./run_unsorted_probes_vary_probe_fixed_build25.sh --plot-only  # re-plot existing results
#
# Env overrides: TOTALRUNS, PROBE_SIZE_LOG_START/END/STEP, ROUNDS, BUILD_SIZE_LOG
#------------------------------------------------------------
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This script lives in build/runscripts/ICDE_Revision -> build dir is two levels up
BUILD_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_DIR="$(cd "${BUILD_DIR}/.." && pwd)"
cd "$BUILD_DIR" || exit 1

BOLD="$(printf '\033[1m')"
RESET="$(printf '\033[0m')"

PLOT_ONLY=0
if [[ "${1:-}" == "--plot-only" ]]; then
  PLOT_ONLY=1
fi

TARGET_DIR="ICDE/UnsortedExperiments"
PLOTS_DIR="${TARGET_DIR}/plots"
RUN_MANIFEST="${TARGET_DIR}/runs_manifest.csv"

XVAL_VALUE=25
YVAL_VALUE=90
GrowthVal=200
TOTALRUNS="${TOTALRUNS:-5}"
ROUNDS="${ROUNDS:-6}"

BUILD_SIZE_LOG="${BUILD_SIZE_LOG:-25}"
BUILD_SIZE_ACTUAL=$((1 << BUILD_SIZE_LOG))

PROBE_SIZE_LOG_START="${PROBE_SIZE_LOG_START:-8}"
PROBE_SIZE_LOG_END="${PROBE_SIZE_LOG_END:-28}"
PROBE_SIZE_LOG_STEP="${PROBE_SIZE_LOG_STEP:-2}"

LSM_CHUNK_SIZE_LOG=16
RQ_SIZE=8

declare -a NODE_CACHE_PAIRS=(
  "5 0"
)

# FliX (sorted probes + sort cost) and the 4 baselines (unsorted probes)
declare -a VARIANT_NAMES=(
  "FliX"
  "HASHTABLE_WARPCORE"
  "HASHTABLE_SLAB"
  "GPU_BTREE"
  "LSM_TREE"
)

declare -a VARIANT_FLAGS=(
  ""
  "-DBASELINES -DHASHTABLE_WARPCORE"
  "-DBASELINES -DHASHTABLE_SLAB"
  "-DBASELINES -DGPU_BTREE"
  "-DBASELINES -DLSM_TREE"
)

FAILURES=0
SUCCESSFUL=0
CSV_FILES=()

sanitize_value() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/_}"
  printf '%s' "$value"
}

print_bold_failure() {
  local message="$1"
  local output_file="$2"
  printf '%sERROR: %s%s\n' "$BOLD" "$message" "$RESET" >&2
  printf '%sOUTPUT FILE: %s%s\n' "$BOLD" "$output_file" "$RESET" >&2
}

build_and_make() {
  local base_flags="$1"
  local extra_flags="$2"
  local output_txt="$3"

  local cmake_cmd=(cmake .. -DIFDEFS:STRING="${base_flags} ${extra_flags}")
  local make_clean_cmd=(make clean)
  local make_build_cmd=(make -j"$(nproc)")

  rm -f CMakeCache.txt build.log

  {
    echo "==================== BUILD ===================="
    echo "PWD=$(pwd)"
    echo "IFDEFS(base)=${base_flags}"
    echo "IFDEFS(extra)=${extra_flags}"
    echo "Running: ${cmake_cmd[*]}"
  } >> "$output_txt"

  "${cmake_cmd[@]}" >> "$output_txt" 2>&1 || return 1

  echo "Running: ${make_clean_cmd[*]}" >> "$output_txt"
  "${make_clean_cmd[@]}" >> "$output_txt" 2>&1 || return 1

  echo "Running: ${make_build_cmd[*]}" >> "$output_txt"
  "${make_build_cmd[@]}" >> "$output_txt" 2>&1 || return 1

  return 0
}

run_one_test() {
  local probe_size_log="$1"
  local node_size_input="$2"
  local cacheline_size_input="$3"
  local variant_name="$4"
  local variant_flags="$5"

  local max_node
  local tile_size
  local probe_size_actual
  probe_size_actual=$((1 << probe_size_log))

  if [[ "$cacheline_size_input" -eq 0 ]]; then
    max_node=$((2 ** node_size_input))
    tile_size=$((2 ** node_size_input))
  elif [[ "$cacheline_size_input" -eq 10 ]]; then
    max_node=14
    tile_size=16
  else
    echo "Unsupported CachelineSizeInput=$cacheline_size_input" >&2
    return 1
  fi

  local x_label y_label
  x_label="$(sanitize_value "$XVAL_VALUE")"
  y_label="$(sanitize_value "$YVAL_VALUE")"

  local out_basename="UnsortedProbes_VaryProbe_X${x_label}Y${y_label}_NS${node_size_input}_CL${cacheline_size_input}_x${GrowthVal}_32bit_BLOG${BUILD_SIZE_LOG}_PLOG${probe_size_log}_${ROUNDS}ROUNDS_${TOTALRUNS}RUNS_${variant_name}"
  local out_txt="${TARGET_DIR}/${out_basename}.txt"
  local out_csv="${TARGET_DIR}/${out_basename}.csv"

  rm -f "$out_txt" "$out_csv" updates.csv
  rm -f data_cache/*.* 2>/dev/null || true

  local common_flags="-DUNSORTED_PROBE_BENCH -DNODESIZE=${node_size_input} -DCACHE_LINE_SIZE=${cacheline_size_input} -DDENSEKEYGEN -DXVAL=${XVAL_VALUE} -DYVAL=${YVAL_VALUE} -DTOTALRUNS=${TOTALRUNS} -DMAIN_32 -DDIV=8 -DTILE_INSERTS -DTILE_INSERTS_C -DINSERTS_TILE_BULK_ONLY -DTILE_DELETES -DDELETES_TILE_BULK -DMAX_NODE=${max_node} -DDEFINE_TILE_SIZE=${tile_size} -DINITIAL_BUILD_SIZE=${BUILD_SIZE_LOG} -DINITIAL_PROBE_SIZE=${probe_size_log} -DROUNDS_NUMBER=${ROUNDS} -DGROWTH_RATE=${GrowthVal} -DLSM_CHUNK_SIZE_LOG=${LSM_CHUNK_SIZE_LOG} -DRQ_SIZE=${RQ_SIZE} -DSKIP_RANGE_QUERIES"
  local all_flags="${common_flags} ${variant_flags}"

  {
    echo "==================== RUN PARAMETERS ===================="
    echo "Variant=${variant_name}"
    echo "Xval=${XVAL_VALUE}  Yval=${YVAL_VALUE}  GrowthVal=${GrowthVal}"
    echo "NodeSizeInput=${node_size_input}  CachelineSizeInput=${cacheline_size_input}"
    echo "TOTALRUNS=${TOTALRUNS}  ROUNDS_NUMBER=${ROUNDS}"
    echo "INITIAL_BUILD_SIZE_LOG=${BUILD_SIZE_LOG} (${BUILD_SIZE_ACTUAL})"
    echo "INITIAL_PROBE_SIZE_LOG=${probe_size_log} (${probe_size_actual})"
    echo "MAX_NODE=${max_node}  DEFINE_TILE_SIZE=${tile_size}"
    echo "OutputTxt=${out_txt}"
    echo "OutputCsv=${out_csv}"
    echo "Flags=${all_flags}"
    echo "NOTE: benchmarks_updates_unsorted.cuh -- baselines: UNSORTED probes; FliX: SORTED probes + sort cost."
    echo "NOTE: ${ROUNDS} insert rounds then ${ROUNDS} delete rounds, ${GrowthVal}% growth."
    date
    echo "========================================================"
  } > "$out_txt"

  echo "------------------------------------------------------------"
  echo "Running ${variant_name}: BLOG=${BUILD_SIZE_LOG} PLOG=${probe_size_log} (${probe_size_actual} probes) NS=${node_size_input} CL=${cacheline_size_input}"
  echo "Output: $out_txt"
  echo "------------------------------------------------------------"

  if ! build_and_make "" "$all_flags" "$out_txt"; then
    print_bold_failure "Build failed for ${variant_name} BLOG=${BUILD_SIZE_LOG} PLOG=${probe_size_log}" "$out_txt"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  {
    echo "==================== EXECUTION ===================="
    echo "Running: ./index_prototype"
  } >> "$out_txt"

  if ! ./index_prototype >> "$out_txt" 2>&1; then
    print_bold_failure "Run failed for ${variant_name} BLOG=${BUILD_SIZE_LOG} PLOG=${probe_size_log}" "$out_txt"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  # Sanity check: this must be the simple unsorted benchmark, never the mixed/concurrent one.
  if grep -q "CONCURRENT MIXED-BATCH" "$out_txt"; then
    print_bold_failure "${variant_name} ran the MIXED-BATCH benchmark instead of benchmark_updates_unsorted" "$out_txt"
    FAILURES=$((FAILURES + 1))
    return 0
  fi
  if ! grep -q "Sort: .* ms, Scatter: .* ms, probe_size:" "$out_txt"; then
    print_bold_failure "${variant_name}: no unsorted-benchmark probe lines found in output" "$out_txt"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  if [[ ! -f updates.csv ]]; then
    print_bold_failure "updates.csv not produced for ${variant_name} BLOG=${BUILD_SIZE_LOG} PLOG=${probe_size_log} (skipped for memory?)" "$out_txt"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  cp updates.csv "$out_csv" || {
    print_bold_failure "Failed to copy updates.csv for ${variant_name} PLOG=${probe_size_log}" "$out_txt"
    FAILURES=$((FAILURES + 1))
    return 0
  }

  printf '%s\n' "${variant_name},${XVAL_VALUE},${YVAL_VALUE},${GrowthVal},${BUILD_SIZE_LOG},${BUILD_SIZE_ACTUAL},${probe_size_log},${probe_size_actual},${TOTALRUNS},${ROUNDS},${node_size_input},${cacheline_size_input},${out_csv},${out_txt}" >> "$RUN_MANIFEST"

  CSV_FILES+=("$out_csv")
  SUCCESSFUL=$((SUCCESSFUL + 1))
  echo "Completed ${variant_name}: saved $out_csv"
}

generate_plots() {
  local manifest_csv="$1"
  local target_dir="$2"
  local plots_dir="$3"

  mkdir -p "$plots_dir"

  python3 - "$manifest_csv" "$target_dir" "$plots_dir" <<'PY'
import csv, os, re, sys, statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

manifest_csv, target_dir, plots_dir = sys.argv[1], sys.argv[2], sys.argv[3]

def num(v):
    try:
        t = str(v).strip()
        return float(t) if t != "" else None
    except (TypeError, ValueError):
        return None

with open(manifest_csv, newline="") as f:
    manifest = list(csv.DictReader(f))
if not manifest:
    print(f"[plot] No runs in {manifest_csv}; nothing to plot.")
    sys.exit(0)

rows = []
for m in manifest:
    p = m["out_csv"]
    if not os.path.isfile(p):
        print(f"[plot] missing {p}")
        continue
    # updates.csv is LONG form: one row per (run, step, measurement) with DESCRIPTION/VALUE.
    # Pivot to one wide row per (run, step).
    wide = {}
    with open(p, newline="") as f:
        for r in csv.DictReader(f, skipinitialspace=True):
            key = (r.get("run"), r.get("step"))
            w = wide.setdefault(key, {k: v for k, v in r.items()
                                      if k not in ("DESCRIPTION", "VALUE")})
            w[r["DESCRIPTION"]] = r["VALUE"]
    for w in wide.values():
        w["variant_name"] = m["variant_name"]
        w["probe_size_log"] = m["probe_size_log"]
        w["build_size_log"] = m["build_size_log"]
        rows.append(w)
if not rows:
    print("[plot] no data rows"); sys.exit(0)

# combined CSV
fields = sorted({k for r in rows for k in r})
combined = os.path.join(target_dir, "combined_results.csv")
with open(combined, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)

def g(r, k):
    v = num(r.get(k))
    return 0.0 if v is None else v

FLIX = "FliX"
COLORS = {
    "FliX": "#1f77b4", "FliX (lookup only)": "#9ecae1",
    "HASHTABLE_WARPCORE": "#d62728", "HASHTABLE_SLAB": "#ff7f0e",
    "GPU_BTREE": "#2ca02c", "LSM_TREE": "#9467bd",
}
LABELS = {
    "FliX": "FliX",
    "HASHTABLE_WARPCORE": "Warpcore", "HASHTABLE_SLAB": "SlabHash",
    "GPU_BTREE": "GPU B-tree", "LSM_TREE": "LSM tree",
}

# Per-(variant, probe size, step) values in ms.
#   lookup[label][pl][step] -> [ms]    sort[pl][step] -> [ms]  (FliX only)
lookup = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
sortc = defaultdict(lambda: defaultdict(list))
for r in rows:
    pl = num(r.get("probe_size_log")); st = num(r.get("step"))
    if pl is None or st is None:
        continue
    pl, st = int(pl), int(st)
    v = r["variant_name"]
    lk = g(r, "probe_time_ms")
    if lk <= 0:
        continue
    lookup[v][pl][st].append(lk)                 # baselines: unsorted probes; FliX: kernel only
    if v == FLIX:
        sortc[pl][st].append(g(r, "sort_time_ms"))

probe_logs = sorted({pl for s in lookup.values() for pl in s})
BASE_ORDER = ["HASHTABLE_WARPCORE", "HASHTABLE_SLAB", "GPU_BTREE", "LSM_TREE"]
order = [o for o in ["FliX"] + BASE_ORDER if o in lookup]

def mean_over_steps(d):
    vals = [statistics.mean(v) for v in d.values() if v]
    return statistics.mean(vals) if vals else None

def mean_lookup(label, pl):
    return mean_over_steps(lookup[label][pl]) if pl in lookup[label] else None

def mean_sort(pl):
    return mean_over_steps(sortc[pl]) if pl in sortc else None

# summary CSV (mean over all steps of the run)
summary = os.path.join(target_dir, "summary_mean_query_time_ms.csv")
with open(summary, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["probe_size_log", "probe_size", "FliX_lookup_ms", "FliX_sort_ms", "FliX_lookup_plus_sort_ms"]
               + [LABELS[o] + "_ms" for o in BASE_ORDER if o in lookup])
    for pl in probe_logs:
        fl, fs = mean_lookup(FLIX, pl), mean_sort(pl)
        tot = (fl + fs) if (fl is not None and fs is not None) else None
        fmt = lambda x: "" if x is None else f"{x:.6f}"
        w.writerow([pl, 1 << pl, fmt(fl), fmt(fs), fmt(tot)]
                   + [fmt(mean_lookup(o, pl)) for o in BASE_ORDER if o in lookup])

outputs = []

# ---- Plot 1: query latency per probe size (grouped bars). FliX bar is STACKED: lookup + sort ----
for scale in ("log", "linear"):
    fig, ax = plt.subplots(figsize=(14, 6))
    n = len(order)
    width = 0.8 / max(n, 1)
    for i, o in enumerate(order):
        xs, ys = [], []
        for k, pl in enumerate(probe_logs):
            v = mean_lookup(o, pl)
            if v is None:
                continue
            xs.append(k - 0.4 + width * (i + 0.5)); ys.append(v)
        if o == FLIX:
            sorts = [(mean_sort(pl) or 0.0) for pl in probe_logs if mean_lookup(o, pl) is not None]
            ax.bar(xs, ys, width, color=COLORS[o], label="FliX: lookup")
            ax.bar(xs, sorts, width, bottom=ys, color="#fdae6b", edgecolor="black",
                   hatch="//", label="FliX: + probe sort")
        else:
            ax.bar(xs, ys, width, color=COLORS[o], label=LABELS[o] + " (unsorted probes)")
    ax.set_yscale(scale)
    ax.set_xticks(range(len(probe_logs)))
    ax.set_xticklabels([f"2^{p}" for p in probe_logs], rotation=45)
    ax.set_xlabel("Probe batch size")
    ax.set_ylabel("Query latency per batch (ms)")
    ax.set_title("Query latency vs probe size (build 2^25; mean over rounds)")
    ax.grid(True, alpha=0.3, axis="y", which="both")
    ax.legend()
    p = os.path.join(plots_dir, f"query_latency_vs_probe_size_{scale}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); outputs.append(p)

# ---- Plot 2: query latency across rounds for selected probe sizes ----
wanted = [pl for pl in (14, 20, 24, 28) if pl in probe_logs] or probe_logs[-3:]
if wanted:
    fig, axes = plt.subplots(1, len(wanted), figsize=(5.2 * len(wanted), 5), squeeze=False)
    for ax, pl in zip(axes[0], wanted):
        for o in order:
            if pl not in lookup[o]:
                continue
            steps = sorted(lookup[o][pl])
            ys = [statistics.mean(lookup[o][pl][s]) for s in steps]
            if o == FLIX:
                ys = [y + (statistics.mean(sortc[pl][s]) if sortc[pl].get(s) else 0.0)
                      for y, s in zip(ys, steps)]
            ax.plot(steps, ys, marker="o", color=COLORS[o],
                    label=LABELS[o] + (" (lookup + sort)" if o == FLIX else ""))
        ax.set_yscale("log")
        ax.set_xlabel("Round (0 = initial build; then inserts; then deletes)")
        ax.set_ylabel("Query latency (ms)")
        ax.set_title(f"Probe size 2^{pl}")
        ax.grid(True, alpha=0.3, which="both")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Query latency across update rounds")
    p = os.path.join(plots_dir, "query_latency_vs_round.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); outputs.append(p)

print(f"[plot] combined CSV: {combined}")
print(f"[plot] summary CSV:  {summary}")
for p in outputs:
    print(f"[plot] {p}")
PY
}

mkdir -p "$TARGET_DIR" "$PLOTS_DIR"

if [[ "$PLOT_ONLY" -eq 1 ]]; then
  if [[ ! -f "$RUN_MANIFEST" ]]; then
    echo "No manifest at ${BUILD_DIR}/${RUN_MANIFEST}; nothing to plot." >&2
    exit 1
  fi
  generate_plots "$RUN_MANIFEST" "$TARGET_DIR" "$PLOTS_DIR"
  exit 0
fi

# ---- Preflight: the benchmark must be selectable from main.cu ----
if [[ ! -f "${SRC_DIR}/benchmarks_updates_unsorted.cuh" ]]; then
  printf '%sERROR: %s/benchmarks_updates_unsorted.cuh not found.%s\n' "$BOLD" "$SRC_DIR" "$RESET" >&2
  exit 1
fi
if ! grep -q "UNSORTED_PROBE_BENCH" "${SRC_DIR}/main.cu"; then
  printf '%sERROR: main.cu does not yet select benchmarks_updates_unsorted.cuh via -DUNSORTED_PROBE_BENCH.%s\n' "$BOLD" "$RESET" >&2
  echo "Add to main.cu (replacing '#include \"benchmarks_updates.cuh\"'):" >&2
  echo '  #ifdef UNSORTED_PROBE_BENCH'                  >&2
  echo '  #include "benchmarks_updates_unsorted.cuh"'   >&2
  echo '  #else'                                         >&2
  echo '  #include "benchmarks_updates.cuh"'            >&2
  echo '  #endif'                                        >&2
  exit 1
fi

cat > "$RUN_MANIFEST" <<'CSV'
variant_name,xval,yval,growthval,build_size_log,build_size_actual,probe_size_log,probe_size_actual,totalruns,rounds,node_size_input,cacheline_size_input,out_csv,out_txt
CSV

for pair in "${NODE_CACHE_PAIRS[@]}"; do
  set -- $pair
  NodeSizeInput="$1"
  CachelineSizeInput="$2"

  for ((probe_size_log=PROBE_SIZE_LOG_START; probe_size_log<=PROBE_SIZE_LOG_END; probe_size_log+=PROBE_SIZE_LOG_STEP)); do
    for idx in "${!VARIANT_NAMES[@]}"; do
      run_one_test "$probe_size_log" "$NodeSizeInput" "$CachelineSizeInput" "${VARIANT_NAMES[$idx]}" "${VARIANT_FLAGS[$idx]}"
    done
  done
done

echo "============================================================"
echo "Unsorted-probe experiment finished."
echo "Successful runs: $SUCCESSFUL"
echo "Failed runs:     $FAILURES"
echo "Target dir:      ${BUILD_DIR}/${TARGET_DIR}"
echo "============================================================"

generate_plots "$RUN_MANIFEST" "$TARGET_DIR" "$PLOTS_DIR"

if [[ "$FAILURES" -ne 0 ]]; then
  printf '%sCompleted with %d failed test(s). See matching .txt files in %s.%s\n' "$BOLD" "$FAILURES" "$TARGET_DIR" "$RESET" >&2
fi
