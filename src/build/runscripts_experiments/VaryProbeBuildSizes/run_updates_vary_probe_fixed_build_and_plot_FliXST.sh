#!/usr/bin/env bash
# file: scripts/run_updates_vary_probe_fixed_build_and_plot.sh
#------------------------------------------------------------
# Update experiment driver
#   - Runs only FliX-ST
#   - Uses the configured build size B=2^25 (no build-size override)
#   - Varies initial probe size log2: P=8,10,...,28
#   - Workload: X=25, Y=90
#   - GrowthVal=300: benchmark uses 300% inserts and 300% deletes
#   - Uses 5 insert rounds and 5 delete rounds
#   - Executes one benchmark run per configuration
#   - Disables range queries
#   - Saves per-run .txt output and updates.csv
#   - Generates merged CSV + plots after the sweep
#------------------------------------------------------------
set -u -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

BOLD="$(printf '\033[1m')"
RESET="$(printf '\033[0m')"

TARGET_ROOT="June2026_Experiments/Updates/FixedBuild25_VaryProbeSize"

XVAL_VALUE=25
YVAL_VALUE=90
GrowthVal=300
TOTALRUNS=5

declare -a ROUNDS_VALUES=( 5 )


BUILD_SIZE_LOG=25
BUILD_SIZE_ACTUAL=$((1 << BUILD_SIZE_LOG))

## UPDATE THIS FOR QUERY BATCH RANGE
PROBE_SIZE_LOG_START=22
PROBE_SIZE_LOG_END=22
PROBE_SIZE_LOG_STEP=2

LSM_CHUNK_SIZE_LOG=16
RQ_SIZE=8
SKIP_RANGE_QUERIES=1

#declare -a NODE_CACHE_PAIRS=(
#  "5 0"
#)

declare -a NODE_CACHE_PAIRS=(
  "3 0"
)

declare -a VARIANT_NAMES=(
  "FliX-ST-New"
)

declare -a VARIANT_FLAGS=(
  "-DSINGLE_THREADED_LOOKUPS"
)

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

  if [[ -f CMakeCache.txt ]]; then
    {
      echo "---- CMakeCache extract ----"
      grep -E '(^|;)NODESIZE(:|=)|(^|;)CACHE_LINE_SIZE(:|=)|(^|;)IFDEFS(:|=)' CMakeCache.txt || true
      echo "----------------------------"
    } >> "$output_txt"
  fi

  {
    echo "Running: ${make_clean_cmd[*]}"
  } >> "$output_txt"
  "${make_clean_cmd[@]}" >> "$output_txt" 2>&1 || return 1

  {
    echo "Running: ${make_build_cmd[*]}"
  } >> "$output_txt"
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

  local x_label
  local y_label
  x_label="$(sanitize_value "$XVAL_VALUE")"
  y_label="$(sanitize_value "$YVAL_VALUE")"

  local out_basename="Updates_VaryProbe_X${x_label}Y${y_label}_RQOFF_DIV8_NS${node_size_input}_CL${cacheline_size_input}_x${GrowthVal}_32bit_BLOG${BUILD_SIZE_LOG}_PLOG${probe_size_log}_${ROUNDS}ROUNDS_${TOTALRUNS}RUNS_${variant_name}"
  local out_txt="${TARGET_DIR}/${out_basename}.txt"
  local out_csv="${TARGET_DIR}/${out_basename}.csv"

  rm -f "$out_txt" "$out_csv" updates.csv
  rm -f data_cache/*.* 2>/dev/null || true

  local common_flags="-DNODESIZE=${node_size_input} -DCACHE_LINE_SIZE=${cacheline_size_input} -DDENSEKEYGEN -DXVAL=${XVAL_VALUE} -DYVAL=${YVAL_VALUE} -DGROWTH_RATE=${GrowthVal} -DTOTALRUNS=${TOTALRUNS} -DMAIN_32 -DDIV=8 -DTILE_INSERTS -DTILE_INSERTS_C -DINSERTS_TILE_BULK_ONLY -DTILE_DELETES -DDELETES_TILE_BULK -DMAX_NODE=${max_node} -DDEFINE_TILE_SIZE=${tile_size} -DINITIAL_BUILD_SIZE=${BUILD_SIZE_LOG} -DINITIAL_PROBE_SIZE=${probe_size_log} -DROUNDS_NUMBER=${ROUNDS} -DLSM_CHUNK_SIZE_LOG=${LSM_CHUNK_SIZE_LOG} -DRQ_SIZE=${RQ_SIZE} -DSKIP_RANGE_QUERIES"
  local all_flags="${common_flags} ${variant_flags}"

  {
    echo "==================== RUN PARAMETERS ===================="
    echo "Variant=${variant_name}"
    echo "Xval=${XVAL_VALUE}"
    echo "Yval=${YVAL_VALUE}"
    echo "GrowthVal=${GrowthVal}"
    echo "NodeSizeInput=${node_size_input}"
    echo "CachelineSizeInput=${cacheline_size_input}"
    echo "RQ_SIZE=${RQ_SIZE}"
    echo "RANGE_QUERY_SIZE=$((1 << RQ_SIZE))"
    echo "SKIP_RANGE_QUERIES=${SKIP_RANGE_QUERIES}"
    echo "TOTALRUNS=${TOTALRUNS}"
    echo "ROUNDS_NUMBER=${ROUNDS}"
    echo "INITIAL_BUILD_SIZE_LOG=${BUILD_SIZE_LOG}"
    echo "INITIAL_BUILD_SIZE_ACTUAL=${BUILD_SIZE_ACTUAL}"
    echo "INITIAL_PROBE_SIZE_LOG=${probe_size_log}"
    echo "INITIAL_PROBE_SIZE_ACTUAL=${probe_size_actual}"
    echo "MAX_NODE=${max_node}"
    echo "DEFINE_TILE_SIZE=${tile_size}"
    echo "OutputTxt=${out_txt}"
    echo "OutputCsv=${out_csv}"
    echo "Flags=${all_flags}"
    echo "NOTE: FliX-ST only (-DSINGLE_THREADED_LOOKUPS)."
    echo "NOTE: INITIAL_BUILD_SIZE is 2^${BUILD_SIZE_LOG}=${BUILD_SIZE_ACTUAL}; no build-size override is enabled."
    echo "NOTE: Probe size is swept from 2^${PROBE_SIZE_LOG_START} to 2^${PROBE_SIZE_LOG_END}, step ${PROBE_SIZE_LOG_STEP} in the log2 domain."
    echo "NOTE: Current benchmarks_updates.cuh config uses 300% inserts and 300% deletes."
    echo "NOTE: With ROUNDS_NUMBER=${ROUNDS} and non-combined updates, this is ${ROUNDS} insert rounds then ${ROUNDS} delete rounds."
    date
    echo "========================================================"
  } > "$out_txt"

  echo "------------------------------------------------------------"
  echo "Running ${variant_name}: X=${XVAL_VALUE} Y=${YVAL_VALUE} BLOG=${BUILD_SIZE_LOG} PLOG=${probe_size_log} PACT=${probe_size_actual} NS=${node_size_input} CL=${cacheline_size_input} RQOFF"
  echo "Output: $out_txt"
  echo "------------------------------------------------------------"

  if ! build_and_make "" "$all_flags" "$out_txt"; then
    print_bold_failure "Build failed for ${variant_name} X=${XVAL_VALUE} Y=${YVAL_VALUE} BLOG=${BUILD_SIZE_LOG} PLOG=${probe_size_log} NS=${node_size_input} CL=${cacheline_size_input}" "$out_txt"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  {
    echo "==================== EXECUTION ===================="
    echo "Running: ./index_prototype"
  } >> "$out_txt"

  if ! ./index_prototype >> "$out_txt" 2>&1; then
    print_bold_failure "Run failed for ${variant_name} X=${XVAL_VALUE} Y=${YVAL_VALUE} BLOG=${BUILD_SIZE_LOG} PLOG=${probe_size_log} NS=${node_size_input} CL=${cacheline_size_input}" "$out_txt"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  if [[ ! -f updates.csv ]]; then
    print_bold_failure "Run completed but updates.csv was not produced for ${variant_name} X=${XVAL_VALUE} Y=${YVAL_VALUE} BLOG=${BUILD_SIZE_LOG} PLOG=${probe_size_log} NS=${node_size_input} CL=${cacheline_size_input}" "$out_txt"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  cp updates.csv "$out_csv" || {
    print_bold_failure "Failed to copy updates.csv for ${variant_name} X=${XVAL_VALUE} Y=${YVAL_VALUE} BLOG=${BUILD_SIZE_LOG} PLOG=${probe_size_log} NS=${node_size_input} CL=${cacheline_size_input}" "$out_txt"
    FAILURES=$((FAILURES + 1))
    return 0
  }

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$variant_name" \
    "$XVAL_VALUE" \
    "$YVAL_VALUE" \
    "$GrowthVal" \
    "$BUILD_SIZE_LOG" \
    "$BUILD_SIZE_ACTUAL" \
    "$probe_size_log" \
    "$probe_size_actual" \
    "$TOTALRUNS" \
    "$ROUNDS" \
    "$node_size_input" \
    "$cacheline_size_input" \
    "$out_csv" \
    "$out_txt" >> "$RUN_MANIFEST"

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
import csv
import math
import os
import re
import sys
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

manifest_csv = sys.argv[1]
target_dir = sys.argv[2]
plots_dir = sys.argv[3]

def parse_number(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None

def sanitize_filename(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "metric"

def read_csv_rows(path):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        return list(reader)

with open(manifest_csv, "r", newline="") as f:
    manifest_rows = list(csv.DictReader(f))

if not manifest_rows:
    print(f"[plot] No successful runs listed in {manifest_csv}. Skipping plot generation.")
    sys.exit(0)

combined_rows = []
for manifest_row in manifest_rows:
    csv_path = manifest_row["out_csv"]
    if not os.path.isfile(csv_path):
        print(f"[plot] Missing CSV, skipping: {csv_path}")
        continue

    run_rows = read_csv_rows(csv_path)
    if not run_rows:
        print(f"[plot] Empty or headerless CSV, skipping: {csv_path}")
        continue

    metadata = dict(manifest_row)
    for run_row in run_rows:
        merged = dict(run_row)
        merged.update(metadata)
        combined_rows.append(merged)

if not combined_rows:
    print("[plot] No data rows found after loading per-run CSV files. Skipping plot generation.")
    sys.exit(0)

preferred_field_order = [
    "variant_name",
    "xval",
    "yval",
    "growthval",
    "build_size_log",
    "build_size_actual",
    "probe_size_log",
    "probe_size_actual",
    "totalruns",
    "rounds",
    "node_size_input",
    "cacheline_size_input",
    "out_csv",
    "out_txt",
]
all_fields = set()
for row in combined_rows:
    all_fields.update(row.keys())
remaining_fields = sorted(field for field in all_fields if field not in preferred_field_order)
combined_fieldnames = preferred_field_order + remaining_fields

combined_csv_path = os.path.join(target_dir, "combined_results.csv")
with open(combined_csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=combined_fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(combined_rows)

excluded_columns = {
    "variant_name",
    "xval",
    "yval",
    "growthval",
    "build_size_log",
    "build_size_actual",
    "probe_size_log",
    "probe_size_actual",
    "totalruns",
    "rounds",
    "node_size_input",
    "cacheline_size_input",
    "out_csv",
    "out_txt",
    "Xval",
    "Yval",
    "GrowthVal",
    "NodeSizeInput",
    "CachelineSizeInput",
    "RQ_SIZE",
    "RANGE_QUERY_SIZE",
    "SKIP_RANGE_QUERIES",
    "TOTALRUNS",
    "ROUNDS_NUMBER",
    "INITIAL_BUILD_SIZE",
    "INITIAL_PROBE_SIZE",
    "MAX_NODE",
    "DEFINE_TILE_SIZE",
    "DIV",
    "LSM_CHUNK_SIZE_LOG",
}
excluded_columns_lower = {name.lower() for name in excluded_columns}

candidate_numeric_columns = []
for field in combined_fieldnames:
    if field.lower() in excluded_columns_lower:
        continue
    numeric_values = [parse_number(row.get(field)) for row in combined_rows]
    if any(value is not None for value in numeric_values):
        candidate_numeric_columns.append(field)

if not candidate_numeric_columns:
    note_path = os.path.join(plots_dir, "README.txt")
    with open(note_path, "w") as f:
        f.write("No numeric metric columns were auto-detected from the collected updates.csv files.\n")
        f.write("combined_results.csv was still generated successfully.\n")
    print(f"[plot] No numeric metrics detected. Wrote note: {note_path}")
    sys.exit(0)

probe_logs = sorted(
    {
        int(parse_number(row["probe_size_log"]))
        for row in combined_rows
        if parse_number(row.get("probe_size_log")) is not None
    }
)
probe_tick_labels = [f"2^{p}" for p in probe_logs]

build_logs = sorted(
    {
        int(parse_number(row["build_size_log"]))
        for row in combined_rows
        if parse_number(row.get("build_size_log")) is not None
    }
)
build_label = ", ".join(f"2^{b}" for b in build_logs) if build_logs else "unknown"

summary_pdf_path = os.path.join(plots_dir, "probe_sweep_all_metrics.pdf")
generated_pngs = []

with PdfPages(summary_pdf_path) as pdf:
    for metric in candidate_numeric_columns:
        series = defaultdict(lambda: defaultdict(list))

        for row in combined_rows:
            variant = row.get("variant_name", "UNKNOWN")
            probe_log_value = parse_number(row.get("probe_size_log"))
            metric_value = parse_number(row.get(metric))
            if probe_log_value is None or metric_value is None:
                continue
            probe_log_int = int(probe_log_value)
            series[variant][probe_log_int].append(metric_value)

        has_data = any(series[variant] for variant in series)
        if not has_data:
            continue

        plt.figure(figsize=(11, 6))
        plotted_any = False

        for variant in sorted(series.keys()):
            xs = sorted(series[variant].keys())
            ys = [statistics.mean(series[variant][x]) for x in xs]
            if not xs:
                continue
            plt.plot(xs, ys, marker="o", label=variant)
            plotted_any = True

        if not plotted_any:
            plt.close()
            continue

        plt.xticks(probe_logs, probe_tick_labels, rotation=45)
        plt.xlabel("Initial probe size")
        plt.ylabel(metric)
        plt.title(f"{metric} vs initial probe size (fixed build size = {build_label})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        png_name = f"{sanitize_filename(metric)}_vs_probe_size.png"
        png_path = os.path.join(plots_dir, png_name)
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        pdf.savefig(bbox_inches="tight")
        plt.close()

        generated_pngs.append(png_path)

index_path = os.path.join(plots_dir, "README.txt")
with open(index_path, "w") as f:
    f.write("Generated plot artifacts\n")
    f.write("========================\n")
    f.write(f"Combined CSV: {combined_csv_path}\n")
    f.write(f"Summary PDF: {summary_pdf_path}\n")
    f.write("PNG files:\n")
    for path in generated_pngs:
        f.write(f"  - {path}\n")

print(f"[plot] Combined CSV: {combined_csv_path}")
print(f"[plot] Summary PDF:  {summary_pdf_path}")
for path in generated_pngs:
    print(f"[plot] PNG:          {path}")
PY
}

for ROUNDS in "${ROUNDS_VALUES[@]}"; do
  TARGET_DIR="${TARGET_ROOT}/X25Y90_Growth${GrowthVal}_R${ROUNDS}_RQOFF_AllBaselines"
  PLOTS_DIR="${TARGET_DIR}/plots"
  RUN_MANIFEST="${TARGET_DIR}/runs_manifest.csv"

  mkdir -p "$TARGET_DIR" "$PLOTS_DIR"

  FAILURES=0
  SUCCESSFUL=0
  CSV_FILES=()

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
  echo "Update fixed-build varying-probe experiments finished."
  echo "Rounds:          $ROUNDS insert + $ROUNDS delete"
  echo "Successful runs: $SUCCESSFUL"
  echo "Failed runs:     $FAILURES"
  echo "Target dir:      $TARGET_DIR"
  echo "CSV files:"
  printf '  %s\n' "${CSV_FILES[@]}"
  echo "============================================================"

  generate_plots "$RUN_MANIFEST" "$TARGET_DIR" "$PLOTS_DIR"

  if [[ "$FAILURES" -ne 0 ]]; then
    printf '%sCompleted with %d failed test(s). See bold output above and matching .txt files in %s.%s\n' "$BOLD" "$FAILURES" "$TARGET_DIR" "$RESET" >&2
  fi
done
