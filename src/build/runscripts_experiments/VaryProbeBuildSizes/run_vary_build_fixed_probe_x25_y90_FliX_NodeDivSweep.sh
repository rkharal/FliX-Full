#!/usr/bin/env bash
#------------------------------------------------------------
# Update experiment driver
#   - Runs FliX and FliX_ST (single-threaded lookups) across every
#     NODESIZE in {3,4,5} x DIV in {1,2,4,8} (CACHE_LINE_SIZE fixed at 0)
#   - Varies initial build size: B=10,12,...,26
#   - Keeps probe size fixed: P=20
#   - Workload: X=25, Y=90
#   - GrowthVal=200: current benchmark uses 200% inserts and 200% deletes
#   - Runs 10 insert rounds followed by 10 delete rounds
#   - Disables range queries
#   - TOTALRUNS=5 (updated for the copied experiment script)
#   - Saves per-run .txt output and updates.csv with parameters
#------------------------------------------------------------
set -u -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

BOLD="$(printf '\033[1m')"
RESET="$(printf '\033[0m')"

TARGET_DIR="June2026_Experiments/Updates/VaryBuildSize_FixedProbe20/X25Y90_Growth200_R10_RQOFF_FliX_NodeDivSweep"
mkdir -p "$TARGET_DIR"

XVAL_VALUE=25
YVAL_VALUE=90
GrowthVal=200
TOTALRUNS=5
ROUNDS=10
PROBESIZE=20
LSM_CHUNK_SIZE_LOG=16
RQ_SIZE=8
SKIP_RANGE_QUERIES=1
CACHE_LINE_SIZE_INPUT=0

#declare -a BUILDSIZES=(10 12 14 16 18 20 22 24 26)

declare -a BUILDSIZES=(28)
declare -a NODESIZE_VALUES=(3 4 5)
declare -a DIV_VALUES=(1 2 4 8)

declare -a VARIANT_NAMES=(
  "FliX"
  "FliX_ST"
)

declare -a VARIANT_FLAGS=(
  ""
  "-DSINGLE_THREADED_LOOKUPS"
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
  local build_size_log="$1"
  local node_size_input="$2"
  local div_value="$3"
  local cacheline_size_input="$4"
  local variant_name="$5"
  local variant_flags="$6"

  local max_node
  local tile_size

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

  local out_basename="Updates_VaryBuild_X${x_label}Y${y_label}_RQOFF_DIV${div_value}_NS${node_size_input}_CL${cacheline_size_input}_x${GrowthVal}_32bit_B${build_size_log}_P${PROBESIZE}_${ROUNDS}ROUNDS_${TOTALRUNS}RUNS_${variant_name}"
  local out_txt="${TARGET_DIR}/${out_basename}.txt"
  local out_csv="${TARGET_DIR}/${out_basename}.csv"

  rm -f "$out_txt" "$out_csv" updates.csv
  rm -f data_cache/*.* 2>/dev/null || true

  local common_flags="-DNODESIZE=${node_size_input} -DCACHE_LINE_SIZE=${cacheline_size_input} -DDENSEKEYGEN -DXVAL=${XVAL_VALUE} -DYVAL=${YVAL_VALUE} -DTOTALRUNS=${TOTALRUNS} -DMAIN_32 -DDIV=${div_value} -DTILE_INSERTS -DTILE_INSERTS_C -DINSERTS_TILE_BULK_ONLY -DTILE_DELETES -DDELETES_TILE_BULK -DMAX_NODE=${max_node} -DDEFINE_TILE_SIZE=${tile_size} -DINITIAL_BUILD_SIZE=${build_size_log} -DINITIAL_PROBE_SIZE=${PROBESIZE} -DROUNDS_NUMBER=${ROUNDS} -DLSM_CHUNK_SIZE_LOG=${LSM_CHUNK_SIZE_LOG} -DRQ_SIZE=${RQ_SIZE} -DSKIP_RANGE_QUERIES"
  local all_flags="${common_flags} ${variant_flags}"

  {
    echo "==================== RUN PARAMETERS ===================="
    echo "Variant=${variant_name}"
    echo "Xval=${XVAL_VALUE}"
    echo "Yval=${YVAL_VALUE}"
    echo "GrowthVal=${GrowthVal}"
    echo "NodeSizeInput=${node_size_input}"
    echo "DIV=${div_value}"
    echo "CachelineSizeInput=${cacheline_size_input}"
    echo "RQ_SIZE=${RQ_SIZE}"
    echo "RANGE_QUERY_SIZE=$((1 << RQ_SIZE))"
    echo "SKIP_RANGE_QUERIES=${SKIP_RANGE_QUERIES}"
    echo "TOTALRUNS=${TOTALRUNS}"
    echo "ROUNDS_NUMBER=${ROUNDS}"
    echo "INITIAL_BUILD_SIZE=${build_size_log}"
    echo "INITIAL_PROBE_SIZE=${PROBESIZE}"
    echo "MAX_NODE=${max_node}"
    echo "DEFINE_TILE_SIZE=${tile_size}"
    echo "OutputTxt=${out_txt}"
    echo "OutputCsv=${out_csv}"
    echo "Flags=${all_flags}"
    echo "NOTE: FliX and FliX_ST NODESIZE x DIV sweep. Build size is swept from 10 to 26 by 2. Probe size is fixed at 20."
    echo "NOTE: Current benchmarks_updates.cuh config uses 200% inserts and 200% deletes."
    echo "NOTE: With ROUNDS_NUMBER=10 and non-combined updates, this is 10 insert rounds then 10 delete rounds."
    date
    echo "========================================================"
  } > "$out_txt"

  echo "------------------------------------------------------------"
  echo "Running ${variant_name}: X=${XVAL_VALUE} Y=${YVAL_VALUE} B=${build_size_log} P=${PROBESIZE} NS=${node_size_input} DIV=${div_value} CL=${cacheline_size_input} RQOFF"
  echo "Output: $out_txt"
  echo "------------------------------------------------------------"

  if ! build_and_make "" "$all_flags" "$out_txt"; then
    print_bold_failure "Build failed for ${variant_name} X=${XVAL_VALUE} Y=${YVAL_VALUE} B=${build_size_log} P=${PROBESIZE} NS=${node_size_input} DIV=${div_value} CL=${cacheline_size_input}" "$out_txt"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  {
    echo "==================== EXECUTION ===================="
    echo "Running: ./index_prototype"
  } >> "$out_txt"

  if ! ./index_prototype >> "$out_txt" 2>&1; then
    print_bold_failure "Run failed for ${variant_name} X=${XVAL_VALUE} Y=${YVAL_VALUE} B=${build_size_log} P=${PROBESIZE} NS=${node_size_input} DIV=${div_value} CL=${cacheline_size_input}" "$out_txt"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  if [[ ! -f updates.csv ]]; then
    print_bold_failure "Run completed but updates.csv was not produced for ${variant_name} X=${XVAL_VALUE} Y=${YVAL_VALUE} B=${build_size_log} P=${PROBESIZE} NS=${node_size_input} DIV=${div_value} CL=${cacheline_size_input}" "$out_txt"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  cp updates.csv "$out_csv" || {
    print_bold_failure "Failed to copy updates.csv for ${variant_name} X=${XVAL_VALUE} Y=${YVAL_VALUE} B=${build_size_log} P=${PROBESIZE} NS=${node_size_input} DIV=${div_value} CL=${cacheline_size_input}" "$out_txt"
    FAILURES=$((FAILURES + 1))
    return 0
  }

  CSV_FILES+=("$out_csv")
  SUCCESSFUL=$((SUCCESSFUL + 1))
  echo "Completed ${variant_name}: saved $out_csv"
}

for build_size_log in "${BUILDSIZES[@]}"; do
  for node_size_input in "${NODESIZE_VALUES[@]}"; do
    for div_value in "${DIV_VALUES[@]}"; do
      for idx in "${!VARIANT_NAMES[@]}"; do
        run_one_test "$build_size_log" "$node_size_input" "$div_value" "$CACHE_LINE_SIZE_INPUT" \
          "${VARIANT_NAMES[$idx]}" "${VARIANT_FLAGS[$idx]}"
      done
    done
  done
done

echo "============================================================"
echo "Update varying-build fixed-probe FliX/FliX_ST node-size/DIV sweep finished."
echo "Successful runs: $SUCCESSFUL"
echo "Failed runs:     $FAILURES"
echo "Target dir:      $TARGET_DIR"
echo "CSV files:"
printf '  %s\n' "${CSV_FILES[@]}"
echo "============================================================"

if [[ "$FAILURES" -ne 0 ]]; then
  printf '%sCompleted with %d failed test(s). See bold output above and matching .txt files in %s.%s\n' "$BOLD" "$FAILURES" "$TARGET_DIR" "$RESET" >&2
fi
