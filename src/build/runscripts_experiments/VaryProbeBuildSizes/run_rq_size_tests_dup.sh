#!/usr/bin/env bash
#------------------------------------------------------------
# RQ_SIZE experiment driver (DUP / parallel-ownership range-query variant)
#   - Runs the new parallel-ownership range-query kernel (-DDUP_RANGE_QUERIES)
#   - Regular FliX and LSM_TREE/GPU_BTREE baselines are left available
#     below but commented out -- uncomment to also run them
#   - Varies RQ_SIZE across 8, 10, 14, 18, 20
#   - Saves per-run .txt output and updates.csv into RQ_TESTS
#   - Prints failures in bold and continues with remaining tests
#------------------------------------------------------------
set -u -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

BOLD="$(printf '\033[1m')"
RESET="$(printf '\033[0m')"

TARGET_DIR="RQ_TESTS"
mkdir -p "$TARGET_DIR"

GrowthVal=200
TOTALRUNS=5
ROUNDS=8
BUILDSIZE=25
PROBESIZE=26
LSM_CHUNK_SIZE_LOG=16

# Workloads: (Xval Yval)
declare -a WORKLOADS=(
  #"6 90"
  #"12 90"
  #"25 90"
  #"50 90"
  "90 90"
)

# NodeSize / Cacheline pairs
declare -a NODE_CACHE_PAIRS=(
  #"0 10"
  #"3 0"
  #"4 0"
  "5 0"
)

declare -a RQ_SIZES=(8 10 12 14 16 18 20)


declare -a VARIANT_NAMES=(
  "FliX-Parallel"
  #"FliX"
  #"LSM_TREE"
  #"GPU_BTREE"
)
declare -a VARIANT_FLAGS=(
  "-DDUP_RANGE_QUERIES"
  #""
  #"-DBASELINES -DLSM_TREE"
  #"-DBASELINES -DGPU_BTREE"
)

FAILURES=0
SUCCESSFUL=0
CSV_FILES=()

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
  local xval="$1"
  local yval="$2"
  local node_size_input="$3"
  local cacheline_size_input="$4"
  local rq_size="$5"
  local variant_name="$6"
  local variant_flags="$7"

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

  local out_basename="RQ_X${xval}Y${yval}_RQ${rq_size}_DIV8_NS${node_size_input}_CL${cacheline_size_input}_x${GrowthVal}_32bit_B${BUILDSIZE}_P${PROBESIZE}_${ROUNDS}ROUNDS_${variant_name}"
  local out_txt="${TARGET_DIR}/${out_basename}.txt"
  local out_csv="${TARGET_DIR}/${out_basename}.csv"

  rm -f "$out_txt" "$out_csv" updates.csv
  rm -f data_cache/*.* 2>/dev/null || true

  local common_flags="-DNODESIZE=${node_size_input} -DCACHE_LINE_SIZE=${cacheline_size_input} -DDENSEKEYGEN -DXVAL=${xval} -DYVAL=${yval} -DTOTALRUNS=${TOTALRUNS} -DMAIN_32 -DDIV=8 -DTILE_INSERTS -DTILE_INSERTS_C -DINSERTS_TILE_BULK_ONLY -DTILE_DELETES -DDELETES_TILE_BULK -DMAX_NODE=${max_node} -DDEFINE_TILE_SIZE=${tile_size} -DINITIAL_BUILD_SIZE=${BUILDSIZE} -DINITIAL_PROBE_SIZE=${PROBESIZE} -DROUNDS_NUMBER=${ROUNDS} -DLSM_CHUNK_SIZE_LOG=${LSM_CHUNK_SIZE_LOG} -DRQ_SIZE=${rq_size}"
  local all_flags="${common_flags} ${variant_flags}"

  {
    echo "==================== RUN PARAMETERS ===================="
    echo "Variant=${variant_name}"
    echo "Xval=${xval}"
    echo "Yval=${yval}"
    echo "GrowthVal=${GrowthVal}"
    echo "NodeSizeInput=${node_size_input}"
    echo "CachelineSizeInput=${cacheline_size_input}"
    echo "RQ_SIZE=${rq_size}"
    echo "RANGE_QUERY_SIZE=$((1 << rq_size))"
    echo "TOTALRUNS=${TOTALRUNS}"
    echo "ROUNDS_NUMBER=${ROUNDS}"
    echo "INITIAL_BUILD_SIZE=${BUILDSIZE}"
    echo "INITIAL_PROBE_SIZE=${PROBESIZE}"
    echo "MAX_NODE=${max_node}"
    echo "DEFINE_TILE_SIZE=${tile_size}"
    echo "OutputTxt=${out_txt}"
    echo "OutputCsv=${out_csv}"
    echo "Flags=${all_flags}"
    echo "NOTE: range-query verification (VERIFY_RANGE_QUERIES) is commented out in benchmarks_updates.cuh -- off for this run."
    date
    echo "========================================================"
  } > "$out_txt"

  echo "------------------------------------------------------------"
  echo "Running ${variant_name}: X=${xval} Y=${yval} NS=${node_size_input} CL=${cacheline_size_input} RQ_SIZE=${rq_size}"
  echo "Output: $out_txt"
  echo "------------------------------------------------------------"

  if ! build_and_make "" "$all_flags" "$out_txt"; then
    print_bold_failure "Build failed for ${variant_name} X=${xval} Y=${yval} NS=${node_size_input} CL=${cacheline_size_input} RQ_SIZE=${rq_size}" "$out_txt"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  {
    echo "==================== EXECUTION ===================="
    echo "Running: ./index_prototype"
  } >> "$out_txt"

  if ! ./index_prototype >> "$out_txt" 2>&1; then
    print_bold_failure "Run failed for ${variant_name} X=${xval} Y=${yval} NS=${node_size_input} CL=${cacheline_size_input} RQ_SIZE=${rq_size}" "$out_txt"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  if [[ ! -f updates.csv ]]; then
    print_bold_failure "Run completed but updates.csv was not produced for ${variant_name} X=${xval} Y=${yval} NS=${node_size_input} CL=${cacheline_size_input} RQ_SIZE=${rq_size}" "$out_txt"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  cp updates.csv "$out_csv" || {
    print_bold_failure "Failed to copy updates.csv for ${variant_name} X=${xval} Y=${yval} NS=${node_size_input} CL=${cacheline_size_input} RQ_SIZE=${rq_size}" "$out_txt"
    FAILURES=$((FAILURES + 1))
    return 0
  }

  CSV_FILES+=("$out_csv")
  SUCCESSFUL=$((SUCCESSFUL + 1))
  echo "Completed ${variant_name}: saved $out_csv"
}

for wl in "${WORKLOADS[@]}"; do
  set -- $wl
  Xval="$1"
  Yval="$2"

  for pair in "${NODE_CACHE_PAIRS[@]}"; do
    set -- $pair
    NodeSizeInput="$1"
    CachelineSizeInput="$2"

    for rq_size in "${RQ_SIZES[@]}"; do
      for idx in "${!VARIANT_NAMES[@]}"; do
        run_one_test "$Xval" "$Yval" "$NodeSizeInput" "$CachelineSizeInput" "$rq_size" "${VARIANT_NAMES[$idx]}" "${VARIANT_FLAGS[$idx]}"
      done
    done
  done
done

echo "============================================================"
echo "RQ_SIZE tests finished."
echo "Successful runs: $SUCCESSFUL"
echo "Failed runs:     $FAILURES"
echo "Target dir:      $TARGET_DIR"
echo "CSV files:"
printf '  %s\n' "${CSV_FILES[@]}"
echo "============================================================"

if [[ "$FAILURES" -ne 0 ]]; then
  printf '%sCompleted with %d failed test(s). See bold output above and matching .txt files in %s.%s\n' "$BOLD" "$FAILURES" "$TARGET_DIR" "$RESET" >&2
fi
