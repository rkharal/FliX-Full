#!/usr/bin/env bash
#------------------------------------------------------------
# Update experiment driver
#   - Runs FliX plus all baseline variants
#   - Sweeps paired initial build/probe sizes
#   - Runs workloads X6Y90, X25Y90, and X90Y90
#   - Uses one benchmark run per experiment
#   - Runs 10 insert rounds followed by 10 delete rounds
#   - Disables range queries
#   - Stores each workload in its own target directory
#   - Saves runtime output only in each experiment .txt file
#------------------------------------------------------------
set -u -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

BOLD="$(printf '\033[1m')"
RESET="$(printf '\033[0m')"

TARGET_ROOT="June2026_Experiments/Updates/VaryBuildProbePairs/X90Y90_Growth200_R10_RQOFF_AllBaselines/"

GrowthVal=200
TOTALRUNS=5
ROUNDS=10
LSM_CHUNK_SIZE_LOG=16
RQ_SIZE=8
SKIP_RANGE_QUERIES=1
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

# Workloads: (X Y)
declare -a WORKLOADS=(
  #"6 90"
  #"25 90"
  "90 90"
)

# NodeSize / Cacheline pairs (exactly one must be 0).
declare -a NODE_CACHE_PAIRS=(
  "5 0"
)

# (BUILDSIZE, PROBESIZE) pairs to sweep.
declare -a BUILD_PROBE_PAIRS=(
  "8 9"
  "10 11"
  "12 13"
  "14 15"
  "16 17"
  "18 19"
  "20 21"
  "22 23"
  "24 25"
  "25 26"
)

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
FAILURE_MESSAGES=()

sanitize_value() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/_}"
  printf '%s' "$value"
}

target_dir_for_workload() {
  local xval="$1"
  local yval="$2"
  local x_label
  local y_label
  x_label="$(sanitize_value "$xval")"
  y_label="$(sanitize_value "$yval")"
  printf '%s/X%sY%s_Growth%s_R%s_RQOFF_AllBaselines' \
    "$TARGET_ROOT" "$x_label" "$y_label" "$GrowthVal" "$ROUNDS"
}

record_failure() {
  local message="$1"
  local output_file="${2:-}"

  printf '%sERROR: %s%s\n' "$BOLD" "$message" "$RESET" >&2
  if [[ -n "$output_file" ]]; then
    printf '%sRUNTIME OUTPUT FILE: %s%s\n' \
      "$BOLD" "$output_file" "$RESET" >&2
  fi
  FAILURES=$((FAILURES + 1))
  FAILURE_MESSAGES+=("$message")
}

build_and_make() {
  local base_flags="$1"
  local extra_flags="$2"

  local cmake_cmd=(cmake .. -DIFDEFS:STRING="${base_flags} ${extra_flags}")
  local make_clean_cmd=(make clean)
  local make_build_cmd=(make -j"$BUILD_JOBS")

  rm -f CMakeCache.txt build.log

  echo "==================== BUILD ===================="
  echo "PWD=$(pwd)"
  echo "IFDEFS(base)=${base_flags}"
  echo "IFDEFS(extra)=${extra_flags}"
  echo "Running: ${cmake_cmd[*]}"
  "${cmake_cmd[@]}" || return 1

  echo "Running: ${make_clean_cmd[*]}"
  "${make_clean_cmd[@]}" || return 1

  echo "Running: ${make_build_cmd[*]}"
  "${make_build_cmd[@]}" || return 1

  return 0
}

run_one_test() {
  local xval="$1"
  local yval="$2"
  local build_size_log="$3"
  local probe_size_log="$4"
  local node_size_input="$5"
  local cacheline_size_input="$6"
  local variant_name="$7"
  local variant_flags="$8"

  local max_node
  local tile_size

  if [[ "$cacheline_size_input" -eq 0 ]]; then
    max_node=$((2 ** node_size_input))
    tile_size=$((2 ** node_size_input))
  elif [[ "$cacheline_size_input" -eq 10 ]]; then
    max_node=14
    tile_size=16
  else
    record_failure \
      "Unsupported CachelineSizeInput=${cacheline_size_input}" ""
    return 0
  fi

  local target_dir
  local x_label
  local y_label
  target_dir="$(target_dir_for_workload "$xval" "$yval")"
  x_label="$(sanitize_value "$xval")"
  y_label="$(sanitize_value "$yval")"
  mkdir -p "$target_dir"

  local out_basename="Updates_VaryBuildProbe_X${x_label}Y${y_label}_RQOFF_DIV8_NS${node_size_input}_CL${cacheline_size_input}_x${GrowthVal}_32bit_B${build_size_log}_P${probe_size_log}_${ROUNDS}ROUNDS_${TOTALRUNS}RUNS_${variant_name}"
  local out_txt="${target_dir}/${out_basename}.txt"
  local out_csv="${target_dir}/${out_basename}.csv"

  rm -f "$out_txt" "$out_csv" updates.csv
  rm -f data_cache/*.* 2>/dev/null || true

  local common_flags="-DNODESIZE=${node_size_input} -DCACHE_LINE_SIZE=${cacheline_size_input} -DDENSEKEYGEN -DXVAL=${xval} -DYVAL=${yval} -DTOTALRUNS=${TOTALRUNS} -DMAIN_32 -DDIV=8 -DTILE_INSERTS -DTILE_INSERTS_C -DINSERTS_TILE_BULK_ONLY -DTILE_DELETES -DDELETES_TILE_BULK -DMAX_NODE=${max_node} -DDEFINE_TILE_SIZE=${tile_size} -DINITIAL_BUILD_SIZE=${build_size_log} -DINITIAL_PROBE_SIZE=${probe_size_log} -DROUNDS_NUMBER=${ROUNDS} -DLSM_CHUNK_SIZE_LOG=${LSM_CHUNK_SIZE_LOG} -DRQ_SIZE=${RQ_SIZE} -DSKIP_RANGE_QUERIES"
  local all_flags="${common_flags} ${variant_flags}"

  echo "------------------------------------------------------------"
  echo "Running ${variant_name}: X=${xval} Y=${yval} B=${build_size_log} P=${probe_size_log} NS=${node_size_input} CL=${cacheline_size_input} RQOFF"
  echo "Runtime output: $out_txt"
  echo "CSV output:     $out_csv"
  echo "------------------------------------------------------------"

  # Build output is printed to the terminal only. It is not stored in out_txt.
  if ! build_and_make "" "$all_flags"; then
    record_failure \
      "Build failed for ${variant_name} X=${xval} Y=${yval} B=${build_size_log} P=${probe_size_log} NS=${node_size_input} CL=${cacheline_size_input}" \
      ""
    return 0
  fi

  echo "==================== RUNTIME EXECUTION ===================="
  echo "Running: ./index_prototype"

  # tee displays runtime output on screen and stores only runtime output in .txt.
  if ! ./index_prototype 2>&1 | tee "$out_txt"; then
    record_failure \
      "Run failed for ${variant_name} X=${xval} Y=${yval} B=${build_size_log} P=${probe_size_log} NS=${node_size_input} CL=${cacheline_size_input}" \
      "$out_txt"
    return 0
  fi

  if [[ ! -f updates.csv ]]; then
    record_failure \
      "Run completed but updates.csv was not produced for ${variant_name} X=${xval} Y=${yval} B=${build_size_log} P=${probe_size_log} NS=${node_size_input} CL=${cacheline_size_input}" \
      "$out_txt"
    return 0
  fi

  if ! cp updates.csv "$out_csv"; then
    record_failure \
      "Failed to copy updates.csv for ${variant_name} X=${xval} Y=${yval} B=${build_size_log} P=${probe_size_log} NS=${node_size_input} CL=${cacheline_size_input}" \
      "$out_txt"
    return 0
  fi

  CSV_FILES+=("$out_csv")
  SUCCESSFUL=$((SUCCESSFUL + 1))
  echo "Completed ${variant_name}: saved $out_csv"
}

if [[ "${#VARIANT_NAMES[@]}" -ne "${#VARIANT_FLAGS[@]}" ]]; then
  echo "ERROR: VARIANT_NAMES and VARIANT_FLAGS must have equal lengths" >&2
  exit 1
fi

for workload in "${WORKLOADS[@]}"; do
  set -- $workload
  Xval="$1"
  Yval="$2"

  for value in "$Xval" "$Yval"; do
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
      echo "ERROR: Non-integer workload value: $value" >&2
      exit 1
    fi
  done

  workload_target_dir="$(target_dir_for_workload "$Xval" "$Yval")"
  mkdir -p "$workload_target_dir"

  echo "============================================================"
  echo "Starting workload X=${Xval}, Y=${Yval}"
  echo "Target dir: $workload_target_dir"
  echo "============================================================"

  for build_probe_pair in "${BUILD_PROBE_PAIRS[@]}"; do
    set -- $build_probe_pair
    BUILDSIZE="$1"
    PROBESIZE="$2"

    for value in "$BUILDSIZE" "$PROBESIZE"; do
      if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Non-integer Build/Probe size: $value" >&2
        exit 1
      fi
    done

    echo "== Build/Probe: BUILDSIZE=${BUILDSIZE} PROBESIZE=${PROBESIZE}"

    for node_cache_pair in "${NODE_CACHE_PAIRS[@]}"; do
      set -- $node_cache_pair
      NodeSizeInput="$1"
      CachelineSizeInput="$2"

      for idx in "${!VARIANT_NAMES[@]}"; do
        run_one_test \
          "$Xval" "$Yval" "$BUILDSIZE" "$PROBESIZE" \
          "$NodeSizeInput" "$CachelineSizeInput" \
          "${VARIANT_NAMES[$idx]}" "${VARIANT_FLAGS[$idx]}"
      done
    done
  done

  echo "Completed workload X=${Xval}, Y=${Yval}"
done

echo "============================================================"
echo "Update varying build/probe-pair experiments finished."
echo "Successful tests: $SUCCESSFUL"
echo "Failed tests:     $FAILURES"
echo "Target root:      $TARGET_ROOT"
echo "Expected tests:   $((${#WORKLOADS[@]} * ${#BUILD_PROBE_PAIRS[@]} * ${#NODE_CACHE_PAIRS[@]} * ${#VARIANT_NAMES[@]}))"
echo "============================================================"

if [[ "$FAILURES" -ne 0 ]]; then
  printf '%sFAILED TESTS:%s\n' "$BOLD" "$RESET" >&2
  printf '  %s\n' "${FAILURE_MESSAGES[@]}" >&2
  exit 1
fi

echo "All runs completed successfully."
