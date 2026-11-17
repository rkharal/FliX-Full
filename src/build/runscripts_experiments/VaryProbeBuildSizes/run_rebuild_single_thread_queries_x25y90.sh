#!/usr/bin/env bash
#------------------------------------------------------------
# Single-threaded query rebuild sweep for FliX
#   - FliX only, with -DSINGLE_THREADED_LOOKUPS
#   - Standard tiled insert/delete update kernels
#   - Workload: X=25, Y=90, growth=300%
#   - 8 insert rounds followed by 8 delete rounds
#   - Build size log defaults to 25; pass another value as arg 1
#   - Probe size log defaults to 26; pass another value as arg 2
#   - Rebuild variants: no rebuild, every 2, every 4, every 8 rounds
#   - Range queries disabled
#   - Results under June2026/Rebuilding/X25Y90
#------------------------------------------------------------
set -u -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

# This login shell may export stale /usr/local compiler paths.
# Pin CMake to the compilers available on this system.
#export CC=/usr/bin/gcc
#export CXX=/usr/bin/g++
#export CUDAHOSTCXX=/usr/bin/g++

BOLD="$(printf '\033[1m')"
RESET="$(printf '\033[0m')"

TARGET_DIR="June2026/Rebuilding/X25Y90"
RUN_MANIFEST="${TARGET_DIR}/runs_manifest.csv"

XVAL_VALUE=25
YVAL_VALUE=90
GROWTH_RATE_VALUE=300
TOTALRUNS=5
ROUNDS=8

BUILD_SIZE_LOG="${1:-15}"
PROBE_SIZE_LOG="${2:-16}"
BUILD_SIZE_ACTUAL=$((1 << BUILD_SIZE_LOG))
PROBE_SIZE_ACTUAL=$((1 << PROBE_SIZE_LOG))

NODE_SIZE_INPUT=3
CACHELINE_SIZE_INPUT=0
DIV_FACTOR=8
LSM_CHUNK_SIZE_LOG=16
RQ_SIZE=8

if [[ "$CACHELINE_SIZE_INPUT" -eq 0 ]]; then
  MAX_NODE=$((2 ** NODE_SIZE_INPUT))
  TILE_SIZE=$((2 ** NODE_SIZE_INPUT))
elif [[ "$CACHELINE_SIZE_INPUT" -eq 10 ]]; then
  MAX_NODE=14
  TILE_SIZE=16
else
  echo "Unsupported CACHELINE_SIZE_INPUT=${CACHELINE_SIZE_INPUT}" >&2
  exit 1
fi

declare -a REBUILD_NAMES=(
  "NoRebuild"
  "RebuildEvery2"
  "RebuildEvery4"
  "RebuildEvery8"
)

declare -a REBUILD_FLAGS=(
  ""
  "-DREBUILD_ON -DREBUILD_INTERVAL=2"
  "-DREBUILD_ON -DREBUILD_INTERVAL=4"
  "-DREBUILD_ON -DREBUILD_INTERVAL=8"
)

mkdir -p "$TARGET_DIR"

cat > "$RUN_MANIFEST" <<'CSV'
variant_name,xval,yval,growth_rate,totalruns,rounds,build_size_log,build_size_actual,probe_size_log,probe_size_actual,node_size_input,cacheline_size_input,div_factor,max_node,tile_size,out_csv,out_txt,flags
CSV

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
  local all_flags="$1"
  local output_txt="$2"

  local cmake_cmd=(cmake .. -DIFDEFS:STRING="${all_flags}")
  local make_clean_cmd=(make clean)
  local make_build_cmd=(make -j"$(nproc)")

  rm -f CMakeCache.txt build.log

  {
    echo "==================== BUILD ===================="
    echo "PWD=$(pwd)"
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
}

run_variant() {
  local variant_name="$1"
  local rebuild_flags="$2"

  local common_flags="-DNODESIZE=${NODE_SIZE_INPUT} -DCACHE_LINE_SIZE=${CACHELINE_SIZE_INPUT} -DDENSEKEYGEN -DXVAL=${XVAL_VALUE} -DYVAL=${YVAL_VALUE} -DGROWTH_RATE=${GROWTH_RATE_VALUE} -DTOTALRUNS=${TOTALRUNS} -DMAIN_32 -DDIV=${DIV_FACTOR} -DTILE_INSERTS -DTILE_INSERTS_C -DINSERTS_TILE_BULK_ONLY -DTILE_DELETES -DDELETES_TILE_BULK -DMAX_NODE=${MAX_NODE} -DDEFINE_TILE_SIZE=${TILE_SIZE} -DINITIAL_BUILD_SIZE=${BUILD_SIZE_LOG} -DINITIAL_PROBE_SIZE=${PROBE_SIZE_LOG} -DROUNDS_NUMBER=${ROUNDS} -DLSM_CHUNK_SIZE_LOG=${LSM_CHUNK_SIZE_LOG} -DRQ_SIZE=${RQ_SIZE} -DSKIP_RANGE_QUERIES -DSINGLE_THREADED_LOOKUPS"
  local all_flags="${common_flags} ${rebuild_flags}"

  local out_basename="SingleThreadQueries_${variant_name}_X${XVAL_VALUE}Y${YVAL_VALUE}_Growth${GROWTH_RATE_VALUE}_B${BUILD_SIZE_LOG}_P${PROBE_SIZE_LOG}_${ROUNDS}Insert_${ROUNDS}Delete_NS${NODE_SIZE_INPUT}_CL${CACHELINE_SIZE_INPUT}_DIV${DIV_FACTOR}"
  local out_txt="${TARGET_DIR}/${out_basename}.txt"
  local out_csv="${TARGET_DIR}/${out_basename}.csv"

  rm -f "$out_txt" "$out_csv" updates.csv
  rm -f data_cache/*.* 2>/dev/null || true

  {
    echo "==================== RUN PARAMETERS ===================="
    echo "Variant=${variant_name}"
    echo "Xval=${XVAL_VALUE}"
    echo "Yval=${YVAL_VALUE}"
    echo "GrowthRate=${GROWTH_RATE_VALUE}"
    echo "TOTALRUNS=${TOTALRUNS}"
    echo "ROUNDS_NUMBER=${ROUNDS}"
    echo "Workload=${ROUNDS} insert rounds then ${ROUNDS} delete rounds"
    echo "INITIAL_BUILD_SIZE_LOG=${BUILD_SIZE_LOG}"
    echo "INITIAL_BUILD_SIZE_ACTUAL=${BUILD_SIZE_ACTUAL}"
    echo "INITIAL_PROBE_SIZE_LOG=${PROBE_SIZE_LOG}"
    echo "INITIAL_PROBE_SIZE_ACTUAL=${PROBE_SIZE_ACTUAL}"
    echo "NodeSizeInput=${NODE_SIZE_INPUT}"
    echo "CachelineSizeInput=${CACHELINE_SIZE_INPUT}"
    echo "DIV=${DIV_FACTOR}"
    echo "MAX_NODE=${MAX_NODE}"
    echo "DEFINE_TILE_SIZE=${TILE_SIZE}"
    echo "RangeQueries=OFF (-DSKIP_RANGE_QUERIES)"
    echo "LookupMode=single-threaded (-DSINGLE_THREADED_LOOKUPS)"
    echo "UpdateMode=tiled inserts/deletes"
    echo "RebuildFlags=${rebuild_flags:-<none>}"
    echo "OutputTxt=${out_txt}"
    echo "OutputCsv=${out_csv}"
    echo "Flags=${all_flags}"
    date
    echo "========================================================"
  } > "$out_txt"

  echo "------------------------------------------------------------"
  echo "Running ${variant_name}: B=2^${BUILD_SIZE_LOG} P=2^${PROBE_SIZE_LOG} X=${XVAL_VALUE} Y=${YVAL_VALUE} growth=${GROWTH_RATE_VALUE}%"
  echo "Output: ${out_txt}"
  echo "------------------------------------------------------------"

  if ! build_and_make "$all_flags" "$out_txt"; then
    print_bold_failure "Build failed for ${variant_name}" "$out_txt"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  {
    echo "==================== EXECUTION ===================="
    echo "Running: ./index_prototype"
  } >> "$out_txt"

  if ! ./index_prototype >> "$out_txt" 2>&1; then
    print_bold_failure "Run failed for ${variant_name}" "$out_txt"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  if [[ ! -f updates.csv ]]; then
    print_bold_failure "Run completed but updates.csv was not produced for ${variant_name}" "$out_txt"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  if ! cp updates.csv "$out_csv"; then
    print_bold_failure "Failed to copy updates.csv for ${variant_name}" "$out_txt"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"%s"\n' \
    "$variant_name" \
    "$XVAL_VALUE" \
    "$YVAL_VALUE" \
    "$GROWTH_RATE_VALUE" \
    "$TOTALRUNS" \
    "$ROUNDS" \
    "$BUILD_SIZE_LOG" \
    "$BUILD_SIZE_ACTUAL" \
    "$PROBE_SIZE_LOG" \
    "$PROBE_SIZE_ACTUAL" \
    "$NODE_SIZE_INPUT" \
    "$CACHELINE_SIZE_INPUT" \
    "$DIV_FACTOR" \
    "$MAX_NODE" \
    "$TILE_SIZE" \
    "$out_csv" \
    "$out_txt" \
    "$all_flags" >> "$RUN_MANIFEST"

  CSV_FILES+=("$out_csv")
  SUCCESSFUL=$((SUCCESSFUL + 1))
  echo "Completed ${variant_name}: saved ${out_csv}"
}

for idx in "${!REBUILD_NAMES[@]}"; do
  run_variant "${REBUILD_NAMES[$idx]}" "${REBUILD_FLAGS[$idx]}"
done

echo "============================================================"
echo "Single-threaded query rebuild sweep finished."
echo "Target dir:      ${TARGET_DIR}"
echo "Build size:      2^${BUILD_SIZE_LOG} (${BUILD_SIZE_ACTUAL})"
echo "Probe size:      2^${PROBE_SIZE_LOG} (${PROBE_SIZE_ACTUAL})"
echo "Rounds:          ${ROUNDS} insert + ${ROUNDS} delete"
echo "Successful runs: ${SUCCESSFUL}"
echo "Failed runs:     ${FAILURES}"
echo "Manifest:        ${RUN_MANIFEST}"
echo "CSV files:"
printf '  %s\n' "${CSV_FILES[@]}"
echo "============================================================"

if [[ "$FAILURES" -ne 0 ]]; then
  printf '%sCompleted with %d failed test(s). See matching .txt files in %s.%s\n' "$BOLD" "$FAILURES" "$TARGET_DIR" "$RESET" >&2
  exit 1
fi
