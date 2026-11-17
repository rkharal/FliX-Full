#!/usr/bin/env bash
#------------------------------------------------------------
# Rebuild interval sweep: Single-Threaded Lookups + Tiled Inserts/Deletes
#   - FliX only, -DSINGLE_THREADED_LOOKUPS, NodeSize=3, DIV=8
#   - First test run: BuildSize=2^15, ProbeSize=2^16, 8 rounds
#   - Runs: NoRebuild, then REBUILD_INTERVAL = 1,2,3,4,5,6,7,8
#   - Output: June2026/Rebuild
#------------------------------------------------------------
set -u -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

# ------------------ Args ------------------
if [[ $# -ne 5 ]]; then
  echo "Usage: $0 Xval Yval GrowthVal NodeSizeInput CachelineSizeInput" >&2
  exit 1
fi

Xval=$1
Yval=$2
GrowthVal=$3
NodeSizeInput=$4
CachelineSizeInput=$5

# Validate numeric
for v in "$Xval" "$Yval" "$GrowthVal" "$NodeSizeInput" "$CachelineSizeInput"; do
  [[ "$v" =~ ^[0-9]+$ ]] || { echo "Error: all args must be non-negative integers." >&2; exit 1; }
done

# Exactly one of NodeSizeInput or CachelineSizeInput must be zero
if ! { { [[ "$NodeSizeInput" -eq 0 ]] && [[ "$CachelineSizeInput" -ne 0 ]]; } || { [[ "$NodeSizeInput" -ne 0 ]] && [[ "$CachelineSizeInput" -eq 0 ]]; }; }; then
  echo "Error: exactly ONE of NodeSizeInput or CachelineSizeInput must be 0." >&2
  echo "Given: NodeSizeInput=$NodeSizeInput, CachelineSizeInput=$CachelineSizeInput" >&2
  exit 1
fi

# Derive NodeSize
if [[ "$CachelineSizeInput" -eq 0 ]]; then
  NodeSize="$NodeSizeInput"
else
  if [[ "$CachelineSizeInput" -eq 10 ]]; then
    NodeSize=16
  else
    echo "Unsupported CachelineSizeInput=$CachelineSizeInput" >&2
    exit 1
  fi
fi

# Derive MaxNode / TileSize
if [[ "$CachelineSizeInput" -eq 10 ]]; then
  MaxNode=14
  TileSize=16
else
  MaxNode=$((2 ** NodeSize))
  TileSize=$((2 ** NodeSize))
fi

# Fixed parameters for this experiment
# Build size is hardcoded to ~100,000,000 via -DOVERRIDE_BUILD_TO_100M
# (benchmarks_updates.cuh -> override_build_size = 100000005, in
# definitions_updates.cuh). INITIAL_BUILD_SIZE is still required to compile
# and is used only as a label/log field when the override is active (actual
# capacity/build_size come from the override, not this log value), so we
# keep it at the nearest ceil(log2(100000000))=27 for sane logging.
BuildSizeLog=27
ProbeSizeLog=26
Rounds=8
Div=8

echo "  Running Single-Threaded-Lookup Rebuild sweep with common parameters: "
echo "  Xval=$Xval, Yval=$Yval, GrowthVal=$GrowthVal, Div=$Div"
echo "  NodeSizeInput=$NodeSizeInput, CachelineSizeInput=$CachelineSizeInput"
echo "  NodeSize=$NodeSize, MaxNode=$MaxNode, TileSize=$TileSize"
echo "  BuildSize=OVERRIDE_BUILD_TO_100M (~100,000,000), ProbeSizeLog=$ProbeSizeLog, Rounds=$Rounds"
echo "------------------------------------------------------------"

TARGET_DIR="June2026/Rebuild"
mkdir -p "$TARGET_DIR"

# ---------- Build helper (per-run log, fail fast) ----------
# NOTE: IFDEFS must be semicolon-separated, not space-separated. CMake's
# add_definitions(${IFDEFS}) only splits on ";" (its list separator) -- a
# space-joined string collapses into ONE compile-definition entry, so only
# the first -D flag is ever recognized and everything after it gets appended
# as literal garbage onto that macro's value (confirmed against this repo's
# CMakeLists.txt + a live cmake repro).
build_and_make() {
  local base_flags="$1"
  local extra_flags="$2"

  local combined="$base_flags"
  if [[ -n "$extra_flags" ]]; then
    combined="${combined};${extra_flags}"
  fi

  local cmake_cmd=(cmake .. -DIFDEFS:STRING="${combined}")
  local make_clean_cmd=(make clean)
  local make_build_cmd=(make -j$(nproc))

  rm -f CMakeCache.txt build.log

  echo "Running: ${cmake_cmd[*]}" | tee build.log
  "${cmake_cmd[@]}" 2>&1 | tee -a build.log || { echo "ERROR: cmake failed. Command: ${cmake_cmd[*]}"; exit 1; }

  echo "Running: ${make_clean_cmd[*]}" | tee -a build.log
  "${make_clean_cmd[@]}" 2>&1 | tee -a build.log || { echo "ERROR: make clean failed. Command: ${make_clean_cmd[*]}"; exit 1; }

  echo "Running: ${make_build_cmd[*]}" | tee -a build.log
  "${make_build_cmd[@]}" 2>&1 | tee -a build.log || { echo "ERROR: make build failed. Command: ${make_build_cmd[*]}"; exit 1; }
}

echo "------------------------------------------------------------"
echo "===== Starting Single-Threaded-Lookup Rebuild sweep: DIV=$Div ====="

CSV_FILES=()
OUT_BASENAME="Rebuild_X${Xval}Y${Yval}_Growth${GrowthVal}_NS${NodeSizeInput}_CL${CachelineSizeInput}_DIV${Div}_B100M_P${ProbeSizeLog}_${Rounds}ROUNDS"

BASE_IFDEFS="-DNODESIZE=${NodeSizeInput};-DCACHE_LINE_SIZE=${CachelineSizeInput};-DDENSEKEYGEN;-DXVAL=${Xval};-DYVAL=${Yval};-DGROWTH_RATE=${GrowthVal};-DTOTALRUNS=5;-DMAIN_32;-DDIV=${Div};-DTILE_INSERTS;-DTILE_INSERTS_C;-DINSERTS_TILE_BULK_ONLY;-DTILE_DELETES;-DDELETES_TILE_BULK;-DMAX_NODE=${MaxNode};-DDEFINE_TILE_SIZE=${TileSize};-DINITIAL_BUILD_SIZE=${BuildSizeLog};-DINITIAL_PROBE_SIZE=${ProbeSizeLog};-DROUNDS_NUMBER=${Rounds};-DLSM_CHUNK_SIZE_LOG=16;-DPD=100;-DPU=0;-DSINGLE_THREADED_LOOKUPS;-DOVERRIDE_BUILD_TO_100M"

rm -f data_cache/*.* || true

# --------------------------------------- NO REBUILD
OUT_TXT="${OUT_BASENAME}_norebuild.txt"
OUT_CSV="${OUT_BASENAME}_norebuild.csv"
build_and_make "$BASE_IFDEFS" ""
./index_prototype &> "$OUT_TXT"
cp updates.csv "$OUT_CSV"
mv "$OUT_TXT" "$OUT_CSV" "$TARGET_DIR"/
CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
echo "NoRebuild run completed."

# --------------------------------------- REBUILD EVERY N (toggle individually below)
run_rebuild_interval() {
  local interval="$1"
  local out_txt="${OUT_BASENAME}_every${interval}.txt"
  local out_csv="${OUT_BASENAME}_every${interval}.csv"
  build_and_make "$BASE_IFDEFS" "-DREBUILD_ON;-DREBUILD_INTERVAL=${interval}"
  ./index_prototype &> "$out_txt"
  cp updates.csv "$out_csv"
  mv "$out_txt" "$out_csv" "$TARGET_DIR"/
  CSV_FILES+=("${TARGET_DIR}/${out_csv}")
  echo "Rebuild every ${interval} run completed."
}

# run_rebuild_interval 1
# run_rebuild_interval 2
# run_rebuild_interval 3
# run_rebuild_interval 4
# run_rebuild_interval 5
# run_rebuild_interval 6
# run_rebuild_interval 7
run_rebuild_interval 8

echo "============================================================"
echo "Rebuild sweep finished."
echo "Target dir:      $TARGET_DIR"
echo "Build size:      2^${BuildSizeLog}"
echo "Probe size:      2^${ProbeSizeLog}"
echo "Rounds:          $Rounds"
echo "CSV files:"
printf '  %s\n' "${CSV_FILES[@]}"
echo "============================================================"
