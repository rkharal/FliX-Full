#!/usr/bin/env bash
#------------------------------------------------------------
# HeatMap Trials: Single Threaded + Tiled Tests (Parametrized Node/Cache)
# Runs full experiment + plotting for each DIV in {1,2,4,8}
#------------------------------------------------------------
set -u -o pipefail

cd ..

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
    NodeSize=0
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

echo "Running HeatMap Trials with common parameters:"
echo "  Xval=$Xval, Yval=$Yval, GrowthVal=$GrowthVal"
echo "  NodeSizeInput=$NodeSizeInput, CachelineSizeInput=$CachelineSizeInput"
echo "  NodeSize=$NodeSize, MaxNode=$MaxNode, TileSize=$TileSize"
echo "------------------------------------------------------------"

# Node/Cache CMake defines
if [[ "$CachelineSizeInput" -eq 0 ]]; then
  NodeCacheString=" -DNODESIZE=${NodeSizeInput}"
else
  NodeCacheString=" -DNODESIZE=0 -DCACHE_LINE_SIZE=${CachelineSizeInput}"
fi

TARGET_DIR="july7Exp/hybrid_params_tests/32bit/BaselineTests"
mkdir -p "$TARGET_DIR"

# ---------- Build helper (per-run log, fail fast) ----------
build_and_make() {
  local base_flags="$1"
  local extra_flags="$2"

  local cmake_cmd=(cmake .. -DIFDEFS:STRING="${base_flags} ${extra_flags}")
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

# ======================= OUTER LOOP OVER DIV =======================
#for Div in 1 2 4 8; do

for Div in 1 8; do

  echo "===== Starting experiments with DIV=$Div ====="

  CSV_FILES=()
  # Include DIV in the *first half* of the basename, right after R_X..Y..
  OUT_BASENAME="(B_X${Xval}Y${Yval}_DIV${Div})NS${NodeSizeInput}_CL${CachelineSizeInput}_x${GrowthVal}_32bit_25_26_4rounds"

  # Base flags for Single-threaded and Tiled runs with current DIV
  BASE_IFDEFS_SINGLE="-DDENSEKEYGEN -DXVAL=${Xval} -DYVAL=${Yval} -DTOTALRUNS=1 -DMAIN_32 -DDIV=${Div} -DPROCESS_INSERTS_SINGLETHREAD -DMAX_NODE=${MaxNode} -DDEFINE_TILE_SIZE=${TileSize}${NodeCacheString}"
  BASE_IFDEFS_TILED="-DDENSEKEYGEN -DXVAL=${Xval} -DYVAL=${Yval} -DTOTALRUNS=1 -DMAIN_32 -DDIV=${Div} -DTILE_INSERTS -DTILE_INSERTS_C -DINSERTS_TILE_BULK_ONLY -DMAX_NODE=${MaxNode} -DDEFINE_TILE_SIZE=${TileSize}${NodeCacheString}"
  BASE_IFDEFS_BASELINES="-DDENSEKEYGEN -DXVAL=${Xval} -DYVAL=${Yval} -DTOTALRUNS=1 -DMAIN_32 -DDIV=${Div} -DPROCESS_INSERTS_SINGLETHREAD -DALL_SHIFT_RIGHT -DBASELINES -DMAX_NODE=${MaxNode} -DDEFINE_TILE_SIZE=${TileSize}${NodeCacheString}"


##-DBASELINES -DHASHTABLE -DMIN_KEYS_INSERT=3 -DNODE_THRESHOLD=75 -DALL_SHIFT_RIGHT -DMAX_NODE=16 -DDEFINE_TILE_SIZE=16 -DTOTALRUNS=1 -DDIV=1 -DREGKEYGEN -DPROCESS_INSERTS_SINGLETHREAD -DST_DELETE_TOMBSTONES -DBASELINES -DHASHTABLE_SLAB -DMAIN_32
  # --- Experiments ---
  rm -f data_cache/*.* || true

  # ST: Shift Right
  OUT_TXT="${OUT_BASENAME}+{AllShftRght_ST}.txt"
  OUT_CSV="${OUT_BASENAME}+{AllShftRght_ST}.csv"
  build_and_make "$BASE_IFDEFS_SINGLE" "-DALL_SHIFT_RIGHT"
  ./index_prototype &> "$OUT_TXT"
  cp updates.csv "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
  echo "Single Threaded All Shift Right completed (DIV=$Div)."

  # ST: Bulk
  OUT_TXT="${OUT_BASENAME}+{AllBULK_ST}.txt"
  OUT_CSV="${OUT_BASENAME}+{AllBULK_ST}.csv"
  build_and_make "$BASE_IFDEFS_SINGLE" ""
  ./index_prototype &> "$OUT_TXT"
  cp updates.csv "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
  echo "Single Threaded All Bulk completed (DIV=$Div)."

  # ST: Hybrid (MK=8)
  build_and_make "$BASE_IFDEFS_SINGLE" "-DMIN_KEYS_INSERT=8 -DHYBRID"
  OUT_TXT="${OUT_BASENAME}+{Hybrid_ST_NTNA_MK8}.txt"
  OUT_CSV="${OUT_BASENAME}+{Hybrid_ST_NTNA_MK8}.csv"
  ./index_prototype &> "$OUT_TXT"
  cp updates.csv "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
  echo "Single Threaded Hybrid (MK=8) completed (DIV=$Div)."

  # TILED: Shift Right
  OUT_TXT="${OUT_BASENAME}+{AllShftRght_TILED}.txt"
  OUT_CSV="${OUT_BASENAME}+{AllShftRght_TILED}.csv"
  build_and_make "$BASE_IFDEFS_TILED" "-DALL_SHIFT_RIGHT"
  ./index_prototype &> "$OUT_TXT"
  cp updates.csv "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
  echo "Tiled All Shift Right completed (DIV=$Div)."

  # TILED: Bulk
  OUT_TXT="${OUT_BASENAME}+{AllBULK_TILED}.txt"
  OUT_CSV="${OUT_BASENAME}+{AllBULK_TILED}.csv"
  build_and_make "$BASE_IFDEFS_TILED" ""
  ./index_prototype &> "$OUT_TXT"
  cp updates.csv "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
  echo "Tiled All Bulk completed (DIV=$Div)."

  # TILED: Hybrid (MK=8)
  build_and_make "$BASE_IFDEFS_TILED" "-DMIN_KEYS_INSERT=8 -DHYBRID"
  OUT_TXT="${OUT_BASENAME}+{Hybrid_TILED_NTNA_MK8}.txt"
  OUT_CSV="${OUT_BASENAME}+{Hybrid_TILED_NTNA_MK8}.csv"
  ./index_prototype &> "$OUT_TXT"
  cp updates.csv "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
  echo "Tiled Hybrid (MK=8) completed (DIV=$Div)."

#-----------------------------------------
# Updated with NEW BASELINES
# ------------------------ HASHTABLES RUN ----------------------
OUT_TXT="${OUT_BASENAME}+{HASHTABLE_SLAB}.txt"
OUT_CSV="${OUT_BASENAME}+{HASHTABLE_SLAB}.csv"
build_and_make "$BASE_IFDEFS_BASELINES" "-DHASHTABLE_SLAB"
./index_prototype &> "$OUT_TXT"
cp updates.csv "$OUT_CSV"
mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
echo "SLAB HASHTABLE run complete."

OUT_TXT="${OUT_BASENAME}+{HASHTABLE_WARPCORE}.txt"
OUT_CSV="${OUT_BASENAME}+{HASHTABLE_WARPCORE}.csv"
build_and_make "$BASE_IFDEFS_BASELINES" "-DHASHTABLE_WARPCORE"
./index_prototype &> "$OUT_TXT"
cp updates.csv "$OUT_CSV"
mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
echo "WARPCORE HASHTABLE run complete."

# ------------------------ GPU_BTREE RUN -------------------------------
#rm -f data_cache/*.*

OUT_TXT="${OUT_BASENAME}+{GPU_BTREE}.txt"
OUT_CSV="${OUT_BASENAME}+{GPU_BTREE}.csv"
build_and_make "$BASE_IFDEFS_BASELINES" "-DGPU_BTREE"
./index_prototype &> "$OUT_TXT"
cp updates.csv "$OUT_CSV"
mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
echo "GPU_BTREE run complete."

#------------------------------------------------------

  # --- Plot for this DIV ---
  cd "$TARGET_DIR"

  ST_FIXED_FILES=( ${OUT_BASENAME}+{AllShftRght_ST}.csv ${OUT_BASENAME}+{AllBULK_ST}.csv )
  readarray -t ST_HYBRID < <(ls ${OUT_BASENAME}+{Hybrid_ST_NT*_MK*}.csv 2>/dev/null | sort -V)
  TILE_FIXED_FILES=( ${OUT_BASENAME}+{AllShftRght_TILED}.csv ${OUT_BASENAME}+{AllBULK_TILED}.csv )
  readarray -t TILE_HYBRID < <(ls ${OUT_BASENAME}+{Hybrid_TILED_NT*_MK*}.csv 2>/dev/null | sort -V)
  BASELINE_FILES="${OUT_BASENAME}+{HASHTABLE_SLAB}.csv ${OUT_BASENAME}+{HASHTABLE_WARPCORE}.csv ${OUT_BASENAME}+{GPU_BTREE}.csv"

  ALL_FILES=(
    "${ST_FIXED_FILES[@]}"
    "${ST_HYBRID[@]}"
    "${TILE_FIXED_FILES[@]}"
    "${TILE_HYBRID[@]}"
    "${BASELINE_FILES[@]}"
  )

  FILE_COUNT=${#ALL_FILES[@]}

  echo "Ordered file list to plot for DIV=$Div (count=$FILE_COUNT):"
  printf '  %s\n' "${ALL_FILES[@]}"

  # Execute plotting for this DIV
  # python3 auto_4_Rounds_updates_correctlabels_Ncolours_noDel_Colours_Baselines_merged.py "$FILE_COUNT" "${ALL_FILES[@]}"
  python3 auto_4_Rounds_Baselines_SingleRun_TEST_exclude.py "$FILE_COUNT" "$NodeSizeInput" "$OUT_BASENAME" "${ALL_FILES[@]}"

  # return to build root for next DIV
  cd - >/dev/null 2>&1

done

echo "All experiments (across DIV=1,2,4,8) completed and results stored in $TARGET_DIR"


# python3 auto_4_Rounds_Baselines_SingleRun_TEST_exclude.py 9 5 "$OUT_BASENAME" "${ALL_FILES[@]}"
