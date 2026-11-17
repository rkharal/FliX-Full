#!/usr/bin/env bash
#===============================================================================
# HeatMap Trials: Single-Threaded & Tiled + Baselines
# Runs experiments for DIV in {1,8}, captures outputs, and plots with Python.
#
# Usage:
#   ./run_heatmap_trials.sh Xval Yval GrowthVal NodeSizeInput CachelineSizeInput
#   - Exactly ONE of NodeSizeInput or CachelineSizeInput must be 0
#   - If CachelineSizeInput==10, NodeSize is inferred as 16
#
# Why set -euo pipefail? We want early, loud failure on any unexpected issue.
# Why explicit arrays for BASELINE_FILES? Prevents accidental single-element join.
# Why compute paths from script location? Avoids fragile "cd .." assumptions.
#===============================================================================
set -euo pipefail
shopt -s nullglob

# ------------------ Paths ------------------
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)
BUILD_DIR="${REPO_DIR}"
mkdir -p "${BUILD_DIR}"

# Where to store artifacts
TARGET_DIR="${REPO_DIR}/july7Exp/hybrid_params_tests/32bit/BaselineTests"
mkdir -p "${TARGET_DIR}"

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
  # NOTE: If you intentionally want MaxNode!=TileSize here, adjust accordingly.
  MaxNode=14
  TileSize=16
else
  MaxNode=$((2 ** NodeSize))
  TileSize=$((2 ** NodeSize))
fi

# Detect core count for make -j
if command -v nproc >/dev/null 2>&1; then NPROC=$(nproc); else NPROC=$(getconf _NPROCESSORS_ONLN || echo 4); fi

echo "Running HeatMap Trials with common parameters:"
echo "  Xval=$Xval, Yval=$Yval, GrowthVal=$GrowthVal"
echo "  NodeSizeInput=$NodeSizeInput, CachelineSizeInput=$CachelineSizeInput"
echo "  NodeSize=$NodeSize, MaxNode=$MaxNode, TileSize=$TileSize"
echo "  Build dir: $BUILD_DIR"
echo "  Target dir: $TARGET_DIR"
echo "------------------------------------------------------------"

# Node/Cache CMake defines
if [[ "$CachelineSizeInput" -eq 0 ]]; then
  NodeCacheString=" -DNODESIZE=${NodeSizeInput}"
else
  NodeCacheString=" -DNODESIZE=0 -DCACHE_LINE_SIZE=${CachelineSizeInput}"
fi

# ---------- Build helper (per-run log, fail fast) ----------
build_and_make() {
  local base_flags="$1"
  local extra_flags="$2"

  pushd "${BUILD_DIR}" >/dev/null 2>&1
  rm -f CMakeCache.txt build.log

  local cmake_cmd=(cmake -S "${REPO_DIR}" -B . -DIFDEFS:STRING="${base_flags} ${extra_flags}")
  local make_clean_cmd=(make clean)
  local make_build_cmd=(make -j"${NPROC}")

  echo "Running: ${cmake_cmd[*]}" | tee build.log
  "${cmake_cmd[@]}" 2>&1 | tee -a build.log || { echo "ERROR: cmake failed. Command: ${cmake_cmd[*]}"; popd >/dev/null 2>&1; exit 1; }

  echo "Running: ${make_clean_cmd[*]}" | tee -a build.log
  "${make_clean_cmd[@]}" 2>&1 | tee -a build.log || { echo "ERROR: make clean failed. Command: ${make_clean_cmd[*]}"; popd >/dev/null 2>&1; exit 1; }

  echo "Running: ${make_build_cmd[*]}" | tee -a build.log
  "${make_build_cmd[@]}" 2>&1 | tee -a build.log || { echo "ERROR: make build failed. Command: ${make_build_cmd[*]}"; popd >/dev/null 2>&1; exit 1; }
  popd >/dev/null 2>&1
}

# ======================= OUTER LOOP OVER DIV =======================
for Div in 1 8; do
  echo "===== Starting experiments with DIV=$Div ====="

  # Include DIV in the *first half* of the basename, right after R_X..Y..
  OUT_BASENAME="(B_X${Xval}Y${Yval}_DIV${Div})NS${NodeSizeInput}_CL${CachelineSizeInput}_x${GrowthVal}_32bit_25_26_4rounds"

  # Common flags across ST/TILED/Baselines for this DIV
  COMMON_FLAGS="-DDENSEKEYGEN -DXVAL=${Xval} -DYVAL=${Yval} -DTOTALRUNS=1 -DMAIN_32 -DDIV=${Div} -DMAX_NODE=${MaxNode} -DDEFINE_TILE_SIZE=${TileSize}${NodeCacheString}"

  # Base flags
  BASE_IFDEFS_SINGLE="${COMMON_FLAGS} -DPROCESS_INSERTS_SINGLETHREAD"
  BASE_IFDEFS_TILED="${COMMON_FLAGS} -DTILE_INSERTS -DTILE_INSERTS_C -DINSERTS_TILE_BULK_ONLY"
  # Keep baselines clean of ST/TILED-specific toggles unless required by the implementation
  BASE_IFDEFS_BASELINES="${COMMON_FLAGS} -DBASELINES"

  # --- Clean cache per DIV ---
  # rm -f "${REPO_DIR}"/data_cache/*.* || true

  # ------------------------ ST: Shift Right ------------------------
  
  rm -f data_cache/*.* || true
  OUT_TXT="${OUT_BASENAME}+{AllShftRght_ST}.txt"
  OUT_CSV="${OUT_BASENAME}+{AllShftRght_ST}.csv"
  build_and_make "$BASE_IFDEFS_SINGLE" "-DALL_SHIFT_RIGHT"
  "${BUILD_DIR}/index_prototype" &> "$OUT_TXT"
  cp "${BUILD_DIR}/updates.csv" "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  echo "Single Threaded All Shift Right completed (DIV=$Div)."

  # ------------------------ ST: Bulk -------------------------------
  OUT_TXT="${OUT_BASENAME}+{AllBULK_ST}.txt"
  OUT_CSV="${OUT_BASENAME}+{AllBULK_ST}.csv"
  build_and_make "$BASE_IFDEFS_SINGLE" ""
  "${BUILD_DIR}/index_prototype" &> "$OUT_TXT"
  cp "${BUILD_DIR}/updates.csv" "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  echo "Single Threaded All Bulk completed (DIV=$Div)."

  # ------------------------ ST: Hybrid (MK=8) ----------------------
  build_and_make "$BASE_IFDEFS_SINGLE" "-DMIN_KEYS_INSERT=8 -DHYBRID"
  OUT_TXT="${OUT_BASENAME}+{Hybrid_ST_NTNA_MK8}.txt"
  OUT_CSV="${OUT_BASENAME}+{Hybrid_ST_NTNA_MK8}.csv"
  "${BUILD_DIR}/index_prototype" &> "$OUT_TXT"
  cp "${BUILD_DIR}/updates.csv" "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  echo "Single Threaded Hybrid (MK=8) completed (DIV=$Div)."

  # ------------------------ TILED: Shift Right ---------------------
  OUT_TXT="${OUT_BASENAME}+{AllShftRght_TILED}.txt"
  OUT_CSV="${OUT_BASENAME}+{AllShftRght_TILED}.csv"
  build_and_make "$BASE_IFDEFS_TILED" "-DALL_SHIFT_RIGHT"
  "${BUILD_DIR}/index_prototype" &> "$OUT_TXT"
  cp "${BUILD_DIR}/updates.csv" "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  echo "Tiled All Shift Right completed (DIV=$Div)."

  # ------------------------ TILED: Bulk ----------------------------
  OUT_TXT="${OUT_BASENAME}+{AllBULK_TILED}.txt"
  OUT_CSV="${OUT_BASENAME}+{AllBULK_TILED}.csv"
  build_and_make "$BASE_IFDEFS_TILED" ""
  "${BUILD_DIR}/index_prototype" &> "$OUT_TXT"
  cp "${BUILD_DIR}/updates.csv" "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  echo "Tiled All Bulk completed (DIV=$Div)."

  # ------------------------ TILED: Hybrid (MK=8) -------------------
  build_and_make "$BASE_IFDEFS_TILED" "-DMIN_KEYS_INSERT=8 -DHYBRID"
  OUT_TXT="${OUT_BASENAME}+{Hybrid_TILED_NTNA_MK8}.txt"
  OUT_CSV="${OUT_BASENAME}+{Hybrid_TILED_NTNA_MK8}.csv"
  "${BUILD_DIR}/index_prototype" &> "$OUT_TXT"
  cp "${BUILD_DIR}/updates.csv" "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  echo "Tiled Hybrid (MK=8) completed (DIV=$Div)."

  # ====================== NEW BASELINES ============================
  # ------------------------ HASHTABLE: SLAB ------------------------
  OUT_TXT="${OUT_BASENAME}+{HASHTABLE_SLAB}.txt"
  OUT_CSV="${OUT_BASENAME}+{HASHTABLE_SLAB}.csv"
  build_and_make "$BASE_IFDEFS_BASELINES" "-DHASHTABLE_SLAB"
  "${BUILD_DIR}/index_prototype" &> "$OUT_TXT"
  cp "${BUILD_DIR}/updates.csv" "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  echo "SLAB HASHTABLE run complete."

  # ------------------------ HASHTABLE: WARPCORE --------------------
  OUT_TXT="${OUT_BASENAME}+{HASHTABLE_WARPCORE}.txt"
  OUT_CSV="${OUT_BASENAME}+{HASHTABLE_WARPCORE}.csv"
  build_and_make "$BASE_IFDEFS_BASELINES" "-DHASHTABLE_WARPCORE"
  "${BUILD_DIR}/index_prototype" &> "$OUT_TXT"
  cp "${BUILD_DIR}/updates.csv" "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  echo "WARPCORE HASHTABLE run complete."

  # ------------------------ GPU_BTREE ------------------------------
  OUT_TXT="${OUT_BASENAME}+{GPU_BTREE}.txt"
  OUT_CSV="${OUT_BASENAME}+{GPU_BTREE}.csv"
  build_and_make "$BASE_IFDEFS_BASELINES" "-DGPU_BTREE"
  "${BUILD_DIR}/index_prototype" &> "$OUT_TXT"
  cp "${BUILD_DIR}/updates.csv" "$OUT_CSV"
  mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
  echo "GPU_BTREE run complete."

  # ====================== PLOTTING FOR THIS DIV ====================
  pushd "$TARGET_DIR" >/dev/null 2>&1

  # Fixed ST/TILED files
  ST_FIXED_FILES=(
    "${OUT_BASENAME}+{AllShftRght_ST}.csv"
    "${OUT_BASENAME}+{AllBULK_ST}.csv"
  )
  mapfile -t ST_HYBRID < <(ls ${OUT_BASENAME}+{Hybrid_ST_NT*_MK*}.csv 2>/dev/null | sort -V)

  TILE_FIXED_FILES=(
    "${OUT_BASENAME}+{AllShftRght_TILED}.csv"
    "${OUT_BASENAME}+{AllBULK_TILED}.csv"
  )
  mapfile -t TILE_HYBRID < <(ls ${OUT_BASENAME}+{Hybrid_TILED_NT*_MK*}.csv 2>/dev/null | sort -V)

  BASELINE_FILES=(
    "${OUT_BASENAME}+{HASHTABLE_SLAB}.csv"
    "${OUT_BASENAME}+{HASHTABLE_WARPCORE}.csv"
    "${OUT_BASENAME}+{GPU_BTREE}.csv"
  )

  ALL_FILES=(
    "${ST_FIXED_FILES[@]}"
    "${ST_HYBRID[@]}"
    "${TILE_FIXED_FILES[@]}"
    "${TILE_HYBRID[@]}"
    "${BASELINE_FILES[@]}"
  )

  # Flatten quotes to preserve filenames with spaces/braces while counting
  FILE_COUNT=${#ALL_FILES[@]}

  echo "Ordered file list to plot for DIV=$Div (count=$FILE_COUNT):"
  printf '  %s\n' "${ALL_FILES[@]}"

  # Execute plotting for this DIV
  python3 "${REPO_DIR}/auto_4_Rounds_Baselines_SingleRun.py" "$FILE_COUNT" "$NodeSizeInput" "$OUT_BASENAME" "${ALL_FILES[@]}"

  popd >/dev/null 2>&1

done

echo "All experiments (across DIV=1,8) completed and results stored in $TARGET_DIR"
