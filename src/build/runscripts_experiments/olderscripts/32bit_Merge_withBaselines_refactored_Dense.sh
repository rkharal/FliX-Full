#!/bin/bash
#------------------------------------------------------------
# Combined Script: Single Threaded Tests + TiledTests (Ordered Output)
#------------------------------------------------------------

# go to correct directory
cd ..

# ======================= COMMON OUTPUT & TARGET =======================
TARGET_DIR="july7Exp/hybrid_params_tests/32bit/MergedTests"
mkdir -p "$TARGET_DIR"

CSV_FILES=()

# Shared output base name for all runs
OUT_BASENAME="Dense(X25Y90)x200_25_26_Dense_Merge_4rounds_200Growth_NS4_32bit"

# ======================= COMMON CMAKE FLAGS =======================
# Base flags for Single Threaded Runs
BASE_IFDEFS_SINGLE="-DDENSEKEYGEN -DTOTALRUNS=1 -DMAIN_32 -DDIV=1 -DPROCESS_INSERTS_SINGLETHREAD -DMAX_NODE=16 -DDEFINE_TILE_SIZE=16"

# Base flags for TiledTests Runs
BASE_IFDEFS_TILED="-DDENSEKEYGEN -DTOTALRUNS=1 -DMAIN_32 -DDIV=4 -DMAX_NODE=16 -DDEFINE_TILE_SIZE=16"

# Helper to build and compile with additional flags per run (fail-fast with clear error output, tee to build.log)
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


# ================================================================
# === Part 1: SINGLE THREADED EXPERIMENTS (from first script) ===
# ================================================================

# --- Run 1: Single_Thread_SHIFT RIGHT ONLY ---
rm -f data_cache/*.*
OUT_TXT="${OUT_BASENAME}+{AllShftRght_ST}.txt"
OUT_CSV="${OUT_BASENAME}+{AllShftRght_ST}.csv"
build_and_make "$BASE_IFDEFS_SINGLE" " -DALL_SHIFT_RIGHT"
./index_prototype &> "$OUT_TXT"
cp updates.csv "$OUT_CSV"
mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
echo "Single Threaded All Shift Right experiment completed."

# --- Run 2: Single_Thread_BULK ONLY ---
OUT_TXT="${OUT_BASENAME}+{AllBULK_ST}.txt"
OUT_CSV="${OUT_BASENAME}+{AllBULK_ST}.csv"
build_and_make "$BASE_IFDEFS_SINGLE" ""
./index_prototype &> "$OUT_TXT"
cp updates.csv "$OUT_CSV"
mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
echo "Single Threaded All Bulk experiment completed."

# --- Hybrid Configs ---
NODE_THRESHOLDS=(75 90)
MIN_KEYS_INSERTS=(0 1 2 3 4 5 6 7 8)
for threshold in "${NODE_THRESHOLDS[@]}"; do
    for min_keys in "${MIN_KEYS_INSERTS[@]}"; do
        build_and_make "$BASE_IFDEFS_SINGLE" "-DMIN_KEYS_INSERT=${min_keys} -DNODE_THRESHOLD=${threshold} -DHYBRID"
        OUT_TXT="${OUT_BASENAME}+{Hybrid_ST_NT${threshold}_MK${min_keys}}.txt"
        OUT_CSV="${OUT_BASENAME}+{Hybrid_ST_NT${threshold}_MK${min_keys}}.csv"
        ./index_prototype &> "$OUT_TXT"
        cp updates.csv "$OUT_CSV"
        mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
        CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
        echo "Single Threaded Hybrid: NODE_THRESHOLD=$threshold, MIN_KEYS_INSERT=$min_keys"
    done
done

# ================================================================
# === Part 2: TILED TESTS (from second script) ===
# ================================================================

# ------------------------ GPU_BTREE RUN -------------------------------
build_and_make "$BASE_IFDEFS_SINGLE" "-DBASELINES"
OUT_TXT="${OUT_BASENAME}+{GPU_BTREE}.txt"
OUT_CSV="${OUT_BASENAME}+{GPU_BTREE}.csv"
./index_prototype &> "$OUT_TXT"
cp updates.csv "$OUT_CSV"
mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
echo "GPU_BTREE run complete."

# --------------------- Run 1: All Shift Right ---------------------
OUT_TXT="${OUT_BASENAME}+{AllShftRght_TILED}.txt"
OUT_CSV="${OUT_BASENAME}+{AllShftRght_TILED}.csv"
build_and_make "$BASE_IFDEFS_TILED" "-DTILE_INSERTS -DTILE_INSERTS_C -DINSERTS_TILE_BULK_ONLY -DALL_SHIFT_RIGHT"
./index_prototype &> "$OUT_TXT"
cp updates.csv "$OUT_CSV"
mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
echo "Tiled All Shift Right experiment completed."

# --------------------- Run 2: All Bulk ----------------------------
OUT_TXT="${OUT_BASENAME}+{AllBULK_TILED}.txt"
OUT_CSV="${OUT_BASENAME}+{AllBULK_TILED}.csv"
build_and_make "$BASE_IFDEFS_TILED" "-DTILE_INSERTS -DTILE_INSERTS_C -DINSERTS_TILE_BULK_ONLY"
./index_prototype &> "$OUT_TXT"
cp updates.csv "$OUT_CSV"
mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
echo "Tiled All Bulk experiment completed."

# ----------------- Run 3: Hybrid Configs --------------------------
NODE_THRESHOLDS=(75 90)
MIN_KEYS_INSERTS=(0 1 2 3 4 5 6 7 8)
for threshold in "${NODE_THRESHOLDS[@]}"; do
    for min_keys in "${MIN_KEYS_INSERTS[@]}"; do
        build_and_make "$BASE_IFDEFS_TILED" "-DTILE_INSERTS -DTILE_INSERTS_C -DINSERTS_TILE_BULK_ONLY -DMIN_KEYS_INSERT=${min_keys} -DNODE_THRESHOLD=${threshold} -DHYBRID"
        OUT_TXT="${OUT_BASENAME}+{Hybrid_TILED_NT${threshold}_MK${min_keys}}.txt"
        OUT_CSV="${OUT_BASENAME}+{Hybrid_TILED_NT${threshold}_MK${min_keys}}.csv"
        ./index_prototype &> "$OUT_TXT"
        cp updates.csv "$OUT_CSV"
        mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
        CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")
        echo "Tiled Hybrid: NODE_THRESHOLD=$threshold, MIN_KEYS_INSERT=$min_keys"
    done
done

# ================================================================
# === Final Plotting Step (Ordered Combination) ===
# ================================================================

cd "$TARGET_DIR"

# Build ordered groups
# 1) ST fixed (AllShftRght_ST, AllBULK_ST)
ST_FIXED_FILES=( ${OUT_BASENAME}+{AllShftRght_ST}.csv ${OUT_BASENAME}+{AllBULK_ST}.csv )

# 2) ST hybrids (Hybrid_ST_NT*_MK*)
#    Sorted by version-aware order
readarray -t ST_HYBRID < <(ls ${OUT_BASENAME}+{Hybrid_ST_NT*_MK*}.csv 2>/dev/null | sort -V)

# 3) TILE fixed (AllShftRght_TILED, AllBULK_TILED)
TILE_FIXED_FILES=( ${OUT_BASENAME}+{AllShftRght_TILED}.csv ${OUT_BASENAME}+{AllBULK_TILED}.csv )

# 4) TILE hybrids (Hybrid_TILED_NT*_MK*)
readarray -t TILE_HYBRID < <(ls ${OUT_BASENAME}+{Hybrid_TILED_NT*_MK*}.csv 2>/dev/null | sort -V)

# 5) Baseline GPU_BTREE (last)
BASELINE_FILES=( ${OUT_BASENAME}+{GPU_BTREE}.csv )

# Combine in required order
ALL_FILES=(
  "${ST_FIXED_FILES[@]}"
  "${ST_HYBRID[@]}"
  "${TILE_FIXED_FILES[@]}"
  "${TILE_HYBRID[@]}"
  "${BASELINE_FILES[@]}"
)

# Count files robustly
FILE_COUNT=${#ALL_FILES[@]}

echo "Ordered file list to plot (count=$FILE_COUNT):"
printf '  %s\n' "${ALL_FILES[@]}"

# Run plot script with ordered arguments
python3 auto_4_Rounds_updates_correctlabels_Ncolours_noDel_Colours_Baselines_merged.py "$FILE_COUNT" "${ALL_FILES[@]}"

echo "All experiments completed and results stored in $TARGET_DIR"
