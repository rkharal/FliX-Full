#!/bin/bash

#TARGET_DIR="july7Exp/hybrid_params_tests/32bit/NoShiftRange"
TARGET_DIR="july7Exp/hybrid_params_tests/32bit/UniformDistRegGen"
mkdir -p "$TARGET_DIR"

CSV_FILES=()

# Shared output base name
OUT_BASENAME="32bit_25_26_ST_Inserts_DiV1_Uniform_RegGen_100_4rounds"

# --- Run 1: Single_Thread_SHIFT RIGHT ONLY ---
rm -f data_cache/*.*
#rm CMakeCache.txt ; cmake ..  -DIFDEFS:STRING='-DDENSEKEYGEN -DTOTALRUNS=1 -DMAIN_32 -DDIV=1 -DPROCESS_INSERTS_SINGLETHREAD -DMAX_NODE=16 -DDEFINE_TILE_SIZE=16  -DMIN_KEYS_INSERT=3 -DNODE_THRESHOLD=75 -DALL_SHIFT_RIGHT' && make clean && make -j
rm CMakeCache.txt ; cmake ..  -DIFDEFS:STRING='-DREGKEYGEN -DTOTALRUNS=1 -DMAIN_32 -DDIV=1 -DPROCESS_INSERTS_SINGLETHREAD -DMAX_NODE=16 -DDEFINE_TILE_SIZE=16  -DMIN_KEYS_INSERT=3 -DNODE_THRESHOLD=75 -DALL_SHIFT_RIGHT' && make clean && make -j

OUT_TXT="${OUT_BASENAME}+{AllShftRght}.txt"
OUT_CSV="${OUT_BASENAME}+{AllShftRght}.csv"

./index_prototype &> "$OUT_TXT"
cp updates.csv "$OUT_CSV"
mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")

echo "All Shift Right experiment completed."

# --- Run 2: Single_Thread_BULK ONLY ---
rm CMakeCache.txt ; cmake ..  -DIFDEFS:STRING='-DREGKEYGEN -DTOTALRUNS=1 -DMAIN_32 -DDIV=1 -DPROCESS_INSERTS_SINGLETHREAD  -DMAX_NODE=16 -DDEFINE_TILE_SIZE=16  -DMIN_KEYS_INSERT=3 -DNODE_THRESHOLD=75 ' && make clean && make -j
OUT_TXT="${OUT_BASENAME}+{AllBULK}.txt"
OUT_CSV="${OUT_BASENAME}+{AllBULK}.csv"

./index_prototype &> "$OUT_TXT"
cp updates.csv "$OUT_CSV"
mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")

echo "All Bulk experiment completed."

# ------------------------------- Run 3: Loop Across Hybrid Configurations

NODE_THRESHOLDS=(50 75 90 100)
MIN_KEYS_INSERTS=(1 2 3 4 5 6 7)

for threshold in "${NODE_THRESHOLDS[@]}"; do
    for min_keys in "${MIN_KEYS_INSERTS[@]}"; do
        rm CMakeCache.txt
        cmake .. -DIFDEFS:STRING="--DREGKEYGEN -DTOTALRUNS=1 -DMAIN_32 -DDIV=1 -DPROCESS_INSERTS_SINGLETHREAD -DMAX_NODE=16 -DDEFINE_TILE_SIZE=16 -DMIN_KEYS_INSERT=${min_keys} -DNODE_THRESHOLD=${threshold} -DHYBRID " && make clean && make -j

        OUT_TXT="${OUT_BASENAME}+{Hybrid_NT${threshold}_MK${min_keys}}.txt"
        OUT_CSV="${OUT_BASENAME}+{Hybrid_NT${threshold}_MK${min_keys}}.csv"

        ./index_prototype &> "$OUT_TXT"
        cp updates.csv "$OUT_CSV"
        mv "${OUT_BASENAME}+"* "$TARGET_DIR"/
        CSV_FILES+=("${TARGET_DIR}/${OUT_CSV}")

        echo "Experiment completed with NODE_THRESHOLD=$threshold and MIN_KEYS_INSERT=$min_keys"
    done
done


echo "Hybrid experiments completed."


##-------------------------------------------------------------------------

# --
# -------------------------------------------------------------------------

cd "$TARGET_DIR"
# Final step: Call Python plotting script
#python3 auto_4_Rounds_updates_correctlabels.py 10 100 ${#CSV_FILES[@]} "${CSV_FILES[@]}"
HYBRID_FILES=$(ls *{Hybrid_NT*}.csv | sort -V)
ALL_FILES="$HYBRID_FILES 32bit_25_26_ST_Inserts_test_DiV1_x10y100_NOShiftRange_4rounds+{AllShftRght}.csv 32bit_25_26_ST_Inserts_test_DiV1_x10y100_NOShiftRange_4rounds+{AllBULK}.csv"
FILE_COUNT=$(echo $ALL_FILES | wc -w)

python3 auto_4_Rounds_updates_correctlabels_Ncolours_noDel.py 10 100 $FILE_COUNT $ALL_FILES

#python3 auto_4_Rounds_updates_correctlabels_Ncolours_noDel.py 10 100 $(ls *{Hybrid_NT*}.csv | sort -V | wc -l) $(ls *{Hybrid_NT*}.csv | sort -V)
#python3 update_plots_4rounds_variable.py ${#CSV_FILES[@]} "${CSV_FILES[@]}"

echo "All experiments completed and results stored in $TARGET_DIR"