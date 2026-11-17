#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../RQ_TESTS"

RQ_SIZES=(8 10 12 14 16 18 20)

for rq in "${RQ_SIZES[@]}"; do
    base="RQ_X90Y90_RQ${rq}_DIV8_NS5_CL0_x200_32bit_B25_P26_8ROUNDS"
    RQONLY_LOG_SCALE=1 python3 plot_range_query_rqonly_3files.py \
        "${base}_FliX.csv" \
        "${base}_FliX-Parallel.csv" \
        "${base}_FliX-naive.csv" \
        "${base}_GPU_BTREE.csv" \
        "${base}_LSM_TREE.csv"
done

echo "All log-scale RQ-only plots are in ${SCRIPT_DIR}/RQ_PLOTS_RQONLY/LOG"
