#!/usr/bin/env bash
# Compare FliX update/query timings with and without selected warm-up calls.
#
# Fixed experiment:
#   X=25, Y=90, growth=200%, node size=5, DIV=4
#   build size=2^25, probe size=2^26, four insertion rounds followed by
#   four deletion rounds, and single-threaded point queries.
#
# Output:
#   ./WarmupTests/*_FliX_Nowarmpus.{txt,csv}
#   ./WarmupTests/*_FliX_warmpus.{txt,csv}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/WarmupTests"
mkdir -p "${OUTPUT_DIR}"
cd "${BUILD_DIR}"

XVAL=25
YVAL=90
GROWTH_RATE_VALUE=200
NODE_SIZE=5
DIV_VALUE=4
BUILD_SIZE_LOG=25
PROBE_SIZE_LOG=26
ROUNDS=4
TOTAL_RUNS=5
MAX_NODE=$((2 ** NODE_SIZE))
TILE_SIZE=$((2 ** NODE_SIZE))

PREFIX="X${XVAL}Y${YVAL}_Growth${GROWTH_RATE_VALUE}_NS${NODE_SIZE}_DIV${DIV_VALUE}_B${BUILD_SIZE_LOG}_P${PROBE_SIZE_LOG}_R${ROUNDS}"

COMMON_FLAGS="-DNODESIZE=${NODE_SIZE};-DCACHE_LINE_SIZE=0;-DDENSEKEYGEN;-DXVAL=${XVAL};-DYVAL=${YVAL};-DGROWTH_RATE=${GROWTH_RATE_VALUE};-DTOTALRUNS=${TOTAL_RUNS};-DMAIN_32;-DDIV=${DIV_VALUE};-DTILE_INSERTS;-DTILE_INSERTS_C;-DINSERTS_TILE_BULK_ONLY;-DTILE_DELETES;-DDELETES_TILE_BULK;-DMAX_NODE=${MAX_NODE};-DDEFINE_TILE_SIZE=${TILE_SIZE};-DINITIAL_BUILD_SIZE=${BUILD_SIZE_LOG};-DINITIAL_PROBE_SIZE=${PROBE_SIZE_LOG};-DROUNDS_NUMBER=${ROUNDS};-DLSM_CHUNK_SIZE_LOG=16;-DPD=100;-DPU=0;-DSINGLE_THREADED_LOOKUPS"

build_and_run() {
    local label="$1"
    local warmup_flags="$2"
    local flags="${COMMON_FLAGS}"
    local output_base="${PREFIX}_FliX_${label}"
    local output_txt="${OUTPUT_DIR}/${output_base}.txt"
    local output_csv="${OUTPUT_DIR}/${output_base}.csv"
    local build_log="${OUTPUT_DIR}/${output_base}_build.log"
    local params_log="${OUTPUT_DIR}/${output_base}_parameters.txt"

    if [[ -n "${warmup_flags}" ]]; then
        flags="${flags};${warmup_flags}"
    fi

    rm -f CMakeCache.txt build.log updates.csv
    rm -f data_cache/*.* 2>/dev/null || true

    {
        echo "label=${label}"
        echo "X=${XVAL} Y=${YVAL} growth=${GROWTH_RATE_VALUE}"
        echo "node_size=${NODE_SIZE} div=${DIV_VALUE}"
        echo "build_size_log=${BUILD_SIZE_LOG} probe_size_log=${PROBE_SIZE_LOG}"
        echo "rounds=${ROUNDS} total_runs=${TOTAL_RUNS}"
        echo "IFDEFS=${flags}"
    } > "${params_log}"

    echo "Building ${label} configuration..."
    cmake .. -DIFDEFS:STRING="${flags}" 2>&1 | tee "${build_log}"
    make clean 2>&1 | tee -a "${build_log}"
    make -j"$(nproc)" 2>&1 | tee -a "${build_log}"

    echo "Running ${label} configuration..."
    ./index_prototype > "${output_txt}" 2>&1

    if [[ ! -s updates.csv ]]; then
        echo "ERROR: updates.csv was not generated for ${label}." >&2
        exit 1
    fi
    cp updates.csv "${output_csv}"
    echo "Completed: ${output_base}"
}

# Round 0 still performs the full measured query-only workload. Only its
# preliminary 20-key point-query warm-up is disabled. Insert, miss-query, and
# deletion warm-ups are also absent.
build_and_run "Nowarmpus" "-DDISABLE_QUERY_HIT_WARMUP"

# Restore the existing 20-key hit-query warm-up and enable the existing
# 10-key dummy insertion immediately before the first measured insertion.
# Miss queries and deletions still have no explicit warm-up.
build_and_run "warmpus" "-DENABLE_INSERT_WARMUP"

echo "All warm-up comparison runs completed."
echo "Results: ${OUTPUT_DIR}"
