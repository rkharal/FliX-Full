#!/usr/bin/env bash
# Warm-up comparison with single-threaded insertions and queries.
# This variant uses build size 2^22, probe size 2^26, and node size 3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/WarmupTests/SingleThreadedInsertsQueries_NS3_Probe26_Growth400_FullQueryWarmup_Insert5000Warmup"
mkdir -p "${OUTPUT_DIR}"
cd "${BUILD_DIR}"

XVAL=25
YVAL=90
GROWTH_RATE_VALUE=400
NODE_SIZE=3
DIV_VALUE=4
BUILD_SIZE_LOG=22
PROBE_SIZE_LOG=26
ROUNDS=4
TOTAL_RUNS=5
MAX_NODE=$((2 ** NODE_SIZE))
TILE_SIZE=$((2 ** NODE_SIZE))

CUDA_HOME="/usr/local/cuda-12.8"
CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"
C_COMPILER="/usr/local/bin/gcc"
CXX_COMPILER="/usr/local/bin/g++"
CUDA_ARCHITECTURE=86
OPTIX_HOME="/home/rkharal/OPTIX/cgRT_Optimization/rtx-index-updates/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64"

export PATH="${CUDA_HOME}/bin:/usr/local/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDACXX="${CUDA_COMPILER}"
export CUDA_PATH="${CUDA_HOME}"
export OptiX_INSTALL_DIR="${OPTIX_HOME}"

PREFIX="X${XVAL}Y${YVAL}_Growth${GROWTH_RATE_VALUE}_NS${NODE_SIZE}_DIV${DIV_VALUE}_B${BUILD_SIZE_LOG}_P${PROBE_SIZE_LOG}_R${ROUNDS}_FliX_STInsert_STQuery_FullQueryWarmup_Insert5000"

# Tiled insertion flags are intentionally absent so that the single-threaded
# shift-right insertion path is selected. Deletions remain tiled and bulk.
COMMON_FLAGS="-DNODESIZE=${NODE_SIZE};-DCACHE_LINE_SIZE=0;-DDENSEKEYGEN;-DXVAL=${XVAL};-DYVAL=${YVAL};-DGROWTH_RATE=${GROWTH_RATE_VALUE};-DTOTALRUNS=${TOTAL_RUNS};-DMAIN_32;-DDIV=${DIV_VALUE};-DPROCESS_INSERTS_SINGLETHREAD;-DALL_SHIFT_RIGHT;-DTILE_DELETES;-DDELETES_TILE_BULK;-DMAX_NODE=${MAX_NODE};-DDEFINE_TILE_SIZE=${TILE_SIZE};-DINITIAL_BUILD_SIZE=${BUILD_SIZE_LOG};-DINITIAL_PROBE_SIZE=${PROBE_SIZE_LOG};-DROUNDS_NUMBER=${ROUNDS};-DLSM_CHUNK_SIZE_LOG=16;-DPD=100;-DPU=0;-DSINGLE_THREADED_LOOKUPS"

build_and_run() {
    local label="$1"
    local warmup_flags="$2"
    local flags="${COMMON_FLAGS}"
    local output_base="${PREFIX}_${label}"
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
        echo "insertion_path=single-threaded shift-right"
        echo "query_path=single-threaded"
        echo "deletion_path=tiled bulk"
        echo "X=${XVAL} Y=${YVAL} growth=${GROWTH_RATE_VALUE}"
        echo "node_size=${NODE_SIZE} div=${DIV_VALUE}"
        echo "build_size_log=${BUILD_SIZE_LOG} probe_size_log=${PROBE_SIZE_LOG}"
        echo "rounds=${ROUNDS} total_runs=${TOTAL_RUNS}"
        echo "IFDEFS=${flags}"
    } > "${params_log}"

    echo "Building ${label} B22/P26 configuration..."
    cmake .. \
        -DCMAKE_C_COMPILER="${C_COMPILER}" \
        -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
        -DCMAKE_CUDA_COMPILER="${CUDA_COMPILER}" \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURE}" \
        -DCUDAToolkit_ROOT="${CUDA_HOME}" \
        -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
        -DOptiX_INSTALL_DIR="${OPTIX_HOME}" \
        -DIFDEFS:STRING="${flags}" 2>&1 | tee "${build_log}"
    make clean 2>&1 | tee -a "${build_log}"
    make -j"$(nproc)" 2>&1 | tee -a "${build_log}"

    echo "Running ${label} B22/P26 configuration..."
    ./index_prototype > "${output_txt}" 2>&1

    if [[ ! -s updates.csv ]]; then
        echo "ERROR: updates.csv was not generated for ${label}." >&2
        exit 1
    fi
    cp updates.csv "${output_csv}"
    echo "Completed: ${output_base}"
}

build_and_run "NoWarmups" "-DDISABLE_QUERY_HIT_WARMUP"
build_and_run "Warmups" "-DENABLE_INSERT_WARMUP;-DFULL_QUERY_HIT_WARMUP;-DINSERT_WARMUP_KEY_COUNT=5000"

PLOT_SCRIPT="${OUTPUT_DIR}/plot_NS3_Probe26_Growth400_full_query_insert5000_warmup_comparison.py"
if [[ -x /home/rkharal/anaconda3/bin/python3.9 ]]; then
    /home/rkharal/anaconda3/bin/python3.9 "${PLOT_SCRIPT}"
else
    python3 "${PLOT_SCRIPT}"
fi

echo "All B22/P26 single-thread insertion/query warm-up runs completed."
echo "Results: ${OUTPUT_DIR}"
