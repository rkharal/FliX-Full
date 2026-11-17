#!/usr/bin/env bash
# Threads-per-block (TPB) assignment sweep for FliX tile-bulk lookups.
#
# Fixed experiment:
#   X=90, Y=90, growth=300%, node size=5, build size=2^25, probe size=2^26,
#   8 insertion rounds followed by 8 deletion rounds.
#   Lookups use the default multi-threaded tile-bulk kernel (no
#   -DSINGLE_THREADED_LOOKUPS). Insert warm-up and full query hit warm-up
#   are both enabled.
#
# Swept:
#   DIV (threads-per-block divisor: launch block size = MAXBLOCKSIZE/DIV)
#   over 1, 2, 4, 8, 16.
#
# Output:
#   ./TPB_assignment_tests/*_DIV<n>.{txt,csv}
#   ./TPB_assignment_tests/*_DIV<n>_build.log
#   ./TPB_assignment_tests/*_DIV<n>_parameters.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/TPB_assignment_tests"
mkdir -p "${OUTPUT_DIR}"
cd "${BUILD_DIR}"

XVAL=90
YVAL=90
GROWTH_RATE_VALUE=300
NODE_SIZE=5
BUILD_SIZE_LOG=25
PROBE_SIZE_LOG=26
ROUNDS=8
TOTAL_RUNS=5
MAX_NODE=$((2 ** NODE_SIZE))
TILE_SIZE=$((2 ** NODE_SIZE))

declare -a DIV_VALUES=(1 2 4 8 16)

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

PREFIX="X${XVAL}Y${YVAL}_Growth${GROWTH_RATE_VALUE}_NS${NODE_SIZE}_B${BUILD_SIZE_LOG}_P${PROBE_SIZE_LOG}_R${ROUNDS}_FliX_TileBulk_Warmups"

# Tile-bulk lookups/queries use the default lookup kernel, so
# -DSINGLE_THREADED_LOOKUPS is intentionally absent. Insert warm-up uses the
# code's default warm-up key count (10); query hit warm-up uses the full
# probe count via -DFULL_QUERY_HIT_WARMUP.
COMMON_FLAGS="-DNODESIZE=${NODE_SIZE};-DCACHE_LINE_SIZE=0;-DDENSEKEYGEN;-DXVAL=${XVAL};-DYVAL=${YVAL};-DGROWTH_RATE=${GROWTH_RATE_VALUE};-DTOTALRUNS=${TOTAL_RUNS};-DMAIN_32;-DTILE_INSERTS;-DTILE_INSERTS_C;-DINSERTS_TILE_BULK_ONLY;-DTILE_DELETES;-DDELETES_TILE_BULK;-DMAX_NODE=${MAX_NODE};-DDEFINE_TILE_SIZE=${TILE_SIZE};-DINITIAL_BUILD_SIZE=${BUILD_SIZE_LOG};-DINITIAL_PROBE_SIZE=${PROBE_SIZE_LOG};-DROUNDS_NUMBER=${ROUNDS};-DLSM_CHUNK_SIZE_LOG=16;-DPD=100;-DPU=0;-DENABLE_INSERT_WARMUP;-DFULL_QUERY_HIT_WARMUP"

build_and_run() {
    local div_value="$1"
    local flags="${COMMON_FLAGS};-DDIV=${div_value}"
    local output_base="${PREFIX}_DIV${div_value}"
    local output_txt="${OUTPUT_DIR}/${output_base}.txt"
    local output_csv="${OUTPUT_DIR}/${output_base}.csv"
    local build_log="${OUTPUT_DIR}/${output_base}_build.log"
    local params_log="${OUTPUT_DIR}/${output_base}_parameters.txt"

    rm -f CMakeCache.txt build.log updates.csv
    rm -f data_cache/*.* 2>/dev/null || true

    {
        echo "div=${div_value}"
        echo "lookup_path=tile-bulk (multi-threaded)"
        echo "insert_path=tiled bulk"
        echo "delete_path=tiled bulk"
        echo "X=${XVAL} Y=${YVAL} growth=${GROWTH_RATE_VALUE}"
        echo "node_size=${NODE_SIZE} div=${div_value}"
        echo "build_size_log=${BUILD_SIZE_LOG} probe_size_log=${PROBE_SIZE_LOG}"
        echo "rounds=${ROUNDS} total_runs=${TOTAL_RUNS}"
        echo "insert_warmup=enabled (default key count)"
        echo "query_hit_warmup=enabled (full)"
        echo "IFDEFS=${flags}"
    } > "${params_log}"

    echo "Building DIV=${div_value} configuration..."
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

    echo "Running DIV=${div_value} configuration..."
    ./index_prototype > "${output_txt}" 2>&1

    if [[ ! -s updates.csv ]]; then
        echo "ERROR: updates.csv was not generated for DIV=${div_value}." >&2
        exit 1
    fi
    cp updates.csv "${output_csv}"
    echo "Completed: ${output_base}"
}

for div_value in "${DIV_VALUES[@]}"; do
    build_and_run "${div_value}"
done

echo "All TPB assignment (DIV sweep) runs completed."
echo "Results: ${OUTPUT_DIR}"
