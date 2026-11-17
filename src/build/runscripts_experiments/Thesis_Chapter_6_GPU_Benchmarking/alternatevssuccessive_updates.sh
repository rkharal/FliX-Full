#!/usr/bin/env bash
# Successive vs. alternating insert/delete schedule comparison across
# GPU B-tree, LSM-tree, HT-Slab, and FliX.
#
# Fixed experiment:
#   X=50, Y=90, growth=200%, node size=3 (8 keys/node), DIV=4
#   build size=2^25, probe size=2^26, 4 rounds per phase.
#   Tiled insertions, tiled deletions, tiled (multi-threaded) point queries.
#   No insert warm-up cycle, no query-hit warm-up cycle.
#
# Swept:
#   SCHEDULE   Successive  = 4 insertion rounds followed by 4 deletion rounds
#              Alternating = insert, delete, insert, delete... (4 of each)
#   STRUCTURE  FliX, LSM-tree (LSMu), HT-Slab, GPU B-tree
#
#   FliX data is generated for completeness but is intentionally excluded from
#   the default plot -- the comparison of interest is the three baselines.
#
#   The alternating schedule requires -DALTERNATING_INSERT_DELETE, which wires
#   up compile_flags::ALTERNATE_INSERT_DELETE_ROUNDS added to
#   ../benchmarks_updates.cuh (test_configuration_options' alternating_insert_delete
#   field was previously hardcoded to false).
#
# Output:
#   ./alternatevssuccessive_updates_distinct_batches/*_<Schedule>+{<Structure>}.{txt,csv}
#   ./alternatevssuccessive_updates_distinct_batches/*_<Schedule>+{<Structure>}_build.log
#   ./alternatevssuccessive_updates_distinct_batches/*_<Schedule>+{<Structure>}_parameters.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/alternatevssuccessive_updates_distinct_batches"
mkdir -p "${OUTPUT_DIR}"
cd "${BUILD_DIR}"

XVAL=50
YVAL=90
GROWTH_RATE_VALUE=200
NODE_SIZE=3
DIV_VALUE=4
BUILD_SIZE_LOG=25
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

PREFIX="X${XVAL}Y${YVAL}_Growth${GROWTH_RATE_VALUE}_NS${NODE_SIZE}_DIV${DIV_VALUE}_B${BUILD_SIZE_LOG}_P${PROBE_SIZE_LOG}_R${ROUNDS}"

# Tiled insertion/deletion/query paths; no warm-ups of any kind.
# -DBASELINES plus one of -DLSM_TREE / -DHASHTABLE_SLAB / -DGPU_BTREE selects a
# baseline; FliX is the default build with none of those flags set.
COMMON_FLAGS="-DNODESIZE=${NODE_SIZE};-DCACHE_LINE_SIZE=0;-DDENSEKEYGEN;-DXVAL=${XVAL};-DYVAL=${YVAL};-DGROWTH_RATE=${GROWTH_RATE_VALUE};-DTOTALRUNS=${TOTAL_RUNS};-DMAIN_32;-DDIV=${DIV_VALUE};-DTILE_INSERTS;-DTILE_INSERTS_C;-DINSERTS_TILE_BULK_ONLY;-DTILE_DELETES;-DDELETES_TILE_BULK;-DMAX_NODE=${MAX_NODE};-DDEFINE_TILE_SIZE=${TILE_SIZE};-DINITIAL_BUILD_SIZE=${BUILD_SIZE_LOG};-DINITIAL_PROBE_SIZE=${PROBE_SIZE_LOG};-DROUNDS_NUMBER=${ROUNDS};-DLSM_CHUNK_SIZE_LOG=16;-DPD=100;-DPU=0;-DDISABLE_QUERY_HIT_WARMUP"

STRUCTURE_ORDER=(FliX LSMu HashSlab GPUBTree)
declare -A STRUCTURE_FLAGS=(
    [FliX]=""
    [LSMu]="-DBASELINES;-DLSM_TREE"
    [HashSlab]="-DBASELINES;-DHASHTABLE_SLAB"
    [GPUBTree]="-DBASELINES;-DGPU_BTREE"
)

SCHEDULE_ORDER=(Successive Alternating)
declare -A SCHEDULE_FLAGS=(
    [Successive]=""
    [Alternating]="-DALTERNATING_INSERT_DELETE"
)

build_and_run() {
    local schedule="$1"
    local structure="$2"
    local structure_flags="${STRUCTURE_FLAGS[$structure]}"
    local schedule_flags="${SCHEDULE_FLAGS[$schedule]}"

    local flags="${COMMON_FLAGS}"
    [[ -n "${structure_flags}" ]] && flags="${flags};${structure_flags}"
    [[ -n "${schedule_flags}" ]] && flags="${flags};${schedule_flags}"

    local output_base="${PREFIX}_${schedule}+{${structure}}"
    local output_txt="${OUTPUT_DIR}/${output_base}.txt"
    local output_csv="${OUTPUT_DIR}/${output_base}.csv"
    local build_log="${OUTPUT_DIR}/${output_base}_build.log"
    local params_log="${OUTPUT_DIR}/${output_base}_parameters.txt"

    rm -f CMakeCache.txt build.log updates.csv
    rm -f data_cache/*.* 2>/dev/null || true

    {
        echo "schedule=${schedule}"
        echo "structure=${structure}"
        echo "insert_path=tiled bulk"
        echo "delete_path=tiled bulk"
        echo "lookup_path=tile-bulk (multi-threaded)"
        echo "X=${XVAL} Y=${YVAL} growth=${GROWTH_RATE_VALUE}"
        echo "node_size=${NODE_SIZE} div=${DIV_VALUE}"
        echo "build_size_log=${BUILD_SIZE_LOG} probe_size_log=${PROBE_SIZE_LOG}"
        echo "rounds=${ROUNDS} total_runs=${TOTAL_RUNS}"
        echo "insert_warmup=disabled query_hit_warmup=disabled"
        echo "IFDEFS=${flags}"
    } > "${params_log}"

    echo "Building ${schedule}/${structure} configuration..."
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

    echo "Running ${schedule}/${structure} configuration..."
    ./index_prototype > "${output_txt}" 2>&1

    if [[ ! -s updates.csv ]]; then
        echo "ERROR: updates.csv was not generated for ${schedule}/${structure}." >&2
        exit 1
    fi
    cp updates.csv "${output_csv}"
    echo "Completed: ${output_base}"
}

for schedule in "${SCHEDULE_ORDER[@]}"; do
    for structure in "${STRUCTURE_ORDER[@]}"; do
        build_and_run "${schedule}" "${structure}"
    done
done

echo "All successive-vs-alternating update schedule runs completed."
echo "Results: ${OUTPUT_DIR}"

PLOT_SCRIPT="${OUTPUT_DIR}/plot_alternatevssuccessive_updates.py"
if [[ -x /home/rkharal/anaconda3/bin/python3.9 ]]; then
    /home/rkharal/anaconda3/bin/python3.9 "${PLOT_SCRIPT}"
else
    python3 "${PLOT_SCRIPT}"
fi
