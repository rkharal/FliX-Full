#!/usr/bin/env bash
# Build/probe size sweep to detect whether round-1 hit-query time is affected
# by (a) the presence of step 0 and (b) the query-hit/insert warm-up cycles,
# for both single-threaded-query/tiled-insert and tiled-query/
# single-threaded-insert configurations.
#
# Sweep: BUILD_SIZE_LOG = 10, 12, 14, ..., 22 (step 2), PROBE_SIZE_LOG =
# BUILD_SIZE_LOG + 1 (paired), i.e. (10,11), (12,13), (14,15), (16,17),
# (18,19), (20,21), (22,23) -- 7 points.
#
# Fixed: X90/Y90, growth=200%, node size=4, DIV=1 (matching the most recent
# script in this dir), 4 insertion rounds followed by 4 deletion rounds,
# tiled deletes throughout.
#
# Structural configs:
#   a. TiledInsert_STQuery : tiled inserts, tiled deletes, single-threaded lookups
#   b. STInsert_TiledQuery : single-threaded shift-right inserts, tiled deletes, tiled lookups
#
# Warm-up configs (per structural config):
#   NoWarmup_NoStep0  : -DSKIP_STEP0_PROBES -- step 0 is skipped entirely (no
#     update, no insertion warm-up, no query preparation/sort, no hit lookup,
#     no miss lookup -- see benchmarks_updates.cuh's `if (step == 0) { #ifdef
#     SKIP_STEP0_PROBES ... continue; #endif }`). -DDISABLE_QUERY_HIT_WARMUP is
#     also set for clarity/redundancy, though SKIP_STEP0_PROBES alone already
#     makes that check unreachable. No insert warm-up flag (already off by
#     default). NOTE: without SKIP_STEP0_PROBES, step 0 ALWAYS performs a real,
#     timed hit/miss-query probe regardless of DISABLE_QUERY_HIT_WARMUP -- that
#     flag only removes one extra *dummy* lookup call issued immediately
#     before the real one. This was gotten wrong in the immediately preceding
#     script (WarmupTests_FliX_TiledInsert_STQuery_X90Y90_25_27.sh) -- its
#     "NoWarmup_NoStep0" label is inaccurate; step 0 still ran a real probe
#     there. This script is the first to use the actually-correct flag.
#   WithStep0_Warmups : -DENABLE_INSERT_WARMUP, query-hit warm-up left at
#     default-on (step 0 runs its dummy lookup then the real one; insert
#     warm-up runs a dummy insert at step 1, independent of step 0).
#
# 7 build/probe pairs x 2 structures x 2 warmup configs = 28 build+run cycles.
#
# No plotting. A companion Python script (detect_round1_hitquery_warmup_effect.py)
# reads every pair of CSVs and prints/writes a table comparing round-1
# hit-query time (NoWarmup vs WithWarmup) for every (structure, build/probe)
# combination, to directly detect whether/how much warm-up changes round-1
# query time.
#
# Output:
#   ./BuildProbeSweep_WarmupDetection_X90Y90_B10_to_B22/*.{txt,csv,build.log,parameters.txt}
#   ./BuildProbeSweep_WarmupDetection_X90Y90_B10_to_B22/round1_hitquery_warmup_detection_summary.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/BuildProbeSweep_WarmupDetection_X90Y90_B10_to_B22"
mkdir -p "${OUTPUT_DIR}"
cd "${BUILD_DIR}"

XVAL=90
YVAL=90
GROWTH_RATE_VALUE=200
NODE_SIZE=4
DIV_VALUE=1
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

BASE_FLAGS="-DNODESIZE=${NODE_SIZE};-DCACHE_LINE_SIZE=0;-DDENSEKEYGEN;-DXVAL=${XVAL};-DYVAL=${YVAL};-DGROWTH_RATE=${GROWTH_RATE_VALUE};-DTOTALRUNS=${TOTAL_RUNS};-DMAIN_32;-DDIV=${DIV_VALUE};-DTILE_DELETES;-DDELETES_TILE_BULK;-DMAX_NODE=${MAX_NODE};-DDEFINE_TILE_SIZE=${TILE_SIZE};-DROUNDS_NUMBER=${ROUNDS};-DLSM_CHUNK_SIZE_LOG=16;-DPD=100;-DPU=0"

STRUCTURE_ORDER=(TiledInsert_STQuery STInsert_TiledQuery)
declare -A STRUCTURE_FLAGS=(
    [TiledInsert_STQuery]="-DTILE_INSERTS;-DTILE_INSERTS_C;-DINSERTS_TILE_BULK_ONLY;-DSINGLE_THREADED_LOOKUPS"
    [STInsert_TiledQuery]="-DPROCESS_INSERTS_SINGLETHREAD;-DALL_SHIFT_RIGHT"
)

WARMUP_ORDER=(NoWarmup_NoStep0 WithStep0_Warmups)
declare -A WARMUP_FLAGS=(
    [NoWarmup_NoStep0]="-DSKIP_STEP0_PROBES;-DDISABLE_QUERY_HIT_WARMUP"
    [WithStep0_Warmups]="-DENABLE_INSERT_WARMUP"
)

BUILD_SIZES=(10 12 14 16 18 20 22)

build_and_run() {
    local build_log2="$1"
    local probe_log2="$2"
    local structure="$3"
    local warmup="$4"

    local flags="${BASE_FLAGS};-DINITIAL_BUILD_SIZE=${build_log2};-DINITIAL_PROBE_SIZE=${probe_log2};${STRUCTURE_FLAGS[$structure]};${WARMUP_FLAGS[$warmup]}"

    local output_base="X${XVAL}Y${YVAL}_Growth${GROWTH_RATE_VALUE}_NS${NODE_SIZE}_DIV${DIV_VALUE}_B${build_log2}_P${probe_log2}_R${ROUNDS}_FliX_${structure}_${warmup}"
    local output_txt="${OUTPUT_DIR}/${output_base}.txt"
    local output_csv="${OUTPUT_DIR}/${output_base}.csv"
    local build_log="${OUTPUT_DIR}/${output_base}_build.log"
    local params_log="${OUTPUT_DIR}/${output_base}_parameters.txt"

    rm -f CMakeCache.txt build.log updates.csv
    rm -f data_cache/*.* 2>/dev/null || true

    {
        echo "structure=${structure}"
        echo "warmup=${warmup}"
        echo "build_size_log=${build_log2} probe_size_log=${probe_log2}"
        echo "X=${XVAL} Y=${YVAL} growth=${GROWTH_RATE_VALUE}"
        echo "node_size=${NODE_SIZE} div=${DIV_VALUE}"
        echo "rounds=${ROUNDS} total_runs=${TOTAL_RUNS}"
        echo "IFDEFS=${flags}"
    } > "${params_log}"

    echo "Building B${build_log2}/P${probe_log2} ${structure} ${warmup}..."
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

    echo "Running B${build_log2}/P${probe_log2} ${structure} ${warmup}..."
    ./index_prototype > "${output_txt}" 2>&1

    if [[ ! -s updates.csv ]]; then
        echo "ERROR: updates.csv was not generated for ${output_base}." >&2
        exit 1
    fi
    cp updates.csv "${output_csv}"
    echo "Completed: ${output_base}"
}

for build_log2 in "${BUILD_SIZES[@]}"; do
    probe_log2=$((build_log2 + 1))
    for structure in "${STRUCTURE_ORDER[@]}"; do
        for warmup in "${WARMUP_ORDER[@]}"; do
            build_and_run "${build_log2}" "${probe_log2}" "${structure}" "${warmup}"
        done
    done
done

echo "All 28 build/probe sweep runs completed."
echo "Results: ${OUTPUT_DIR}"

PYTHON_BIN="python3"
if [[ -x /home/rkharal/anaconda3/bin/python3.9 ]]; then
    PYTHON_BIN="/home/rkharal/anaconda3/bin/python3.9"
fi
"${PYTHON_BIN}" "${OUTPUT_DIR}/detect_round1_hitquery_warmup_effect.py"
"${PYTHON_BIN}" "${OUTPUT_DIR}/plot_round1_pct_change.py"
