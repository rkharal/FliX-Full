#!/usr/bin/env bash
# X90/Y90 warm-up comparison with tiled inserts, tiled deletes, and
# single-threaded (point-query) lookups. Build size 2^25, probe size 2^27.
#
# Experiment 1 (NoWarmup_NoStep0): strict no-step-0 execution. Step 0 exits
#   before any warm-up, update, sort, hit-query, miss-query, or other benchmark
#   kernel. The first benchmark kernel in the measured rounds is the step-1
#   insertion kernel, followed later by the step-1 query kernels.
# Experiment 2 (WithStep0_Warmups): step 0 performs an untimed full-probe-batch
#   hit-query warm-up followed by the normal timed step-0 hit and miss queries.
#   The step-1 insertion warm-up is also enabled.
#
# Output:
#   ./WarmupTests/TiledInsert_STQuery_X90Y90_B25P27_Growth200_NS4_DIV1/*_NoWarmup_NoStep0.{txt,csv}
#   ./WarmupTests/TiledInsert_STQuery_X90Y90_B25P27_Growth200_NS4_DIV1/*_WithStep0_Warmups.{txt,csv}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/WarmupTests/TiledInsert_STQuery_X90Y90_B25P27_Growth200_NS4_DIV1"
mkdir -p "${OUTPUT_DIR}"
cd "${BUILD_DIR}"

XVAL=90
YVAL=90
GROWTH_RATE_VALUE=200
NODE_SIZE=4
DIV_VALUE=1
BUILD_SIZE_LOG=25
PROBE_SIZE_LOG=27
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

PREFIX="X${XVAL}Y${YVAL}_Growth${GROWTH_RATE_VALUE}_NS${NODE_SIZE}_DIV${DIV_VALUE}_B${BUILD_SIZE_LOG}_P${PROBE_SIZE_LOG}_R${ROUNDS}_FliX_TiledInsert_STQuery_25_27"
FULL_PROBE_COUNT=$((2 ** PROBE_SIZE_LOG))

# Tiled inserts, tiled deletes, single-threaded (point-query) lookups.
COMMON_FLAGS="-DNODESIZE=${NODE_SIZE};-DCACHE_LINE_SIZE=0;-DDENSEKEYGEN;-DXVAL=${XVAL};-DYVAL=${YVAL};-DGROWTH_RATE=${GROWTH_RATE_VALUE};-DTOTALRUNS=${TOTAL_RUNS};-DMAIN_32;-DDIV=${DIV_VALUE};-DTILE_INSERTS;-DTILE_INSERTS_C;-DINSERTS_TILE_BULK_ONLY;-DTILE_DELETES;-DDELETES_TILE_BULK;-DSINGLE_THREADED_LOOKUPS;-DMAX_NODE=${MAX_NODE};-DDEFINE_TILE_SIZE=${TILE_SIZE};-DINITIAL_BUILD_SIZE=${BUILD_SIZE_LOG};-DINITIAL_PROBE_SIZE=${PROBE_SIZE_LOG};-DROUNDS_NUMBER=${ROUNDS};-DLSM_CHUNK_SIZE_LOG=16;-DPD=100;-DPU=0"

validate_warmup_behavior() {
    local label="$1"
    local output_txt="$2"
    local output_csv="$3"

    if [[ "${label}" == "NoWarmup_NoStep0" ]]; then
        if ! grep -q "step 0: STRICT SKIP" "${output_txt}"; then
            echo "ERROR: strict step-0 skip marker is missing from ${output_txt}." >&2
            return 1
        fi
        if grep -q "Point-query hit warm-up size:" "${output_txt}"; then
            echo "ERROR: query warm-up unexpectedly ran in ${label}." >&2
            return 1
        fi
        if grep -q "Performing Dummy Insert Round" "${output_txt}"; then
            echo "ERROR: insertion warm-up unexpectedly ran in ${label}." >&2
            return 1
        fi
        if awk -F',' '
            NR > 1 {
                step = $14;
                gsub(/[[:space:]]/, "", step);
                if (step == "0") found = 1;
            }
            END { exit(found ? 0 : 1); }
        ' "${output_csv}"; then
            echo "ERROR: ${label} recorded step-0 measurements despite strict skip." >&2
            return 1
        fi
    elif [[ "${label}" == "WithStep0_Warmups" ]]; then
        if ! grep -q "Point-query hit warm-up size: ${FULL_PROBE_COUNT}" "${output_txt}"; then
            echo "ERROR: full-probe query warm-up (${FULL_PROBE_COUNT}) was not confirmed." >&2
            return 1
        fi
        if ! grep -q "Performing Dummy Insert Round" "${output_txt}"; then
            echo "ERROR: step-1 insertion warm-up was not confirmed." >&2
            return 1
        fi
        if ! awk -F',' '
            NR > 1 {
                step = $14;
                description = $20;
                gsub(/[[:space:]]/, "", step);
                gsub(/[[:space:]]/, "", description);
                if (step == "0" && description == "probe_time_ms") found = 1;
            }
            END { exit(found ? 0 : 1); }
        ' "${output_csv}"; then
            echo "ERROR: timed step-0 hit-query result is missing from ${output_csv}." >&2
            return 1
        fi
    else
        echo "ERROR: unknown warm-up experiment label ${label}." >&2
        return 1
    fi

    echo "Validated warm-up behavior: ${label}"
}

build_and_run() {
    local label="$1"
    local extra_flags="$2"
    local flags="${COMMON_FLAGS}"
    local output_base="${PREFIX}_${label}"
    local output_txt="${OUTPUT_DIR}/${output_base}.txt"
    local output_csv="${OUTPUT_DIR}/${output_base}.csv"
    local build_log="${OUTPUT_DIR}/${output_base}_build.log"
    local params_log="${OUTPUT_DIR}/${output_base}_parameters.txt"

    if [[ -n "${extra_flags}" ]]; then
        flags="${flags};${extra_flags}"
    fi

    rm -f CMakeCache.txt build.log updates.csv
    rm -f data_cache/*.* 2>/dev/null || true

    {
        echo "label=${label}"
        echo "insertion_path=tiled bulk"
        echo "deletion_path=tiled bulk"
        echo "query_path=single-threaded"
        echo "X=${XVAL} Y=${YVAL} growth=${GROWTH_RATE_VALUE}"
        echo "node_size=${NODE_SIZE} div=${DIV_VALUE}"
        echo "build_size_log=${BUILD_SIZE_LOG} probe_size_log=${PROBE_SIZE_LOG}"
        echo "rounds=${ROUNDS} total_runs=${TOTAL_RUNS}"
        echo "IFDEFS=${flags}"
    } > "${params_log}"

    echo "Building ${label} X90/Y90 tiled-insert/ST-query configuration..."
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

    echo "Running ${label} X90/Y90 tiled-insert/ST-query configuration..."
    ./index_prototype > "${output_txt}" 2>&1

    if [[ ! -s updates.csv ]]; then
        echo "ERROR: updates.csv was not generated for ${label}." >&2
        exit 1
    fi
    cp updates.csv "${output_csv}"
    validate_warmup_behavior "${label}" "${output_txt}" "${output_csv}"
    echo "Completed: ${output_base}"
}

# Experiment 1: strict no-warm-up/no-step-0 build. Both definitions are kept
# explicit: DISABLE_QUERY_HIT_WARMUP disables the query warm-up path, while
# SKIP_STEP0_PROBES prevents every step-0 benchmark kernel from launching.
build_and_run "NoWarmup_NoStep0" "-DDISABLE_QUERY_HIT_WARMUP;-DSKIP_STEP0_PROBES"

# Experiment 2: step 0 runs the untimed full-batch query warm-up and then the
# normal timed step-0 hit/miss queries. Step 1 also warms the insertion path
# before timing the first real insertion batch.
build_and_run "WithStep0_Warmups" "-DENABLE_INSERT_WARMUP"

echo "Both configurations completed."
echo "Results: ${OUTPUT_DIR}"

PYTHON_BIN="python3"
if [[ -x /home/rkharal/anaconda3/bin/python3.9 ]]; then
    PYTHON_BIN="/home/rkharal/anaconda3/bin/python3.9"
fi

"${PYTHON_BIN}" "${OUTPUT_DIR}/plot_nowarmup_nostep0.py"
"${PYTHON_BIN}" "${OUTPUT_DIR}/plot_withstep0_warmups.py"

echo "Plots created in: ${OUTPUT_DIR}"
