#!/usr/bin/env bash
# Warm-up comparison using alternating update rounds:
#   R1 insert, R2 delete, R3 insert, R4 delete,
#   R5 insert, R6 delete, R7 insert, R8 delete.
#
# Configuration: X90/Y90, growth 400%, node-size log 4, DIV 1,
# build size 2^12, probe size 2^12, tiled inserts/deletes, and
# single-threaded point queries.
#
# Experiment 1 (NoWarmup_NoStep0): step 0 and all warm-ups are disabled.
# Experiment 2 (WithStep0_Warmups): full query warm-up and timed queries at
# step 0, insertion warm-up before R1, and one untimed 1,000-key deletion
# warm-up before the first real deletion at R2. Each deletion batch contains
# 4,096 keys, so the warm-up covers approximately 24.4% of the batch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/WarmupTests/TiledInsert_STQuery_X90Y90_B12P12_Growth400_NS4_DIV1_Alternating_CorrectedFullQuery_DeleteWarmup1000"
mkdir -p "${OUTPUT_DIR}"
cd "${BUILD_DIR}"

XVAL=90
YVAL=90
GROWTH_RATE_VALUE=400
NODE_SIZE=4
DIV_VALUE=1
BUILD_SIZE_LOG=12
PROBE_SIZE_LOG=12
ROUNDS=4
TOTAL_RUNS=5
MAX_NODE=$((2 ** NODE_SIZE))
TILE_SIZE=$((2 ** NODE_SIZE))
FULL_PROBE_COUNT=$((2 ** PROBE_SIZE_LOG))
DELETE_WARMUP_COUNT=1000
DELETE_BATCH_COUNT=$(( (2 ** BUILD_SIZE_LOG) * GROWTH_RATE_VALUE / (100 * ROUNDS) ))
EXPECTED_DELETE_WARMUP_COUNT=$(( DELETE_WARMUP_COUNT < DELETE_BATCH_COUNT ? DELETE_WARMUP_COUNT : DELETE_BATCH_COUNT ))

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

PREFIX="X${XVAL}Y${YVAL}_Growth${GROWTH_RATE_VALUE}_NS${NODE_SIZE}_DIV${DIV_VALUE}_B${BUILD_SIZE_LOG}_P${PROBE_SIZE_LOG}_R${ROUNDS}_FliX_Alternating_TiledInsert_STQuery_12_12"

# ALTERNATING_INSERT_DELETE selects insert/delete/insert/delete scheduling.
COMMON_FLAGS="-DNODESIZE=${NODE_SIZE};-DCACHE_LINE_SIZE=0;-DDENSEKEYGEN;-DXVAL=${XVAL};-DYVAL=${YVAL};-DGROWTH_RATE=${GROWTH_RATE_VALUE};-DTOTALRUNS=${TOTAL_RUNS};-DMAIN_32;-DDIV=${DIV_VALUE};-DTILE_INSERTS;-DTILE_INSERTS_C;-DINSERTS_TILE_BULK_ONLY;-DTILE_DELETES;-DDELETES_TILE_BULK;-DSINGLE_THREADED_LOOKUPS;-DALTERNATING_INSERT_DELETE;-DMAX_NODE=${MAX_NODE};-DDEFINE_TILE_SIZE=${TILE_SIZE};-DINITIAL_BUILD_SIZE=${BUILD_SIZE_LOG};-DINITIAL_PROBE_SIZE=${PROBE_SIZE_LOG};-DROUNDS_NUMBER=${ROUNDS};-DLSM_CHUNK_SIZE_LOG=16;-DPD=100;-DPU=0"

validate_alternating_schedule() {
    local output_csv="$1"

    if ! awk -F',' '
        NR > 1 {
            step = $14; do_insert = $15; do_delete = $16;
            gsub(/[[:space:]]/, "", step);
            gsub(/[[:space:]]/, "", do_insert);
            gsub(/[[:space:]]/, "", do_delete);
            if (step >= 1 && step <= 8) {
                seen[step] = 1;
                if ((step % 2 == 1 && !(do_insert == 1 && do_delete == 0)) ||
                    (step % 2 == 0 && !(do_insert == 0 && do_delete == 1))) {
                    bad = 1;
                }
            }
        }
        END {
            for (step = 1; step <= 8; ++step) {
                if (!seen[step]) bad = 1;
            }
            exit(bad ? 1 : 0);
        }
    ' "${output_csv}"; then
        echo "ERROR: ${output_csv} does not contain the expected alternating R1-R8 schedule." >&2
        return 1
    fi

    echo "Validated alternating schedule: odd rounds insert, even rounds delete."
}

validate_warmup_behavior() {
    local label="$1"
    local output_txt="$2"
    local output_csv="$3"

    validate_alternating_schedule "${output_csv}"

    if [[ "${label}" == "NoWarmup_NoStep0" ]]; then
        grep -q "step 0: STRICT SKIP" "${output_txt}" || {
            echo "ERROR: strict step-0 skip marker is missing." >&2
            return 1
        }
        if grep -Eq "Point-query hit warm-up size:|Performing Dummy Insert Round|Performing Dummy Delete Round" "${output_txt}"; then
            echo "ERROR: a warm-up unexpectedly ran in ${label}." >&2
            return 1
        fi
        if awk -F',' '
            NR > 1 {
                step = $14; gsub(/[[:space:]]/, "", step);
                if (step == 0) found = 1;
            }
            END { exit(found ? 0 : 1); }
        ' "${output_csv}"; then
            echo "ERROR: ${label} recorded step-0 measurements." >&2
            return 1
        fi
    elif [[ "${label}" == "WithStep0_Warmups" ]]; then
        grep -q "Point-query hit warm-up size: ${FULL_PROBE_COUNT}" "${output_txt}" || {
            echo "ERROR: full-probe query warm-up was not confirmed." >&2
            return 1
        }
        grep -q "Performing Dummy Insert Round" "${output_txt}" || {
            echo "ERROR: insertion warm-up was not confirmed." >&2
            return 1
        }
        grep -q "Dummy delete size: ${EXPECTED_DELETE_WARMUP_COUNT}" "${output_txt}" || {
            echo "ERROR: effective ${EXPECTED_DELETE_WARMUP_COUNT}-key deletion warm-up was not confirmed." >&2
            return 1
        }
        local delete_warmup_runs
        delete_warmup_runs=$(grep -c "Performing Dummy Delete Round" "${output_txt}" || true)
        if [[ "${delete_warmup_runs}" -ne 1 ]]; then
            echo "ERROR: expected one deletion warm-up, found ${delete_warmup_runs}." >&2
            return 1
        fi
        if ! awk -F',' '
            NR > 1 {
                step = $14; description = $20;
                gsub(/[[:space:]]/, "", step);
                gsub(/[[:space:]]/, "", description);
                if (step == 0 && description == "probe_time_ms") found = 1;
            }
            END { exit(found ? 0 : 1); }
        ' "${output_csv}"; then
            echo "ERROR: timed step-0 hit-query result is missing." >&2
            return 1
        fi
    else
        echo "ERROR: unknown experiment label ${label}." >&2
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
        echo "schedule=alternating insert/delete (R1 insert, R2 delete, through R8 delete)"
        echo "insertion_path=tiled bulk"
        echo "deletion_path=tiled bulk"
        echo "query_path=single-threaded"
        echo "X=${XVAL} Y=${YVAL} growth=${GROWTH_RATE_VALUE}"
        echo "node_size=${NODE_SIZE} div=${DIV_VALUE}"
        echo "build_size_log=${BUILD_SIZE_LOG} probe_size_log=${PROBE_SIZE_LOG}"
        echo "rounds=${ROUNDS} total_runs=${TOTAL_RUNS}"
        echo "full_query_warmup_count=${FULL_PROBE_COUNT}"
        echo "delete_warmup_requested_count=${DELETE_WARMUP_COUNT}"
        echo "delete_batch_count=${DELETE_BATCH_COUNT}"
        echo "delete_warmup_effective_count=${EXPECTED_DELETE_WARMUP_COUNT}"
        echo "IFDEFS=${flags}"
    } > "${params_log}"

    echo "Building ${label} alternating-update configuration..."
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

    echo "Running ${label} alternating-update configuration..."
    ./index_prototype > "${output_txt}" 2>&1

    if [[ ! -s updates.csv ]]; then
        echo "ERROR: updates.csv was not generated for ${label}." >&2
        exit 1
    fi
    cp updates.csv "${output_csv}"
    validate_warmup_behavior "${label}" "${output_txt}" "${output_csv}"
    echo "Completed: ${output_base}"
}

# No step 0 and no query, insertion, or deletion warm-up.
build_and_run "NoWarmup_NoStep0" "-DDISABLE_QUERY_HIT_WARMUP;-DSKIP_STEP0_PROBES"

# Full-query warm-up at step 0, insertion warm-up at R1, and one 1,000-key
# deletion warm-up immediately before the first real 4,096-key deletion at R2.
build_and_run "WithStep0_Warmups" "-DENABLE_INSERT_WARMUP;-DENABLE_DELETE_WARMUP;-DDELETE_WARMUP_KEY_COUNT=${DELETE_WARMUP_COUNT}"

echo "Both alternating configurations completed and validated."
echo "Results: ${OUTPUT_DIR}"

