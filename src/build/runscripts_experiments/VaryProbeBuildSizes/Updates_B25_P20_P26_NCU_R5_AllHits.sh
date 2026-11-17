#!/usr/bin/env bash
# Profile only the all-hit lookup in update round 5.
#
# Workload:
#   - Build size: 2^25
#   - Probe sizes: 2^20 and 2^26 only
#   - X=25, Y=90
#   - GROWTH_RATE=300
#   - Five insertion rounds followed by five deletion rounds
#   - Range queries disabled
#   - One benchmark run per configuration
#
# Variants:
#   - FliX:    NODESIZE=5, DIV=8
#   - FliX_ST: NODESIZE=3, DIV=8, SINGLE_THREADED_LOOKUPS
#   - LSM_TREE, GPU_BTREE, HASHTABLE_SLAB, HASHTABLE_WARPCORE
#     (NODESIZE and DIV do not affect the baseline implementations)
#
# Lookup-launch selection:
#   - Matching launch 0:  step-0 warm-up hit lookup
#   - Matching launch 1:  step-0 measured hit lookup
#   - Matching launch 2:  step-0 miss lookup
#   - Matching launches 3..10: hit and miss lookups in steps 1..4
#   - Matching launch 11: step-5 all-hit lookup (the requested launch)
#
# Usage:
#   ./Updates_B25_P20_P26_NCU_R5_AllHits.sh --smoke
#   ./Updates_B25_P20_P26_NCU_R5_AllHits.sh --full
#   ./Updates_B25_P20_P26_NCU_R5_AllHits.sh --full --resume

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit 1

RESULT_ROOT="${UPDATES_NCU_RESULT_ROOT_OVERRIDE:-$SCRIPT_DIR/profile_updates_growth300_round5_all_hits}"
SMOKE_ROOT="$RESULT_ROOT/_smoke_test"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
NCU_BIN="${NCU_BIN:-$CUDA_HOME/bin/ncu}"
OPTIX_HOME="${OPTIX_HOME:-/home/rkharal/OPTIX/cgRT_Optimization/rtx-index-updates/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64}"
JOBS="${JOBS:-$(nproc)}"

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export OptiX_INSTALL_DIR="$OPTIX_HOME"

TOTALRUNS=5
ROUNDS=5
BUILD_SIZE_LOG=25
XVAL_VALUE=25
YVAL_VALUE=90
GROWTH_RATE_VALUE=300
LSM_CHUNK_SIZE_LOG=16
RQ_SIZE=8
CACHE_LINE_SIZE=0

PROFILED_STEP=5
ALL_HIT_LOOKUP_LAUNCH_SKIP=11

if [[ -n "${UPDATES_NCU_PROBE_SIZES_OVERRIDE:-}" ]]; then
  IFS=' ' read -r -a PROBE_SIZES <<< "$UPDATES_NCU_PROBE_SIZES_OVERRIDE"
else
  PROBE_SIZES=(20 26)
fi

# Keep this order consistent with the approved experiment/plot ordering.
VARIANT_NAMES=(
  "FliX"
  "FliX_ST"
  "LSM_TREE"
  "GPU_BTREE"
  "HASHTABLE_SLAB"
  "HASHTABLE_WARPCORE"
)

VARIANT_FLAGS=(
  ""
  "-DSINGLE_THREADED_LOOKUPS"
  "-DBASELINES -DLSM_TREE"
  "-DBASELINES -DGPU_BTREE"
  "-DBASELINES -DHASHTABLE_SLAB"
  "-DBASELINES -DHASHTABLE_WARPCORE"
)

# Baselines receive harmless compile defaults because the common headers require
# these macros, but their implementations do not use the FliX node/tile layout.
VARIANT_NODE_SIZES=(5 3 5 5 5 5)
VARIANT_DIVS=(8 8 8 8 8 8)

# These are GPU kernel symbols, not the host API method names.
VARIANT_LOOKUP_KERNELS=(
  "lookup_kernel_tile_ordered"
  "lookup_kernel_tile_ordered_single_threaded"
  "lsm_lookup_kernel"
  "modified_find_kernel"
  "hashtable_slab_search_kernel"
  "warpcore_lookup_kernel"
)

# Compact counter set used by Skew_B25_VaryProbeSize_NCU_STEP3.sh. It captures
# runtime, instruction work, SM utilization, achieved resident-warp occupancy,
# lane participation/divergence, cache traffic, cache hits/misses, and DRAM
# traffic without the much larger replay count of every predefined NCU section.
NCU_METRICS="gpu__time_duration.sum,smsp__inst_executed.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__thread_inst_executed_pred_on_per_inst_executed.ratio,smsp__sass_average_branch_targets_threads_uniform.pct,l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum,lts__t_sector_hit_rate.pct,lts__t_sectors_lookup_hit.sum,lts__t_sectors_lookup_miss.sum,dram__sectors_read.sum,dram__sectors_write.sum"

MODE=""
RESUME=0
FAILURES=0
SUCCESSES=0
SKIPPED=0

usage() {
  cat <<'EOF'
Usage: Updates_B25_P20_P26_NCU_R5_AllHits.sh --smoke|--full [--resume]

  --smoke   Build and profile only FliX/P20, then export and validate it.
  --full    Profile all six variants at P20 and P26 (12 configurations).
  --resume  With --full, skip configurations that already pass validation.

The selected NCU launch is the all-hit lookup in update step 5.
EOF
}

variant_index() {
  local requested="$1"
  local index
  for index in "${!VARIANT_NAMES[@]}"; do
    if [[ "${VARIANT_NAMES[$index]}" == "$requested" ]]; then
      printf '%s\n' "$index"
      return 0
    fi
  done
  return 1
}

calculate_layout() {
  local node_size="$1"
  local cache_line_size="$2"

  if [[ "$cache_line_size" -eq 0 ]]; then
    CALCULATED_MAX_NODE=$((2 ** node_size))
    CALCULATED_TILE_SIZE=$((2 ** node_size))
  elif [[ "$cache_line_size" -eq 10 ]]; then
    CALCULATED_MAX_NODE=14
    CALCULATED_TILE_SIZE=16
  else
    echo "Unsupported CACHE_LINE_SIZE=$cache_line_size" >&2
    return 1
  fi
}

build_configuration() {
  local probe_log="$1"
  local variant_name="$2"
  local variant_flags="$3"
  local node_size="$4"
  local div_factor="$5"
  local output_log="$6"

  calculate_layout "$node_size" "$CACHE_LINE_SIZE" || return 1

  local common_flags
  local all_flags
  local cmake_definitions

  common_flags="-DNODESIZE=${node_size} -DCACHE_LINE_SIZE=${CACHE_LINE_SIZE} -DDENSEKEYGEN -DXVAL=${XVAL_VALUE} -DYVAL=${YVAL_VALUE} -DGROWTH_RATE=${GROWTH_RATE_VALUE} -DTOTALRUNS=${TOTALRUNS} -DMAIN_32 -DDIV=${div_factor} -DTILE_INSERTS -DTILE_INSERTS_C -DINSERTS_TILE_BULK_ONLY -DTILE_DELETES -DDELETES_TILE_BULK -DMAX_NODE=${CALCULATED_MAX_NODE} -DDEFINE_TILE_SIZE=${CALCULATED_TILE_SIZE} -DINITIAL_BUILD_SIZE=${BUILD_SIZE_LOG} -DINITIAL_PROBE_SIZE=${probe_log} -DROUNDS_NUMBER=${ROUNDS} -DLSM_CHUNK_SIZE_LOG=${LSM_CHUNK_SIZE_LOG} -DRQ_SIZE=${RQ_SIZE} -DSKIP_RANGE_QUERIES"
  all_flags="${common_flags} ${variant_flags}"
  cmake_definitions="${all_flags// /;}"

  {
    echo "BUILD_START=$(date --iso-8601=seconds)"
    echo "PWD=$PWD"
    echo "VARIANT=$variant_name"
    echo "IFDEFS=$all_flags"
    echo "COMMAND=cmake .. -DIFDEFS:STRING=<flags>"
  } > "$output_log"

  rm -f CMakeCache.txt build.log
  cmake .. \
    -DCMAKE_CUDA_COMPILER="$CUDACXX" \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DOptiX_INSTALL_DIR="$OPTIX_HOME" \
    -DIFDEFS:STRING="$cmake_definitions" >> "$output_log" 2>&1 || return 1

  make clean >> "$output_log" 2>&1 || return 1
  make -j"$JOBS" >> "$output_log" 2>&1 || return 1

  {
    echo "BUILD_END=$(date --iso-8601=seconds)"
    echo "CMAKE_CACHE_IFDEFS:"
    grep -E '^IFDEFS:' CMakeCache.txt || true
  } >> "$output_log"
}

run_unprofiled_benchmark() {
  local config_dir="$1"
  local output_log="$config_dir/benchmark_unprofiled.log"

  rm -f updates.csv
  ./index_prototype > "$output_log" 2>&1 || return 1
  [[ -s updates.csv ]] || return 1
  cp updates.csv "$config_dir/updates_unprofiled.csv"
}

run_ncu_round5_all_hits() {
  local config_dir="$1"
  local lookup_kernel="$2"
  local report_base="$config_dir/round5_all_hits_lookup"
  local output_log="$config_dir/ncu_round5_all_hits_lookup.log"

  rm -f updates.csv "$report_base.ncu-rep"

  "$NCU_BIN" \
    --kernel-name-base function \
    --kernel-name "$lookup_kernel" \
    --launch-skip "$ALL_HIT_LOOKUP_LAUNCH_SKIP" \
    --launch-count 1 \
    --metrics "$NCU_METRICS" \
    --import-source yes \
    --force-overwrite \
    --export "$report_base" \
    ./index_prototype \
    2>&1 | tee "$output_log"

  local ncu_status=${PIPESTATUS[0]}
  [[ "$ncu_status" -eq 0 ]] || return "$ncu_status"
  [[ -s "$report_base.ncu-rep" ]] || return 1
  [[ -s updates.csv ]] || return 1
  cp updates.csv "$config_dir/updates_profiled.csv"
}

export_report() {
  local config_dir="$1"
  local report="$config_dir/round5_all_hits_lookup.ncu-rep"

  "$NCU_BIN" --import "$report" --page source --print-source cuda,sass \
    > "$config_dir/round5_all_hits_lookup_source.txt" 2>&1 || return 1

  "$NCU_BIN" --import "$report" --page details \
    > "$config_dir/round5_all_hits_lookup_details.txt" 2>&1 || return 1

  "$NCU_BIN" --import "$report" --page raw \
    > "$config_dir/round5_all_hits_lookup_raw.txt" 2>&1 || return 1

  "$NCU_BIN" --import "$report" --page raw --csv \
    > "$config_dir/round5_all_hits_lookup_raw.csv" 2>&1 || return 1
}

validate_configuration() {
  local config_dir="$1"
  local expected_probe_log="$2"
  local expected_variant="$3"
  local expected_kernel="$4"

  local ncu_log="$config_dir/ncu_round5_all_hits_lookup.log"
  local raw="$config_dir/round5_all_hits_lookup_raw.txt"
  local validation="$config_dir/validation.txt"
  local expected_probe_count=$((1 << expected_probe_log))
  local errors=0

  : > "$validation"

  check() {
    local description="$1"
    shift
    if "$@"; then
      echo "PASS: $description" >> "$validation"
    else
      echo "FAIL: $description" >> "$validation"
      errors=$((errors + 1))
    fi
  }

  check "NCU report exists" \
    test -s "$config_dir/round5_all_hits_lookup.ncu-rep"
  check "raw text export exists" \
    test -s "$raw"
  check "raw CSV export exists" \
    test -s "$config_dir/round5_all_hits_lookup_raw.csv"
  check "source export exists" \
    test -s "$config_dir/round5_all_hits_lookup_source.txt"
  check "details export exists" \
    test -s "$config_dir/round5_all_hits_lookup_details.txt"
  check "profiled updates CSV exists" \
    test -s "$config_dir/updates_profiled.csv"

  check "compiled probe log is P${expected_probe_log}" \
    grep -q "probe_size_log: ${expected_probe_log}" "$ncu_log"
  check "observed probe count is ${expected_probe_count}" \
    grep -q "probe_size: ${expected_probe_count}" "$ncu_log"
  check "five update batches are configured" \
    grep -q "batch_count: ${ROUNDS}" "$ncu_log"
  check "insert growth is 300 percent" \
    grep -q "total_inserts_percentage_of_build_size: ${GROWTH_RATE_VALUE}" "$ncu_log"
  check "delete growth is 300 percent" \
    grep -q "total_deletes_percentage_of_build_size: ${GROWTH_RATE_VALUE}" "$ncu_log"

  check "exactly one GPU kernel launch was profiled" \
    test "$(grep -c '==PROF== Profiling "' "$ncu_log")" -eq 1
  check "profiled GPU kernel is ${expected_kernel}" \
    grep -q "==PROF== Profiling \".*${expected_kernel}" "$ncu_log"
  check "profiled launch occurs inside update step 5" \
    awk '
      /----> Update step 5:/ { in_step5=1; next }
      in_step5 && /----> Update step 6:/ { in_step5=0 }
      in_step5 && /==PROF== Profiling "/ { found=1 }
      END { exit(found ? 0 : 1) }
    ' "$ncu_log"
  check "step 5 is an insertion round" \
    awk '
      /----> Update step 5:/ { in_step5=1; next }
      in_step5 && /----> step 5: insert/ { found=1 }
      in_step5 && /----> Update step 6:/ { in_step5=0 }
      END { exit(found ? 0 : 1) }
    ' "$ncu_log"
  check "step-5 miss lookup is absent after exhausting the generated miss range" \
    awk '
      /----> Update step 5:/ { in_step5=1; next }
      in_step5 && /----> Update step 6:/ { in_step5=0 }
      in_step5 && /--Probe Miss Time: -> 0([.]0+)? ms/ { found=1 }
      END { exit(found ? 0 : 1) }
    ' "$ncu_log"
  check "NCU reported no error" \
    sh -c "! grep -q '==ERROR==' \"$ncu_log\""

  local metric
  local required_metrics
  IFS=',' read -r -a required_metrics <<< "$NCU_METRICS"
  for metric in "${required_metrics[@]}"; do
    check "metric present: $metric" grep -q "$metric" "$raw"
  done

  local passes
  passes="$(sed -n 's/.*- \([0-9][0-9]*\) passes.*/\1/p' "$ncu_log" | head -1)"

  {
    echo "EXPECTED_VARIANT=$expected_variant"
    echo "EXPECTED_KERNEL=$expected_kernel"
    echo "PROFILED_OPERATION=round_5_all_hits_lookup"
    echo "LOOKUP_LAUNCH_SKIP=$ALL_HIT_LOOKUP_LAUNCH_SKIP"
    echo "NCU_REPLAY_PASSES=${passes:-unknown}"
    echo "VALIDATION_ERRORS=$errors"
  } >> "$validation"

  [[ "$errors" -eq 0 ]]
}

record_metadata() {
  local config_dir="$1"
  local probe_log="$2"
  local variant_name="$3"
  local variant_flags="$4"
  local node_size="$5"
  local div_factor="$6"
  local lookup_kernel="$7"
  local parameters_apply="yes"

  if [[ "$variant_name" != "FliX" && "$variant_name" != "FliX_ST" ]]; then
    parameters_apply="no (baseline compile defaults only)"
  fi

  {
    printf 'VARIANT\t%s\n' "$variant_name"
    printf 'VARIANT_FLAGS\t%s\n' "$variant_flags"
    printf 'LOOKUP_GPU_KERNEL\t%s\n' "$lookup_kernel"
    printf 'XVAL\t%s\n' "$XVAL_VALUE"
    printf 'YVAL\t%s\n' "$YVAL_VALUE"
    printf 'GROWTH_RATE\t%s\n' "$GROWTH_RATE_VALUE"
    printf 'BUILD_LOG\t%s\n' "$BUILD_SIZE_LOG"
    printf 'PROBE_LOG\t%s\n' "$probe_log"
    printf 'PROBE_COUNT\t%s\n' "$((1 << probe_log))"
    printf 'ROUNDS\t%s\n' "$ROUNDS"
    printf 'INSERT_ROUNDS\t%s\n' "$ROUNDS"
    printf 'DELETE_ROUNDS\t%s\n' "$ROUNDS"
    printf 'PROFILED_STEP\t%s\n' "$PROFILED_STEP"
    printf 'PROFILED_OPERATION\t%s\n' "all_hits_lookup"
    printf 'LOOKUP_LAUNCH_SKIP\t%s\n' "$ALL_HIT_LOOKUP_LAUNCH_SKIP"
    printf 'NODE_SIZE\t%s\n' "$node_size"
    printf 'DIV\t%s\n' "$div_factor"
    printf 'FLIX_LAYOUT_PARAMETERS_APPLY\t%s\n' "$parameters_apply"
    printf 'RANGE_QUERIES\t%s\n' "disabled"
  } > "$config_dir/metadata.tsv"
}

run_configuration() {
  local run_root="$1"
  local probe_log="$2"
  local variant_name="$3"
  local run_unprofiled="$4"

  local index
  index="$(variant_index "$variant_name")" || {
    echo "Unknown variant: $variant_name" >&2
    return 1
  }

  local variant_flags="${VARIANT_FLAGS[$index]}"
  local node_size="${VARIANT_NODE_SIZES[$index]}"
  local div_factor="${VARIANT_DIVS[$index]}"
  local lookup_kernel="${VARIANT_LOOKUP_KERNELS[$index]}"
  local config_dir="$run_root/P${probe_log}/${variant_name}"

  mkdir -p "$config_dir"

  if [[ "$RESUME" -eq 1 ]] && \
     validate_configuration "$config_dir" "$probe_log" "$variant_name" "$lookup_kernel" \
       >/dev/null 2>&1; then
    echo "SKIP valid existing result: P=$probe_log variant=$variant_name"
    SKIPPED=$((SKIPPED + 1))
    return 0
  fi

  record_metadata \
    "$config_dir" "$probe_log" "$variant_name" "$variant_flags" \
    "$node_size" "$div_factor" "$lookup_kernel"

  rm -f "$config_dir/FAILED" "$config_dir/COMPLETE"
  rm -f data_cache/*.* 2>/dev/null || true

  echo "============================================================"
  echo "START P=$probe_log variant=$variant_name at $(date --iso-8601=seconds)"
  echo "Directory: $config_dir"
  echo "GPU lookup kernel: $lookup_kernel"
  echo "NCU selection: --launch-skip $ALL_HIT_LOOKUP_LAUNCH_SKIP --launch-count 1"

  if ! build_configuration \
      "$probe_log" "$variant_name" "$variant_flags" \
      "$node_size" "$div_factor" "$config_dir/build.log"; then
    echo "ERROR: build failed for P=$probe_log variant=$variant_name" \
      | tee "$config_dir/FAILED"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  if [[ "$run_unprofiled" -eq 1 ]]; then
    if ! run_unprofiled_benchmark "$config_dir"; then
      echo "ERROR: unprofiled smoke benchmark failed for P=$probe_log variant=$variant_name" \
        | tee "$config_dir/FAILED"
      FAILURES=$((FAILURES + 1))
      return 0
    fi
  fi

  if ! run_ncu_round5_all_hits "$config_dir" "$lookup_kernel"; then
    echo "ERROR: NCU failed for P=$probe_log variant=$variant_name" \
      | tee "$config_dir/FAILED"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  if ! export_report "$config_dir"; then
    echo "ERROR: NCU report export failed for P=$probe_log variant=$variant_name" \
      | tee "$config_dir/FAILED"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  if ! validate_configuration \
      "$config_dir" "$probe_log" "$variant_name" "$lookup_kernel"; then
    echo "ERROR: validation failed for P=$probe_log variant=$variant_name" \
      | tee "$config_dir/FAILED"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  touch "$config_dir/COMPLETE"
  SUCCESSES=$((SUCCESSES + 1))
  echo "PASS P=$probe_log variant=$variant_name at $(date --iso-8601=seconds)"
}

write_manifest() {
  local manifest="$1"

  {
    echo "START=$(date --iso-8601=seconds)"
    echo "HOST=$(hostname)"
    echo "BUILD_LOG=$BUILD_SIZE_LOG"
    echo "PROBE_LOGS=${PROBE_SIZES[*]}"
    echo "VARIANTS=${VARIANT_NAMES[*]}"
    echo "CONFIGURATIONS=$((${#PROBE_SIZES[@]} * ${#VARIANT_NAMES[@]}))"
    echo "ROUNDS=$ROUNDS"
    echo "INSERT_ROUNDS=$ROUNDS"
    echo "DELETE_ROUNDS=$ROUNDS"
    echo "GROWTH_RATE=$GROWTH_RATE_VALUE"
    echo "PROFILED_STEP=$PROFILED_STEP"
    echo "PROFILED_OPERATION=all_hits_lookup"
    echo "LOOKUP_LAUNCH_SKIP=$ALL_HIT_LOOKUP_LAUNCH_SKIP"
    echo "NCU_METRICS=$NCU_METRICS"
  } > "$manifest"
}

main() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --smoke|--full)
        MODE="${1#--}"
        ;;
      --resume)
        RESUME=1
        ;;
      --help|-h)
        usage
        return 0
        ;;
      *)
        echo "Unknown argument: $1" >&2
        usage >&2
        return 2
        ;;
    esac
    shift
  done

  if [[ -z "$MODE" ]]; then
    usage >&2
    return 2
  fi

  local expected_array_size="${#VARIANT_NAMES[@]}"
  if [[ "${#VARIANT_FLAGS[@]}" -ne "$expected_array_size" ||
        "${#VARIANT_NODE_SIZES[@]}" -ne "$expected_array_size" ||
        "${#VARIANT_DIVS[@]}" -ne "$expected_array_size" ||
        "${#VARIANT_LOOKUP_KERNELS[@]}" -ne "$expected_array_size" ]]; then
    echo "Variant configuration arrays have inconsistent lengths." >&2
    return 2
  fi

  if [[ ! -x "$NCU_BIN" ]]; then
    echo "NCU executable not found: $NCU_BIN" >&2
    return 2
  fi

  mkdir -p "$RESULT_ROOT" data_cache

  local run_root="$RESULT_ROOT"
  local manifest="$RESULT_ROOT/full_sweep_started.txt"

  if [[ "$MODE" == "smoke" ]]; then
    run_root="$SMOKE_ROOT"
    rm -rf "$SMOKE_ROOT"
    mkdir -p "$SMOKE_ROOT"
    run_configuration "$run_root" 20 "FliX" 1
  else
    write_manifest "$manifest"

    local probe_log
    local variant_name
    for probe_log in "${PROBE_SIZES[@]}"; do
      for variant_name in "${VARIANT_NAMES[@]}"; do
        run_configuration "$run_root" "$probe_log" "$variant_name" 0
      done
    done
  fi

  {
    echo "END=$(date --iso-8601=seconds)"
    echo "SUCCESSES=$SUCCESSES"
    echo "SKIPPED=$SKIPPED"
    echo "FAILURES=$FAILURES"
  } | tee "$run_root/${MODE}_summary.txt"

  [[ "$FAILURES" -eq 0 ]]
}

main "$@"
