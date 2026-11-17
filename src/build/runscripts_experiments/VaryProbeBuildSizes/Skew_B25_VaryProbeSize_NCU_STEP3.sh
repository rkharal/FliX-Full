#!/usr/bin/env bash
# Query-only skew sweep with lightweight Nsight Compute profiling of step 3.
#
# Usage:
#   ./Skew_B25_VaryProbeSize_NCU_STEP3.sh --smoke
#   ./Skew_B25_VaryProbeSize_NCU_STEP3.sh --full
#   ./Skew_B25_VaryProbeSize_NCU_STEP3.sh --full --resume
#
# The smoke run builds and runs X=0.0001/P8 once without NCU, then profiles it.
# The full run profiles all configured X values and probe sizes.  NCU's fourth
# matching real lookup is step 3 because match 0 is the step-0 warm-up lookup.

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit 1

RESULT_ROOT="$SCRIPT_DIR/profile_4_full_skew_results"
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
ROUNDS=8
BUILDSIZE=25
LSM_CHUNK_SIZE_LOG=16
RQ_SIZE=8
PD=100
PU=0
YVAL=90
NODE_SIZE=5
CACHE_LINE_SIZE=0
MAX_NODE=32
TILE_SIZE=32

PROBE_SIZES=(8 10 12 14 15 16 18 20 22 24 26 28)
X_VALUES=(0.0001 0.001 0.01 0.1 1 10 20 50 100)

# A deliberately compact counter set.  It preserves the core P14/P15/P16
# comparisons without collecting the broad sections that caused 35 replays.
NCU_METRICS="gpu__time_duration.sum,smsp__inst_executed.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__thread_inst_executed_pred_on_per_inst_executed.ratio,smsp__sass_average_branch_targets_threads_uniform.pct,l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum,lts__t_sector_hit_rate.pct,lts__t_sectors_lookup_hit.sum,lts__t_sectors_lookup_miss.sum,dram__sectors_read.sum,dram__sectors_write.sum"

MODE=""
RESUME=0
FAILURES=0
SUCCESSES=0
SKIPPED=0

sanitize_value() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/_}"
  printf '%s' "$value"
}

usage() {
  cat <<'EOF'
Usage: Skew_B25_VaryProbeSize_NCU_STEP3.sh --smoke|--full [--resume]

  --smoke   Build, run, profile, export, and validate X=0.0001/P8.
  --full    Profile and validate the complete X/probe grid.
  --resume  Skip configurations that already pass validation.
EOF
}

build_configuration() {
  local xval="$1"
  local probe_log="$2"
  local output_log="$3"
  local common_flags
  local cmake_definitions
  common_flags="-DNODESIZE=${NODE_SIZE} -DCACHE_LINE_SIZE=${CACHE_LINE_SIZE} -DDENSEKEYGEN -DYVAL=${YVAL} -DTOTALRUNS=${TOTALRUNS} -DMAIN_32 -DDIV=8 -DTILE_INSERTS -DTILE_INSERTS_C -DINSERTS_TILE_BULK_ONLY -DTILE_DELETES -DDELETES_TILE_BULK -DMAX_NODE=${MAX_NODE} -DDEFINE_TILE_SIZE=${TILE_SIZE} -DINITIAL_BUILD_SIZE=${BUILDSIZE} -DINITIAL_PROBE_SIZE=${probe_log} -DROUNDS_NUMBER=${ROUNDS} -DLSM_CHUNK_SIZE_LOG=${LSM_CHUNK_SIZE_LOG} -DXVAL=${xval} -DPD=${PD} -DPU=${PU}"
  cmake_definitions="${common_flags// /;}"

  {
    echo "BUILD_START=$(date --iso-8601=seconds)"
    echo "PWD=$PWD"
    echo "IFDEFS=$common_flags"
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
  echo "BUILD_END=$(date --iso-8601=seconds)" >> "$output_log"
}

run_unprofiled_benchmark() {
  local config_dir="$1"
  local output_log="$config_dir/benchmark_unprofiled.log"
  rm -f updates.csv
  ./index_prototype > "$output_log" 2>&1 || return 1
  [[ -s updates.csv ]] || return 1
  cp updates.csv "$config_dir/updates_unprofiled.csv"
}

run_ncu_step3() {
  local config_dir="$1"
  local report_base="$config_dir/lookup_step3_memory"
  local output_log="$config_dir/ncu_lookup_step3_memory.log"

  rm -f updates.csv "$report_base.ncu-rep"
  "$NCU_BIN" \
    --kernel-name-base function \
    --kernel-name lookup_kernel_tile_ordered \
    --launch-skip 4 \
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
  local report="$config_dir/lookup_step3_memory.ncu-rep"

  "$NCU_BIN" --import "$report" --page source --print-source cuda,sass \
    > "$config_dir/lookup_step3_memory_source.txt" 2>&1 || return 1
  "$NCU_BIN" --import "$report" --page details \
    > "$config_dir/lookup_step3_memory_details.txt" 2>&1 || return 1
  "$NCU_BIN" --import "$report" --page raw \
    > "$config_dir/lookup_step3_memory_raw.txt" 2>&1 || return 1
  "$NCU_BIN" --import "$report" --page raw --csv \
    > "$config_dir/lookup_step3_memory_raw.csv" 2>&1 || return 1
}

validate_configuration() {
  local config_dir="$1"
  local expected_probe_log="$2"
  local ncu_log="$config_dir/ncu_lookup_step3_memory.log"
  local raw="$config_dir/lookup_step3_memory_raw.txt"
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

  check "NCU report exists" test -s "$config_dir/lookup_step3_memory.ncu-rep"
  check "raw text export exists" test -s "$raw"
  check "raw CSV export exists" test -s "$config_dir/lookup_step3_memory_raw.csv"
  check "source export exists" test -s "$config_dir/lookup_step3_memory_source.txt"
  check "details export exists" test -s "$config_dir/lookup_step3_memory_details.txt"
  check "profiled updates CSV exists" test -s "$config_dir/updates_profiled.csv"
  check "compiled probe log is P${expected_probe_log}" grep -q "probe_size_log: ${expected_probe_log}" "$ncu_log"
  check "observed probe count is ${expected_probe_count}" grep -q "probe_size: ${expected_probe_count}" "$ncu_log"
  check "query-only mode is active" grep -q "query_only_mode: YES" "$ncu_log"
  check "insert percentage is zero" grep -q "total_inserts_percentage_of_build_size: 0" "$ncu_log"
  check "delete percentage is zero" grep -q "total_deletes_percentage_of_build_size: 0" "$ncu_log"
  check "exactly one lookup launch was profiled" test "$(grep -c '==PROF== Profiling "lookup_kernel_tile_ordered"' "$ncu_log")" -eq 1
  check "profiled launch occurs within step 3" awk '
    /Update step 3:/ { in_step3=1; next }
    in_step3 && /Update step 4:/ { in_step3=0 }
    in_step3 && /==PROF== Profiling "lookup_kernel_tile_ordered"/ { found=1 }
    END { exit(found ? 0 : 1) }
  ' "$ncu_log"
  check "NCU reported no error" sh -c "! grep -q '==ERROR==' \"$ncu_log\""

  local metric
  IFS=',' read -r -a required_metrics <<< "$NCU_METRICS"
  for metric in "${required_metrics[@]}"; do
    check "metric present: $metric" grep -q "$metric" "$raw"
  done

  local passes
  passes="$(sed -n 's/.*- \([0-9][0-9]*\) passes.*/\1/p' "$ncu_log" | head -1)"
  echo "NCU_REPLAY_PASSES=${passes:-unknown}" >> "$validation"
  echo "VALIDATION_ERRORS=$errors" >> "$validation"
  [[ "$errors" -eq 0 ]]
}

record_metadata() {
  local config_dir="$1"
  local xval="$2"
  local probe_log="$3"
  cat > "$config_dir/metadata.tsv" <<EOF
XVAL	$xval
YVAL	$YVAL
PROBE_LOG	$probe_log
PROBE_COUNT	$((1 << probe_log))
BUILD_LOG	$BUILDSIZE
ROUNDS	$ROUNDS
PROFILED_STEP	3
PD	$PD
PU	$PU
EOF
}

run_configuration() {
  local run_root="$1"
  local xval="$2"
  local probe_log="$3"
  local run_unprofiled="$4"
  local x_label
  x_label="$(sanitize_value "$xval")"
  local config_dir="$run_root/X${x_label}/P${probe_log}"
  mkdir -p "$config_dir"

  if [[ "$RESUME" -eq 1 ]] && validate_configuration "$config_dir" "$probe_log" >/dev/null 2>&1; then
    echo "SKIP valid existing result: X=$xval P=$probe_log"
    SKIPPED=$((SKIPPED + 1))
    return 0
  fi

  record_metadata "$config_dir" "$xval" "$probe_log"
  echo "============================================================"
  echo "START X=$xval P=$probe_log at $(date --iso-8601=seconds)"
  echo "Directory: $config_dir"

  if ! build_configuration "$xval" "$probe_log" "$config_dir/build.log"; then
    echo "ERROR: build failed for X=$xval P=$probe_log" | tee "$config_dir/FAILED"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  if [[ "$run_unprofiled" -eq 1 ]]; then
    if ! run_unprofiled_benchmark "$config_dir"; then
      echo "ERROR: unprofiled smoke benchmark failed for X=$xval P=$probe_log" | tee "$config_dir/FAILED"
      FAILURES=$((FAILURES + 1))
      return 0
    fi
  fi

  if ! run_ncu_step3 "$config_dir"; then
    echo "ERROR: NCU failed for X=$xval P=$probe_log" | tee "$config_dir/FAILED"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  if ! export_report "$config_dir"; then
    echo "ERROR: report export failed for X=$xval P=$probe_log" | tee "$config_dir/FAILED"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  if ! validate_configuration "$config_dir" "$probe_log"; then
    echo "ERROR: validation failed for X=$xval P=$probe_log" | tee "$config_dir/FAILED"
    FAILURES=$((FAILURES + 1))
    return 0
  fi

  rm -f "$config_dir/FAILED"
  touch "$config_dir/COMPLETE"
  SUCCESSES=$((SUCCESSES + 1))
  echo "PASS X=$xval P=$probe_log at $(date --iso-8601=seconds)"
}

main() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --smoke|--full) MODE="${1#--}" ;;
      --resume) RESUME=1 ;;
      --help|-h) usage; return 0 ;;
      *) echo "Unknown argument: $1" >&2; usage >&2; return 2 ;;
    esac
    shift
  done

  if [[ -z "$MODE" ]]; then
    usage >&2
    return 2
  fi
  if [[ ! -x "$NCU_BIN" ]]; then
    echo "NCU executable not found: $NCU_BIN" >&2
    return 2
  fi
  if [[ ! -x ./index_prototype && "$MODE" == "full" ]]; then
    echo "NOTE: index_prototype will be produced by the first build."
  fi

  mkdir -p "$RESULT_ROOT" data_cache
  local run_root="$RESULT_ROOT"
  local manifest

  if [[ "$MODE" == "smoke" ]]; then
    run_root="$SMOKE_ROOT"
    rm -rf "$SMOKE_ROOT"
    mkdir -p "$SMOKE_ROOT"
    : > data_cache/keys_cache.txt
    run_configuration "$run_root" "0.0001" 8 1
  else
    manifest="$RESULT_ROOT/full_sweep_started.txt"
    {
      echo "START=$(date --iso-8601=seconds)"
      echo "HOST=$(hostname)"
      echo "BUILD_LOG=$BUILDSIZE"
      echo "PROBE_LOGS=${PROBE_SIZES[*]}"
      echo "X_VALUES=${X_VALUES[*]}"
      echo "CONFIGURATIONS=$((${#PROBE_SIZES[@]} * ${#X_VALUES[@]}))"
      echo "NCU_METRICS=$NCU_METRICS"
    } > "$manifest"
    if [[ "$RESUME" -eq 0 ]]; then
      : > data_cache/keys_cache.txt
    fi
    local probe_log xval
    for probe_log in "${PROBE_SIZES[@]}"; do
      for xval in "${X_VALUES[@]}"; do
        run_configuration "$run_root" "$xval" "$probe_log" 0
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
