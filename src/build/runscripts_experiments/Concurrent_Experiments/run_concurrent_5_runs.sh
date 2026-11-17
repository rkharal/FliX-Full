#!/usr/bin/env bash
# Shared implementation for the relocated five-run mixed-batch drivers.
set -euo pipefail
[[ $# == 3 ]] || { echo 'Usage: run_concurrent_5_runs.sh MODE OUTPUT_DIR --dry-run|--run' >&2; exit 2; }
kind="$1"; output="$2"; action="$3"
[[ "$kind" == reuse || "$kind" == old ]] || exit 2
[[ "$action" == --dry-run || "$action" == --run ]] || exit 2
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd -- "$here/../../.." && pwd)"
validator="$here/validate_concurrent_5_runs.py"
configs=('5 5 90 33554432 8 standard' '25 25 50 33554432 8 standard' '45 45 10 8388608 4 adaptive')
variants=(FliX-ST FliX-TL)
[[ "$kind" == old ]] && variants+=(GPU_BTREE SLABHASH)
flags_for() {
 local v="$1" i="$2" d="$3" q="$4" build="$5" rounds="$6" policy="$7"
 local node=5 width=32 div=8 extra='' adaptive=''
 case "$v" in
 FliX-ST) node=3; width=8; div=1; extra='-DFLIX_MIXED_SINGLE_THREADED';;
 FliX-TL) ;;
 GPU_BTREE) extra='-DBASELINES -DGPU_BTREE';;
 SLABHASH) extra='-DBASELINES -DHASHTABLE_SLAB';;
 esac
 [[ "$kind" == reuse ]] && extra="$extra -DFLIX_MIXED_REUSE_NODES"
 [[ "$policy" == adaptive ]] && adaptive='-DCONCURRENT_SMALL_BUILD_ADAPTIVE_DELETES'
 printf '%s' "-DNODESIZE=$node -DCACHE_LINE_SIZE=0 -DREGKEYGEN -DTOTALRUNS=5 -DMAIN_32 -DDIV=$div -DMAX_NODE=$width -DDEFINE_TILE_SIZE=$width $extra -DInsert_mix=$i -DDel_mix=$d -DQuery_mix=$q -DCONCURRENT_ROUNDS=$rounds -DCONCURRENT_BUILD_SIZE=$build -DCONCURRENT_BATCH_SIZE=100000000 $adaptive"
}
echo "Mode=$kind; outputs=$output; 100M batch; five runs; source=$source_dir"
for config in "${configs[@]}"; do
 read -r i d q build rounds policy <<< "$config"
 for v in "${variants[@]}"; do echo "$v $i/$d/$q: $(flags_for "$v" "$i" "$d" "$q" "$build" "$rounds" "$policy")"; done
done
[[ "$action" == --run ]] || { echo 'Dry run: no files written or GPU work performed.'; exit 0; }
[[ "$(hostname -s | tr '[:upper:]' '[:lower:]')" == zyra ]] || { echo 'Run this configuration on Zyra.' >&2; exit 1; }
cc="${CC:-$(command -v gcc)}"; cxx="${CXX:-$(command -v g++)}"; nvcc="${CUDACXX:-$(command -v nvcc || true)}"
[[ -n "$nvcc" ]] || { echo 'nvcc not found' >&2; exit 1; }
for compiler in "$cc" "$cxx" "$nvcc"; do command -v "$compiler" >/dev/null; done
cuda_root="$(cd -- "$(dirname -- "$(readlink -f -- "$(command -v "$nvcc")")")/.." && pwd)"
export PATH="$cuda_root/bin:$PATH" LD_LIBRARY_PATH="$cuda_root/lib64:${LD_LIBRARY_PATH:-}"
export FLIX_CONCURRENT_KEY_CACHE="${FLIX_CONCURRENT_KEY_CACHE:-/home/rkharal/OPTIX/Concurrent_Key_Cache}"
optix="${OptiX_INSTALL_DIR:-/home/rkharal/OPTIX/cgRT_Optimization/rtx-index-updates/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64}"
[[ -f "$optix/include/optix.h" ]] || { echo 'OptiX not found' >&2; exit 1; }
tmp="$(mktemp -d /tmp/flix-concurrent5.XXXXXXXX)"
trap 'rm -rf -- "$tmp"' EXIT
session="Zyra_${kind}_5runs_$(date +%Y%m%dT%H%M%S)_$$"
for config in "${configs[@]}"; do
 read -r i d q build rounds policy <<< "$config"
 for v in "${variants[@]}"; do
  flags="$(flags_for "$v" "$i" "$d" "$q" "$build" "$rounds" "$policy")"
  base="$output/${session}_I${i}_D${d}_Q${q}_${v}"
  cmake -S "$source_dir" -B "$tmp/build" -DCMAKE_C_COMPILER="$cc" -DCMAKE_CXX_COMPILER="$cxx" \
   -DCMAKE_CUDA_COMPILER="$nvcc" -DCUDA_TOOLKIT_ROOT_DIR="$cuda_root" -DCUDAToolkit_ROOT="$cuda_root" \
   -DOptiX_INSTALL_DIR="$optix" -DIFDEFS:STRING=" $flags"
  cmake --build "$tmp/build" --clean-first --parallel "${BUILD_JOBS:-8}"
  mkdir -p "$tmp/runtime"; rm -f "$tmp/runtime/updates.csv"
  (cd "$tmp/runtime" && "$tmp/build/index_prototype") 2>&1 | tee "$base.txt"
  [[ -s "$tmp/runtime/updates.csv" ]] || { echo 'Missing updates.csv' >&2; exit 1; }
  cp -- "$tmp/runtime/updates.csv" "$base.csv"
  python3 -B "$validator" "$base.csv" "$base.txt" "$v" "$i" "$d" "$q" "$build" "$rounds"
 done
done
echo 'All requested experiments passed five-run validation.'
