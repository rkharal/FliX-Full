#!/usr/bin/env bash
#------------------------------------------------------------
# Driver Script: Run Lookup/Rebuild tests across multiple workloads
#   - Iterates over (Xval, Yval) workloads and (NodeSize, Cacheline) pairs
#   - Exits on first failure with clear context
#------------------------------------------------------------
set -u -o pipefail

# Target script to run (ensure it's executable)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit 1
REBUILD_SCRIPT="$SCRIPT_DIR/Rebuild.sh"

# REBUILD_SCRIPT="./RunBaselines_Only.sh"


# Growth factor
GrowthVal=300

# Workloads: (Xval Yval)
declare -a WORKLOADS=(
  "25 90"
  "90 90"
)

# NodeSize / Cacheline pairs
declare -a NODE_CACHE_PAIRS=(
  "3 0"
)

# Optional preflight
if [[ ! -x "$REBUILD_SCRIPT" ]]; then
  echo "ERROR: Target script not executable: $REBUILD_SCRIPT" >&2
  exit 1
fi

for wl in "${WORKLOADS[@]}"; do
  set -- $wl
  Xval="$1"
  Yval="$2"

  echo "================================================================"
  echo "Workload: Xval=$Xval, Yval=$Yval, GrowthVal=$GrowthVal"
  echo "Script:   $REBUILD_SCRIPT"
  echo "================================================================"

  for pair in "${NODE_CACHE_PAIRS[@]}"; do
    set -- $pair
    NodeSizeInput="$1"
    CachelineSizeInput="$2"

    echo "------------------------------------------------------------"
    echo "Running REBUILD with:"
    echo "  NodeSizeInput=$NodeSizeInput, CachelineSizeInput=$CachelineSizeInput"
    echo "  Params: Xval=$Xval Yval=$Yval GrowthVal=$GrowthVal"
    echo "------------------------------------------------------------"
    rm -f data_cache/*.* || true
    "$REBUILD_SCRIPT" "$Xval" "$Yval" "$GrowthVal" "$NodeSizeInput" "$CachelineSizeInput" || {
      echo "ERROR: Run failed for Workload=($Xval,$Yval) NodeSize=$NodeSizeInput Cacheline=$CachelineSizeInput" >&2
      exit 1
    }
  done

  echo "✓ Completed workload Xval=$Xval, Yval=$Yval"
done

echo "All runs completed successfully."
