#!/usr/bin/env bash
# Run the validated round-5 all-hit NCU experiment for probe size 2^16 only.
#
# This launcher uses Updates_B25_P20_P26_NCU_R5_AllHits.sh with isolated
# configuration overrides. It profiles the same six algorithms, lookup kernels,
# launch selection, and NCU metrics without rerunning P20 or P26.
#
# Usage:
#   ./Updates_B25_P16_NCU_R5_AllHits.sh --full
#   ./Updates_B25_P16_NCU_R5_AllHits.sh --full --resume

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PROFILER="$SCRIPT_DIR/Updates_B25_P20_P26_NCU_R5_AllHits.sh"
P16_RESULT_ROOT="$SCRIPT_DIR/profile_updates_growth300_round5_all_hits_P16"

usage() {
  cat <<'EOF'
Usage: Updates_B25_P16_NCU_R5_AllHits.sh --full [--resume]

  --full    Profile all six variants at P16.
  --resume  Skip P16 configurations that already pass validation.

Results:
  profile_updates_growth300_round5_all_hits_P16/P16/<variant>/
EOF
}

if [[ ! -x "$BASE_PROFILER" ]]; then
  echo "Base profiler is not executable: $BASE_PROFILER" >&2
  exit 2
fi

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

for argument in "$@"; do
  case "$argument" in
    --full|--resume)
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --smoke)
      echo "This P16-only launcher uses --full; it contains only six configurations." >&2
      echo "Run: ./Updates_B25_P16_NCU_R5_AllHits.sh --full" >&2
      exit 2
      ;;
    *)
      echo "Unknown argument: $argument" >&2
      usage >&2
      exit 2
      ;;
  esac
done

export UPDATES_NCU_PROBE_SIZES_OVERRIDE="16"
export UPDATES_NCU_RESULT_ROOT_OVERRIDE="$P16_RESULT_ROOT"

exec "$BASE_PROFILER" "$@"
