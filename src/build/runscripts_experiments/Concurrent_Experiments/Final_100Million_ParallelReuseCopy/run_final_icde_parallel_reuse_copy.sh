#!/usr/bin/env bash
# The historical parallel-copy kernel is not present in the active source.
set -euo pipefail
if [[ "${1:---dry-run}" == --dry-run ]]; then
  echo 'Historical parallel-copy experiment: four configurations, 100M batch, TOTALRUNS=5.'
  echo 'The active source has no FLIX_MIXED_PARALLEL_REUSE_COPY implementation; this experiment cannot be reproduced from it.'
  exit 0
fi
echo 'Cannot run: the historical parallel-copy implementation is absent from the active source.' >&2
exit 1
