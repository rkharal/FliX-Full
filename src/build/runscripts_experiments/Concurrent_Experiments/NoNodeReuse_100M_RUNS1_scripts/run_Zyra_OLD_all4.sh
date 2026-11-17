#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/../run_concurrent_5_runs.sh" old "$HERE" "${1:---dry-run}"
