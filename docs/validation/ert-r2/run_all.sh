#!/bin/sh
# Run every R2 job in parallel (one container each), then the comparison.
#   PYTHON=/path/to/resipy-env/bin/python docs/validation/ert-r2/run_all.sh [output-dir]
# Takes about 30 minutes on an M1 under amd64 emulation; the field inversions are the slowest.
set -e
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
out="${1:-$here/output}"
py="${PYTHON:-python}"
export PATH="$here/bin:$PATH" PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$out"
pids=""
for job in fwd syn_dd824 syn_field214 field214 field214_default; do
  "$py" "$here/r2_comparison.py" "$job" --out "$out" > "$out/log_$job.txt" 2>&1 &
  pids="$pids $!"
done
status=0
for p in $pids; do wait "$p" || status=1; done
[ "$status" = 0 ] || echo "a job failed; see $out/log_*.txt" >&2
"$py" "$here/r2_comparison.py" compare --out "$out"
exit $status
