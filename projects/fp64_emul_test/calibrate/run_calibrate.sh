#!/usr/bin/env bash
# run_calibrate.sh — run hipblaslt-bench over bench_calibrate.yaml with
# profiling enabled, producing emul_calib.csv for calibrate_efficiencies.py.
#
# Usage (from any directory):
#   ./run_calibrate.sh [path-to-hipblaslt-bench]
#
# The optional first argument overrides the default bench path below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${1:-/home/pghysels/rocm-libraries/projects/hipblaslt/build/release/clients/hipblaslt-bench}"
YAML="$SCRIPT_DIR/bench_calibrate.yaml"
CSV="$SCRIPT_DIR/emul_calib.csv"

# workspace, cold_iters, and iters are specified per-entry in bench_calibrate.yaml.

echo "==================================================================="
echo "  fp64 emulation kernel efficiency calibration"
echo "==================================================================="
echo "  Bench      : $BENCH"
echo "  YAML       : $YAML"
echo "  Output CSV : $CSV"
echo "  (workspace=200 GiB, cold_iters=10, iters=100 set in YAML)"
echo "-------------------------------------------------------------------"

if [[ ! -x "$BENCH" ]]; then
    echo "ERROR: hipblaslt-bench not found or not executable at: $BENCH" >&2
    exit 1
fi

# Remove stale CSV so the Python script sees only this run's data.
rm -f "$CSV"

echo "Starting benchmark run... (this will take several minutes)"
echo ""

HIPBLASLT_EMULATION_FUSED=off          \
HIPBLASLT_EMULATION_STRATEGY=eager     \
HIPBLASLT_EMULATION_PROFILE="$CSV"     \
HIPBLASLT_EMULATE_DOUBLE_PRECISION=1   \
    "$BENCH" --yaml "$YAML"

echo ""
echo "Done.  Profile written to: $CSV"
echo "Row count: $(( $(wc -l < "$CSV") - 1 )) data rows (excluding header)"
echo ""
echo "Run calibrate_efficiencies.py to compute updated efficiency constants:"
echo "  python3 $SCRIPT_DIR/calibrate_efficiencies.py $CSV"
