#!/bin/bash
# Run the EasySteer validation suites: one pytest process per GPU module
# (vLLM engines do not reliably release GPU state within a process), CPU
# units in a single process.
#
# Usage: GPU_ID=2 ./run_suites.sh [dense|moe|cpu|all]
# Env: GPU_ID, STEER_TEST_MODEL/VECTOR/MOE_MODEL/QWEN3 (see conftest.py)

set -u
cd "$(dirname "$0")"
GROUP="${1:-all}"

CPU_SUITES=(cpu)
DENSE_SUITES=(
  e2e/test_vanilla_parity.py
  e2e/test_apply_semantics.py
  e2e/test_routing.py
  e2e/test_prefix_cache.py
  e2e/test_server_steering.py
  e2e/test_golden_sentiment.py
  e2e/test_require_preload.py
  e2e/test_piecewise.py
  e2e/test_fullgraph.py
  e2e/test_capture.py
  e2e/test_capture_unified.py
  e2e/test_trigger_positions.py
  e2e/test_openai_server.py
  e2e/test_capture_chunked.py
  e2e/test_payload_steering.py
)
MOE_SUITES=(
  moe/test_moe.py
  moe/test_moe_fullgraph.py
  moe/test_moe_compiled.py
  moe/test_steermoe.py
  moe/test_qwen3_smoke.py
)

case "$GROUP" in
  cpu)   SUITES=("${CPU_SUITES[@]}") ;;
  dense) SUITES=("${DENSE_SUITES[@]}") ;;
  moe)   SUITES=("${MOE_SUITES[@]}") ;;
  all)   SUITES=("${CPU_SUITES[@]}" "${DENSE_SUITES[@]}" "${MOE_SUITES[@]}") ;;
  *) echo "unknown group: $GROUP" >&2; exit 2 ;;
esac

FAILED=()
for suite in "${SUITES[@]}"; do
  echo "=== $suite ==="
  python -m pytest -q "$suite"
  status=$?
  if [ $status -ne 0 ]; then
    FAILED+=("$suite")
  fi
  sleep 10  # let the previous engine's GPU memory drain before the next boot
done

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "OVERALL: PASS (${#SUITES[@]} suites)"
else
  echo "OVERALL: FAIL — ${FAILED[*]}"
  exit 1
fi
