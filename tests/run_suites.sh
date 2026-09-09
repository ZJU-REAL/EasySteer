#!/bin/bash
# Run isolated GPU engine groups; only identical named profiles share an
# engine. CPU units run in a single process.
#
# Usage: GPU_ID=2 ./tests/run_suites.sh --help
# Env: see tests/README.md; no models are selected or downloaded by this script.

set -uo pipefail
GROUP="${1:-all}"
TEST_DIR="$(cd "$(dirname "$0")" && pwd)" || exit 2

if [ "$GROUP" = --help ] || [ "$GROUP" = -h ]; then
  echo "Usage: GPU_ID=0 $0 [cpu|kernels|baseline|dense|moe|moe-core|extended|golden|all]"
  echo "See tests/README.md for required model variables and optional results."
  exit 0
fi

CPU_SUITES=(cpu)
KERNEL_SUITES=(kernels/test_graph_additive.py)
# Small migration baseline: use the existing behavioral tests and avoid
# hardware-specific goldens, the 51-scale sweep, and extra MoE checkpoints.
BASELINE_SUITES=(
  cpu
  "${KERNEL_SUITES[@]}"
  e2e/test_vanilla_parity.py
  e2e/test_apply_semantics.py
  e2e/test_routing.py::TestMultiVector
  e2e/test_prefix_cache.py
  e2e/test_piecewise.py
  e2e/test_trigger_positions.py
  "e2e/test_capture_unified.py e2e/test_server_steering.py"
)
DENSE_SUITES=(
  e2e/test_vanilla_parity.py
  e2e/test_apply_semantics.py
  e2e/test_routing.py
  e2e/test_prefix_cache.py
  e2e/test_require_preload.py
  e2e/test_piecewise.py
  e2e/test_fullgraph.py
  e2e/test_fullgraph_large_capacity.py
  e2e/test_capacity_backpressure.py
  e2e/test_capacity_backpressure_fullgraph.py
  e2e/test_capture.py
  "e2e/test_capture_unified.py e2e/test_server_steering.py"
  e2e/test_trigger_positions.py
  e2e/test_openai_server.py
  e2e/test_capture_chunked.py
  e2e/test_payload_steering.py
)
MOE_CORE_SUITES=(
  "moe/test_moe.py moe/test_steermoe.py"
  moe/test_moe_fullgraph.py
  moe/test_moe_compiled.py
)
MOE_SUITES=(
  "${MOE_CORE_SUITES[@]}"
  moe/test_qwen3_smoke.py
)
EXTENDED_SUITES=(e2e/test_routing.py::TestScaleSweep)
GOLDEN_SUITES=(e2e/test_golden_sentiment.py)

case "$GROUP" in
  cpu)   SUITES=("${CPU_SUITES[@]}") ;;
  kernels) SUITES=("${KERNEL_SUITES[@]}") ;;
  baseline) SUITES=("${BASELINE_SUITES[@]}") ;;
  dense) SUITES=("${DENSE_SUITES[@]}") ;;
  moe)   SUITES=("${MOE_SUITES[@]}") ;;
  moe-core) SUITES=("${MOE_CORE_SUITES[@]}") ;;
  extended) SUITES=("${EXTENDED_SUITES[@]}") ;;
  golden) SUITES=("${GOLDEN_SUITES[@]}") ;;
  all)   SUITES=("${CPU_SUITES[@]}" "${KERNEL_SUITES[@]}" "${DENSE_SUITES[@]}" "${MOE_SUITES[@]}") ;;
  *) echo "unknown group: $GROUP" >&2; exit 2 ;;
esac

# Validate before any test starts. CPU-only runs need no model variables.
MODEL_VARS=()
case "$GROUP" in
  baseline|dense|extended|golden) MODEL_VARS=(STEER_TEST_MODEL) ;;
  moe-core) MODEL_VARS=(STEER_TEST_MOE_MODEL) ;;
  moe) MODEL_VARS=(STEER_TEST_MOE_MODEL STEER_TEST_QWEN3) ;;
  all) MODEL_VARS=(STEER_TEST_MODEL STEER_TEST_MOE_MODEL STEER_TEST_QWEN3) ;;
esac
for name in "${MODEL_VARS[@]}"; do
  if [ -z "${!name:-}" ]; then
    echo "Set $name to the test model's local directory (see tests/README.md)." >&2
    exit 2
  fi
  # Resolve existing relative directories against the caller's cwd before
  # changing to tests/. An explicitly supplied Hugging Face ID is passed on.
  if [ -d "${!name}" ]; then
    model_path="$(cd "${!name}" && pwd)" || exit 2
    export "$name=$model_path"
  fi
done

STEER_TEST_VECTOR="${STEER_TEST_VECTOR:-$TEST_DIR/../vectors/happy_diffmean.gguf}"
case "$GROUP" in
  baseline|dense|extended|golden|all)
    if [ ! -f "$STEER_TEST_VECTOR" ]; then
      echo "Steering vector not found: $STEER_TEST_VECTOR" >&2
      exit 2
    fi
    STEER_TEST_VECTOR="$(cd "$(dirname "$STEER_TEST_VECTOR")" && pwd)/$(basename "$STEER_TEST_VECTOR")"
    ;;
esac
export STEER_TEST_VECTOR

if [ "$GROUP" = golden ]; then
  if [ ! -f "${STEER_TEST_GOLDEN:-}" ]; then
    echo "Set STEER_TEST_GOLDEN to a recorded JSON file (see tests/README.md)." >&2
    exit 2
  fi
  STEER_TEST_GOLDEN="$(cd "$(dirname "$STEER_TEST_GOLDEN")" && pwd)/$(basename "$STEER_TEST_GOLDEN")"
  export STEER_TEST_GOLDEN
fi

TEST_PYTHON="$(command -v "${STEER_TEST_PYTHON:-python}")" || {
  echo "Python interpreter not found; set STEER_TEST_PYTHON to its executable." >&2
  exit 2
}
if [[ "$TEST_PYTHON" != /* ]]; then
  TEST_PYTHON="$PWD/$TEST_PYTHON"
fi
GPU_PAUSE="${STEER_TEST_GPU_PAUSE:-0}"
if [[ ! "$GPU_PAUSE" =~ ^[0-9]+$ ]]; then
  echo "STEER_TEST_GPU_PAUSE must be a non-negative integer (seconds)." >&2
  exit 2
fi

# Optional artifacts for comparing runs. Choose a fresh directory per run;
# refusing an existing one avoids overwriting an earlier baseline.
RESULTS_DIR="${STEER_TEST_RESULTS_DIR:-}"
if [ -n "$RESULTS_DIR" ]; then
  if [ -e "$RESULTS_DIR" ]; then
    echo "Results directory already exists; choose a fresh one: $RESULTS_DIR" >&2
    exit 2
  fi
  mkdir -p "$RESULTS_DIR" || exit 2
  RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)" || exit 2
  printf 'suite\texit_code\n' > "$RESULTS_DIR/summary.tsv" || exit 2
fi

cd "$TEST_DIR" || exit 2
FAILED=()
PREVIOUS_GPU=0
for suite in "${SUITES[@]}"; do
  if [ "$PREVIOUS_GPU" -eq 1 ] && [ "$GPU_PAUSE" -gt 0 ]; then
    sleep "$GPU_PAUSE"  # optional additional pause after bounded cleanup
  fi
  echo "=== $suite ==="
  read -r -a suite_paths <<< "$suite"
  pytest_args=(-m pytest -q --durations=0 -o junit_logging=all "${suite_paths[@]}")
  [ "$GROUP" = extended ] && pytest_args+=(--steer-extended)
  process_args=("$TEST_DIR/run_process.py")
  if [ -n "$RESULTS_DIR" ]; then
    artifact_name="${suite//\//_}"
    artifact_name="${artifact_name//:/_}"
    artifact_name="${artifact_name// /__}"
    export STEER_TEST_ARTIFACT_PREFIX="$RESULTS_DIR/$artifact_name"
    pytest_args+=("--junitxml=$RESULTS_DIR/$artifact_name.xml")
    process_args+=(--result "$RESULTS_DIR/$artifact_name.process.json")
    "$TEST_PYTHON" "${process_args[@]}" -- "$TEST_PYTHON" "${pytest_args[@]}" 2>&1 | tee "$RESULTS_DIR/$artifact_name.log"
  else
    unset STEER_TEST_ARTIFACT_PREFIX
    "$TEST_PYTHON" "${process_args[@]}" -- "$TEST_PYTHON" "${pytest_args[@]}"
  fi
  status=$?
  if [ -n "$RESULTS_DIR" ]; then
    printf '%s\t%s\n' "$suite" "$status" >> "$RESULTS_DIR/summary.tsv" || exit 2
  fi
  if [ $status -ne 0 ]; then
    FAILED+=("$suite")
  fi
  PREVIOUS_GPU=1
  [ "$suite" = cpu ] && PREVIOUS_GPU=0
done

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "OVERALL: PASS (${#SUITES[@]} suites)"
else
  echo "OVERALL: FAIL — ${FAILED[*]}"
  exit 1
fi
