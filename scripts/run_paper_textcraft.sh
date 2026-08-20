#!/usr/bin/env bash
set -euo pipefail

for depth in 2 3 4; do
  for suffix in _react _spec_cyc "" _depdag_retry; do
    python -m luna.benchmarks.cli run \
      --config "configs/experiments/textcraft/depth${depth}${suffix}.yaml" \
      --dry-run
  done
done

