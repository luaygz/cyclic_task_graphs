#!/usr/bin/env bash
set -euo pipefail

python -m luna.benchmarks.cli run \
  --benchmark textcraft \
  --mode gen-cyc \
  --config configs/experiments/textcraft/depth2.yaml \
  --seeds 0 \
  --max-cases 1 \
  --concurrency 1 \
  --output-dir outputs/smoke/textcraft \
  --dry-run

