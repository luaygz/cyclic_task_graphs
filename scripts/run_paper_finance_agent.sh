#!/usr/bin/env bash
set -euo pipefail

for config in react spec_cyc gen_cyc depdag_retry; do
  python -m luna.benchmarks.cli run \
    --config "configs/experiments/finance_agent/${config}.yaml" \
    --dry-run
done

