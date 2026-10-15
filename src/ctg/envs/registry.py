"""Benchmark lookup. Imports are lazy so optional dependencies (alfworld) load only when used."""

from __future__ import annotations

from ctg.config import BenchmarkName
from ctg.envs.base import Benchmark


def get_benchmark(name: BenchmarkName) -> Benchmark:
    match name:
        case BenchmarkName.TEXTCRAFT:
            from ctg.envs.textcraft.benchmark import TextCraftBenchmark

            return TextCraftBenchmark()
        case BenchmarkName.ALFWORLD:
            from ctg.envs.alfworld.benchmark import ALFWorldBenchmark

            return ALFWorldBenchmark()
        case BenchmarkName.FINANCE_AGENT:
            from ctg.envs.finance.benchmark import FinanceAgentBenchmark

            return FinanceAgentBenchmark()
