import sys

from luna.benchmarks.registry import get_adapter_class


def test_finance_import_does_not_import_other_adapters():
    for name in list(sys.modules):
        if name.startswith("luna.benchmarks.alfworld") or name.startswith("luna.benchmarks.textcraft"):
            sys.modules.pop(name)
    assert get_adapter_class("finance_agent").__name__ == "FinanceAgentAdapter"
    assert not any(name.startswith("luna.benchmarks.alfworld") for name in sys.modules)
    assert not any(name.startswith("luna.benchmarks.textcraft") for name in sys.modules)

