import pytest

from luna.benchmarks.alfworld.adapter import require_alfworld_data


def test_missing_alfworld_data_error_is_actionable(monkeypatch):
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)
    with pytest.raises(RuntimeError, match="alfworld-download"):
        require_alfworld_data()

