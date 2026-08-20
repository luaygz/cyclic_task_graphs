import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_dry_run_creates_nothing(tmp_path):
    output = tmp_path / "must-not-exist"
    command = [
        sys.executable,
        "-m",
        "luna.benchmarks.cli",
        "run",
        "--config",
        "configs/experiments/textcraft/depth2.yaml",
        "--seeds",
        "0",
        "--output-dir",
        str(output),
        "--dry-run",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    assert '"seed_count": 1' in result.stdout
    assert '"graph_hash":' in result.stdout
    assert not output.exists()

