from pathlib import Path

from luna.actor.graphs import load_graph
from luna.config import load_seed_manifest


ROOT = Path(__file__).resolve().parents[2]


def test_paper_split_counts():
    expected = {
        "textcraft_depth2_test.json": 203,
        "textcraft_depth3_test.json": 82,
        "textcraft_depth4_test.json": 11,
        "alfworld_eval_out_of_distribution.json": 134,
        "finance_agent_test.json": 35,
    }
    for filename, count in expected.items():
        manifest = load_seed_manifest(ROOT / "configs" / "splits" / filename)
        assert len(manifest.seeds) == count
        assert len(manifest.seeds) == len(set(manifest.seeds))
        assert len(manifest.sha256) == 64


def test_graph_manifests_are_normalized_and_valid():
    paths = sorted((ROOT / "configs" / "graphs").glob("*.json"))
    assert len(paths) == 3
    hashes = {load_graph(path).sha256 for path in paths}
    assert len(hashes) == 3

