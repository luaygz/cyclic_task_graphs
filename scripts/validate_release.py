#!/usr/bin/env python3
"""Validate public-release structure, manifests, configs, and leak controls."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from luna.actor.graphs import load_graph
from luna.benchmarks.finance_agent.data import load_cases
from luna.config import load_run_config, load_seed_manifest

EXPECTED_SPLITS = {
    "textcraft_depth2_test.json": 203,
    "textcraft_depth3_test.json": 82,
    "textcraft_depth4_test.json": 11,
    "alfworld_eval_out_of_distribution.json": 134,
    "finance_agent_test.json": 35,
}
SKIP_PARTS = {".git", ".venv", "dist", "build", "__pycache__", ".pytest_cache", "outputs"}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".json", ".csv", ".cff", ".sh", ".txt", ""}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-release", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    validate_structure(errors)
    validate_manifests(errors)
    validate_configs(errors)
    scan_tree(errors)
    validate_owner_gates(errors if args.strict_release else warnings)
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print(f"release validation failed with {len(errors)} error(s)")
        return 1
    print("release validation passed")
    return 0


def validate_structure(errors: list[str]) -> None:
    required = [
        "README.md", "LICENSE", "CITATION.cff", "THIRD_PARTY_NOTICES.md", "pyproject.toml",
        ".env.example", ".gitignore", "compose.yaml", "configs/experiments/alfworld",
        "configs/experiments/textcraft", "configs/experiments/finance_agent", "configs/splits",
        "configs/graphs", "src/luna/actor", "src/luna/benchmarks/alfworld",
        "src/luna/benchmarks/textcraft", "src/luna/benchmarks/finance_agent", "src/luna/storage",
        "scripts/bootstrap_db.py", "scripts/verify_db.py", "scripts/import_graphs.py",
        "scripts/export_graph.py", "scripts/export_results.py", "scripts/run_smoke.sh",
        "results/paper_manifest.csv", "docs/SOURCE_MANIFEST.md",
    ]
    for relative in required:
        if not (ROOT / relative).exists():
            errors.append(f"missing required path: {relative}")
    forbidden = [".auth", ".env", "mongo-dump", "rebuttal_audit"]
    for name in forbidden:
        if (ROOT / name).exists():
            errors.append(f"forbidden private path exists: {name}")


def validate_manifests(errors: list[str]) -> None:
    for filename, expected in EXPECTED_SPLITS.items():
        path = ROOT / "configs" / "splits" / filename
        try:
            manifest = load_seed_manifest(path)
        except Exception as exc:
            errors.append(f"invalid seed manifest {filename}: {exc}")
            continue
        if len(manifest.seeds) != expected:
            errors.append(f"{filename}: expected {expected} seeds, got {len(manifest.seeds)}")
        if len(manifest.seeds) != len(set(manifest.seeds)):
            errors.append(f"{filename}: contains duplicate seeds")
    graph_paths = sorted((ROOT / "configs" / "graphs").glob("*.json"))
    if len(graph_paths) != 3:
        errors.append(f"expected 3 graph manifests, got {len(graph_paths)}")
    for path in graph_paths:
        try:
            graph = load_graph(path)
            if len(graph.sha256) != 64:
                errors.append(f"invalid graph hash: {path}")
        except Exception as exc:
            errors.append(f"invalid graph {path.name}: {exc}")
    try:
        cases = load_cases()
        if len(cases) != 50 or any(not case.rubric for case in cases):
            errors.append("Finance Agent CSV must contain 50 cases with non-empty rubrics")
    except Exception as exc:
        errors.append(f"Finance Agent CSV validation failed: {exc}")


def validate_configs(errors: list[str]) -> None:
    paths = sorted((ROOT / "configs" / "experiments").glob("*/*.yaml"))
    if len(paths) < 20:
        errors.append(f"expected at least 20 experiment configs, got {len(paths)}")
    for path in paths:
        try:
            config = load_run_config(path).resolved(ROOT)
            if config.mode == "gen-cyc":
                load_graph(config.graph_manifest)
        except Exception as exc:
            errors.append(f"invalid experiment config {path.relative_to(ROOT)}: {exc}")


def scan_tree(errors: list[str]) -> None:
    secret_patterns = [
        re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
        re.compile(r"(?i)(api[_-]?key|secret|password)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
        re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    ]
    personal_path = re.compile(r"(?:/home/[^/\s]+|/Users/[^/\s]+|[A-Za-z]:\\Users\\[^\\\s]+)")
    personal_identity = re.compile(r"(?i)\b(?:luay|samer)(?:[._ -][a-z]+)?\b|[\w.+-]+@(?!example\.org\b|invalid\b)[\w.-]+\.[A-Za-z]{2,}")
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.relative_to(ROOT) == Path("scripts/validate_release.py"):
            # This file necessarily contains the detector signatures themselves.
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(ROOT)
        for pattern in secret_patterns:
            if pattern.search(text):
                errors.append(f"possible secret in {relative}: {pattern.pattern}")
        if personal_path.search(text):
            errors.append(f"personal absolute path in {relative}")
        if personal_identity.search(text):
            errors.append(f"personal name or email in {relative}")


def validate_owner_gates(messages: list[str]) -> None:
    if "NOT YET SELECTED" in (ROOT / "LICENSE").read_text(encoding="utf-8"):
        messages.append("project license has not been selected")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    if "REPLACE WITH" in citation or "REPLACE_WITH" in citation:
        messages.append("CITATION.cff still contains owner-controlled placeholders")
    manifest = (ROOT / "results" / "paper_manifest.csv").read_text(encoding="utf-8")
    if "REPLACE_WITH_APPROVED_RUN_ID" in manifest:
        messages.append("paper result run IDs have not been approved")


if __name__ == "__main__":
    raise SystemExit(main())

