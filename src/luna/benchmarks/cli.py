"""Command-line interface with a strictly side-effect-free dry-run path."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from luna.actor.graphs import load_graph
from luna.actor.model import OpenAIResponsesModel
from luna.benchmarks.runner import BenchmarkRunner
from luna.config import (
    ServiceSettings,
    load_run_config,
    load_seed_manifest,
    parse_seeds,
    repository_root,
    seed_list_hash,
)
from luna.storage.mongo import MongoStore, required_definition_aliases
from luna.storage.redis_bus import RedisBus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="luna-benchmark", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run or dry-run an experiment")
    run.add_argument("--benchmark", choices=("alfworld", "textcraft", "finance_agent"))
    run.add_argument("--mode", choices=("react", "spec-cyc", "gen-cyc", "depdag-retry"))
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--split")
    run.add_argument("--depth", type=int, choices=(2, 3, 4))
    run.add_argument("--seeds", help="comma-separated integers or ranges, e.g. 0,3-5")
    run.add_argument("--seeds-file", type=Path)
    run.add_argument("--max-cases", type=int)
    run.add_argument("--concurrency", type=int)
    run.add_argument("--graph-manifest", type=Path)
    run.add_argument("--resume", nargs="?", const="latest")
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--yes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return asyncio.run(run_command(args))
    raise AssertionError(args.command)


async def run_command(args: argparse.Namespace) -> int:
    root = repository_root()
    config_path = _resolve(root, args.config)
    overrides = {
        "benchmark": args.benchmark,
        "mode": args.mode,
        "split": args.split,
        "depth": args.depth,
        "max_cases": args.max_cases,
        "concurrency": args.concurrency,
        "graph_manifest": args.graph_manifest,
        "output_dir": args.output_dir,
    }
    config = load_run_config(config_path, overrides).resolved(root)
    seeds, manifest_hash = resolve_seeds(args, config, root)
    if config.max_cases is not None:
        seeds = seeds[: config.max_cases]
    graph = load_graph(config.graph_manifest) if config.graph_manifest else None
    if graph is not None and graph.benchmark != config.benchmark:
        raise ValueError(f"graph benchmark {graph.benchmark} does not match {config.benchmark}")
    summary = {
        "benchmark": config.benchmark,
        "split": config.split,
        "depth": config.depth,
        "method": config.mode,
        "config": config.model_dump(mode="json"),
        "seed_count": len(seeds),
        "seeds": seeds,
        "seed_hash": seed_list_hash(seeds),
        "seed_manifest_hash": manifest_hash,
        "graph_alias": graph.alias if graph else None,
        "graph_hash": graph.sha256 if graph else None,
        "database_name": ServiceSettings.from_env().mongodb_db,
        "output_path": str(config.output_dir),
        "api_cost_warning": "A non-dry run may call paid model and search APIs.",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    if not args.yes:
        answer = input("Proceed with this potentially paid benchmark run? [y/N] ")
        if answer.strip().lower() != "y":
            return 1
    preflight_local(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.storage_enabled:
        settings = ServiceSettings.from_env()
        store = MongoStore(settings.mongodb_uri, settings.mongodb_db)
        bus = RedisBus(settings.redis_url)
        preflight_services(store, bus)
        if graph is not None:
            stored = store.get_graph(graph.alias, graph.version)
            if stored is None:
                raise RuntimeError(
                    f"graph {graph.alias}@{graph.version} is not imported; run scripts/import_graphs.py"
                )
            if stored.sha256 != graph.sha256:
                raise RuntimeError("stored graph hash does not match the selected local manifest")
            graph = stored
    else:
        store = None
        bus = None
    model = OpenAIResponsesModel(
        model=config.model.executor_model,
        reasoning_effort=config.model.reasoning_effort,
        judge_model=config.model.judge_model,
    )
    record = await BenchmarkRunner(config, model, store=store, bus=bus).run(
        seeds, generalized_graph=graph, resume=args.resume
    )
    print(json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


def resolve_seeds(args: argparse.Namespace, config, root: Path) -> tuple[list[int], str | None]:
    if args.seeds and args.seeds_file:
        raise ValueError("choose either --seeds or --seeds-file")
    if args.seeds:
        return parse_seeds(args.seeds), None
    path = _resolve(root, args.seeds_file) if args.seeds_file else default_seed_manifest(root, config)
    manifest = load_seed_manifest(path)
    if (manifest.benchmark, manifest.split, manifest.depth) != (
        config.benchmark,
        config.split,
        config.depth,
    ):
        raise ValueError("seed manifest benchmark/split/depth does not match the run config")
    return manifest.seeds, manifest.sha256


def default_seed_manifest(root: Path, config) -> Path:
    if config.benchmark == "textcraft":
        name = f"textcraft_depth{config.depth}_{config.split}.json"
    else:
        name = f"{config.benchmark}_{config.split}.json"
    return root / "configs" / "splits" / name


def preflight_local(config) -> None:
    if config.model.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for non-dry live runs")
    if config.benchmark == "alfworld":
        from luna.benchmarks.alfworld.adapter import require_alfworld_data

        require_alfworld_data()
    if config.benchmark == "finance_agent":
        missing = [name for name in ("EXA_API_KEY", "SEC_EDGAR_API_KEY", "SEC_USER_AGENT") if not os.getenv(name)]
        if missing:
            raise RuntimeError(f"Finance Agent live run is missing: {', '.join(missing)}")
    output_parent = config.output_dir
    while not output_parent.exists() and output_parent != output_parent.parent:
        output_parent = output_parent.parent
    if not output_parent.is_dir() or not os.access(output_parent, os.W_OK | os.X_OK):
        raise RuntimeError(f"output path is not writable: {config.output_dir}")


def preflight_services(store: MongoStore, bus: RedisBus) -> None:
    store.ping()
    bus.ping()
    store.ensure_indexes()
    missing = set(required_definition_aliases()) - store.definition_aliases()
    if missing:
        raise RuntimeError(f"database bootstrap is incomplete; missing aliases: {sorted(missing)}")


def _resolve(root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (root / path).resolve()


if __name__ == "__main__":
    sys.exit(main())

