"""Run one experiment cell: every (repeat, instance) episode, recorded as JSON and resumable.

Gen-Cyc plans its graph once per run and reuses it for every instance and repeat (Sec. 3.2.2). Spec-Cyc,
DepDAG and DepDAG-NoRetry plan per episode; ReAct has no graph. Only the planner sees n-shot summaries
(Sec. 4.1.6).
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import importlib.util
import logging
import os
import random
import subprocess
import traceback
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, overload

from ctg.config import (
    REPO_ROOT,
    BenchmarkName,
    Budgets,
    Controller,
    ExecutorMode,
    ExperimentConfig,
    ModelSpec,
    Provider,
    Role,
    load_model_presets,
    resolve_models,
)
from ctg.control.react import run_react_episode
from ctg.control.workflow import EpisodeResult, run_graph_episode
from ctg.envs.base import Benchmark, Environment, ExecutorProfile, InstanceRef
from ctg.envs.registry import get_benchmark
from ctg.errors import BackendUnavailableError
from ctg.graphs.model import CyclicGraph, DependencyGraph
from ctg.graphs.planner import PlannedGraph, plan_cyclic, plan_dependency
from ctg.llm.base import LLM, CallSink, LLMOutputError, SpendCapExceededError
from ctg.llm.client import create_llm
from ctg.runs.nshot import load_nshot
from ctg.runs.records import (
    EndReason,
    EpisodeRecord,
    Outcome,
    PreviousAttempt,
    RoleModel,
    RunConditions,
    RunManifest,
)
from ctg.runs.splits import SPLITS_DIR, select_instances
from ctg.runs.store import RunInUseError, RunStore, SchemaVersionError, StoredGraph

log = logging.getLogger(__name__)

PROVIDER_KEYS: dict[Provider, str] = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.OPENROUTER: "OPENROUTER_API_KEY",
}

ALFWORLD_MODULES = ("alfworld", "textworld", "fast_downward")
"""Top-level modules of the `alfworld` extra that the ALFWorld game processes import."""

# Dry-run cost estimate. Deliberately rough: an order of magnitude to budget a run, not a prediction.
EPISODE_TOKENS: dict[str, tuple[int, int]] = {
    "textcraft_depth2": (85_000, 95_000),
    "textcraft_depth3": (200_000, 255_000),
    "textcraft_depth4": (265_000, 690_000),
    "alfworld": (45_000, 230_000),
    "finance_agent": (85_000, 150_000),
}
"""(ReAct, graph controller) average total tokens per episode: the all-win averages of the paper's Table 12
(gpt-4o-mini), rounded up. Failed episodes usually run to the budget and use more."""

_SHARES_WITH_PLANNING = {Role.PLANNER: 0.10, Role.EXECUTOR: 0.60, Role.ROUTER: 0.30}
ROLE_SHARES: dict[Controller, dict[Role, float]] = {
    Controller.REACT: {Role.EXECUTOR: 1.0},
    Controller.DEPDAG: _SHARES_WITH_PLANNING,
    Controller.DEPDAG_NORETRY: _SHARES_WITH_PLANNING,
    Controller.SPEC_CYC: _SHARES_WITH_PLANNING,
    Controller.GEN_CYC: {Role.EXECUTOR: 0.65, Role.ROUTER: 0.35},
}
"""Assumed split of an episode's tokens across roles (the paper reports totals only)."""

OUTPUT_TOKEN_SHARE = 0.05
"""Assumed share of output tokens; agent prompts resend long histories, so input dominates."""

GEN_CYC_PLANNING_TOKENS = 20_000
FINANCE_EXTRA_TOKENS: dict[Role, int] = {Role.JUDGE: 5_000, Role.TOOL: 30_000}
"""Finance-Agent per-episode tokens for the answer judge and the document summaries of retrieval tools."""


class RunSetupError(RuntimeError):
    """The experiment cannot start. The message lists every problem found."""


@dataclass(frozen=True)
class EpisodeKey:
    repeat: int
    instance: InstanceRef


@dataclass(frozen=True)
class CostEstimate:
    episodes: int
    usd: float
    unpriced: tuple[str, ...]
    """Roles left out of `usd` because their model preset has no prices (e.g. OpenRouter models)."""


@dataclass(frozen=True)
class DryRunReport:
    name: str
    run_dir: Path
    total: int
    completed: int
    scheduled: int
    estimate: CostEstimate | None
    notes: tuple[str, ...]
    problems: tuple[str, ...]


@dataclass
class RunReport:
    name: str
    run_dir: Path
    total: int
    completed_before: int
    scheduled: int
    finished: int = 0
    won: int = 0
    output_failures: int = 0
    """Finished episodes whose model output stayed unusable (losses, complete on resume)."""
    errors: int = 0
    not_started: int = 0
    spent_usd: float = 0.0
    """Cost of the calls made in this invocation (as priced by the client)."""
    stop_reason: str | None = None
    halts_sweep: bool = False
    """The run stopped on a spend cap or an unavailable backend. The CLI then starts no further configs;
    other stops (a failed Gen-Cyc planning) end only this run."""


@dataclass
class _Preparation:
    """Everything resolved before any model call. `problems` must be empty to run."""

    config: ExperimentConfig
    store: RunStore
    benchmark: Benchmark
    budgets: Budgets
    models: dict[Role, ModelSpec] = field(default_factory=dict[Role, ModelSpec])
    instances: list[InstanceRef] = field(default_factory=list[InstanceRef])
    nshot: tuple[str, ...] = ()
    nshot_sha256: str | None = None
    gen_cyc: StoredGraph | None = None
    completed: set[tuple[int, str]] = field(default_factory=set[tuple[int, str]])
    problems: list[str] = field(default_factory=list[str])
    notes: list[str] = field(default_factory=list[str])

    @property
    def total(self) -> int:
        return self.config.repeats * len(self.instances)

    def pending(self, limit: int | None) -> list[EpisodeKey]:
        """Episodes still to run, repeat-major in frozen-split order, truncated to `limit`."""
        keys = [
            EpisodeKey(repeat, instance)
            for repeat in range(self.config.repeats)
            for instance in self.instances
            if (repeat, instance.id) not in self.completed
        ]
        return keys if limit is None else keys[:limit]

    def role_models(self) -> dict[Role, RoleModel]:
        return {role: RoleModel.of(role, self.models[role]) for role in _used_roles(self.config)}

    def conditions(self) -> RunConditions:
        return RunConditions(nshot_sha256=self.nshot_sha256, models=self.role_models(), budgets=self.budgets)


@overload
async def run_experiment(
    config: ExperimentConfig,
    runs_dir: Path,
    *,
    dry_run: Literal[False] = False,
    limit: int | None = None,
    max_cost_usd: float | None = None,
    splits_dir: Path = SPLITS_DIR,
) -> RunReport: ...
@overload
async def run_experiment(
    config: ExperimentConfig,
    runs_dir: Path,
    *,
    dry_run: Literal[True],
    limit: int | None = None,
    max_cost_usd: float | None = None,
    splits_dir: Path = SPLITS_DIR,
) -> DryRunReport: ...


async def run_experiment(
    config: ExperimentConfig,
    runs_dir: Path,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    max_cost_usd: float | None = None,
    splits_dir: Path = SPLITS_DIR,
) -> RunReport | DryRunReport:
    """Run (or with `dry_run`, only validate and cost) the pending episodes of `config`.

    Completed episodes in `runs_dir/<name>` are skipped; errored ones are rerun. A run holds the run
    directory's lock throughout, so a second process on the same directory gets a `RunSetupError`.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    run_dir = runs_dir / config.name
    prep = _prepare(config, run_dir, splits_dir)
    if dry_run:
        return _dry_run_report(prep, prep.pending(limit))
    if prep.problems:
        raise RunSetupError("\n".join(prep.problems))
    try:
        with prep.store.lock():
            return await _run_locked(config, run_dir, splits_dir, limit, max_cost_usd)
    except RunInUseError as exc:
        raise RunSetupError(str(exc)) from exc


async def _run_locked(
    config: ExperimentConfig, run_dir: Path, splits_dir: Path, limit: int | None, max_cost_usd: float | None
) -> RunReport:
    """Prepare again under the lock (another process may have changed the directory since the first check),
    create the manifest of a new run, and execute."""
    prep = _prepare(config, run_dir, splits_dir)
    if prep.problems:
        raise RunSetupError("\n".join(prep.problems))
    manifest = prep.store.manifest() or _create_manifest(prep)
    llm = create_llm(run=manifest.run_id, run_cap_usd=max_cost_usd)
    try:
        return await _execute(prep, prep.pending(limit), llm)
    finally:
        await llm.aclose()


def _create_manifest(prep: _Preparation) -> RunManifest:
    config = prep.config
    manifest = RunManifest(
        name=config.name,
        run_id=f"{config.name}/{uuid.uuid4().hex}",
        config=config,
        conditions=prep.conditions(),
        ctg_version=importlib.metadata.version("ctg"),
        git_commit=_git_commit(),
        created_at=_now(),
    )
    prep.store.write_manifest(manifest)
    return manifest


def _prepare(config: ExperimentConfig, run_dir: Path, splits_dir: Path) -> _Preparation:
    benchmark = get_benchmark(config.benchmark)
    prep = _Preparation(
        config=config,
        store=RunStore(run_dir),
        benchmark=benchmark,
        budgets=config.budgets or benchmark.budgets(config),
    )
    problems = prep.problems
    try:
        prep.models = resolve_models(config.models, load_model_presets())
    except KeyError as exc:
        problems.append(str(exc.args[0]))
    try:
        prep.instances = select_instances(config, splits_dir)
    except (FileNotFoundError, ValueError) as exc:
        problems.append(str(exc))
    if config.nshot is not None:
        path = config.nshot if config.nshot.is_absolute() else REPO_ROOT / config.nshot
        try:
            prep.nshot = tuple(summary.text for summary in load_nshot(path, config.benchmark).summaries)
            prep.nshot_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            prep.notes.append(f"n-shot: {len(prep.nshot)} summaries from {path}")
        except (FileNotFoundError, ValueError) as exc:
            problems.append(f"n-shot: {exc}")
    try:
        mismatch = _manifest_mismatch(prep)
        if mismatch is not None:
            problems.append(mismatch)
        prep.completed = prep.store.completed()
        if config.controller is Controller.GEN_CYC:
            prep.gen_cyc = prep.store.gen_cyc_graph()
    except SchemaVersionError as exc:
        problems.append(str(exc))
    except ValueError as exc:
        problems.append(f"{run_dir} holds unreadable records: {exc}")
    if config.controller is Controller.GEN_CYC:
        prep.notes.append(
            f"Gen-Cyc graph: reusing {prep.store.graph_path}"
            if prep.gen_cyc is not None
            else "Gen-Cyc graph: planned once at run start and saved to graph.json"
        )
    if prep.models:
        problems.extend(f"missing environment variable {name}" for name in _missing_env(config, prep.models))
    if config.benchmark is BenchmarkName.ALFWORLD:
        from ctg.envs.alfworld.games import games_dir

        missing = [module for module in ALFWORLD_MODULES if importlib.util.find_spec(module) is None]
        if missing:
            problems.append(
                f"ALFWorld needs the alfworld extra ({', '.join(missing)} not installed); "
                "run `uv sync --extra alfworld`"
            )
        if not games_dir().is_dir():
            problems.append(f"ALFWorld games not found at {games_dir()}; run `alfworld-download`")
    if config.benchmark is BenchmarkName.TEXTCRAFT:
        from ctg.envs.textcraft.recipes import load_recipe_book

        try:
            load_recipe_book()
        except (FileNotFoundError, ValueError) as exc:
            problems.append(f"TextCraft recipes: {exc}")
    return prep


def _manifest_mismatch(prep: _Preparation) -> str | None:
    """What changed since the run directory was created, or None if the run may resume.

    Conditions this invocation could not resolve (unknown preset, unreadable n-shot file) are skipped; they
    are reported as problems of their own.
    """
    store, config = prep.store, prep.config
    manifest = store.manifest()
    if manifest is None:
        return None
    stored = manifest.config.model_dump(exclude={"concurrency"})
    current = config.model_dump(exclude={"concurrency"})
    changed = sorted(key for key in current if stored.get(key) != current[key])
    conditions = manifest.conditions
    if (
        "nshot" not in changed
        and prep.nshot_sha256 is not None
        and prep.nshot_sha256 != conditions.nshot_sha256
    ):
        changed.append(f"contents of the n-shot file {config.nshot}")
    if prep.models and "models" not in changed:
        current_models = prep.role_models()
        changed.extend(
            f"{role.value} model settings (preset {getattr(config.models, role.value)!r})"
            for role, model in current_models.items()
            if role in conditions.models and conditions.models[role] != model
        )
    if "budgets" not in changed and conditions.budgets != prep.budgets:
        changed.append("budgets")
    if not changed:
        return None
    return (
        f"{store.root} was created with a different configuration (changed: {', '.join(changed)}); "
        "use a new experiment name or remove the run directory"
    )


def _used_roles(config: ExperimentConfig) -> list[Role]:
    roles = (
        [Role.EXECUTOR]
        if config.controller is Controller.REACT
        else [Role.PLANNER, Role.EXECUTOR, Role.ROUTER]
    )
    if config.benchmark is BenchmarkName.FINANCE_AGENT:
        roles += [Role.JUDGE, Role.TOOL]
    return roles


def _missing_env(config: ExperimentConfig, models: dict[Role, ModelSpec]) -> list[str]:
    providers = {models[role].provider for role in _used_roles(config)}
    names = [PROVIDER_KEYS[provider] for provider in Provider if provider in providers]
    if config.benchmark is BenchmarkName.FINANCE_AGENT:
        from ctg.envs.finance.benchmark import required_env_vars

        names += required_env_vars(config.web_search)
    return [name for name in names if not os.environ.get(name, "").strip()]


def estimate_cost(
    config: ExperimentConfig, models: dict[Role, ModelSpec], episodes: int, plan_gen_cyc: bool
) -> CostEstimate:
    """Rough USD cost of `episodes` episodes from the documented token constants and preset prices."""
    react_tokens, graph_tokens = EPISODE_TOKENS[_setting(config)]
    per_episode = react_tokens if config.controller is Controller.REACT else graph_tokens
    tokens = {role: per_episode * share * episodes for role, share in ROLE_SHARES[config.controller].items()}
    if plan_gen_cyc:
        tokens[Role.PLANNER] = tokens.get(Role.PLANNER, 0.0) + GEN_CYC_PLANNING_TOKENS
    if config.benchmark is BenchmarkName.FINANCE_AGENT:
        for role, count in FINANCE_EXTRA_TOKENS.items():
            tokens[role] = float(count * episodes)
    usd = 0.0
    unpriced: list[str] = []
    for role, count in tokens.items():
        if count == 0:
            continue
        spec = models[role]
        if spec.input_usd_per_mtok is None or spec.output_usd_per_mtok is None:
            unpriced.append(f"{role.value} ({spec.name})")
            continue
        blended = (
            1 - OUTPUT_TOKEN_SHARE
        ) * spec.input_usd_per_mtok + OUTPUT_TOKEN_SHARE * spec.output_usd_per_mtok
        usd += count * blended / 1_000_000
    return CostEstimate(episodes=episodes, usd=usd, unpriced=tuple(unpriced))


def _setting(config: ExperimentConfig) -> str:
    match config.benchmark:
        case BenchmarkName.TEXTCRAFT:
            return f"textcraft_depth{config.textcraft_depth}"
        case BenchmarkName.ALFWORLD:
            return "alfworld"
        case BenchmarkName.FINANCE_AGENT:
            return "finance_agent"


def _dry_run_report(prep: _Preparation, pending: list[EpisodeKey]) -> DryRunReport:
    """Counts and the estimate need the instance list; without one the problems say why."""
    config = prep.config
    notes = [
        f"budgets: C_g={prep.budgets.global_tool_calls}, C_l={prep.budgets.local_tool_calls}",
        *prep.notes,
    ]
    if prep.instances:
        notes.insert(
            0, f"{config.variant} {config.split}: {len(prep.instances)} instances × {config.repeats} repeats"
        )
    estimate = None
    if prep.models:
        roles = ", ".join(
            f"{role.value}={getattr(config.models, role.value)}" for role in _used_roles(config)
        )
        notes.insert(0, f"models: {roles}")
        if prep.instances:
            plan_gen_cyc = config.controller is Controller.GEN_CYC and prep.gen_cyc is None and bool(pending)
            estimate = estimate_cost(config, prep.models, len(pending), plan_gen_cyc)
    return DryRunReport(
        name=config.name,
        run_dir=prep.store.root,
        total=prep.total,
        completed=prep.total - len(prep.pending(None)),
        scheduled=len(pending),
        estimate=estimate,
        notes=tuple(notes),
        problems=tuple(prep.problems),
    )


def episode_rng(seed: int, repeat: int, instance_id: str) -> random.Random:
    """Per-episode randomness (fault injection), independent of scheduling order and concurrency."""
    digest = hashlib.sha256(f"{seed}:{repeat}:{instance_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _now() -> datetime:
    return datetime.now(UTC)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


async def _execute(prep: _Preparation, pending: list[EpisodeKey], llm: LLM) -> RunReport:
    config, store = prep.config, prep.store
    report = RunReport(
        name=config.name,
        run_dir=store.root,
        total=prep.total,
        completed_before=prep.total - len(prep.pending(None)),
        scheduled=len(pending),
    )
    log.info(
        "%s: %d episodes to run (%d of %d already complete), concurrency %d",
        config.name,
        len(pending),
        report.completed_before,
        report.total,
        config.concurrency,
    )
    if not pending:
        return report
    roster = prep.benchmark.executor_roster(config.executors is ExecutorMode.SPECIALIST)
    gen_cyc = prep.gen_cyc
    if config.controller is Controller.GEN_CYC and gen_cyc is None:
        try:
            gen_cyc = await _plan_gen_cyc(prep, roster, llm, report)
        except Exception as exc:
            report.stop_reason = _planning_failure(exc)
            report.halts_sweep = isinstance(exc, SpendCapExceededError | BackendUnavailableError)
            report.not_started = len(pending)
            expected = report.halts_sweep or isinstance(exc, LLMOutputError)
            log.error("%s: %s; no episode started", config.name, report.stop_reason, exc_info=not expected)
            return report
    await _Episodes(prep, roster, llm, gen_cyc, report).run(pending)
    return report


async def _plan_gen_cyc(
    prep: _Preparation, roster: Sequence[ExecutorProfile], llm: LLM, report: RunReport
) -> StoredGraph:
    sink = CallSink()
    try:
        planned = await plan_cyclic(
            llm.scoped(sink), prep.models[Role.PLANNER], prep.benchmark, roster, None, prep.nshot
        )
    finally:
        report.spent_usd += sum(call.cost_usd for call in sink.calls)
    if not isinstance(planned.graph, CyclicGraph):
        raise TypeError(f"plan_cyclic returned a {type(planned.graph).__name__}")
    stored = StoredGraph(graph=planned.graph, notes=planned.notes, llm_calls=sink.calls, created_at=_now())
    prep.store.write_gen_cyc_graph(stored)
    log.info("%s: planned the Gen-Cyc graph (%d subtasks)", prep.config.name, len(planned.graph.subtasks))
    return stored


def _planning_failure(exc: Exception) -> str:
    match exc:
        case SpendCapExceededError():
            return f"spend cap reached while planning the Gen-Cyc graph: {exc}"
        case BackendUnavailableError():
            return f"backend unavailable while planning the Gen-Cyc graph: {exc}"
        case _:
            return f"planning the Gen-Cyc graph failed: {type(exc).__name__}: {exc}"


@dataclass
class _Attempt:
    """What one episode has reached so far; an ERROR record keeps all of it."""

    started: datetime
    previous: list[PreviousAttempt]
    """The stored ERROR attempts of this episode, oldest first, carried into the new record."""
    sink: CallSink = field(default_factory=CallSink)
    env: Environment | None = None
    objective: str = ""
    initial_state: dict[str, Any] = field(default_factory=dict[str, Any])
    graph: CyclicGraph | DependencyGraph | None = None
    notes: dict[str, Any] = field(default_factory=dict[str, Any])


class _Episodes:
    def __init__(
        self,
        prep: _Preparation,
        roster: Sequence[ExecutorProfile],
        llm: LLM,
        gen_cyc: StoredGraph | None,
        report: RunReport,
    ) -> None:
        self.prep = prep
        self.config = prep.config
        self.roster = roster
        self.llm = llm
        self.gen_cyc = gen_cyc
        self.report = report

    async def run(self, keys: Sequence[EpisodeKey]) -> None:
        semaphore = asyncio.Semaphore(self.config.concurrency)

        async def guarded(key: EpisodeKey) -> None:
            async with semaphore:
                if self.report.stop_reason is not None:
                    self.report.not_started += 1
                    return
                await self._episode(key)

        async with asyncio.TaskGroup() as group:
            for key in keys:
                group.create_task(guarded(key))

    async def _episode(self, key: EpisodeKey) -> None:
        attempt = _Attempt(started=_now(), previous=self._previous_attempts(key))
        try:
            record = await self._attempt(key, attempt)
            self.prep.store.write_episode(record)
            self._count(record, attempt.sink)
        finally:
            if attempt.env is not None:
                await self._close(attempt.env, key)

    def _previous_attempts(self, key: EpisodeKey) -> list[PreviousAttempt]:
        """The attempts kept in the stored ERROR record of `key` followed by that record's own."""
        stored = self.prep.store.episode(key.repeat, key.instance.id)
        if stored is None or stored.outcome.end_reason is not EndReason.ERROR:
            return []
        latest = PreviousAttempt(
            error=stored.error or "",
            llm_calls=stored.llm_calls,
            started_at=stored.started_at,
            finished_at=stored.finished_at,
        )
        return [*stored.previous_attempts, latest]

    async def _attempt(self, key: EpisodeKey, attempt: _Attempt) -> EpisodeRecord:
        """Play the episode and record whatever it reached.

        Model output that stays unusable ends the episode as an OUTPUT_FAILURE loss: during control the
        controller returns the trajectory so far with `failure` set, which is recorded as the error; during
        planning `LLMOutputError` is raised and the record has no graph. Any other exception becomes an ERROR
        record. A spend cap or a backend that cannot serve the run would fail every remaining episode, so
        either also stops the run.
        """
        llm = self.llm.scoped(attempt.sink)
        try:
            env = await self.prep.benchmark.make_env(key.instance, self.config, llm, self.prep.models)
            attempt.env = env
            attempt.objective = env.objective
            attempt.initial_state = env.initial_state()
            attempt.graph, attempt.notes = await self._plan(env, llm)
            result = await self._control(attempt.graph, env, llm, key)
            return self._record(
                key, attempt, result.outcome, result=result, details=env.details(), error=result.failure
            )
        except LLMOutputError:
            return self._failure(key, attempt, EndReason.OUTPUT_FAILURE)
        except Exception as exc:
            if isinstance(exc, SpendCapExceededError):
                self._stop(key, "spend cap reached", exc)
            elif isinstance(exc, BackendUnavailableError):
                self._stop(key, "backend unavailable", exc)
            return self._failure(key, attempt, EndReason.ERROR)

    def _failure(self, key: EpisodeKey, attempt: _Attempt, end_reason: EndReason) -> EpisodeRecord:
        """Record the exception being handled, with the steps the environment reached before it."""
        tool_calls = attempt.env.tool_calls if attempt.env is not None else 0
        outcome = Outcome(won=False, end_reason=end_reason, tool_calls=tool_calls, segments=0)
        details = attempt.env.details() if attempt.env is not None else None
        return self._record(key, attempt, outcome, details=details, error=traceback.format_exc())

    async def _plan(
        self, env: Environment, llm: LLM
    ) -> tuple[CyclicGraph | DependencyGraph | None, dict[str, Any]]:
        planner = self.prep.models[Role.PLANNER]
        planned: PlannedGraph
        match self.config.controller:
            case Controller.REACT:
                return None, {}
            case Controller.GEN_CYC:
                if self.gen_cyc is None:
                    raise RuntimeError("Gen-Cyc episode without a planned graph")
                return self.gen_cyc.graph, dict(self.gen_cyc.notes)
            case Controller.SPEC_CYC:
                planned = await plan_cyclic(
                    llm, planner, self.prep.benchmark, self.roster, env.objective, self.prep.nshot
                )
            case Controller.DEPDAG | Controller.DEPDAG_NORETRY:
                planned = await plan_dependency(
                    llm,
                    planner,
                    self.prep.benchmark,
                    self.roster,
                    env.objective,
                    retry=self.config.controller is Controller.DEPDAG,
                    nshot=self.prep.nshot,
                )
        return planned.graph, planned.notes

    async def _control(
        self, graph: CyclicGraph | DependencyGraph | None, env: Environment, llm: LLM, key: EpisodeKey
    ) -> EpisodeResult:
        if graph is None:
            return await run_react_episode(env, self.prep.benchmark, llm, self.prep.models, self.prep.budgets)
        rng = episode_rng(self.config.seed, key.repeat, key.instance.id)
        return await run_graph_episode(
            graph,
            env,
            self.prep.benchmark,
            llm,
            self.prep.models,
            self.prep.budgets,
            self.config.perturbation,
            rng,
        )

    def _record(
        self,
        key: EpisodeKey,
        attempt: _Attempt,
        outcome: Outcome,
        *,
        result: EpisodeResult | None = None,
        details: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> EpisodeRecord:
        config = self.config
        env = attempt.env
        return EpisodeRecord(
            run=config.name,
            benchmark=config.benchmark,
            variant=config.variant,
            instance_id=key.instance.id,
            instance=dict(key.instance.meta),
            objective=attempt.objective,
            repeat=key.repeat,
            controller=config.controller,
            executors=config.executors,
            perturbation=config.perturbation,
            graph=attempt.graph,
            planner_notes=attempt.notes,
            initial_state=attempt.initial_state,
            segments=result.segments if result is not None else [],
            react_steps=result.react_steps if result is not None else [],
            steps=list(env.steps) if env is not None else [],
            env_details=details or {},
            llm_calls=list(attempt.sink.calls),
            previous_attempts=attempt.previous,
            outcome=outcome,
            error=error,
            started_at=attempt.started,
            finished_at=_now(),
        )

    def _stop(self, key: EpisodeKey, cause: str, exc: Exception) -> None:
        """Start no further episode, here or in the configs after this one. The first cause is reported."""
        if self.report.stop_reason is not None:
            return
        self.report.stop_reason = f"{cause} during r{key.repeat}/{key.instance.id}: {exc}"
        self.report.halts_sweep = True
        log.error(
            "%s: %s. No new episodes will start; episodes in flight finish and are recorded.",
            self.config.name,
            self.report.stop_reason,
        )

    def _count(self, record: EpisodeRecord, sink: CallSink) -> None:
        report = self.report
        report.finished += 1
        report.won += record.outcome.won
        report.output_failures += record.outcome.end_reason is EndReason.OUTPUT_FAILURE
        report.errors += record.outcome.end_reason is EndReason.ERROR
        cost = sum(call.cost_usd for call in sink.calls)
        report.spent_usd += cost
        log.info(
            "[%d/%d] r%d %s: %s (k=%d, %d segments, $%.4f)",
            report.finished,
            report.scheduled,
            record.repeat,
            record.instance_id,
            record.outcome.end_reason.value,
            record.outcome.tool_calls,
            record.outcome.segments,
            cost,
        )

    async def _close(self, env: Environment, key: EpisodeKey) -> None:
        try:
            await env.close()
        except Exception:
            # The episode is already recorded; a failed cleanup must not abort the remaining episodes.
            log.exception("closing the environment of r%d/%s failed", key.repeat, key.instance.id)
