"""Tables shaped like the paper's, rendered as Markdown or CSV.

A `Table` holds typed cells. Markdown shows `mean ± std`, `value [low, high]` and `low–high`; CSV splits
those cells into separate numeric columns and leaves unavailable values empty (Markdown shows "D/A",
as the paper does). Numbers that round to zero print without a minus sign.

Every table built from runs notes excluded error episodes and names incomplete runs.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ctg.analysis.audit import TransitionAudit
from ctg.analysis.exploration import ExplorationComparison
from ctg.analysis.failures import FailureLabel, FinanceDiagnostics, finance_diagnostics
from ctg.analysis.load import Run, run_notes
from ctg.analysis.metrics import MeanStd, Robustness, run_metrics
from ctg.analysis.routing import Range, RoutingSummary, RunRouting, run_routing, summarize_routing
from ctg.analysis.tokens import RunUsage, TokenComparison, TokenScope, run_usage
from ctg.config import Controller, Role

MISSING = "D/A"

SR_NOTE = "SR: mean ± std over complete repeats; the pooled rate (std D/A) when a run has no complete repeat."

VARIANT_LABELS = {
    "textcraft_depth2": "TextCraft-2",
    "textcraft_depth3": "TextCraft-3",
    "textcraft_depth4": "TextCraft-4",
    "alfworld_train": "ALFWorld (train)",
    "alfworld_valid_seen": "ALFWorld (valid seen)",
    "alfworld_valid_unseen": "ALFWorld (valid unseen)",
    "finance_agent": "Finance-Agent",
}
CONTROLLER_LABELS = {
    Controller.REACT: "ReAct",
    Controller.DEPDAG: "DepDAG",
    Controller.DEPDAG_NORETRY: "DepDAG-NoRetry",
    Controller.SPEC_CYC: "Spec-Cyc",
    Controller.GEN_CYC: "Gen-Cyc",
}
CONTROLLER_ORDER = tuple(CONTROLLER_LABELS)


def variant_label(variant: str) -> str:
    return VARIANT_LABELS.get(variant, variant)


def variant_order(variant: str) -> tuple[int, str]:
    order = list(VARIANT_LABELS)
    return (order.index(variant) if variant in order else len(order), variant)


def run_order(run: Run) -> tuple[tuple[int, str], int]:
    return variant_order(run.variant), CONTROLLER_ORDER.index(run.controller)


@dataclass(frozen=True)
class Interval:
    value: float
    low: float
    high: float


Cell = str | int | float | MeanStd | Interval | Range | None


@dataclass(frozen=True)
class Column:
    name: str
    digits: int = 1
    signed: bool = False


@dataclass(frozen=True)
class Table:
    key: str
    """Short slug used in CSV file names."""
    title: str
    columns: tuple[Column, ...]
    rows: tuple[tuple[Cell, ...], ...]
    notes: tuple[str, ...] = ()


def _number(value: float, column: Column) -> str:
    return f"{value:+z.{column.digits}f}" if column.signed else f"{value:z.{column.digits}f}"


def format_cell(cell: Cell, column: Column) -> str:
    match cell:
        case None:
            return MISSING
        case str():
            return cell
        case bool() | int():
            return str(cell)
        case float():
            return _number(cell, column)
        case MeanStd(mean=mean, std=std):
            spread = MISSING if std is None else f"{std:z.{column.digits}f}"
            return f"{_number(mean, column)} ± {spread}"
        case Interval(value=value, low=low, high=high):
            d = column.digits
            return f"{value:z.{d}f} [{low:z.{d}f}, {high:z.{d}f}]"
        case Range(low=low, high=high):
            low_text, high_text = _number(low, column), _number(high, column)
            return low_text if low_text == high_text else f"{low_text}–{high_text}"


def to_markdown(table: Table) -> str:
    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

    out = [f"### {table.title}", "", line([c.name for c in table.columns])]
    out.append("|" + "|".join("---" for _ in table.columns) + "|")
    out.extend(
        line([format_cell(cell, col) for cell, col in zip(row, table.columns, strict=True)])
        for row in table.rows
    )
    if table.notes:
        out.append("")
        out.extend(table.notes)
    return "\n".join(out)


def render_markdown(tables: Sequence[Table]) -> str:
    return "\n\n".join(to_markdown(t) for t in tables) + "\n"


def _csv_suffixes(cell: Cell) -> tuple[str, ...]:
    match cell:
        case MeanStd():
            return (" mean", " std")
        case Interval():
            return ("", " low", " high")
        case Range():
            return (" low", " high")
        case _:
            return ("",)


def _csv_values(cell: Cell, width: int) -> tuple[str, ...]:
    match cell:
        case None:
            return ("",) * width
        case MeanStd(mean=mean, std=std):
            return (repr(mean), "" if std is None else repr(std))
        case Interval(value=value, low=low, high=high):
            return (repr(value), repr(low), repr(high))
        case Range(low=low, high=high):
            return (repr(low), repr(high))
        case float():
            return (repr(cell),)
        case _:
            return (str(cell),)


def to_csv(table: Table) -> str:
    suffixes = [
        _csv_suffixes(next((row[i] for row in table.rows if row[i] is not None), None))
        for i in range(len(table.columns))
    ]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([col.name + s for col, ss in zip(table.columns, suffixes, strict=True) for s in ss])
    for row in table.rows:
        writer.writerow(
            [v for cell, ss in zip(row, suffixes, strict=True) for v in _csv_values(cell, len(ss))]
        )
    return buffer.getvalue()


def write_csv(tables: Sequence[Table], path: Path) -> list[Path]:
    """One table goes to `path`; several go to `<stem>.<table key><suffix>` next to it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(tables) == 1:
        targets = [path]
    else:
        targets = [path.with_name(f"{path.stem}.{t.key}{path.suffix or '.csv'}") for t in tables]
    for table, target in zip(tables, targets, strict=True):
        target.write_text(to_csv(table), encoding="utf-8")
    return targets


def _notes(runs: Sequence[Run], *extra: str) -> tuple[str, ...]:
    return (*extra, *run_notes(runs))


def _percent(fraction: float) -> float:
    return 100.0 * fraction


# Run overview


def summary_table(runs: Sequence[Run]) -> Table:
    rows: list[tuple[Cell, ...]] = []
    for run in sorted(runs, key=run_order):
        m, usage = run_metrics(run), run_usage(run)
        wilson = m.wilson
        visits = m.visitation
        rows.append(
            (
                run.name,
                variant_label(run.variant),
                CONTROLLER_LABELS[run.controller],
                run.config.executors.value,
                m.episodes,
                m.errors,
                m.wins,
                m.sr,
                Interval(_percent(m.wins / m.episodes), _percent(wilson[0]), _percent(wilson[1]))
                if wilson
                else None,
                m.tc,
                m.str_,
                visits.ats if visits else None,
                visits.aus if visits else None,
                visits.asr if visits else None,
                len(m.transitions.all),
                len(m.transitions.won),
                len(m.transitions.never_won),
                usage.tokens_per_case,
                usage.spend_usd,
            )
        )
    columns = (
        Column("Run"),
        Column("Benchmark"),
        Column("Graph"),
        Column("Executors"),
        Column("Test cases"),
        Column("Errors"),
        Column("Wins"),
        Column("SR (%)"),
        Column("Pooled SR (Wilson 95% CI, %)"),
        Column("TC"),
        Column("STR"),
        Column("ATS"),
        Column("AUS"),
        Column("ASR"),
        Column("UT (all)"),
        Column("UT (won)"),
        Column("UT (never won)"),
        Column("Tokens / test case", digits=0),
        Column("Spend (USD)", digits=4),
    )
    return Table("summary", "Run summary", columns, tuple(rows), _notes(runs, SR_NOTE))


# Table 2 / 13: benchmark rows x controller columns


def _cells_by_variant_and_controller(runs: Sequence[Run]) -> dict[tuple[str, Controller], Run]:
    cells: dict[tuple[str, Controller], Run] = {}
    for run in runs:
        key = (run.variant, run.controller)
        if key in cells:
            raise ValueError(
                f"{cells[key].name} and {run.name} both fill {variant_label(run.variant)} / "
                f"{CONTROLLER_LABELS[run.controller]}; pass one run per benchmark and controller"
            )
        cells[key] = run
    return cells


def main_table(runs: Sequence[Run]) -> Table:
    cells = _cells_by_variant_and_controller(runs)
    controllers = [c for c in CONTROLLER_ORDER if any(c is k[1] for k in cells)]
    variants = sorted({k[0] for k in cells}, key=variant_order)
    columns = [Column("Benchmark")]
    for controller in controllers:
        label = CONTROLLER_LABELS[controller]
        columns += [Column(f"{label} SR (%)"), Column(f"{label} TC"), Column(f"{label} STR")]
    rows: list[tuple[Cell, ...]] = []
    for variant in variants:
        row: list[Cell] = [variant_label(variant)]
        for controller in controllers:
            run = cells.get((variant, controller))
            m = run_metrics(run) if run else None
            row += [m.sr, m.tc, m.str_] if m else [None, None, None]
        rows.append(tuple(row))
    return Table(
        "main",
        "Overall comparison (Table 2 layout)",
        tuple(columns),
        tuple(rows),
        _notes(
            runs,
            "SR: mean ± std over complete repeats; the pooled rate (std D/A) when a run has no complete "
            "repeat. TC: tool calls on successful episodes. STR = SR / TC.",
        ),
    )


# Table 15 / 16 and 19 / 20: configuration sweeps


_CONFIG_COLUMNS = (
    Column("Benchmark"),
    Column("Graph"),
    Column("Executors"),
    Column("Planner"),
    Column("Executor"),
    Column("Router"),
)


def _config_cells(run: Run) -> tuple[Cell, ...]:
    models = run.config.models
    return (
        variant_label(run.variant),
        CONTROLLER_LABELS[run.controller],
        run.config.executors.value,
        models.planner,
        models.executor,
        models.router,
    )


def sweep_tables(runs: Sequence[Run]) -> list[Table]:
    ordered = sorted(runs, key=run_order)
    metrics = [run_metrics(r) for r in ordered]
    sweep_rows: list[tuple[Cell, ...]] = []
    ut_rows: list[tuple[Cell, ...]] = []
    for run, m in zip(ordered, metrics, strict=True):
        visits = m.visitation
        sweep_rows.append(
            (
                *_config_cells(run),
                m.sr,
                m.tc,
                m.str_,
                visits.ats if visits else None,
                visits.aus if visits else None,
                visits.asr if visits else None,
            )
        )
        t = m.transitions
        ut_rows.append((*_config_cells(run), m.sr, len(t.all), len(t.won), len(t.never_won)))
    sweep_columns = (
        *_CONFIG_COLUMNS,
        Column("SR (%)"),
        Column("TC"),
        Column("STR"),
        Column("ATS"),
        Column("AUS"),
        Column("ASR"),
    )
    ut_columns = (
        *_CONFIG_COLUMNS,
        Column("SR (%)"),
        Column("UT (all)"),
        Column("UT (won)"),
        Column("UT (never won)"),
    )
    return [
        Table(
            "sweep",
            "Model-tier sweep (Table 15 layout)",
            sweep_columns,
            tuple(sweep_rows),
            _notes(
                runs,
                SR_NOTE,
                "ATS / AUS / ASR: subtask visits, unique subtasks and revisits on successful episodes.",
            ),
        ),
        Table(
            "unique_transitions",
            "Unique transitions (Table 19 layout)",
            ut_columns,
            tuple(ut_rows),
            _notes(
                runs,
                SR_NOTE,
                "UT: distinct executed (source, target) transitions over all, won and never-won episodes.",
            ),
        ),
    ]


# Tables 3 / 17 / 18: fault injection


def robustness_table(pairs: Sequence[tuple[Run, Run, Robustness]]) -> Table:
    rows: list[tuple[Cell, ...]] = []
    for nominal, _, r in sorted(pairs, key=lambda p: run_order(p[0])):
        rows.append(
            (
                variant_label(nominal.variant),
                CONTROLLER_LABELS[nominal.controller],
                nominal.config.executors.value,
                r.nominal.sr,
                r.nominal.tc,
                r.perturbed.sr,
                r.perturbed.tc,
                r.delta_sr,
                r.delta_tc,
                r.relative_drop,
                r.srr,
            )
        )
    columns = (
        Column("Benchmark"),
        Column("Graph"),
        Column("Executors"),
        Column("Unperturbed SR (%)"),
        Column("Unperturbed TC"),
        Column("Perturbed SR (%)"),
        Column("Perturbed TC"),
        Column("ΔSR (pp)", signed=True),
        Column("ΔTC", signed=True),
        Column("Rel. diff. (%)"),
        Column("SRR", digits=3),
    )
    runs = [run for nominal, perturbed, _ in pairs for run in (nominal, perturbed)]
    return Table(
        "robustness",
        "Robustness under fault injection (Tables 3 and 18 layout)",
        columns,
        tuple(rows),
        _notes(
            runs,
            SR_NOTE,
            "SRR = SR_perturbed / SR_nominal; ΔTC = TC_perturbed − TC_nominal (successful episodes).",
        ),
    )


# Table 12: shared-win token comparison, plus spend


def token_tables(baseline: Run, comparisons: Sequence[tuple[Run, TokenComparison]]) -> list[Table]:
    rows: list[tuple[Cell, ...]] = []

    def scope_row(compare: Run, scope_name: str, scope: TokenScope) -> tuple[Cell, ...]:
        return (
            variant_label(compare.variant),
            f"{baseline.name} vs {compare.name}",
            scope_name,
            scope.cases_a,
            scope.cases_b,
            scope.avg_a,
            scope.avg_b,
            scope.excess,
            scope.relative_pct,
        )

    for compare, comparison in comparisons:
        rows.append(scope_row(compare, "Shared wins", comparison.shared_wins))
        rows.append(scope_row(compare, "All wins", comparison.all_wins))
    columns = (
        Column("Benchmark"),
        Column("Compared pair"),
        Column("Scope"),
        Column("Baseline cases"),
        Column("Compared cases"),
        Column("Baseline avg total", digits=2),
        Column("Compared avg total", digits=2),
        Column("Excess (compared − baseline)", digits=2, signed=True),
        Column("Rel. diff. (%)", digits=2, signed=True),
    )
    runs = [baseline, *(c for c, _ in comparisons)]
    comparison_table = Table(
        "tokens",
        "Shared-win token comparison (Table 12 layout)",
        columns,
        tuple(rows),
        _notes(
            runs,
            "Average input + output tokens per test case over planner, executor, router and tool calls; "
            "judge tokens excluded. Shared wins: (instance, repeat) pairs won by both runs.",
        ),
    )
    return [comparison_table, usage_table(runs)]


def usage_table(runs: Sequence[Run]) -> Table:
    usages: list[RunUsage] = [run_usage(r) for r in runs]
    roles = [role for role in Role if any(role in u.by_role for u in usages)]
    columns = (
        Column("Run"),
        Column("Test cases"),
        Column("Tokens / test case", digits=0),
        Column("Judge tokens / test case", digits=0),
        *(Column(f"{role.value.capitalize()} (USD)", digits=4) for role in roles),
        Column("Total (USD)", digits=4),
    )
    rows = tuple(
        (
            u.run,
            u.episodes,
            u.tokens_per_case,
            u.judge_tokens_per_case,
            *(u.by_role[role].cost_usd if role in u.by_role else 0.0 for role in roles),
            u.spend_usd,
        )
        for u in usages
    )
    return Table(
        "spend",
        "Token usage and spend",
        columns,
        rows,
        _notes(
            runs,
            "Spend (USD): one-time Gen-Cyc planning plus every call of every stored episode, including error "
            "episodes and the earlier error attempts of episodes re-run on resume. Tokens / test case: the "
            "final attempt of each non-error episode.",
        ),
    )


# Table 14: DepDAG transition audit


def audit_table(audit: TransitionAudit, runs: Sequence[Run]) -> Table:
    rows: tuple[tuple[Cell, ...], ...] = (
        ("DepDAG runs inspected", audit.runs),
        ("Test cases inspected", audit.test_cases),
        ("Typed transitions inspected", audit.transitions),
        ("Same-subtask retry transitions", audit.retry),
        ("DAG-forward transitions", audit.forward),
        ("Terminal transitions", audit.terminal),
        ("Other transition types", audit.other),
        ("Invalid transitions", audit.invalid),
    )
    return Table(
        "audit",
        "DepDAG transition audit (Table 14 layout)",
        (Column("Quantity"), Column("Count")),
        rows,
        _notes(runs),
    )


# Tables 21-24: graph size and routing behavior


_ROUTING_TABLE_NUMBERS = {Controller.GEN_CYC: (21, 22), Controller.SPEC_CYC: (23, 24)}


def routing_tables(runs: Sequence[Run]) -> list[Table]:
    ordered = sorted(runs, key=run_order)
    per_run: list[RunRouting] = [run_routing(r) for r in ordered]
    tables = [
        Table(
            "routing_runs",
            "Graph size and routing per run",
            (
                Column("Run"),
                Column("Benchmark"),
                Column("Graph"),
                Column("Test cases"),
                Column("Nodes"),
                Column("Criterion words"),
                Column("Self-loops / test case"),
                Column("Inter-node / test case"),
            ),
            tuple(
                (
                    r.run,
                    variant_label(r.variant),
                    CONTROLLER_LABELS[r.controller],
                    r.test_cases,
                    r.nodes,
                    r.criterion_words,
                    r.self_loops,
                    r.inter_node,
                )
                for r in per_run
            ),
            _notes(
                runs, "Self-loops: cyclic self / DepDAG retry. Inter-node: cyclic cross / DepDAG forward."
            ),
        )
    ]
    summaries = summarize_routing(per_run)
    for controller in CONTROLLER_ORDER:
        group: list[RoutingSummary] = [s for s in summaries if s.controller is controller]
        if not group:
            continue
        label = CONTROLLER_LABELS[controller]
        group_notes = _notes([r for r in ordered if r.controller is controller])
        size_number, routing_number = _ROUTING_TABLE_NUMBERS.get(controller, (None, None))
        size_ref = f" (Table {size_number} layout)" if size_number else ""
        routing_ref = f" (Table {routing_number} layout)" if routing_number else ""
        tables.append(
            Table(
                f"graph_size_{controller.value}",
                f"Observed {label} graph size by benchmark{size_ref}",
                (
                    Column("Benchmark"),
                    Column("Runs"),
                    Column("Average nodes (observed range)"),
                    Column("Avg. criterion length (words)"),
                ),
                tuple((variant_label(s.variant), s.runs, s.nodes, s.criterion_words) for s in group),
                group_notes,
            )
        )
        tables.append(
            Table(
                f"routing_{controller.value}",
                f"Observed {label} routing behavior by benchmark{routing_ref}",
                (
                    Column("Benchmark"),
                    Column("Runs"),
                    Column("Avg. self-loops / test case"),
                    Column("Avg. inter-node transitions / test case"),
                ),
                tuple((variant_label(s.variant), s.runs, s.self_loops, s.inter_node) for s in group),
                group_notes,
            )
        )
    return tables


# Tables 9-11: Finance-Agent coverage, tool use and failure labels


def _setting(run: Run) -> str:
    return f"{CONTROLLER_LABELS[run.controller]} / {run.config.models.executor}"


def _access_path(run: Run) -> str | None:
    providers = sorted({c.provider.value for e in run.valid for c in e.llm_calls if c.role is Role.EXECUTOR})
    return ", ".join(providers) if providers else None


def wilson_table(runs: Sequence[Run]) -> Table:
    rows: list[tuple[Cell, ...]] = []
    for run in sorted(runs, key=run_order):
        m = run_metrics(run)
        wilson = m.wilson
        models = run.config.models
        rows.append(
            (
                variant_label(run.variant),
                CONTROLLER_LABELS[run.controller],
                "-" if run.controller is Controller.REACT else f"{models.planner} / {models.router}",
                models.executor,
                _access_path(run),
                f"{m.wins}/{m.episodes}",
                Interval(_percent(m.wins / m.episodes), _percent(wilson[0]), _percent(wilson[1]))
                if wilson
                else None,
                m.tc,
                m.tc_all,
            )
        )
    columns = (
        Column("Benchmark"),
        Column("Controller"),
        Column("Planner / router"),
        Column("Executor / ReAct model"),
        Column("Access path"),
        Column("Wins"),
        Column("SR (Wilson 95% CI, %)"),
        Column("TC (success)", digits=2),
        Column("TC (all)", digits=2),
    )
    return Table(
        "wilson",
        "Success with Wilson 95% intervals (Table 9 layout)",
        columns,
        tuple(rows),
        _notes(runs, "Wilson intervals (Eq. 9, z = 1.96) over wins pooled across repeats."),
    )


def finance_tables(runs: Sequence[Run]) -> list[Table]:
    ordered = sorted(runs, key=run_order)
    diagnostics: list[FinanceDiagnostics] = [finance_diagnostics(r) for r in ordered]
    tool_use = Table(
        "finance_tool_use",
        "Finance-Agent tool-use diagnostics (Table 10 layout)",
        (
            Column("Controller / model"),
            Column("Wins"),
            Column("Final-answer rate", digits=3),
            Column("Source-submission rate", digits=3),
            Column("Budget-exhaustion rate", digits=3),
            Column("Mean tool calls", digits=2),
        ),
        tuple(
            (
                _setting(run),
                f"{d.wins}/{d.episodes}",
                d.final_answer_rate,
                d.source_submission_rate,
                d.budget_exhaustion_rate,
                d.tool_calls.mean if d.tool_calls else None,
            )
            for run, d in zip(ordered, diagnostics, strict=True)
        ),
        _notes(runs, "Rates are fractions of non-error episodes."),
    )
    labels = list(FailureLabel)
    failure_modes = Table(
        "finance_failures",
        "Trace-derived Finance-Agent failure labels (Table 11 layout)",
        (Column("Setting"), Column("Failed episodes"), *(Column(label.display_name) for label in labels)),
        tuple(
            (_setting(run), d.failed, *(d.failures[label] for label in labels))
            for run, d in zip(ordered, diagnostics, strict=True)
        ),
        _notes(
            runs, "Labels are deterministic trace heuristics (ctg.analysis.failures), not human-validated."
        ),
    )
    return [tool_use, failure_modes]


# A6 / Fig. 4: unique environment states


def exploration_table(comparison: ExplorationComparison, runs: Sequence[Run]) -> Table:
    notes = [
        f"Unique environment states per episode, averaged over each instance's non-error episodes: "
        f"{comparison.cyclic_run} (cyclic) vs {comparison.react_run} (ReAct), over "
        f"{len(comparison.instances)} shared instances."
    ]
    if comparison.mean_difference is not None:
        notes.append(f"Mean difference (cyclic − ReAct): {comparison.mean_difference:+z.2f}.")
    if comparison.unmatched:
        notes.append(f"Instances evaluated by only one run (left out): {comparison.unmatched}.")
    return Table(
        "exploration",
        "Unique environment states per episode (A6)",
        (
            Column("Instance"),
            Column("ReAct", digits=2),
            Column("Cyclic", digits=2),
            Column("Difference (cyclic − ReAct)", digits=2, signed=True),
        ),
        tuple((i.instance_id, i.react, i.cyclic, i.difference) for i in comparison.instances),
        _notes(runs, *notes),
    )
