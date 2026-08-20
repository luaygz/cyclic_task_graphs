"""Versioned graph loading and deterministic per-case graph construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from luna.actor.schemas import GraphEdge, GraphManifest, Subtask


def load_graph(path: Path) -> GraphManifest:
    with path.open("r", encoding="utf-8") as handle:
        return GraphManifest.model_validate(json.load(handle))


def build_case_graph(benchmark: str, task_prompt: str, dependency_dag: bool) -> GraphManifest:
    task_hash = hashlib.sha256(task_prompt.encode("utf-8")).hexdigest()[:12]
    if benchmark == "alfworld":
        definitions = [
            ("inspect", "Inspect and localize", ["tool.alfworld_select_command"], "Identify the next relevant object or receptacle."),
            ("manipulate", "Acquire or manipulate", ["tool.alfworld_select_command"], "Advance the required object state."),
            ("place", "Place and verify", ["tool.alfworld_select_command"], "Complete the household goal."),
        ]
    elif benchmark == "textcraft":
        definitions = [
            ("plan", "Resolve recipe dependencies", ["tool.textcraft_inventory"], "Identify the next missing raw or crafted item."),
            ("gather", "Gather raw materials", ["tool.textcraft_get_item"], "Acquire enough non-craftable ingredients."),
            ("craft", "Craft intermediates and target", ["tool.textcraft_craft"], "Craft the target item using an allowed recipe."),
        ]
    elif benchmark == "finance_agent":
        definitions = [
            ("research", "Find authoritative sources", ["tool.finance_agent_web_search", "tool.finance_agent_edgar_search"], "Locate evidence relevant to the question."),
            ("retrieve", "Parse and analyze evidence", ["tool.finance_agent_parse_html", "tool.finance_agent_retrieve_information"], "Extract the facts and figures needed for an answer."),
            ("answer", "Synthesize final answer", ["tool.finance_agent_final_answer"], "Submit a sourced answer."),
        ]
    else:
        raise ValueError(f"unsupported benchmark: {benchmark}")
    subtasks = []
    for index, (node_id, title, tools, criterion) in enumerate(definitions):
        dependencies = [definitions[index - 1][0]] if dependency_dag and index > 0 else []
        subtasks.append(
            Subtask(
                id=node_id,
                title=title,
                agent_alias=f"agent.{benchmark}_select_command",
                tool_aliases=tools,
                prompt=f"{title}. Work only on this phase of the task.\n\n{task_prompt}",
                success_criterion=criterion,
                dependencies=dependencies,
                retry_limit=2 if dependency_dag else 0,
            )
        )
    if dependency_dag:
        edges = [GraphEdge(source=definitions[i][0], target=definitions[i + 1][0]) for i in range(len(definitions) - 1)]
        graph_type = "dependency_dag"
    else:
        edges = [GraphEdge(source=definitions[i][0], target=definitions[(i + 1) % len(definitions)][0]) for i in range(len(definitions))]
        graph_type = "cyclic"
    return GraphManifest(
        alias=f"graph.{benchmark}.case.{task_hash}",
        version="1.0.0",
        benchmark=benchmark,
        graph_type=graph_type,
        entrypoint=definitions[0][0],
        subtasks=subtasks,
        edges=edges,
        metadata={"task_hash": task_hash, "generated_per_case": True},
    )


def normalized_graph_json(graph: GraphManifest) -> str:
    return json.dumps(graph.normalized(), indent=2, sort_keys=True) + "\n"

