"""Rendering of the router's rolling memory M_m for prompts."""

from __future__ import annotations

from ctg.runs.records import Memory

NO_MEMORY = "None yet: no segment has been analyzed."


def render_memory(memory: Memory | None) -> str:
    """Markdown view of M_m. Empty lists are left out to keep the prompt short."""
    if memory is None:
        return NO_MEMORY
    sections = [
        f"Summary so far: {memory.summary}",
        f"The last segment made progress on its subtask: {'yes' if memory.subtask_progress else 'no'}",
    ]
    if memory.held_resources:
        sections.append(
            "Held resources:\n"
            + "\n".join(f"- {resource.name}: {resource.quantity}" for resource in memory.held_resources)
        )
    _add_list(sections, "Environment notes", memory.environment_notes)
    if memory.inconsistencies:
        lines = [
            f"- {item.description} Cause: {item.cause} Correction plan: {'; '.join(item.correction_plan)}"
            for item in memory.inconsistencies
        ]
        sections.append("Inconsistencies:\n" + "\n".join(lines))
    _add_list(sections, "Failure analysis", memory.failure_analysis)
    _add_list(sections, "Analysis of the last segment", memory.segment_analysis)
    return "\n\n".join(sections)


def _add_list(sections: list[str], title: str, items: list[str]) -> None:
    if items:
        sections.append(f"{title}:\n" + "\n".join(f"- {item}" for item in items))
