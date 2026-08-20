"""Curated definitions used by the explicit idempotent bootstrap step."""

from __future__ import annotations


def definitions() -> list[dict]:
    records = [
        {"kind": "llm", "alias": f"llm.{role}", "provider": "openai", "provider_model": "NOT_RECORDED"}
        for role in ("executor", "planner", "router", "judge")
    ]
    records.extend(
        [
            {"kind": "agent", "alias": "agent.alfworld_select_command", "tools": ["tool.alfworld_select_command"], "max_tool_calls": 75},
            {"kind": "agent", "alias": "agent.textcraft_select_command", "tools": ["tool.textcraft_inventory", "tool.textcraft_get_item", "tool.textcraft_craft"], "max_tool_calls": 100},
            # Deliberately ports the uncommitted FinanceAgent max-tool-call setting.
            {"kind": "agent", "alias": "agent.finance_agent_select_command", "tools": ["tool.finance_agent_web_search", "tool.finance_agent_edgar_search", "tool.finance_agent_parse_html", "tool.finance_agent_retrieve_information", "tool.finance_agent_final_answer"], "max_tool_calls": 20, "parallel_tool_calls": False},
            {"kind": "agent", "alias": "agent.summarizer", "tools": [], "max_tool_calls": 0},
        ]
    )
    tool_aliases = {
        "tool.alfworld_select_command", "tool.textcraft_inventory", "tool.textcraft_get_item",
        "tool.textcraft_craft", "tool.finance_agent_web_search", "tool.finance_agent_edgar_search",
        "tool.finance_agent_parse_html", "tool.finance_agent_retrieve_information",
        "tool.finance_agent_final_answer",
    }
    records.extend({"kind": "tool", "alias": alias} for alias in sorted(tool_aliases))
    return records

