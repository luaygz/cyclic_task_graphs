# Deliberate port decisions

- The monolithic runner was replaced by a lazy three-entry registry.
- `.env` loading and directory creation no longer happen during import.
- Database bootstrap is explicit; runner startup never mutates agents.
- Redis cache clearing is run-namespaced rather than database-wide.
- The ALFWorld generic-prompt `task_prompt =+ {game_summaries}` typo is absent;
  prompts are composed with normal string concatenation.
- Benchmark run labels use the selected benchmark and split. Finance Agent can
  no longer inherit an ALFWorld dataset label.
- SEC HTML requests require `SEC_USER_AGENT`; no personal identifier is stored.
- EDGAR rate limiting uses three attempts and reports that exact count.
- Long Finance inputs are summarized in bounded chunks and recursively reduced.
- The Finance executor definition explicitly records a 20-call overall limit.
- Stale async demos were not ported.
- TextCraft recipe traversal copies stored lists before recursion, preventing
  reset from appending dependency recipes to the global recipe index.

