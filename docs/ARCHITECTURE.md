# Architecture

Configuration flows in one direction:

```text
CLI + YAML + explicit environment
            |
            v
 immutable RunConfig + seed/graph hashes
            |
            v
 lazy benchmark adapter --> DecisionModel
            |                    |
            +------ steps -------+
            |
            +--> MongoDB immutable records
            +--> Redis progress events/cache namespace
            +--> JSONL, state, prompt, and summary files
```

`luna.benchmarks.registry` stores import strings, so importing Finance Agent
does not import ALFWorld or TextCraft. Configuration loading performs no
directory creation and no `.env` loading. The CLI returns from `--dry-run`
before constructing service clients or adapters.

MongoDB collections are `definitions`, `graphs`, `benchmark_runs`,
`test_cases`, and `task_outputs`. The benchmark runner inserts run records once
after execution; it never rewrites definitions. `bootstrap_db.py` is the only
definition upsert path. Redis keys and channels are scoped beneath
`luna:run:<run-id>:`; no database-wide flush is issued.

Graph JSON is normalized before hashing. An existing alias/version cannot be
overwritten with a different hash, so any graph change requires a version bump.

