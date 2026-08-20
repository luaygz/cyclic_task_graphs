# Reproducibility and validation

## No-services checks

```bash
python scripts/validate_release.py
python -m compileall -q src scripts tests
pytest -q
python -m build
python -m venv /tmp/luna-wheel-check
/tmp/luna-wheel-check/bin/pip install dist/luna_benchmarks-*.whl
/tmp/luna-wheel-check/bin/python -c \
  'import luna; from luna.benchmarks.registry import benchmark_names; print(benchmark_names())'
```

`pytest` loads packaged TextCraft data, programmatically solves one case,
parses every Finance rubric, mocks every Finance network/tool family, checks an
actionable ALFWorld missing-data error, and runs each benchmark label through
all four modes with a fake model.

## Service checks

```bash
docker compose up -d
python scripts/bootstrap_db.py
python scripts/bootstrap_db.py
python scripts/import_graphs.py
python scripts/import_graphs.py
python scripts/verify_db.py
LUNA_RUN_SERVICE_TESTS=1 pytest -q tests/integration
```

For each graph, use `scripts/export_graph.py` and compare the printed hash to
the corresponding local manifest. Do not use `docker compose down -v` unless
volume deletion is explicitly intended.

## Paper commands

All paper shell scripts include `--dry-run`. Review their seed count/hash,
graph hash, database name, model mapping, and output path. A live command is a
paid operation and requires explicit authorization, configured services and
keys, and removal of `--dry-run`.

Historical settings absent from immutable source records must be exported as
`NOT_RECORDED`.

