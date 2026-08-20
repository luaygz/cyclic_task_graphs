# Complete cyclic subtask graphs benchmark experiments

This standalone repository contains the public experiment harness for
ALFWorld, TextCraft, and the Vals AI Finance Agent benchmark. It preserves four
execution methods: `react`, per-case specialized cyclic graphs (`spec-cyc`), a
versioned reusable generalized graph (`gen-cyc`), and per-case dependency DAGs
with retry (`depdag-retry`). It contains no history, secrets, browser sessions,
MongoDB dumps, private notebooks, or private application code.

> **License:** this repository is licensed under the MIT License; see `LICENSE`.
> `CITATION.cff` contains owner-controlled paper metadata. Third-party code and
> data retain the licenses documented in `THIRD_PARTY_NOTICES.md`.

## Quick start

Python 3.11 is required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[storage,openai,textcraft,finance,dev]'
cp .env.example .env
```

The package never loads `.env` implicitly. Export the variables you need in
your shell or use an environment manager explicitly.

Start the loopback-bound services without deleting their named volumes:

```bash
docker compose up -d
python scripts/bootstrap_db.py --provider-model gpt-4o-mini
python scripts/bootstrap_db.py --provider-model gpt-4o-mini  # idempotency check
python scripts/import_graphs.py
python scripts/import_graphs.py                               # idempotency check
python scripts/verify_db.py
```

Run the side-effect-free representative preflight:

```bash
python -m luna.benchmarks.cli run \
  --benchmark textcraft \
  --mode gen-cyc \
  --config configs/experiments/textcraft/depth2.yaml \
  --seeds 0 \
  --max-cases 1 \
  --concurrency 1 \
  --output-dir outputs/smoke/textcraft \
  --dry-run
```

Dry-run reads configuration and manifests only. It does not create directories,
connect to MongoDB or Redis, initialize an adapter, or contact an API. Remove
`--dry-run` only after reviewing the printed seed and graph hashes, then add
`--yes` for an explicitly authorized live run.

## Benchmark extras

TextCraft recipes are packaged in the wheel and need no external path.

ALFWorld code and game data remain external:

```bash
python -m pip install -e '.[alfworld]'
alfworld-download
export ALFWORLD_DATA=/path/to/alfworld-data
```

`ALFWORLD_DATA` must contain `json_2.1.1/valid_unseen` and the `logic/`
files. CUDA is disabled unless `LUNA_USE_CUDA=true` is explicitly set.

Finance Agent live runs require `EXA_API_KEY`, `SEC_EDGAR_API_KEY`, and a
responsible `SEC_USER_AGENT`. Final answers are graded against every
rubric item by the configured `judge_model` using structured output. Automated tests mock all network, summarization,
and judging paths. Live tests are opt-in and are never run by validation.

## Reproducibility

The paper scripts are dry-run-only safeguards:

```bash
scripts/run_paper_alfworld.sh
scripts/run_paper_textcraft.sh
scripts/run_paper_finance_agent.sh
```

Expected test counts are ALFWorld OOD 134, TextCraft 203/82/11 at depths
2/3/4, and Finance Agent 35. Each run stores the full immutable configuration,
seed hash, graph hash, provider model names, routing and tool limits, source
commit, timestamps, usage, exceptions, and output paths.

See `docs/REPRODUCIBILITY.md` for clean-wheel validation and service tests,
`docs/DATA_PROVENANCE.md` for dataset terms, and `docs/SOURCE_MANIFEST.md` for
the complete private-source copy ledger.

## Results policy

Only run IDs marked `approved` in `results/paper_manifest.csv` are exported by
`scripts/export_results.py`. The export excludes raw pages, prompts,
trajectories, authentication state, and full MongoDB documents. Unknown
historical fields are recorded as `NOT_RECORDED`, never inferred.

