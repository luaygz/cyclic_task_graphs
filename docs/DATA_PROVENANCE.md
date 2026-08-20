# Data and code provenance

## TextCraft

The environment and Minecraft 1.16.5 crafting recipes are adapted from the
MIT-licensed [ADaPT repository](https://github.com/archiki/ADaPT). The public
port adds exact-depth selection, `importlib.resources` loading, deterministic
split files, and a fix preventing recipe-index mutation during traversal.

The source experiment enumerates 291 depth-2 and 117 depth-3 seeds, shuffles
each with Python `random.Random(42)`, and holds out the first 88 and 35 for
training. The remaining test counts are 203 and 82. All 11 depth-4 seeds are
test cases.

## ALFWorld

ALFWorld is not vendored. Install and download it from the
[official MIT-licensed repository](https://github.com/alfworld/alfworld). The
primary manifest uses the official `valid_unseen` / `eval_out_of_distribution`
split with 134 game indices. Consult ALFWorld for downstream ALFRED and
TextWorld data terms.

## Finance Agent

The 50-row `public.csv` and tool design come from the MIT-licensed
[Vals AI Finance Agent repository](https://github.com/vals-ai/finance-agent).
The split is reproduced by shuffling indices 0..49 with
`random.Random(42)`, assigning 15 to train and 35 to test. The CSV is kept
unmodified; every rubric is parsed and validated during release tests.

Web and SEC content retrieved during a run is transient research input and is
not part of curated exports. Users are responsible for API terms and the SEC's
request-identification and rate guidance.

