# Third-party notices

This repository's own code is MIT-licensed (see `LICENSE`). It builds on the following benchmarks and data, which keep their own terms.

## TextCraft (ADaPT)

The TextCraft environment in `src/ctg/envs/textcraft/` reimplements the benchmark introduced in:

> Archiki Prasad, Alexander Koller, Mareike Hartmann, Peter Clark, Ashish Sabharwal, Mohit Bansal, Tushar Khot. *ADaPT: As-Needed Decomposition and Planning with Language Models.* Findings of NAACL 2024. https://github.com/archiki/ADaPT (MIT License)

## Minecraft recipe data

TextCraft reads crafting recipe definitions from Minecraft: Java Edition 1.16.5. They are not distributed with this repository. `ctg data fetch textcraft` downloads them from [InventivetalentDev/minecraft-assets](https://github.com/InventivetalentDev/minecraft-assets) at the pinned commit `1b7a1d48b0343c0b8da80604598707df428dc32a` (`data/minecraft/recipes/`) and checks every file against the SHA-256 digests in `data/textcraft/recipes.lock.json`. The recipe data is Mojang game data and is not covered by this repository's MIT license. Minecraft is a trademark of Mojang Synergies AB.

## ALFWorld

The ALFWorld environment uses the `alfworld` package and its game files, downloaded separately with `alfworld-download`:

> Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, Matthew Hausknecht. *ALFWorld: Aligning Text and Embodied Environments for Interactive Learning.* ICLR 2021. https://github.com/alfworld/alfworld (MIT License)

## Finance Agent Benchmark

`data/finance_agent/public.csv` is the public validation set of the Finance Agent Benchmark by Vals AI, licensed under CC BY 4.0:

> Antoine Bigeard, Rayan Krishnan, Shirley Wu, Langston Nashold. *Finance Agent Benchmark: Benchmarking LLMs on Real-world Financial Research Tasks.* arXiv:2508.00828, 2025. https://github.com/vals-ai/finance-agent
