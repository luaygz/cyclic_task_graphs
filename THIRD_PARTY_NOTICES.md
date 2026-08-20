# Third-party notices

This repository contains or interfaces with the following third-party work.
The project license does not replace these licenses.

## TextCraft / ADaPT

The TextCraft environment and Minecraft 1.16.5 recipe corpus are adapted from
[archiki/ADaPT](https://github.com/archiki/ADaPT), released under the MIT
License. Copyright remains with the ADaPT authors and contributors. The LUNA
port adds exact-depth selection, repository-resource loading, and deterministic
split manifests. The required notice is included in
[`licenses/ADAPT-MIT.txt`](licenses/ADAPT-MIT.txt).

## Vals AI Finance Agent

The Finance Agent tool design and `public.csv` are adapted from
[vals-ai/finance-agent](https://github.com/vals-ai/finance-agent), released
under the MIT License. Copyright (c) 2025 Vals AI, Inc. The required copyright
and permission notice is included in
[`licenses/VALS-FINANCE-AGENT-MIT.txt`](licenses/VALS-FINANCE-AGENT-MIT.txt).

## ALFWorld

ALFWorld is an external dependency and is not vendored. It is released under
the MIT License by its authors and contributors. Users obtain code and data
through the [official ALFWorld repository](https://github.com/alfworld/alfworld).
ALFWorld also derives environments from ALFRED and TextWorld; consult upstream
for their terms. The notice applicable to the adapted environment wrapper is
included in [`licenses/ALFWORLD-MIT.txt`](licenses/ALFWORLD-MIT.txt).

## MongoDB, Redis, and Python dependencies

MongoDB and Redis are runtime services supplied by their respective container
images. Python packages are installed from their upstream distributions and
retain their own licenses. Generate an environment-specific dependency report
before redistribution.

