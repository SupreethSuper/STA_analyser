# STA_analyser
The main aim of the project is to create STA analyser software, for general purpose application, but software should be optimised for GPU STA analysis

## Project layout

This repo expects to sit as a **sibling folder** alongside three other folders used by the project. The common parent folder's name and location do not matter (it can stay `Downloads`, or be a differently named folder later) -- what matters is that all four folders sit directly next to each other, with **exactly** these names:

```
<any-parent-folder>/
├── STA_analyser/                   <- this repo
├── TimingPredict/                  <- GNN timing-prediction repo (DAC-2022, extended locally)
├── netlists/                       <- raw dataset: OpenLane/sky130 netlists + techlib (not version-controlled, ~3.75 GB)
└── tiny-gpu-for-test-openlane/     <- tiny-gpu fork used as an OpenLane test design
```

Folder names are case-insensitive on Windows but **case-sensitive on macOS/Linux**. Use the exact casing shown above (`STA_analyser`, `TimingPredict`, `netlists`, `tiny-gpu-for-test-openlane`) on every machine so the same code resolves paths identically for every teammate.

STA_analyser scripts never hardcode `../netlists` or similar relative paths directly. Every script imports the shared constants from [`paths.py`](./paths.py) -- `NETLISTS_DIR`, `TIMING_PREDICT_DIR`, `TINY_GPU_DIR` -- which are computed from the script's own file location (not the current working directory), so they resolve correctly no matter where a script is run from. If a sibling folder is missing or misnamed, importing `paths` fails immediately with a message describing exactly what to place and where, instead of a confusing error buried later.

**Do not commit netlist data** (`.def`/`.sdf`/`.spef`/`.lib`/etc.) into this repo. It belongs only in the `netlists/` sibling folder and the team's Drive link -- this repo's git history already had a large netlist dump added and then removed once; keep it that way.

## Setup

1. Pick any parent folder (e.g. `Downloads`, or anything else -- the name doesn't matter).
2. Clone this repo into it:
   ```
   git clone https://github.com/SupreethSuper/STA_analyser.git
   ```
3. Get the other 3 folders from the team's shared Drive link / repos and place them as direct siblings of `STA_analyser/` (same parent folder), named exactly:
   - `TimingPredict/`
   - `netlists/`
   - `tiny-gpu-for-test-openlane/`
4. Verify the layout -- from inside `STA_analyser/`, run:
   ```
   python paths.py
   ```
   If all 3 sibling folders are found, it prints their resolved paths and exits cleanly. If one is missing or misnamed, it raises a clear error naming exactly which folder to add and where.
