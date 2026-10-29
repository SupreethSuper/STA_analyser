"""
Centralized filesystem paths for the sibling project folders STA_analyser
depends on.

STA_analyser is expected to sit as one of 4 sibling folders under a common
parent directory. The parent folder's name/location does not matter (it
can be Downloads, or anything else later) -- only these 4 child names,
spelled exactly as below, matter:

    <any_parent_folder>/
        STA_analyser/                   <- this repo
        TimingPredict/
        netlists/
        tiny-gpu-for-test-openlane/

Every script in this repo should import the constants below instead of
hardcoding its own relative path (e.g. "../netlists"). Paths are anchored
to this file's own location (Path(__file__).resolve()), not the current
working directory, so they resolve correctly no matter how/where a script
is invoked.

Usage:
    from paths import NETLISTS_DIR, TIMING_PREDICT_DIR, TINY_GPU_DIR

    sdc = NETLISTS_DIR / "aes128" / "aes128.cts_1.sdc"
"""

from pathlib import Path

# Directory containing this file, i.e. .../STA_analyser
STA_ANALYSER_DIR = Path(__file__).resolve().parent

# Common parent of all 4 sibling folders. Its name/location is irrelevant.
_PARENT_DIR = STA_ANALYSER_DIR.parent

# --- Sibling folder locations (exact, case-sensitive names) -------------
NETLISTS_DIR = _PARENT_DIR / "netlists"
TIMING_PREDICT_DIR = _PARENT_DIR / "TimingPredict"
TINY_GPU_DIR = _PARENT_DIR / "tiny-gpu-for-test-openlane"

_REQUIRED_SIBLINGS = {
    "netlists": NETLISTS_DIR,
    "TimingPredict": TIMING_PREDICT_DIR,
    "tiny-gpu-for-test-openlane": TINY_GPU_DIR,
}


def _check_siblings() -> None:
    missing = [name for name, p in _REQUIRED_SIBLINGS.items() if not p.is_dir()]
    if not missing:
        return
    lines = [
        "STA_analyser: missing required sibling folder(s): " + ", ".join(missing),
        "",
        "STA_analyser expects to sit next to these folders, all directly",
        f"inside the same parent directory ({_PARENT_DIR}):",
        "",
        "    <parent>/",
        "        STA_analyser/                     (this repo)",
        "        TimingPredict/                    <- clone/pull from the team repo",
        "        netlists/                         <- download+extract from the team Drive link",
        "        tiny-gpu-for-test-openlane/       <- clone from the team repo",
        "",
        f"Place the missing folder(s) directly inside: {_PARENT_DIR}",
        "using exactly those names (case-sensitive on Mac/Linux).",
        "See STA_analyser/README.md -> 'Project layout' for setup steps.",
    ]
    raise FileNotFoundError("\n".join(lines))


_check_siblings()


if __name__ == "__main__":
    # Smoke test: `python paths.py` from inside STA_analyser/
    for name, p in _REQUIRED_SIBLINGS.items():
        print(f"[OK] {name}: {p}")
    print("All sibling folders found.")
