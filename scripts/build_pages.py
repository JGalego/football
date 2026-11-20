"""Build the lite edition of the notebook for GitHub Pages.

The lite edition must never contain StatsBomb or Metrica row-level data (their terms do not allow republishing it).
So this script does not export from the project folder, where `public/` may hold those files. It stages a clean
copy containing only an allowlist of open files, then exports that.

Usage:
    python scripts/build_pages.py --out _site/live       # WebAssembly site: fully interactive, but the browser must download and boot Python (minutes)
    python scripts/build_pages.py --out _static --html   # run the lite notebook on the server and export static HTML: opens instantly, not interactive

The Pages workflow publishes the static page at the site root and the WebAssembly site under /live/.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Files that may be published. Everything else in public/ is ignored on purpose.
ALLOW_FILES = [
    "datasets.json", "timeline.json", "frontier.json", "atlas.json", "media.json", "stats.json", "player_photos.json",  # research tables and aggregate counts
    "intl_results.csv.gz", "intl_shootouts.csv",  # international results (CC0)
    "tm_age_curve.csv",  # Transfermarkt valuations by age (CC0, aggregated)
    "statsbomb-logo.png",  # attribution logo, if you added it from StatsBomb's media pack
]
ALLOW_DIRS = ["media", "snapshots"]  # Wikimedia Commons derivatives (credited in the notebook) and static snapshots

FORBIDDEN = ("actions_", "shots.csv", "shot_frames", "appearances", "matches.csv", "players.csv", "teams.csv", "tracking_", "final_")


def stage(dest: Path) -> None:
    # The lite edition never runs the plotly cells (they need Metrica tracking data) and Pyodide does not bundle plotly,
    # so drop it from the staged copy's dependency header. The real notebook keeps it for the full edition.
    source = (ROOT / "the_beautiful_data.py").read_text()
    source = "\n".join(line for line in source.split("\n") if not line.startswith('#     "plotly'))
    (dest / "the_beautiful_data.py").write_text(source)
    shutil.copy(ROOT / "head.html", dest / "head.html")  # tab icon (a football)
    pub = dest / "public"
    pub.mkdir()
    for name in ALLOW_FILES:
        src = ROOT / "public" / name
        if src.exists():
            shutil.copy(src, pub / name)
        elif name not in ("statsbomb-logo.png",):
            print(f"  ! missing public/{name} (run scripts/prepare_data.py first)", file=sys.stderr)
    for d in ALLOW_DIRS:
        src = ROOT / "public" / d
        if src.is_dir():
            shutil.copytree(src, pub / d)
    leaked = [p.name for p in pub.rglob("*") if p.is_file() and p.name.startswith(FORBIDDEN)]
    if leaked:
        sys.exit(f"refusing to build: row-level files in the staged site: {leaked}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--html", action="store_true", help="export static HTML (executes the notebook) instead of a WebAssembly site")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        stage(work)
        if out.exists():
            shutil.rmtree(out)
        marimo = [sys.executable, "-m", "marimo", "export"]
        if args.html:
            out.mkdir(parents=True)
            cmd = [*marimo, "html", "--no-include-code", "the_beautiful_data.py", "-o", str(out / "index.html")]
        else:
            cmd = [*marimo, "html-wasm", "the_beautiful_data.py", "-o", str(out), "--mode", "run", "--no-show-code"]
        # The static copy is told it is static (FOOTBALL_STATIC) so it can show a banner linking to the interactive build.
        env = {**os.environ, "FOOTBALL_STATIC": "1"} if args.html else None
        subprocess.run(cmd, cwd=work, check=True, env=env)
    leaked = [p.name for p in out.rglob("*") if p.is_file() and p.name.startswith(FORBIDDEN)]
    if leaked:
        sys.exit(f"refusing to publish: row-level files in the output: {leaked}")
    print(f"built {out}  ({sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
