"""Download the raw public datasets used by the notebook into data/raw/.

Everything here is open and free to download:
  * StatsBomb Open Data (events, lineups, 360 freeze frames) - https://github.com/statsbomb/open-data
  * Metrica Sports sample tracking + event data - https://github.com/metrica-sports/sample-data
  * International football results 1872-present (martj42, CC0) - https://github.com/martj42/international_results
  * Photo, audio and video from Wikimedia Commons (CC BY / CC BY-SA / CC0) - listed in public/media.json
  * Transfermarkt players + valuations (dcaribou/transfermarkt-datasets, CC0) - https://github.com/dcaribou/transfermarkt-datasets

Usage:  python scripts/download_data.py [--open-only]
"""

from __future__ import annotations

import argparse
import json
import os
import time
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from _progress import Progress, line_buffered  # sibling module in scripts/

ROOT = Path(__file__).resolve().parents[1]

# Wikimedia asks automated clients to identify themselves with contact details and rate-limits generic ones (HTTP 429).
# Set FOOTBALL_USER_AGENT to something like "my-football-notebook/1.0 (https://example.org/contact)" if downloads get throttled.
USER_AGENT = os.environ.get("FOOTBALL_USER_AGENT", "football-notebook/1.0 (open-source research notebook)")
RAW = ROOT / "data" / "raw"

SB = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
METRICA = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
INTL = "https://raw.githubusercontent.com/martj42/international_results/master"
TM = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"  # dcaribou/transfermarkt-datasets (CC0)

# (competition_id, season_id): StatsBomb competitions used in the notebook
SB_COMPETITIONS = [
    (43, 106),  # FIFA World Cup 2022 (360)
    (43, 3),  # FIFA World Cup 2018
    (55, 282),  # UEFA Euro 2024 (360)
    (55, 43),  # UEFA Euro 2020 (360)
    (223, 282),  # Copa America 2024
    (1267, 107),  # African Cup of Nations 2023 (360)
    (72, 107),  # Women's World Cup 2023 (360)
    (72, 30),  # Women's World Cup 2019
    (53, 106),  # UEFA Women's Euro 2022 (360)
    (53, 315),  # UEFA Women's Euro 2025 (360)
    (9, 281),  # 1. Bundesliga 2023/2024 - Leverkusen's unbeaten season (360)
]


BYTES = [0]  # bytes downloaded so far (a list so worker threads can update it)
_BYTES_LOCK = threading.Lock()


def fetch(url: str, dest: Path, retries: int = 5) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True  # already downloaded
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            with _BYTES_LOCK:
                BYTES[0] += len(data)
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:  # genuinely missing (e.g. a 360 frame StatsBomb never published)
                print(f"  ! missing {url}", file=sys.stderr)
                return False
            # 429 / 5xx: honour Retry-After if the server sends one, otherwise back off
            wait = int(exc.headers.get("Retry-After", 0) or 0) or 15 * (attempt + 1)
            time.sleep(min(wait, 120))
            if attempt == retries - 1:
                print(f"  ! failed {url}: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
            if attempt == retries - 1:
                print(f"  ! failed {url}: {exc}", file=sys.stderr)
    return False


def main() -> None:
    line_buffered()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--open-only", action="store_true",
                    help="skip StatsBomb and Metrica (whose terms do not allow republishing): fetch only CC0 / openly licensed data. Used for the GitHub Pages build.")
    args = ap.parse_args()

    jobs: list[tuple[str, Path]] = []
    if not args.open_only:
        sb = RAW / "statsbomb"
        print("Step 1: listing the StatsBomb matches ...", flush=True)
        fetch(f"{SB}/competitions.json", sb / "competitions.json")
        comps = json.loads((sb / "competitions.json").read_text())
        has360 = {(c["competition_id"], c["season_id"]): bool(c.get("match_available_360")) for c in comps}

        for n_comp, (cid, sid) in enumerate(SB_COMPETITIONS, 1):
            print(f"  competition {n_comp}/{len(SB_COMPETITIONS)} (id {cid}, season {sid})", flush=True)
            mpath = sb / "matches" / str(cid) / f"{sid}.json"
            fetch(f"{SB}/matches/{cid}/{sid}.json", mpath)
            for m in json.loads(mpath.read_text()):
                mid = m["match_id"]
                jobs.append((f"{SB}/events/{mid}.json", sb / "events" / f"{mid}.json"))
                jobs.append((f"{SB}/lineups/{mid}.json", sb / "lineups" / f"{mid}.json"))
                if has360.get((cid, sid)):
                    jobs.append((f"{SB}/three-sixty/{mid}.json", sb / "three-sixty" / f"{mid}.json"))

        for game in (1, 2):
            g = f"Sample_Game_{game}"
            for suffix in ("RawTrackingData_Home_Team", "RawTrackingData_Away_Team", "RawEventsData"):
                name = f"{g}_{suffix}.csv"
                jobs.append((f"{METRICA}/{g}/{name}", RAW / "metrica" / name))

    for name in ("players.csv.gz", "player_valuations.csv.gz"):
        jobs.append((f"{TM}/{name}", RAW / "transfermarkt" / name))

    for name in ("results.csv", "shootouts.csv", "goalscorers.csv"):
        jobs.append((f"{INTL}/{name}", RAW / "international" / name))

    # Wikimedia Commons files for the photo / audio / video demos (see public/media.json for licences).
    # Fetched one at a time: Wikimedia rate-limits parallel clients.
    media = json.loads((ROOT / "public" / "media.json").read_text())
    print(f"Step 2: downloading {len(media)} media files from Wikimedia Commons (one at a time, politely) ...", flush=True)
    for i, m in enumerate(media, 1):
        before = BYTES[0]
        got = fetch(m.get("download_url", m["file_url"]), RAW / "media" / m["raw_name"], retries=6)  # photos: the 1280 px thumbnail (Wikimedia throttles big originals)
        print(f"  [{i}/{len(media)}] {m['raw_name']}: " + (f"ok, {(BYTES[0] - before) / 1e6:.1f} MB" if BYTES[0] > before else "already there" if got else "FAILED"), flush=True)
        if not got:
            print(f"  ! could not fetch {m['raw_name']}", file=sys.stderr)
        time.sleep(2)

    todo = [j for j in jobs if not (j[1].exists() and j[1].stat().st_size > 0)]
    print(f"Step 3: {len(jobs):,} data files, {len(jobs) - len(todo):,} already downloaded, {len(todo):,} to fetch into {RAW} (8 at a time) ...", flush=True)
    progress = Progress("Downloading", len(todo))
    ok = [True] * (len(jobs) - len(todo))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch, *j): j for j in todo}
        for fut in as_completed(futures):
            ok.append(fut.result())
            progress.update(nbytes=0, note=futures[fut][1].name)
            progress.bytes = BYTES[0]  # fetch() counts the bytes; mirror them for the speed readout
    print(f"Done: {sum(ok)}/{len(ok)} files present, {BYTES[0] / 1e6:,.0f} MB downloaded this run.", flush=True)


if __name__ == "__main__":
    main()
