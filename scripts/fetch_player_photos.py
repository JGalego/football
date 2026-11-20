"""Find openly licensed face photos for the players in the notebook.

For every player with enough minutes: search Wikidata by name, keep only items that are footballers, disambiguate by
nationality, take the item's Commons photo (P18), check its licence on Commons, download a small thumbnail, centre a crop
on the face (OpenCV) and save 112 x 112 px to public/players/<player_id>.jpg. Credits go to public/player_photos.json and
are shown in the notebook, because CC BY / CC BY-SA require attribution.

Only photos with a free licence are kept, and a player gets a photo only if a face is found. The rest simply have none.

    uv run python scripts/fetch_player_photos.py            # everything (about 30-60 minutes because of Wikimedia's rate limits; resumable)
    uv run python scripts/fetch_player_photos.py --limit 40 # a quick trial

Needs the `dev` dependency group (OpenCV). Set FOOTBALL_USER_AGENT if Wikimedia throttles you (HTTP 429).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
CACHE = ROOT / "data" / "raw" / "wikidata"
OUT = PUBLIC / "players"
UA = os.environ.get("FOOTBALL_USER_AGENT", "football-notebook/1.0 (open-source research notebook)")
WD = "https://www.wikidata.org/w/api.php"
COMMONS = "https://commons.wikimedia.org/w/api.php"
FOOTBALLER = "Q937857"  # association football player
THROTTLE = 1.0  # seconds between API requests: be polite to Wikimedia
DOWNLOAD_GAP = 2.0  # seconds between thumbnail downloads: upload.wikimedia.org answers HTTP 429 to anything faster
FREE_LICENCE = re.compile(r"^(cc[ -]?by|cc[ -]?0|cc0|public domain|pd|attribution)", re.I)

# StatsBomb country names -> labels Wikidata may use for citizenship (normalised, see norm())
COUNTRY_ALIASES = {
    "england": {"england", "united kingdom"}, "scotland": {"scotland", "united kingdom"}, "wales": {"wales", "united kingdom"},
    "northern ireland": {"northern ireland", "united kingdom"}, "korea republic": {"south korea"}, "ir iran": {"iran"},
    "china pr": {"china", "people s republic of china"}, "united states": {"united states of america", "united states"},
    "czech republic": {"czechia", "czech republic"}, "turkey": {"turkey", "turkiye"}, "turkiye": {"turkey", "turkiye"},
    "cote d ivoire": {"ivory coast", "cote d ivoire"}, "cape verde islands": {"cape verde"}, "congo dr": {"democratic republic of the congo"},
    "republic of ireland": {"ireland"}, "bosnia and herzegovina": {"bosnia and herzegovina"}, "north macedonia": {"north macedonia"},
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s)).strip()


_last = [0.0]


def fetch(url: str, params: dict | None = None, binary: bool = False, retries: int = 6):
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        wait = (DOWNLOAD_GAP if binary else THROTTLE) - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            return data if binary else json.loads(data)
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 400):
                return None
            delay = int(exc.headers.get("Retry-After", 0) or 0) or 20 * (attempt + 1)
            print(f"  .. HTTP {exc.code}, waiting {min(delay, 120)} s", file=sys.stderr)
            time.sleep(min(delay, 120))
        except Exception as exc:  # noqa: BLE001
            print(f"  .. {exc}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    return None


def load_cache(name: str) -> dict:
    p = CACHE / name
    return json.loads(p.read_text()) if p.exists() else {}


def save_cache(name: str, data: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / name).write_text(json.dumps(data))


def search(name: str, cache: dict) -> list[dict]:
    if name in cache:
        return cache[name]
    res = fetch(WD, dict(action="wbsearchentities", search=name, language="en", uselang="en", type="item", limit=8, format="json"))
    hits = [dict(id=h["id"], label=h.get("label", ""), desc=h.get("description", "")) for h in (res or {}).get("search", [])]
    cache[name] = hits
    return hits


def parse_entity(e: dict) -> dict:
    claims = e.get("claims", {})

    def vals(prop):
        out = []
        for c in claims.get(prop, []):
            v = (c.get("mainsnak", {}).get("datavalue") or {}).get("value")
            if v is not None:
                out.append((c.get("rank"), v))
        return out

    occ = {v.get("id") for _, v in vals("P106") if isinstance(v, dict)}
    imgs = [v for rank, v in sorted(vals("P18"), key=lambda t: t[0] != "preferred") if isinstance(v, str)]
    return dict(label=(e.get("labels", {}).get("en") or {}).get("value", ""), footballer=FOOTBALLER in occ,
                countries=[v.get("id") for _, v in vals("P27") if isinstance(v, dict)], image=imgs[0] if imgs else None,
                enwiki=((e.get("sitelinks") or {}).get("enwiki") or {}).get("title", ""))


def entities(ids: list[str], cache: dict) -> None:
    todo = [i for i in ids if i not in cache]
    for k in range(0, len(todo), 50):
        chunk = todo[k:k + 50]
        res = fetch(WD, dict(action="wbgetentities", ids="|".join(chunk), props="claims|labels|sitelinks", sitefilter="enwiki", languages="en", format="json"))
        for qid, e in ((res or {}).get("entities") or {}).items():
            cache[qid] = parse_entity(e)


def entities_by_title(titles: list[str], cache: dict, done: set) -> None:
    """Resolve English Wikipedia titles to Wikidata items, 50 per request (about 50x fewer requests than searching one name at a time)."""
    todo = [t for t in titles if t not in done]
    for k in range(0, len(todo), 50):
        chunk = todo[k:k + 50]
        res = fetch(WD, dict(action="wbgetentities", sites="enwiki", titles="|".join(chunk), props="claims|labels|sitelinks", sitefilter="enwiki",
                             languages="en", format="json"))
        for qid, e in ((res or {}).get("entities") or {}).items():
            if qid.startswith("Q"):
                cache[qid] = parse_entity(e)
        done.update(chunk)
        if (k // 50) % 10 == 9:
            print(f"  resolved {min(k + 50, len(todo))}/{len(todo)} titles", flush=True)


def country_labels(qids: set[str], cache: dict) -> None:
    todo = [q for q in qids if q not in cache]
    for k in range(0, len(todo), 50):
        res = fetch(WD, dict(action="wbgetentities", ids="|".join(todo[k:k + 50]), props="labels", languages="en", format="json"))
        for q, e in ((res or {}).get("entities") or {}).items():
            cache[q] = (e.get("labels", {}).get("en") or {}).get("value", "")


def pick(player: str, full_name: str, country: str, hits_by_name: dict[str, list[dict]], ents: dict, clabels: dict) -> str | None:
    want_names = {norm(player), norm(full_name)}
    ok_countries = {norm(country)} | COUNTRY_ALIASES.get(norm(country), set())
    cands = {}
    for hits in hits_by_name.values():
        for h in hits:
            e = ents.get(h["id"])
            if e and e["footballer"] and e["image"]:
                cands[h["id"]] = (h, e)
    if not cands:
        return None
    scored = []
    for qid, (h, e) in cands.items():
        country_ok = any(norm(clabels.get(c, "")) in ok_countries for c in e["countries"])
        name_ok = norm(e["label"]) in want_names or norm(h["label"]) in want_names
        scored.append((country_ok, name_ok, qid))
    exact = [s for s in scored if s[0] and s[1]]
    if len(exact) == 1:
        return exact[0][2]
    if len(exact) > 1:
        return None  # two footballers of the same name and nationality: do not guess
    by_country = [s for s in scored if s[0]]
    if len(by_country) == 1:
        return by_country[0][2]
    only_name = [s for s in scored if s[1]]
    if len(only_name) == 1 and len(scored) == 1:  # a single footballer by that name, nationality missing on Wikidata
        return only_name[0][2]
    return None


def commons_info(filenames: list[str], cache: dict) -> None:
    todo = [f for f in dict.fromkeys(filenames) if f not in cache]
    for k in range(0, len(todo), 40):
        chunk = todo[k:k + 40]
        res = fetch(COMMONS, dict(action="query", titles="|".join("File:" + f for f in chunk), prop="imageinfo", iiprop="url|extmetadata|size|mime",
                                  iiurlwidth=330, format="json", redirects=1))
        pages = ((res or {}).get("query") or {}).get("pages", {})
        redirects = {r["to"]: r["from"] for r in ((res or {}).get("query") or {}).get("redirects", [])}
        norm_map = {n["to"]: n["from"] for n in ((res or {}).get("query") or {}).get("normalized", [])}
        for page in pages.values():
            ii = (page.get("imageinfo") or [{}])[0]
            title = page.get("title", "")
            original = redirects.get(title, title)
            original = norm_map.get(original, original).removeprefix("File:")
            md = ii.get("extmetadata", {})
            lic = (md.get("LicenseShortName") or {}).get("value", "")
            artist = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", (md.get("Artist") or {}).get("value", "")))).strip()
            info = dict(title=title.removeprefix("File:"), license=lic, author=artist or "unknown", page=ii.get("descriptionurl", ""),
                        thumb=ii.get("thumburl"), mime=ii.get("mime", ""), width=ii.get("thumbwidth"))
            for f in chunk:
                if f.replace("_", " ") in (original.replace("_", " "), title.removeprefix("File:")) or f == original:
                    cache[f] = info


_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def face_crop(jpg: bytes, size: int = 112):
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    faces = _cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(max(30, w // 14), max(30, w // 14)))
    if len(faces) == 0:
        return None
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    if fw < 0.07 * w:  # a tiny face in a wide action shot would be a blurry smudge at this size
        return None
    side = int(min(min(w, h), fw * 2.3))
    cx, cy = x + fw / 2, y + fh / 2 + fh * 0.1
    x0 = int(np.clip(cx - side / 2, 0, w - side))
    y0 = int(np.clip(cy - side / 2, 0, h - side))
    crop = cv2.resize(img[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes() if ok else None


YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
_yunet = [None]


def crop_ok(jpg: bytes) -> bool:
    """Second opinion on a finished crop: a modern face detector (YuNet) must find a clear, centred face.

    The classic Haar detector used for cropping sometimes locks onto a shirt, a knee or a trophy. A rejected player simply has no
    photo (initials are shown). The check does not look at skin tone, only at whether a face is there.
    """
    if _yunet[0] is None:
        model = CACHE.parent / "models" / "yunet.onnx"
        if not model.exists():
            model.parent.mkdir(parents=True, exist_ok=True)
            data = fetch(YUNET_URL, binary=True)
            if not data:
                return True  # cannot check: keep the crop rather than lose it
            model.write_bytes(data)
        _yunet[0] = cv2.FaceDetectorYN.create(str(model), "", (224, 224), score_threshold=0.6, nms_threshold=0.3)
    img = cv2.resize(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR), (224, 224), interpolation=cv2.INTER_CUBIC)
    _, faces = _yunet[0].detect(img)
    for f in (faces if faces is not None else []):
        x, y, w, h = f[:4]
        if w >= 0.30 * 224 and abs(x + w / 2 - 112) < 56 and abs(y + h / 2 - 120) < 60:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-minutes", type=float, default=270)
    ap.add_argument("--limit", type=int, default=0, help="only the first N players (by minutes), for a trial")
    args = ap.parse_args()

    players = pd.read_csv(PUBLIC / "players.csv.gz")
    mins = pd.read_csv(PUBLIC / "appearances.csv.gz").groupby("player_id").minutes.sum()
    players = players.assign(minutes=players.player_id.map(mins).fillna(0))
    players = players[players.minutes >= args.min_minutes].sort_values("minutes", ascending=False)
    if args.limit:
        players = players.head(args.limit)
    print(f"{len(players)} players to look up")

    scache, ecache, ccache, icache = load_cache("search.json"), load_cache("entities.json"), load_cache("countries.json"), load_cache("commons.json")
    tcache = load_cache("titles_done.json")
    done_titles = set(tcache.get("done", []))

    # Pass 1: batch-resolve every name (and its "(footballer)" disambiguated form) as an English Wikipedia title.
    variants = {}
    for r in players.itertuples():
        for nm in dict.fromkeys([r.player, r.full_name]):
            variants.setdefault(r.player_id, []).extend([nm, f"{nm} (footballer)"])
    all_titles = sorted({t for v in variants.values() for t in v})
    print(f"pass 1: {len(all_titles)} candidate Wikipedia titles")
    entities_by_title(all_titles, ecache, done_titles)
    save_cache("entities.json", ecache)
    save_cache("titles_done.json", {"done": sorted(done_titles)})
    index: dict[str, list[str]] = {}
    for qid, e in ecache.items():
        if e["footballer"]:
            for name in {e["label"], re.sub(r"\s*\(.*\)$", "", e.get("enwiki", ""))}:
                if name:
                    index.setdefault(norm(name), []).append(qid)

    def hits_from_index(r) -> dict[str, list[dict]]:
        out = {}
        for nm in dict.fromkeys([r.player, r.full_name]):
            qs = list(dict.fromkeys(index.get(norm(nm), [])))
            if qs:
                out[nm] = [dict(id=q, label=ecache[q]["label"], desc="") for q in qs]
        return out

    hits: dict[int, dict[str, list[dict]]] = {r.player_id: hits_from_index(r) for r in players.itertuples()}
    country_labels({c for e in ecache.values() for c in e.get("countries", [])}, ccache)
    unresolved = [r for r in players.itertuples() if not pick(r.player, r.full_name, r.country, hits[r.player_id], ecache, ccache)]
    print(f"pass 1 resolved {len(players) - len(unresolved)} of {len(players)}; pass 2 searches the other {len(unresolved)}")

    # Pass 2: the slow name search, only for the players pass 1 could not settle.
    for n, r in enumerate(unresolved, 1):
        found = dict(hits[r.player_id])
        for nm in dict.fromkeys([r.player, r.full_name]):
            res = [h for h in search(nm, scache) if "footbal" in h["desc"].lower()]
            if res:
                found[nm] = found.get(nm, []) + res
                break
        hits[r.player_id] = found
        if n % 100 == 0:
            save_cache("search.json", scache)
            print(f"  searched {n}/{len(unresolved)}", flush=True)
    save_cache("search.json", scache)
    qids = sorted({h["id"] for f in hits.values() for hs in f.values() for h in hs})
    entities(qids, ecache)
    save_cache("entities.json", ecache)
    country_labels({c for q in qids for c in (ecache.get(q) or {}).get("countries", [])}, ccache)
    save_cache("countries.json", ccache)

    chosen: dict[int, tuple[str, str]] = {}
    for r in players.itertuples():
        qid = pick(r.player, r.full_name, r.country, hits.get(r.player_id, {}), ecache, ccache)
        if qid:
            chosen[r.player_id] = (qid, ecache[qid]["image"])
    print(f"matched a footballer with a photo: {len(chosen)} of {len(players)}")

    commons_info([f for _, f in chosen.values()], icache)
    save_cache("commons.json", icache)

    OUT.mkdir(parents=True, exist_ok=True)
    names = dict(zip(players.player_id, players.player))
    credits: dict[str, dict] = {}
    skipped = dict(licence=0, no_thumb=0, no_face=0, download=0, low_quality=0)
    rejected = set(load_cache("rejected.json").get("ids", []))  # crops that failed the YuNet check: never fetch them again
    for n, (pid, (qid, fname)) in enumerate(chosen.items(), 1):
        info = icache.get(fname)
        if not info or not info.get("thumb") or not info.get("mime", "").startswith("image/"):
            skipped["no_thumb"] += 1
            continue
        if not FREE_LICENCE.match(info["license"] or ""):
            skipped["licence"] += 1
            continue
        if pid in rejected:
            skipped["low_quality"] += 1
            continue
        dest = OUT / f"{pid}.jpg"
        jpg = None
        if dest.exists():  # resumed run: re-check what is already on disk
            if not crop_ok(dest.read_bytes()):
                dest.unlink()
                rejected.add(pid)
                skipped["low_quality"] += 1
                continue
        else:
            raw = fetch(info["thumb"], binary=True)
            if not raw:
                skipped["download"] += 1
                continue
            jpg = face_crop(raw)
            if jpg is None:
                skipped["no_face"] += 1
                continue
            if not crop_ok(jpg):
                rejected.add(pid)
                skipped["low_quality"] += 1
                continue
            dest.write_bytes(jpg)
        credits[str(pid)] = dict(player=names.get(pid, ""), title=info["title"], author=info["author"], license=info["license"], page=info["page"], wikidata=qid)
        if n % 25 == 0:  # save as we go: the credits stay in step with the photos, and a crash loses nothing
            (PUBLIC / "player_photos.json").write_text(json.dumps(credits, ensure_ascii=False, indent=1))
        if n % 50 == 0:
            print(f"  photos {n}/{len(chosen)}  kept {len(credits)}", flush=True)
    save_cache("rejected.json", {"ids": sorted(rejected)})
    (PUBLIC / "player_photos.json").write_text(json.dumps(credits, ensure_ascii=False, indent=1))
    print(f"done: {len(credits)} photos kept | skipped: {skipped}")


if __name__ == "__main__":
    main()
