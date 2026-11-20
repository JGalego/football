"""Turn the raw downloads in data/raw/ into the compact files the notebook reads from public/.

Run `python scripts/download_data.py` first, then `python scripts/prepare_data.py`.

Conventions used everywhere in the notebook:
  * Pitch is 105 x 68 metres, origin at the bottom-left corner, y pointing up.
  * Event data is oriented so the team performing the action attacks left -> right
    (StatsBomb already does this; we only rescale 120x80 yards and flip y).
  * Tracking data (Metrica) is oriented so the HOME team always attacks left -> right.
"""

from __future__ import annotations

import gzip
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from _progress import Progress, human_time, line_buffered  # sibling module in scripts/

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SB = RAW / "statsbomb"
OUT = ROOT / "public"
OUT.mkdir(exist_ok=True)

L, W = 105.0, 68.0  # metric pitch
GOAL_Y = W / 2
POST_HALF = 7.32 / 2

COMPS = {
    (43, 106): ("wc22", "Men's World Cup 2022"),
    (43, 3): ("wc18", "Men's World Cup 2018"),
    (55, 282): ("eu24", "Men's Euro 2024"),
    (55, 43): ("eu20", "Men's Euro 2020"),
    (223, 282): ("ca24", "Copa América 2024"),
    (1267, 107): ("afc23", "Africa Cup of Nations 2023"),
    (72, 107): ("wwc23", "Women's World Cup 2023"),
    (72, 30): ("wwc19", "Women's World Cup 2019"),
    (53, 106): ("weu22", "Women's Euro 2022"),
    (53, 315): ("weu25", "Women's Euro 2025"),
    (9, 281): ("b2324", "Bundesliga 2023/24 (Leverkusen)"),
}


# ----------------------------------------------------------------------------- helpers
# StatsBomb uses a 120 x 80 yard pitch (y pointing down). A plain rescale would misplace
# the box and goal by up to a metre, so we map piecewise-linearly between pitch markings
# (the same idea as mplsoccer's Standardizer).
SB_X = [0, 6, 12, 18, 60, 102, 108, 114, 120]
M_X = [0, 5.5, 11, 16.5, 52.5, 88.5, 94, 99.5, 105]
SB_Y = [0, 18, 30, 36, 40, 44, 50, 62, 80]
M_Y = [68, 54.16, 43.16, 37.66, 34, 30.34, 24.84, 13.84, 0]
SB_GOAL_H, M_GOAL_H = 2.67, 2.44  # crossbar height in yards / metres


def sx(x: float) -> float:
    return round(float(np.interp(x, SB_X, M_X)), 2)


def sy(y: float) -> float:
    return round(float(np.interp(y, SB_Y, M_Y)), 2)


def loc(p):
    if not p:
        return (np.nan, np.nan)
    return (sx(p[0]), sy(p[1]))


def name(d, key="name"):
    return d.get(key) if isinstance(d, dict) else None


def clock_to_min(s: str | None) -> float | None:
    if s is None:
        return None
    m, sec = s.split(":")
    return int(m) + int(sec) / 60.0


def in_triangle(p, a, b, c) -> bool:
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1, d2, d3 = sign(p, a, b), sign(p, b, c), sign(p, c, a)
    neg = d1 < 0 or d2 < 0 or d3 < 0
    pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (neg and pos)


def write_csv(df: pd.DataFrame, fname: str) -> None:
    path = OUT / fname
    if fname.endswith(".gz"):
        df.to_csv(path, index=False, compression={"method": "gzip", "compresslevel": 9, "mtime": 0})
    else:
        df.to_csv(path, index=False)
    print(f"  wrote {fname:38s} {len(df):>9,} rows  {path.stat().st_size / 1e6:6.2f} MB")


# ----------------------------------------------------------------------------- StatsBomb
SPADL_PASS = {"Corner": "corner", "Free Kick": "freekick", "Throw-in": "throw_in", "Goal Kick": "goal_kick"}
DEF_SUCCESS = {"Won", "Success In Play", "Success Out", "Success"}


def to_actions(ev: dict) -> dict | None:
    """Map one StatsBomb event to a SPADL-like action (Decroos et al., 2019), or None."""
    t = ev["type"]["name"]
    x, y = loc(ev.get("location"))
    a = dict(type=None, result=1, ex=x, ey=y, body="foot", recipient=None)
    if t == "Pass":
        p = ev["pass"]
        ptype = name(p.get("type"))
        a["type"] = SPADL_PASS.get(ptype, "cross" if p.get("cross") else "pass")
        a["result"] = 0 if p.get("outcome") else 1
        a["ex"], a["ey"] = loc(p.get("end_location"))
        bp = name(p.get("body_part")) or ""
        a["body"] = "head" if "Head" in bp else ("other" if bp and "Foot" not in bp else "foot")
        a["recipient"] = (p.get("recipient") or {}).get("id")
        a["assist"] = 1 if p.get("goal_assist") else (2 if p.get("shot_assist") else 0)
    elif t == "Carry":
        a["type"] = "carry"
        a["ex"], a["ey"] = loc(ev["carry"].get("end_location"))
    elif t == "Dribble":
        a["type"] = "take_on"
        a["result"] = 1 if name(ev["dribble"].get("outcome")) == "Complete" else 0
    elif t == "Shot":
        s = ev["shot"]
        st = name(s.get("type"))
        a["type"] = "penalty" if st == "Penalty" else ("shot_freekick" if st == "Free Kick" else "shot")
        a["result"] = 1 if name(s.get("outcome")) == "Goal" else 0
        a["ex"], a["ey"] = loc(s.get("end_location"))
        bp = name(s.get("body_part")) or ""
        a["body"] = "head" if bp == "Head" else ("other" if bp == "Other" else "foot")
        a["xg"] = round(s.get("statsbomb_xg", np.nan), 4)
    elif t == "Duel" and name(ev["duel"].get("type")) == "Tackle":
        a["type"] = "tackle"
        a["result"] = 1 if name(ev["duel"].get("outcome")) in DEF_SUCCESS else 0
    elif t == "Interception":
        a["type"] = "interception"
        a["result"] = 1 if name(ev["interception"].get("outcome")) in DEF_SUCCESS else 0
    elif t == "Clearance":
        a["type"] = "clearance"
    elif t == "Ball Recovery":
        a["type"] = "recovery"
        a["result"] = 0 if ev.get("ball_recovery", {}).get("recovery_failure") else 1
    elif t == "Foul Committed":
        a["type"] = "foul"
        a["result"] = 0
    elif t == "Miscontrol":
        a["type"] = "bad_touch"
        a["result"] = 0
    elif t == "Dispossessed":
        a["type"] = "dispossessed"
        a["result"] = 0
    elif t == "Goal Keeper":
        gk = name(ev["goalkeeper"].get("type")) or ""
        if gk in {"Shot Saved", "Save", "Saved To Post", "Shot Saved Off Target", "Shot Saved to Post"}:
            a["type"] = "keeper_save"
        elif gk in {"Collected", "Punch", "Smother", "Keeper Sweeper"}:
            a["type"] = "keeper_claim"
        else:
            return None
    elif t == "Pressure":
        a["type"] = "pressure"
    elif t == "Block":
        a["type"] = "block"
    elif t == "Own Goal Against":
        a["type"] = "own_goal"
        a["result"] = 0
    else:
        return None
    return a


def shot_features(ev: dict) -> dict:
    s = ev["shot"]
    x, y = loc(ev["location"])
    post_a, post_b = (L, GOAL_Y - POST_HALF), (L, GOAL_Y + POST_HALF)
    dx, dy = L - x, GOAL_Y - y
    dist = math.hypot(dx, dy)
    # angle subtended by the goal mouth, in degrees
    a1 = math.atan2(post_a[1] - y, post_a[0] - x)
    a2 = math.atan2(post_b[1] - y, post_b[0] - x)
    angle = abs(math.degrees(a2 - a1))
    ff = s.get("freeze_frame") or []
    opp = [(loc(p["location"]), (p.get("position") or {}).get("name") == "Goalkeeper") for p in ff if not p.get("teammate")]
    mates = [loc(p["location"]) for p in ff if p.get("teammate")]
    tri = ((x, y), post_a, post_b)
    n_def_cone = sum(1 for (p, gk) in opp if not gk and in_triangle(p, *tri))
    n_mate_cone = sum(1 for p in mates if in_triangle(p, *tri))
    gk = [p for (p, is_gk) in opp if is_gk]
    if gk:
        g = gk[0]
        gk_dist_goal = math.hypot(L - g[0], GOAL_Y - g[1])
        gk_dist_shooter = math.hypot(g[0] - x, g[1] - y)
        gk_in_cone = int(in_triangle(g, *tri))
    else:
        gk_dist_goal, gk_dist_shooter, gk_in_cone = np.nan, np.nan, 0
    dists = [math.hypot(p[0] - x, p[1] - y) for (p, is_gk) in opp if not is_gk]
    end = s.get("end_location") or [np.nan, np.nan, np.nan]
    return dict(
        x=x,
        y=y,
        distance=round(dist, 2),
        angle=round(angle, 2),
        body_part=name(s.get("body_part")),
        shot_type=name(s.get("type")),
        technique=name(s.get("technique")),
        play_pattern=name(ev.get("play_pattern")),
        first_time=int(bool(s.get("first_time"))),
        under_pressure=int(bool(ev.get("under_pressure"))),
        one_on_one=int(bool(s.get("one_on_one"))),
        open_goal=int(bool(s.get("open_goal"))),
        follows_dribble=int(bool(s.get("follows_dribble"))),
        n_def_cone=n_def_cone,
        n_mate_cone=n_mate_cone,
        gk_in_cone=gk_in_cone,
        gk_dist_goal=round(gk_dist_goal, 2) if gk else np.nan,
        gk_dist_shooter=round(gk_dist_shooter, 2) if gk else np.nan,
        nearest_def=round(min(dists), 2) if dists else np.nan,
        n_def_5m=sum(1 for d in dists if d < 5),
        has_ff=int(bool(ff)),
        end_x=sx(end[0]) if len(end) > 0 and end[0] == end[0] else np.nan,
        end_y=sy(end[1]) if len(end) > 1 and end[1] == end[1] else np.nan,
        end_z=round(end[2] * M_GOAL_H / SB_GOAL_H, 2) if len(end) > 2 and end[2] is not None else np.nan,
        sb_xg=round(s.get("statsbomb_xg", np.nan), 4),
        outcome=name(s.get("outcome")),
        goal=int(name(s.get("outcome")) == "Goal"),
    )


def read_360(mid: int) -> list | None:
    """The 360 freeze frames of a match, or None when there are none or the file is unreadable.

    StatsBomb's open-data repository has at least one malformed 360 file (three-sixty/3845506.json), so a bad file must not stop the build.
    """
    tfile = SB / "three-sixty" / f"{mid}.json"
    if not tfile.exists():
        return None
    try:
        return json.loads(tfile.read_text())
    except ValueError as exc:
        print(f"  ! skipping the unreadable 360 file for match {mid}: {exc}", flush=True)
        return None


def prepare_statsbomb() -> None:
    print("StatsBomb ...", flush=True)
    matches, teams, players, appearances, shots, frames = [], {}, {}, [], [], []
    total = sum(len(json.loads((SB / "matches" / str(cid) / f"{sid}.json").read_text())) for (cid, sid) in COMPS if (SB / "matches" / str(cid) / f"{sid}.json").exists())
    progress = Progress("  Parsing matches", total, every=5.0)
    actions_by_comp: dict[str, list] = defaultdict(list)
    final_id = None

    for (cid, sid), (key, label) in COMPS.items():
        mfile = SB / "matches" / str(cid) / f"{sid}.json"
        if not mfile.exists():
            print(f"  ! missing {mfile}")
            continue
        for m in json.loads(mfile.read_text()):
            mid = m["match_id"]
            efile, lfile = SB / "events" / f"{mid}.json", SB / "lineups" / f"{mid}.json"
            progress.update(note=f"{label}: {m['home_team']['home_team_name']} v {m['away_team']['away_team_name']}")
            if not efile.exists() or not lfile.exists():
                continue
            frames360 = read_360(mid)
            gender = "women" if "Women" in m["competition"]["competition_name"] else "men"
            stage = (m.get("competition_stage") or {}).get("name")
            home, away = m["home_team"], m["away_team"]
            matches.append(
                dict(
                    match_id=mid,
                    comp=key,
                    competition=label,
                    gender=gender,
                    date=m["match_date"],
                    stage=stage,
                    home_id=home["home_team_id"],
                    away_id=away["away_team_id"],
                    home=home["home_team_name"],
                    away=away["away_team_name"],
                    home_score=m["home_score"],
                    away_score=m["away_score"],
                    has360=int(frames360 is not None),
                )
            )
            teams[home["home_team_id"]] = (home["home_team_name"], gender)
            teams[away["away_team_id"]] = (away["away_team_name"], gender)
            if key == "wc22" and stage == "Final":
                final_id = mid

            events = json.loads(efile.read_text())
            end_min = max(e["minute"] + e["second"] / 60 for e in events)
            matches[-1]["n_events"] = len(events)
            if frames360 is not None:
                matches[-1]["n_360_frames"] = len(frames360)
                matches[-1]["n_360_players"] = sum(len(f.get("freeze_frame") or []) for f in frames360)

            for tl in json.loads(lfile.read_text()):
                for p in tl["lineup"]:
                    players[p["player_id"]] = (p.get("player_nickname") or p["player_name"], p["player_name"], (p.get("country") or {}).get("name"))
                    mins, pos_mins = 0.0, defaultdict(float)
                    started = 0
                    for pos in p.get("positions", []):
                        a = clock_to_min(pos["from"]) or 0.0
                        b = clock_to_min(pos["to"]) if pos["to"] else end_min
                        mins += max(0.0, b - a)
                        pos_mins[pos["position"]] += max(0.0, b - a)
                        if pos.get("start_reason") == "Starting XI":
                            started = 1
                    if mins > 0:
                        appearances.append(
                            dict(
                                match_id=mid,
                                team_id=tl["team_id"],
                                player_id=p["player_id"],
                                minutes=round(mins, 1),
                                started=started,
                                position=max(pos_mins, key=pos_mins.get),
                                jersey=p.get("jersey_number"),
                            )
                        )

            for ev in events:
                a = to_actions(ev)
                if a is not None:
                    x, y = loc(ev.get("location"))
                    actions_by_comp[key].append(
                        dict(
                            match_id=mid,
                            idx=ev["index"],
                            period=ev["period"],
                            minute=ev["minute"],
                            second=ev["second"],
                            possession=ev.get("possession"),
                            team_id=ev["team"]["id"],
                            player_id=(ev.get("player") or {}).get("id"),
                            type=a["type"],
                            result=a["result"],
                            body=a["body"],
                            x=x,
                            y=y,
                            ex=a["ex"],
                            ey=a["ey"],
                            under_pressure=int(bool(ev.get("under_pressure"))),
                            recipient_id=a.get("recipient"),
                            assist=a.get("assist", 0),
                            xg=a.get("xg", np.nan),
                        )
                    )
                if ev["type"]["name"] == "Shot":
                    f = shot_features(ev)
                    shots.append(
                        dict(
                            shot_id=ev["id"],
                            match_id=mid,
                            comp=key,
                            gender=gender,
                            team_id=ev["team"]["id"],
                            player_id=ev["player"]["id"],
                            period=ev["period"],
                            minute=ev["minute"],
                            second=ev["second"],
                            **f,
                        )
                    )
                    if key in {"wc22", "wwc23", "eu24", "weu25"}:
                        for p in ev["shot"].get("freeze_frame") or []:
                            px, py = loc(p["location"])
                            frames.append(
                                dict(
                                    shot_id=ev["id"],
                                    x=px,
                                    y=py,
                                    teammate=int(bool(p.get("teammate"))),
                                    keeper=int((p.get("position") or {}).get("name") == "Goalkeeper"),
                                    player_id=(p.get("player") or {}).get("id"),
                                )
                            )

    matches_df = pd.DataFrame(matches).sort_values(["comp", "date"])
    write_csv(matches_df, "matches.csv")
    write_csv(
        pd.DataFrame([(k, v[0], v[1]) for k, v in teams.items()], columns=["team_id", "team", "gender"]).sort_values("team"),
        "teams.csv",
    )
    write_csv(
        pd.DataFrame([(k, *v) for k, v in players.items()], columns=["player_id", "player", "full_name", "country"]),
        "players.csv.gz",
    )
    print("  Writing the compact files ...", flush=True)
    write_csv(pd.DataFrame(appearances), "appearances.csv.gz")
    write_csv(pd.DataFrame(shots), "shots.csv.gz")
    write_csv(pd.DataFrame(frames), "shot_frames.csv.gz")
    for key, rows in actions_by_comp.items():
        df = pd.DataFrame(rows)
        for c in ("player_id", "recipient_id", "possession"):
            df[c] = df[c].astype("Int64")
        write_csv(df, f"actions_{key}.csv.gz")

    # Raw JSON for one iconic match, so the notebook can show the real thing.
    if final_id:
        for sub, fname in (("events", "final_events.json.gz"), ("three-sixty", "final_360.json.gz")):
            src = SB / sub / f"{final_id}.json"
            if src.exists():
                with src.open("rb") as fi, gzip.GzipFile(OUT / fname, "wb", compresslevel=9, mtime=0) as fo:
                    shutil.copyfileobj(fi, fo)
                print(f"  wrote {fname:38s} {(OUT / fname).stat().st_size / 1e6:6.2f} MB")


# ----------------------------------------------------------------------------- Metrica
def read_metrica_tracking(path: Path, team: str) -> pd.DataFrame:
    with path.open() as fh:
        fh.readline()
        jerseys = fh.readline().strip().split(",")
        header = fh.readline().strip().split(",")
    cols = ["period", "frame", "time"]
    for i, h in enumerate(header[3:], start=3):
        if h.startswith("Player"):
            n = jerseys[i]
            cols += [f"{team}{n}_x", f"{team}{n}_y"]
        elif h == "Ball":
            cols += ["ball_x", "ball_y"]
    df = pd.read_csv(path, skiprows=3, header=None)
    df.columns = cols[: df.shape[1]]
    return df


def prepare_metrica(game: int = 2, step: int = 5) -> None:
    print(f"Metrica sample game {game} ...")
    mdir = RAW / "metrica"
    home = read_metrica_tracking(mdir / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv", "H")
    away = read_metrica_tracking(mdir / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv", "A")
    df = home.merge(away.drop(columns=["period", "time", "ball_x", "ball_y"]), on="frame")
    xcols = [c for c in df.columns if c.endswith("_x")]
    ycols = [c for c in df.columns if c.endswith("_y")]
    df[xcols] = df[xcols] * L
    df[ycols] = (1 - df[ycols]) * W

    # Orient so the home team attacks left -> right in both halves:
    # in each period, the home goalkeeper (first home player listed) should sit at small x.
    home_cols = [c[:-2] for c in xcols if c.startswith("H")]
    gk = home_cols[0]
    flip = {}
    for p in (1, 2):
        flip[p] = df.loc[df.period == p, f"{gk}_x"].mean() > L / 2
        if flip[p]:
            m = df.period == p
            df.loc[m, xcols] = L - df.loc[m, xcols]
            df.loc[m, ycols] = W - df.loc[m, ycols]

    out = df.iloc[::step].copy()
    out[xcols + ycols] = out[xcols + ycols].round(2)
    out["time"] = out["time"].round(2)
    write_csv(out, f"tracking_g{game}_5hz.csv.gz")

    ev = pd.read_csv(mdir / f"Sample_Game_{game}_RawEventsData.csv")
    ev.columns = ["team", "type", "subtype", "period", "start_frame", "start_time", "end_frame", "end_time", "from", "to", "x", "y", "ex", "ey"]
    ev["x"], ev["ex"] = ev["x"] * L, ev["ex"] * L
    ev["y"], ev["ey"] = (1 - ev["y"]) * W, (1 - ev["ey"]) * W
    for p in (1, 2):
        if flip[p]:
            m = ev.period == p
            ev.loc[m, ["x", "ex"]] = L - ev.loc[m, ["x", "ex"]]
            ev.loc[m, ["y", "ey"]] = W - ev.loc[m, ["y", "ey"]]
    ev[["x", "y", "ex", "ey"]] = ev[["x", "y", "ex", "ey"]].round(2)
    write_csv(ev, f"tracking_g{game}_events.csv")


# ----------------------------------------------------------------------------- International results
def prepare_international() -> None:
    print("International results ...")
    idir = RAW / "international"
    res = pd.read_csv(idir / "results.csv")
    write_csv(res, "intl_results.csv.gz")
    so = pd.read_csv(idir / "shootouts.csv")
    write_csv(so, "intl_shootouts.csv")


# ----------------------------------------------------------------------------- Transfermarkt
def prepare_transfermarkt() -> None:
    """Aggregate every Transfermarkt valuation into an age curve (CC0 source, tiny output)."""
    print("Transfermarkt ...")
    tdir = RAW / "transfermarkt"
    players = pd.read_csv(tdir / "players.csv.gz", usecols=["player_id", "date_of_birth", "position"])
    vals = pd.read_csv(tdir / "player_valuations.csv.gz", usecols=["player_id", "date", "market_value_in_eur"])
    d = vals.merge(players, on="player_id").dropna(subset=["date_of_birth", "market_value_in_eur"])
    age = (pd.to_datetime(d["date"]) - pd.to_datetime(d["date_of_birth"])).dt.days / 365.25
    d = d.assign(age=age.round().astype("Int64"), value=d.market_value_in_eur / 1e6)
    d = d[d.age.between(16, 38) & (d.value > 0)]
    rows = []
    for pos, g in [("All", d)] + list(d.groupby("position")):
        if pos == "Missing":
            continue
        q = g.groupby("age").value.quantile([0.25, 0.5, 0.75, 0.95]).unstack()
        q.columns = ["p25", "p50", "p75", "p95"]
        q["n"] = g.groupby("age").size()
        rows.append(q.reset_index().assign(position=pos))
    write_csv(pd.concat(rows).round(4), "tm_age_curve.csv")


# ----------------------------------------------------------------------------- Media (photo, audio, video)
MEDIA_SRC = ROOT / "data" / "raw" / "media"


def _ffmpeg(*args: str) -> None:
    import subprocess

    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def _gray_frames(src: Path, fps: int, w: int, h: int) -> np.ndarray:
    import subprocess

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-vf", f"fps={fps},scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w)


def prepare_media() -> None:
    """Shrink the openly licensed Wikimedia Commons media (see public/media.json) into analysis-ready files. Needs ffmpeg and Pillow."""
    print("Media ...")
    out = OUT / "media"
    out.mkdir(exist_ok=True)
    from PIL import Image

    photo = Image.open(MEDIA_SRC / "photo_france_japan_2012_1280.jpg").convert("RGB")
    photo.resize((1280, round(photo.height * 1280 / photo.width)), Image.LANCZOS).save(out / "match_photo.jpg", quality=88)
    for name, src, secs in (("crowd_frosinone", "audio_frosinone_chant.ogg", 45), ("crowd_strasbourg", "audio_strasbourg_vive_le_vent.ogg", 38)):
        _ffmpeg("-i", str(MEDIA_SRC / src), "-t", str(secs), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out / f"{name}.wav"))
    lat, rep_clip = MEDIA_SRC / "video_latvia_gibraltar_2026.webm", MEDIA_SRC / "video_spring_cup_2026.webm"
    _ffmpeg("-i", str(lat), "-vf", "scale=640:-2", "-c:v", "libvpx-vp9", "-crf", "40", "-b:v", "0", "-c:a", "libopus", "-b:a", "48k", str(out / "stand_clip.webm"))
    _ffmpeg("-i", str(lat), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out / "stand_clip_audio.wav"))
    _ffmpeg("-i", str(rep_clip), "-vf", "scale=480:-2", "-c:v", "libvpx-vp9", "-crf", "44", "-b:v", "0", "-c:a", "libopus", "-b:a", "24k", str(out / "match_report.webm"))  # small: marimo caps a cell output at 5 MB
    np.savez_compressed(out / "stand_clip_frames.npz", frames=_gray_frames(lat, 10, 128, 72), fps=10)
    np.savez_compressed(out / "match_report_frames.npz", frames=_gray_frames(rep_clip, 12, 128, 72), fps=12)
    for f in sorted(out.iterdir()):
        print(f"  wrote media/{f.name:28s} {f.stat().st_size / 1e6:6.2f} MB")


# ----------------------------------------------------------------------------- Aggregate counts
def prepare_stats() -> None:
    """Headline counts for the notebook's first screen and data-stack chart. Aggregates only, so the file is safe to publish."""
    print("Stats ...")
    matches = pd.read_csv(OUT / "matches.csv")
    shots = pd.read_csv(OUT / "shots.csv.gz", usecols=["goal"])
    trk = pd.read_csv(OUT / "tracking_g2_5hz.csv.gz")
    intl = pd.read_csv(OUT / "intl_results.csv.gz", usecols=["date"])
    positions = int(trk.filter(regex=r"^[HA]\d+_x$").notna().sum(axis=1).add(1).sum() * 5)  # 5 Hz sample -> 25 Hz
    stats = dict(
        n_matches=len(matches), n_competitions=int(matches.comp.nunique()), n_events=int(matches.n_events.sum()),
        n_360_players=int(matches.n_360_players.sum()), n_shots=len(shots), n_goals=int(shots.goal.sum()), n_intl=len(intl),
        events_per_match=float(matches.n_events.mean()), freeze_players_per_match=float(matches.loc[matches.has360 == 1, "n_360_players"].mean()),
        tracking_positions_per_match=positions,
    )
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"  wrote stats.json  {stats}")


STAGES = {
    "statsbomb": prepare_statsbomb,
    "metrica": lambda: prepare_metrica(2),
    "international": prepare_international,
    "transfermarkt": prepare_transfermarkt,
    "media": prepare_media,
    "stats": prepare_stats,
}

if __name__ == "__main__":
    import sys
    import time

    line_buffered()
    todo = sys.argv[1:] or list(STAGES)
    t_all = time.monotonic()
    for i, stage in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {stage}", flush=True)
        t0 = time.monotonic()
        STAGES[stage]()
        print(f"[{i}/{len(todo)}] {stage} done in {human_time(time.monotonic() - t0)}", flush=True)
    print(f"All done in {human_time(time.monotonic() - t_all)}.", flush=True)
