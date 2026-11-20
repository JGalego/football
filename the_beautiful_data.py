# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair>=5.5",
#     "marimo>=0.25",
#     "networkx>=3.2",
#     "numpy>=1.26",
#     "pandas>=2.2",
#     "pillow>=10",
#     "plotly>=5.20",
#     "scikit-learn>=1.4",
#     "scipy>=1.12",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="The Beautiful Data", html_head_file="head.html")

with app.setup(hide_code=True):
    import base64
    import functools
    import hashlib
    import importlib
    import pickle
    from xml.sax.saxutils import escape as xml_escape
    import gzip
    import io
    import json
    import math
    import os
    import urllib.request
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import networkx as nx
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageDraw
    from scipy import ndimage as ndi
    from scipy.io import wavfile
    from scipy.signal.windows import tukey
    from scipy.spatial import ConvexHull
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, PoissonRegressor
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # ── Pitch: metres, origin bottom-left, y up. Event data: acting team attacks left → right.
    L, W = 105.0, 68.0

    # ── SPADL-style action families
    PASS_TYPES = ["pass", "cross", "corner", "freekick", "throw_in", "goal_kick"]
    MOVE_TYPES = PASS_TYPES + ["carry"]
    SHOT_TYPES = ["shot", "penalty", "shot_freekick"]

    # ── Design tokens: one validated palette, stepped separately for light and dark.
    DARK = mo.app_meta().theme == "dark"
    PALETTES = {
        "light": dict(
            surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
            axis="#c3c2b7", line="#c9c8c0", mid="#f0efec", fill="#efeee9",
            blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
            magenta="#e87ba4", green="#008300", violet="#6250d6", red="#e34948",
            seq=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
            seq2=["#fde3d6", "#f9bfa3", "#f39a70", "#eb6834", "#c94f1f", "#9c3b15", "#6e290e"],
        ),
        "dark": dict(
            surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
            axis="#383835", line="#4a4944", mid="#383835", fill="#262624",
            blue="#3987e5", orange="#d95926", aqua="#199e70", yellow="#c98500",
            magenta="#d55181", green="#0ca30c", violet="#9085e9", red="#e66767",
            seq=["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
            seq2=["#6e290e", "#9c3b15", "#c94f1f", "#eb6834", "#f39a70", "#f9bfa3", "#fde3d6"],
        ),
    }
    C = PALETTES["dark" if DARK else "light"]
    CAT = [C[k] for k in ("blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red")]

    @alt.theme.register("beautiful_data", enable=True)
    def beautiful_data_theme():
        return alt.theme.ThemeConfig(
            {
                "config": {
                    "background": C["surface"],
                    "font": "system-ui, -apple-system, 'Segoe UI', sans-serif",
                    "view": {"stroke": None},
                    "axis": {
                        "labelColor": C["muted"], "titleColor": C["ink2"], "gridColor": C["grid"],
                        "domainColor": C["axis"], "tickColor": C["axis"], "labelFontSize": 11,
                        "titleFontSize": 11, "titleFontWeight": "normal", "gridWidth": 1,
                    },
                    "legend": {
                        "labelColor": C["ink2"], "titleColor": C["ink2"], "titleFontWeight": "normal",
                        "labelFontSize": 11, "titleFontSize": 11,
                    },
                    "header": {"labelColor": C["ink2"], "titleColor": C["ink2"]},
                    "title": {
                        "color": C["ink"], "fontSize": 13, "anchor": "start", "fontWeight": 600,
                        "subtitleColor": C["muted"], "subtitleFontSize": 11, "offset": 8,
                    },
                    "text": {"color": C["ink2"], "fontSize": 11},
                    "range": {"category": CAT},
                    "line": {"strokeWidth": 2},
                    "rule": {"color": C["axis"]},
                }
            }
        )

    alt.data_transformers.disable_max_rows()

    # ── Data access: works from a local checkout and from a WebAssembly export.
    def load_bytes(name: str) -> bytes:
        location = mo.notebook_location() / "public" / name
        path = Path(str(location))
        if path.exists():
            return path.read_bytes()
        try:
            with urllib.request.urlopen(str(location)) as response:
                status, body = getattr(response, "status", 200), response.read()
            if status != 200:  # in the browser a missing file may come back as an error page instead of raising
                raise FileNotFoundError(name)
            return body
        except FileNotFoundError:
            raise
        except Exception:
            try:
                from js import XMLHttpRequest  # WebAssembly (Pyodide): synchronous binary XHR, allowed in marimo's web worker
            except ImportError:
                raise FileNotFoundError(name) from None
            req = XMLHttpRequest.new()
            req.open("GET", str(location), False)
            req.responseType = "arraybuffer"
            req.send(None)
            if req.status != 200:
                raise FileNotFoundError(name)
            return bytes(req.response.to_py())

    def has_file(name: str) -> bool:
        try:
            load_bytes(name)
            return True
        except Exception:
            return False

    # Full edition: the row-level StatsBomb / Metrica files are on disk (local run or the Hugging Face Space).
    # Lite edition (GitHub Pages): they are not, and the sections that need them show static snapshots instead.
    FULL = has_file("matches.csv") and load_bytes("matches.csv")[:8] == b"match_id"  # not an HTML error page
    SPACE_URL = "https://huggingface.co/spaces/jgalego/football"
    PAGES_URL = "https://jgalego.github.io/football"
    REPO_URL = "https://github.com/JGalego/football"

    def read_csv(name: str, **kw) -> pd.DataFrame:
        raw = load_bytes(name)
        if name.endswith(".gz"):
            raw = gzip.decompress(raw)
        return pd.read_csv(io.BytesIO(raw), **kw)

    def read_json(name: str):
        raw = load_bytes(name)
        if name.endswith(".gz"):
            raw = gzip.decompress(raw)
        return json.loads(raw)

    # The pre-rendered static copy of the lite edition (built with FOOTBALL_STATIC=1) would otherwise embed every audio and video clip in the page
    # (about 6 MB). There they point at the copies the interactive build ships under live/ and only load when played.
    STATIC = bool(os.environ.get("FOOTBALL_STATIC"))

    def media_video(path: str, width: int = 520):
        if STATIC:
            return mo.Html(f"<video src='live/public/{path}' controls preload='metadata' style='width:{width}px;max-width:100%;border-radius:8px'></video>")
        return mo.video(load_bytes(path), width=width, rounded=True)

    def media_audio(path: str, raw: bytes):
        if STATIC:
            return mo.Html(f"<audio src='live/public/{path}' controls preload='none' style='width:100%'></audio>")
        return mo.audio(raw)

    @functools.cache
    def actions_for(comp: str) -> pd.DataFrame:
        """SPADL-style actions for one competition (cached; do not mutate)."""
        return read_csv(f"actions_{comp}.csv.gz")

    # ── StatsBomb (120×80 yd, y down) → metres, piecewise between pitch markings.
    SB_X = [0, 6, 12, 18, 60, 102, 108, 114, 120]
    M_X = [0, 5.5, 11, 16.5, 52.5, 88.5, 94, 99.5, 105]
    SB_Y = [0, 18, 30, 36, 40, 44, 50, 62, 80]
    M_Y = [68, 54.16, 43.16, 37.66, 34, 30.34, 24.84, 13.84, 0]

    def sb_xy(p):
        return float(np.interp(p[0], SB_X, M_X)), float(np.interp(p[1], SB_Y, M_Y))

    # ── Pitch drawing (Altair). `view` is "full" or "vhalf" (attacking half, goal at the top).
    def arc_pts(cx, cy, r, a0, a1, n=40):
        t = np.radians(np.linspace(a0, a1, n))
        return list(zip(cx + r * np.cos(t), cy + r * np.sin(t)))

    def pitch_segments():
        segs = [[(0, 0), (L, 0), (L, W), (0, W), (0, 0)], [(L / 2, 0), (L / 2, W)],
                arc_pts(L / 2, W / 2, 9.15, 0, 360, 90)]
        a = math.degrees(math.acos(5.5 / 9.15))
        for x0, s in ((0.0, 1), (L, -1)):
            segs.append([(x0, W / 2 - 20.16), (x0 + s * 16.5, W / 2 - 20.16), (x0 + s * 16.5, W / 2 + 20.16), (x0, W / 2 + 20.16)])
            segs.append([(x0, W / 2 - 9.16), (x0 + s * 5.5, W / 2 - 9.16), (x0 + s * 5.5, W / 2 + 9.16), (x0, W / 2 + 9.16)])
            segs.append([(x0, W / 2 - 3.66), (x0 - s * 1.5, W / 2 - 3.66), (x0 - s * 1.5, W / 2 + 3.66), (x0, W / 2 + 3.66)])
            spot = x0 + s * 11
            segs.append(arc_pts(spot, W / 2, 9.15, -a, a) if s == 1 else arc_pts(spot, W / 2, 9.15, 180 - a, 180 + a))
        for cx, cy, a0 in ((0, 0, 0), (L, 0, 90), (L, W, 180), (0, W, 270)):
            segs.append(arc_pts(cx, cy, 1.0, a0, a0 + 90, 8))
        return segs

    PITCH_SEGS = pitch_segments()
    PITCH_DF = pd.DataFrame([dict(x=x, y=y, seg=i, o=j) for i, s in enumerate(PITCH_SEGS) for j, (x, y) in enumerate(s)])
    SPOTS_DF = pd.DataFrame(dict(x=[11.0, L / 2, L - 11], y=[W / 2] * 3))

    def xy(xf="x", yf="y", view="full"):
        """x/y encodings on the shared pitch scales."""
        if view == "vhalf":
            return dict(
                x=alt.X(f"{yf}:Q", scale=alt.Scale(domain=[-3, 71], reverse=True, nice=False, zero=False), axis=None),
                y=alt.Y(f"{xf}:Q", scale=alt.Scale(domain=[50, 107], nice=False, zero=False), axis=None),
            )
        return dict(
            x=alt.X(f"{xf}:Q", scale=alt.Scale(domain=[-3, 108], nice=False, zero=False), axis=None),
            y=alt.Y(f"{yf}:Q", scale=alt.Scale(domain=[-3, 71], nice=False, zero=False), axis=None),
        )

    def xy2(xf="ex", yf="ey", view="full"):
        if view == "vhalf":
            return dict(x2=alt.X2(f"{yf}:Q"), y2=alt.Y2(f"{xf}:Q"))
        return dict(x2=alt.X2(f"{xf}:Q"), y2=alt.Y2(f"{yf}:Q"))

    def pitch(*layers, under=(), width=560, view="full", title=None):
        lines = alt.Chart(PITCH_DF).mark_line(color=C["line"], strokeWidth=1.2, clip=True).encode(
            **xy(view=view), detail="seg:N", order="o:Q")
        spots = alt.Chart(SPOTS_DF).mark_circle(color=C["line"], size=14, opacity=1, clip=True).encode(**xy(view=view))
        height = round(width * (57 / 74 if view == "vhalf" else 74 / 111))
        chart = alt.layer(*under, lines, spots, *layers).properties(width=width, height=height)
        return chart.properties(title=title) if title else chart

    def pitch_guide(width=480):
        """Reference diagram for the conventions: the pitch with its markings, measurements (m) and the direction of attack."""
        ink, blue, orange = C["ink2"], C["blue"], C["orange"]

        def panel(xd, yd, w, dims, ext=(), notes=(), extra=(), fs=11, title=None):
            X = alt.X("x:Q", scale=alt.Scale(domain=xd, nice=False, zero=False), axis=None)
            Y = alt.Y("y:Q", scale=alt.Scale(domain=yd, nice=False, zero=False), axis=None)
            tk = 3.5 * (xd[1] - xd[0]) / w  # end-tick half-length: 3.5 px, expressed in metres

            def rules(rows, **mark):
                return alt.Chart(pd.DataFrame(rows, columns=["x", "y", "x2", "y2"])).mark_rule(clip=True, **mark).encode(
                    x=X, y=Y, x2="x2:Q", y2="y2:Q")

            def label(rows, color, size, align="center", baseline="middle"):
                base = alt.Chart(pd.DataFrame(rows, columns=["x", "y", "t", "a"])).encode(
                    x=X, y=Y, text="t:N", angle=alt.Angle("a:Q", scale=None))
                kw = dict(fontSize=size, align=align, baseline=baseline)
                return [base.mark_text(color=C["surface"], stroke=C["surface"], strokeWidth=4, strokeJoin="round", **kw),
                        base.mark_text(color=color, **kw)]

            ticks = [(px, py - tk, px, py + tk) if ya == yb else (px - tk, py, px + tk, py)
                     for xa, ya, xb, yb, *_ in dims for px, py in ((xa, ya), (xb, yb))]
            layers = [
                alt.Chart(PITCH_DF).mark_line(color=ink, strokeWidth=1.4, clip=True).encode(x=X, y=Y, detail="seg:N", order="o:Q"),
                alt.Chart(SPOTS_DF).mark_circle(color=ink, size=16, opacity=1, clip=True).encode(x=X, y=Y),
                rules(ext, color=blue, strokeWidth=1, strokeDash=[2, 2], opacity=0.7),
                rules([d[:4] for d in dims], color=blue, strokeWidth=1.4),
                rules(ticks, color=blue, strokeWidth=1.4),
                *[f(X, Y) for f in extra],
                *label([(lx, ly, t, a) for _, _, _, _, t, lx, ly, a in dims], blue, fs),
            ]
            for rows, color, size, align, baseline in notes:
                layers += label(rows, color, size, align, baseline)
            return alt.layer(*layers).properties(width=w, height=round(w * (yd[1] - yd[0]) / (xd[1] - xd[0])), title=title)

        def arrow(X, Y):  # direction of attack, above the touchline
            shaft = pd.DataFrame(dict(x=[30.0], y=[73.0], x2=[75.0], y2=[73.0]))
            head = pd.DataFrame(dict(x=[75.0], y=[73.0]))
            return alt.layer(
                alt.Chart(shaft).mark_rule(color=orange, strokeWidth=2).encode(x=X, y=Y, x2="x2:Q", y2="y2:Q"),
                alt.Chart(head).mark_point(shape="triangle-right", filled=True, color=orange, size=70, opacity=1).encode(x=X, y=Y))

        c9 = L / 2 + 9.15  # centre-circle radius
        full = panel(
            [-12.0, 114.0], [-10.0, 79.0], width,
            dims=[(0, -5, L, -5, "105 m · touchline", L / 2, -8.4, 0),
                  (-5, 0, -5, W, "68 m · goal line", -8.6, W / 2, 270),
                  (L / 2, W / 2, c9, W / 2, "", 0, 0, 0)],
            notes=[([(c9 + 1.5, W / 2, "9.15 m", 0)], blue, 11, "left", "middle"),
                   ([(1.8, 1.8, "(0, 0)", 0)], C["muted"], 10, "left", "bottom"),
                   ([(L - 1.8, W - 1.8, f"({L:.0f}, {W:.0f})", 0)], C["muted"], 10, "right", "top"),
                   ([(0, 73, "own goal", 0)], C["ink2"], 11, "left", "middle"),
                   ([(L, 73, "opponent's goal", 0)], C["ink2"], 11, "right", "middle"),
                   ([(L / 2, 76.8, "direction of attack", 0)], orange, 11, "center", "middle")],
            extra=[arrow], title=alt.Title("The pitch", subtitle="metres from the bottom-left corner, y pointing up"))
        zoom = panel(
            [78.5, 111.5], [7.5, 60.5], round(width * 0.45),
            dims=[(88.5, 10.5, L, 10.5, "16.5 m", 96.75, 8.7, 0),
                  (82.5, 13.84, 82.5, 54.16, "40.32 m", 80.7, W / 2, 270),
                  (99.5, 21.5, L, 21.5, "5.5 m", 102.25, 19.7, 0),
                  (100.9, 24.84, 100.9, 43.16, "18.32 m", 103.3, W / 2, 270),
                  (94, 47.5, L, 47.5, "11 m", 99.5, 49.3, 0),
                  (108.5, 30.34, 108.5, 37.66, "7.32 m", 110.3, W / 2, 270)],
            ext=[(88.5, 13.84, 88.5, 9.8), (88.5, 13.84, 81.8, 13.84), (88.5, 54.16, 81.8, 54.16), (99.5, 24.84, 99.5, 20.8),
                 (94, 35.2, 94, 48.2), (106.5, 30.34, 109.2, 30.34), (106.5, 37.66, 109.2, 37.66)],
            fs=10, title=alt.Title("Penalty area", subtitle="the other end is a mirror image"))
        return alt.hconcat(full, zoom, spacing=24)

    def grid_cells(nx, ny, x0=0.0, x1=L, y0=0.0, y1=W):
        """A regular grid over (part of) the pitch, as cell edges + centres."""
        xs, ys = np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1)
        gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        return pd.DataFrame(dict(
            i=gx.ravel(), j=gy.ravel(), x0=xs[gx.ravel()], x1=xs[gx.ravel() + 1],
            y0=ys[gy.ravel()], y1=ys[gy.ravel() + 1],
            cx=(xs[gx.ravel()] + xs[gx.ravel() + 1]) / 2, cy=(ys[gy.ravel()] + ys[gy.ravel() + 1]) / 2,
        ))

    def inside_polygon(px, py, poly):
        """Vectorised ray-casting point-in-polygon test."""
        px, py = np.asarray(px, float), np.asarray(py, float)
        res = np.zeros(px.shape, bool)
        n = len(poly)
        for k in range(n):
            (x1, y1), (x2, y2) = poly[k], poly[(k + 1) % n]
            cross = ((y1 > py) != (y2 > py)) & (px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-12) + x1)
            res ^= cross
        return res

    def two_teams(home, away):
        return alt.Scale(domain=[home, away], range=[C["blue"], C["orange"]])

    def picked(chart, df):
        """Rows of `df` selected by clicking an (optionally layered) Altair chart; empty if nothing is selected."""
        sel = chart.selections or {}
        if not any(bool(v) for v in sel.values()):
            return df.iloc[0:0]
        return chart.apply_selection(df)

    def further(*links):
        return mo.md("↳ **Go further** · " + " · ".join(f"[{t}]({u})" for t, u in links))

    def note(text):
        return mo.md(f"<span style='color:{C['muted']};font-size:0.85em'>{text}</span>")

    STATS = read_json("stats.json")  # aggregate counts only: safe to publish

    # Player photos: openly licensed Wikimedia Commons portraits (scripts/fetch_player_photos.py), face-cropped to 112 px.
    try:
        PHOTOS = {int(k): v for k, v in read_json("player_photos.json").items()}
    except Exception:
        PHOTOS = {}

    @functools.cache
    def face_uri(player_id, size: int = 112) -> str | None:
        """The player's photo as a data URI (works in charts, static exports and the browser build alike), or None."""
        try:
            pid = int(player_id)
            if pid not in PHOTOS:
                return None
            raw = load_bytes(f"players/{pid}.jpg")
            if size != 112:  # a smaller copy for hover cards, to keep chart data light
                small = io.BytesIO()
                Image.open(io.BytesIO(raw)).convert("RGB").resize((size, size), Image.LANCZOS).save(small, "JPEG", quality=82)
                raw = small.getvalue()
            return "data:image/jpeg;base64," + base64.b64encode(raw).decode()
        except Exception:
            return None

    def initials_of(name) -> str:
        return "".join(w[0] for w in str(name).replace("-", " ").split()[:2]).upper()

    @functools.cache
    def initials_uri(name: str) -> str:
        """A tiny round badge with the player's initials, for players without a free photo."""
        svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='88' height='88'><circle cx='44' cy='44' r='42' fill='{C['fill']}' stroke='{C['axis']}' stroke-width='2'/>"
               f"<text x='44' y='55' font-family='system-ui,sans-serif' font-size='32' text-anchor='middle' fill='{C['muted']}'>{xml_escape(initials_of(name))}</text></svg>")
        return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()

    def card_face(player_id, name) -> str:
        return face_uri(player_id, 44) or initials_uri(str(name))

    def chip(player_id, name, size: int = 22) -> str:
        """HTML for an inline mention: a round face (or initials) followed by the name. Use inside markdown or mo.Html."""
        uri = face_uri(player_id, 44 if size <= 44 else 112) or initials_uri(str(name))
        return (f"<span style='display:inline-flex;align-items:center;gap:6px;vertical-align:middle'>"
                f"<img src='{uri}' alt='' style='width:{size}px;height:{size}px;border-radius:50%;object-fit:cover;border:1px solid {C['grid']}'>{xml_escape(str(name))}</span>")

    def photo_data(df, pid: str, name: str, size: int = 80):
        """Prepare a chart's data for tooltips with a player photo.

        Returns the DataFrame with an extra `_pid` column and a lookup table (one small photo, or initials, per player).
        Use as `alt.Chart(data).transform_lookup(lookup="_pid", from_=faces)` and give the marks `tooltip=photo_tip(...)`.
        """
        d = df.copy()
        d["_pid"] = pd.to_numeric(d[pid], errors="coerce").fillna(-1).astype(int)
        names = d.assign(_name=d[name].astype(str)).drop_duplicates("_pid")[["_pid", "_name"]]
        names["_face"] = [face_uri(p, size) or initials_uri(n) for p, n in zip(names["_pid"], names["_name"])]
        return d, alt.LookupData(names[["_pid", "_face"]], "_pid", ["_face"])

    def photo_tip(name_field: str, *details):
        """Tooltip channels that Vega-Tooltip draws as a heading (the name), a photo, then a table of the details."""
        return [alt.Tooltip(f"{name_field}:N", title="title"), alt.Tooltip("_face:N", title="image"), *details]

    def avatar(player_id, name: str, size: int = 48):
        """A round face photo, or the player's initials when there is no free photo."""
        uri = face_uri(player_id)
        ring = f"border:1px solid {C['grid']};border-radius:50%;width:{size}px;height:{size}px;flex:none"
        if uri:
            return mo.Html(f"<img src='{uri}' alt='{name}' style='{ring};object-fit:cover'>")
        return mo.Html(f"<div style='{ring};background:{C['fill']};color:{C['muted']};display:flex;align-items:center;justify-content:center;font-size:{size * 0.34:.0f}px'>{xml_escape(initials_of(name))}</div>")

    def cache_version() -> str:
        h = hashlib.sha256(load_bytes("stats.json"))
        try:  # any edit to the notebook invalidates the cache, so it can never serve stale results
            h.update((mo.notebook_dir() / "the_beautiful_data.py").read_bytes())
        except Exception:
            pass
        return h.hexdigest()[:12]

    def cached(name: str, key: tuple, compute):
        """Run `compute()` once per (notebook version, key) and keep the result on disk for every later session."""
        try:
            folder = (mo.notebook_dir() or Path.cwd()) / ".cache"
            path = folder / f"{name}-{hashlib.sha256(repr((cache_version(), key)).encode()).hexdigest()[:16]}.pkl"
            if path.exists():
                return pickle.loads(path.read_bytes())
        except Exception:
            folder = path = None
        value = compute()
        if path is not None:
            try:
                folder.mkdir(exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
                tmp.replace(path)
            except Exception:
                pass  # read-only or browser file system: just recompute next time
        return value

    def snapshot(name: str):
        """Lite edition: a static picture of a section that needs row-level data (None in the full edition)."""
        if FULL:
            return None
        try:
            img = Image.open(io.BytesIO(load_bytes(f"snapshots/{name}.jpg")))
        except Exception:
            return mo.md("*(snapshot not available)*")
        return mo.vstack([
            mo.image(img, rounded=True, width="100%"),
            mo.md(f"<span style='color:{C['muted']};font-size:0.85em'>Static snapshot. [Run it live in the full notebook]({SPACE_URL}).</span>"),
        ], gap=0.3)


@app.cell(hide_code=True)
def _():
    mo.sidebar(
        [
            mo.md("### The Beautiful Data\n<span style='font-size:0.85em'>An interactive field guide to the beautiful game.</span>"),
            mo.nav_menu(
                {
                    "#kick-off": "Kick-off",
                    "#first-half-the-data": "**📊 First half · The data**",
                    "#1-the-data-stack": "1 · The data stack",
                    "#2-anatomy-of-an-event": "2 · Anatomy of an event",
                    "#3-a-match-as-a-story": "3 · A match as a story",
                    "#4-where-things-happen": "4 · Where things happen",
                    "#5-tracking-25-snapshots-a-second": "5 · Tracking data",
                    "#6-the-engine-room-physical-data": "6 · Physical data",
                    "#7-from-pixels-to-coordinates": "7 · From pixels to coordinates",
                    "#pixels-sound-and-video-three-from-scratch-demos": "↳ Photo, sound & video demos",
                    "#8-150-years-of-results": "8 · 150 years of results",
                    "#9-bodies-and-markets": "9 · Bodies and markets",
                    "#second-half-the-models": "**🧠 Second half · The models**",
                    "#10-expected-goals-xg": "10 · Expected goals",
                    "#11-expected-threat-xt": "11 · Expected threat",
                    "#12-valuing-every-action-vaep": "12 · Valuing every action",
                    "#13-pitch-control": "13 · Pitch control",
                    "#14-ghost-team": "14 · Ghost team",
                    "#15-passing-networks": "15 · Passing networks",
                    "#16-player-fingerprints": "16 · Player fingerprints",
                    "#17-forecasting-matches": "17 · Forecasting matches",
                    "#18-penalties-and-game-theory": "18 · Penalties & game theory",
                    "#19-from-data-to-words": "19 · From data to words",
                    "#extra-time-horizons": "**🔭 Extra time · Horizons**",
                    "#20-a-timeline-of-football-analytics": "20 · Timeline",
                    "#21-the-frontier": "21 · The frontier",
                    "#full-time-references": "References",
                },
                orientation="vertical",
            ),
        ],
        width="260px",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # The Beautiful Data

    **An interactive field guide to the beautiful game.**
    """)
    return


@app.cell(hide_code=True)
def _():
    # Only the pre-rendered static copy of the lite edition (built with FOOTBALL_STATIC=1) shows this; the interactive builds show nothing.
    mo.callout(mo.md(
        "**Fast static copy.** Charts are pre-rendered so this page opens instantly, but sliders, dropdowns and players are frozen. "
        "[Open the interactive version](live/index.html) (it downloads Python into your browser and takes a minute or two to start)."),
        kind="neutral") if STATIC else None
    return


@app.cell(hide_code=True)
def _():
    # A wall of football GIFs from GIPHY: goals, saves, tackles, skills, set pieces, cards, trophies, crowds, the women's game.
    # Each tile is Giphy's own small animated "preview" rendition (12-49 KB), loaded lazily as it scrolls into view, and links to the clip on GIPHY.
    # Needs no data and no kernel work, so it also appears instantly on the static page. `prefers-reduced-motion` gets a still frame instead.
    _N, _M = 6, 8  # rows x columns of the wall (at most as many tiles as there are clips below)
    _clips = [
        ("3o7bu2D938PkrKrcYw", "Goal celebration"),
        ("noV8SksgLVNJK", "Save"),
        ("3oEduHRlWsv3k4A48w", "Tackle"),
        ("SMearnvJ1ghbdXpkzC", "Dribble"),
        ("j5LM22Mtjcfs4ZdCBo", "Skills"),
        ("9rgeyUcLL4NusBIPQg", "Nutmeg"),
        ("LSZQkdPUz6IgM", "Free kick"),
        ("8A4HVoSlNAxfQ7Y4CT", "Penalty"),
        ("THOKxpYQHaakkKsSsY", "Bicycle kick"),
        ("26FL4BhNXmo8PNLYk", "Goal"),
        ("xT8qBqVM3NpG1J802k", "Free kick goal"),
        ("cRRuFyTrAezSIWz52S", "Yellow card"),
        ("dFrGgACie2s8Ux4Ygg", "Red card"),
        ("l46C6ESBdxfQKXEn6", "Corner kick"),
        ("gLWTWCg96RA0btDUm0", "Trophy"),
        ("3otPoRKgy8Urlwe9So", "Celebration"),
        ("8Js7tAjS32CdGvjiOR", "Fans"),
        ("4EEy1DQfXG9KEeaCQI", "Stadium"),
        ("cnJNVjiIWcDGIMAuIe", "Women's goal"),
        ("no6HN9ItkB79posq8o", "Women's celebration"),
        ("Gbz7uJxsfgrPProUjj", "Women's football"),
        ("b7hZe96PaAly4RjZZR", "Shot"),
        ("4RALum4ThckN2", "World Cup"),
        ("j2qJeo2Dhm7ubDKcou", "Throw-in"),
        ("3og0IHHIBTcYxTr8qs", "Goal celebration"),
        ("m2MaJEv0A52PcVhcaV", "Save"),
        ("WCX1h3y8emgj9Prjem", "Tackle"),
        ("MtI011el1MbFmQRFcs", "Dribble"),
        ("7ckMJ2ymiJhnqS9Crd", "Skills"),
        ("55eFoH5ZowIslGNcmH", "Nutmeg"),
        ("Je9Wbvf3EXXTW", "Penalty"),
        ("3FbDBpuRJuUd6VyD3E", "Bicycle kick"),
        ("5FOsu6mtaurTKc4Fxs", "Goal"),
        ("NGDi0TBHQ7UWaoAmr8", "Red card"),
        ("QBMrpkBiaOX90n3rbD", "Trophy"),
        ("kqLgJfjrD8MKC8kyZl", "Celebration"),
        ("tqqH0G3f9tQNHwJuev", "Fans"),
        ("5BYt33NDLURd1Oq8UN", "Stadium"),
        ("ulxLtUzAjrakGBmEOe", "Women's goal"),
        ("Jm9jMGt1XdA4TYaJ7i", "Women's celebration"),
        ("DWTfFGvAg32PFP9uX5", "Throw-in"),
        ("xUOxeYaa0VdPVX6qwU", "Goal celebration"),
        ("KcPvmYNQL8TrNlZolu", "Save"),
        ("QpTiFUjE9rc3Zh01ra", "Tackle"),
        ("l3vRhE2oWFNfgPims", "Dribble"),
        ("kttzuTIE4bDCznFNDi", "Skills"),
        ("d9HYTEzbDO4ULqDZ8z", "Nutmeg"),
        ("xX0bddamE2MdhFpw47", "Bicycle kick"),
        ("tjOqm4ByG4AwM", "Goal"),
        ("kC8x2sN2DnHlGfFNAS", "Trophy"),
        ("kexqhPUFLtnzAd2788", "Celebration"),
        ("sQSMIM6eniTdLkla9k", "Women's goal"),
        ("GURjAoOV2R7N95asyN", "Women's celebration"),
        ("vgHYAr2NFzcT3QVuLC", "Throw-in"),
        ("xPYlrYkCu05iTkpsM2", "Goal celebration"),
        ("mEgimsNRRuEQ1hi1xX", "Tackle"),
        ("mHjCRi4kOMrHa", "Dribble"),
        ("xThta0elszjqKfn1Di", "Bicycle kick"),
        ("HwKOUD4d4918Q", "Goal"),
        ("5Yeeg9ugcLTlC9MfVT", "Celebration"),
        ("Oj4YFnnVGOh3jcLmIo", "Women's goal"),
        ("waHQVqaKne0TcpNV0I", "Women's celebration"),
        ("24okY552tjzCwuGEM0", "Bicycle kick"),
        ("26xBR3EG96YmhwbPG", "Goal"),
        ("l41YmrFwNJYvpOi8E", "Goal"),
        ("xT1Ra3E0ofBiuAJXRC", "Goal"),
        ("IeL5zvS2I1qaSzzt9W", "Goal"),
        ("VKi4Iw5mnl3XOaaZmh", "Goal"),
        ("iF1AybL3n0eSLagITq", "Goal"),
        ("ggLCPifsljVFILuoUC", "Goal"),
        ("Kh8FG9fiF8GGgG6YUO", "Goal"),
    ]
    _tiles = "".join(
        f"<a class='gw-t' href='https://giphy.com/gifs/{gid}' target='_blank' rel='noopener noreferrer' title='{xml_escape(what)} · via GIPHY'>"
        f"<picture><source media='(prefers-reduced-motion: reduce)' srcset='https://media.giphy.com/media/{gid}/200_s.gif'>"
        f"<img src='https://media.giphy.com/media/{gid}/giphy-preview.gif' alt='Football GIF: {xml_escape(what)}' loading='lazy' decoding='async'></picture></a>"
        for gid, what in _clips[: _N * _M])
    _css = f"""
    .gw-grid{{display:grid;grid-template-columns:repeat({_M},minmax(0,1fr));gap:6px;margin:6px 0 2px}}
    .gw-t{{display:block;aspect-ratio:4/3;overflow:hidden;border-radius:8px;background:{C['fill']};border:1px solid {C['grid']}}}
    .gw-t img{{display:block;width:100%;height:100%;object-fit:cover;transition:transform .25s ease}}
    .gw-t:hover img{{transform:scale(1.08)}}
    @media (max-width:820px){{.gw-grid{{grid-template-columns:repeat({max(2, _M // 2)},minmax(0,1fr))}}}}
    @media (max-width:480px){{.gw-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
    """
    mo.vstack([
        mo.Html(f"<div><style>{_css}</style><div class='gw-grid'>{_tiles}</div></div>"),
        mo.md(f"<span style='color:{C['muted']};font-size:0.85em'>GIFs via <a href='https://giphy.com' target='_blank' rel='noopener noreferrer'>GIPHY</a>. Click one to open it there.</span>"),
    ], gap=0.4)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Every pass, sprint, heartbeat and transfer now leaves a trace. This notebook goes from the raw traces
    (results, events, tracking, video, bodies, money, words) to the models people have built on them
    (expected goals, pitch control, action values, networks, forecasts) and on to what comes next.

    Everything runs on **public data** and every model is **built from scratch** in front of you,
    so change the sliders and see what happens.
    """)
    return


@app.cell
def _():
    mo.stop(not FULL)
    COMP_ORDER = ["wc22", "eu24", "ca24", "afc23", "eu20", "wc18", "wwc23", "weu25", "weu22", "wwc19", "b2324"]
    matches = read_csv("matches.csv")
    teams = read_csv("teams.csv")
    players = read_csv("players.csv.gz")
    appearances = read_csv("appearances.csv.gz")
    TEAM = dict(zip(teams.team_id, teams.team))
    PLAYER = dict(zip(players.player_id, players.player))
    COMPS = {k: matches.loc[matches.comp == k, "competition"].iloc[0] for k in COMP_ORDER if k in set(matches.comp)}
    shots = read_csv("shots.csv.gz").assign(
        team=lambda d: d.team_id.map(TEAM), player=lambda d: d.player_id.map(PLAYER)
    )

    def _label(r):
        stage = "" if r.stage in (None, "Regular Season") or pd.isna(r.stage) else f" · {r.stage}"
        return f"{COMPS[r.comp]}{stage} · {r.home} {r.home_score}–{r.away_score} {r.away} ({r.date})"

    MATCH_OPTIONS = {_label(r): r.match_id for r in matches.itertuples()}
    FINAL_ID = int(matches.query("comp == 'wc22' and stage == 'Final'").match_id.iloc[0])
    FINAL_LABEL = next(k for k, v in MATCH_OPTIONS.items() if v == FINAL_ID)
    return (
        COMPS,
        COMP_ORDER,
        FINAL_ID,
        FINAL_LABEL,
        MATCH_OPTIONS,
        PLAYER,
        TEAM,
        appearances,
        matches,
        players,
        shots,
        teams,
    )


@app.cell(hide_code=True)
def _(shots):
    # Every goal, one half-pitch for men's football and one for women's: a ray from where the shot was struck to where it crossed the line.
    # Hover a dot for the scorer's photo and the details.
    _goals = shots.query("goal == 1 and period < 5").assign(
        when=lambda d: d.team + " · " + d.minute.astype(str) + "′", detail=lambda d: d.shot_type + " · xG " + d.sb_xg.round(2).astype(str))
    _panels = []
    for _who, _title, _col in (("men", "Men's football", C["blue"]), ("women", "Women's football", C["orange"])):
        _d, _faces = photo_data(_goals[_goals.gender == _who][["x", "y", "end_x", "end_y", "player_id", "player", "when", "detail"]], "player_id", "player")
        _rays = alt.Chart(_d).mark_rule(strokeWidth=0.8, opacity=0.22, color=_col).encode(**xy(view="vhalf"), **xy2("end_x", "end_y", view="vhalf"))
        _dots = alt.Chart(_d).transform_lookup(lookup="_pid", from_=_faces).mark_circle(size=14, opacity=0.65, color=_col).encode(
            **xy(view="vhalf"), tooltip=photo_tip("player", alt.Tooltip("when:N", title="team · minute"), alt.Tooltip("detail:N", title="shot")))
        _panels.append(pitch(_rays, _dots, view="vhalf", width=400, title=alt.Title(_title, subtitle=f"{len(_d):,} goals · hover a dot")))
    alt.hconcat(*_panels, spacing=24).properties()
    return


@app.cell(hide_code=True)
def _():
    mo.hstack(
        [
            mo.stat(f"{STATS['n_matches']:,}", label="matches", caption=f"{STATS['n_competitions']} competitions, men's & women's"),
            mo.stat(f"{STATS['n_events'] / 1e6:.1f} M", label="events", caption="hand-tagged on-ball actions"),
            mo.stat(f"{STATS['n_360_players'] / 1e6:.1f} M", label="player snapshots", caption="from 360 freeze frames"),
            mo.stat(f"{STATS['n_shots']:,}", label="shots", caption=f"{STATS['n_goals']:,} goals"),
            mo.stat(f"{STATS['n_intl']:,}", label="international results", caption="since 1872"),
        ],
        widths="equal",
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _():
    _lite = None if FULL else mo.callout(mo.md(
        f"**Lite edition.** Parts built on StatsBomb and Metrica data (sections 2–7, 10–16, 18 and 19) appear as static snapshots, because those providers' terms "
        f"do not allow publishing their data. Everything else here is live. [Open the full interactive notebook]({SPACE_URL}) or [run it yourself]({REPO_URL})."), kind="info")
    _logo = mo.image(Image.open(io.BytesIO(load_bytes("statsbomb-logo.png"))), alt="StatsBomb", height=44,
                      style={"background": "#fcfcfb", "padding": "7px 12px", "border-radius": "8px", "border": "1px solid #e1e0d9"}) if has_file("statsbomb-logo.png") else mo.md("")
    _credit = mo.hstack([mo.md("Event and 360 data provided by **StatsBomb** ([open data](https://github.com/statsbomb/open-data))."), _logo], justify="start", gap=1, align="center")
    mo.vstack([x for x in (_lite, snapshot("hero"), _credit) if x is not None], gap=0.8)
    return


@app.cell(hide_code=True)
def _():
    mo.vstack([
        mo.md(r"""
        ## Kick-off

        This guide is laid out like a match:

        - In the **first half**, we look into the data, what gets recorded, at what resolution, and what you can download.
        - In the **Second half**, we check out some classic football models, each rebuilt from scratch on open data.
        - During **extra time**, we survey 70+ years of football analytics on one timeline, and what's being worked on now.

        **Data:** [StatsBomb Open Data](https://github.com/statsbomb/open-data) (events and 360 freeze frames),
        [Metrica Sports sample data](https://github.com/metrica-sports/sample-data) (tracking) and
        [international results since 1872](https://github.com/martj42/international_results) (CC0).
        `scripts/download_data.py` and `scripts/prepare_data.py` rebuild everything in `public/` from source.

        **Conventions:** every position is in metres on a 105 × 68 m pitch, the size FIFA recommends for stadiums.
        Real pitches vary a little, so provider coordinates are rescaled to it. In event data the team on the ball
        always attacks **left → right**, whichever way it really played; in tracking data it is the home team.
        Half-pitch maps are turned so the goal being attacked is at the top.
        """),
        pitch_guide(),
    ], gap=0.8)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # First half · The data
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1 · The data stack

    A match can be described with **one number** (the result) or with **a trillion** (the pixels of the broadcast).
    Each layer up the stack is more expensive and less open, and lets you ask finer questions.
    **Click a dot** to see what each layer is, who records it and what you can build with it.
    """)
    return


@app.cell
def _():
    _ev = STATS["events_per_match"]
    _f360 = STATS["freeze_players_per_match"]
    _trk_positions = STATS["tracking_positions_per_match"]
    DATA_STACK = pd.DataFrame(
        [
            ("Results", "Outcomes", 1, "Open", "Final score, date, venue, competition.", "Federations, volunteers, Wikipedia", "martj42/international_results (CC0), openfootball, football-data.co.uk", "Elo ratings, Poisson goal models, tournament simulators", False),
            ("Line-ups & formations", "Outcomes", 40, "Open", "Who started, who came on, which shape.", "Clubs, data providers", "StatsBomb lineups, Transfermarkt", "Minutes played, chemistry, rotation analysis", False),
            ("Transfers & valuations", "Markets", 3, "Open samples", "Fees, contracts and crowd-estimated market values.", "Clubs, Transfermarkt community", "dcaribou/transfermarkt-datasets", "Valuation models, transfer networks, ageing curves", False),
            ("Betting odds", "Markets", 2e4, "Open samples", "Prices for every market, before and during the game.", "Bookmakers, exchanges", "football-data.co.uk (closing odds)", "Implied probabilities, market efficiency, calibration benchmarks", False),
            ("Event data", "Actions", _ev, "Open samples", "Every on-ball action: who, what, where, when, how.", "StatsBomb/Hudl, Opta/Stats Perform, Wyscout (human taggers)", "StatsBomb Open Data, Wyscout public dataset", "xG, xT, VAEP, passing networks, player profiles", True),
            ("360 freeze frames", "Actions", _f360, "Open samples", "Every visible player's position at each event.", "StatsBomb/Hudl (from broadcast video)", "StatsBomb 360 (Euros, World Cups, Bundesliga 23/24)", "Context-aware xG, pressure, passing options", True),
            ("Optical tracking", "Positions", _trk_positions, "Open samples", "x, y of all 22 players and the ball, 25 times a second.", "Stadium camera systems (e.g. TRACAB, Hawk-Eye, Second Spectrum)", "Metrica sample games, DFL/IDSSE Bundesliga sample, PFF FC World Cup 2022", "Pitch control, physical metrics, off-ball runs, team shape", True),
            ("Broadcast tracking", "Positions", 6e5, "Open samples", "Positions estimated from the TV feed: only who's on screen.", "SkillCorner, PFF FC and others (computer vision)", "SkillCorner open data", "Tracking at scale for scouting in lower leagues", False),
            ("Skeletal tracking", "Bodies", 1.8e8, "Mostly closed", "~29 body points per player, ~50 times a second.", "Hawk-Eye (semi-automated offside)", "Research pose datasets", "Offside lines, biomechanics, injury mechanics", False),
            ("Connected ball", "Bodies", 2.9e6, "Mostly closed", "An inertial sensor inside the ball, ~500 Hz.", "adidas / KINEXON (World Cup 2022 onward)", "None open", "Exact kick moments, touch detection, offside timing", False),
            ("Wearables (GPS/IMU)", "Bodies", 1.2e7, "Mostly closed", "Vests with GPS (~10 Hz) and accelerometers (~100 Hz).", "Catapult, STATSports and similar", "SoccerMon (aggregated)", "Load monitoring, injury-risk studies, return-to-play", False),
            ("Heart rate & wellness", "Bodies", 1.3e5, "Mostly closed", "Heart rate, sleep, soreness, mood questionnaires.", "Club sports science departments", "SoccerMon, PMData", "Readiness models, fatigue, recovery planning", False),
            ("Broadcast video", "Pixels", 6e11, "Open samples", "50 frames a second, ~2 million pixels each.", "Broadcasters, club analysts", "SoccerNet (labels open; video under NDA)", "Action spotting, automatic highlights, broadcast tracking", False),
            ("Commentary & reports", "Words", 5e3, "Open samples", "Live text, scout reports, press conferences.", "Media, clubs", "SoccerNet-Caption, live text archives", "Automatic commentary, retrieval, scouting assistants", False),
            ("Fan & social data", "Words", 1e6, "Mostly closed", "Posts, searches, ticketing, viewing figures.", "Platforms, clubs, broadcasters", "Platform APIs (restricted)", "Sentiment, engagement, personalised broadcasts", False),
        ],
        columns=["layer", "family", "per_match", "openness", "what", "who", "open", "build", "measured"],
    )
    DATA_STACK["shown"] = DATA_STACK.per_match.map(lambda v: f"{v:,.0f}" if v < 1e6 else f"{v:.1e}")
    _short = {"Line-ups & formations": "Line-ups", "Transfers & valuations": "Transfers & values", "Heart rate & wellness": "Heart rate",
              "Wearables (GPS/IMU)": "Wearables", "Connected ball": "Ball sensor", "Commentary & reports": "Commentary", "Fan & social data": "Fans & social"}
    DATA_STACK["label"] = DATA_STACK.layer.map(lambda n: _short.get(n, n))
    DATA_STACK["pos"] = DATA_STACK.sort_values("per_match").groupby("family").cumcount().reindex(DATA_STACK.index).map(lambda k: "above" if k % 2 == 0 else "below")
    return (DATA_STACK,)


@app.cell(hide_code=True)
def _(DATA_STACK):
    _sel = alt.selection_point(fields=["layer"], on="click", empty=False, name="layer_pick")
    _fam = ["Outcomes", "Markets", "Actions", "Positions", "Bodies", "Pixels", "Words"]
    _base = alt.Chart(DATA_STACK).encode(
        x=alt.X("per_match:Q", scale=alt.Scale(type="log", domain=[0.5, 3e12]), title="records per match (log scale, order of magnitude)",
                axis=alt.Axis(format="~s", values=[1, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12])),
        y=alt.Y("family:N", sort=_fam, title=None, axis=alt.Axis(labelColor=C["ink2"], labelFontSize=12, domain=False, ticks=False)),
    )
    _dots = _base.mark_circle(size=170, stroke=C["surface"], strokeWidth=2).encode(
        color=alt.Color("openness:N", scale=alt.Scale(domain=["Open", "Open samples", "Mostly closed"], range=[C["aqua"], C["blue"], C["orange"]]),
                        legend=alt.Legend(title=None, orient="top")),
        opacity=alt.condition(_sel, alt.value(1), alt.value(0.8)),
        size=alt.condition(_sel, alt.value(320), alt.value(170)),
        tooltip=[alt.Tooltip("layer:N", title="Layer"), alt.Tooltip("shown:N", title="Records / match"), alt.Tooltip("what:N", title="What")],
    ).add_params(_sel)
    _above = _base.mark_text(align="center", dy=-15, fontSize=11, color=C["ink2"]).encode(text="label:N").transform_filter("datum.pos == 'above'")
    _below = _base.mark_text(align="center", dy=17, fontSize=11, color=C["ink2"]).encode(text="label:N").transform_filter("datum.pos == 'below'")
    stack_chart = mo.ui.altair_chart(
        alt.layer(_dots, _above, _below).properties(width=640, height=360), chart_selection=False, legend_selection=False
    )
    return (stack_chart,)


@app.cell(hide_code=True)
def _(DATA_STACK, stack_chart):
    _picked = picked(stack_chart, DATA_STACK)
    _row = DATA_STACK.set_index("layer").loc[_picked.layer.iloc[0]] if len(_picked) else DATA_STACK.set_index("layer").loc["Event data"]
    _name = _picked.layer.iloc[0] if len(_picked) else "Event data"
    _src = "measured from the data in this notebook" if _row.measured else "order-of-magnitude estimate"
    _card = mo.md(f"""
    ### {_name}
    {_row.what}

    | | |
    |---|---|
    | **Records per match** | {_row.shown} <span style='color:{C["muted"]}'>({_src})</span> |
    | **Recorded by** | {_row.who} |
    | **Open sources** | {_row.open} |
    | **You can build** | {_row.build} |
    """)
    mo.vstack([stack_chart, _card])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The atlas: every activity, every modality

    Matches are only one source. Training sessions, medical rooms, scouting trips, referees' headsets, broadcasts,
    stands, ticket offices, video games and Sunday-league apps all generate data too, as tables, signals, images,
    sound, text, graphs and even blood. **Click a cell** to see what is recorded there, what you can extract from it and where to find open examples.
    """)
    return


@app.cell
def _():
    atlas = pd.DataFrame(read_json("atlas.json"))
    ATLAS_ACT = ["Match play", "Training & preparation", "Body & health", "Scouting & recruitment", "Officiating", "Broadcast & media",
                 "Fans & community", "Business & operations", "Stadium & environment", "Games, fantasy & betting", "Grassroots & youth"]
    ATLAS_MOD = ["Tabular", "Time series & sensors", "Spatial & tracking", "Image", "Video", "Audio", "Text", "Graph", "3D & pose", "Biological"]
    _cnt = atlas.groupby(["activity", "modality"]).agg(n=("name", "size"), open=("openness", lambda s: (s != "Mostly closed").sum()),
                                                        names=("name", lambda s: " · ".join(s))).reset_index()
    _pick = alt.selection_point(fields=["activity", "modality"], on="click", empty=False, name="atlas_pick")
    _base = alt.Chart(_cnt).encode(
        x=alt.X("modality:N", sort=ATLAS_MOD, title=None, axis=alt.Axis(orient="top", labelAngle=-35, labelColor=C["ink2"], domain=False, ticks=False)),
        y=alt.Y("activity:N", sort=ATLAS_ACT, title=None, axis=alt.Axis(labelColor=C["ink2"], domain=False, ticks=False)),
    )
    _cells = _base.mark_rect(cornerRadius=4, stroke=C["surface"], strokeWidth=3).encode(
        color=alt.Color("n:Q", scale=alt.Scale(range=C["seq"][1:6]), legend=alt.Legend(title="kinds of data", orient="bottom")),
        strokeWidth=alt.condition(_pick, alt.value(3), alt.value(3)),
        stroke=alt.condition(_pick, alt.value(C["ink"]), alt.value(C["surface"])),
        tooltip=[alt.Tooltip("activity:N"), alt.Tooltip("modality:N"), alt.Tooltip("n:Q", title="kinds"), alt.Tooltip("open:Q", title="with open sources"), alt.Tooltip("names:N", title="")],
    ).add_params(_pick)
    _txt = _base.mark_text(fontSize=11).encode(text="n:Q", color=alt.condition("datum.n >= 4", alt.value("#ffffff"), alt.value(C["ink"])))
    atlas_chart = mo.ui.altair_chart(alt.layer(_cells, _txt).properties(width=560, height=alt.Step(30)), chart_selection=False, legend_selection=False)
    return ATLAS_ACT, ATLAS_MOD, atlas, atlas_chart


@app.cell(hide_code=True)
def _(atlas, atlas_chart):
    _sel = picked(atlas_chart, atlas[["activity", "modality"]].drop_duplicates())
    if len(_sel):
        _a, _m = _sel.activity.iloc[0], _sel.modality.iloc[0]
    else:
        _a, _m = "Officiating", "Audio"
    _rows = atlas[(atlas.activity == _a) & (atlas.modality == _m)]
    _badge = {"Open": "open", "Open samples": "open samples", "Mostly closed": "mostly closed"}

    def _item(r):
        ex = " · ".join(f"[{e['title']}]({e['url']})" for e in (r.examples or []))
        return (f"**{r['name']}** <span style='color:{C['muted']}'>({_badge.get(r.openness, r.openness)})</span>  \n"
                f"*Recorded:* {r.what}  \n*Extract:* {r.extract}  \n*Build:* {r.build}" + (f"  \n*Open examples:* {ex}" if ex else ""))

    _cards = [mo.md(_item(r)).style({"padding": "10px 14px", "border": f"1px solid {C['grid']}", "border-radius": "8px"}) for _, r in _rows.iterrows()]
    _detail = mo.vstack([mo.md(f"#### {_a} × {_m}"), mo.hstack(_cards, widths="equal", wrap=True, align="stretch", gap=1)])
    _tot = mo.md(f"<span style='color:{C['muted']}'>{len(atlas)} kinds of data · {(atlas.openness != 'Mostly closed').sum()} with open sources</span>")
    mo.vstack([atlas_chart, _tot, _detail])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### An open-data catalogue

    Football data is mostly proprietary, but a lot has been released for research and learning.
    Filter by data type; every entry was checked against its source.
    """)
    return


@app.cell
def _():
    catalogue = pd.DataFrame(read_json("datasets.json"))
    catalogue_type = mo.ui.multiselect(sorted(catalogue.data_type.unique()), label="Data types (empty = all)")
    return catalogue, catalogue_type


@app.cell(hide_code=True)
def _(catalogue, catalogue_type):
    _cat = catalogue if not catalogue_type.value else catalogue[catalogue.data_type.isin(catalogue_type.value)]
    mo.vstack([
        catalogue_type,
        mo.ui.table(
            _cat[["name", "data_type", "coverage", "license", "year", "url"]].sort_values(["data_type", "year"]).reset_index(drop=True),
            selection=None, page_size=10, show_column_summaries=False,
            format_mapping={"url": lambda u: mo.md(f"[link]({u})")},
            wrapped_columns=["coverage"],
        ),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2 · Anatomy of an event

    **Event data** turns a match into ~3,500 rows. A trained tagger (increasingly helped by computer vision)
    logs each on-ball action with a timestamp, a location and dozens of qualifiers. Since 2021 StatsBomb's
    **360** frames add where every *visible* player stood at that instant.

    Below is the raw JSON for the **2022 World Cup final**, Argentina 3–3 France, event by event.
    """)
    return


@app.cell(hide_code=True)
def _():
    snapshot("s02")
    return


@app.cell
def _():
    mo.stop(not FULL)
    final_events = read_json("final_events.json.gz")
    final_360 = {f["event_uuid"]: f for f in read_json("final_360.json.gz")}
    final_types = pd.Series([e["type"]["name"] for e in final_events]).value_counts()
    anat_type = mo.ui.dropdown(list(final_types.index), value="Shot", label="Event type")
    return anat_type, final_360, final_events, final_types


@app.cell
def _(anat_type, final_events):
    anat_pool = [e for e in final_events if e["type"]["name"] == anat_type.value]
    _first_goal = next((k for k, e in enumerate(anat_pool) if e.get("shot", {}).get("outcome", {}).get("name") == "Goal"), 0)
    anat_i = mo.ui.slider(1, len(anat_pool), value=_first_goal + 1, label=f"{anat_type.value} #", show_value=True, full_width=True)
    return anat_i, anat_pool


@app.cell(hide_code=True)
def _(PLAYER, anat_i, anat_pool, anat_type, final_360):
    _ev = anat_pool[anat_i.value - 1]
    _ff = final_360.get(_ev["id"])
    _team = _ev["team"]["name"]
    _home, _away = "Argentina", "France"
    _other = _away if _team == _home else _home
    _layers, _under = [], []
    if _ff:
        _poly = [sb_xy(_ff["visible_area"][k : k + 2]) for k in range(0, len(_ff["visible_area"]), 2)]
        _g = grid_cells(70, 45)
        _g = _g[inside_polygon(_g.cx, _g.cy, _poly)]
        _under.append(alt.Chart(_g).mark_rect(color=C["fill"], opacity=1).encode(**xy("x0", "y0"), **xy2("x1", "y1")))
        _pp = pd.DataFrame([dict(zip(("x", "y"), sb_xy(p["location"])), team=_team if p["teammate"] else _other,
                                 role="actor" if p.get("actor") else ("keeper" if p.get("keeper") else "player")) for p in _ff["freeze_frame"]])
        _layers.append(alt.Chart(_pp).mark_circle(size=130, opacity=1, stroke=C["surface"], strokeWidth=2).encode(
            **xy(), color=alt.Color("team:N", scale=two_teams(_home, _away), legend=alt.Legend(title=None, orient="bottom")),
            shape=alt.Shape("role:N", scale=alt.Scale(domain=["player", "keeper", "actor"], range=["circle", "square", "diamond"]), legend=None),
            tooltip=["team:N", "role:N"]))
    if "location" in _ev:
        _x, _y = sb_xy(_ev["location"])
        _end = None
        for _k in ("pass", "carry", "shot"):
            if _k in _ev and "end_location" in _ev[_k]:
                _end = sb_xy(_ev[_k]["end_location"][:2])
        _ed = pd.DataFrame([dict(x=_x, y=_y, ex=_end[0] if _end else _x, ey=_end[1] if _end else _y)])
        _layers.append(alt.Chart(_ed).mark_rule(color=C["ink"], strokeWidth=2).encode(**xy(), **xy2()))
        _layers.append(alt.Chart(_ed).mark_circle(color=C["ink"], size=60, opacity=1).encode(**xy("ex", "ey")))
        _layers.append(alt.Chart(_ed).mark_point(color=C["ink"], size=260, strokeWidth=2).encode(**xy()))
    _who = PLAYER.get((_ev.get("player") or {}).get("id"), "—")
    _title = f"{_ev['minute']}′{_ev['second']:02d}″ · {_who} ({_team}) · {_ev['type']['name']}"
    _sub = "Shaded: the area the broadcast camera could see. Diamond: the player on the ball." if _ff else "No 360 frame for this event."
    _chart = pitch(*_layers, under=_under, width=520, title=alt.Title(_title, subtitle=_sub))
    _json = mo.Html(f"<div style='max-height:390px;overflow:auto;font-size:0.85em'>{mo.json(_ev).text}</div>")
    _who_id = (_ev.get("player") or {}).get("id")
    _head = (mo.hstack([avatar(_who_id, _who, 56), mo.md(f"**{_who}** · {_team} · {_ev['type']['name']} · {_ev['minute']}′")], justify="start", align="center", gap=1)
             if _who_id else mo.md(""))
    mo.vstack([mo.hstack([anat_type, anat_i], widths=[1, 3]), _head, mo.hstack([_chart, _json], widths=[3, 2], align="start")])
    return


@app.cell(hide_code=True)
def _(final_types):
    _d = final_types.rename_axis("type").reset_index(name="n").head(18)
    alt.Chart(_d).mark_bar(color=C["blue"], cornerRadiusEnd=3, height=12).encode(
        x=alt.X("n:Q", title="events in the final"), y=alt.Y("type:N", sort="-x", title=None),
        tooltip=["type:N", "n:Q"],
    ).properties(width=520, height=330, title=alt.Title(f"{final_types.sum():,} events in one match", subtitle="Passes and ball receipts dominate; shots are rare."))
    return


@app.cell(hide_code=True)
def _():
    further(
        ("StatsBomb open-data spec", "https://github.com/statsbomb/open-data/tree/master/doc"),
        ("kloppy (load any provider)", "https://github.com/PySport/kloppy"),
        ("SPADL: a common action language (Decroos et al. 2019)", "https://arxiv.org/abs/1802.07127"),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3 · A match as a story

    Goals are rare, about 2.6 a game, so the scoreline is a noisy summary. Adding up the **expected goals**
    of every shot gives a smoother account of who was creating chances. Pick any match.
    """)
    return


@app.cell(hide_code=True)
def _():
    snapshot("s03")
    return


@app.cell
def _(FINAL_LABEL, MATCH_OPTIONS):
    story_match = mo.ui.dropdown(MATCH_OPTIONS, value=FINAL_LABEL, searchable=True, label="Match", full_width=True)
    story_match
    return (story_match,)


@app.cell(hide_code=True)
def _(matches, shots, story_match):
    _m = matches.set_index("match_id").loc[story_match.value]
    _s = shots.query("match_id == @story_match.value and period < 5").assign(t=lambda d: d.minute + d.second / 60).sort_values("t")
    _end = max(90.0, float(_s.t.max()) if len(_s) else 90.0)
    _rows = []
    for _team in (_m.home, _m.away):
        _ts = _s[_s.team == _team]
        _rows += [dict(team=_team, t=0.0, xg=0.0)]
        _rows += [dict(team=_team, t=r.t, xg=v) for r, v in zip(_ts.itertuples(), _ts.sb_xg.cumsum())]
        _rows += [dict(team=_team, t=_end, xg=float(_ts.sb_xg.sum()))]
    _race = pd.DataFrame(_rows)
    _goals = _s[_s.goal == 1].merge(_race, on=["team", "t"], how="left")
    _scale = two_teams(_m.home, _m.away)
    _lines = alt.Chart(_race).mark_line(interpolate="step-after").encode(
        x=alt.X("t:Q", title="minute", scale=alt.Scale(domain=[0, _end], nice=False)), y=alt.Y("xg:Q", title="cumulative xG"),
        color=alt.Color("team:N", scale=_scale, legend=alt.Legend(title=None, orient="top")),
    )
    _gd, _gf = photo_data(_goals, "player_id", "player")
    _gpts = alt.Chart(_gd).transform_lookup(lookup="_pid", from_=_gf).mark_circle(size=110, opacity=1, stroke=C["surface"], strokeWidth=2).encode(
        x="t:Q", y="xg:Q", color=alt.Color("team:N", scale=_scale, legend=None),
        tooltip=photo_tip("player", alt.Tooltip("minute:Q", title="Minute"), alt.Tooltip("sb_xg:Q", title="xG of the shot", format=".2f")),
    )
    _race_chart = alt.layer(_lines, _gpts).properties(
        width=560, height=230,
        title=alt.Title(f"{_m.home} {_m.home_score}–{_m.away_score} {_m.away}",
                        subtitle=f"xG {_s[_s.team == _m.home].sb_xg.sum():.2f} – {_s[_s.team == _m.away].sb_xg.sum():.2f} · dots are goals"),
    )
    # Shot map: home attacks right, away attacks left
    _sm = _s.assign(px=lambda d: np.where(d.team == _m.home, d.x, L - d.x), py=lambda d: np.where(d.team == _m.home, d.y, W - d.y),
                    result=lambda d: np.where(d.goal == 1, "Goal", "No goal"))
    _sd, _sf = photo_data(_sm, "player_id", "player")
    _shotmap = pitch(
        alt.Chart(_sd).transform_lookup(lookup="_pid", from_=_sf).mark_point(filled=True, stroke=C["surface"], strokeWidth=1).encode(
            **xy("px", "py"), size=alt.Size("sb_xg:Q", scale=alt.Scale(range=[20, 500], domain=[0, 1]), legend=None),
            color=alt.Color("team:N", scale=_scale, legend=None),
            opacity=alt.Opacity("result:N", scale=alt.Scale(domain=["Goal", "No goal"], range=[1, 0.35]), legend=alt.Legend(title=None, orient="bottom")),
            tooltip=photo_tip("player", "minute:Q", alt.Tooltip("sb_xg:Q", format=".2f", title="xG"), "outcome:N", "body_part:N"),
        ),
        width=560, title=alt.Title("Every shot, sized by xG", subtitle=f"{_m.home} attack →   ← {_m.away} attack"),
    )
    mo.vstack([_race_chart, _shotmap])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4 · Where things happen

    Aggregate thousands of events and a team's or a player's **habits** appear: where they receive, where they
    shoot from, where they press. Pick a competition, a team and an action.
    """)
    return


@app.cell(hide_code=True)
def _():
    snapshot("s04")
    return


@app.cell
def _(COMPS):
    em_comp = mo.ui.dropdown({v: k for k, v in COMPS.items()}, value=COMPS["wc22"], label="Competition")
    return (em_comp,)


@app.cell
def _(TEAM, em_comp):
    em_actions = actions_for(em_comp.value)
    _teams = sorted({TEAM[t] for t in em_actions.team_id.unique()})
    em_team = mo.ui.dropdown(_teams, value="Argentina" if "Argentina" in _teams else _teams[0], label="Team", searchable=True)
    return em_actions, em_team


@app.cell
def _(PLAYER, TEAM, em_actions, em_team):
    _tid = next(k for k, v in TEAM.items() if v == em_team.value and k in set(em_actions.team_id))
    em_team_actions = em_actions[em_actions.team_id == _tid]
    _counts = em_team_actions.player_id.dropna().astype(int).value_counts()
    _opts = {"All players": -1, **{PLAYER.get(p, str(p)): p for p in _counts.index}}
    em_player = mo.ui.dropdown(_opts, value="Lionel Messi" if "Lionel Messi" in _opts else "All players", label="Player", searchable=True)
    em_kind = mo.ui.dropdown(["Passes", "Carries", "Shots", "Take-ons", "Pressures", "Defensive actions"], value="Passes", label="Action")
    em_view = mo.ui.radio(["Map", "Heatmap"], value="Heatmap", inline=True, label="View")
    return em_kind, em_player, em_team_actions, em_view


@app.cell(hide_code=True)
def _(em_comp, em_kind, em_player, em_team, em_team_actions, em_view):
    _kinds = {
        "Passes": ["pass", "cross", "corner", "freekick", "throw_in", "goal_kick"], "Carries": ["carry"],
        "Shots": ["shot", "penalty", "shot_freekick"], "Take-ons": ["take_on"], "Pressures": ["pressure"],
        "Defensive actions": ["tackle", "interception", "clearance", "block", "recovery"],
    }
    _a = em_team_actions[em_team_actions.type.isin(_kinds[em_kind.value])]
    if em_player.value != -1:
        _a = _a[_a.player_id == em_player.value]
    _a = _a.assign(outcome=np.where(_a.result == 1, "Successful", "Unsuccessful"))
    _who = em_team.value if em_player.value == -1 else em_player.selected_key
    if em_view.value == "Heatmap":
        _g = grid_cells(21, 14)
        _ix = np.clip((_a.x / L * 21).astype(int), 0, 20)
        _iy = np.clip((_a.y / W * 14).astype(int), 0, 13)
        _cnt = pd.Series(1, index=pd.MultiIndex.from_arrays([_ix, _iy])).groupby(level=[0, 1]).sum()
        _g["n"] = [_cnt.get((i, j), 0) for i, j in zip(_g.i, _g.j)]
        _g["share"] = _g.n / max(1, _g.n.sum())
        _layer = alt.Chart(_g[_g.n > 0]).mark_rect(stroke=C["surface"], strokeWidth=1).encode(
            **xy("x0", "y0"), **xy2("x1", "y1"),
            color=alt.Color("share:Q", scale=alt.Scale(range=C["seq"]), legend=alt.Legend(title="share of actions", format=".1%", orient="bottom")),
            tooltip=[alt.Tooltip("n:Q", title="actions"), alt.Tooltip("share:Q", format=".1%")],
        )
        _chart = pitch(under=[_layer], width=620)
    else:
        _a = _a.sample(min(len(_a), 1500), random_state=0)
        _oc = alt.Scale(domain=["Successful", "Unsuccessful"], range=[C["blue"], C["muted"]])
        _moves = em_kind.value in ("Passes", "Carries", "Shots")
        _ls = []
        if _moves:
            _ls.append(alt.Chart(_a).mark_rule(strokeWidth=1, opacity=0.45).encode(**xy(), **xy2(), color=alt.Color("outcome:N", scale=_oc, legend=None)))
            _ls.append(alt.Chart(_a).mark_circle(size=14, opacity=0.8).encode(**xy("ex", "ey"), color=alt.Color("outcome:N", scale=_oc, legend=None)))
        _ls.append(alt.Chart(_a).mark_circle(size=22 if _moves else 40, opacity=0.6).encode(
            **xy(), color=alt.Color("outcome:N", scale=_oc, legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["type:N", "outcome:N", "minute:Q"]))
        _chart = pitch(*_ls, width=620)
    _n = len(em_team_actions[em_team_actions.type.isin(_kinds[em_kind.value])]) if em_player.value == -1 else len(_a)
    _title = f"{_who} · {em_kind.value.lower()} · {em_comp.selected_key}"
    mo.vstack([
        mo.hstack([em_comp, em_team, em_player], justify="start", gap=1),
        mo.hstack([em_kind, em_view], justify="start", gap=1),
        mo.hstack([avatar(em_player.value, _who, 56), mo.md(f"**{_who}** · {em_team.value}")], justify="start", align="center", gap=1) if em_player.value != -1 else mo.md(""),
        _chart.properties(title=alt.Title(_title, subtitle=f"{_n:,} actions · attacking left → right")),
    ])
    return


@app.cell(hide_code=True)
def _():
    further(
        ("mplsoccer (pitch plots in matplotlib)", "https://github.com/andrewRowlinson/mplsoccer"),
        ("Friends of Tracking (video course)", "https://www.youtube.com/@friendsoftracking755"),
        ("Soccermatics (David Sumpter's course)", "https://soccermatics.readthedocs.io/"),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5 · Tracking: 25 snapshots a second

    Events record what happens **to the ball**, about 3,500 moments a match. **Tracking data** records where
    **everyone** is, 25 times a second, which comes to roughly 3 million positions a match. This is Metrica Sports'
    anonymised sample game; the home side (blue) always attacks left → right.
    """)
    return


@app.cell(hide_code=True)
def _():
    snapshot("s05")
    return


@app.cell
def _():
    mo.stop(not FULL)
    trk = read_csv("tracking_g2_5hz.csv.gz")
    trk_events = read_csv("tracking_g2_events.csv")
    TRK_HOME = sorted({c[:-2] for c in trk.columns if c.startswith("H") and c.endswith("_x")}, key=lambda s: int(s[1:]))
    TRK_AWAY = sorted({c[:-2] for c in trk.columns if c.startswith("A") and c.endswith("_x")}, key=lambda s: int(s[1:]))
    _goals = trk_events[(trk_events.type == "SHOT") & trk_events.subtype.str.contains("GOAL", na=False)]
    TRK_GOALS = {
        f"{int(r.start_time // 60)}′ · {'Home' if r.team == 'Home' else 'Away'} goal by #{r['from'].replace('Player', '')}": int(r.start_frame)
        for _, r in _goals.iterrows()
    }
    return TRK_AWAY, TRK_GOALS, TRK_HOME, trk, trk_events


@app.cell
def _(TRK_GOALS):
    trk_goal = mo.ui.dropdown(TRK_GOALS, value=list(TRK_GOALS)[0], label="Replay")
    trk_goal
    return (trk_goal,)


@app.cell(hide_code=True)
def _(TRK_AWAY, TRK_HOME, trk, trk_goal):
    go = importlib.import_module("plotly.graph_objects")  # dynamic on purpose: the lite (WebAssembly) edition never needs plotly

    _f = trk_goal.value
    _seq = trk[(trk.frame >= _f - 25 * 12) & (trk.frame <= _f + 25 * 3)]

    def _frame_traces(r):
        hx = [r[f"{p}_x"] for p in TRK_HOME if pd.notna(r[f"{p}_x"])]
        hy = [r[f"{p}_y"] for p in TRK_HOME if pd.notna(r[f"{p}_x"])]
        ht = [p[1:] for p in TRK_HOME if pd.notna(r[f"{p}_x"])]
        ax = [r[f"{p}_x"] for p in TRK_AWAY if pd.notna(r[f"{p}_x"])]
        ay = [r[f"{p}_y"] for p in TRK_AWAY if pd.notna(r[f"{p}_x"])]
        at = [p[1:] for p in TRK_AWAY if pd.notna(r[f"{p}_x"])]
        mk = dict(size=17, line=dict(width=2, color=C["surface"]))
        return [
            go.Scatter(x=hx, y=hy, text=ht, mode="markers+text", textfont=dict(color="#ffffff", size=9), marker=dict(color=C["blue"], **mk), name="Home", hoverinfo="skip"),
            go.Scatter(x=ax, y=ay, text=at, mode="markers+text", textfont=dict(color="#ffffff", size=9), marker=dict(color=C["orange"], **mk), name="Away", hoverinfo="skip"),
            go.Scatter(x=[r.ball_x], y=[r.ball_y], mode="markers", marker=dict(color=C["ink"], size=9, line=dict(width=1.5, color=C["surface"])), name="Ball", hoverinfo="skip"),
        ]

    _rows = list(_seq.itertuples(index=False))
    _rows = [r._asdict() for r in _rows]
    _shapes = [dict(type="path", path="M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in seg), line=dict(color=C["line"], width=1.2)) for seg in PITCH_SEGS]
    _fig = go.Figure(
        data=_frame_traces(pd.Series(_rows[0])),
        frames=[go.Frame(data=_frame_traces(pd.Series(r)), name=str(k)) for k, r in enumerate(_rows)],
    )
    _fig.update_layout(
        shapes=_shapes, width=700, height=470, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=C["surface"], plot_bgcolor=C["surface"], font=dict(color=C["ink2"], family="system-ui"),
        xaxis=dict(range=[-3, 108], visible=False, fixedrange=True), yaxis=dict(range=[-3, 71], visible=False, scaleanchor="x", fixedrange=True),
        legend=dict(orientation="h", y=1.06, x=0), showlegend=True,
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.02, xanchor="left", yanchor="top", showactive=False, buttons=[
            dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=200, redraw=False), transition=dict(duration=200, easing="linear"), fromcurrent=True)]),
            dict(label="❚❚", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
        ])],
        sliders=[dict(active=0, y=-0.02, x=0.12, len=0.88, currentvalue=dict(visible=False), pad=dict(t=0), steps=[
            dict(method="animate", label="", args=[[str(k)], dict(frame=dict(duration=0, redraw=False), mode="immediate")]) for k in range(len(_rows))])],
    )
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6 · The engine room: physical data

    Differentiate positions once to get **speed**; bucket it and you get the numbers on a sports scientist's
    report: distance, high-speed running, sprints. Wearable GPS vests measure the same things in training,
    where no stadium cameras exist.
    """)
    return


@app.cell(hide_code=True)
def _():
    snapshot("s06")
    return


@app.cell
def _(TRK_AWAY, TRK_HOME, trk):
    def _speeds(df, pid):
        out = []
        for _, g in df.groupby("period"):
            dx, dy = g[f"{pid}_x"].diff(), g[f"{pid}_y"].diff()
            dt = g["time"].diff()
            v = np.sqrt(dx**2 + dy**2) / dt
            v = v.where(v < 12).rolling(5, center=True, min_periods=1).mean()  # drop glitches, 1 s smoothing
            out.append(v)
        return pd.concat(out)

    ZONES = [("Walk", 0, 2), ("Jog", 2, 4), ("Run", 4, 5.5), ("High-speed run", 5.5, 7), ("Sprint", 7, 99)]
    def _compute_physical():
        _rows, _series = [], {}
        for _pid in TRK_HOME + TRK_AWAY:
            if trk[f"{_pid}_x"].notna().sum() < 300:
                continue
            _v = _speeds(trk, _pid)
            _series[_pid] = _v
            _rec = dict(player=f"{'Home' if _pid[0] == 'H' else 'Away'} #{_pid[1:]}", pid=_pid, team="Home" if _pid[0] == "H" else "Away",
                        minutes=_v.notna().sum() * 0.2 / 60, max_kmh=_v.max() * 3.6)
            for _z, _lo, _hi in ZONES:
                _rec[_z] = float((_v.where((_v >= _lo) & (_v < _hi)) * 0.2).sum() / 1000)
            _fast = (_v >= 7).astype(int)
            _rec["sprints"] = int(((_fast.diff() == 1) & (_fast.rolling(5).sum().shift(-4) == 5)).sum())
            _rows.append(_rec)
        physical = pd.DataFrame(_rows)
        physical["km"] = physical[[z for z, _, _ in ZONES]].sum(axis=1)
        return physical, _series

    physical, speed_series = cached("physical", (), _compute_physical)
    return ZONES, physical, speed_series


@app.cell(hide_code=True)
def _(ZONES, physical):
    _long = physical.melt(id_vars=["player", "team", "km"], value_vars=[z for z, _, _ in ZONES], var_name="zone", value_name="dist")
    _zone_scale = alt.Scale(domain=[z for z, _, _ in ZONES], range=C["seq"][1:6] if not DARK else C["seq"][1:6][::-1])
    alt.Chart(_long).mark_bar(height=11, stroke=C["surface"], strokeWidth=1).encode(
        x=alt.X("dist:Q", title="distance covered (km)", stack="zero"),
        y=alt.Y("player:N", sort=alt.EncodingSortField("km", order="descending"), title=None),
        color=alt.Color("zone:N", scale=_zone_scale, sort=[z for z, _, _ in ZONES], legend=alt.Legend(title=None, orient="top")),
        order=alt.Order("zone_order:Q"),
        tooltip=["player:N", "zone:N", alt.Tooltip("dist:Q", format=".2f", title="km")],
    ).transform_calculate(
        zone_order="indexof(['Walk','Jog','Run','High-speed run','Sprint'], datum.zone)"
    ).properties(width=560, height=alt.Step(14), title=alt.Title("Distance by speed zone", subtitle="Zones in m/s: walk <2, jog 2–4, run 4–5.5, high-speed 5.5–7, sprint >7 (25.2 km/h)"))
    return


@app.cell
def _(physical):
    phys_player = mo.ui.dropdown(
        dict(zip(physical.sort_values("sprints", ascending=False).player, physical.sort_values("sprints", ascending=False).pid)),
        value=physical.sort_values("sprints", ascending=False).player.iloc[0], label="Player", searchable=True,
    )
    return (phys_player,)


@app.cell(hide_code=True)
def _(phys_player, physical, speed_series, trk):
    _v = speed_series[phys_player.value]
    _d = pd.DataFrame(dict(t=trk.loc[_v.index, "time"] / 60, v=_v * 3.6)).dropna()
    _d = _d.groupby((_d.t * 6).round() / 6).v.max().reset_index()  # 10-second peaks
    _p = physical.set_index("pid").loc[phys_player.value]
    _area = alt.Chart(_d).mark_area(color=C["blue"], opacity=0.25, line=dict(color=C["blue"], strokeWidth=1.2)).encode(
        x=alt.X("t:Q", title="match time (min)"), y=alt.Y("v:Q", title="peak speed (km/h, 10-s windows)"),
        tooltip=[alt.Tooltip("t:Q", format=".1f", title="minute"), alt.Tooltip("v:Q", format=".1f", title="km/h")],
    )
    _rule = alt.Chart(pd.DataFrame(dict(v=[25.2]))).mark_rule(color=C["orange"], strokeWidth=1.5).encode(y="v:Q")
    _lbl = alt.Chart(pd.DataFrame(dict(v=[25.2], t=[1]))).mark_text(align="left", dy=-6, color=C["ink2"]).encode(x="t:Q", y="v:Q", text=alt.value("sprint threshold"))
    mo.vstack([
        mo.hstack([
            phys_player,
            mo.stat(f"{_p.km:.1f} km", label="distance"),
            mo.stat(f"{_p.max_kmh:.1f} km/h", label="top speed"),
            mo.stat(f"{int(_p.sprints)}", label="sprints (>25.2 km/h for ≥1 s)"),
        ], widths=[2, 1, 1, 1.5], align="center"),
        alt.layer(_area, _rule, _lbl).properties(width=600, height=200),
        note("Speeds come from 5 Hz positions smoothed over 1 s, which slightly understates peak speed compared with 25 Hz."),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 7 · From pixels to coordinates

    Most tracking data now comes from **video**. A detector finds the players in each frame, and a 3 × 3
    **homography** maps pixels on the grass to metres on the pitch. Move the virtual broadcast camera below,
    then **click anywhere on the TV image** to send that pixel back to the pitch.
    """)
    return


@app.cell(hide_code=True)
def _():
    snapshot("s07")
    return


@app.cell
def _(trk):
    cam_pan = mo.ui.slider(5, 100, value=70, step=1, label="Pan (look at x =)", show_value=True)
    cam_tilt = mo.ui.slider(5, 63, value=34, step=1, label="Tilt (look at y =)", show_value=True)
    cam_zoom = mo.ui.slider(600, 2600, value=1100, step=50, label="Zoom (focal length, px)", show_value=True)
    cam_time = mo.ui.slider(0, int(trk.time.max()) - 1, value=1260, step=1, label="Match second", show_value=True)
    return cam_pan, cam_tilt, cam_time, cam_zoom


@app.cell
def _(cam_pan, cam_tilt, cam_zoom):
    CAM_W, CAM_H = 1280, 720
    CAM_POS = np.array([L / 2, -32.0, 22.0])  # behind the near touchline, 22 m up

    def camera(target, f):
        fwd = np.asarray(target, float) - CAM_POS
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.vstack([right, down, fwd])
        K = np.array([[f, 0, CAM_W / 2], [0, f, CAM_H / 2], [0, 0, 1]])
        return K, R

    def project(P, K, R):
        """World points (n×3) → pixels (n×2) and depth."""
        pc = (np.asarray(P, float) - CAM_POS) @ R.T
        uv = pc @ K.T
        return uv[:, :2] / uv[:, 2:3], pc[:, 2]

    CAM_K, CAM_R = camera([cam_pan.value, cam_tilt.value, 0.0], cam_zoom.value)
    # Ground-plane homography: pitch (X, Y, 1) → image (u, v, 1)
    CAM_H_MAT = CAM_K @ np.column_stack([CAM_R[:, 0], CAM_R[:, 1], -CAM_R @ CAM_POS])
    return CAM_H, CAM_H_MAT, CAM_K, CAM_POS, CAM_R, CAM_W, project


@app.cell(hide_code=True)
def _(CAM_H, CAM_H_MAT, CAM_K, CAM_R, CAM_W, TRK_AWAY, TRK_HOME, project, trk, cam_time):
    # Pitch lines, densified so perspective bends them correctly at the image border
    _rows = []
    for _s, _seg in enumerate(PITCH_SEGS):
        for _k in range(len(_seg) - 1):
            (_x1, _y1), (_x2, _y2) = _seg[_k], _seg[_k + 1]
            for _t in np.linspace(0, 1, 12, endpoint=_k == len(_seg) - 2):
                _rows.append((_s, _x1 + (_x2 - _x1) * _t, _y1 + (_y2 - _y1) * _t))
    _pl = pd.DataFrame(_rows, columns=["seg", "X", "Y"])
    _uv, _z = project(np.column_stack([_pl.X, _pl.Y, np.zeros(len(_pl))]), CAM_K, CAM_R)
    _pl = _pl.assign(u=_uv[:, 0], v=_uv[:, 1], ok=_z > 1).reset_index().rename(columns={"index": "o"})
    _pl = _pl[_pl.ok]

    _r = trk.iloc[(trk.time - cam_time.value).abs().argmin()]
    _ppl = [(p, "Home" if p[0] == "H" else "Away", _r[f"{p}_x"], _r[f"{p}_y"]) for p in TRK_HOME + TRK_AWAY if pd.notna(_r[f"{p}_x"])]
    _pp = pd.DataFrame(_ppl, columns=["pid", "team", "X", "Y"])
    _feet, _zf = project(np.column_stack([_pp.X, _pp.Y, np.zeros(len(_pp))]), CAM_K, CAM_R)
    _head, _ = project(np.column_stack([_pp.X, _pp.Y, np.full(len(_pp), 1.8)]), CAM_K, CAM_R)
    _pp = _pp.assign(u=_feet[:, 0], v=_feet[:, 1], uh=_head[:, 0], vh=_head[:, 1], ok=_zf > 1)
    _pp = _pp[_pp.ok]
    _ball, _ = project([[_r.ball_x, _r.ball_y, 0.2]], CAM_K, CAM_R) if pd.notna(_r.ball_x) else (np.array([[np.nan, np.nan]]), None)

    _U = alt.Scale(domain=[0, CAM_W], nice=False, zero=False)
    _V = alt.Scale(domain=[CAM_H, 0], nice=False, zero=False)
    cam_grid = pd.DataFrame([(u, v) for u in np.arange(40, CAM_W, 80) for v in np.arange(40, CAM_H, 80)], columns=["u", "v"])
    _pick = alt.selection_point(fields=["u", "v"], on="click", empty=False, name="pix")
    _img = alt.layer(
        alt.Chart(pd.DataFrame(dict(u=[0], v=[0], u2=[CAM_W], v2=[CAM_H]))).mark_rect(color=C["fill"]).encode(
            x=alt.X("u:Q", scale=_U, axis=None), y=alt.Y("v:Q", scale=_V, axis=None), x2="u2:Q", y2="v2:Q"),
        alt.Chart(_pl).mark_line(color=C["ink2"], strokeWidth=1.2, clip=True).encode(
            x=alt.X("u:Q", scale=_U, axis=None), y=alt.Y("v:Q", scale=_V, axis=None), detail="seg:N", order="o:Q"),
        alt.Chart(_pp).mark_rule(strokeWidth=4, clip=True, strokeCap="round").encode(
            x="u:Q", y="v:Q", x2="uh:Q", y2="vh:Q", color=alt.Color("team:N", scale=alt.Scale(domain=["Home", "Away"], range=[C["blue"], C["orange"]]), legend=None)),
        alt.Chart(pd.DataFrame(dict(u=[_ball[0, 0]], v=[_ball[0, 1]]))).mark_circle(color=C["ink"], size=40, opacity=1, clip=True).encode(x="u:Q", y="v:Q"),
        alt.Chart(cam_grid).mark_point(shape="cross", size=40, color=C["muted"], strokeWidth=1).encode(
            x="u:Q", y="v:Q", opacity=alt.condition(_pick, alt.value(1), alt.value(0.35))).add_params(_pick),
    ).properties(width=448, height=252, title=alt.Title("What the camera sees", subtitle="Click a + to project that pixel onto the pitch"))
    cam_image = mo.ui.altair_chart(_img, chart_selection=False, legend_selection=False)
    return cam_grid, cam_image


@app.cell(hide_code=True)
def _(CAM_H, CAM_H_MAT, CAM_W, TRK_AWAY, TRK_HOME, cam_image, cam_pan, cam_tilt, cam_time, cam_zoom, trk):
    # Camera footprint: back-project the image corners through the inverse homography
    _Hinv = np.linalg.inv(CAM_H_MAT)

    def _to_pitch(u, v):
        p = _Hinv @ np.array([u, v, 1.0])
        return p[:2] / p[2], p[2]

    _corners = []
    for _u, _v in [(0, CAM_H), (CAM_W, CAM_H), (CAM_W, 0), (0, 0)]:
        _p, _s = _to_pitch(_u, _v)
        if _s <= 0:  # above the horizon: clamp far away along the same direction
            _p, _ = _to_pitch(_u, CAM_H * 0.02)
            _p = _p + (_p - np.array([L / 2, -32.0])) * 3
        _corners.append(_p)
    _g = grid_cells(70, 45)
    _g = _g[inside_polygon(_g.cx, _g.cy, _corners)]
    _r = trk.iloc[(trk.time - cam_time.value).abs().argmin()]
    _pp = pd.DataFrame([(("Home" if p[0] == "H" else "Away"), _r[f"{p}_x"], _r[f"{p}_y"]) for p in TRK_HOME + TRK_AWAY if pd.notna(_r[f"{p}_x"])], columns=["team", "x", "y"])
    _layers = [
        alt.Chart(_pp).mark_circle(size=70, opacity=1, stroke=C["surface"], strokeWidth=1.5).encode(
            **xy(), color=alt.Color("team:N", scale=alt.Scale(domain=["Home", "Away"], range=[C["blue"], C["orange"]]), legend=None)),
        alt.Chart(pd.DataFrame(dict(x=[_r.ball_x], y=[_r.ball_y]))).mark_circle(color=C["ink"], size=30, opacity=1).encode(**xy()),
    ]
    _readout = "Click a **+** on the camera image."
    _sel = picked(cam_image, cam_grid)
    if len(_sel):
        _u0, _v0 = float(_sel.u.iloc[0]), float(_sel.v.iloc[0])
        _p, _s = _to_pitch(_u0, _v0)
        if _s > 0:
            _layers.append(alt.Chart(pd.DataFrame(dict(x=[_p[0]], y=[_p[1]]))).mark_point(shape="cross", size=260, strokeWidth=3, color=C["ink"], clip=True).encode(**xy()))
            _readout = f"Pixel **({_u0:.0f}, {_v0:.0f})** → pitch **({_p[0]:.1f} m, {_p[1]:.1f} m)**"
        else:
            _readout = f"Pixel ({_u0:.0f}, {_v0:.0f}) is above the horizon: it never touches the grass."
    _top = pitch(*_layers, under=[alt.Chart(_g).mark_rect(color=C["fill"]).encode(**xy("x0", "y0"), **xy2("x1", "y1"))],
                 width=420, title=alt.Title("…and where that is on the pitch", subtitle="Shaded: the camera's footprint on the grass"))
    _Hn = CAM_H_MAT / CAM_H_MAT[2, 2]
    _hm = "\n".join("| " + " | ".join(f"{v:9.3f}" for v in row) + " |" for row in _Hn)
    mo.vstack([
        mo.hstack([mo.vstack([cam_pan, cam_tilt]), mo.vstack([cam_zoom, cam_time])], widths="equal"),
        mo.hstack([cam_image, _top], widths="equal", align="start"),
        mo.md(_readout),
        mo.accordion({"The homography for this view (8 numbers; the 9th is a scale)": mo.md(
            "| | | |\n|---|---|---|\n" + _hm + "\n\nCalibrating a broadcast camera means estimating these numbers for **every frame**, "
            "usually from line intersections the network has learned to find.")}),
    ])
    return


@app.cell(hide_code=True)
def _():
    further(
        ("SoccerNet (benchmarks: spotting, calibration, tracking, game-state reconstruction)", "https://www.soccer-net.org/"),
        ("roboflow/sports (detection + pitch keypoints)", "https://github.com/roboflow/sports"),
        ("SkillCorner open broadcast tracking", "https://github.com/SkillCorner/opendata"),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Pixels, sound and video: three from-scratch demos

    Not everything in football arrives as a table. Below, three openly licensed files from
    [Wikimedia Commons](https://commons.wikimedia.org/) (credits at the end of this section) go through the same
    first steps a real pipeline would take: **find the players in a photo and split the teams**, **listen to a crowd**,
    and **cut a film into shots**. No deep learning: just colour, frequency and difference.
    """)
    return


@app.cell
def _():
    ph_sat = mo.ui.slider(0.10, 0.60, value=0.25, step=0.01, label="Grass saturation cut-off", show_value=True)
    ph_h = mo.ui.slider(15, 90, value=25, step=1, label="Smallest player (px tall)", show_value=True)
    ph_view = mo.ui.radio(["Photo", "Grass mask", "Players found", "Teams"], value="Teams", inline=True, label="View")
    ph_img = np.asarray(Image.open(io.BytesIO(load_bytes("media/match_photo.jpg"))).convert("RGB"))
    return ph_h, ph_img, ph_sat, ph_view


@app.cell(hide_code=True)
def _(ph_h, ph_img, ph_sat, ph_view):
    _hsv = np.asarray(Image.fromarray(ph_img).convert("HSV")).astype(float)
    _hue, _sat, _val = _hsv[..., 0] * 360 / 255, _hsv[..., 1] / 255, _hsv[..., 2] / 255
    # 1. grass = green hue with some saturation; the pitch begins where most of a row is grass
    _grass = (_hue > 60) & (_hue < 170) & (_sat > ph_sat.value) & (_val > 0.15)
    _top = int(np.argmax(_grass.mean(1) > 0.5))
    _below = np.zeros_like(_grass)
    _below[_top:] = True
    # 2. anything on the pitch that is not grass is a candidate; a tall, thin structuring element drops lines
    _fg = ndi.binary_opening(_below & ~_grass, structure=np.ones((9, 3)))
    _lab, _n = ndi.label(_fg)
    _found = []
    for _i, _sl in enumerate(ndi.find_objects(_lab)):
        _hh, _ww = _sl[0].stop - _sl[0].start, _sl[1].stop - _sl[1].start
        _m = _lab[_sl] == _i + 1
        if _m.sum() < 150 or _hh < ph_h.value or _hh < _ww:
            continue
        _torso = _m.copy()
        _torso[int(_hh * 0.5):] = False
        _torso[: int(_hh * 0.1)] = False
        _px = ph_img[_sl][_torso]
        if len(_px):
            _found.append(dict(box=(_sl[1].start, _sl[0].start, _sl[1].stop, _sl[0].stop), rgb=np.median(_px, 0)))
    # 3. two clusters of torso colour = two kits
    _hexes = {"light": C["orange"], "dark": C["blue"]}
    _teams = []
    if len(_found) >= 2:
        _X = np.array([f["rgb"] for f in _found])
        _km = KMeans(2, n_init=10, random_state=0).fit(_X)
        _dark = int(np.argmin(_km.cluster_centers_.mean(1)))
        _teams = ["dark" if l == _dark else "light" for l in _km.labels_]
        _centres = {"dark": _km.cluster_centers_[_dark], "light": _km.cluster_centers_[1 - _dark]}
    _canvas = Image.fromarray(ph_img)
    if ph_view.value == "Grass mask":
        _rgb = np.where(_grass[..., None], np.array([46, 96, 52]), np.array([235, 235, 232])).astype(np.uint8)
        _rgb[:_top] = (120, 120, 120)
        _rgb[_fg] = (255, 140, 0)
        _canvas = Image.fromarray(_rgb)
    elif ph_view.value in ("Players found", "Teams"):
        _d = ImageDraw.Draw(_canvas)
        for _k, _f in enumerate(_found):
            _col = tuple(int(_hexes[_teams[_k]][i:i + 2], 16) for i in (1, 3, 5)) if (ph_view.value == "Teams" and _teams) else (255, 255, 255)
            _d.rectangle(_f["box"], outline=_col, width=4)
    _swatch = lambda rgb: f"<span style='display:inline-block;width:1em;height:1em;border-radius:3px;background:rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])});vertical-align:-2px;border:1px solid {C['axis']}'></span>"
    _counts = {t: _teams.count(t) for t in ("light", "dark")}
    _readout = (mo.md(f"**{len(_found)} players found** · {_swatch(_centres['light'])} light kit: {_counts['light']} · {_swatch(_centres['dark'])} dark kit: {_counts['dark']}") if _teams
                else mo.md("Too few players found. Lower the size limit."))
    mo.vstack([
        mo.md("#### Photo → players → teams"),
        mo.hstack([ph_sat, ph_h], justify="start", gap=2), ph_view,
        mo.image(_canvas, width=720, rounded=True), _readout,
        note("Steps: (1) mask the grass by hue; (2) keep tall non-grass blobs on the pitch; (3) take each blob's median torso colour; (4) k-means with k = 2. "
             "Where two players overlap, their blobs merge into one box and get one team. That is the occlusion problem, and it is why real systems track players over time."),
    ])
    return


@app.cell
def _():
    CROWD = {"Strasbourg supporters: “Vive le vent” (38 s)": "media/crowd_strasbourg.wav",
             "Frosinone away fans: “Dai Frosinone” (45 s)": "media/crowd_frosinone.wav",
             "Latvia v Gibraltar, from the stand (17 s)": "media/stand_clip_audio.wav"}
    crowd_pick = mo.ui.dropdown(list(CROWD), value=list(CROWD)[0], label="Recording", full_width=True)
    return CROWD, crowd_pick


@app.cell(hide_code=True)
def _(CROWD, crowd_pick):
    _raw = load_bytes(CROWD[crowd_pick.value])
    _sr, _x = wavfile.read(io.BytesIO(_raw))
    _x = _x.astype(float) / 32768
    _win = int(_sr * 0.1)
    _n = len(_x) // _win
    _chunks = _x[: _n * _win].reshape(_n, _win)
    _loud = 20 * np.log10(np.sqrt((_chunks**2).mean(1)) + 1e-9).clip(-60, None)
    # The same numbers as scipy.signal.spectrogram(..., noverlap=0, scaling="spectrum", mode="magnitude"), computed with a plain reshape:
    # scipy builds a virtual sliding-window view of the whole clip that is far larger than the 2 GB a 32-bit browser (Pyodide) can address.
    _frames = _chunks - _chunks.mean(1, keepdims=True)
    _w = tukey(_win, 0.25, sym=False)
    _S = np.abs(np.fft.rfft(_frames * _w, axis=1)).T / _w.sum()
    _f, _t = np.fft.rfftfreq(_win, 1 / _sr), (np.arange(_n) * _win + _win / 2) / _sr
    _keep = _f < 4000
    _bands = _S[_keep].reshape(-1, 5, _S.shape[1]).mean(1) if _keep.sum() % 5 == 0 else _S[_keep][: (_keep.sum() // 5) * 5].reshape(-1, 5, _S.shape[1]).mean(1)
    _db = (20 * np.log10(_bands + 1e-9)).clip(-90, None)
    _fb = _f[_keep][: _bands.shape[0] * 5].reshape(-1, 5).mean(1)
    _centroid = (_S[_keep] * _f[_keep, None]).sum(0) / (_S[_keep].sum(0) + 1e-9)
    # thin the grid for the browser: 0.2 s x 100 Hz cells
    _tc = (_t.size // 2) * 2
    _bc = (_db.shape[0] // 2) * 2
    _g = _db[:_bc, :_tc].reshape(_bc // 2, 2, _tc // 2, 2).mean((1, 3))
    _tt, _ff = _t[:_tc].reshape(-1, 2).mean(1), _fb[:_bc].reshape(-1, 2).mean(1)
    _cells = pd.DataFrame(dict(t0=np.tile(_tt - 0.1, len(_ff)), t1=np.tile(_tt + 0.1, len(_ff)),
                               f0=np.repeat(_ff - 50, len(_tt)), f1=np.repeat(_ff + 50, len(_tt)), db=_g.ravel()))
    _tx = alt.Scale(domain=[0, float(_t.max()) + 0.1], nice=False)
    _spec = alt.layer(
        alt.Chart(_cells).mark_rect().encode(
            x=alt.X("t0:Q", scale=_tx, axis=None), x2="t1:Q", y=alt.Y("f0:Q", title="Hz", scale=alt.Scale(domain=[0, 4000], nice=False)), y2="f1:Q",
            color=alt.Color("db:Q", scale=alt.Scale(range=C["seq"], domain=[float(np.percentile(_db, 5)), float(np.percentile(_db, 99.5))], clamp=True), legend=alt.Legend(title="dB", orient="right")),
            tooltip=[alt.Tooltip("t0:Q", format=".1f", title="s"), alt.Tooltip("f0:Q", format=".0f", title="Hz"), alt.Tooltip("db:Q", format=".0f", title="dB")]),
        alt.Chart(pd.DataFrame(dict(t=_t, c=_centroid))).mark_line(color=C["orange"], strokeWidth=1.2).encode(x=alt.X("t:Q", scale=_tx), y=alt.Y("c:Q", scale=alt.Scale(domain=[0, 4000], nice=False))),
    ).properties(width=620, height=220)
    _lo = alt.Chart(pd.DataFrame(dict(t=_t, l=_loud))).mark_line(color=C["blue"], strokeWidth=1.4).encode(
        x=alt.X("t:Q", title="seconds", scale=_tx), y=alt.Y("l:Q", title="dB", scale=alt.Scale(domain=[float(_loud.min()) - 2, float(_loud.max()) + 2], nice=False)),
        tooltip=[alt.Tooltip("t:Q", format=".1f", title="s"), alt.Tooltip("l:Q", format=".0f", title="dB")]).properties(width=620, height=90)
    _fig = alt.vconcat(_spec, _lo, spacing=4)
    _loudest = _t[int(np.argmax(_loud))]
    _dom = _fb[int(np.argmax(_db.mean(1)))]
    mo.vstack([
        mo.md("#### Sound → spectrogram"),
        crowd_pick, media_audio(CROWD[crowd_pick.value], _raw),
        _fig,
        mo.hstack([mo.stat(f"{_loudest:.1f} s", label="loudest moment"), mo.stat(f"{_dom:.0f} Hz", label="strongest band (0–4 kHz)"), mo.stat(f"{np.percentile(_loud, 95) - np.percentile(_loud, 5):.0f} dB", label="loudness range (5th–95th percentile)", caption="small = compressed or constant singing")], widths="equal"),
        note("A spectrogram turns sound into a picture: time across, pitch up, energy as colour. The orange line is the spectral centroid, the 'brightness' of the sound. "
             "Sung chants sit in bands of rising pitch; whistles and shrieks climb above 2 kHz. Crowd loudness has been used to study home advantage and referee decisions."),
    ])
    return


@app.cell
def _():
    vid_clip = np.load(io.BytesIO(load_bytes("media/match_report_frames.npz")))
    CLIP_FR, CLIP_FPS = vid_clip["frames"].astype(float), int(vid_clip["fps"])
    vid_stand = np.load(io.BytesIO(load_bytes("media/stand_clip_frames.npz")))
    STAND_FR, STAND_FPS = vid_stand["frames"].astype(float), int(vid_stand["fps"])
    cut_thr = mo.ui.slider(2, 9, value=3, step=0.5, label="Cut sensitivity: how many times above its neighbours", show_value=True, full_width=True)
    return CLIP_FPS, CLIP_FR, STAND_FPS, STAND_FR, cut_thr


@app.cell(hide_code=True)
def _(CLIP_FPS, CLIP_FR, cut_thr):
    _d = np.abs(np.diff(CLIP_FR, axis=0)).mean((1, 2))                     # how much the picture changed
    _med = np.array([np.median(np.delete(_d[max(0, i - 6): i + 7], i - max(0, i - 6))) for i in range(len(_d))])
    _score = _d / (_med + 1e-6)                                              # compared with the local normal
    _cand = np.where((_score > cut_thr.value) & (_d > np.percentile(_d, 80)))[0] + 1
    _cuts = []
    for _c in _cand:
        if not _cuts or _c - _cuts[-1] > 0.5 * CLIP_FPS:
            _cuts.append(int(_c))
    _starts = [0] + _cuts
    _t = np.arange(len(_d)) / CLIP_FPS
    _df = pd.DataFrame(dict(t=_t, change=_d, score=_score))
    _line = alt.Chart(_df).mark_line(color=C["blue"], strokeWidth=1.2).encode(
        x=alt.X("t:Q", title="seconds", scale=alt.Scale(domain=[0, float(_t.max())], nice=False)), y=alt.Y("change:Q", title="frame-to-frame change"),
        tooltip=[alt.Tooltip("t:Q", format=".1f"), alt.Tooltip("change:Q", format=".1f")])
    _marks = alt.Chart(pd.DataFrame(dict(t=np.array(_cuts) / CLIP_FPS))).mark_rule(color=C["orange"], strokeWidth=2).encode(x="t:Q", tooltip=[alt.Tooltip("t:Q", format=".1f", title="cut at (s)")])
    _tiles = []
    for _k, _s in enumerate(_starts):
        _im = Image.fromarray(CLIP_FR[min(_s + 2, len(CLIP_FR) - 1)].astype(np.uint8)).resize((192, 108), Image.BICUBIC)
        _tiles.append(mo.image(_im, caption=f"shot {_k + 1} · {_s / CLIP_FPS:.1f} s", rounded=True))
    mo.vstack([
        mo.md("#### Video → shots"),
        mo.md("A 2026 news report on a friendly match: presenter, flag, line-ups, play, an interview. A **cut** is a frame that differs from its predecessor far more than neighbouring frames do. "
              "That is the first step of automatic highlight and replay detection."),
        cut_thr,
        alt.layer(_line, _marks).properties(width=620, height=150, title=alt.Title(f"{len(_cuts)} cuts found", subtitle="Blue: how much the picture changes each frame. Orange: detected cuts")),
        mo.hstack(_tiles, wrap=True, justify="start", gap=0.6),
        mo.accordion({"Watch the clip (48 s)": media_video("media/match_report.webm")}),
    ])
    return


@app.cell(hide_code=True)
def _(STAND_FPS, STAND_FR):
    _motion = np.abs(np.diff(STAND_FR, axis=0)).mean((1, 2))
    _sr, _x = wavfile.read(io.BytesIO(load_bytes("media/stand_clip_audio.wav")))
    _x = _x.astype(float) / 32768
    _w = int(_sr / STAND_FPS)
    _nn = min(len(_motion), len(_x) // _w)
    _loud = 20 * np.log10(np.sqrt((_x[: _nn * _w].reshape(_nn, _w) ** 2).mean(1)) + 1e-9).clip(-60, None)
    _tt = np.arange(_nn) / STAND_FPS
    _r = float(np.corrcoef(_motion[:_nn], _loud)[0, 1])
    _df = pd.DataFrame(dict(t=_tt, motion=_motion[:_nn], loud=_loud))
    _a = alt.Chart(_df).mark_line(color=C["blue"]).encode(x=alt.X("t:Q", title=None, scale=alt.Scale(domain=[0, float(_tt.max())], nice=False)), y=alt.Y("motion:Q", title="motion (frame change)"),
                                                         tooltip=[alt.Tooltip("t:Q", format=".1f"), alt.Tooltip("motion:Q", format=".1f")]).properties(width=620, height=110)
    _b = alt.Chart(_df).mark_line(color=C["orange"]).encode(x=alt.X("t:Q", title="seconds"), y=alt.Y("loud:Q", title="loudness (dB)", scale=alt.Scale(zero=False)),
                                                           tooltip=[alt.Tooltip("t:Q", format=".1f"), alt.Tooltip("loud:Q", format=".0f")]).properties(width=620, height=110)
    mo.vstack([
        mo.md("#### Video + sound → one clip, two signals"),
        mo.md("A phone clip from the stand at Latvia v Gibraltar: one continuous take, no cuts, so the cut detector above finds nothing. "
              "The same file carries a picture and a soundtrack, and each gives its own time series."),
        media_video("media/stand_clip.webm"),
        alt.vconcat(_a, _b, spacing=4),
        mo.md(f"Correlation between motion and loudness in this clip: **{_r:+.2f}**. One clip proves nothing, but it shows how pictures and sound can be fused into features."),
    ])
    return


@app.cell(hide_code=True)
def _():
    _media = pd.DataFrame(read_json("media.json"))
    _rows = "\n".join(f"| {r.kind} | [{r.title[:70]}{'…' if len(r.title) > 70 else ''}]({r.page_url}) | {r.author} | {r.license} |" for r in _media.itertuples())
    mo.accordion({"Credits for the media above": mo.md("All files are from Wikimedia Commons; the notebook resizes or re-encodes them, which changes the files but not the licences.\n\n| | Title | Author | Licence |\n|---|---|---|---|\n" + _rows)})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 8 · 150 years of results

    The oldest and most open layer: **international results since 1872**, maintained by volunteers and released as CC0.
    Even here there are patterns. Goals per game have fallen, and **home advantage** has shrunk.
    """)
    return


@app.cell
def _():
    intl = read_csv("intl_results.csv.gz", parse_dates=["date"]).dropna(subset=["home_score", "away_score"])
    intl = intl.assign(
        year=intl.date.dt.year, decade=(intl.date.dt.year // 10) * 10, goals=intl.home_score + intl.away_score,
        outcome=np.select([intl.home_score > intl.away_score, intl.home_score < intl.away_score], ["Home win", "Away win"], "Draw"),
    )
    intl_scope = mo.ui.dropdown(["All matches", "Friendlies only", "Competitive only", "FIFA World Cup only"], value="All matches", label="Matches")
    return intl, intl_scope


@app.cell(hide_code=True)
def _(intl, intl_scope):
    _d = {"All matches": intl, "Friendlies only": intl[intl.tournament == "Friendly"],
          "Competitive only": intl[intl.tournament != "Friendly"], "FIFA World Cup only": intl[intl.tournament == "FIFA World Cup"]}[intl_scope.value]
    _by = _d.groupby("decade").agg(n=("goals", "size"), gpg=("goals", "mean")).reset_index()
    _vol = alt.Chart(_by).mark_bar(color=C["blue"], cornerRadiusEnd=3, width=14).encode(
        x=alt.X("decade:Q", title=None, axis=alt.Axis(format="d")), y=alt.Y("n:Q", title="matches"),
        tooltip=[alt.Tooltip("decade:Q", format="d"), alt.Tooltip("n:Q", format=",")],
    ).properties(width=270, height=170, title="Matches per decade")
    _gpg = alt.Chart(_by).mark_line(color=C["blue"], point=alt.OverlayMarkDef(size=36, color=C["blue"])).encode(
        x=alt.X("decade:Q", title=None, axis=alt.Axis(format="d")), y=alt.Y("gpg:Q", title="goals per match", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("decade:Q", format="d"), alt.Tooltip("gpg:Q", format=".2f")],
    ).properties(width=270, height=170, title="Goals per match")
    _ha = _d[~_d.neutral.astype(bool)].groupby(["decade", "outcome"]).size().rename("n").reset_index()
    _ha["share"] = _ha.n / _ha.groupby("decade").n.transform("sum")
    _ha_chart = alt.Chart(_ha).mark_bar(width=16, stroke=C["surface"], strokeWidth=1).encode(
        x=alt.X("decade:Q", title=None, axis=alt.Axis(format="d"), scale=alt.Scale(domain=[1865, 2025])),
        y=alt.Y("share:Q", title="share of non-neutral matches", axis=alt.Axis(format=".0%"), stack="normalize"),
        color=alt.Color("outcome:N", scale=alt.Scale(domain=["Home win", "Draw", "Away win"], range=[C["blue"], C["axis"], C["orange"]]), legend=alt.Legend(title=None, orient="top")),
        order=alt.Order("ord:Q"),
        tooltip=[alt.Tooltip("decade:Q", format="d"), "outcome:N", alt.Tooltip("share:Q", format=".0%")],
    ).transform_calculate(ord="indexof(['Home win','Draw','Away win'], datum.outcome)").properties(width=580, height=180, title="Home advantage, decade by decade")
    mo.vstack([intl_scope, mo.hstack([_vol, _gpg], justify="start", gap=2), _ha_chart])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 9 · Bodies and markets

    Two more layers sit outside the 105 × 68 m rectangle: **physiological** data from training and recovery,
    and **market** data about players as assets. Both are mostly private, so this section explains how they work.
    """)
    return


@app.cell
def _():
    body_spike = mo.ui.slider(0, 120, value=60, step=10, label="Load spike in week 20 (% above normal)", show_value=True)
    body_acute = mo.ui.slider(3, 10, value=7, step=1, label="Acute window (days)", show_value=True)
    body_chronic = mo.ui.slider(14, 42, value=28, step=7, label="Chronic window (days)", show_value=True)
    return body_acute, body_chronic, body_spike


@app.cell(hide_code=True)
def _(body_acute, body_chronic, body_spike):
    # A synthetic season of daily training load (arbitrary units): pre-season ramp, weekly match cycle, one spike.
    _rng = np.random.default_rng(7)
    _days = np.arange(7 * 36)
    _week = np.array([0.55, 0.8, 1.0, 0.7, 0.45, 1.6, 0.0])  # Mon..Sun, match on Saturday
    _ramp = np.clip(0.45 + _days / 60, 0, 1.0)
    _load = 500 * _week[_days % 7] * _ramp * _rng.normal(1, 0.12, len(_days))
    _spike = (_days >= 7 * 20) & (_days < 7 * 21)
    _load = np.where(_spike, _load * (1 + body_spike.value / 100), _load).clip(0)
    _s = pd.Series(_load)
    _acwr = _s.rolling(body_acute.value).mean() / _s.rolling(body_chronic.value).mean()
    _d = pd.DataFrame(dict(day=_days, load=_load, acwr=_acwr, zone=np.where(_acwr > 1.5, "ACWR > 1.5", "ACWR ≤ 1.5")))
    _bars = alt.Chart(_d).mark_bar(color=C["blue"], width=2).encode(
        x=alt.X("day:Q", title=None, scale=alt.Scale(domain=[0, 252], nice=False)), y=alt.Y("load:Q", title="daily load (a.u.)"), tooltip=["day:Q", alt.Tooltip("load:Q", format=".0f")]
    ).properties(width=600, height=120, title=alt.Title("A synthetic season of training load", subtitle="Weekly match cycle, pre-season ramp, one overload week"))
    _band = alt.Chart(pd.DataFrame(dict(lo=[0.8], hi=[1.3]))).mark_rect(color=C["fill"]).encode(y="lo:Q", y2="hi:Q")
    _line = alt.Chart(_d.dropna()).mark_line(color=C["blue"]).encode(
        x=alt.X("day:Q", title="day of season", scale=alt.Scale(domain=[0, 252], nice=False)), y=alt.Y("acwr:Q", title="acute : chronic ratio", scale=alt.Scale(domain=[0.4, 2.2], clamp=True)),
        tooltip=["day:Q", alt.Tooltip("acwr:Q", format=".2f")],
    )
    _hot = alt.Chart(_d[_d.acwr > 1.5]).mark_circle(color=C["orange"], size=40, opacity=1).encode(x="day:Q", y="acwr:Q")
    _ratio = alt.layer(_band, _line, _hot).properties(width=600, height=170, title=alt.Title("Acute : chronic workload ratio", subtitle="Shaded: the oft-quoted 0.8–1.3 'sweet spot'. Orange: days above 1.5"))
    mo.vstack([
        mo.md("### Bodies: load monitoring\nClubs track every player's **load** (GPS distance, accelerations, heart rate, perceived exertion) to balance fitness against injury risk. "
              "A popular summary is the **acute : chronic workload ratio**: last week's load against the last month's. Push a spike in and watch it light up."),
        mo.hstack([body_spike, body_acute, body_chronic], widths="equal"),
        mo.vstack([_bars, _ratio], gap=0),
        mo.callout(mo.md("**Use with care.** ACWR became popular after [Gabbett (2016)](https://doi.org/10.1136/bjsports-2015-095788), and was later criticised as a statistical artefact by "
                         "[Impellizzeri et al. (2020)](https://doi.org/10.1123/ijspp.2019-0864) and others. Open, real-world data: [SoccerMon](https://zenodo.org/records/10033832) "
                         "(two elite women's teams; GPS, wellness, injuries; CC BY 4.0). The load series above is **synthetic**."), kind="warn"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Markets: what is a player worth?

    [Transfermarkt](https://www.transfermarkt.com/)'s crowd-sourced market values are the best-known open price signal in football,
    and [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) republishes them as CC0 tables.
    Studies have found the crowd's estimates track real fees well ([Müller, Simons & Weinmann 2017](https://doi.org/10.1016/j.ejor.2017.05.005)).
    The shape of a career shows up when you line valuations up by age.
    """)
    return


@app.cell
def _():
    tm_age = read_csv("tm_age_curve.csv")
    tm_pos = mo.ui.radio(["All", "Attack", "Midfield", "Defender", "Goalkeeper"], value="All", inline=True, label="Position")
    return tm_age, tm_pos


@app.cell(hide_code=True)
def _(tm_age, tm_pos):
    _d = tm_age[tm_age.position == tm_pos.value]
    _band = alt.Chart(_d).mark_area(color=C["blue"], opacity=0.18).encode(
        x=alt.X("age:Q", title="age at valuation", scale=alt.Scale(domain=[16, 38], nice=False)),
        y=alt.Y("p25:Q", title="market value (€ millions, log scale)", scale=alt.Scale(type="log")), y2="p75:Q")
    _top = alt.Chart(_d).mark_line(color=C["orange"]).encode(x="age:Q", y="p95:Q", tooltip=["age:Q", alt.Tooltip("p95:Q", format=".1f", title="top 5% (€m)")])
    _med = alt.Chart(_d).mark_line(color=C["blue"]).encode(
        x="age:Q", y="p50:Q",
        tooltip=["age:Q", alt.Tooltip("p50:Q", format=".2f", title="median (€m)"), alt.Tooltip("n:Q", format=",", title="valuations")])
    _lbl = alt.Chart(pd.concat([_d.nlargest(1, "p50").assign(t="median", v=lambda x: x.p50), _d.nlargest(1, "p95").assign(t="top 5%", v=lambda x: x.p95)])).mark_text(
        align="left", dx=6, dy=-8, color=C["ink2"]).encode(x="age:Q", y="v:Q", text="t:N")
    mo.vstack([
        tm_pos,
        alt.layer(_band, _med, _top, _lbl).properties(width=600, height=260, title=alt.Title(
            "Market value by age", subtitle="Median (blue, shaded band = middle 50%) and the top 5% (orange) of valuations at each age")),
        note("Source: transfermarkt-datasets (CC0), all player valuations since 2004, aggregated by age at valuation."),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Second half · The models
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 10 · Expected goals (xG)

    **xG is the probability that a shot becomes a goal**, learned from thousands of shots like it. It is the
    "hello world" of football models: a classifier on distance, angle and context. Three are trained below,
    each scored only on matches it never saw (4-fold, split by match), next to StatsBomb's commercial model.
    """)
    return


@app.cell(hide_code=True)
def _():
    snapshot("s10")
    return


@app.cell
def _():
    mo.stop(not FULL)
    xg_scope = mo.ui.radio(["Men's + women's", "Men's only", "Women's only"], value="Men's + women's", inline=True, label="Train on")
    xg_scope
    return (xg_scope,)


@app.cell
def _(shots, xg_scope):
    def xg_features(d):
        f = pd.DataFrame(index=d.index)
        f["distance"] = d.distance
        f["angle"] = np.radians(d.angle)
        f["head"] = (d.body_part == "Head").astype(int)
        f["other_body"] = (d.body_part == "Other").astype(int)
        f["penalty"] = (d.shot_type == "Penalty").astype(int)
        f["free_kick"] = (d.shot_type == "Free Kick").astype(int)
        f["first_time"] = d.first_time
        f["under_pressure"] = d.under_pressure
        f["one_on_one"] = d.one_on_one
        f["open_goal"] = d.open_goal
        f["follows_dribble"] = d.follows_dribble
        f["volley"] = d.technique.isin(["Volley", "Half Volley"]).astype(int)
        f["from_corner"] = (d.play_pattern == "From Corner").astype(int)
        f["from_counter"] = (d.play_pattern == "From Counter").astype(int)
        f["n_def_cone"] = d.n_def_cone.fillna(0)
        f["gk_in_cone"] = d.gk_in_cone.fillna(1)
        f["gk_dist_goal"] = d.gk_dist_goal.fillna(1.0).clip(0, 40)
        f["nearest_def"] = d.nearest_def.fillna(8).clip(0, 20)
        f["n_def_5m"] = d.n_def_5m.fillna(0)
        return f

    XG_MODELS = {
        "Logistic · distance + angle": (["distance", "angle"], lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))),
        "Logistic · + context & freeze frame": (None, lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
        "Gradient boosting · all features": (None, lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)),
    }
    xg_X = xg_features(shots)
    _y = shots.goal.values
    _scope = {"Men's + women's": np.ones(len(shots), bool), "Men's only": (shots.gender == "men").values,
              "Women's only": (shots.gender == "women").values}[xg_scope.value]
    def _train_xg():
        xg_oof, xg_fitted = {}, {}
        for _name, (_cols, _make) in XG_MODELS.items():
            _cols = _cols or list(xg_X.columns)
            _Xs, _ys, _idx = xg_X.loc[_scope, _cols].values, _y[_scope], np.flatnonzero(_scope)
            _pred = np.full(len(shots), np.nan)
            for _tr, _te in GroupKFold(n_splits=4).split(_Xs, _ys, groups=shots.match_id.values[_scope]):
                _pred[_idx[_te]] = _make().fit(_Xs[_tr], _ys[_tr]).predict_proba(_Xs[_te])[:, 1]
            _full = _make().fit(_Xs, _ys)
            if (~_scope).any():
                _pred[~_scope] = _full.predict_proba(xg_X.loc[~_scope, _cols].values)[:, 1]
            xg_oof[_name] = _pred
            xg_fitted[_name] = (_cols, _full)
        return xg_oof, xg_fitted

    xg_oof, xg_fitted = cached("xg", (xg_scope.value,), _train_xg)
    return XG_MODELS, xg_X, xg_features, xg_fitted, xg_oof


@app.cell(hide_code=True)
def _(shots, xg_oof, xg_scope):
    _rows = []
    for _name, _p in {**xg_oof, "StatsBomb xG (commercial)": shots.sb_xg.values}.items():
        _r = dict(model=_name)
        for _g, _lab in (("men", "Men"), ("women", "Women")):
            _m = (shots.gender == _g).values & ~np.isnan(_p)
            _pp = np.clip(_p[_m], 1e-4, 1 - 1e-4)
            _r[f"{_lab} · log loss ↓"] = log_loss(shots.goal.values[_m], _pp)
            _r[f"{_lab} · AUC ↑"] = roc_auc_score(shots.goal.values[_m], _pp)
        _rows.append(_r)
    _t = pd.DataFrame(_rows)
    mo.vstack([
        mo.ui.table(_t, selection=None, show_column_summaries=False, show_download=False,
                    format_mapping={c: "{:.3f}" for c in _t.columns if c != "model"}),
        note(f"Trained on: {xg_scope.value}. Out-of-scope shots are scored by the model trained on the rest, which shows how well a model "
             "<b>transfers</b> between men's and women's football."),
    ])
    return


@app.cell
def _(XG_MODELS):
    xg_model = mo.ui.dropdown(list(XG_MODELS), value="Gradient boosting · all features", label="Model")
    xg_body = mo.ui.radio(["Foot", "Head"], value="Foot", inline=True, label="Body part")
    xg_def = mo.ui.slider(0, 4, value=0, label="Defenders in the shooting cone", show_value=True)
    xg_near = mo.ui.slider(0.5, 10, value=4, step=0.5, label="Nearest defender (m)", show_value=True)
    xg_press = mo.ui.switch(value=False, label="Under pressure")
    return xg_body, xg_def, xg_model, xg_near, xg_press


@app.cell
def _(xg_body, xg_def, xg_features, xg_fitted, xg_model, xg_near, xg_press):
    _g = grid_cells(36, 46, x0=52.5, x1=L, y0=0, y1=W)
    _dx, _dy = L - _g.cx, W / 2 - _g.cy
    _a1 = np.arctan2(W / 2 - 3.66 - _g.cy, L - _g.cx)
    _a2 = np.arctan2(W / 2 + 3.66 - _g.cy, L - _g.cx)
    _fake = pd.DataFrame(dict(
        distance=np.hypot(_dx, _dy), angle=np.degrees(np.abs(_a2 - _a1)),
        body_part="Head" if xg_body.value == "Head" else "Right Foot", shot_type="Open Play", technique="Normal", play_pattern="Regular Play",
        first_time=0, under_pressure=int(xg_press.value), one_on_one=0, open_goal=0, follows_dribble=0,
        n_def_cone=xg_def.value, gk_in_cone=1, gk_dist_goal=1.0, nearest_def=xg_near.value, n_def_5m=min(xg_def.value + 1, 3) if xg_near.value < 5 else 0,
    ))
    _cols, _model = xg_fitted[xg_model.value]
    xg_surface = _g.assign(xg=_model.predict_proba(xg_features(_fake)[_cols].values)[:, 1])
    _pick = alt.selection_point(fields=["i", "j"], on="click", empty=False, name="xg_cell")
    _heat = alt.Chart(xg_surface).mark_rect().encode(
        **xy("x0", "y0", "vhalf"), **xy2("x1", "y1", "vhalf"),
        color=alt.Color("xg:Q", scale=alt.Scale(range=C["seq"], domain=[0, 0.6], clamp=True), legend=alt.Legend(title="xG", orient="bottom", format=".0%")),
        opacity=alt.condition(_pick, alt.value(1), alt.value(0.92)),
        stroke=alt.condition(_pick, alt.value(C["ink"]), alt.value(None)),
        tooltip=[alt.Tooltip("xg:Q", format=".1%", title="xG")],
    ).add_params(_pick)
    xg_chart = mo.ui.altair_chart(pitch(view="vhalf", width=520, under=[_heat]), chart_selection=False, legend_selection=False)
    return xg_chart, xg_surface


@app.cell(hide_code=True)
def _(shots, xg_body, xg_chart, xg_def, xg_model, xg_near, xg_press, xg_surface):
    _sel = picked(xg_chart, xg_surface)
    if len(_sel):
        _c = _sel.iloc[0]
    else:
        _c = xg_surface.iloc[((xg_surface.cx - 94) ** 2 + (xg_surface.cy - 34) ** 2).argmin()]
    _near = shots[(shots.shot_type == "Open Play") & (np.hypot(shots.x - _c.cx, shots.y - _c.cy) < 2.5)
                  & ((shots.body_part == "Head") == (xg_body.value == "Head"))]
    _rate = f"{_near.goal.mean():.0%}" if len(_near) else "–"
    _readout = mo.vstack([
        mo.stat(f"{_c.xg:.0%}", label="model xG at this spot", caption=f"{L - _c.cx:.0f} m out, {abs(_c.cy - W / 2):.0f} m off-centre"),
        mo.stat(_rate, label="real goal rate nearby", caption=f"{len(_near)} open-play {xg_body.value.lower()} shots within 2.5 m"),
        note("Click a cell to move the probe."),
    ])
    mo.vstack([
        mo.hstack([xg_model, xg_body, xg_press], justify="start", gap=1.5),
        mo.hstack([xg_def, xg_near], justify="start", gap=1.5),
        mo.hstack([xg_chart, _readout], widths=[3, 1.2], align="center"),
    ])
    return


@app.cell(hide_code=True)
def _(shots, xg_model, xg_oof):
    _p = xg_oof[xg_model.value]
    _rows = []
    for _name, _pred in ((f"Ours ({xg_model.value.split(' · ')[0]})", _p), ("StatsBomb", shots.sb_xg.values)):
        _d = pd.DataFrame(dict(p=_pred, y=shots.goal.values)).dropna()
        _d["bin"] = pd.qcut(_d.p.rank(method="first"), 12, labels=False)
        _b = _d.groupby("bin").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).assign(model=_name)
        _rows.append(_b)
    _cal = pd.concat(_rows)
    _diag = alt.Chart(pd.DataFrame(dict(a=[0, 0.8], b=[0, 0.8]))).mark_line(color=C["axis"], strokeWidth=1).encode(x="a:Q", y="b:Q")
    _lines = alt.Chart(_cal).mark_line(point=alt.OverlayMarkDef(size=40)).encode(
        x=alt.X("pred:Q", title="predicted xG (bin average)", scale=alt.Scale(domain=[0, 0.8])),
        y=alt.Y("obs:Q", title="observed goal rate", scale=alt.Scale(domain=[0, 0.8])),
        color=alt.Color("model:N", scale=alt.Scale(range=[C["blue"], C["orange"]]), legend=alt.Legend(title=None, orient="top")),
        tooltip=["model:N", alt.Tooltip("pred:Q", format=".3f"), alt.Tooltip("obs:Q", format=".3f"), "n:Q"],
    )
    alt.layer(_diag, _lines).properties(width=330, height=300, title=alt.Title("Calibration", subtitle="On the diagonal = a 30% chance scores 30% of the time"))
    return


@app.cell
def _():
    mo.stop(not FULL)
    xg_min_shots = mo.ui.slider(5, 40, value=12, step=1, label="Minimum non-penalty shots", show_value=True)
    return (xg_min_shots,)


@app.cell(hide_code=True)
def _(shots, xg_min_shots, xg_model, xg_oof):
    _d = shots.assign(ours=xg_oof[xg_model.value]).query("shot_type != 'Penalty' and period < 5")
    _p = _d.groupby(["player_id", "player", "team"]).agg(shots=("goal", "size"), goals=("goal", "sum"), xg=("ours", "sum")).reset_index()
    _p = _p[_p.shots >= xg_min_shots.value].assign(diff=lambda d: d.goals - d.xg)
    _lab = pd.concat([_p.nlargest(4, "diff"), _p.nsmallest(2, "diff"), _p.nlargest(1, "xg")]).drop_duplicates()
    _mx = float(max(_p.goals.max(), _p.xg.max())) + 1
    _diag = alt.Chart(pd.DataFrame(dict(a=[0, _mx]))).mark_line(color=C["axis"], strokeWidth=1).encode(x="a:Q", y="a:Q")
    _pd, _pf = photo_data(_p, "player_id", "player")
    _ld, _lf = photo_data(_lab, "player_id", "player", size=44)
    _tip = photo_tip("player", "team:N", "shots:Q", "goals:Q", alt.Tooltip("xg:Q", format=".1f"), alt.Tooltip("diff:Q", format="+.1f", title="goals − xG"))
    _dots = alt.Chart(_pd).transform_lookup(lookup="_pid", from_=_pf).mark_circle(size=50, color=C["blue"], opacity=0.6, stroke=C["surface"], strokeWidth=1).encode(
        x=alt.X("xg:Q", title="expected goals (our model, out of sample)"), y=alt.Y("goals:Q", title="non-penalty goals"), tooltip=_tip)
    _faces = alt.Chart(_ld).transform_lookup(lookup="_pid", from_=_lf).mark_image(width=22, height=22, xOffset=-17).encode(x="xg:Q", y="goals:Q", url="_face:N", tooltip=_tip)
    _txt = alt.Chart(_ld).transform_lookup(lookup="_pid", from_=_lf).mark_text(align="left", dx=7, fontSize=10, color=C["ink2"]).encode(x="xg:Q", y="goals:Q", text="player:N", tooltip=_tip)
    mo.vstack([
        xg_min_shots,
        alt.layer(_diag, _dots, _faces, _txt).properties(width=600, height=320, title=alt.Title(
            "Finishing: goals vs expectation", subtitle="Above the line = scored more than the chances suggested. Small samples: most of this is luck.")),
    ])
    return


@app.cell
def _(PLAYER, matches, shots, xg_model, xg_oof):
    shot_frames = read_csv("shot_frames.csv.gz")
    _d = shots.assign(ours=xg_oof[xg_model.value]).query("shot_type != 'Penalty' and period < 5 and shot_id in @shot_frames.shot_id")
    _m = matches.set_index("match_id")
    def _lab(r, tag):
        mm = _m.loc[r.match_id]
        opp = mm.away if r.team == mm.home else mm.home
        return f"{tag} {r.ours:.2f} xG · {r.player} ({r.team}) v {opp}, {r.minute}′ · {mm.competition}"
    _opts = {}
    for r in _d[_d.goal == 1].nsmallest(8, "ours").itertuples():
        _opts[_lab(r, "⚽ Goal from")] = r.shot_id
    for r in _d[_d.goal == 0].nlargest(8, "ours").itertuples():
        _opts[_lab(r, "✗ Miss from")] = r.shot_id
    xg_pick = mo.ui.dropdown(_opts, value=list(_opts)[0], label="Shot", full_width=True)
    return shot_frames, xg_pick


@app.cell(hide_code=True)
def _(PLAYER, shot_frames, shots, xg_model, xg_oof, xg_pick):
    _s = shots.assign(ours=xg_oof[xg_model.value]).set_index("shot_id").loc[xg_pick.value]
    _f = shot_frames[shot_frames.shot_id == xg_pick.value].assign(
        who=lambda d: np.where(d.teammate == 1, "Teammate", np.where(d.keeper == 1, "Goalkeeper", "Defender")),
        name=lambda d: d.player_id.map(PLAYER))
    _f["name"] = _f["name"].fillna(_f["who"])
    _fd, _ffaces = photo_data(_f, "player_id", "name")
    _cone = pd.DataFrame(dict(x=[_s.x, L, L, _s.x], y=[_s.y, W / 2 - 3.66, W / 2 + 3.66, _s.y], o=range(4)))
    _shooter = pd.DataFrame(dict(x=[_s.x], y=[_s.y], ex=[_s.end_x if pd.notna(_s.end_x) else L], ey=[_s.end_y if pd.notna(_s.end_y) else W / 2], name=[_s.player], player_id=[_s.player_id]))
    _sd, _sfaces = photo_data(_shooter, "player_id", "name")
    _ch = pitch(
        alt.Chart(_cone).mark_line(color=C["muted"], strokeWidth=1).encode(**xy(view="vhalf"), order="o:Q"),
        alt.Chart(_shooter).mark_rule(color=C["ink"], strokeWidth=2).encode(**xy(view="vhalf"), **xy2(view="vhalf")),
        alt.Chart(_fd).transform_lookup(lookup="_pid", from_=_ffaces).mark_point(filled=True, size=140, opacity=1, stroke=C["surface"], strokeWidth=1.5).encode(
            **xy(view="vhalf"), color=alt.Color("who:N", scale=alt.Scale(domain=["Teammate", "Defender", "Goalkeeper"], range=[C["blue"], C["orange"], C["ink"]]), legend=alt.Legend(title=None, orient="bottom")),
            shape=alt.Shape("who:N", scale=alt.Scale(domain=["Teammate", "Defender", "Goalkeeper"], range=["circle", "circle", "square"]), legend=None),
            tooltip=photo_tip("name", "who:N")),
        alt.Chart(_sd).transform_lookup(lookup="_pid", from_=_sfaces).mark_point(shape="diamond", filled=True, size=220, color=C["blue"], stroke=C["ink"], strokeWidth=2).encode(**xy(view="vhalf"), tooltip=photo_tip("name")),
        view="vhalf", width=480,
        title=alt.Title(f"{_s.player}: {'GOAL' if _s.goal else _s.outcome}", subtitle=f"Our xG {_s.ours:.2f} · StatsBomb xG {_s.sb_xg:.2f} · {_s.distance:.0f} m · {_s.body_part} · {int(_s.n_def_cone)} defender(s) in the cone"),
    )
    mo.vstack([
        mo.md("### Why that number?\nThe freeze frame shows what the model saw: the shooting cone, who is in it and where the keeper stands. These are the season's most surprising goals and misses."),
        mo.hstack([avatar(_s.player_id, _s.player, 56), xg_pick], widths=[1, 14], align="center", gap=1), _ch,
    ])
    return


@app.cell(hide_code=True)
def _():
    further(
        ("Pollard & Reep 1997 (an early shot model)", "https://doi.org/10.1111/1467-9884.00108"),
        ("Rathke 2017, xG and shot efficiency", "https://doi.org/10.14198/jhse.2017.12.Proc2.05"),
        ("soccer_xg (open xG toolkit)", "https://github.com/ML-KULeuven/soccer_xg"),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 11 · Expected threat (xT)

    xG only rewards the shot. **Expected threat** ([Karun Singh, 2019](https://karun.in/blog/expected-threat.html)) also values the
    passes and carries before it. Split the pitch into a grid and ask, for each cell: *if I have the ball here, how likely
    is my team to score in the next few actions?* The answer is a **Markov chain** solved by value iteration, so
    drag the slider from 1 upwards to watch value flow backwards from the goal.
    """)
    return


@app.cell(hide_code=True)
def _():
    snapshot("s11")
    return


@app.cell
def _(COMPS):
    xt_comps = mo.ui.multiselect({v: k for k, v in COMPS.items()}, value=[COMPS[k] for k in ("wc22", "eu24", "wwc23", "weu25")], label="Learn from")
    xt_iter = mo.ui.slider(0, 25, value=25, label="Moves ahead (iterations)", show_value=True, full_width=True)
    return xt_comps, xt_iter


@app.cell
def _(xt_comps):
    XT_NX, XT_NY = 16, 12

    def xt_cell(x, y):
        return (np.clip((np.asarray(x) / L * XT_NX).astype(int), 0, XT_NX - 1) * XT_NY
                + np.clip((np.asarray(y) / W * XT_NY).astype(int), 0, XT_NY - 1))

    xt_actions = pd.concat([actions_for(c) for c in (xt_comps.value or ["wc22"])], ignore_index=True)
    _moves = xt_actions[xt_actions.type.isin(MOVE_TYPES)].dropna(subset=["x", "ex"])
    _shots = xt_actions[xt_actions.type.isin(["shot", "shot_freekick"])]
    _N = XT_NX * XT_NY
    _nm = np.bincount(xt_cell(_moves.x, _moves.y), minlength=_N)
    _ns = np.bincount(xt_cell(_shots.x, _shots.y), minlength=_N)
    _ng = np.bincount(xt_cell(_shots.x, _shots.y), weights=_shots.result, minlength=_N)
    _ok = _moves[_moves.result == 1]
    xt_T = np.zeros((_N, _N))
    np.add.at(xt_T, (xt_cell(_ok.x, _ok.y), xt_cell(_ok.ex, _ok.ey)), 1)
    xt_T /= np.maximum(_nm, 1)[:, None]
    _tot = np.maximum(_nm + _ns, 1)
    _s, _m, _g = _ns / _tot, _nm / _tot, _ng / np.maximum(_ns, 1)
    xt_hist = [np.zeros(_N)]
    for _ in range(25):
        xt_hist.append(_s * _g + _m * (xt_T @ xt_hist[-1]))
    xt_grid = grid_cells(XT_NX, XT_NY).assign(cell=lambda d: d.i * XT_NY + d.j)
    return XT_NX, XT_NY, xt_T, xt_actions, xt_cell, xt_grid, xt_hist


@app.cell
def _(xt_grid, xt_hist, xt_iter):
    _d = xt_grid.assign(xt=xt_hist[xt_iter.value])
    _pick = alt.selection_point(fields=["cell"], on="click", empty=False, name="xt_pick")
    _heat = alt.Chart(_d).mark_rect(stroke=C["surface"], strokeWidth=1).encode(
        **xy("x0", "y0"), **xy2("x1", "y1"),
        color=alt.Color("xt:Q", scale=alt.Scale(range=C["seq"], type="pow", exponent=0.5, domain=[0, max(0.05, float(xt_hist[-1].max()))]),
                        legend=alt.Legend(title="xT", orient="bottom", format=".2f")),
        strokeWidth=alt.condition(_pick, alt.value(3), alt.value(1)),
        stroke=alt.condition(_pick, alt.value(C["ink"]), alt.value(C["surface"])),
        tooltip=[alt.Tooltip("xt:Q", format=".3f", title="xT")],
    ).add_params(_pick)
    xt_chart = mo.ui.altair_chart(pitch(under=[_heat], width=440, title=alt.Title("Expected threat", subtitle="Click a cell to see where the ball goes next")),
                                  chart_selection=False, legend_selection=False)
    return (xt_chart,)


@app.cell(hide_code=True)
def _(xt_T, xt_chart, xt_comps, xt_grid, xt_iter):
    _sel = picked(xt_chart, xt_grid)
    _c = int(_sel.cell.iloc[0]) if len(_sel) else int(xt_grid.query("i == 9 and j == 6").cell.iloc[0])
    _d = xt_grid.assign(p=xt_T[_c])
    _src = xt_grid[xt_grid.cell == _c]
    _heat = alt.Chart(_d[_d.p > 0.002]).mark_rect(stroke=C["surface"], strokeWidth=1).encode(
        **xy("x0", "y0"), **xy2("x1", "y1"),
        color=alt.Color("p:Q", scale=alt.Scale(range=C["seq2"]), legend=alt.Legend(title="share of moves", orient="bottom", format=".0%")),
        tooltip=[alt.Tooltip("p:Q", format=".1%", title="probability")],
    )
    _box = alt.Chart(_src).mark_rect(fill=None, stroke=C["ink"], strokeWidth=2.5).encode(**xy("x0", "y0"), **xy2("x1", "y1"))
    _moves = pitch(_box, under=[_heat], width=440, title=alt.Title("Where successful moves from that cell end up", subtitle="The transition matrix, one row at a time"))
    mo.vstack([xt_comps, xt_iter, mo.hstack([xt_chart, _moves], widths="equal", align="start")])
    return


@app.cell
def _():
    mo.stop(not FULL)
    xt_min = mo.ui.slider(90, 900, value=270, step=90, label="Minimum minutes played", show_value=True)
    return (xt_min,)


@app.cell(hide_code=True)
def _(PLAYER, TEAM, appearances, xt_actions, xt_cell, xt_hist, xt_min):
    _v = xt_hist[-1]
    _ok = xt_actions[xt_actions.type.isin(MOVE_TYPES) & (xt_actions.result == 1)].dropna(subset=["ex"])
    _ok = _ok.assign(xt=_v[xt_cell(_ok.ex, _ok.ey)] - _v[xt_cell(_ok.x, _ok.y)])
    _mins = appearances[appearances.match_id.isin(xt_actions.match_id.unique())].groupby("player_id").minutes.sum()
    _p = _ok.groupby("player_id").agg(xt=("xt", "sum"), team=("team_id", "first")).join(_mins, how="inner")
    _p = _p[_p.minutes >= xt_min.value].assign(per90=lambda d: d.xt / d.minutes * 90, name=lambda d: d.index.map(PLAYER), team=lambda d: d.team.map(TEAM))
    _top = _p.nlargest(15, "per90").reset_index()
    _top = _top.assign(label=_top.name + " · " + _top.team)
    _td, _tf = photo_data(_top, "player_id", "name")
    _x = alt.X("per90:Q", title="xT added per 90 minutes (passes + carries)", scale=alt.Scale(domain=[0, float(_top.per90.max()) * 1.1], nice=False))
    _y = alt.Y("label:N", sort=list(_top.label), title=None)
    _bars = alt.Chart(_td).transform_lookup(lookup="_pid", from_=_tf).mark_bar(color=C["blue"], cornerRadiusEnd=3, height=12).encode(
        x=_x, y=_y, tooltip=photo_tip("name", "team:N", alt.Tooltip("minutes:Q", format=".0f"), alt.Tooltip("per90:Q", format=".3f")))
    _faces = alt.Chart(_td).transform_lookup(lookup="_pid", from_=_tf).mark_image(width=24, height=24, xOffset=17).encode(x=_x, y=_y, url="_face:N")
    mo.vstack([
        xt_min,
        alt.layer(_bars, _faces).properties(width=480, height=alt.Step(28), title="Who moves the ball into danger?"),
        note("Photos: Wikimedia Commons, credited under References."),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 12 · Valuing every action (VAEP)

    xT values only moves, and only by location. **VAEP** ([Decroos et al., KDD 2019](https://arxiv.org/abs/1802.07127)) asks a
    classifier instead: *given the last two actions, how likely is my team to **score**, and to **concede**, within the next
    10 actions?* An action's value is how much it shifts those two probabilities, so tackles, dribbles and clearances get
    credit too. Built here from scratch with gradient boosting, and every action is scored by a model that never saw its match.
    """)
    return


@app.cell(hide_code=True)
def _():
    snapshot("s12")
    return


@app.cell
def _(COMPS):
    vaep_comp = mo.ui.dropdown({v: k for k, v in COMPS.items()}, value=COMPS["wc22"], label="Competition")
    vaep_comp
    return (vaep_comp,)


@app.cell
def _(matches, vaep_comp):
    def _compute_vaep():
        _a = actions_for(vaep_comp.value)
        _a = _a[_a.type != "pressure"].sort_values(["match_id", "period", "idx"]).reset_index(drop=True)
        _teams = matches.set_index("match_id")[["home_id", "away_id"]]
        _a["opp_id"] = _a.match_id.map(_teams.home_id + _teams.away_id) - _a.team_id
        _a["t"] = _a.minute * 60 + _a.second
        _goal_for = np.where(_a.type.isin(SHOT_TYPES) & (_a.result == 1), _a.team_id, np.where(_a.type == "own_goal", _a.opp_id, -1))
        _a["goal_for"] = _goal_for
        _same = (_a.match_id == _a.match_id.shift(1)) & (_a.period == _a.period.shift(1))
        _prev = _a.shift(1)
        _st = (_prev.team_id == _a.team_id) & _same

        def _persp(col, flip):
            v = np.where(_st, _prev[col], flip - _prev[col])
            return np.where(_same, v, _a[col].fillna(0))

        _types = sorted(_a.type.unique())
        _ex, _ey = _a.ex.fillna(_a.x), _a.ey.fillna(_a.y)
        F = pd.DataFrame({
            "x": _a.x, "y": _a.y, "ex": _ex, "ey": _ey, "dx": _ex - _a.x,
            "d_goal": np.hypot(L - _a.x, W / 2 - _a.y), "d_goal_end": np.hypot(L - _ex, W / 2 - _ey),
            "ang_end": np.abs(np.arctan2(W / 2 - _ey, L - _ex)), "result": _a.result,
            "head": (_a.body == "head").astype(int), "under_pressure": _a.under_pressure,
            "p_x": _persp("x", L), "p_y": _persp("y", W), "p_ex": _persp("ex", L), "p_ey": _persp("ey", W),
            "p_result": np.where(_same, _prev.result, 1), "same_team": _st.astype(int),
            "dt": np.where(_same, (_a.t - _prev.t).clip(0, 60), 0),
        })
        for _ty in _types:
            F[f"t_{_ty}"] = (_a.type == _ty).astype(int)
            F[f"p_{_ty}"] = ((_prev.type == _ty) & _same).astype(int)
        _scores = np.zeros(len(_a), bool)
        _concedes = np.zeros(len(_a), bool)
        for _k in range(10):
            _gf = pd.Series(_goal_for).shift(-_k)
            _sm = _a.match_id.shift(-_k) == _a.match_id
            _scores |= ((_gf == _a.team_id) & _sm).values
            _concedes |= ((_gf == _a.opp_id) & _sm).values
        _rng = np.random.default_rng(0)
        _fold = _a.match_id.map(dict(zip(_a.match_id.unique(), _rng.integers(0, 2, _a.match_id.nunique())))).values
        _ps, _pc = np.zeros(len(_a)), np.zeros(len(_a))
        for _f in (0, 1):
            _tr, _te = _fold != _f, _fold == _f
            for _lab, _out in ((_scores, _ps), (_concedes, _pc)):
                _m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=0).fit(F[_tr], _lab[_tr])
                _out[_te] = _m.predict_proba(F[_te])[:, 1]
        _pps, _ppc = pd.Series(_ps).shift(1).values, pd.Series(_pc).shift(1).values
        _prev_s = np.where(_st, _pps, _ppc)
        _prev_c = np.where(_st, _ppc, _pps)
        _reset = (~_same.values) | (pd.Series(_goal_for).shift(1).fillna(-1).values != -1)
        _prev_s, _prev_c = np.where(_reset, 0, _prev_s), np.where(_reset, 0, _prev_c)
        vaep_actions = _a.assign(p_score=_ps, p_concede=_pc, off=_ps - _prev_s, deff=-(_pc - _prev_c))
        vaep_actions["value"] = vaep_actions.off + vaep_actions.deff
        vaep_auc = (roc_auc_score(_scores, _ps), roc_auc_score(_concedes, _pc), _scores.mean(), _concedes.mean())
        return vaep_actions, vaep_auc

    vaep_actions, vaep_auc = cached("vaep", (vaep_comp.value,), _compute_vaep)
    return vaep_actions, vaep_auc


@app.cell
def _():
    mo.stop(not FULL)
    vaep_min = mo.ui.slider(90, 630, value=270, step=90, label="Minimum minutes", show_value=True)
    return (vaep_min,)


@app.cell(hide_code=True)
def _(PLAYER, TEAM, appearances, vaep_actions, vaep_auc, vaep_min, xt_cell, xt_hist):
    _mins = appearances[appearances.match_id.isin(vaep_actions.match_id.unique())].groupby("player_id").minutes.sum()
    _xtv = xt_hist[-1]
    _mv = vaep_actions[vaep_actions.type.isin(MOVE_TYPES) & (vaep_actions.result == 1)].dropna(subset=["ex"])
    _xt = (_xtv[xt_cell(_mv.ex, _mv.ey)] - _xtv[xt_cell(_mv.x, _mv.y)])
    _xt = pd.Series(_xt, index=_mv.index).groupby(_mv.player_id).sum()
    _p = vaep_actions.groupby("player_id").agg(vaep=("value", "sum"), team=("team_id", "first")).join(_mins, how="inner").join(_xt.rename("xt"))
    _p = _p[_p.minutes >= vaep_min.value].fillna(0).assign(
        vaep90=lambda d: d.vaep / d.minutes * 90, xt90=lambda d: d.xt / d.minutes * 90,
        name=lambda d: d.index.map(PLAYER), team=lambda d: d.team.map(TEAM))
    _top = _p.nlargest(15, "vaep90").reset_index()
    _td, _tf = photo_data(_top, "player_id", "name")
    _vx = alt.X("vaep90:Q", title="VAEP per 90", scale=alt.Scale(domain=[0, float(_top.vaep90.max()) * 1.15], nice=False))
    _vy = alt.Y("name:N", sort=list(_top.name), title=None)
    _bars = alt.layer(
        alt.Chart(_td).transform_lookup(lookup="_pid", from_=_tf).mark_bar(color=C["blue"], cornerRadiusEnd=3, height=12).encode(
            x=_vx, y=_vy, tooltip=photo_tip("name", "team:N", alt.Tooltip("minutes:Q", format=".0f"), alt.Tooltip("vaep90:Q", format=".3f"))),
        alt.Chart(_td).transform_lookup(lookup="_pid", from_=_tf).mark_image(width=24, height=24, xOffset=17).encode(x=_vx, y=_vy, url="_face:N"),
    ).properties(width=270, height=alt.Step(28), title="Top 15 by VAEP per 90")
    _lab = pd.concat([_p.nlargest(4, "vaep90"), _p.nlargest(3, "xt90")]).drop_duplicates()
    _pd, _pf = photo_data(_p.reset_index(), "player_id", "name")
    _ld, _lf = photo_data(_lab.reset_index(), "player_id", "name")
    _tip = photo_tip("name", "team:N", alt.Tooltip("xt90:Q", format=".3f"), alt.Tooltip("vaep90:Q", format=".3f"))
    _sc = alt.layer(
        alt.Chart(_pd).transform_lookup(lookup="_pid", from_=_pf).mark_circle(size=40, color=C["blue"], opacity=0.5).encode(
            x=alt.X("xt90:Q", title="xT per 90"), y=alt.Y("vaep90:Q", title="VAEP per 90"), tooltip=_tip),
        alt.Chart(_ld).transform_lookup(lookup="_pid", from_=_lf).mark_image(width=20, height=20, xOffset=-14).encode(x="xt90:Q", y="vaep90:Q", url="_face:N", tooltip=_tip),
        alt.Chart(_ld).transform_lookup(lookup="_pid", from_=_lf).mark_text(align="left", dx=6, fontSize=10, color=C["ink2"]).encode(x="xt90:Q", y="vaep90:Q", text="name:N", tooltip=_tip),
    ).properties(width=300, height=300, title=alt.Title("Two ways to value a player", subtitle="Agree on many, disagree on shooters and defenders"))
    mo.vstack([
        mo.hstack([vaep_min, mo.stat(f"{vaep_auc[0]:.2f}", label="AUC · scoring model", caption=f"base rate {vaep_auc[2]:.1%}"),
                   mo.stat(f"{vaep_auc[1]:.2f}", label="AUC · conceding model", caption=f"base rate {vaep_auc[3]:.1%}")], widths=[2, 1, 1]),
        mo.hstack([_bars, _sc], align="start"),
    ])
    return


@app.cell(hide_code=True)
def _(MATCH_OPTIONS, vaep_actions):
    _ids = set(vaep_actions.match_id.unique())
    _opts = {k: v for k, v in MATCH_OPTIONS.items() if v in _ids}
    vaep_match = mo.ui.dropdown(_opts, value=next((k for k in _opts if " · Final · " in k), list(_opts)[0]), label="Match", searchable=True, full_width=True)
    return (vaep_match,)


@app.cell(hide_code=True)
def _(PLAYER, TEAM, matches, vaep_actions, vaep_match):
    _m = matches.set_index("match_id").loc[vaep_match.value]
    _a = vaep_actions[vaep_actions.match_id == vaep_match.value].assign(
        team=lambda d: d.team_id.map(TEAM), player=lambda d: d.player_id.map(PLAYER), tmin=lambda d: d.minute + d.second / 60)
    _sc = two_teams(_m.home, _m.away)
    _top = _a.nlargest(12, "value")
    _fd, _ff = photo_data(_a[_a.value.abs() > 0.005], "player_id", "player")
    _td, _tf = photo_data(_top, "player_id", "player")
    _flow = alt.Chart(_fd).transform_lookup(lookup="_pid", from_=_ff).mark_bar(width=2).encode(
        x=alt.X("tmin:Q", title="minute"), y=alt.Y("value:Q", title="action value"),
        color=alt.Color("team:N", scale=_sc, legend=alt.Legend(title=None, orient="top")),
        tooltip=photo_tip("player", "type:N", "minute:Q", alt.Tooltip("value:Q", format="+.3f")),
    ).properties(width=600, height=170, title=alt.Title("Every action's value, minute by minute", subtitle="Spikes are chances created, scored, or prevented"))
    _arrows = pitch(
        alt.Chart(_top).mark_rule(strokeWidth=2.5).encode(**xy(), **xy2(), color=alt.Color("team:N", scale=_sc, legend=None)),
        alt.Chart(_td).transform_lookup(lookup="_pid", from_=_tf).mark_circle(size=120, opacity=1, stroke=C["surface"], strokeWidth=2).encode(
            **xy(), color=alt.Color("team:N", scale=_sc, legend=None),
            tooltip=photo_tip("player", "type:N", "minute:Q", alt.Tooltip("value:Q", format="+.3f"))),
        width=600, title=alt.Title("The 12 most valuable actions", subtitle="Each from its team's perspective (attacking →). Hover for who and when."),
    )
    mo.vstack([vaep_match, _flow, _arrows])
    return


@app.cell(hide_code=True)
def _():
    further(
        ("socceraction (SPADL, VAEP, xT)", "https://github.com/ML-KULeuven/socceraction"),
        ("EPV: Fernández, Bornn & Cervone (2021)", "https://doi.org/10.1007/s10994-021-05989-6"),
        ("un-xPass: valuing creativity (KDD 2023)", "https://github.com/ML-KULeuven/un-xPass"),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 13 · Pitch control

    *Who owns which part of the pitch?* Three answers, from naive to physical:
    **Voronoi** (whoever is closest; [Taki & Hasegawa, 2000](https://doi.org/10.1109/CGI.2000.852338)),
    **time-to-intercept** (who gets there first, given speed, reaction time and the ball's flight; [Spearman et al., 2017](https://web.archive.org/web/20170625101337/http://www.sloansportsconference.com/wp-content/uploads/2017/02/1621.pdf))
    and **influence fields** (each player as a Gaussian stretched by their velocity; [Fernández & Bornn, 2018](https://www.sloansportsconference.com/research-papers/wide-open-spaces-a-statistical-technique-for-measuring-space-creation-in-professional-soccer)).
    Scrub through the build-up to a goal.
    """)
    return


@app.cell(hide_code=True)
def _():
    snapshot("s13")
    return


@app.cell
def _(TRK_GOALS):
    pc_moment = mo.ui.dropdown(TRK_GOALS, value=list(TRK_GOALS)[0], label="Goal")
    pc_offset = mo.ui.slider(-12, 1, value=-3, step=0.2, label="Seconds before the shot", show_value=True, full_width=True)
    pc_model = mo.ui.radio(["Voronoi", "Time-to-intercept (Spearman)", "Influence (Fernández & Bornn)"], value="Time-to-intercept (Spearman)", inline=True)
    pc_vmax = mo.ui.slider(3, 9, value=5, step=0.5, label="Max running speed (m/s)", show_value=True)
    pc_react = mo.ui.slider(0.2, 1.2, value=0.7, step=0.1, label="Reaction time (s)", show_value=True)
    return pc_model, pc_moment, pc_offset, pc_react, pc_vmax


@app.cell
def _():
    def pc_voronoi(grid, home, away, **_):
        dh = np.linalg.norm(grid[:, None] - home[None], axis=2).min(1)
        da = np.linalg.norm(grid[:, None] - away[None], axis=2).min(1)
        return (dh < da).astype(float)

    def pc_spearman(grid, home, away, home_v, away_v, ball, vmax=5.0, react=0.7, sigma=0.45, lam=4.3, ball_speed=15.0, dt=0.04, horizon=10.0):
        def tti(pos, vel):
            r = pos + vel * react
            return react + np.linalg.norm(grid[:, None] - r[None], axis=2) / vmax
        th, ta = tti(home, home_v), tti(away, away_v)
        tb = np.linalg.norm(grid - ball, axis=1) / ball_speed
        h, a = np.zeros(len(grid)), np.zeros(len(grid))
        k = np.pi / np.sqrt(3) / sigma
        for step in np.arange(dt, horizon, dt):
            t = (tb + step)[:, None]
            fh = (1 / (1 + np.exp(-k * (t - th)))).sum(1)  # expected number of home players who could reach it by t
            fa = (1 / (1 + np.exp(-k * (t - ta)))).sum(1)
            rem = 1 - h - a
            # exact update for a constant hazard over dt: never hands out more than what remains
            take = rem * (1 - np.exp(-lam * dt * (fh + fa)))
            share = fh / np.maximum(fh + fa, 1e-12)
            h, a = h + take * share, a + take * (1 - share)
            if rem.max() < 0.01:
                break
        return np.clip(h / np.maximum(h + a, 1e-9), 0, 1)

    def pc_influence(grid, home, away, home_v, away_v, ball, **_):
        def field(pos, vel):
            out = np.zeros(len(grid))
            for p, v in zip(pos, vel):
                speed = np.linalg.norm(v)
                theta = np.arctan2(v[1], v[0])
                rad = min(4 + np.linalg.norm(p - ball) ** 3 / (18**3 / 10), 10)
                srat = min(speed**2 / 13**2, 0.99)
                rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                s = np.diag([(rad + rad * srat) / 2, (rad - rad * srat) / 2])
                cov = rot @ s @ s @ rot.T
                mu = p + v * 0.5
                inv = np.linalg.inv(cov)
                d = grid - mu
                d0 = p - mu
                out += np.exp(-0.5 * np.einsum("ij,jk,ik->i", d, inv, d)) / np.exp(-0.5 * d0 @ inv @ d0)
            return out
        return 1 / (1 + np.exp(-(field(home, home_v) - field(away, away_v))))

    return pc_influence, pc_spearman, pc_voronoi


@app.cell(hide_code=True)
def _(TRK_AWAY, TRK_HOME, pc_influence, pc_model, pc_moment, pc_offset, pc_react, pc_spearman, pc_vmax, pc_voronoi, trk):
    _f0 = pc_moment.value + int(round(pc_offset.value * 25))
    _k = int((trk.frame - _f0).abs().argmin())
    _r, _rb, _ra = trk.iloc[_k], trk.iloc[max(_k - 2, 0)], trk.iloc[min(_k + 2, len(trk) - 1)]
    _dtv = max(float(_ra.time - _rb.time), 0.2)

    def _team(ids):
        ids = [p for p in ids if pd.notna(_r[f"{p}_x"])]
        pos = np.array([[_r[f"{p}_x"], _r[f"{p}_y"]] for p in ids])
        vel = np.array([[(_ra[f"{p}_x"] - _rb[f"{p}_x"]) / _dtv, (_ra[f"{p}_y"] - _rb[f"{p}_y"]) / _dtv] for p in ids])
        return ids, pos, np.nan_to_num(vel)

    _hid, _hp, _hv = _team(TRK_HOME)
    _aid, _ap, _av = _team(TRK_AWAY)
    _ball = np.array([_r.ball_x, _r.ball_y]) if pd.notna(_r.ball_x) else np.array([L / 2, W / 2])
    _g = grid_cells(50, 32)
    _pts = _g[["cx", "cy"]].values
    _fn = {"Voronoi": pc_voronoi, "Time-to-intercept (Spearman)": pc_spearman, "Influence (Fernández & Bornn)": pc_influence}[pc_model.value]
    _g["home"] = _fn(_pts, _hp, _ap, home_v=_hv, away_v=_av, ball=_ball, vmax=pc_vmax.value, react=pc_react.value)
    _heat = alt.Chart(_g).mark_rect().encode(
        **xy("x0", "y0"), **xy2("x1", "y1"),
        color=alt.Color("home:Q", scale=alt.Scale(domain=[0, 0.5, 1], range=[C["orange"], C["mid"], C["blue"]], interpolate="rgb"),
                        legend=alt.Legend(title="P(home controls)", orient="bottom", format=".0%")),
        tooltip=[alt.Tooltip("home:Q", format=".0%", title="home control")],
    )
    _pl = pd.DataFrame(np.vstack([_hp, _ap]), columns=["x", "y"]).assign(
        vx=np.concatenate([_hv[:, 0], _av[:, 0]]), vy=np.concatenate([_hv[:, 1], _av[:, 1]]),
        team=["Home"] * len(_hp) + ["Away"] * len(_ap), num=[p[1:] for p in _hid + _aid])
    _pl = _pl.assign(ex=_pl.x + _pl.vx, ey=_pl.y + _pl.vy)
    _ts = alt.Scale(domain=["Home", "Away"], range=[C["blue"], C["orange"]])
    _chart = pitch(
        alt.Chart(_pl).mark_rule(color=C["ink"], strokeWidth=1.5).encode(**xy(), **xy2()),
        alt.Chart(_pl).mark_circle(size=150, opacity=1, stroke=C["surface"], strokeWidth=2).encode(**xy(), color=alt.Color("team:N", scale=_ts, legend=None), tooltip=["team:N", "num:N"]),
        alt.Chart(_pl).mark_text(fontSize=8, color="#ffffff", fontWeight=600).encode(**xy(), text="num:N"),
        alt.Chart(pd.DataFrame(dict(x=[_ball[0]], y=[_ball[1]]))).mark_circle(color=C["ink"], size=60, opacity=1, stroke=C["surface"], strokeWidth=1.5).encode(**xy()),
        under=[_heat], width=640,
        title=alt.Title(f"{pc_model.value}", subtitle="Lines show where each player will be in one second"),
    )
    _share = _g.home.mean()
    _final = _g[_g.cx > 70].home.mean()
    _params = mo.hstack([pc_vmax, pc_react], justify="start", gap=1.5) if pc_model.value.startswith("Time") else mo.md("")
    mo.vstack([
        mo.hstack([pc_moment, pc_model], justify="start", gap=1.5), pc_offset, _params,
        mo.hstack([_chart, mo.vstack([mo.stat(f"{_share:.0%}", label="pitch owned by home"), mo.stat(f"{_final:.0%}", label="of the final third (x > 70 m)")])], widths=[4, 1], align="center"),
    ])
    return


@app.cell(hide_code=True)
def _():
    further(
        ("LaurieOnTracking (pitch control in Python)", "https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking"),
        ("databallpy (tracking + events toolkit)", "https://github.com/Alek050/databallpy"),
        ("floodlight", "https://github.com/floodlight-sports/floodlight"),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 14 · Ghost team

    Teams move as a unit around the ball. Learn that mapping from tracking data, **ball position → all 22 positions**,
    and you get a "ghost" of how this team *usually* sets up ([Le et al., 2017](https://www.sloansportsconference.com/research-papers/data-driven-ghosting-using-deep-imitation-learning)).
    Clubs compare the ghost with what actually happened. Move the ball and switch possession.
    """)
    return


@app.cell(hide_code=True)
def _():
    snapshot("s14")
    return


@app.cell
def _(trk, trk_events):
    _start = trk.iloc[0]
    GH_HOME = [c[:-2] for c in trk.columns if c.startswith("H") and c.endswith("_x") and pd.notna(_start[c])]
    GH_AWAY = [c[:-2] for c in trk.columns if c.startswith("A") and c.endswith("_x") and pd.notna(_start[c])]
    _cols = [f"{p}_{a}" for p in GH_HOME + GH_AWAY for a in "xy"]
    _ev = trk_events[trk_events.team.isin(["Home", "Away"])].sort_values("start_frame")[["start_frame", "team"]]
    _d = pd.merge_asof(trk.sort_values("frame"), _ev, left_on="frame", right_on="start_frame", direction="backward")
    _d = _d.dropna(subset=_cols + ["ball_x", "team"])
    gh_X = np.column_stack([_d.ball_x / L, _d.ball_y / W, (_d.team == "Home").astype(float)])
    gh_Y = _d[_cols].values
    def _fit_ghosts():
        gh_models = {
            "Nearest real frames (k-NN)": KNeighborsRegressor(n_neighbors=25, weights="distance").fit(gh_X * [1, 1, 3], gh_Y),
            "Neural network (MLP)": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=400, random_state=0)).fit(gh_X, gh_Y),
        }
        return gh_models

    gh_models = cached("ghost", (), _fit_ghosts)
    gh_n = len(_d)
    return GH_AWAY, GH_HOME, gh_models, gh_n


@app.cell
def _():
    mo.stop(not FULL)
    gh_bx = mo.ui.slider(0, 105, value=70, step=1, label="Ball x (m)", show_value=True, full_width=True)
    gh_by = mo.ui.slider(0, 68, value=50, step=1, label="Ball y (m)", show_value=True, full_width=True)
    gh_poss = mo.ui.radio(["Home", "Away"], value="Home", inline=True, label="In possession")
    gh_kind = mo.ui.radio(["Nearest real frames (k-NN)", "Neural network (MLP)"], value="Neural network (MLP)", inline=True, label="Model")
    return gh_bx, gh_by, gh_kind, gh_poss


@app.cell(hide_code=True)
def _(GH_AWAY, GH_HOME, gh_bx, gh_by, gh_kind, gh_models, gh_n, gh_poss):
    _x = np.array([[gh_bx.value / L, gh_by.value / W, 1.0 if gh_poss.value == "Home" else 0.0]])
    _scale = [1, 1, 3] if gh_kind.value.startswith("Nearest") else [1, 1, 1]
    _pred = gh_models[gh_kind.value].predict(_x * _scale)[0].reshape(-1, 2)
    _n = len(GH_HOME)
    _pl = pd.DataFrame(_pred, columns=["x", "y"]).assign(team=["Home"] * _n + ["Away"] * len(GH_AWAY), num=[p[1:] for p in GH_HOME + GH_AWAY])
    _ts = alt.Scale(domain=["Home", "Away"], range=[C["blue"], C["orange"]])
    _hulls, _stats = [], {}
    for _t, _pts in (("Home", _pred[1:_n]), ("Away", _pred[_n + 1 :])):  # outfield players (first listed = goalkeeper)
        _h = ConvexHull(_pts)
        _v = _pts[_h.vertices]
        _hulls.append(pd.DataFrame(np.vstack([_v, _v[:1]]), columns=["x", "y"]).assign(team=_t, o=range(len(_v) + 1)))
        _stats[_t] = (np.ptp(_pts[:, 0]), np.ptp(_pts[:, 1]), _h.volume)
    _hull = pd.concat(_hulls)
    _chart = pitch(
        alt.Chart(_hull).mark_line(strokeWidth=1.5, opacity=0.7).encode(**xy(), color=alt.Color("team:N", scale=_ts, legend=None), detail="team:N", order="o:Q"),
        alt.Chart(_pl).mark_circle(size=170, opacity=1, stroke=C["surface"], strokeWidth=2).encode(**xy(), color=alt.Color("team:N", scale=_ts, legend=alt.Legend(title=None, orient="bottom"))),
        alt.Chart(_pl).mark_text(fontSize=8, color="#ffffff", fontWeight=600).encode(**xy(), text="num:N"),
        alt.Chart(pd.DataFrame(dict(x=[gh_bx.value], y=[gh_by.value]))).mark_circle(color=C["ink"], size=90, opacity=1, stroke=C["surface"], strokeWidth=2).encode(**xy()),
        width=620, title=alt.Title("Where would everyone be?", subtitle=f"Learned from {gh_n:,} frames (5 Hz) while all 22 starters were on the pitch"),
    )
    _tbl = mo.md(
        "| | depth | width | area |\n|---|---|---|---|\n"
        + "\n".join(f"| **{t}** | {d:.0f} m | {w:.0f} m | {a:,.0f} m² |" for t, (d, w, a) in _stats.items())
    )
    mo.vstack([
        mo.hstack([gh_poss, gh_kind], justify="start", gap=2),
        gh_bx, gh_by,
        mo.hstack([_chart, _tbl], widths=[4, 1.4], align="center"),
        note("Lines: each team's outfield shape (convex hull). Teams are more compact without the ball. The k-NN ghost averages the closest real moments; the MLP produces a smooth function you can query anywhere."),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 15 · Passing networks

    Players are nodes and passes are edges. Network science then gives you **hubs**, **bridges** and **structure**
    ([Duch, Waitzman & Amaral, 2010](https://doi.org/10.1371/journal.pone.0010937)). Nodes sit at each player's average
    on-ball location, and only the starting XI until the first substitution is shown.
    """)
    return


@app.cell(hide_code=True)
def _():
    snapshot("s15")
    return


@app.cell
def _(FINAL_LABEL, MATCH_OPTIONS):
    net_match = mo.ui.dropdown(MATCH_OPTIONS, value=FINAL_LABEL, searchable=True, label="Match", full_width=True)
    net_min = mo.ui.slider(1, 10, value=3, label="Minimum passes per link", show_value=True)
    return net_match, net_min


@app.cell(hide_code=True)
def _(PLAYER, TEAM, appearances, matches, net_match, net_min):
    _m = matches.set_index("match_id").loc[net_match.value]
    _acts = actions_for(_m.comp)
    _acts = _acts[_acts.match_id == net_match.value]
    _app = appearances[appearances.match_id == net_match.value]
    _charts, _tables = [], {}
    for _tid, _col in ((_m.home_id, C["blue"]), (_m.away_id, C["orange"])):
        _xi = set(_app[(_app.team_id == _tid) & (_app.started == 1)].player_id)
        _jersey = dict(zip(_app.player_id, _app.jersey))
        _ta = _acts[_acts.team_id == _tid]
        _subs = _ta[~_ta.player_id.isin(_xi) & _ta.player_id.notna()]
        _cut = (_subs.period * 1000 + _subs.minute).min() if len(_subs) else 10**9
        _ta = _ta[(_ta.period * 1000 + _ta.minute) < _cut]
        _pas = _ta[_ta.type.isin(PASS_TYPES) & (_ta.result == 1) & _ta.player_id.isin(_xi) & _ta.recipient_id.isin(_xi)]
        _loc = pd.concat([_pas[["player_id", "x", "y"]], _pas[["recipient_id", "ex", "ey"]].set_axis(["player_id", "x", "y"], axis=1)])
        _nodes = _loc.groupby("player_id").agg(x=("x", "mean"), y=("y", "mean"), touches=("x", "size")).reset_index()
        _nodes["name"] = _nodes.player_id.map(PLAYER)
        _nodes["num"] = _nodes.player_id.map(_jersey).astype("Int64").astype(str)
        _pairs = _pas.assign(a=np.minimum(_pas.player_id, _pas.recipient_id), b=np.maximum(_pas.player_id, _pas.recipient_id)).groupby(["a", "b"]).size().rename("n").reset_index()
        _pairs = _pairs[_pairs.n >= net_min.value]
        _pos = _nodes.set_index("player_id")
        _edges = _pairs.assign(x=_pairs.a.map(_pos.x), y=_pairs.a.map(_pos.y), ex=_pairs.b.map(_pos.x), ey=_pairs.b.map(_pos.y))
        _G = nx.DiGraph()
        for (_p, _r), _n in _pas.groupby(["player_id", "recipient_id"]).size().items():
            _G.add_edge(int(_p), int(_r), weight=int(_n), distance=1 / _n)
        _bet = nx.betweenness_centrality(_G, weight="distance")
        _pr = nx.pagerank(_G, weight="weight")
        _nodes = _nodes.assign(betweenness=_nodes.player_id.map(_bet).fillna(0), pagerank=_nodes.player_id.map(_pr).fillna(0))
        _t = _nodes.sort_values("pagerank", ascending=False).reset_index(drop=True)
        _t["player"] = _t.player_id.astype(int).astype(str) + "|" + _t.name  # shown as photo + name by the table's format_mapping
        _tables[TEAM[_tid]] = _t[["num", "player", "touches", "pagerank", "betweenness"]]
        _nd, _nf = photo_data(_nodes, "player_id", "name")
        _ch = pitch(
            alt.Chart(_edges).mark_rule(color=_col, opacity=0.55, strokeCap="round").encode(
                **xy(), **xy2(), strokeWidth=alt.StrokeWidth("n:Q", scale=alt.Scale(range=[1, 9]), legend=None), tooltip=[alt.Tooltip("n:Q", title="passes")]),
            alt.Chart(_nd).transform_lookup(lookup="_pid", from_=_nf).mark_circle(color=_col, opacity=1, stroke=C["surface"], strokeWidth=2).encode(
                **xy(), size=alt.Size("touches:Q", scale=alt.Scale(range=[120, 700]), legend=None),
                tooltip=photo_tip("name", "touches:Q", alt.Tooltip("pagerank:Q", format=".3f"), alt.Tooltip("betweenness:Q", format=".3f"))),
            alt.Chart(_nd).transform_lookup(lookup="_pid", from_=_nf).mark_text(color="#ffffff", fontSize=10, fontWeight=600).encode(
                **xy(), text="num:N", tooltip=photo_tip("name", "touches:Q", alt.Tooltip("pagerank:Q", format=".3f"), alt.Tooltip("betweenness:Q", format=".3f"))),
            width=420, title=alt.Title(TEAM[_tid], subtitle=f"{len(_pas)} completed passes between starters · attacking →"),
        )
        _charts.append(_ch)
    mo.vstack([
        net_match, net_min, mo.hstack(_charts, widths="equal"),
        mo.ui.tabs({t: mo.ui.table(d, selection=None, page_size=11, show_column_summaries=False, show_download=False, format_mapping={"pagerank": "{:.3f}", "betweenness": "{:.3f}", "player": lambda v: mo.Html(chip(int(v.split("|", 1)[0]), v.split("|", 1)[1]))}) for t, d in _tables.items()}),
        note("PageRank: how often passing 'flows' through a player. Betweenness: how often they sit on the shortest route between two teammates."),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 16 · Player fingerprints

    Describe every player by **what they do per 90 minutes** (passes, progressive carries, pressures, shots and so on), then
    compress that profile to two dimensions with PCA. Similar players end up near each other, which is the core
    idea behind data-driven scouting and recruitment.
    """)
    return


@app.cell(hide_code=True)
def _():
    snapshot("s16")
    return


@app.cell
def _():
    mo.stop(not FULL)
    FP_CORPORA = {
        "Men's internationals 2022–24": ["wc22", "eu24", "ca24", "afc23"],
        "Women's internationals 2022–25": ["wwc23", "weu22", "weu25"],
        "Men's internationals 2018–21": ["wc18", "eu20"],
    }
    fp_corpus = mo.ui.radio(list(FP_CORPORA), value="Men's internationals 2022–24", label="Players from", inline=True)
    fp_min = mo.ui.slider(180, 900, value=300, step=60, label="Minimum minutes", show_value=True)
    return FP_CORPORA, fp_corpus, fp_min


@app.cell
def _(FP_CORPORA, PLAYER, TEAM, appearances, fp_corpus, fp_min):
    _a = pd.concat([actions_for(c) for c in FP_CORPORA[fp_corpus.value]], ignore_index=True)
    _app = appearances[appearances.match_id.isin(_a.match_id.unique())]
    _pos = _app.groupby("player_id").position.agg(lambda s: s.mode().iloc[0])
    _mins = _app.groupby("player_id").minutes.sum()
    _team = _app.groupby("player_id").team_id.agg(lambda s: s.mode().iloc[0])

    def _grp(p):
        if "Goalkeeper" in p:
            return "GK"
        if "Back" in p:
            return "Defender"
        if "Forward" in p or "Striker" in p or "Wing" in p:
            return "Forward"
        return "Midfielder"

    _d0 = np.hypot(L - _a.x, W / 2 - _a.y)
    _d1 = np.hypot(L - _a.ex, W / 2 - _a.ey)
    _a = _a.assign(prog=(_d1 < 0.75 * _d0) & (_a.result == 1), box=(_a.x > L - 16.5) & ((_a.y - W / 2).abs() < 20.16))
    _is = lambda *t: _a.type.isin(t)
    _f = pd.DataFrame({
        "passes": _is(*PASS_TYPES),
        "pass_ok": _is(*PASS_TYPES) & (_a.result == 1),
        "progressive passes": _is(*PASS_TYPES) & _a.prog,
        "crosses": _is("cross"),
        "key passes": _is(*PASS_TYPES) & (_a.assist > 0),
        "carries": _is("carry"),
        "progressive carries": _is("carry") & _a.prog,
        "take-ons": _is("take_on"),
        "touches in box": _a.box & ~_is("pressure"),
        "shots": _is("shot", "shot_freekick"),
        "xG": np.where(_is("shot", "shot_freekick"), _a.xg.fillna(0), 0),
        "pressures": _is("pressure"),
        "tackles + interceptions": _is("tackle", "interception"),
        "recoveries": _is("recovery"),
        "clearances + blocks": _is("clearance", "block"),
    }).astype(float).assign(player_id=_a.player_id)
    _agg = _f.groupby("player_id").sum()
    _agg = _agg[_agg.index.isin(_mins[_mins >= fp_min.value].index)]
    _p90 = _agg.div(_mins.reindex(_agg.index), axis=0) * 90
    _p90["pass %"] = _agg.pass_ok / _agg.passes.clip(lower=1)
    _p90 = _p90.drop(columns="pass_ok")
    _meta = pd.DataFrame(dict(name=_p90.index.map(PLAYER), team=_team.reindex(_p90.index).map(TEAM),
                              group=_pos.reindex(_p90.index).map(_grp), minutes=_mins.reindex(_p90.index)), index=_p90.index)
    _keep = _meta.group != "GK"
    fp_stats, fp_meta = _p90[_keep], _meta[_keep]
    fp_z = (fp_stats - fp_stats.mean()) / fp_stats.std()
    _pca = PCA(n_components=2, random_state=0).fit(fp_z.values)
    fp_xy = pd.DataFrame(_pca.transform(fp_z.values), columns=["pc1", "pc2"], index=fp_z.index).join(fp_meta)
    _load = pd.DataFrame(_pca.components_.T, index=fp_z.columns, columns=["pc1", "pc2"])
    fp_axes = {k: (" · ".join(_load[k].nsmallest(2).index), " · ".join(_load[k].nlargest(2).index), _pca.explained_variance_ratio_[i]) for i, k in enumerate(["pc1", "pc2"])}
    _default = next((n for n in ("Lionel Messi", "Aitana Bonmatí", "Luka Modrić") if n in set(fp_meta.name)), fp_meta.sort_values("minutes").name.iloc[-1])
    fp_player = mo.ui.dropdown(dict(sorted(zip(fp_meta.name + " · " + fp_meta.team, fp_meta.index))), value=f"{_default} · {fp_meta.set_index('name').loc[_default, 'team'] if _default in set(fp_meta.name) else ''}".strip(" ·"), searchable=True, label="Player")
    return fp_axes, fp_meta, fp_player, fp_stats, fp_xy, fp_z


@app.cell(hide_code=True)
def _(fp_axes, fp_corpus, fp_min, fp_player, fp_stats, fp_xy, fp_z):
    _pid = fp_player.value
    _v = fp_z.loc[_pid].values
    _sim = (fp_z.values @ _v) / (np.linalg.norm(fp_z.values, axis=1) * np.linalg.norm(_v) + 1e-9)
    _sim = pd.Series(_sim, index=fp_z.index).drop(_pid).nlargest(8)
    _xy = fp_xy.assign(role=np.where(fp_xy.index == _pid, "selected", np.where(fp_xy.index.isin(_sim.index), "similar", "other")))
    _gs = alt.Scale(domain=["Defender", "Midfielder", "Forward"], range=[C["blue"], C["orange"], C["aqua"]])
    _xd, _xf = photo_data(_xy.rename_axis("player_id").reset_index(), "player_id", "name")
    _tip = photo_tip("name", "team:N", "group:N", alt.Tooltip("minutes:Q", format=".0f"))
    _pts = alt.Chart(_xd).transform_lookup(lookup="_pid", from_=_xf).mark_circle(stroke=C["surface"], strokeWidth=1).encode(
        x=alt.X("pc1:Q", title=f"← {fp_axes['pc1'][0]}   ·   {fp_axes['pc1'][1]} →"),
        y=alt.Y("pc2:Q", title=f"← {fp_axes['pc2'][0]}   ·   {fp_axes['pc2'][1]} →"),
        color=alt.Color("group:N", scale=_gs, legend=alt.Legend(title=None, orient="top")),
        size=alt.condition("datum.role != 'other'", alt.value(130), alt.value(36)),
        opacity=alt.condition("datum.role != 'other'", alt.value(1), alt.value(0.45)),
        tooltip=_tip,
    )
    _shown = _xd[_xd.role != "other"]
    _faces = alt.Chart(_shown).transform_lookup(lookup="_pid", from_=_xf).mark_image(width=22, height=22, xOffset=-16).encode(x="pc1:Q", y="pc2:Q", url="_face:N", tooltip=_tip)
    _lab = alt.Chart(_shown).transform_lookup(lookup="_pid", from_=_xf).mark_text(align="left", dx=7, fontSize=10, color=C["ink2"]).encode(x="pc1:Q", y="pc2:Q", text="name:N", tooltip=_tip)
    _map = alt.layer(_pts, _faces, _lab).properties(width=430, height=380, title=alt.Title("A map of playing styles", subtitle=f"PCA of {fp_z.shape[1]} per-90 stats · {len(fp_z)} outfield players · selected + 8 most similar labelled"))
    _twin = _sim.index[0]
    _pct = fp_stats.rank(pct=True)
    _bars = pd.DataFrame(dict(stat=fp_stats.columns, sel=_pct.loc[_pid].values, twin=_pct.loc[_twin].values))
    _who = fp_xy.loc[_pid, "name"]
    _twin_name = fp_xy.loc[_twin, "name"]
    _b = alt.Chart(_bars).mark_bar(color=C["blue"], cornerRadiusEnd=3, height=10).encode(
        x=alt.X("sel:Q", title="percentile among peers", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])), y=alt.Y("stat:N", title=None, sort=None),
        tooltip=["stat:N", alt.Tooltip("sel:Q", format=".0%", title=_who)])
    _t = alt.Chart(_bars).mark_tick(color=C["orange"], thickness=3, size=14).encode(x="twin:Q", y=alt.Y("stat:N", sort=None), tooltip=["stat:N", alt.Tooltip("twin:Q", format=".0%", title=_twin_name)])
    _prof = alt.layer(_b, _t).properties(width=190, height=alt.Step(17), title=alt.Title(_who, subtitle=f"bars: {_who} · ticks: {_twin_name} (most similar)"))
    _card = mo.hstack([avatar(_pid, _who, 72), mo.md(f"**{_who}**  \n{fp_xy.loc[_pid, 'team']} · {fp_xy.loc[_pid, 'group'].lower()} · {fp_xy.loc[_pid, 'minutes']:.0f} min")],
                      justify="start", align="center", gap=1)
    _items = [mo.vstack([avatar(i, fp_xy.loc[i, "name"], 52),
                         mo.md(f"<span style='font-size:0.75em;line-height:1.2;display:block;text-align:center;max-width:78px'>{fp_xy.loc[i, 'name']}<br><span style='color:{C['muted']}'>{s:.2f}</span></span>")],
                        align="center", gap=0.2) for i, s in _sim.items()]
    _list = mo.vstack([mo.md("**Most similar** (cosine similarity)"), mo.hstack(_items, wrap=True, justify="start", align="start", gap=0.7)], gap=0.4)
    mo.vstack([mo.hstack([fp_corpus, fp_min], justify="start", gap=2), fp_player, mo.hstack([_map, mo.vstack([_card, _prof, _list])], align="start"),
               note("Photos: Wikimedia Commons, credited under References. A player without a free photo shows initials.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 17 · Forecasting matches

    Three classic models, all rebuilt on 150 years of international results:
    **Elo ratings** (a running score that moves after every match, as on [eloratings.net](https://www.eloratings.net/)),
    a **Poisson goals model** with an attack and a defence strength per team ([Maher, 1982](https://doi.org/10.1111/j.1467-9574.1982.tb00782.x);
    [Dixon & Coles, 1997](https://doi.org/10.1111/1467-9876.00065)), and a **Monte Carlo** tournament simulator built on top.
    """)
    return


@app.cell
def _():
    elo_k = mo.ui.slider(0.5, 2.0, value=1.0, step=0.1, label="K-factor scale", show_value=True)
    elo_hfa = mo.ui.slider(0, 200, value=100, step=10, label="Home advantage (Elo points)", show_value=True)
    return elo_hfa, elo_k


@app.cell
def _(elo_hfa, elo_k, intl):
    def _k(t):
        if t == "FIFA World Cup":
            return 60
        if t in ("UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup", "Gold Cup", "CONCACAF Championship"):
            return 50
        if "qualification" in t or t in ("UEFA Nations League", "CONCACAF Nations League", "Confederations Cup"):
            return 40
        return 20 if t == "Friendly" else 30

    _r, _hist, _exp = {}, [], []
    _K = intl.tournament.map(_k).values * elo_k.value
    for _i, _m in enumerate(intl.itertuples()):
        rh, ra = _r.get(_m.home_team, 1500.0), _r.get(_m.away_team, 1500.0)
        dr = rh - ra + (0 if _m.neutral else elo_hfa.value)
        e = 1 / (10 ** (-dr / 400) + 1)
        w = 1.0 if _m.home_score > _m.away_score else (0.5 if _m.home_score == _m.away_score else 0.0)
        gd = abs(_m.home_score - _m.away_score)
        g = 1 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
        delta = _K[_i] * g * (w - e)
        _r[_m.home_team], _r[_m.away_team] = rh + delta, ra - delta
        _hist.append((_m.date, _m.home_team, rh + delta))
        _hist.append((_m.date, _m.away_team, ra - delta))
        _exp.append((e, w, _m.year))
    elo_hist = pd.DataFrame(_hist, columns=["date", "team", "elo"])
    elo_eval = pd.DataFrame(_exp, columns=["e", "w", "year"])
    _last = intl.groupby("home_team").date.max().combine(intl.groupby("away_team").date.max(), max, fill_value=pd.Timestamp("1900-01-01"))
    _active = _last[_last >= intl.date.max() - pd.Timedelta(days=4 * 365)].index
    elo_now = pd.Series(_r).loc[lambda s: s.index.isin(_active)].sort_values(ascending=False)
    return elo_eval, elo_hist, elo_now


@app.cell
def _(elo_now):
    elo_teams = mo.ui.multiselect(sorted(elo_now.index), value=[t for t in ["Brazil", "Argentina", "Germany", "Spain", "Portugal"] if t in elo_now.index], max_selections=8, label="Teams")
    return (elo_teams,)


@app.cell(hide_code=True)
def _(elo_eval, elo_hfa, elo_hist, elo_k, elo_now, elo_teams, intl):
    _h = elo_hist[elo_hist.team.isin(elo_teams.value)].assign(year=lambda d: d.date.dt.year)
    _h = _h.groupby(["team", "year"]).elo.last().reset_index()
    _order = list(elo_teams.value)
    _sc = alt.Scale(domain=_order, range=CAT[: len(_order)])
    _end = _h.sort_values("year").groupby("team").tail(1)
    _lines = alt.layer(
        alt.Chart(_h).mark_line(strokeWidth=1.8).encode(
            x=alt.X("year:Q", title=None, axis=alt.Axis(format="d")), y=alt.Y("elo:Q", title="Elo rating", scale=alt.Scale(zero=False)),
            color=alt.Color("team:N", scale=_sc, legend=alt.Legend(title=None, orient="top")), tooltip=["team:N", "year:Q", alt.Tooltip("elo:Q", format=".0f")]),
        alt.Chart(_end).mark_text(align="left", dx=5, fontSize=10, color=C["ink2"]).encode(x="year:Q", y="elo:Q", text="team:N"),
    ).properties(width=470, height=260, title=alt.Title("Elo ratings over time", subtitle="Every team starts at 1500; ratings move after each match by K × (result − expectation)"))
    _top = elo_now.head(15).rename_axis("team").reset_index(name="elo")
    _lo = float(_top.elo.min() - 40)
    _top_ch = alt.layer(
        alt.Chart(_top.assign(lo=_lo)).mark_rule(color=C["axis"], strokeWidth=1.5).encode(
            x=alt.X("lo:Q", title="Elo today", scale=alt.Scale(zero=False, domain=[_lo, float(_top.elo.max() + 15)])), x2="elo:Q", y=alt.Y("team:N", sort="-x", title=None)),
        alt.Chart(_top).mark_circle(color=C["blue"], size=70, opacity=1).encode(
            x="elo:Q", y=alt.Y("team:N", sort="-x", title=None), tooltip=["team:N", alt.Tooltip("elo:Q", format=".0f")]),
    ).properties(width=170, height=alt.Step(16), title=f"Top 15 · {intl.date.max():%b %Y}")
    _ev = elo_eval[(elo_eval.year >= 2000) & (elo_eval.w != 0.5)]
    _acc = ((_ev.e > 0.5) == (_ev.w == 1)).mean()
    _cal = elo_eval[elo_eval.year >= 2000].assign(bin=lambda d: (d.e * 10).clip(0, 9.99).astype(int)).groupby("bin").agg(e=("e", "mean"), w=("w", "mean"), n=("w", "size")).reset_index()
    _cal_ch = alt.layer(
        alt.Chart(pd.DataFrame(dict(a=[0, 1]))).mark_line(color=C["axis"], strokeWidth=1).encode(x="a:Q", y="a:Q"),
        alt.Chart(_cal).mark_line(color=C["blue"], point=alt.OverlayMarkDef(size=40, color=C["blue"])).encode(
            x=alt.X("e:Q", title="Elo expected score"), y=alt.Y("w:Q", title="actual score (win 1, draw ½)"), tooltip=[alt.Tooltip("e:Q", format=".2f"), alt.Tooltip("w:Q", format=".2f"), "n:Q"]),
    ).properties(width=220, height=200, title="Is Elo calibrated? (since 2000)")
    mo.vstack([
        mo.hstack([elo_teams, elo_k, elo_hfa], justify="start", gap=1),
        mo.hstack([_lines, _top_ch], align="start"),
        mo.hstack([_cal_ch, mo.vstack([mo.stat(f"{_acc:.0%}", label="winner called correctly", caption="decisive matches since 2000"),
                                       note("A single number per team, updated after each match, already gets most results right and is well calibrated.")])], justify="start", gap=2, align="center"),
    ])
    return


@app.cell
def _(intl):
    pois_years = mo.ui.slider(2, 8, value=4, step=1, label="Fit on the last N years", show_value=True)
    pois_decay = mo.ui.switch(value=True, label="Weight recent matches more (half-life 1.5 years)")
    return pois_decay, pois_years


@app.cell
def _(intl, pois_decay, pois_years):
    _cut = intl.date.max() - pd.Timedelta(days=365 * pois_years.value)
    _d = intl[intl.date >= _cut]
    _cnt = pd.concat([_d.home_team, _d.away_team]).value_counts()
    POIS_TEAMS = sorted(_cnt[_cnt >= 10].index)
    _d = _d[_d.home_team.isin(POIS_TEAMS) & _d.away_team.isin(POIS_TEAMS)]
    _idx = {t: i for i, t in enumerate(POIS_TEAMS)}
    _n, _T = len(_d), len(POIS_TEAMS)
    _X = np.zeros((2 * _n, 2 * _T + 1))
    _r = np.arange(_n)
    _h, _a = _d.home_team.map(_idx).values, _d.away_team.map(_idx).values
    _X[_r, _h] = 1
    _X[_r, _T + _a] = 1
    _X[_r, 2 * _T] = (~_d.neutral.astype(bool)).values
    _X[_n + _r, _a] = 1
    _X[_n + _r, _T + _h] = 1
    _yv = np.concatenate([_d.home_score.values, _d.away_score.values])
    _age = (intl.date.max() - _d.date).dt.days.values / 365
    _wt = np.tile(0.5 ** (_age / 1.5) if pois_decay.value else np.ones(_n), 2)
    _glm = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(_X, _yv, sample_weight=_wt)
    POIS = dict(b0=_glm.intercept_, att=dict(zip(POIS_TEAMS, _glm.coef_[:_T])), dfn=dict(zip(POIS_TEAMS, _glm.coef_[_T:2 * _T])), home=_glm.coef_[-1], n=_n)
    return POIS, POIS_TEAMS


@app.cell
def _(POIS_TEAMS):
    pois_a = mo.ui.dropdown(POIS_TEAMS, value="Portugal" if "Portugal" in POIS_TEAMS else POIS_TEAMS[0], label="Team A", searchable=True)
    pois_b = mo.ui.dropdown(POIS_TEAMS, value="Spain" if "Spain" in POIS_TEAMS else POIS_TEAMS[1], label="Team B", searchable=True)
    pois_venue = mo.ui.radio(["Neutral", "Team A at home"], value="Neutral", inline=True)
    return pois_a, pois_b, pois_venue


@app.cell(hide_code=True)
def _(POIS, pois_a, pois_b, pois_decay, pois_venue, pois_years):
    from scipy.stats import poisson as _poisson
    _home = POIS["home"] if pois_venue.value != "Neutral" else 0.0
    _la = np.exp(POIS["b0"] + POIS["att"][pois_a.value] + POIS["dfn"][pois_b.value] + _home)
    _lb = np.exp(POIS["b0"] + POIS["att"][pois_b.value] + POIS["dfn"][pois_a.value])
    _g = np.arange(7)
    _M = np.outer(_poisson.pmf(_g, _la), _poisson.pmf(_g, _lb))
    _M /= _M.sum()
    _d = pd.DataFrame([(i, j, _M[i, j]) for i in _g for j in _g], columns=["a", "b", "p"])
    _d["lab"] = np.where(_d.p >= 0.04, (_d.p * 100).round(0).astype(int).astype(str) + "%", "")
    _best = _d.loc[_d.p.idxmax()]
    _heat = alt.Chart(_d).mark_rect(stroke=C["surface"], strokeWidth=2).encode(
        x=alt.X("b:O", title=f"{pois_b.value} goals", axis=alt.Axis(labelAngle=0)), y=alt.Y("a:O", title=f"{pois_a.value} goals", sort="descending"),
        color=alt.Color("p:Q", scale=alt.Scale(range=C["seq"]), legend=None), tooltip=[alt.Tooltip("a:O", title=pois_a.value), alt.Tooltip("b:O", title=pois_b.value), alt.Tooltip("p:Q", format=".1%")])
    _txt = alt.Chart(_d).mark_text(fontSize=10).encode(x="b:O", y=alt.Y("a:O", sort="descending"), text="lab:N",
                                                       color=alt.condition("datum.p > 0.08", alt.value("#ffffff"), alt.value(C["ink2"])))
    _pw, _pd = np.tril(_M, -1).sum(), np.trace(_M)
    mo.vstack([
        mo.hstack([pois_years, pois_decay], justify="start", gap=2),
        mo.hstack([pois_a, pois_b, pois_venue], justify="start", gap=1.5),
        mo.hstack([
            alt.layer(_heat, _txt).properties(width=300, height=300, title=alt.Title("Scoreline probabilities", subtitle=f"Most likely: {int(_best.a)}–{int(_best.b)}")),
            mo.vstack([
                mo.stat(f"{_pw:.0%}", label=f"{pois_a.value} win"), mo.stat(f"{_pd:.0%}", label="draw"), mo.stat(f"{1 - _pw - _pd:.0%}", label=f"{pois_b.value} win"),
                note(f"Expected goals: {_la:.2f} – {_lb:.2f}. Fitted on {POIS['n']:,} matches."),
            ]),
        ], justify="start", gap=2, align="center"),
    ])
    return


@app.cell
def _(POIS, POIS_TEAMS):
    _strength = pd.Series({t: POIS["att"][t] - POIS["dfn"][t] for t in POIS_TEAMS}).sort_values(ascending=False)
    sim_teams = mo.ui.multiselect(POIS_TEAMS, value=list(_strength.index[:8]), max_selections=8, label="Eight teams (bracket: 1v8, 4v5, 2v7, 3v6 in the order picked)")
    sim_run = mo.ui.run_button(label="Simulate 20,000 tournaments")
    return sim_run, sim_teams


@app.cell(hide_code=True)
def _(POIS, sim_run, sim_teams):
    _teams = list(sim_teams.value)
    if len(_teams) != 8:
        _out = mo.callout(mo.md("Pick exactly **eight** teams."), kind="warn")
    else:
        _rng = np.random.default_rng()
        _N = 20000

        def _play(a, b):
            la = np.exp(POIS["b0"] + POIS["att"][a] + POIS["dfn"][b])
            lb = np.exp(POIS["b0"] + POIS["att"][b] + POIS["dfn"][a])
            return la, lb

        def _round(sides):
            nxt = []
            for k in range(0, len(sides), 2):
                a, b = sides[k], sides[k + 1]
                ga = np.zeros(_N)
                gb = np.zeros(_N)
                for ta in np.unique(a):
                    for tb in np.unique(b):
                        m = (a == ta) & (b == tb)
                        if m.any():
                            la, lb = _play(ta, tb)
                            ga[m], gb[m] = _rng.poisson(la, m.sum()), _rng.poisson(lb, m.sum())
                coin = _rng.random(_N) < 0.5  # draws go to penalties: a coin flip
                nxt.append(np.where((ga > gb) | ((ga == gb) & coin), a, b))
            return nxt

        _seed = [_teams[i] for i in (0, 7, 3, 4, 1, 6, 2, 5)]
        _sides = [np.full(_N, t, dtype=object) for t in _seed]
        while len(_sides) > 1:
            _sides = _round(_sides)
        _win = pd.Series(_sides[0]).value_counts(normalize=True).reindex(_teams).fillna(0).rename_axis("team").reset_index(name="p")
        _out = alt.Chart(_win).mark_bar(color=C["blue"], cornerRadiusEnd=3, height=14).encode(
            x=alt.X("p:Q", title="probability of winning the tournament", axis=alt.Axis(format=".0%")), y=alt.Y("team:N", sort="-x", title=None),
            tooltip=["team:N", alt.Tooltip("p:Q", format=".1%")],
        ).properties(width=480, height=alt.Step(22), title=alt.Title("20,000 simulated knockouts", subtitle="Neutral venues; drawn ties decided by a coin flip"))
    mo.vstack([mo.md("### A tournament simulator"), sim_teams, sim_run, _out])
    return


@app.cell(hide_code=True)
def _():
    further(
        ("penaltyblog (Dixon-Coles & friends)", "https://github.com/martineastwood/penaltyblog"),
        ("World Football Elo Ratings", "https://www.eloratings.net/about"),
        ("Open International Soccer Database (2017 challenge)", "https://doi.org/10.1007/s10994-018-5726-0"),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 18 · Penalties and game theory

    A penalty is a two-player game: the kicker picks a side and the keeper dives at the same instant.
    Game theory says both should **randomise** so the other can't exploit them, and
    [Palacios-Huerta (2003)](https://doi.org/10.1111/1467-937X.00249) found that professionals do, almost exactly.
    """)
    return


@app.cell(hide_code=True)
def _():
    snapshot("s18")
    return


@app.cell(hide_code=True)
def _(shots):
    _p = shots[(shots.shot_type == "Penalty") & shots.end_z.notna() & shots.end_y.notna()].assign(
        result=lambda d: np.select([d.goal == 1, d.outcome.isin(["Saved", "Saved To Post", "Saved to Post"])], ["Scored", "Saved"], "Missed"))
    _p = _p[(_p.end_y.between(W / 2 - 6, W / 2 + 6)) & (_p.end_z < 4)]
    _frame = pd.DataFrame(dict(x=[W / 2 - 3.66, W / 2 - 3.66, W / 2 + 3.66, W / 2 + 3.66], y=[0, 2.44, 2.44, 0], o=range(4)))
    _X = alt.Scale(domain=[W / 2 + 5.5, W / 2 - 5.5], nice=False)
    _Y = alt.Scale(domain=[0, 3.6], nice=False)
    _rs = alt.Scale(domain=["Scored", "Saved", "Missed"], range=[C["blue"], C["orange"], C["muted"]])
    _pd, _pf = photo_data(_p, "player_id", "player")
    _goal = alt.layer(
        alt.Chart(_frame).mark_line(color=C["ink2"], strokeWidth=3).encode(x=alt.X("x:Q", scale=_X, axis=None), y=alt.Y("y:Q", scale=_Y, axis=None), order="o:Q"),
        alt.Chart(_pd).transform_lookup(lookup="_pid", from_=_pf).mark_circle(size=55, opacity=0.75, stroke=C["surface"], strokeWidth=1).encode(
            x=alt.X("end_y:Q", scale=_X, axis=None), y=alt.Y("end_z:Q", scale=_Y, axis=None),
            color=alt.Color("result:N", scale=_rs, legend=alt.Legend(title=None, orient="bottom")),
            tooltip=photo_tip("player", "team:N", "result:N", "minute:Q")),
    ).properties(width=560, height=230, title=alt.Title(f"{len(_p)} penalties, seen from behind the kicker", subtitle=f"Converted {(_p.result == 'Scored').mean():.0%} · shoot-outs included"))
    _p = _p.assign(side=pd.cut(_p.end_y - W / 2, [-99, -1.22, 1.22, 99], labels=["Right third", "Centre", "Left third"]),
                   height=np.where(_p.end_z > 1.2, "High", "Low"))
    _tab = _p.groupby(["height", "side"], observed=True).agg(n=("goal", "size"), rate=("goal", "mean")).reset_index()
    _hm = alt.Chart(_tab).mark_rect(stroke=C["surface"], strokeWidth=3).encode(
        x=alt.X("side:N", sort=["Left third", "Centre", "Right third"], title=None, axis=alt.Axis(labelAngle=0, orient="top")), y=alt.Y("height:N", sort=["High", "Low"], title=None),
        color=alt.Color("rate:Q", scale=alt.Scale(range=C["seq"], domain=[0.5, 1]), legend=None), tooltip=["side:N", "height:N", "n:Q", alt.Tooltip("rate:Q", format=".0%")])
    _ht = alt.Chart(_tab).mark_text(fontSize=12).encode(x=alt.X("side:N", sort=["Left third", "Centre", "Right third"]), y=alt.Y("height:N", sort=["High", "Low"]),
                                                       text=alt.Text("rate:Q", format=".0%"), color=alt.condition("datum.rate > 0.8", alt.value("#ffffff"), alt.value(C["ink"])))
    mo.vstack([_goal, alt.layer(_hm, _ht).properties(width=560, height=110, title="Conversion rate by target zone")])
    return


@app.cell
def _():
    pen_ll = mo.ui.slider(0.3, 1.0, value=0.58, step=0.01, label="Kick natural side · keeper dives natural side", show_value=True)
    pen_lr = mo.ui.slider(0.3, 1.0, value=0.95, step=0.01, label="Kick natural side · keeper dives other side", show_value=True)
    pen_rl = mo.ui.slider(0.3, 1.0, value=0.93, step=0.01, label="Kick other side · keeper dives natural side", show_value=True)
    pen_rr = mo.ui.slider(0.3, 1.0, value=0.70, step=0.01, label="Kick other side · keeper dives other side", show_value=True)
    return pen_ll, pen_lr, pen_rl, pen_rr


@app.cell(hide_code=True)
def _(pen_ll, pen_lr, pen_rl, pen_rr):
    _a, _b, _c, _d = pen_ll.value, pen_lr.value, pen_rl.value, pen_rr.value
    _den = _a - _b - _c + _d
    _ok = abs(_den) > 1e-9
    _q = (_d - _b) / _den if _ok else np.nan  # keeper: P(dive natural side)
    _p = (_d - _c) / _den if _ok else np.nan  # kicker: P(kick natural side)
    _interior = _ok and 0 < _q < 1 and 0 < _p < 1
    _v = _q * _a + (1 - _q) * _b if _interior else np.nan
    _qs = np.linspace(0, 1, 51)
    _lines = pd.DataFrame(dict(q=np.tile(_qs, 2), score=np.concatenate([_qs * _a + (1 - _qs) * _b, _qs * _c + (1 - _qs) * _d]),
                               kick=["Kick natural side"] * 51 + ["Kick other side"] * 51))
    _ks = alt.Scale(domain=["Kick natural side", "Kick other side"], range=[C["blue"], C["orange"]])
    _layers = [alt.Chart(_lines).mark_line().encode(
        x=alt.X("q:Q", title="how often the keeper dives to the kicker's natural side", axis=alt.Axis(format=".0%")),
        y=alt.Y("score:Q", title="kicker's chance of scoring", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0.3, 1])),
        color=alt.Color("kick:N", scale=_ks, legend=alt.Legend(title=None, orient="top")))]
    if _interior:
        _layers.append(alt.Chart(pd.DataFrame(dict(q=[_q], score=[_v]))).mark_circle(size=140, color=C["ink"], opacity=1).encode(x="q:Q", y="score:Q"))
        _layers.append(alt.Chart(pd.DataFrame(dict(q=[_q], score=[_v]))).mark_text(align="left", dx=10, dy=12, color=C["ink2"]).encode(x="q:Q", y="score:Q", text=alt.value("equilibrium")))
    _chart = alt.layer(*_layers).properties(width=440, height=260, title=alt.Title("The keeper's dilemma", subtitle="Where the lines cross, the kicker no longer cares which side to choose"))
    _stats = mo.vstack([
        mo.stat(f"{_p:.0%}" if _interior else "–", label="kicker: natural side", caption="Nash prediction · observed 40%"),
        mo.stat(f"{_q:.0%}" if _interior else "–", label="keeper: natural side", caption="Nash prediction · observed 42%"),
        mo.stat(f"{_v:.0%}" if _interior else "–", label="scoring probability"),
    ]) if _interior else mo.callout(mo.md("No mixed equilibrium: one side is always better, so a pure strategy wins."), kind="info")
    mo.vstack([
        mo.md("**Scoring probabilities** (defaults from Palacios-Huerta's 1,417 professional penalties; *natural side* = left for a right-footer):"),
        mo.hstack([mo.vstack([pen_ll, pen_lr]), mo.vstack([pen_rl, pen_rr])], widths="equal"),
        mo.hstack([_chart, _stats], widths=[3, 1], align="center"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Test a published claim yourself

    [Apesteguia & Palacios-Huerta (2010)](https://doi.org/10.1257/aer.100.5.2548) reported that the team kicking **first** in a shoot-out wins
    **60.5%** of the time, which they attributed to psychological pressure. [Kocher, Lenz & Sutter (2012)](https://doi.org/10.1287/mnsc.1120.1516)
    used four times the data and found **53.3%**, not significantly different from 50%. Here is what the open international record says.
    """)
    return


@app.cell
def _():
    so = read_csv("intl_shootouts.csv", parse_dates=["date"])
    so_years = mo.ui.range_slider(1970, int(so.date.dt.year.max()), value=[1970, int(so.date.dt.year.max())], step=1, label="Years", full_width=True)
    return so, so_years


@app.cell(hide_code=True)
def _(so, so_years):
    _d = so[so.first_shooter.notna() & so.date.dt.year.between(*so_years.value)]
    _n = len(_d)
    _k = int((_d.first_shooter == _d.winner).sum())
    _p = _k / max(_n, 1)
    _z = 1.96
    _c = (_p + _z**2 / (2 * _n)) / (1 + _z**2 / _n) if _n else np.nan
    _hw = _z * np.sqrt(_p * (1 - _p) / _n + _z**2 / (4 * _n**2)) / (1 + _z**2 / _n) if _n else np.nan
    _ref = pd.DataFrame(dict(who=["Apesteguia & Palacios-Huerta 2010", "Kocher, Lenz & Sutter 2012", "This data"], p=[0.605, 0.533, _p],
                             lo=[np.nan, np.nan, _c - _hw], hi=[np.nan, np.nan, _c + _hw]))
    _sc = alt.Scale(domain=[0.35, 0.75])
    _ch = alt.layer(
        alt.Chart(pd.DataFrame(dict(p=[0.5]))).mark_rule(color=C["axis"], strokeWidth=1.5).encode(x=alt.X("p:Q", scale=_sc)),
        alt.Chart(_ref.dropna()).mark_rule(color=C["blue"], strokeWidth=3).encode(x=alt.X("lo:Q", scale=_sc), x2="hi:Q", y=alt.Y("who:N", sort=None, title=None)),
        alt.Chart(_ref).mark_circle(size=130, opacity=1).encode(
            x=alt.X("p:Q", scale=_sc, title="share of shoot-outs won by the team kicking first", axis=alt.Axis(format=".0%", values=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])),
            y=alt.Y("who:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.condition("datum.who == 'This data'", alt.value(C["blue"]), alt.value(C["muted"])), tooltip=["who:N", alt.Tooltip("p:Q", format=".1%")]),
    ).properties(width=520, height=130)
    mo.vstack([so_years, _ch, mo.md(f"{so_years.value[0]}–{so_years.value[1]}: **{_k} of {_n}** shoot-outs with a known first kicker went to the team that kicked first (**{_p:.1%}**, 95% CI {_c - _hw:.1%}–{_c + _hw:.1%}).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 19 · From data to words

    Event data is already structured, so a few templates turn it into **commentary**. Large language models make it fluent;
    the data keeps it honest. Below is a template-based commentary for the 2022 World Cup final, with each chance's xG.
    Then comes the compact, *grounded* context you would hand an LLM instead of asking it to remember the game.
    """)
    return


@app.cell(hide_code=True)
def _():
    snapshot("s19")
    return


@app.cell
def _():
    mo.stop(not FULL)
    words_range = mo.ui.range_slider(0, 125, value=[78, 82], step=1, label="Minutes", show_value=True, full_width=True)
    words_range
    return (words_range,)


@app.cell(hide_code=True)
def _(PLAYER, final_events, words_range):
    def _name(e):
        p = e.get("player") or {}
        return PLAYER.get(p.get("id"), p.get("name", e["team"]["name"]))

    def _who(e):  # for display: a face next to the name (the LLM prompt below keeps plain names)
        return chip((e.get("player") or {}).get("id"), _name(e), 20)

    _lines, _ctx = [], []
    _score = {"Argentina": 0, "France": 0}
    for _e in final_events:
        _t = _e["type"]["name"]
        _mn = _e["minute"]
        _ts = f"{_mn}′"
        _team = _e["team"]["name"]
        _txt = None
        if _t == "Shot":
            _s = _e["shot"]
            _xg = _s.get("statsbomb_xg", 0)
            _o = _s["outcome"]["name"]
            _kind = "penalty" if _s["type"]["name"] == "Penalty" else ("header" if _s["body_part"]["name"] == "Head" else "shot")
            _q = "a gilt-edged chance" if _xg > 0.4 else ("a decent opening" if _xg > 0.12 else "a speculative effort")
            if _o == "Goal":
                if _e["period"] < 5:
                    _score[_team] += 1
                _txt = f"**GOAL!** {_who(_e)} ({_team}) scores with a {_kind}. {_q.capitalize()} (xG {_xg:.2f}). Argentina {_score['Argentina']}–{_score['France']} France."
            else:
                _txt = f"{_who(_e)} tries a {_kind}, {_q} (xG {_xg:.2f}): {_o.lower()}."
            _ctx.append(f"{_mn}' SHOT team={_team} player={_name(_e)} type={_kind} xg={_xg:.2f} outcome={_o}")
        elif _t == "Substitution":
            _rep = _e["substitution"]["replacement"]
            _txt = f"{_team} change: {_who(_e)} off, {chip(_rep['id'], PLAYER.get(_rep['id'], _rep['name']), 20)} on."
            _ctx.append(f"{_mn}' SUB team={_team} off={_name(_e)} on={_e['substitution']['replacement']['name']}")
        elif _t == "Foul Committed" and "card" in _e.get("foul_committed", {}):
            _txt = f"{_e['foul_committed']['card']['name']} for {_who(_e)} ({_team})."
            _ctx.append(f"{_mn}' CARD team={_team} player={_name(_e)} card={_e['foul_committed']['card']['name']}")
        elif _t == "Pass" and _e["pass"].get("goal_assist"):
            _rcp = _e["pass"].get("recipient") or {}
            _txt = f"{_who(_e)} picks out {chip(_rcp.get('id'), PLAYER.get(_rcp.get('id'), _rcp.get('name', '')), 20)}…"
            _ctx.append(f"{_mn}' ASSIST team={_team} player={_name(_e)}")
        elif _t == "Dribble" and _e["dribble"]["outcome"]["name"] == "Complete" and _e.get("location", [0])[0] > 90:
            _txt = f"{_who(_e)} beats a man in the final third."
        if _txt and words_range.value[0] <= _mn <= words_range.value[1]:
            _lines.append(f"<div style='margin:2px 0'><span style='color:{C['muted']};display:inline-block;width:3em'>{_ts}</span>{mo.md(_txt).text}</div>")
    _ctx = [c for c in _ctx if words_range.value[0] <= int(c.split("'")[0]) <= words_range.value[1]]
    _prompt = (
        "You are a football commentator. Using ONLY the events below (do not add facts), write three lively sentences "
        "about this passage of the 2022 World Cup final, Argentina v France.\n\n" + "\n".join(_ctx)
    )
    mo.vstack([
        mo.Html(f"<div style='max-height:280px;overflow:auto;padding:8px 12px;border:1px solid {C['grid']};border-radius:8px'>"
                + ("".join(_lines) or "<em>Nothing notable in this window. Widen it.</em>") + "</div>"),
        mo.accordion({f"The grounded LLM prompt for this window (~{len(_prompt) // 4} tokens)": mo.md(f"```text\n{_prompt}\n```")}),
    ])
    return


@app.cell(hide_code=True)
def _():
    further(
        ("SoccerNet-Caption (dense video commentary)", "https://www.soccer-net.org/tasks/dense-video-captioning"),
        ("MatchTime (EMNLP 2024)", "https://github.com/jyrao/MatchTime"),
        ("Large Events Model (Mendes-Neves et al.)", "https://arxiv.org/abs/2405.08397"),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Extra time · Horizons
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 20 · A timeline of football analytics

    From Charles Reep's notebooks in the 1950s to graph neural networks designing corner kicks: seventy-odd landmarks,
    each checked against its source. **Click a dot** for the details; filled dots have open code or data.
    """)
    return


@app.cell
def _():
    timeline = pd.DataFrame(read_json("timeline.json"))
    tl_kinds = mo.ui.multiselect(sorted(timeline.kind.unique()), value=sorted(timeline.kind.unique()), label="Kinds")
    tl_open = mo.ui.switch(value=False, label="Only open code / data")
    return timeline, tl_kinds, tl_open


@app.cell
def _(timeline, tl_kinds, tl_open):
    _d = timeline[timeline.kind.isin(tl_kinds.value)]
    if tl_open.value:
        _d = _d[_d.open]
    _d = _d.assign(slot=_d.groupby(["year", "theme"]).cumcount(), status=np.where(_d.open, "Open code / data", "Closed or paper only"), is_open=_d.open.astype(int))  # 0/1: marimo may send booleans to the browser as text
    _themes = list(timeline.groupby("theme").year.min().sort_values().index)
    _pick = alt.selection_point(fields=["title"], on="click", empty=False, name="tl_pick")
    _ch = alt.Chart(_d).mark_point(size=110, strokeWidth=2).encode(
        x=alt.X("year:Q", title=None, scale=alt.Scale(domain=[1965, 2027], nice=False), axis=alt.Axis(format="d", values=list(range(1970, 2030, 10)))),
        y=alt.Y("theme:N", sort=_themes, title=None, axis=alt.Axis(labelColor=C["ink2"], domain=False, ticks=False)),
        yOffset=alt.YOffset("slot:Q", scale=alt.Scale(domain=[-1, 4])),
        color=alt.Color("status:N", scale=alt.Scale(domain=["Open code / data", "Closed or paper only"], range=[C["blue"], C["orange"]]), legend=alt.Legend(title=None, orient="top")),
        fill=alt.condition("datum.is_open == 1", alt.value(C["blue"]), alt.value(C["surface"])),
        size=alt.condition(_pick, alt.value(300), alt.value(110)),
        tooltip=["year:Q", "title:N", "who:N", "kind:N"],
    ).add_params(_pick).properties(width=640, height=alt.Step(30))
    tl_chart = mo.ui.altair_chart(_ch, chart_selection=False, legend_selection=False)
    return (tl_chart,)


@app.cell(hide_code=True)
def _(timeline, tl_chart, tl_kinds, tl_open):
    _sel = picked(tl_chart, timeline)
    if len(_sel):
        _t = _sel.title.iloc[0]
        _r = timeline.set_index("title").loc[_t]
        _links = f"[source]({_r.url})" + (f" · [code / data]({_r.code})" if isinstance(_r.code, str) and _r.code else "")
        _card = mo.callout(mo.md(f"**{_t}** ({_r.year})  \n{_r.who}{' · ' + _r.venue if isinstance(_r.venue, str) else ''}\n\n{_r.summary}\n\n{_links}"), kind="neutral")
    else:
        _card = mo.callout(mo.md("Click any dot."), kind="neutral")
    mo.vstack([mo.hstack([tl_kinds, tl_open], justify="start", gap=2), tl_chart, _card])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 21 · The frontier

    What's being built now (mostly 2021–2026), grouped by area, with links to the paper and, where it exists, the code.
    The **weekend projects** are things you could start tonight with the open data in this notebook.
    """)
    return


@app.cell(hide_code=True)
def _():
    frontier = pd.DataFrame(read_json("frontier.json"))
    _ideas = frontier[frontier.summary.str.startswith("IDEA:")]
    _items = frontier[~frontier.summary.str.startswith("IDEA:")].sort_values("year", ascending=False)

    def _card(r):
        code = f" · [code]({r.code})" if isinstance(r.code, str) and r.code else ""
        venue = f" · {r.venue}" if isinstance(r.venue, str) and r.venue else ""
        return f"- **[{r.title}]({r.url})** ({r.year}{venue})  \n  {r.who}. {r.summary}{code}"

    _tabs = {a: mo.md("\n".join(_card(r) for r in g.itertuples())) for a, g in _items.groupby("area")}
    _idea_cards = [
        mo.callout(mo.md(f"**{r.title.removeprefix('Weekend idea: ').capitalize()}**  \n<span style='color:{C['muted']}'>{r.area}</span>\n\n{r.summary.removeprefix('IDEA: ')}\n\n↳ [where the idea comes from]({r.url})"), kind="success")
        for r in _ideas.itertuples()
    ]
    mo.vstack([
        mo.ui.tabs(_tabs),
        mo.md("### Weekend projects"),
        mo.hstack(_idea_cards[:3], widths="equal", align="stretch"),
        mo.hstack(_idea_cards[3:6], widths="equal", align="stretch"),
    ])
    return


@app.cell(hide_code=True)
def _(timeline):
    _papers = timeline[timeline.kind == "paper"].sort_values(["year", "who"])
    _plist = "\n".join(f"* {r.who} ({r.year}). [{r.title}]({r.url}). *{r.venue}*." if isinstance(r.venue, str) else f"* {r.who} ({r.year}). [{r.title}]({r.url})." for r in _papers.itertuples())
    mo.md(f"""
    # Full time · References

    ### Data
    * **[StatsBomb Open Data](https://github.com/statsbomb/open-data)**: events, line-ups and 360 frames. Used under the
      StatsBomb Public Data User Agreement, which requires attribution: *data provided by StatsBomb (Hudl)*.
    * **[Metrica Sports sample data](https://github.com/metrica-sports/sample-data)**: anonymised tracking and events.
    * **[International football results 1872–present](https://github.com/martj42/international_results)** by martj42 (CC0).
    * **[transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets)** by dcaribou (CC0).

    ### Books
    * Anderson, C. & Sally, D. (2013). **The Numbers Game: Why Everything You Know About Football Is Wrong**. Penguin.
    * Sumpter, D. (2016). **Soccermatics: Mathematical Adventures in the Beautiful Game**. Bloomsbury Sigma.
    * Biermann, C. (2019). **Football Hackers: The Science and Art of a Data Revolution**. Blink.
    * Palacios-Huerta, I. (2014). **Beautiful Game Theory: How Soccer Can Help Economics**. Princeton University Press.
    * Kuper, S. & Szymanski, S. (2009, updated 2022). **Soccernomics**. Nation Books.
    * Graham, I. (2024). **How to Win the Premier League: The Inside Story of Football's Data Revolution**. Century.

    ### Packages
    * [marimo](https://marimo.io): *reactive Python notebooks*
    * [pandas](https://pandas.pydata.org) · [NumPy](https://numpy.org) · [SciPy](https://scipy.org) · [scikit-learn](https://scikit-learn.org) · [NetworkX](https://networkx.org)
    * [Vega-Altair](https://altair-viz.github.io) · [Plotly](https://plotly.com/python/)
    * Football-specific, for going further: [kloppy](https://github.com/PySport/kloppy) · [mplsoccer](https://github.com/andrewRowlinson/mplsoccer) ·
      [socceraction](https://github.com/ML-KULeuven/socceraction) · [floodlight](https://github.com/floodlight-sports/floodlight) ·
      [databallpy](https://github.com/Alek050/databallpy) · [penaltyblog](https://github.com/martineastwood/penaltyblog) ·
      [statsbombpy](https://github.com/statsbomb/statsbombpy) · [worldfootballR](https://github.com/JaseZiv/worldfootballR)

    ### Papers
    {_plist}
    """)
    return


@app.cell(hide_code=True)
def _():
    _c = pd.DataFrame([
        dict(player=v.get("player", ""), photo=f"[{v['title'][:70]}]({v['page']})", author=v["author"][:90], licence=v["license"])
        for v in sorted(PHOTOS.values(), key=lambda v: v.get("player", ""))
    ])
    mo.stop(_c.empty)
    mo.accordion({f"Credits for the {len(_c)} player photos": mo.vstack([
        mo.md("Portraits from [Wikimedia Commons](https://commons.wikimedia.org/), each under the licence shown. Cropped to the face and resized."),
        mo.ui.table(_c, selection=None, page_size=10, show_column_summaries=False, format_mapping={"photo": lambda t: mo.md(t)}),
    ])})
    return


if __name__ == "__main__":
    app.run()
