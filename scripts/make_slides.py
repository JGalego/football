"""Present the notebook as a slide deck with marimo's own slides layout.

marimo takes a notebook's layout from `layout_file` in `marimo.App(...)` and has no command-line switch for it, so setting
it on the_beautiful_data.py would turn the Pages site and the Space into slides too. Instead this script writes three
untracked files (see .gitignore), so this script is the one source to edit:

  * the_beautiful_data_slides.py: a copy of the notebook with slide-sized text (below) whose `marimo.App(...)` points at:
  * layouts/the_beautiful_data.slides.json: a marimo slides layout (one entry per cell: slide type and speaker notes);
  * layouts/the_beautiful_data.slides.head.html: head.html plus a little CSS/JS that fits each slide's content to the slide.

Slide text. A notebook paragraph is a wall of text on a slide, so in the copy each section intro keeps its heading and gets
the one-line takeaway in SLIDE_TEXT; the notebook's own paragraph becomes that slide's speaker notes (press S).

Slide structure. The layout matches cells by position, so it is worked out from the notebook rather than written by hand;
re-run this script after adding or moving cells:

  * a "# Half" banner, a "## N · Section" intro, the title, the kick-off and full time each start a new slide;
  * every other cell with output becomes a sub-slide under its section (→ next section, ↓ its charts);
  * a "### Sub-heading" or a lone control (e.g. a match picker) shares a slide with the cell after it, as a fragment;
  * links, credits, the sidebar, and pages of fine print (SKIP_*) are left out; marimo drops cells without output itself.

    uv run python scripts/make_slides.py          # write the files
    uv run python scripts/make_slides.py --run    # ...then present: uv run marimo run the_beautiful_data_slides.py

During the talk: space or → / ↓ to move on, S for the speaker view, F for full screen.
Tweaks made in marimo's slide editor (marimo edit the_beautiful_data_slides.py) are lost the next time this script runs.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "the_beautiful_data.py"
SLIDES_NOTEBOOK = ROOT / "the_beautiful_data_slides.py"
LAYOUT = Path("layouts") / "the_beautiful_data.slides.json"  # paths relative to the notebook, as marimo stores them
HEAD = Path("layouts") / "the_beautiful_data.slides.head.html"

DECK = {"transition": "slide", "verticalAlign": "center"}

# What each text cell says on its slide, keyed by its heading ("## 10 · Expected goals (xG)" -> "10"; "### Markets: ..."
# -> "Markets"). One idea per slide; the notebook's paragraph moves to the speaker notes.
SLIDE_TEXT = {
    "intro": "Every pass, sprint, heartbeat and transfer now leaves a trace.\n\n**Public data** only · every model **built from scratch**",
    # (the empty div keeps the agenda from being squeezed by the pitch guide beside it)
    "Kick-off": "- **First half** · the data\n- **Second half** · the models\n- **Extra time** · history, and what's next"
                '\n\n<div style="width:20em"></div>',
    "1": "From **one number** per match to **a trillion**",
    "The atlas": "Matches are only one source. **Click a cell.**",
    "2": "Every on-ball action, **~3,500 rows** a match",
    "3": "Goals are rare; **expected goals** tell the story of the chances",
    "4": "Add up thousands of events and **habits** appear",
    "5": "Events follow the ball; tracking follows **everyone**, 25 times a second",
    "6": "Positions → **speed** → distance, high-speed running, sprints",
    "7": "A **homography** maps pixels on the grass to metres on the pitch",
    "Pixels, sound and video": "Photo → teams · sound → spectrogram · film → shots. **No deep learning.**",
    "8": "**International results since 1872**: fewer goals, less home advantage",
    "9": "Two layers off the pitch: **bodies** and **markets**",
    "Markets": "Crowd-sourced values track real fees, and trace a **career** by age",
    "10": "The probability that a **shot** becomes a goal",
    "11": "Credit the **passes and carries** before the shot",
    "12": "Value **every action** by how it moves the odds of scoring and conceding",
    "13": "Who owns which part of the pitch? **Voronoi → intercept → influence**",
    "14": "Learn **ball → 22 positions** and you get a team's usual shape",
    "15": "Players are **nodes**, passes are **edges**",
    "16": "Profile every player **per 90 minutes**; similar players sit together",
    "17": "**Elo**, a **Poisson** goals model and a **Monte Carlo** tournament",
    "18": "A two-player game, and the pros **randomise** as theory says",
    "Test a published claim yourself": "Does kicking **first** win shoot-outs? **60.5%** (2010) vs **53.3%** (2012)",
    "19": "Structured events + **templates** (or an LLM) = commentary",
    "20": "From Reep's notebooks in the 1950s to **graph neural networks**",
    "21": "What's being built now, and **weekend projects** you could start tonight.\n\nBrowse all 74 in the notebook.",
    "Full time": (
        "### Data\n"
        "- **StatsBomb Open Data**: events and 360 frames · *data provided by StatsBomb (Hudl)*\n"
        "- **Metrica Sports** sample tracking data\n"
        "- **International results since 1872** and **transfermarkt-datasets** (both CC0)\n\n"
        "### Further reading\n"
        "- Anderson & Sally, *The Numbers Game* (2013) · Sumpter, *Soccermatics* (2016)\n"
        "- Biermann, *Football Hackers* (2019) · Palacios-Huerta, *Beautiful Game Theory* (2014)\n"
        "- Graham, *How to Win the Premier League* (2024)\n\n"
        "Built with **marimo** · github.com/JGalego/football"
    ),
}
# Fine print that cannot be read at slide size: a text cell by heading (it and the cell it introduces), or a cell by output.
SKIP_HEADINGS = {"An open-data catalogue"}
SKIP_CALLS = ("mo.sidebar(", "mo.callout(", "further(", "mo.accordion(", "mo.vstack([mo.ui.tabs(")
# Layouts made for a scrolling page that waste a 16:9 slide: charts stacked vertically go side by side, and a 12-row
# table and a scroll box go. Exact edits to the copy; each must match once, so a change in the notebook stops the script, not the talk.
SLIDE_CODE = [
    ("    mo.vstack([_race_chart, _shotmap])",  # 3 · the xG race and the shot map
     '    mo.hstack([_race_chart, _shotmap], justify="start", align="center", gap=2)'),
    ("    mo.vstack([atlas_chart, _tot, _detail])",  # 1 · the atlas and the clicked cell's card
     '    mo.hstack([mo.vstack([atlas_chart, _tot]), _detail], justify="start", gap=2, widths=[3, 2])'),
    ("    mo.vstack([vaep_match, _flow, _arrows])",  # 12 · a match's action values and its 12 best actions
     '    mo.vstack([vaep_match, mo.hstack([_flow, _arrows], justify="start", align="center", gap=2)])'),
    ('        mo.image(_canvas, width=720, rounded=True), _readout,',  # 7 · photo demo: picture beside its readout
     '        mo.hstack([mo.image(_canvas, width=720, rounded=True), _readout], justify="start", align="center", gap=2),'),
    ('        media_video("media/stand_clip.webm"),\n        alt.vconcat(_a, _b, spacing=4),',  # 7 · the clip beside its two signals
     '        mo.hstack([media_video("media/stand_clip.webm"), alt.vconcat(_a, _b, spacing=4)], justify="start", align="center", gap=2),'),
    ("<div style='max-height:280px;overflow:auto;", "<div style='"),  # 19 · the commentary feed, without its own scrollbar
    # kick-off: the agenda beside the pitch guide (the markdown wrapped in a vstack, as marimo renders it display:contents)
    ('    mo.vstack([\n        mo.md("## Kick-off', '    mo.hstack([\n        mo.vstack([mo.md("## Kick-off'),
    ('"),\n        pitch_guide(),\n    ], gap=0.8)', '")]),\n        pitch_guide(),\n    ], gap=3, align="center", justify="start")'),
]
SLIDE_DROP_LINES = ["mo.ui.tabs({t: mo.ui.table(d, selection=None, page_size=11,"]  # 15 · the centrality table
# Cells that join the slide before them: the headline numbers and the StatsBomb credit go with the every-goal chart.
JOIN_PREVIOUS = ("mo.hstack([mo.stat(", "mo.vstack([x for x in (_lite, snapshot('hero')")

# The notebook's charts have fixed pixel widths sized for a scrolling page, so on a 16:9 slide they sit small in a corner.
# This sets CSS zoom on each slide's content (zoom re-wraps text, unlike a transform) to the largest factor that fits
# without scrolling, re-checking as charts finish drawing and when the window changes. Fragments show straight away: this
# deck only uses them to put two cells on one slide, not for click-by-click reveals. The intro TV is kept small.
SLIDES_HEAD = """
<!-- Slides only (scripts/make_slides.py): fit each slide's content to the slide, and show fragments at once. -->
<style>
  .reveal .slides section .fragment { opacity: 1 !important; visibility: visible !important; transform: none !important; }
  /* The intro television: a modest size beside the title rather than filling the slide (the slide's zoom still applies). */
  .reveal iframe[title^="A television"] { max-width: 340px !important; }
</style>
<script>
(() => {
  const MIN = 0.5, MAX = 1.8;
  const fits = (box) => box.scrollHeight <= box.clientHeight && box.scrollWidth <= box.clientWidth;
  const fit = (content) => {
    const box = content.closest(".overflow-auto");
    if (!box || !box.clientHeight) return;
    let lo = MIN, hi = MAX;
    content.style.zoom = hi;
    if (fits(box)) return;
    for (let i = 0; i < 8; i++) {  // binary search for the largest zoom that fits
      const mid = (lo + hi) / 2;
      content.style.zoom = mid;
      if (fits(box)) lo = mid; else hi = mid;
    }
    content.style.zoom = lo;
  };
  const present = () => document.querySelectorAll(".reveal section.present .mo-slide-content");
  let timer = 0;
  const refit = () => { clearTimeout(timer); timer = setTimeout(() => present().forEach(fit), 100); };
  // Slides change by class and outputs arrive as new nodes; charts and images can also grow without either (a canvas
  // resized by attribute, a photo loading), so also re-check the visible slide a few times a second and refit on overflow.
  new MutationObserver(refit).observe(document.documentElement, { subtree: true, childList: true, attributes: true, attributeFilter: ["class"] });
  window.addEventListener("resize", refit);
  document.addEventListener("fullscreenchange", refit);
  setInterval(() => present().forEach((c) => { const box = c.closest(".overflow-auto"); if (box && !fits(box)) fit(c); }), 300);
})();
</script>
"""


# ----------------------------------------------------------------------------- reading the notebook
def cells(tree: ast.Module) -> list[ast.stmt]:
    """The notebook's cells in marimo's order: the setup block, then every @app.cell / @app.function / @app.class_definition."""
    out: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.With) and "app.setup" in ast.unparse(node.items[0].context_expr) or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and any(
            ast.unparse(d).split("(")[0] in ("app.cell", "app.function", "app.class_definition") for d in node.decorator_list
        ):
            out.append(node)
    return out


def output_of(cell: ast.stmt) -> ast.Expr | None:
    """A cell's displayed expression (its last statement before `return`), or None if it shows nothing."""
    if not isinstance(cell, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    body = [s for s in cell.body if not isinstance(s, ast.Return)]
    return body[-1] if body and isinstance(body[-1], ast.Expr) else None


def md_args(node: ast.AST) -> list[ast.expr]:
    """The string arguments of the mo.md(...) calls inside a node, outermost first."""
    return [n.args[0] for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == "mo.md"
            and n.args and isinstance(n.args[0], (ast.Constant, ast.JoinedStr))]


def text_of(arg: ast.expr) -> str:
    """The markdown of a mo.md(...) argument; f-string placeholders show as {…}."""
    if isinstance(arg, ast.JoinedStr):
        return textwrap.dedent("".join(str(v.value) if isinstance(v, ast.Constant) else "{…}" for v in arg.values)).strip()
    return textwrap.dedent(str(arg.value) if isinstance(arg, ast.Constant) else "").strip()


def heading(md: str) -> tuple[int, str] | None:
    m = re.search(r"^(#{1,3}) (.+)$", md, flags=re.MULTILINE)
    return (len(m.group(1)), m.group(2).strip()) if m else None


def key_of(md: str) -> str | None:
    h = heading(md)
    if h is None:
        return "intro" if md.startswith("Every pass, sprint") else None
    if m := re.match(r"(\d+) ·", h[1]):
        return m.group(1)
    return next((k for k in SLIDE_TEXT if h[1] == k or h[1].startswith(k + ":") or h[1].startswith(k + " ·")), h[1])


def plain(md: str) -> str:
    """Markdown to speaker-note text."""
    md = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md)
    md = re.sub(r"<[^>]+>", "", md)
    md = re.sub(r"[*_`]", "", md)
    return re.sub(r"\s+", " ", md).strip()


# ----------------------------------------------------------------------------- building the deck
def slide_copy(source: str, tree: ast.Module) -> tuple[str, dict[int, str]]:
    """The notebook with SLIDE_TEXT swapped into its text cells, and {cell index: speaker notes from the original text}."""
    data = source.encode()
    starts = [0] + [i + 1 for i, b in enumerate(data) if b == ord("\n")]
    edits, notes = [], {}
    for i, cell in enumerate(cells(tree)):
        out = output_of(cell)
        for arg in md_args(out) if out else []:
            md = text_of(arg)
            key = key_of(md)
            if key not in SLIDE_TEXT:
                continue
            h = heading(md)
            first_line = md.splitlines()[0]
            new = SLIDE_TEXT[key] if h is None else f"{first_line}\n\n{SLIDE_TEXT[key]}"
            body = md if h is None else md.split("\n", 1)[1] if "\n" in md else ""
            if note := plain(body):
                notes[i] = note[:4000]
            lo = starts[arg.lineno - 1] + arg.col_offset
            hi = starts[(arg.end_lineno or arg.lineno) - 1] + (arg.end_col_offset or 0)
            edits.append((lo, hi, json.dumps(new, ensure_ascii=False).encode()))  # always "..." (a valid Python literal)
    for lo, hi, new in sorted(edits, reverse=True):
        data = data[:lo] + new + data[hi:]
    return data.decode(), notes


def build_layout(tree: ast.Module, notes: dict[int, str]) -> list[dict]:
    layout: list[dict] = []
    pair_next = skip_next = False
    all_cells = cells(tree)
    for i, cell in enumerate(all_cells):
        out = output_of(cell)
        expr = ast.unparse(out) if out else ""
        mds = [text_of(a) for a in md_args(out)] if out else []
        h = next((heading(m) for m in mds if heading(m)), None)
        entry: dict = {}
        if not expr or expr.startswith(SKIP_CALLS) or skip_next or (h and h[1] in SKIP_HEADINGS):
            entry["type"] = "skip"
            if expr:  # an output cell left out: so is the cell it introduces, and nothing pairs with it
                skip_next, pair_next = bool(h and h[1] in SKIP_HEADINGS), False
        elif h and (h[0] == 1 or (h[0] == 2 and (re.match(r"\d+ ·", h[1]) or h[1] == "Kick-off"))):
            entry["type"] = "slide"
            pair_next = False
        elif pair_next or expr.startswith(JOIN_PREVIOUS):
            entry["type"] = "fragment"
            pair_next = False
        else:
            entry["type"] = "sub-slide"
            # A "### ..." intro or a bare control (a name such as `story_match`) leads into the next cell.
            pair_next = bool(h and h[0] == 3 and expr.startswith("mo.md(")) or expr.isidentifier()
        if i in notes and entry["type"] != "skip":
            entry["speakerNotes"] = notes[i]
        layout.append(entry)
    # The title shares its slide with the television right after it.
    title = next(i for i, c in enumerate(all_cells) if (o := output_of(c)) and any(heading(text_of(a)) == (1, "The Beautiful Data") for a in md_args(o)))
    tv = next(j for j in range(title + 1, len(layout)) if layout[j]["type"] != "skip")
    if (tv_out := output_of(all_cells[tv])) and ast.unparse(tv_out).startswith("mo.Html(f'''<iframe"):
        layout[tv]["type"] = "fragment"
    return layout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="start `marimo run` on the slides copy afterwards")
    args = ap.parse_args()

    source = NOTEBOOK.read_text()
    tree = ast.parse(source)
    copy, notes = slide_copy(source, tree)
    for old, new in SLIDE_CODE:
        if copy.count(old) != 1:
            sys.exit(f"SLIDE_CODE edit matches {copy.count(old)} times, not once: {old.strip()[:70]!r}")
        copy = copy.replace(old, new)
    for snippet in SLIDE_DROP_LINES:
        lines = copy.split("\n")
        hits = [i for i, line in enumerate(lines) if snippet in line]
        if len(hits) != 1:
            sys.exit(f"SLIDE_DROP_LINES snippet matches {len(hits)} lines, not one: {snippet[:70]!r}")
        copy = "\n".join(lines[:hits[0]] + lines[hits[0] + 1:])
    layout = build_layout(tree, notes)
    try:  # cross-check the cell count with marimo's own loader (a private API, so skipped if it moves)
        from marimo._ast.load import load_app

        app = load_app(str(NOTEBOOK))
        n = len(list(app._cell_manager.cell_ids())) if app else len(layout)
        if n != len(layout):
            sys.exit(f"marimo counts {n} cells but this script found {len(layout)}: the layout would be misaligned")
    except (ImportError, AttributeError):
        pass
    outputs = [o for c in cells(tree) if (o := output_of(c))]
    unused = set(SLIDE_TEXT) - {key_of(text_of(a)) for o in outputs for a in md_args(o)}
    if unused:
        sys.exit(f"SLIDE_TEXT entries that match no heading in the notebook: {sorted(unused)}")

    (ROOT / LAYOUT).parent.mkdir(exist_ok=True)
    (ROOT / LAYOUT).write_text(json.dumps({"type": "slides", "data": {"cells": layout, "deck": DECK}}, indent=2, ensure_ascii=False) + "\n")

    app_line = re.search(r"^app = marimo\.App\((.*)\)$", copy, flags=re.MULTILINE)
    if not app_line or "layout_file" in app_line.group(1):
        sys.exit("could not find a plain `app = marimo.App(...)` line to point at the layout")
    head = re.search(r'html_head_file="([^"]+)"', app_line.group(1))
    (ROOT / HEAD).write_text(((ROOT / head.group(1)).read_text() if head else "") + SLIDES_HEAD)
    app_args = re.sub(r',?\s*html_head_file="[^"]+"', "", app_line.group(1))
    copy = copy.replace(app_line.group(0), f'app = marimo.App({app_args}, html_head_file="{HEAD.as_posix()}", layout_file="{LAYOUT.as_posix()}")', 1)
    ast.parse(copy)  # the edited copy must still be valid Python
    SLIDES_NOTEBOOK.write_text("# Generated by scripts/make_slides.py from the_beautiful_data.py: edit those, not this.\n" + copy)

    kinds = {k: sum(e["type"] == k for e in layout) for k in ("slide", "sub-slide", "fragment", "skip")}
    print(f"wrote {LAYOUT} ({len(layout)} cells: {kinds['slide']} slides, {kinds['sub-slide']} sub-slides, {kinds['fragment']} "
          f"fragments, {kinds['skip']} skipped), {HEAD.name} and {SLIDES_NOTEBOOK.name} ({len(notes)} text cells shortened)")
    print(f"present with: uv run marimo run {SLIDES_NOTEBOOK.name}")
    if args.run:
        subprocess.run(["marimo", "run", str(SLIDES_NOTEBOOK)], cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
