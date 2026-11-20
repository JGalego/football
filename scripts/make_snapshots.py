"""Create the static snapshots used by the lite edition (GitHub Pages).

Sections built on StatsBomb / Metrica data cannot be republished as live notebooks, but a picture of the analysis can:
the terms allow sharing analysis (with the StatsBomb logo), not the data. This script renders those sections from a
FULL static export and saves one JPEG per section into public/snapshots/.

    uv run marimo export html --no-include-code the_beautiful_data.py -o /tmp/full.html
    uv run python scripts/make_snapshots.py --html /tmp/full.html

Needs the `dev` dependency group (Playwright) and a Chromium installed for it: `uv run playwright install chromium`.
"""

from __future__ import annotations

import argparse
import asyncio
import io
from pathlib import Path

from PIL import Image
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]

# snapshot name -> heading id of the section whose intro cell stays live; the picture starts right below that cell
SECTIONS = {
    "s02": "2-anatomy-of-an-event", "s03": "3-a-match-as-a-story", "s04": "4-where-things-happen", "s05": "5-tracking-25-snapshots-a-second",
    "s06": "6-the-engine-room-physical-data", "s07": "7-from-pixels-to-coordinates", "s10": "10-expected-goals-xg", "s11": "11-expected-threat-xt",
    "s12": "12-valuing-every-action-vaep", "s13": "13-pitch-control", "s14": "14-ghost-team", "s15": "15-passing-networks",
    "s16": "16-player-fingerprints", "s18": "18-penalties-and-game-theory", "s19": "19-from-data-to-words",
}
# the picture ends at the first following cell that holds a section heading, a "Go further" line, or this text
STOP_TEXT = {"s18": "Scoring probabilities"}

HIDE_MENUS = ".vega-embed details, .vega-embed summary, .vega-actions { display: none !important }"
INJECT_INTO_SHADOW_ROOTS = """(css) => {
  const visit = (root) => {
    for (const el of root.querySelectorAll('*')) {
      if (!el.shadowRoot) continue;
      if (!el.shadowRoot.querySelector('style[data-snap]')) {
        const style = document.createElement('style');
        style.dataset.snap = '1';
        style.textContent = css;
        el.shadowRoot.appendChild(style);
      }
      visit(el.shadowRoot);
    }
  };
  visit(document);
}"""

FIND_REGION = """([anchor, stopText]) => {
  const scroller = (() => { const els = [...document.querySelectorAll('*')].filter(e => e.scrollHeight > e.clientHeight + 50 && ['auto','scroll'].includes(getComputedStyle(e).overflowY));
    els.sort((a, b) => b.scrollHeight - a.scrollHeight); return els[0]; })();
  scroller.setAttribute('data-scroller', '1');
  const cells = [...document.querySelectorAll('.marimo-cell')];
  const intro = document.getElementById(anchor).closest('.marimo-cell');
  let i = cells.indexOf(intro) + 1, end = null;
  for (; i < cells.length; i++) {
    const c = cells[i], t = (c.innerText || '').trim();
    if (c.querySelector('h1,h2')) { end = c; break; }  // a section heading (sub-headings, h3 and below, stay inside)
    if (t.startsWith('↳ Go further')) { end = c; break; }
    if (stopText && t.includes(stopText)) { end = c; break; }
  }
  const sTop = scroller.getBoundingClientRect().top;
  const abs = el => el.getBoundingClientRect().top - sTop + scroller.scrollTop;
  const left = intro.getBoundingClientRect().left, width = intro.getBoundingClientRect().width;
  return { top: abs(intro) + intro.getBoundingClientRect().height, bottom: end ? abs(end) : null, left, width };
}"""


async def main(html: Path, out: Path, scale: float, only: set[str] | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await (await browser.new_context(viewport={"width": 1400, "height": 1500}, device_scale_factor=scale)).new_page()
        await page.goto(html.resolve().as_uri(), wait_until="networkidle", timeout=240000)
        await page.wait_for_timeout(10000)
        # live-page furniture that has no place in a picture: the "..." menu on every chart
        await page.add_style_tag(content=HIDE_MENUS)

        async def hide_menus() -> None:
            """Charts live in shadow roots that page-level CSS cannot reach: put the rule inside every one of them."""
            await page.evaluate(INJECT_INTO_SHADOW_ROOTS, HIDE_MENUS)

        async def shoot(name: str, find) -> None:
            """The page lays out lazily, so measure, scroll there, and measure again until the position is stable."""
            r = await find()
            for _ in range(5):
                height = max(40, r["bottom"] - r["top"])
                await page.set_viewport_size({"width": 1400, "height": int(min(height + 160, 12000))})
                await page.evaluate("t => { document.querySelector('[data-scroller]').scrollTop = t }", r["top"] - 10)
                await page.wait_for_timeout(1500)
                again = await find()
                if abs(again["top"] - r["top"]) < 2 and abs(again["bottom"] - r["bottom"]) < 2:
                    r = again
                    break
                r = again
            height = max(40, r["bottom"] - r["top"])
            sTop = await page.evaluate("document.querySelector('[data-scroller]').getBoundingClientRect().top")
            scroll = await page.evaluate("document.querySelector('[data-scroller]').scrollTop")
            await hide_menus()
            png = await page.screenshot(clip={"x": r["left"], "y": sTop + (r["top"] - scroll), "width": r["width"], "height": height})
            img = Image.open(io.BytesIO(png)).convert("RGB")
            if img.width > 1100:
                img = img.resize((1100, round(img.height * 1100 / img.width)), Image.LANCZOS)
            img.save(out / f"{name}.jpg", quality=84, optimize=True)
            print(f"  {name}.jpg  {img.width}x{img.height}  {(out / f'{name}.jpg').stat().st_size / 1e3:.0f} kB")

        async def find_hero():
            return await page.evaluate("""() => {
              const scroller = [...document.querySelectorAll('*')].filter(e => e.scrollHeight > e.clientHeight + 50 && ['auto','scroll'].includes(getComputedStyle(e).overflowY)).sort((a, b) => b.scrollHeight - a.scrollHeight)[0];
              scroller.setAttribute('data-scroller', '1');
              const el = document.querySelector('.vega-embed canvas, .vega-embed svg');  // crop to the chart itself
              const r = el.getBoundingClientRect(), s = scroller.getBoundingClientRect();
              return { top: r.top - s.top + scroller.scrollTop, bottom: r.bottom - s.top + scroller.scrollTop, left: r.left, width: r.width }; }""")

        if not only or "hero" in only:
            await shoot("hero", find_hero)
        for name, anchor in SECTIONS.items():
            if only and name not in only:
                continue
            probe = await page.evaluate(FIND_REGION, [anchor, STOP_TEXT.get(name)])
            if probe["bottom"] is None:
                print(f"  ! {name}: no end cell found, skipped")
                continue
            await shoot(name, lambda anchor=anchor, name=name: page.evaluate(FIND_REGION, [anchor, STOP_TEXT.get(name)]))
        await browser.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", required=True, type=Path, help="static export of the FULL notebook")
    ap.add_argument("--out", type=Path, default=ROOT / "public" / "snapshots")
    ap.add_argument("--scale", type=float, default=1.25)
    ap.add_argument("--only", nargs="*", help="regenerate just these snapshots, e.g. hero s10")
    args = ap.parse_args()
    asyncio.run(main(args.html, args.out, args.scale, set(args.only) if args.only else None))
