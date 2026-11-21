# The Beautiful Data

<p align="center"><img src="assets/tv.gif" alt="A retro TV switching on and playing football's greatest moments" width="480"></p>

An interactive [marimo](https://marimo.io) notebook to explore [the beautiful game](https://en.wikipedia.org/wiki/The_beautiful_game) through data: what gets recorded (results, events, 360 freeze frames, tracking, bodies, markets, images, sound, video, text), the models people build on it (xG, xT, VAEP, pitch control, ghost teams, passing networks, forecasting, game theory) and where the field is heading.

Every model is built from scratch on public data.

<p align="center"><img src="assets/tv.gif" alt="A retro TV switching on and playing football's greatest moments" width="480"></p>

## Try it online

- **Lite**: [jgalego.github.io/football](https://jgalego.github.io/football), runs in your browser
- **Full**: [Hugging Face Space](https://huggingface.co/spaces/jgalego/football), with the StatsBomb and Metrica sections live

## Run it locally

```bash
uv sync                                    # creates .venv from pyproject.toml
uv run marimo run the_beautiful_data.py    # read-only app (add --watch to reload on edits)
uv run marimo edit the_beautiful_data.py   # edit it

# or, with no setup at all: uv resolves the dependencies declared at the top of the notebook
uvx marimo run --sandbox the_beautiful_data.py
```

The notebook reads prepared files from `public/`. Rebuild them from source:

```bash
uv run python scripts/download_data.py     # ~4 GB: StatsBomb, Metrica, results, Transfermarkt, Wikimedia media -> data/raw/
uv run python scripts/prepare_data.py      # compact files -> public/ (name stages to run some: statsbomb | metrica | international | transfermarkt | media | stats)
```

Player portraits come from Wikimedia Commons: `uv run python scripts/fetch_player_photos.py` matches players to Wikidata footballers, keeps only freely licensed photos with a detectable face, crops them to 112 px in `public/players/` and writes the credits to `public/player_photos.json` (shown in the notebook). It is rate-limited by Wikimedia and takes about an hour; it is resumable and the results are committed, so you only need it to refresh the photos.

The `media` stage needs `ffmpeg` on your PATH. If Wikimedia answers HTTP 429, set `FOOTBALL_USER_AGENT` to something that identifies you. The first run trains a few models (about two minutes) and caches them in `.cache/`; later runs start in well under a minute.

## Deploy

**Lite edition (GitHub Pages).** `.github/workflows/pages.yml` fetches only the open data, builds with `scripts/build_pages.py` and deploys. In the repository settings choose *Pages → Source: GitHub Actions*. `build_pages.py` stages an allowlist of open files and refuses to publish anything that looks like row-level StatsBomb or Metrica data, so a local build is safe too: `uv run python scripts/build_pages.py --out _site`.

**Full edition (Hugging Face Space).** Create a Space with the **Docker** SDK, add a repository secret `HF_TOKEN` (write access) and let `.github/workflows/space.yml` sync the repository to it. The `Dockerfile` downloads the data inside the image build and warms the model cache. Edit `HF_SPACE` in the workflow and `SPACE_URL` in the notebook if your Space is not `jgalego/football`.

**Snapshots.** The lite edition's pictures are made from a full render:

```bash
uv run marimo export html --no-include-code the_beautiful_data.py -o /tmp/full.html
uv run playwright install chromium         # once
uv run python scripts/make_snapshots.py --html /tmp/full.html
```

**StatsBomb logo.** Their agreement asks for the logo on any publication. Download it from their [media pack](https://statsbomb.com/media-pack/) and save it as `public/statsbomb-logo.png`; the notebook shows it automatically.

## Data and licences

| Source | Used for | Terms |
|---|---|---|
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | events, line-ups, 360 frames | [StatsBomb Public Data User Agreement](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf): credit with the logo, do not provide the data to third parties, no commercial use of the data or analysis |
| [Metrica Sports sample data](https://github.com/metrica-sports/sample-data) | tracking | no licence file; the README asks for acknowledgement. Treated like StatsBomb |
| [International results](https://github.com/martj42/international_results) | Elo, Poisson, shoot-outs | CC0 |
| [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) | market value by age | CC0 |
| Wikimedia Commons | photo, audio and video demos, player portraits | per file, see `public/media.json` and `public/player_photos.json` |

The player portraits in `public/players/` are face crops of Wikimedia Commons photos, mostly CC BY-SA and CC BY. They keep their original licences (the cropped copies are adaptations under the same terms) and every photo is credited in the notebook and in `public/player_photos.json`.

`data/raw/` and most of `public/` are git-ignored on purpose. Only small research tables, aggregate counts, CC0 data and the static snapshots are tracked.

## License

The code is released under the [MIT License](LICENSE). The data it uses is not covered by it and keeps its own terms, see [Data and licences](#data-and-licences).
