# Full interactive edition for a Hugging Face Space (Docker SDK).
#
# The StatsBomb and Metrica data are downloaded INSIDE the image build and never enter the git repository or the Space's
# files. Visitors only receive what the notebook renders; the data stays on the server.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Hugging Face runs containers as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1 PATH=/home/user/app/.venv/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=user . .

# 1. fetch the raw data, 2. build the compact files, 3. delete the raw files. One RUN so the ~4 GB never lands in a layer.
# (Set FOOTBALL_USER_AGENT as a Space variable if Wikimedia throttles the build.)
RUN python scripts/download_data.py && python scripts/prepare_data.py && rm -rf data/raw

# Warm the cache of the heavy models so the first visitor does not wait for them.
RUN marimo export html --no-include-code the_beautiful_data.py -o /tmp/warmup.html && rm /tmp/warmup.html

EXPOSE 7860
CMD ["marimo", "run", "the_beautiful_data.py", "--host", "0.0.0.0", "--port", "7860", "--headless"]
