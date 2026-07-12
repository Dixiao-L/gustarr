# Gustarr — CPU image. Torch comes from the PyTorch CPU wheel index so the
# image stays ~2 GB and runs on any host. GPU users: prefer the NixOS module
# (flake ships a CUDA `ml` variant), or build a variant of this image
# installing the cu13 wheels instead of the cpu ones.

# ── build: wheels into a self-contained venv ─────────────────────────
FROM python:3.14-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU torch first (own index), so the `.[ml]` resolve below sees torch>=2.4
# already satisfied and never pulls the multi-GB CUDA build from PyPI.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple "torch>=2.4"

COPY pyproject.toml README.md LICENSE /src/
COPY src /src/src
RUN pip install "/src[ml]"

# ── runtime ──────────────────────────────────────────────────────────
FROM python:3.14-slim

LABEL org.opencontainers.image.title="Gustarr" \
      org.opencontainers.image.description="Learns your media taste, one profile per person, and drives Sonarr/Radarr/Lidarr" \
      org.opencontainers.image.source="https://github.com/Dixiao-L/gustarr" \
      org.opencontainers.image.licenses="MIT"

RUN useradd --system --uid 1000 --user-group --home-dir /var/lib/gustarr \
        --create-home gustarr \
    && mkdir -p /etc/gustarr \
    && chmod 750 /var/lib/gustarr

COPY --from=build /opt/venv /opt/venv

# HF_HOME keeps the sentence-transformers model cache inside the state volume
ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/var/lib/gustarr/hf

USER gustarr
WORKDIR /var/lib/gustarr

# store + model cache; mount a volume here (it IS your taste model)
# config: mount your gustarr.toml at /etc/gustarr/gustarr.toml (a default
# search path) — remember [core] data_dir = "/var/lib/gustarr" and
# [model] device = "cpu" in container deployments.

EXPOSE 8790

ENTRYPOINT ["gustarr"]
# default: serve the approval UI — this process never runs the pipeline.
# Scheduling is a dedicated `gustarr schedule` process: run a second
# container from this same image with command "schedule" and
# [scheduler] nightly = "HH:MM" in the TOML (see the compose example).
# One-shots still work for manual runs or cron-style scheduling, e.g.
#   docker compose run --rm gustarr run nightly
CMD ["web"]
