# vllm-rdna-dash as a container. It manages the vLLM container through the host's
# Docker (or Podman) socket, so it only needs the docker CLI and compose plugin.
FROM docker:27.3.1-cli AS dockercli

FROM python:3.12-slim
ARG COMPOSE_VERSION=v2.29.7
ARG TARGETARCH
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates pciutils \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /usr/local/lib/docker/cli-plugins \
 && arch="$(case "${TARGETARCH:-amd64}" in arm64) echo aarch64 ;; *) echo x86_64 ;; esac)" \
 && curl -fsSL "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-${arch}" \
      -o /usr/local/lib/docker/cli-plugins/docker-compose \
 && chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker

WORKDIR /app
COPY rdna_dash ./rdna_dash
COPY recipes ./recipes
COPY tools ./tools
ENV RDNA_DASH_IN_CONTAINER=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# Where a plain `docker run` keeps its data; compose.yml overrides these when DATA is set.
ENV RDNA_DASH_ROOT=/srv/vllm-rdna XDG_CONFIG_HOME=/srv/vllm-rdna/dash/config XDG_STATE_HOME=/srv/vllm-rdna/dash/state
EXPOSE 7860
ENTRYPOINT ["python", "-m", "rdna_dash"]
CMD ["--host", "0.0.0.0", "--port", "7860"]
