ARG BASE_REGISTRY=docker.io
FROM ${BASE_REGISTRY}/library/python:3.14-slim

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    jq \
    ca-certificates \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI into a world-readable prefix -- NOT root's home.
# The image runs as USER executor, for whom /root is unreadable, so an install
# under /root ships a binary nothing can execute. The installer puts the
# launcher in $HOME/.local/bin and its payload in $HOME/.claude, so point HOME
# at /opt/claude and make the tree readable by every user.
# No `|| true`, no `|| echo` -- a failed install or symlink must fail the build.
RUN mkdir -p /opt/claude && \
    export HOME=/opt/claude && \
    curl -fsSL https://claude.ai/install.sh | bash && \
    ln -s /opt/claude/.local/bin/claude /usr/local/bin/claude && \
    chmod -R a+rX /opt/claude

# Build-time assertion: if `claude` is not on PATH and runnable, fail HERE,
# loudly, instead of shipping an image whose entrypoint dies on
# "claude: command not found".
RUN claude --version

# Install MinIO client (architecture-aware)
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        curl -o /usr/local/bin/mc https://dl.min.io/client/mc/release/linux-amd64/mc; \
    elif [ "$ARCH" = "aarch64" ]; then \
        curl -o /usr/local/bin/mc https://dl.min.io/client/mc/release/linux-arm64/mc; \
    fi && \
    chmod +x /usr/local/bin/mc

# Install Python packages for Claude API and MinIO
RUN pip install --no-cache-dir \
    minio \
    anthropic \
    requests

# Create workspace and artifacts directories
RUN mkdir -p /workspace /artifacts && \
    chmod 755 /workspace /artifacts

# Create non-root user for execution
RUN useradd -m -u 1000 -s /bin/bash executor && \
    chown -R executor:executor /workspace /artifacts

# Copy entrypoint script
COPY docker/scripts/k8s_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy artifact collection script
COPY docker/scripts/collect_artifacts.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/collect_artifacts.sh

USER executor
WORKDIR /workspace

# Same assertion as the non-root user that actually runs the entrypoint --
# root being able to run it proves nothing about `executor`.
RUN claude --version

ENTRYPOINT ["/entrypoint.sh"]
