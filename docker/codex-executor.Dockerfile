ARG BASE_REGISTRY=docker.io
FROM ${BASE_REGISTRY}/library/node:24-alpine

# Install Codex CLI (use latest for CLI tool)
# hadolint ignore=DL3016
RUN npm install -g @openai/codex

# Create .codex directory with correct ownership for node user (uid 1000)
RUN mkdir -p /home/node/.codex && chown -R node:node /home/node/.codex

USER node
WORKDIR /workspace

# AGENT_PROMPT passed via environment
ENTRYPOINT ["sh", "-c", "codex exec --dangerously-bypass-approvals-and-sandbox \"$AGENT_PROMPT\""]
