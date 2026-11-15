#!/bin/bash

# Simple setup for GitHub MCP server via mcp-proxy

set -e

PORT="8080"
GITHUB_TOKEN="${GITHUB_PERSONAL_ACCESS_TOKEN}"

if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GITHUB_PERSONAL_ACCESS_TOKEN required"
    exit 1
fi

echo "Installing mcp-proxy..."
pip install mcp-proxy

echo "Starting GitHub MCP proxy on port $PORT..."
echo "This will run the GitHub MCP server and expose it via HTTP/SSE"

# Run mcp-proxy with GitHub MCP server
mcp-proxy --port=$PORT --host=0.0.0.0 \
  docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_TOKEN ghcr.io/github/github-mcp-server
