# CLI Proxy Setup for Claude Code & Codex

## Overview

This proxy server runs on your host machine and forwards requests from the Docker container to your local Claude Code and Codex CLIs, which have subscription authentication configured.

## Architecture

```
Container (agents) → host.docker.internal:9000 → Proxy Server → Local CLIs → Anthropic/OpenAI
```

## Prerequisites

- Claude Code CLI authenticated locally (subscription/OAuth)
- Codex CLI authenticated locally (ChatGPT subscription)
- Python 3.8+ with FastAPI and Uvicorn

## Installation

### 1. Install Proxy Dependencies

```bash
pip3 install fastapi uvicorn pydantic
```

### 2. Test Proxy Server

```bash
# Start the proxy server
python3 scripts/cli_proxy_server.py --port 9000

# Test Claude endpoint
curl -X POST http://localhost:9000/claude \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!"}'

# Test Codex endpoint
curl -X POST http://localhost:9000/codex \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 2+2?"}'

# Check health
curl http://localhost:9000/health
```

## Running the Proxy

### Option 1: Manual Start

```bash
# Start in foreground
python3 scripts/cli_proxy_server.py --port 9000

# Start in background
nohup python3 scripts/cli_proxy_server.py --port 9000 > /tmp/cli_proxy.log 2>&1 &
```

### Option 2: macOS launchd Service (Auto-start)

Create `~/Library/LaunchAgents/com.worker-management.cli-proxy.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.worker-management.cli-proxy</string>

    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/python3</string>
        <string>/Users/mini2/Movies/worker_management/scripts/cli_proxy_server.py</string>
        <string>--port</string>
        <string>9000</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/mini2/Movies/worker_management</string>

    <key>StandardOutPath</key>
    <string>/tmp/cli-proxy.out.log</string>

    <key>StandardErrorPath</key>
    <string>/tmp/cli-proxy.err.log</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

Load the service:

```bash
launchctl load ~/Library/LaunchAgents/com.worker-management.cli-proxy.plist
launchctl start com.worker-management.cli-proxy
```

Check status:

```bash
launchctl list | grep cli-proxy
tail -f /tmp/cli-proxy.out.log
```

### Option 3: systemd Service (Linux)

Create `/etc/systemd/system/cli-proxy.service`:

```ini
[Unit]
Description=CLI Proxy Server for Claude Code and Codex
After=network.target

[Service]
Type=simple
User=mini2
WorkingDirectory=/Users/mini2/Movies/worker_management
ExecStart=/usr/bin/python3 /Users/mini2/Movies/worker_management/scripts/cli_proxy_server.py --port 9000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable cli-proxy
sudo systemctl start cli-proxy
sudo systemctl status cli-proxy
```

## API Endpoints

### POST /claude
Execute Claude Code CLI

**Request:**
```json
{
  "prompt": "Your prompt here",
  "model": "sonnet",  // optional
  "system_prompt": "You are...",  // optional
  "timeout": 300  // seconds
}
```

**Response:**
```json
{
  "success": true,
  "output": "Claude's response...",
  "error": null,
  "exit_code": 0,
  "execution_time": 4.52
}
```

### POST /codex
Execute Codex CLI

**Request:**
```json
{
  "prompt": "Your prompt here",
  "model": "gpt-5-codex",  // optional
  "timeout": 300  // seconds
}
```

**Response:**
```json
{
  "success": true,
  "output": "Codex response...",
  "error": null,
  "exit_code": 0,
  "execution_time": 2.13
}
```

### GET /health
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "claude_available": true,
  "claude_path": "/opt/homebrew/bin/claude",
  "codex_available": true,
  "codex_path": "/opt/homebrew/bin/codex"
}
```

## Docker Configuration

The agents in the container will connect to the proxy via `host.docker.internal:9000`.

Environment variables in `.env`:
```bash
CLI_PROXY_URL=http://host.docker.internal:9000
CLI_PROXY_ENABLED=true
```

## Troubleshooting

### Proxy not accessible from container

```bash
# Test from container
docker exec worker_management-worker-management-api-1 curl http://host.docker.internal:9000/health
```

### Check proxy logs

```bash
# If running as launchd service
tail -f /tmp/cli-proxy.out.log
tail -f /tmp/cli-proxy.err.log

# If running manually
# Check terminal output
```

### Port already in use

```bash
# Find what's using the port
lsof -i :9000

# Kill the process
pkill -f cli_proxy_server
```

### CLIs not found

Make sure both CLIs are in your PATH:
```bash
which claude  # Should return /opt/homebrew/bin/claude
which codex   # Should return /opt/homebrew/bin/codex
```

## Security Notes

- The proxy runs on localhost only by default
- No authentication is required (local machine only)
- Container accesses via `host.docker.internal` (Docker's special DNS)
- For production, consider adding API key authentication

## Performance

- Claude responses: ~3-5 seconds typical
- Codex responses: ~1-3 seconds typical
- Timeout: 300 seconds default (configurable)

## Maintenance

### Restart proxy

```bash
# launchd
launchctl stop com.worker-management.cli-proxy
launchctl start com.worker-management.cli-proxy

# systemd
sudo systemctl restart cli-proxy

# manual
pkill -f cli_proxy_server
python3 scripts/cli_proxy_server.py --port 9000 &
```

### Update proxy code

```bash
# Stop service
launchctl stop com.worker-management.cli-proxy

# Update code
git pull  # or edit scripts/cli_proxy_server.py

# Restart service
launchctl start com.worker-management.cli-proxy
```
