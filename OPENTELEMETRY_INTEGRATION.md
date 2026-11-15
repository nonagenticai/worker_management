# OpenTelemetry Integration - Configuration Changes

## Overview
OpenTelemetry has been enabled in the worker_management service to send traces to the local Grafana Agent for observability and monitoring.

## Configuration Changes Made

### 1. Environment Variables (.env)
**File**: `/Users/mini2/Movies/worker_management/.env`

**Changes (Lines 16-28)**:
- **OTLP_TRACES_ENDPOINT=http://grafana-agent:4317** - Points to Grafana Agent's OTLP gRPC endpoint
- **OTEL_EXPORTER_OTLP_INSECURE=true** - Enables insecure connections for local development
- **OTEL_SERVICE_NAME=worker-management-service** - Service identifier in traces
- **OTEL_SDK_DISABLED=false** - Changed from `true` to enable OpenTelemetry SDK
- **OTEL_RESOURCE_ATTRIBUTES** - Updated to:
  - `service.name=worker-management-service` (consistent naming)
  - `service.namespace=worker_management`
  - `deployment.environment=local` (changed from production)

### 2. Docker Compose Configuration
**File**: `/Users/mini2/Movies/worker_management/docker-compose.yaml`

**Changes (Lines 64-69)** - Added environment variables:
```yaml
# OpenTelemetry configuration
- OTLP_TRACES_ENDPOINT=${OTLP_TRACES_ENDPOINT:-http://grafana-agent:4317}
- OTEL_EXPORTER_OTLP_INSECURE=${OTEL_EXPORTER_OTLP_INSECURE:-true}
- OTEL_SERVICE_NAME=${OTEL_SERVICE_NAME:-worker-management-service}
- OTEL_SDK_DISABLED=${OTEL_SDK_DISABLED:-false}
- OTEL_RESOURCE_ATTRIBUTES=${OTEL_RESOURCE_ATTRIBUTES:-service.name=worker-management-service,service.namespace=worker_management,deployment.environment=local}
```

All variables include default values using the `${VAR:-default}` syntax, so they work even if .env is missing.

### 3. Network Configuration
**Verified (Lines 76-77, 96-98)**: The service is already connected to `workflow-system-network` (external network), which allows communication with the Grafana Agent.

## Instrumentation Details

### Code Already Instrumented
**File**: `/Users/mini2/Movies/worker_management/src/api/worker_management_api.py`

The application is already instrumented with OpenTelemetry:
- **Lines 62-115**: Conditional OpenTelemetry imports with graceful fallback
- **Lines 118-155**: Tracer provider setup with OTLP and Console exporters
- **Lines 311-318**: FastAPI instrumentation via `FastAPIInstrumentor`

Key features:
- Automatic HTTP request/response tracing
- Custom spans can be created using the `tracer` object
- Console exporter for debugging when `OTEL_DEBUG=true`
- Graceful degradation if OpenTelemetry packages are not installed

## Infrastructure Setup

### Grafana Agent Configuration
**File**: `/Users/mini2/Movies/grafana/agent-config.yaml`

The Grafana Agent is configured to:
- Receive OTLP traces on ports 4317 (gRPC) and 4318 (HTTP)
- Forward traces to Tempo at `tempo:4317`
- Run in insecure mode for local development

### Observability Stack
**File**: `/Users/mini2/Movies/grafana/docker-compose.yml`

Components available:
- **Grafana** (port 3439) - Visualization and dashboards
- **Grafana Agent** (ports 4317/4318) - OTLP receiver and data pipeline
- **Tempo** (port 3200) - Distributed tracing backend
- **Prometheus** (port 9090) - Metrics storage
- **Loki** (port 3100) - Log aggregation

## How to Verify the Integration

### 1. Start the Observability Stack
```bash
cd /Users/mini2/Movies/grafana
docker-compose up -d
```

Verify services are running:
```bash
docker-compose ps
```

Expected output should show all services (grafana, grafana-agent, tempo, prometheus, loki) as "Up".

### 2. Start the Worker Management Service
```bash
cd /Users/mini2/Movies/worker_management
docker-compose up -d
```

### 3. Check Service Logs
```bash
# Check worker-management-api logs for OpenTelemetry initialization
docker logs worker-management-api 2>&1 | grep -i "opentelemetry\|otlp"
```

Expected log messages:
- "OpenTelemetry FastAPI instrumentation available"
- "OpenTelemetry OTLP exporter configured for http://grafana-agent:4317"
- "OpenTelemetry tracing configured successfully"
- "FastAPI app instrumented with OpenTelemetry"

### 4. Generate Test Traces
```bash
# Make some API requests to generate traces
curl http://localhost:8887/health
curl http://localhost:8887/
curl http://localhost:8887/docs
```

### 5. View Traces in Grafana

1. Open Grafana: http://localhost:3439
   - Default credentials: admin/admin (if not changed)

2. Navigate to "Explore" → Select "Tempo" data source

3. Search for traces:
   - Service: `worker-management-service`
   - Look for recent traces in the timeline

4. Expected trace information:
   - Span names: GET /health, GET /, etc.
   - Service name: worker-management-service
   - Resource attributes: service.namespace=worker_management, deployment.environment=local

### 6. Check Grafana Agent Logs
```bash
docker logs grafana-agent 2>&1 | grep -i "trace\|otlp"
```

Should show successful OTLP receiver initialization and trace forwarding.

### 7. Query Tempo Directly
```bash
# Check Tempo health
curl http://localhost:3200/ready

# List recent traces (requires trace ID)
curl "http://localhost:3200/api/search?tags=service.name=worker-management-service"
```

## Troubleshooting

### Issue: No traces appearing in Grafana

**Check 1**: Verify network connectivity
```bash
docker exec worker-management-api ping -c 3 grafana-agent
```

**Check 2**: Verify Grafana Agent is receiving traces
```bash
docker logs grafana-agent | tail -50
```

**Check 3**: Enable debug logging
Add to `.env`:
```bash
OTEL_DEBUG=true
LOG_LEVEL=DEBUG
```
Then restart: `docker-compose restart worker-management-api`

**Check 4**: Verify environment variables in container
```bash
docker exec worker-management-api env | grep OTEL
```

### Issue: Connection refused to grafana-agent:4317

**Possible causes**:
1. Grafana Agent not running: `docker-compose -f /Users/mini2/Movies/grafana/docker-compose.yml ps`
2. Network issue: Verify both services are on `workflow-system-network`
3. Firewall/port blocking: Check Docker network rules

**Fix**: Restart both stacks
```bash
cd /Users/mini2/Movies/grafana && docker-compose restart grafana-agent
cd /Users/mini2/Movies/worker_management && docker-compose restart
```

### Issue: OTEL_SDK_DISABLED is still true

**Check**:
```bash
docker exec worker-management-api env | grep OTEL_SDK_DISABLED
```

**Fix**: The .env file is mounted as read-only. After changing .env, restart:
```bash
docker-compose down && docker-compose up -d
```

## Performance Considerations

### Trace Sampling
By default, all traces are sampled (100% sampling rate). For production:

Add to `.env`:
```bash
OTEL_TRACES_SAMPLER=traceidratio
OTEL_TRACES_SAMPLER_ARG=0.1  # Sample 10% of traces
```

### Resource Usage
- OpenTelemetry adds minimal overhead (< 5% CPU, < 50MB RAM)
- BatchSpanProcessor batches spans for efficiency
- Grafana Agent buffers traces before sending to Tempo

### Disabling Tracing
To temporarily disable without code changes:
```bash
OTEL_SDK_DISABLED=true
```

## Next Steps

### 1. Custom Spans
Add custom instrumentation in your code:

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("custom_operation"):
    # Your code here
    pass
```

### 2. Create Grafana Dashboards
- Navigate to Grafana → Dashboards → New Dashboard
- Add panels with Tempo queries for service metrics
- Track request latency, error rates, throughput

### 3. Set Up Alerts
Configure alerts in Grafana for:
- High error rates
- Slow request times
- Service availability

### 4. Database Query Tracing
Install additional instrumentation:
```bash
uv add opentelemetry-instrumentation-sqlalchemy
```

### 5. External Service Tracing
Trace calls to external APIs:
```bash
uv add opentelemetry-instrumentation-requests
uv add opentelemetry-instrumentation-httpx
```

## References

- [OpenTelemetry Python Docs](https://opentelemetry.io/docs/instrumentation/python/)
- [Grafana Tempo Documentation](https://grafana.com/docs/tempo/latest/)
- [Grafana Agent Configuration](https://grafana.com/docs/agent/latest/)
- [FastAPI OpenTelemetry](https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/fastapi/fastapi.html)

## Configuration Summary

| Variable | Value | Purpose |
|----------|-------|---------|
| OTLP_TRACES_ENDPOINT | http://grafana-agent:4317 | Grafana Agent OTLP gRPC endpoint |
| OTEL_EXPORTER_OTLP_INSECURE | true | Allow insecure connections (local dev) |
| OTEL_SERVICE_NAME | worker-management-service | Service identifier in traces |
| OTEL_SDK_DISABLED | false | Enable OpenTelemetry SDK |
| OTEL_RESOURCE_ATTRIBUTES | service.name=worker-management-service,<br>service.namespace=worker_management,<br>deployment.environment=local | Service metadata |

---

**Status**: OpenTelemetry integration is now ENABLED and ready to use.

**Created**: 2025-10-27
**Last Updated**: 2025-10-27
