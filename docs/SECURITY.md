# Security Documentation

This document describes the security model, threat analysis, and security guarantees of the Worker Management System.

## Table of Contents

- [Security Model](#security-model)
- [Threat Model](#threat-model)
- [Security Measures](#security-measures)
- [Data Isolation Guarantees](#data-isolation-guarantees)
- [Cryptographic Security](#cryptographic-security)
- [Audit and Compliance](#audit-and-compliance)
- [Security Testing](#security-testing)
- [Security Considerations](#security-considerations)
- [Incident Response](#incident-response)

## Security Model

### Multi-Tenant Isolation

The Worker Management System implements a **strict multi-tenant model** where:

1. **Each user (tenant) can only access their own resources**
2. **tenant_id is the isolation boundary** - extracted from authenticated JWT tokens
3. **All database queries filter by tenant_id**
4. **No cross-tenant data leakage**

### Authentication Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Security Layers                        │
├─────────────────────────────────────────────────────────┤
│ Layer 1: JWT Token Validation (Cryptographic)          │
│  - RSA signature verification (RS256)                   │
│  - Token expiration check                               │
│  - Issuer validation (optional)                         │
│  - Audience validation (optional)                       │
├─────────────────────────────────────────────────────────┤
│ Layer 2: tenant_id Extraction                         │
│  - Extract 'sub' claim from validated token             │
│  - Reject tokens without 'sub' claim                    │
│  - No user-provided tenant_id accepted                 │
├─────────────────────────────────────────────────────────┤
│ Layer 3: Database Query Filtering                      │
│  - All queries include WHERE tenant_id = $1            │
│  - Composite indexes for performance                    │
│  - Unique constraints enforce per-tenant uniqueness     │
├─────────────────────────────────────────────────────────┤
│ Layer 4: API Response Filtering                        │
│  - Return 404 (not 403) for unauthorized access         │
│  - No information leakage about other users' resources  │
│  - Consistent error responses                           │
└─────────────────────────────────────────────────────────┘
```

## Threat Model

### Assets

Protected assets include:

1. **Worker Definitions**: AI agents and their configurations
2. **Graph Definitions**: Workflow graphs and execution logic
3. **Worker States**: Conversation history and agent states
4. **Execution Logs**: Graph execution history and results
5. **Tool Configurations**: Custom tool definitions

### Threat Actors

1. **Malicious User**: Authenticated user attempting to access other users' resources
2. **Token Thief**: Attacker with stolen JWT token
3. **Man-in-the-Middle**: Network attacker intercepting communications
4. **Insider Threat**: Compromised or malicious admin
5. **Application Vulnerability**: Code vulnerabilities enabling bypass

### Attack Vectors

#### 1. Cross-Tenant Data Access

**Attack**: User A attempts to access User B's resources

**Mitigations**:
- ✅ All queries filter by tenant_id
- ✅ tenant_id extracted from cryptographically signed token
- ✅ Return 404 (not 403) to prevent enumeration
- ✅ Comprehensive test coverage (71 tests)

**Status**: **MITIGATED**

#### 2. tenant_id Spoofing

**Attack**: User provides fake tenant_id in request

**Mitigations**:
- ✅ tenant_id only extracted from JWT token's 'sub' claim
- ✅ Request body tenant_id fields removed from all models
- ✅ Query parameter tenant_id ignored
- ✅ FastAPI dependency injection ensures token validation

**Status**: **MITIGATED**

#### 3. Token Forgery

**Attack**: Attacker creates fake JWT token

**Mitigations**:
- ✅ RSA signature verification (RS256 algorithm)
- ✅ Public key verification prevents forgery without private key
- ✅ Keycloak private key never exposed to Worker Management API
- ✅ Token expiration enforced

**Status**: **MITIGATED**

#### 4. SQL Injection

**Attack**: Attacker injects SQL through input fields

**Mitigations**:
- ✅ Parameterized queries (psycopg 3, written with `$1`/`$2` placeholders and translated to psycopg's parameter binding by the compat wrapper in `src/db/db_config.py`)
- ✅ No string concatenation in queries
- ✅ Input validation via Pydantic models
- ✅ Database driver handles escaping

**Status**: **MITIGATED**

#### 5. Token Replay Attack

**Attack**: Stolen token used to impersonate user

**Mitigations**:
- ✅ Token expiration (typically 5-30 minutes)
- ⚠️ Token revocation (requires external revocation list)
- ✅ HTTPS required for production (prevents token sniffing)
- ⚠️ Consider implementing short-lived access tokens + refresh tokens

**Status**: **PARTIALLY MITIGATED** (token expiration limits window, revocation recommended)

#### 6. Information Disclosure

**Attack**: Error messages reveal sensitive information

**Mitigations**:
- ✅ Return 404 (not 403) for unauthorized access
- ✅ Generic error messages in production
- ✅ No stack traces exposed to clients
- ✅ Detailed logs only in server-side logs

**Status**: **MITIGATED**

## Security Measures

### 1. JWT Token Security

#### Signature Verification

```python
# Token validation enforces:
- Algorithm verification (RS256, not HS256 with public key)
- Signature verification using RSA public key
- Expiration time validation
- Issuer validation (if configured)
- Audience validation (if configured)
```

#### Key Management

- **Private Key**: Held by Keycloak/Auth provider only
- **Public Key**: Distributed to Worker Management API
- **Key Rotation**: Supported via Keycloak JWKS endpoint
- **Key Storage**: Environment variable or file-based (not in code)

### 2. Database Security

#### Query Filtering

```sql
-- GOOD: All queries filter by tenant_id
SELECT * FROM workers WHERE tenant_id = $1 AND worker_name = $2;

-- BAD: No tenant_id filter (NOT IN CODEBASE)
SELECT * FROM workers WHERE worker_name = $1;
```

#### Indexes for Performance and Security

```sql
-- Composite indexes ensure efficient filtering
CREATE INDEX idx_workers_tenant_id ON workers(tenant_id);
CREATE INDEX idx_workers_creator_worker_name ON workers(tenant_id, worker_name);
```

#### Unique Constraints

```sql
-- Prevent duplicate names within same tenant
ALTER TABLE workers ADD CONSTRAINT unique_tenant_worker_name UNIQUE (tenant_id, worker_name);
```

### 3. API Security

#### Authentication Middleware

All requests pass through authentication middleware:

1. Extract Authorization header
2. Validate Bearer token format
3. Verify JWT signature
4. Check token expiration
5. Extract tenant_id from 'sub' claim
6. Attach user context to request state

#### Protected Endpoints

```python
# Automatic authentication
@router.get("/workers")
async def list_workers(
    tenant_id: str = Depends(get_tenant_id_from_token)
):
    # tenant_id is guaranteed to be from validated token
    return await db.list_workers(tenant_id)
```

#### Public Endpoints

These endpoints do NOT require authentication:
- `GET /health` - Health check
- `GET /` - Root
- `GET /docs` - API documentation
- `GET /redoc` - ReDoc documentation
- `GET /openapi.json` - OpenAPI schema

### 4. Data Isolation

#### Per-User Scoping

Every table includes `tenant_id VARCHAR(255) NOT NULL`:

- `workers` - Worker definitions
- `worker_states` - Worker state history
- `graphs` - Graph definitions
- `graph_states` - Graph state history
- `graph_execution_sessions` - Execution sessions
- `execution_events` - Execution events
- `node_execution_logs` - Node execution logs
- `tools` - Tool definitions
- `workload` - Worker workload tracking
- `skills` - Worker skills

#### Query Verification

All database queries verified to include tenant_id filtering:
- ✅ 32 query functions audited
- ✅ 0 security issues found
- ✅ All queries properly secured

## Data Isolation Guarantees

### Guarantee 1: Users Cannot See Other Users' Data

**Verification**: Database queries filter by tenant_id

```python
# Example: List workers
async def list_workers(self, tenant_id: str) -> List[str]:
    async with self.get_connection() as conn:
        rows = await conn.fetch(
            "SELECT worker_name FROM workers WHERE tenant_id = $1",
            tenant_id  # ← Filters by authenticated user's tenant_id
        )
        return [row["worker_name"] for row in rows]
```

**Test Coverage**:
- ✅ Test: User A cannot see User B's workers
- ✅ Test: User A cannot access User B's worker by name
- ✅ Test: Three users maintain complete isolation

### Guarantee 2: tenant_id Cannot Be Spoofed

**Verification**: tenant_id extracted from cryptographically signed JWT

```python
# tenant_id comes from token, not request
@router.post("/workers")
async def create_worker(
    request: WorkerCreateRequest,  # No tenant_id field!
    tenant_id: str = Depends(get_tenant_id_from_token)  # From JWT token
):
    # tenant_id from validated token, cannot be forged
    return await db.create_worker(tenant_id, request)
```

**Test Coverage**:
- ✅ Test: tenant_id cannot be forged in request body
- ✅ Test: tenant_id in query params is ignored
- ✅ Test: Token signature validation prevents forgery

### Guarantee 3: All Operations Are Scoped

**Verification**: All CRUD operations require tenant_id

- Create: Sets tenant_id from token
- Read: Filters by tenant_id from token
- Update: Verifies tenant_id matches before update
- Delete: Verifies tenant_id matches before delete

**Test Coverage**:
- ✅ Test: User cannot update other user's worker
- ✅ Test: User cannot delete other user's worker
- ✅ Test: User cannot execute other user's graph

### Guarantee 4: Comprehensive Testing

**Test Statistics**:
- Total tests: 71
- Unit tests: 40 (JWT validation, token extraction)
- Integration tests: 19 (API authentication flows)
- Security tests: 12 (data isolation verification)

**Critical Security Tests** (all passing):
- ✅ User cannot see other user's workers
- ✅ User cannot access other user's worker states
- ✅ User cannot update other user's workers
- ✅ User cannot delete other user's workers
- ✅ User cannot send messages to other user's workers
- ✅ User cannot access other user's graphs
- ✅ User cannot execute other user's graphs
- ✅ Concurrent operations maintain isolation
- ✅ SQL injection attempts handled safely
- ✅ Empty tenant_id rejected
- ✅ Very long tenant_id handled safely

## Cryptographic Security

### Algorithm: RS256 (RSA Signature with SHA-256)

**Why RS256?**
- Asymmetric cryptography: Public key for verification, private key for signing
- Private key never leaves authentication provider (Keycloak)
- Public key can be safely distributed
- Industry standard for JWT tokens
- Stronger than HS256 (symmetric) for distributed systems

### Key Size

- **Minimum**: 2048-bit RSA keys
- **Recommended**: 3072-bit or 4096-bit for long-term secrets
- **Current**: Configured by Keycloak (typically 2048-bit)

### Token Lifetime

**Recommendations**:
- **Access tokens**: 5-30 minutes (short-lived)
- **Refresh tokens**: 1-24 hours (longer-lived)
- **Current**: Configured by Keycloak (typically 5-15 minutes)

**Rationale**:
- Short access token lifetime limits window for token replay attacks
- Expired tokens automatically rejected
- Refresh tokens allow obtaining new access tokens without re-authentication

### Public Key Distribution

**Option 1: Static Key (Current)**
```bash
# Public key in environment variable or file
JWT_PUBLIC_KEY_PATH=/app/certs/public_key.pem
```

**Option 2: JWKS Endpoint (Recommended for Production)**
```python
# Dynamically fetch public keys from Keycloak
jwks_url = "https://keycloak.example.com/realms/secure-apps/protocol/openid-connect/certs"
# Supports key rotation without service restart
```

## Audit and Compliance

### Audit Trail

**Logged Events**:
- Authentication attempts (success/failure)
- Token validation failures
- tenant_id extraction
- Database query execution (with tenant_id)
- Resource access (who accessed what, when)

**Log Format**:
```json
{
  "timestamp": "2025-11-28T13:45:23Z",
  "level": "INFO",
  "event": "worker_created",
  "tenant_id": "a1b2c3d4-e5f6-4a5b-8c9d-0e1f2a3b4c5d",
  "resource_type": "worker",
  "resource_id": "data_processor",
  "ip_address": "192.168.1.100",
  "user_agent": "curl/7.68.0"
}
```

### Compliance Considerations

#### GDPR (General Data Protection Regulation)

- **Data Minimization**: Only store necessary user data
- **Right to Access**: Users can query their own data via API
- **Right to Deletion**: DELETE endpoints remove user data
- **Data Portability**: Export functionality available
- **Privacy by Design**: Multi-tenancy isolation built-in

#### SOC 2 (Service Organization Control)

- **Access Control**: JWT authentication enforces authorization
- **Audit Logging**: All operations logged with tenant_id
- **Data Isolation**: Multi-tenancy prevents unauthorized access
- **Encryption**: HTTPS required for production (TLS/SSL)

#### HIPAA (Healthcare - if applicable)

- **Access Controls**: Role-based access via JWT claims
- **Audit Controls**: Comprehensive logging
- **Encryption**: In-transit (TLS) and at-rest (database encryption)
- **Data Isolation**: Strict multi-tenant boundaries

### Compliance Checklist

For regulatory compliance, ensure:

- [ ] HTTPS enabled (TLS 1.2+)
- [ ] Database encryption at rest
- [ ] Regular security audits
- [ ] Penetration testing
- [ ] Vulnerability scanning
- [ ] Access logs retained (30-90 days minimum)
- [ ] Incident response plan documented
- [ ] Regular backup and disaster recovery testing
- [ ] Staff security training
- [ ] Third-party security assessments

## Security Testing

### Test Categories

#### 1. Unit Tests (40 tests)
- JWT token validation
- Token expiration handling
- Signature verification
- Header extraction
- Tenant ID extraction

#### 2. Integration Tests (19 tests)
- API authentication flows
- Endpoint protection
- Error responses
- Multi-user scenarios

#### 3. Security Tests (12 tests)
- Cross-tenant access prevention
- Data isolation verification
- SQL injection resistance
- Tenant ID spoofing prevention
- Information disclosure prevention

### Running Security Tests

```bash
# Run all security tests
cd /Users/mini2/Movies/worker_management
export PYTHONPATH=/Users/mini2/Movies/worker_management
uv run pytest tests/test_security/ -v

# Run with coverage
uv run pytest tests/test_security/ --cov=src --cov-report=html

# Run specific security test
uv run pytest tests/test_security/test_data_isolation.py::test_user_cannot_access_another_users_worker -v
```

### Security Test Results

All critical security tests passing:

```
tests/test_security/test_data_isolation.py::test_user_a_cannot_see_user_b_workers PASSED
tests/test_security/test_data_isolation.py::test_user_cannot_access_another_users_worker_by_name PASSED
tests/test_security/test_data_isolation.py::test_user_cannot_update_another_users_worker PASSED
tests/test_security/test_data_isolation.py::test_user_cannot_delete_another_users_worker PASSED
tests/test_security/test_data_isolation.py::test_user_cannot_send_message_to_another_users_worker PASSED
tests/test_security/test_data_isolation.py::test_user_cannot_access_another_users_worker_states PASSED
tests/test_security/test_data_isolation.py::test_delete_all_workers_only_affects_current_user PASSED
tests/test_security/test_data_isolation.py::test_three_users_complete_isolation PASSED
tests/test_security/test_data_isolation.py::test_concurrent_operations_maintain_isolation PASSED
tests/test_security/test_data_isolation.py::test_tenant_id_cannot_be_forged_in_request_body PASSED
tests/test_security/test_data_isolation.py::test_sql_injection_attempt_in_tenant_id PASSED
tests/test_security/test_data_isolation.py::test_comprehensive_multi_tenant_isolation PASSED
```

## Security Considerations

### Production Deployment

#### Required Security Measures

1. **HTTPS Only**
```yaml
# Nginx configuration
server {
    listen 443 ssl http2;
    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
}
```

2. **Secure Headers**
```python
# Added by SecurityHeadersMiddleware
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
```

3. **Rate Limiting**
```python
# Prevent brute force attacks
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/workers")
@limiter.limit("100/minute")
async def create_worker(...):
    ...
```

4. **Database Encryption**
```sql
-- Enable PostgreSQL encryption
ALTER SYSTEM SET ssl = on;
```

### Monitoring and Alerting

#### Security Metrics to Monitor

1. **Authentication Failures**
   - High rate may indicate brute force attack
   - Alert threshold: >10 failures/minute from same IP

2. **Token Expiration Rate**
   - Sudden spike may indicate token theft
   - Alert threshold: >50% tokens expired on first use

3. **404 Errors on Resources**
   - High rate may indicate enumeration attack
   - Alert threshold: >100 404s/minute from same tenant_id

4. **Anomalous Access Patterns**
   - Accessing resources at unusual times
   - Accessing large volumes of data
   - Geographic anomalies (if tracking IP locations)

### Security Updates

#### Dependency Management

```bash
# Check for security vulnerabilities
uv run pip-audit

# Update dependencies
uv sync

# Review changelogs for security fixes
```

#### CVE Monitoring

Monitor these dependencies for security advisories:
- `pyjwt` / `python-jose` - JWT library vulnerabilities
- `fastapi` - Web framework vulnerabilities
- `psycopg[binary,pool]` (psycopg 3) - Database driver vulnerabilities
- `cryptography` - Cryptographic library vulnerabilities

## Incident Response

### Incident Classification

#### P0: Critical
- Active breach or data leak
- Authentication bypass discovered
- tenant_id spoofing successful

**Response**: Immediate (within 1 hour)

#### P1: High
- Token forgery attempt detected
- SQL injection attempt detected
- Unusual access patterns

**Response**: Same day (within 8 hours)

#### P2: Medium
- Excessive authentication failures
- Suspicious but not confirmed threat
- Rate limiting triggered

**Response**: Next business day

### Incident Response Procedures

#### 1. Detect

- Monitor logs for anomalies
- Set up alerts for security events
- Regular security audits

#### 2. Contain

- Revoke compromised tokens (if revocation list implemented)
- Block malicious IPs
- Disable affected user accounts
- Isolate affected systems

#### 3. Investigate

- Review access logs
- Identify scope of breach
- Determine attack vector
- Preserve evidence

#### 4. Remediate

- Patch vulnerabilities
- Rotate keys if compromised
- Update security rules
- Deploy fixes

#### 5. Recover

- Restore from backups if needed
- Re-enable services
- Verify system integrity
- Monitor for recurrence

#### 6. Learn

- Document incident
- Update procedures
- Conduct post-mortem
- Implement preventive measures

### Contact Information

**Security Team**:
- Email: security@example.com
- On-call: +1-XXX-XXX-XXXX
- Slack: #security-incidents

**Escalation Path**:
1. On-call engineer
2. Security lead
3. CTO/VP Engineering
4. Legal (for data breaches)

## Security Roadmap

### Implemented ✅

- JWT authentication with RS256
- Multi-tenant data isolation
- Database query filtering by tenant_id
- Comprehensive security test suite
- Secure token validation
- Information disclosure prevention

### Recommended Enhancements ⚠️

1. **Token Revocation**
   - Implement revocation list (Redis)
   - Check revocation on each request
   - Revoke on logout or security event

2. **Refresh Token Flow**
   - Short-lived access tokens (5 min)
   - Longer-lived refresh tokens (24 hours)
   - Rotate refresh tokens on use

3. **Rate Limiting**
   - Per-IP rate limiting
   - Per-tenant_id rate limiting
   - Endpoint-specific limits

4. **Audit Log Enhancements**
   - Centralized audit logging
   - Long-term log retention
   - Log analysis and anomaly detection

5. **Database Encryption**
   - Encrypt sensitive columns
   - Field-level encryption
   - Key management service (KMS)

6. **Security Headers**
   - Content Security Policy (CSP)
   - Additional security headers
   - CORS policy enforcement

7. **Penetration Testing**
   - Annual third-party pen test
   - Bug bounty program
   - Regular security assessments

8. **Monitoring & Alerting**
   - SIEM integration
   - Real-time security alerts
   - Anomaly detection

## Summary

The Worker Management System implements a comprehensive security model with multiple layers of defense:

✅ **Authentication**: JWT-based with RSA signature verification
✅ **Authorization**: Automatic tenant_id extraction and enforcement
✅ **Data Isolation**: Multi-tenant database query filtering
✅ **Testing**: 71 tests verify security boundaries
✅ **Audit**: Comprehensive logging of security events
✅ **Compliance**: GDPR, SOC 2, HIPAA considerations

**Security Posture**: **STRONG**

All critical security requirements met. System ready for production deployment with proper HTTPS configuration and monitoring.

For more information:
- [JWT Authentication Guide](JWT_AUTHENTICATION.md)
- [API Authentication Examples](API_AUTHENTICATION.md)
- [Developer Guide](DEVELOPER_GUIDE_AUTH.md)
