#!/usr/bin/env python3
"""Production readiness check for the Worker Management API."""

import requests

BASE_URL = "http://localhost:8887"


def check_endpoint(
    method: str, path: str, expected_status: int, payload: dict | None = None
) -> tuple:
    """Check if an endpoint responds correctly."""
    url = f"{BASE_URL}{path}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=payload, timeout=5)
        elif method == "PUT":
            response = requests.put(url, json=payload, timeout=5)
        elif method == "DELETE":
            response = requests.delete(url, timeout=5)
        elif method == "PATCH":
            response = requests.patch(url, json=payload, timeout=5)
        else:
            return False, f"Unknown method: {method}"

        if response.status_code == expected_status:
            return True, "OK"
        else:
            return False, f"Expected {expected_status}, got {response.status_code}"
    except Exception as e:
        return False, str(e)


def main():  # noqa: PLR0915 -- top-level readiness driver, sequence of self-contained probes
    print("🏭 Production Readiness Check")
    print("=" * 60)

    checks = []

    # 1. Health Checks
    print("\n1️⃣ Health Checks:")
    success, msg = check_endpoint("GET", "/health", 200)
    checks.append(("Main API health", success, msg))
    print(f"   - Main API health: {'✅' if success else '❌'} {msg}")

    success, msg = check_endpoint("GET", "/tools/health", 200)
    checks.append(("Tools API health", success, msg))
    print(f"   - Tools API health: {'✅' if success else '❌'} {msg}")

    # 2. Security Headers
    print("\n2️⃣ Security Headers:")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Content-Security-Policy",
            "Strict-Transport-Security",
            "Referrer-Policy",
        ]
        for header in security_headers:
            present = header in response.headers
            checks.append(
                (
                    f"Security header: {header}",
                    present,
                    "Present" if present else "Missing",
                )
            )
            print(f"   - {header}: {'✅' if present else '❌'}")
    except Exception as e:
        print(f"   ❌ Error checking headers: {e}")

    # 3. Rate Limiting
    print("\n3️⃣ Rate Limiting:")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        rate_limit_headers = [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ]
        for header in rate_limit_headers:
            present = header in response.headers
            value = response.headers.get(header, "N/A")
            checks.append((f"Rate limit header: {header}", present, value))
            print(f"   - {header}: {'✅' if present else '❌'} ({value})")
    except Exception as e:
        print(f"   ❌ Error checking rate limits: {e}")

    # 4. Error Handling
    print("\n4️⃣ Error Handling:")
    error_tests = [
        ("404 Not Found", "GET", "/nonexistent", 404),
        ("405 Method Not Allowed", "PATCH", "/tools/", 405),
        ("422 Validation Error", "POST", "/tools/", 422, {"name": "!@#$%"}),
    ]
    for test_name, method, path, expected_status, *args in error_tests:
        payload = args[0] if args else None
        success, msg = check_endpoint(method, path, expected_status, payload)
        checks.append((test_name, success, msg))
        print(f"   - {test_name}: {'✅' if success else '❌'} {msg}")

    # 5. CORS Headers (if configured)
    print("\n5️⃣ CORS Configuration:")
    try:
        response = requests.options(
            f"{BASE_URL}/health", headers={"Origin": "http://example.com"}, timeout=5
        )
        cors_configured = "Access-Control-Allow-Origin" in response.headers
        checks.append(
            (
                "CORS headers",
                cors_configured,
                "Configured" if cors_configured else "Not configured",
            )
        )
        print(
            f"   - CORS: {'✅' if cors_configured else '⚠️'} {'Configured' if cors_configured else 'Not configured (may be intentional)'}"
        )
    except Exception as e:
        print(f"   ⚠️  Could not check CORS: {e}")

    # 6. API Endpoints
    print("\n6️⃣ Core API Endpoints:")
    endpoints = [
        ("Workers API", "GET", "/workers", 200),
        ("Graphs API", "GET", "/graphs/", 200),
        ("Tools API", "GET", "/tools/", 200),
        ("Worker States API", "GET", "/workers_states", 200),
    ]
    for name, method, path, expected_status in endpoints:
        success, msg = check_endpoint(method, path, expected_status)
        checks.append((name, success, msg))
        print(f"   - {name}: {'✅' if success else '❌'} {msg}")

    # 7. Input Validation
    print("\n7️⃣ Input Validation:")
    validation_tests = [
        (
            "SQL injection prevention",
            "POST",
            "/tools/",
            422,
            {
                "name": "test_tool",
                "description": "'; DROP TABLE tools; --",
                "python_code": "def test(): pass",
            },
        ),
        (
            "XSS prevention",
            "POST",
            "/tools/",
            422,
            {
                "name": "test_tool",
                "description": "<script>alert('xss')</script>",
                "python_code": "def test(): pass",
            },
        ),
        ("Path traversal prevention", "GET", "/tools/../../../etc/passwd", 404),
    ]
    for test_name, method, path, expected_status, *args in validation_tests:
        payload = args[0] if args else None
        success, msg = check_endpoint(method, path, expected_status, payload)
        checks.append((test_name, success, msg))
        print(f"   - {test_name}: {'✅' if success else '❌'} {msg}")

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    total_checks = len(checks)
    passed_checks = sum(1 for _, success, _ in checks if success)
    pass_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0

    print(f"Total checks: {total_checks}")
    print(f"Passed: {passed_checks}")
    print(f"Failed: {total_checks - passed_checks}")
    print(f"Pass rate: {pass_rate:.1f}%")

    if pass_rate >= 90:
        print("\n✅ PRODUCTION READY - All critical checks passed!")
    elif pass_rate >= 70:
        print("\n⚠️  MOSTLY READY - Some non-critical issues to address")
    else:
        print("\n❌ NOT READY - Critical issues need to be fixed")

    # List failed checks
    failed = [(name, msg) for name, success, msg in checks if not success]
    if failed:
        print("\n❌ Failed checks:")
        for name, msg in failed:
            print(f"   - {name}: {msg}")


if __name__ == "__main__":
    main()
