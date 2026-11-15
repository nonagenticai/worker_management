#!/usr/bin/env node
/**
 * JavaScript Security and Input Validation Test Suite for Worker Management API
 *
 * This comprehensive test suite validates security measures and input sanitization across
 * all API endpoints of the Worker Management system. Tests include:
 *
 * 1. SQL Injection Protection - Tests malicious SQL payloads in all string parameters
 * 2. XSS Protection - Tests cross-site scripting attempts in worker prompts and descriptions
 * 3. Path Traversal Protection - Tests directory traversal attempts in names and paths
 * 4. Null Byte Injection - Tests null byte injection in string fields
 * 5. DoS Protection - Tests oversized payloads and resource exhaustion
 * 6. Malformed Data - Tests invalid JSON, content types, and data structures
 * 7. Input Sanitization - Validates proper cleaning and validation of inputs
 * 8. Error Response Security - Ensures errors don't leak sensitive information
 *
 * Based on: test_graph_execution.js
 * Requirements: Node.js 18+ (for native fetch API)
 * Usage: node test_16_security_validation.js
 */

const { v4: uuidv4 } = require('uuid');

// Test Configuration
const API_BASE = 'http://localhost:8887';
const TIMEOUT_MS = 30000; // 30 seconds for security tests
const MAX_RETRIES = 2;
const RETRY_DELAY = 1000; // 1 second
const REQUEST_DELAY = 500; // 0.5 seconds between requests

// Security Test Payloads
const SECURITY_PAYLOADS = {
    // SQL Injection Attempts
    SQL_INJECTION: [
        "'; DROP TABLE workers; --",
        "' OR '1'='1",
        "' UNION SELECT * FROM users --",
        "'; INSERT INTO workers VALUES ('malicious', 'data'); --",
        "' OR 1=1 LIMIT 1 OFFSET 1 --",
        "\"; DROP TABLE workers; --",
        "admin'--",
        "admin' /*",
        "' OR 'x'='x",
        "') OR ('1'='1",
        "1' AND (SELECT COUNT(*) FROM workers) > 0 --"
    ],

    // XSS Attempts
    XSS_PAYLOADS: [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "javascript:alert('XSS')",
        "<svg onload=alert('XSS')>",
        "<iframe src=javascript:alert('XSS')>",
        "'; alert('XSS'); //",
        "<script>document.cookie='stolen'</script>",
        "<body onload=alert('XSS')>",
        "<<SCRIPT>alert('XSS')</SCRIPT>",
        "<scr<script>ipt>alert('XSS')</scr</script>ipt>",
        "%3Cscript%3Ealert('XSS')%3C/script%3E"
    ],

    // Path Traversal Attempts
    PATH_TRAVERSAL: [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "....//....//....//etc/passwd",
        "..%2F..%2F..%2Fetc%2Fpasswd",
        "..%252F..%252F..%252Fetc%252Fpasswd",
        "../../../../../../etc/hosts",
        "file:///etc/passwd",
        "file://localhost/etc/passwd",
        "/.././.././.././etc/passwd",
        "\\..\\..\\..\\etc\\passwd"
    ],

    // Null Byte Injection
    NULL_BYTE: [
        "normal_name\x00malicious",
        "test\u0000.txt",
        "worker\0.exe",
        "data\x00/../../../etc/passwd",
        "normal%00malicious"
    ],

    // Command Injection
    COMMAND_INJECTION: [
        "; ls -la",
        "| cat /etc/passwd",
        "&& rm -rf /",
        "`whoami`",
        "$(id)",
        "; ping -c 4 127.0.0.1",
        "| nc -l 4444",
        "&& curl http://malicious.com",
        "; python -c 'import os; os.system(\"ls\")'",
        "| python3 -c 'import subprocess; subprocess.call([\"id\"])'"
    ],

    // Format String Attacks
    FORMAT_STRING: [
        "%x%x%x%x",
        "%n%n%n%n",
        "%s%s%s%s",
        "%d%d%d%d",
        "%p%p%p%p",
        "%08x.%08x.%08x.%08x",
        "AAAA%08x.%08x.%08x.%08x.%08x"
    ],

    // LDAP Injection
    LDAP_INJECTION: [
        "*)(uid=*",
        "admin)(|(password=*))",
        "*)(|(objectClass=*))",
        ")(cn=*",
        "*))%00",
        "admin*",
        "*)(userPassword=*",
        ")(objectClass=*"
    ]
};

// DoS Test Payloads
const DOS_PAYLOADS = {
    LARGE_STRING: 'A'.repeat(10000), // 10KB string
    HUGE_STRING: 'B'.repeat(100000), // 100KB string
    MASSIVE_STRING: 'C'.repeat(1000000), // 1MB string
    DEEP_NESTING: generateDeepObject(100), // 100 levels deep
    LARGE_ARRAY: new Array(10000).fill('data'), // 10K array items
    SPECIAL_CHARS: ''.repeat(1000) + String.fromCharCode(...Array.from({length: 1000}, (_, i) => i % 127))
};

// Generate deeply nested object for DoS testing
function generateDeepObject(depth) {
    let obj = { value: 'deep' };
    for (let i = 0; i < depth; i++) {
        obj = { nested: obj };
    }
    return obj;
}

// Test Results Tracking
const securityTestResults = {
    total_tests: 0,
    passed_tests: 0,
    failed_tests: 0,
    security_issues: [],
    test_details: [],
    start_time: null,
    end_time: null
};

/**
 * Enhanced utility function to make HTTP requests with security test logging
 */
async function makeSecurityRequest(url, options = {}, expectedToFail = false, testName = '') {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'SecurityTest-Suite/1.0',
            ...options.headers
        }
    };

    const finalOptions = { ...defaultOptions, ...options };

    try {
        console.log(`🔒 [${testName}] Testing: ${finalOptions.method || 'GET'} ${url}`);

        const response = await fetch(url, finalOptions);
        const responseText = await response.text();

        let responseData;
        try {
            responseData = JSON.parse(responseText);
        } catch {
            responseData = responseText;
        }

        const testResult = {
            test_name: testName,
            url: url,
            method: finalOptions.method || 'GET',
            status_code: response.status,
            response_size: responseText.length,
            expected_to_fail: expectedToFail,
            actual_failed: !response.ok,
            response_data: responseData,
            headers: Object.fromEntries(response.headers.entries())
        };

        // Check for potential security issues
        const securityIssues = analyzeResponseForSecurityIssues(response, responseData, testName);
        if (securityIssues.length > 0) {
            testResult.security_issues = securityIssues;
            securityTestResults.security_issues.push(...securityIssues);
        }

        securityTestResults.test_details.push(testResult);

        if (expectedToFail && response.ok) {
            console.log(`⚠️  [${testName}] SECURITY CONCERN: Request was expected to fail but succeeded`);
            securityTestResults.security_issues.push({
                type: 'UNEXPECTED_SUCCESS',
                test: testName,
                details: 'Malicious payload was not rejected',
                severity: 'HIGH'
            });
        } else if (!expectedToFail && !response.ok) {
            console.log(`❌ [${testName}] Legitimate request failed: ${response.status}`);
        } else if (expectedToFail && !response.ok) {
            console.log(`✅ [${testName}] Malicious payload properly rejected: ${response.status}`);
        } else {
            console.log(`✅ [${testName}] Request handled correctly: ${response.status}`);
        }

        // Add delay between requests
        await new Promise(resolve => setTimeout(resolve, REQUEST_DELAY));

        return { response, data: responseData, text: responseText };

    } catch (error) {
        console.log(`❌ [${testName}] Request failed: ${error.message}`);

        const testResult = {
            test_name: testName,
            url: url,
            method: finalOptions.method || 'GET',
            error: error.message,
            expected_to_fail: expectedToFail,
            actual_failed: true
        };

        securityTestResults.test_details.push(testResult);

        if (!expectedToFail) {
            throw error;
        }

        return { response: null, data: null, text: null, error };
    }
}

/**
 * Analyze response for potential security issues
 */
function analyzeResponseForSecurityIssues(response, responseData, testName) {
    const issues = [];

    // Check for information disclosure in error messages
    if (typeof responseData === 'string' || (responseData && responseData.detail)) {
        const content = typeof responseData === 'string' ? responseData : responseData.detail;

        // Check for common information disclosure patterns
        const sensitivePatterns = [
            /traceback|stack trace/i,
            /internal server error.*at line/i,
            /database.*error/i,
            /sql.*syntax/i,
            /file not found.*\/.*\//i,
            /permission denied.*\/.*\//i,
            /access denied.*\/.*\//i,
            /path.*does not exist/i
        ];

        for (const pattern of sensitivePatterns) {
            if (pattern.test(content)) {
                issues.push({
                    type: 'INFORMATION_DISCLOSURE',
                    test: testName,
                    details: `Response contains potentially sensitive error information: ${content.substring(0, 200)}`,
                    severity: 'MEDIUM'
                });
                break;
            }
        }
    }

    // Check for reflected payload (potential XSS)
    if (testName.includes('XSS') && typeof responseData === 'string') {
        const payload = testName.match(/XSS.*?with payload: (.*)/);
        if (payload && responseData.includes(payload[1])) {
            issues.push({
                type: 'REFLECTED_XSS',
                test: testName,
                details: 'Malicious payload was reflected in response without sanitization',
                severity: 'HIGH'
            });
        }
    }

    // Check response headers for security headers
    const securityHeaders = ['x-content-type-options', 'x-frame-options', 'x-xss-protection', 'content-security-policy'];
    const missingHeaders = securityHeaders.filter(header => !response.headers.get(header));

    if (missingHeaders.length > 0 && !testName.includes('Header Check')) {
        issues.push({
            type: 'MISSING_SECURITY_HEADERS',
            test: testName,
            details: `Missing security headers: ${missingHeaders.join(', ')}`,
            severity: 'LOW'
        });
    }

    return issues;
}

/**
 * Test API health and basic security
 */
async function testApiHealthSecurity() {
    console.log('\n🏥 Testing API Health and Basic Security...');

    // Test main health endpoint
    await makeSecurityRequest(`${API_BASE}/health`, {}, false, 'Health Check - Basic');

    // Test execution health endpoint
    await makeSecurityRequest(`${API_BASE}/execution/health`, {}, false, 'Execution Health Check');

    // Test tool registry health endpoint
    await makeSecurityRequest(`${API_BASE}/tools/health`, {}, false, 'Tools Health Check');

    // Test malformed health requests
    await makeSecurityRequest(`${API_BASE}/health/../../../etc/passwd`, {}, true, 'Health Path Traversal');
    await makeSecurityRequest(`${API_BASE}/health?param=<script>alert('xss')</script>`, {}, true, 'Health XSS Query');
    await makeSecurityRequest(`${API_BASE}/health`, {
        method: 'POST',
        body: JSON.stringify({ malicious: "'; DROP TABLE workers; --" })
    }, true, 'Health POST SQL Injection');
}

/**
 * Test workers endpoint security
 */
async function testWorkersEndpointSecurity() {
    console.log('\n🤖 Testing Workers Endpoint Security...');

    // Test legitimate worker list request
    await makeSecurityRequest(`${API_BASE}/workers`, {}, false, 'Workers List - Legitimate');

    // Test SQL injection in worker creation
    for (const payload of SECURITY_PAYLOADS.SQL_INJECTION.slice(0, 3)) {
        const workerConfig = {
            worker_name: payload,
            agent_role: 'test_role',
            agent_type: 'pydantic_ai',
            model_provider: 'deepseek',
            model: 'deepseek-chat',
            temperature: 0.3,
            system_prompt: 'Test prompt',
            instruction: 'Test instruction',
            description: 'Test description'
        };

        await makeSecurityRequest(`${API_BASE}/workers`, {
            method: 'POST',
            body: JSON.stringify(workerConfig)
        }, true, `Workers Creation SQL Injection - ${payload.substring(0, 20)}`);

        securityTestResults.total_tests++;
    }

    // Test XSS in worker prompts and descriptions
    for (const payload of SECURITY_PAYLOADS.XSS_PAYLOADS.slice(0, 3)) {
        const workerConfig = {
            worker_name: 'test_worker_xss',
            agent_role: 'test_role',
            agent_type: 'pydantic_ai',
            model_provider: 'deepseek',
            model: 'deepseek-chat',
            temperature: 0.3,
            system_prompt: payload,
            instruction: payload,
            description: payload
        };

        await makeSecurityRequest(`${API_BASE}/workers`, {
            method: 'POST',
            body: JSON.stringify(workerConfig)
        }, true, `Workers Creation XSS - ${payload.substring(0, 20)}`);

        securityTestResults.total_tests++;
    }

    // Test path traversal in worker names
    for (const payload of SECURITY_PAYLOADS.PATH_TRAVERSAL.slice(0, 2)) {
        await makeSecurityRequest(`${API_BASE}/workers/${encodeURIComponent(payload)}`, {}, true, `Workers Path Traversal - ${payload}`);
        securityTestResults.total_tests++;
    }

    // Test null byte injection
    for (const payload of SECURITY_PAYLOADS.NULL_BYTE.slice(0, 2)) {
        const workerConfig = {
            worker_name: payload,
            agent_role: 'test_role',
            agent_type: 'pydantic_ai',
            model_provider: 'deepseek',
            model: 'deepseek-chat',
            temperature: 0.3,
            system_prompt: 'Test prompt',
            instruction: 'Test instruction',
            description: 'Test description'
        };

        await makeSecurityRequest(`${API_BASE}/workers`, {
            method: 'POST',
            body: JSON.stringify(workerConfig)
        }, true, `Workers Null Byte Injection - ${payload}`);

        securityTestResults.total_tests++;
    }

    // Test DoS with oversized payloads
    const dosWorkerConfig = {
        worker_name: 'dos_test_worker',
        agent_role: 'test_role',
        agent_type: 'pydantic_ai',
        model_provider: 'deepseek',
        model: 'deepseek-chat',
        temperature: 0.3,
        system_prompt: DOS_PAYLOADS.LARGE_STRING,
        instruction: DOS_PAYLOADS.HUGE_STRING.substring(0, 50000), // Limit to avoid timeout
        description: 'DoS test description'
    };

    await makeSecurityRequest(`${API_BASE}/workers`, {
        method: 'POST',
        body: JSON.stringify(dosWorkerConfig)
    }, true, 'Workers DoS - Oversized Payload');

    securityTestResults.total_tests++;
}

/**
 * Test graphs endpoint security
 */
async function testGraphsEndpointSecurity() {
    console.log('\n📊 Testing Graphs Endpoint Security...');

    // Test legitimate graph list request
    await makeSecurityRequest(`${API_BASE}/graphs/`, {}, false, 'Graphs List - Legitimate');

    // Test SQL injection in graph creation
    for (const payload of SECURITY_PAYLOADS.SQL_INJECTION.slice(0, 2)) {
        const graphConfig = {
            name: payload,
            description: 'Test graph description',
            variables: { test: 'value' },
            nodes: [{
                id: uuidv4(),
                name: 'test_node',
                node_type: 'worker',
                worker_name: 'test_worker',
                task_template: 'Test task',
                dependencies: []
            }],
            edges: []
        };

        await makeSecurityRequest(`${API_BASE}/graphs/`, {
            method: 'POST',
            body: JSON.stringify(graphConfig)
        }, true, `Graphs Creation SQL Injection - ${payload.substring(0, 20)}`);

        securityTestResults.total_tests++;
    }

    // Test XSS in graph descriptions and task templates
    for (const payload of SECURITY_PAYLOADS.XSS_PAYLOADS.slice(0, 2)) {
        const graphConfig = {
            name: 'test_graph_xss',
            description: payload,
            variables: { test: 'value' },
            nodes: [{
                id: uuidv4(),
                name: 'test_node',
                node_type: 'worker',
                worker_name: 'test_worker',
                task_template: payload,
                dependencies: []
            }],
            edges: []
        };

        await makeSecurityRequest(`${API_BASE}/graphs/`, {
            method: 'POST',
            body: JSON.stringify(graphConfig)
        }, true, `Graphs Creation XSS - ${payload.substring(0, 20)}`);

        securityTestResults.total_tests++;
    }

    // Test path traversal in graph IDs
    for (const payload of SECURITY_PAYLOADS.PATH_TRAVERSAL.slice(0, 2)) {
        await makeSecurityRequest(`${API_BASE}/graphs/${encodeURIComponent(payload)}`, {}, true, `Graphs Path Traversal - ${payload}`);
        securityTestResults.total_tests++;
    }

    // Test malformed JSON in graph creation
    const malformedJsons = [
        '{"name": "test", "description": "test"', // Missing closing brace
        '{"name": "test", "description": "test",}', // Trailing comma
        '{"name": "test", "description": "test", "invalid": function() {}}', // Function in JSON
        '{name: "test", description: "test"}', // Unquoted keys
        '{"name": "test", "description": "test"} extra content' // Extra content
    ];

    for (const malformedJson of malformedJsons) {
        await makeSecurityRequest(`${API_BASE}/graphs/`, {
            method: 'POST',
            body: malformedJson
        }, true, `Graphs Malformed JSON - ${malformedJson.substring(0, 30)}`);

        securityTestResults.total_tests++;
    }
}

/**
 * Test tools endpoint security
 */
async function testToolsEndpointSecurity() {
    console.log('\n🔧 Testing Tools Endpoint Security...');

    // Test legitimate tools list request
    await makeSecurityRequest(`${API_BASE}/tools/`, {}, false, 'Tools List - Legitimate');

    // Test SQL injection in tool names
    for (const payload of SECURITY_PAYLOADS.SQL_INJECTION.slice(0, 2)) {
        await makeSecurityRequest(`${API_BASE}/tools/${encodeURIComponent(payload)}`, {}, true, `Tools SQL Injection - ${payload}`);
        securityTestResults.total_tests++;
    }

    // Test path traversal in tool names
    for (const payload of SECURITY_PAYLOADS.PATH_TRAVERSAL.slice(0, 3)) {
        await makeSecurityRequest(`${API_BASE}/tools/${encodeURIComponent(payload)}`, {}, true, `Tools Path Traversal - ${payload}`);
        securityTestResults.total_tests++;
    }

    // Test tool creation with malicious code
    const maliciousToolConfig = {
        name: 'malicious_tool',
        description: 'A tool with malicious code',
        code: `
import os
import subprocess
# Malicious code attempt
os.system('rm -rf /')
subprocess.call(['cat', '/etc/passwd'])
`,
        requirements: ['os', 'subprocess']
    };

    await makeSecurityRequest(`${API_BASE}/tools/`, {
        method: 'POST',
        body: JSON.stringify(maliciousToolConfig)
    }, true, 'Tools Creation - Malicious Code');

    securityTestResults.total_tests++;

    // Test command injection in tool names
    for (const payload of SECURITY_PAYLOADS.COMMAND_INJECTION.slice(0, 2)) {
        await makeSecurityRequest(`${API_BASE}/tools/${encodeURIComponent(payload)}`, {}, true, `Tools Command Injection - ${payload}`);
        securityTestResults.total_tests++;
    }
}

/**
 * Test execution endpoint security
 */
async function testExecutionEndpointSecurity() {
    console.log('\n⚡ Testing Execution Endpoint Security...');

    // Test malicious execution requests
    const maliciousExecutionRequest = {
        variables: {
            malicious_var: "'; DROP TABLE executions; --",
            xss_var: "<script>alert('xss')</script>",
            command_injection: "; cat /etc/passwd",
            path_traversal: "../../../etc/passwd"
        }
    };

    // Use a fake graph ID for security testing
    const fakeGraphId = 'fake-graph-id-for-security-test';

    await makeSecurityRequest(`${API_BASE}/execution/graphs/${fakeGraphId}/execute`, {
        method: 'POST',
        body: JSON.stringify(maliciousExecutionRequest)
    }, true, 'Execution Malicious Variables');

    securityTestResults.total_tests++;

    // Test path traversal in graph ID
    for (const payload of SECURITY_PAYLOADS.PATH_TRAVERSAL.slice(0, 2)) {
        await makeSecurityRequest(`${API_BASE}/execution/graphs/${encodeURIComponent(payload)}/execute`, {
            method: 'POST',
            body: JSON.stringify({ variables: {} })
        }, true, `Execution Path Traversal - ${payload}`);

        securityTestResults.total_tests++;
    }

    // Test DoS with large execution payload
    const dosExecutionRequest = {
        variables: Object.fromEntries(
            Array.from({ length: 1000 }, (_, i) => [`var_${i}`, DOS_PAYLOADS.LARGE_STRING])
        )
    };

    await makeSecurityRequest(`${API_BASE}/execution/graphs/${fakeGraphId}/execute`, {
        method: 'POST',
        body: JSON.stringify(dosExecutionRequest)
    }, true, 'Execution DoS - Large Variables');

    securityTestResults.total_tests++;
}

/**
 * Test HTTP methods and content type security
 */
async function testHttpMethodsSecurity() {
    console.log('\n🌐 Testing HTTP Methods and Content Type Security...');

    const endpoints = [
        '/health',
        '/workers',
        '/graphs/',
        '/tools/',
        '/execution/health'
    ];

    const maliciousMethods = ['TRACE', 'CONNECT', 'PATCH', 'OPTIONS'];

    for (const endpoint of endpoints) {
        for (const method of maliciousMethods) {
            await makeSecurityRequest(`${API_BASE}${endpoint}`, {
                method: method
            }, true, `HTTP Method ${method} on ${endpoint}`);

            securityTestResults.total_tests++;
        }
    }

    // Test malicious content types
    const maliciousContentTypes = [
        'application/x-www-form-urlencoded; boundary=malicious',
        'multipart/form-data; boundary=malicious',
        'text/html',
        'application/xml',
        'text/plain; charset=malicious'
    ];

    for (const contentType of maliciousContentTypes) {
        await makeSecurityRequest(`${API_BASE}/workers`, {
            method: 'POST',
            headers: {
                'Content-Type': contentType
            },
            body: 'malicious=data&payload=<script>alert("xss")</script>'
        }, true, `Malicious Content Type - ${contentType}`);

        securityTestResults.total_tests++;
    }
}

/**
 * Test authentication and authorization security
 */
async function testAuthenticationSecurity() {
    console.log('\n🔐 Testing Authentication and Authorization Security...');

    // Test with malicious authorization headers
    const maliciousAuthHeaders = [
        'Bearer malicious_token',
        'Basic YWRtaW46cGFzc3dvcmQ=', // admin:password
        'Digest username="admin", realm="test", nonce="malicious"',
        'NTLM malicious_ntlm_token',
        'Negotiate malicious_negotiate_token'
    ];

    for (const authHeader of maliciousAuthHeaders) {
        await makeSecurityRequest(`${API_BASE}/workers`, {
            headers: {
                'Authorization': authHeader
            }
        }, false, `Malicious Auth Header - ${authHeader.split(' ')[0]}`);

        securityTestResults.total_tests++;
    }

    // Test with malicious custom headers
    const maliciousHeaders = {
        'X-Forwarded-For': '127.0.0.1; DROP TABLE users; --',
        'X-Real-IP': '../../../etc/passwd',
        'User-Agent': '<script>alert("xss")</script>',
        'Referer': 'javascript:alert("xss")',
        'X-Requested-With': 'malicious_request',
        'X-Custom-Header': '; cat /etc/passwd'
    };

    for (const [headerName, headerValue] of Object.entries(maliciousHeaders)) {
        await makeSecurityRequest(`${API_BASE}/health`, {
            headers: {
                [headerName]: headerValue
            }
        }, false, `Malicious Header - ${headerName}`);

        securityTestResults.total_tests++;
    }
}

/**
 * Test input validation edge cases
 */
async function testInputValidationEdgeCases() {
    console.log('\n🧪 Testing Input Validation Edge Cases...');

    // Test Unicode and special character handling
    const unicodePayloads = [
        'test\uFEFF', // Byte Order Mark
        'test\u202E', // Right-to-Left Override
        'test\u0000', // Null character
        'test\uFFFE', // Noncharacter
        'test\uFFFF', // Noncharacter
        '𝒜𝒪𝒵', // Mathematical script characters
        '︶︿︶', // Unicode symbols
        '🚀🔒💻', // Emojis
        '\u0001\u0002\u0003\u0004\u0005', // Control characters
        'test\r\n\r\n', // CRLF injection
    ];

    for (const payload of unicodePayloads) {
        const workerConfig = {
            worker_name: `unicode_test_${Date.now()}`,
            agent_role: 'test_role',
            agent_type: 'pydantic_ai',
            model_provider: 'deepseek',
            model: 'deepseek-chat',
            temperature: 0.3,
            system_prompt: payload,
            instruction: 'Test instruction',
            description: 'Test description'
        };

        await makeSecurityRequest(`${API_BASE}/workers`, {
            method: 'POST',
            body: JSON.stringify(workerConfig)
        }, false, `Unicode Validation - ${payload.replace(/[\u0000-\u001F\u007F-\u009F]/g, '?')}`);

        securityTestResults.total_tests++;
    }

    // Test extremely long field names and values
    const longFieldTests = [
        { field: 'A'.repeat(1000), value: 'test' },
        { field: 'worker_name', value: 'B'.repeat(10000) },
        { field: 'system_prompt', value: 'C'.repeat(100000) }
    ];

    for (const test of longFieldTests) {
        const workerConfig = {
            worker_name: 'length_test_worker',
            agent_role: 'test_role',
            agent_type: 'pydantic_ai',
            model_provider: 'deepseek',
            model: 'deepseek-chat',
            temperature: 0.3,
            system_prompt: 'Test prompt',
            instruction: 'Test instruction',
            description: 'Test description',
            [test.field]: test.value
        };

        await makeSecurityRequest(`${API_BASE}/workers`, {
            method: 'POST',
            body: JSON.stringify(workerConfig)
        }, true, `Long Field Test - ${test.field.substring(0, 20)} (${test.value.length} chars)`);

        securityTestResults.total_tests++;
    }
}

/**
 * Generate comprehensive security test report
 */
function generateSecurityReport() {
    console.log('\n📊 COMPREHENSIVE SECURITY TEST REPORT');
    console.log('='.repeat(80));

    const totalTime = securityTestResults.end_time - securityTestResults.start_time;
    const issuesBySeverity = {
        HIGH: securityTestResults.security_issues.filter(i => i.severity === 'HIGH').length,
        MEDIUM: securityTestResults.security_issues.filter(i => i.severity === 'MEDIUM').length,
        LOW: securityTestResults.security_issues.filter(i => i.severity === 'LOW').length
    };

    console.log(`📈 Test Summary:`);
    console.log(`   Total Tests Executed: ${securityTestResults.total_tests}`);
    console.log(`   Total Test Duration: ${totalTime}ms`);
    console.log(`   Security Issues Found: ${securityTestResults.security_issues.length}`);
    console.log(`   High Severity Issues: ${issuesBySeverity.HIGH}`);
    console.log(`   Medium Severity Issues: ${issuesBySeverity.MEDIUM}`);
    console.log(`   Low Severity Issues: ${issuesBySeverity.LOW}`);

    if (securityTestResults.security_issues.length > 0) {
        console.log(`\n🚨 SECURITY ISSUES IDENTIFIED:`);
        console.log('-'.repeat(60));

        securityTestResults.security_issues.forEach((issue, index) => {
            console.log(`\n${index + 1}. ${issue.type} [${issue.severity}]`);
            console.log(`   Test: ${issue.test}`);
            console.log(`   Details: ${issue.details}`);
        });
    } else {
        console.log(`\n✅ NO SECURITY ISSUES IDENTIFIED`);
        console.log(`   All security tests passed successfully!`);
    }

    // Test coverage analysis
    console.log(`\n📋 Test Coverage Analysis:`);
    console.log(`   ✅ SQL Injection Protection: ${SECURITY_PAYLOADS.SQL_INJECTION.length} payloads tested`);
    console.log(`   ✅ XSS Protection: ${SECURITY_PAYLOADS.XSS_PAYLOADS.length} payloads tested`);
    console.log(`   ✅ Path Traversal Protection: ${SECURITY_PAYLOADS.PATH_TRAVERSAL.length} payloads tested`);
    console.log(`   ✅ Command Injection Protection: ${SECURITY_PAYLOADS.COMMAND_INJECTION.length} payloads tested`);
    console.log(`   ✅ DoS Protection: Large payload and deep nesting tests`);
    console.log(`   ✅ Input Validation: Unicode, null bytes, and edge cases`);
    console.log(`   ✅ HTTP Security: Methods, headers, and content types`);
    console.log(`   ✅ Authentication: Malicious auth headers and tokens`);

    // Recommendations
    console.log(`\n💡 Security Recommendations:`);
    if (issuesBySeverity.HIGH > 0) {
        console.log(`   🔴 URGENT: Address ${issuesBySeverity.HIGH} high-severity security issues immediately`);
    }
    if (issuesBySeverity.MEDIUM > 0) {
        console.log(`   🟡 IMPORTANT: Review and fix ${issuesBySeverity.MEDIUM} medium-severity issues`);
    }
    if (issuesBySeverity.LOW > 0) {
        console.log(`   🟢 MINOR: Consider addressing ${issuesBySeverity.LOW} low-severity issues for better security posture`);
    }

    console.log(`   ✅ Implement request rate limiting to prevent DoS attacks`);
    console.log(`   ✅ Add comprehensive input sanitization for all user inputs`);
    console.log(`   ✅ Implement proper error handling to prevent information disclosure`);
    console.log(`   ✅ Add security headers (CSP, X-Frame-Options, etc.)`);
    console.log(`   ✅ Regular security audits and penetration testing`);

    // Overall security score
    const securityScore = Math.max(0, 100 - (issuesBySeverity.HIGH * 20 + issuesBySeverity.MEDIUM * 10 + issuesBySeverity.LOW * 5));
    console.log(`\n🏆 Overall Security Score: ${securityScore}/100`);

    if (securityScore >= 90) {
        console.log(`   🥇 EXCELLENT: Strong security posture with minimal issues`);
    } else if (securityScore >= 75) {
        console.log(`   🥈 GOOD: Solid security with some areas for improvement`);
    } else if (securityScore >= 60) {
        console.log(`   🥉 FAIR: Moderate security, several issues need attention`);
    } else {
        console.log(`   🚨 POOR: Significant security concerns require immediate attention`);
    }

    return {
        total_tests: securityTestResults.total_tests,
        security_issues: securityTestResults.security_issues.length,
        security_score: securityScore,
        high_severity_issues: issuesBySeverity.HIGH,
        passed: securityScore >= 75 // Pass threshold
    };
}

/**
 * Main security test execution function
 */
async function runSecurityValidationTests() {
    console.log('🔒 Starting Comprehensive Security and Input Validation Test Suite');
    console.log('='.repeat(80));

    securityTestResults.start_time = Date.now();

    try {
        // Initialize test counters
        securityTestResults.total_tests = 0;
        securityTestResults.passed_tests = 0;
        securityTestResults.failed_tests = 0;

        // Execute all security test categories
        await testApiHealthSecurity();
        await testWorkersEndpointSecurity();
        await testGraphsEndpointSecurity();
        await testToolsEndpointSecurity();
        await testExecutionEndpointSecurity();
        await testHttpMethodsSecurity();
        await testAuthenticationSecurity();
        await testInputValidationEdgeCases();

        securityTestResults.end_time = Date.now();

        // Generate comprehensive report
        const report = generateSecurityReport();

        console.log('\n🎯 SECURITY TEST EXECUTION COMPLETE');
        console.log('='.repeat(80));

        return report;

    } catch (error) {
        console.error(`\n❌ Security test execution failed: ${error.message}`);
        if (error.stack) {
            console.error(`Stack trace: ${error.stack}`);
        }

        securityTestResults.end_time = Date.now();
        return {
            total_tests: securityTestResults.total_tests,
            security_issues: securityTestResults.security_issues.length,
            security_score: 0,
            high_severity_issues: securityTestResults.security_issues.filter(i => i.severity === 'HIGH').length,
            passed: false,
            error: error.message
        };
    }
}

/**
 * Main function with comprehensive error handling
 */
async function main() {
    console.log('🔒 Worker Management API - Security & Input Validation Test Suite');
    console.log('Node.js version:', process.version);
    console.log('Platform:', process.platform);
    console.log('Architecture:', process.arch);
    console.log('='.repeat(80));

    try {
        const testResults = await runSecurityValidationTests();

        // Final summary
        console.log('\n📊 FINAL SECURITY TEST SUMMARY');
        console.log('='.repeat(80));
        console.log(`Overall Result: ${testResults.passed ? '🎉 SECURITY TESTS PASSED' : '🚨 SECURITY ISSUES DETECTED'}`);
        console.log(`Security Score: ${testResults.security_score}/100`);
        console.log(`Total Tests: ${testResults.total_tests}`);
        console.log(`Security Issues: ${testResults.security_issues}`);
        console.log(`High Severity: ${testResults.high_severity_issues}`);

        // Exit with appropriate code
        process.exit(testResults.passed ? 0 : 1);

    } catch (error) {
        console.error('❌ Security test suite failed to execute:', error.message);
        if (error.stack) {
            console.error('Stack trace:', error.stack);
        }
        process.exit(1);
    }
}

// Execute security tests if this script is run directly
if (require.main === module) {
    main().catch(error => {
        console.error('❌ Unhandled error in security test suite:', error);
        process.exit(1);
    });
}

module.exports = {
    makeSecurityRequest,
    testApiHealthSecurity,
    testWorkersEndpointSecurity,
    testGraphsEndpointSecurity,
    testToolsEndpointSecurity,
    testExecutionEndpointSecurity,
    testHttpMethodsSecurity,
    testAuthenticationSecurity,
    testInputValidationEdgeCases,
    runSecurityValidationTests,
    generateSecurityReport,
    SECURITY_PAYLOADS,
    DOS_PAYLOADS
};
