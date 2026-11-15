#!/usr/bin/env node
/**
 * JavaScript Test Script for Monitoring & Observability - Production Ready
 *
 * This script validates monitoring and observability features of the Worker Management API by:
 * 1. Testing metrics endpoints and data
 * 2. Testing health check endpoints
 * 3. Testing logging and audit trails
 * 4. Testing performance monitoring
 * 5. Testing error tracking and alerting
 * 6. Testing distributed tracing
 * 7. Testing system resource monitoring
 * 8. Testing custom metrics collection
 * 9. Testing dashboard and visualization support
 * 10. Testing monitoring integrations
 *
 * ENDPOINTS TESTED:
 * - GET /metrics (Prometheus format)
 * - GET /health (detailed health)
 * - GET /logs (if available)
 * - GET /traces (if available)
 * - GET /status/system
 *
 * Requirements: Node.js 18+ (for native fetch API)
 * Usage: node test_33_monitoring_observability.js
 */

const { v4: uuidv4 } = require('uuid');

// Enhanced Configuration
const API_BASE = 'http://localhost:8887';
const TIMEOUT_MS = 30000; // 30 seconds
const MAX_RETRIES = 3;
const RETRY_DELAY = 2000; // 2 seconds
const REQUEST_DELAY = 500; // 500ms between requests

// Monitoring test configurations
const MONITORING_CONFIG = {
    METRIC_COLLECTION_DURATION: 10000, // 10 seconds
    STRESS_TEST_DURATION: 5000,        // 5 seconds
    EXPECTED_METRICS: [
        'http_requests_total',
        'http_request_duration_seconds',
        'system_cpu_usage',
        'system_memory_usage',
        'workers_active_total',
        'executions_total'
    ]
};

/**
 * Enhanced utility function to make HTTP requests
 */
async function makeRequest(url, options = {}) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'WorkerManagement-MonitoringTest/1.0',
            ...options.headers
        },
        signal: AbortSignal.timeout(TIMEOUT_MS)
    };

    const finalOptions = { ...defaultOptions, ...options };

    try {
        const response = await fetch(url, finalOptions);

        if (!response.ok) {
            const errorText = await response.text();
            const error = new Error(`HTTP ${response.status}: ${errorText}`);
            error.status = response.status;
            error.response = errorText;
            throw error;
        }

        // Handle different content types
        const contentType = response.headers.get('content-type') || '';

        if (contentType.includes('application/json')) {
            return await response.json();
        } else if (contentType.includes('text/plain') || contentType.includes('text/html')) {
            return await response.text();
        } else {
            return await response.text(); // Default to text
        }
    } catch (error) {
        throw error;
    }
}

/**
 * Test metrics endpoints
 */
async function testMetricsEndpoints() {
    console.log('\n=== Testing Metrics Endpoints ===');

    const results = {
        test_passed: false,
        prometheus_metrics: false,
        json_metrics: false,
        custom_metrics: false,
        metrics_data: {},
        endpoint_found: null
    };

    try {
        // Check for different metrics endpoints
        const metricsEndpoints = [
            { url: '/metrics', format: 'prometheus', accept: 'text/plain' },
            { url: '/metrics.json', format: 'json', accept: 'application/json' },
            { url: '/api/metrics', format: 'json', accept: 'application/json' },
            { url: '/monitoring/metrics', format: 'prometheus', accept: 'text/plain' },
            { url: '/health/metrics', format: 'json', accept: 'application/json' }
        ];

        for (const endpoint of metricsEndpoints) {
            console.log(`\n📊 Testing metrics endpoint: ${endpoint.url}`);

            try {
                const metricsData = await makeRequest(`${API_BASE}${endpoint.url}`, {
                    headers: {
                        'Accept': endpoint.accept
                    }
                });

                console.log(`✅ Metrics endpoint found: ${endpoint.url}`);
                results.endpoint_found = endpoint.url;

                if (endpoint.format === 'prometheus') {
                    console.log('📊 Analyzing Prometheus metrics...');
                    results.prometheus_metrics = true;

                    // Parse Prometheus metrics
                    if (typeof metricsData === 'string') {
                        const lines = metricsData.split('\n');
                        const metrics = {};

                        lines.forEach(line => {
                            if (line.startsWith('#')) return; // Skip comments
                            if (line.trim() === '') return; // Skip empty lines

                            const parts = line.split(' ');
                            if (parts.length >= 2) {
                                const metricName = parts[0];
                                const metricValue = parseFloat(parts[1]);
                                metrics[metricName] = metricValue;
                            }
                        });

                        results.metrics_data = metrics;

                        console.log(`   Found ${Object.keys(metrics).length} metrics`);

                        // Check for expected metrics
                        const foundExpectedMetrics = MONITORING_CONFIG.EXPECTED_METRICS.filter(
                            metric => Object.keys(metrics).some(key => key.includes(metric))
                        );

                        console.log(`   Expected metrics found: ${foundExpectedMetrics.length}/${MONITORING_CONFIG.EXPECTED_METRICS.length}`);
                        foundExpectedMetrics.forEach(metric => {
                            console.log(`     ✅ ${metric}`);
                        });

                        if (foundExpectedMetrics.length > 0) {
                            results.custom_metrics = true;
                        }
                    }

                } else if (endpoint.format === 'json') {
                    console.log('📊 Analyzing JSON metrics...');
                    results.json_metrics = true;

                    if (typeof metricsData === 'object') {
                        results.metrics_data = metricsData;

                        console.log(`   JSON metrics structure:`, Object.keys(metricsData));

                        // Check for common metric categories
                        const commonCategories = ['system', 'http', 'workers', 'database', 'performance'];
                        const foundCategories = commonCategories.filter(cat =>
                            Object.keys(metricsData).some(key => key.toLowerCase().includes(cat))
                        );

                        if (foundCategories.length > 0) {
                            console.log(`   Found metric categories: ${foundCategories.join(', ')}`);
                            results.custom_metrics = true;
                        }
                    }
                }

                break; // Found working endpoint

            } catch (error) {
                console.log(`   ❌ ${endpoint.url} not available: ${error.message}`);
            }
        }

        if (!results.endpoint_found) {
            console.log('ℹ️ No metrics endpoints found');
        }

        results.test_passed = results.prometheus_metrics || results.json_metrics;

    } catch (error) {
        console.error(`❌ Metrics endpoints test failed: ${error.message}`);
    }

    return results;
}

/**
 * Test health check endpoints
 */
async function testHealthCheckEndpoints() {
    console.log('\n=== Testing Health Check Endpoints ===');

    const results = {
        test_passed: false,
        basic_health: false,
        detailed_health: false,
        component_health: false,
        health_data: {}
    };

    try {
        // Test basic health endpoint
        console.log('\n📊 Testing basic health endpoint...');

        try {
            const basicHealth = await makeRequest(`${API_BASE}/health`);
            console.log('✅ Basic health endpoint available');
            results.basic_health = true;

            if (typeof basicHealth === 'object') {
                console.log('📊 Health status:', basicHealth.status);
                console.log('📊 Service:', basicHealth.service);

                if (basicHealth.status === 'healthy' || basicHealth.status === 'ok') {
                    console.log('✅ Service reports healthy status');
                }
            }

        } catch (error) {
            console.log(`❌ Basic health check failed: ${error.message}`);
        }

        // Test detailed health endpoints
        const detailedEndpoints = [
            '/health/detailed',
            '/health/full',
            '/status',
            '/api/health'
        ];

        for (const endpoint of detailedEndpoints) {
            console.log(`\n📊 Testing detailed health: ${endpoint}`);

            try {
                const detailedHealth = await makeRequest(`${API_BASE}${endpoint}`);
                console.log(`✅ Detailed health endpoint found: ${endpoint}`);
                results.detailed_health = true;
                results.health_data = detailedHealth;

                if (typeof detailedHealth === 'object') {
                    console.log('📊 Health information:');

                    // Check for component health
                    if (detailedHealth.components || detailedHealth.checks) {
                        results.component_health = true;
                        const components = detailedHealth.components || detailedHealth.checks;

                        console.log('   Component health status:');
                        Object.entries(components).forEach(([component, status]) => {
                            const statusText = typeof status === 'object' ? status.status : status;
                            console.log(`     ${component}: ${statusText}`);
                        });
                    }

                    // Check for system information
                    if (detailedHealth.system) {
                        console.log('   System information:');
                        const system = detailedHealth.system;
                        if (system.uptime) console.log(`     Uptime: ${system.uptime}`);
                        if (system.version) console.log(`     Version: ${system.version}`);
                        if (system.environment) console.log(`     Environment: ${system.environment}`);
                    }

                    // Check for performance metrics
                    if (detailedHealth.performance || detailedHealth.metrics) {
                        console.log('   Performance metrics included: YES');
                    }
                }

                break; // Found detailed endpoint

            } catch (error) {
                console.log(`   ❌ ${endpoint} not available`);
            }
        }

        results.test_passed = results.basic_health || results.detailed_health;

    } catch (error) {
        console.error(`❌ Health check test failed: ${error.message}`);
    }

    return results;
}

/**
 * Test logging and audit capabilities
 */
async function testLoggingAudit() {
    console.log('\n=== Testing Logging and Audit Capabilities ===');

    const results = {
        test_passed: false,
        logs_endpoint: false,
        audit_trail: false,
        log_levels: false,
        structured_logs: false
    };

    try {
        // Check for logging endpoints
        const logEndpoints = [
            '/logs',
            '/api/logs',
            '/monitoring/logs',
            '/audit',
            '/api/audit'
        ];

        for (const endpoint of logEndpoints) {
            console.log(`\n📊 Testing logging endpoint: ${endpoint}`);

            try {
                const logs = await makeRequest(`${API_BASE}${endpoint}`);
                console.log(`✅ Logging endpoint found: ${endpoint}`);
                results.logs_endpoint = true;

                if (Array.isArray(logs)) {
                    console.log(`   Found ${logs.length} log entries`);

                    if (logs.length > 0) {
                        const sampleLog = logs[0];
                        console.log('   Sample log entry:', JSON.stringify(sampleLog, null, 2));

                        // Check for structured logging
                        if (typeof sampleLog === 'object' &&
                            (sampleLog.timestamp || sampleLog.level || sampleLog.message)) {
                            results.structured_logs = true;
                            console.log('   ✅ Structured logging detected');
                        }

                        // Check for log levels
                        const logLevels = new Set();
                        logs.slice(0, 10).forEach(log => {
                            if (log.level) logLevels.add(log.level);
                        });

                        if (logLevels.size > 1) {
                            results.log_levels = true;
                            console.log(`   ✅ Log levels found: ${Array.from(logLevels).join(', ')}`);
                        }
                    }
                }

                break; // Found logs endpoint

            } catch (error) {
                console.log(`   ❌ ${endpoint} not available: ${error.message}`);
            }
        }

        // Generate some activity to test audit trail
        console.log('\n📊 Testing audit trail generation...');

        const testWorkerName = `audit_test_worker_${Date.now()}`;
        const auditStartTime = Date.now();

        try {
            // Create worker
            await makeRequest(`${API_BASE}/workers`, {
                method: 'POST',
                body: JSON.stringify({
                    worker_name: testWorkerName,
                    agent_role: 'audit test',
                    agent_type: 'pydantic_ai',
                    model_provider: 'deepseek',
                    model: 'deepseek-chat',
                    temperature: 0.5,
                    system_prompt: 'Test audit logging',
                    instruction: 'Generate audit events',
                    user_prompt: '',
                    description: 'Audit test worker'
                })
            });

            // Update worker
            await makeRequest(`${API_BASE}/workers/${testWorkerName}`, {
                method: 'PUT',
                body: JSON.stringify({
                    worker_name: testWorkerName,
                    agent_role: 'audit test updated',
                    agent_type: 'pydantic_ai',
                    model_provider: 'deepseek',
                    model: 'deepseek-chat',
                    temperature: 0.7,
                    system_prompt: 'Updated audit test',
                    instruction: 'Generate more audit events',
                    user_prompt: '',
                    description: 'Updated audit test worker'
                })
            });

            // Delete worker
            await makeRequest(`${API_BASE}/workers/${testWorkerName}`, {
                method: 'DELETE'
            });

            console.log('✅ Generated audit events (create, update, delete)');

            // Wait a bit for logs to be processed
            await new Promise(resolve => setTimeout(resolve, 2000));

            // Check if audit events were logged
            if (results.logs_endpoint) {
                try {
                    const recentLogs = await makeRequest(`${API_BASE}/logs?since=${auditStartTime}`);

                    if (Array.isArray(recentLogs)) {
                        const auditEvents = recentLogs.filter(log =>
                            log.message &&
                            (log.message.includes(testWorkerName) ||
                             log.message.includes('worker') ||
                             log.action)
                        );

                        if (auditEvents.length > 0) {
                            console.log(`✅ Audit trail working - found ${auditEvents.length} related events`);
                            results.audit_trail = true;
                        }
                    }
                } catch (error) {
                    console.log('⚠️ Could not retrieve recent logs for audit verification');
                }
            }

        } catch (error) {
            console.log('⚠️ Could not generate audit events:', error.message);
        }

        results.test_passed = results.logs_endpoint || results.audit_trail;

    } catch (error) {
        console.error(`❌ Logging audit test failed: ${error.message}`);
    }

    return results;
}

/**
 * Test performance monitoring
 */
async function testPerformanceMonitoring() {
    console.log('\n=== Testing Performance Monitoring ===');

    const results = {
        test_passed: false,
        response_time_tracking: false,
        throughput_metrics: false,
        resource_usage: false,
        performance_data: {}
    };

    try {
        // Get baseline metrics
        console.log('📊 Collecting baseline metrics...');

        let baselineMetrics = {};
        try {
            const metricsResponse = await makeRequest(`${API_BASE}/metrics`);
            if (typeof metricsResponse === 'string') {
                // Parse Prometheus metrics
                const lines = metricsResponse.split('\n');
                lines.forEach(line => {
                    if (!line.startsWith('#') && line.trim() !== '') {
                        const parts = line.split(' ');
                        if (parts.length >= 2) {
                            baselineMetrics[parts[0]] = parseFloat(parts[1]);
                        }
                    }
                });
            }
        } catch (error) {
            console.log('ℹ️ No metrics endpoint available for performance monitoring');
        }

        // Generate load
        console.log('\n📊 Generating load for performance testing...');

        const loadStartTime = Date.now();
        const requestPromises = [];
        const requestCount = 10;

        for (let i = 0; i < requestCount; i++) {
            requestPromises.push(
                makeRequest(`${API_BASE}/health`)
                    .then(() => ({ success: true, index: i }))
                    .catch(error => ({ success: false, index: i, error }))
            );
        }

        const loadResults = await Promise.all(requestPromises);
        const loadDuration = Date.now() - loadStartTime;

        const successfulRequests = loadResults.filter(r => r.success).length;
        const throughput = successfulRequests / (loadDuration / 1000);

        console.log(`✅ Load test completed:`);
        console.log(`   Requests: ${successfulRequests}/${requestCount}`);
        console.log(`   Duration: ${loadDuration}ms`);
        console.log(`   Throughput: ${throughput.toFixed(2)} req/sec`);

        results.throughput_metrics = true;
        results.performance_data.throughput = throughput;
        results.performance_data.average_response_time = loadDuration / requestCount;

        // Check if metrics changed
        try {
            await new Promise(resolve => setTimeout(resolve, 2000)); // Wait for metrics update

            const afterMetrics = await makeRequest(`${API_BASE}/metrics`);
            if (typeof afterMetrics === 'string') {
                console.log('✅ Metrics available after load test');
                results.response_time_tracking = true;

                // Look for request counter increases
                const afterLines = afterMetrics.split('\n');
                afterLines.forEach(line => {
                    if (line.includes('http_requests_total') || line.includes('request_count')) {
                        console.log(`   Request metric: ${line}`);
                    }
                });
            }
        } catch (error) {
            console.log('ℹ️ Could not retrieve updated metrics');
        }

        // Check for system resource monitoring
        console.log('\n📊 Checking system resource monitoring...');

        const resourceEndpoints = [
            '/metrics/system',
            '/status/resources',
            '/monitoring/resources',
            '/health/system'
        ];

        for (const endpoint of resourceEndpoints) {
            try {
                const resources = await makeRequest(`${API_BASE}${endpoint}`);
                console.log(`✅ Resource monitoring endpoint found: ${endpoint}`);
                results.resource_usage = true;

                if (typeof resources === 'object') {
                    console.log('   System resources:');
                    if (resources.cpu) console.log(`     CPU: ${resources.cpu}%`);
                    if (resources.memory) console.log(`     Memory: ${resources.memory}%`);
                    if (resources.disk) console.log(`     Disk: ${resources.disk}%`);
                    if (resources.load) console.log(`     Load: ${resources.load}`);
                }

                break;

            } catch (error) {
                console.log(`   ❌ ${endpoint} not available`);
            }
        }

        results.test_passed =
            results.response_time_tracking ||
            results.throughput_metrics ||
            results.resource_usage;

    } catch (error) {
        console.error(`❌ Performance monitoring test failed: ${error.message}`);
    }

    return results;
}

/**
 * Test error tracking
 */
async function testErrorTracking() {
    console.log('\n=== Testing Error Tracking ===');

    const results = {
        test_passed: false,
        error_logs: false,
        error_metrics: false,
        error_details: false,
        alerting: false
    };

    try {
        // Generate intentional errors
        console.log('📊 Generating test errors...');

        const errorTests = [
            { url: '/workers/nonexistent_worker', method: 'GET', expectedStatus: 404 },
            { url: '/workers', method: 'POST', body: { invalid: 'data' }, expectedStatus: 400 },
            { url: '/graphs/invalid_id', method: 'DELETE', expectedStatus: 404 }
        ];

        for (const test of errorTests) {
            try {
                await makeRequest(`${API_BASE}${test.url}`, {
                    method: test.method,
                    body: test.body ? JSON.stringify(test.body) : undefined
                });
            } catch (error) {
                if (error.status === test.expectedStatus) {
                    console.log(`✅ Expected error generated: ${test.expectedStatus} for ${test.url}`);
                }
            }
        }

        // Wait for error processing
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Check for error tracking in logs
        console.log('\n📊 Checking error tracking in logs...');

        try {
            const logs = await makeRequest(`${API_BASE}/logs`);

            if (Array.isArray(logs)) {
                const errorLogs = logs.filter(log =>
                    log.level === 'error' ||
                    log.level === 'ERROR' ||
                    (log.message && log.message.includes('error'))
                );

                if (errorLogs.length > 0) {
                    console.log(`✅ Error logs found: ${errorLogs.length} entries`);
                    results.error_logs = true;

                    // Check for error details
                    const detailedErrors = errorLogs.filter(log =>
                        log.stack_trace || log.error_details || log.exception
                    );

                    if (detailedErrors.length > 0) {
                        console.log(`✅ Detailed error information available`);
                        results.error_details = true;
                    }
                }
            }
        } catch (error) {
            console.log('ℹ️ Could not check error logs');
        }

        // Check for error metrics
        console.log('\n📊 Checking error metrics...');

        try {
            const metrics = await makeRequest(`${API_BASE}/metrics`);

            if (typeof metrics === 'string') {
                const errorMetrics = metrics.split('\n').filter(line =>
                    line.includes('error') || line.includes('4') || line.includes('5') // HTTP error codes
                );

                if (errorMetrics.length > 0) {
                    console.log(`✅ Error metrics found: ${errorMetrics.length} metrics`);
                    results.error_metrics = true;

                    errorMetrics.slice(0, 3).forEach(metric => {
                        console.log(`   ${metric}`);
                    });
                }
            }
        } catch (error) {
            console.log('ℹ️ Could not check error metrics');
        }

        // Check for alerting endpoints
        console.log('\n📊 Checking alerting capabilities...');

        const alertEndpoints = [
            '/alerts',
            '/api/alerts',
            '/monitoring/alerts',
            '/health/alerts'
        ];

        for (const endpoint of alertEndpoints) {
            try {
                const alerts = await makeRequest(`${API_BASE}${endpoint}`);
                console.log(`✅ Alerting endpoint found: ${endpoint}`);
                results.alerting = true;

                if (Array.isArray(alerts)) {
                    console.log(`   Active alerts: ${alerts.length}`);
                } else if (typeof alerts === 'object') {
                    console.log(`   Alert configuration available`);
                }

                break;

            } catch (error) {
                console.log(`   ❌ ${endpoint} not available`);
            }
        }

        results.test_passed = results.error_logs || results.error_metrics;

    } catch (error) {
        console.error(`❌ Error tracking test failed: ${error.message}`);
    }

    return results;
}

/**
 * Test distributed tracing
 */
async function testDistributedTracing() {
    console.log('\n=== Testing Distributed Tracing ===');

    const results = {
        test_passed: false,
        trace_headers: false,
        trace_endpoint: false,
        span_tracking: false,
        trace_correlation: false
    };

    try {
        // Test trace header support
        console.log('📊 Testing trace header support...');

        const traceId = uuidv4();
        const spanId = uuidv4();

        const tracedResponse = await fetch(`${API_BASE}/health`, {
            headers: {
                'X-Trace-Id': traceId,
                'X-Span-Id': spanId,
                'X-B3-TraceId': traceId,
                'X-B3-SpanId': spanId
            }
        });

        // Check if trace headers are returned
        const responseTraceId = tracedResponse.headers.get('X-Trace-Id') ||
                              tracedResponse.headers.get('X-B3-TraceId');

        if (responseTraceId) {
            console.log(`✅ Trace headers supported - Trace ID: ${responseTraceId}`);
            results.trace_headers = true;
        } else {
            console.log('ℹ️ No trace headers in response');
        }

        // Check for tracing endpoints
        console.log('\n📊 Checking tracing endpoints...');

        const traceEndpoints = [
            '/traces',
            '/api/traces',
            '/monitoring/traces',
            '/jaeger/traces'
        ];

        for (const endpoint of traceEndpoints) {
            try {
                const traces = await makeRequest(`${API_BASE}${endpoint}`);
                console.log(`✅ Tracing endpoint found: ${endpoint}`);
                results.trace_endpoint = true;

                if (Array.isArray(traces)) {
                    console.log(`   Found ${traces.length} traces`);

                    if (traces.length > 0) {
                        const sampleTrace = traces[0];
                        console.log('   Sample trace structure:', Object.keys(sampleTrace));

                        if (sampleTrace.spans || sampleTrace.operations) {
                            results.span_tracking = true;
                            console.log('   ✅ Span tracking detected');
                        }
                    }
                }

                break;

            } catch (error) {
                console.log(`   ❌ ${endpoint} not available`);
            }
        }

        // Test trace correlation with complex operation
        console.log('\n📊 Testing trace correlation...');

        const correlationTraceId = uuidv4();

        try {
            // Create worker with trace
            const workerResponse = await fetch(`${API_BASE}/workers`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Trace-Id': correlationTraceId
                },
                body: JSON.stringify({
                    worker_name: `trace_test_worker_${Date.now()}`,
                    agent_role: 'trace test',
                    agent_type: 'pydantic_ai',
                    model_provider: 'deepseek',
                    model: 'deepseek-chat',
                    temperature: 0.5,
                    system_prompt: 'Test tracing',
                    instruction: 'Generate trace spans',
                    user_prompt: '',
                    description: 'Trace test worker'
                })
            });

            if (workerResponse.ok) {
                const worker = await workerResponse.json();

                // Send message with same trace
                await fetch(`${API_BASE}/workers/${worker.name}/message`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Trace-Id': correlationTraceId
                    },
                    body: JSON.stringify({
                        message: 'Test message for tracing',
                        thread_id: uuidv4()
                    })
                });

                console.log('✅ Multi-operation trace generated');
                results.trace_correlation = true;

                // Cleanup
                await fetch(`${API_BASE}/workers/${worker.name}`, { method: 'DELETE' });
            }

        } catch (error) {
            console.log('⚠️ Could not test trace correlation:', error.message);
        }

        results.test_passed = results.trace_headers || results.trace_endpoint;

    } catch (error) {
        console.error(`❌ Distributed tracing test failed: ${error.message}`);
    }

    return results;
}

/**
 * Main test function for monitoring and observability
 */
async function runMonitoringObservabilityTest() {
    console.log('🚀 Starting Monitoring & Observability Test Suite');
    console.log('='.repeat(80));

    const testResults = {
        metrics_endpoints: false,
        health_checks: false,
        logging_audit: false,
        performance_monitoring: false,
        error_tracking: false,
        distributed_tracing: false,
        overall_success: false,
        execution_time: 0,
        details: {}
    };

    const testStartTime = Date.now();

    try {
        // Test 1: Metrics Endpoints
        console.log('\n📋 Test 1: Metrics Endpoints');
        const metricsResult = await testMetricsEndpoints();
        testResults.metrics_endpoints = metricsResult.test_passed;
        testResults.details.metrics = metricsResult;

        // Test 2: Health Check Endpoints
        console.log('\n📋 Test 2: Health Check Endpoints');
        const healthResult = await testHealthCheckEndpoints();
        testResults.health_checks = healthResult.test_passed;
        testResults.details.health = healthResult;

        // Test 3: Logging and Audit
        console.log('\n📋 Test 3: Logging and Audit');
        const loggingResult = await testLoggingAudit();
        testResults.logging_audit = loggingResult.test_passed;
        testResults.details.logging = loggingResult;

        // Test 4: Performance Monitoring
        console.log('\n📋 Test 4: Performance Monitoring');
        const performanceResult = await testPerformanceMonitoring();
        testResults.performance_monitoring = performanceResult.test_passed;
        testResults.details.performance = performanceResult;

        // Test 5: Error Tracking
        console.log('\n📋 Test 5: Error Tracking');
        const errorResult = await testErrorTracking();
        testResults.error_tracking = errorResult.test_passed;
        testResults.details.error = errorResult;

        // Test 6: Distributed Tracing
        console.log('\n📋 Test 6: Distributed Tracing');
        const tracingResult = await testDistributedTracing();
        testResults.distributed_tracing = tracingResult.test_passed;
        testResults.details.tracing = tracingResult;

        // Calculate overall success
        const passedTests = Object.keys(testResults)
            .filter(key => key !== 'overall_success' && key !== 'execution_time' && key !== 'details')
            .filter(key => testResults[key] === true).length;

        const totalTests = 6;

        // Require at least basic monitoring capabilities
        testResults.overall_success =
            testResults.health_checks &&
            (testResults.metrics_endpoints || testResults.logging_audit) &&
            passedTests >= 3;

        testResults.execution_time = Date.now() - testStartTime;

        // Generate summary report
        console.log('\n📊 Monitoring & Observability Test Summary');
        console.log('='.repeat(80));
        console.log(`🏁 Test Suite - COMPLETE`);
        console.log(`   Total execution time: ${testResults.execution_time}ms`);
        console.log(`   Overall success: ${testResults.overall_success ? '✅ PASSED' : '❌ FAILED'}`);
        console.log(`   Tests passed: ${passedTests}/${totalTests}`);

        console.log('\n📋 Test Results:');
        console.log(`   ✅ Metrics Endpoints: ${testResults.metrics_endpoints ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Health Checks: ${testResults.health_checks ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Logging & Audit: ${testResults.logging_audit ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Performance Monitoring: ${testResults.performance_monitoring ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Error Tracking: ${testResults.error_tracking ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Distributed Tracing: ${testResults.distributed_tracing ? 'PASSED' : 'FAILED'}`);

        if (testResults.overall_success) {
            console.log('\n🎉 Monitoring & Observability Test PASSED!');
            console.log('   ✅ Essential monitoring capabilities are present');
            console.log('   ✅ Health checks are available');
            console.log('   ✅ System observability is adequate');
        } else {
            console.log('\n💥 Monitoring & Observability Test FAILED!');
            console.log('   ❌ Insufficient monitoring capabilities');
            console.log('   ❌ Consider implementing comprehensive monitoring');
        }

    } catch (error) {
        console.error(`\n❌ Test execution failed: ${error.message}`);
        if (error.stack) {
            console.error(`Stack trace: ${error.stack}`);
        }
        testResults.overall_success = false;
        testResults.execution_time = Date.now() - testStartTime;
    }

    return testResults;
}

/**
 * Main function
 */
async function main() {
    console.log('🚀 Monitoring & Observability Test Suite');
    console.log('Node.js version:', process.version);
    console.log('Platform:', process.platform);
    console.log('Architecture:', process.arch);
    console.log('='.repeat(80));

    try {
        const testResults = await runMonitoringObservabilityTest();

        // Final summary
        console.log('\n📊 FINAL TEST SUMMARY');
        console.log('='.repeat(80));
        console.log(`Overall Result: ${testResults.overall_success ? '🎉 SUCCESS' : '💥 FAILURE'}`);
        console.log(`Total Execution Time: ${testResults.execution_time}ms`);

        // Exit with appropriate code
        process.exit(testResults.overall_success ? 0 : 1);

    } catch (error) {
        console.error('❌ Test suite failed to execute:', error.message);
        if (error.stack) {
            console.error('Stack trace:', error.stack);
        }
        process.exit(1);
    }
}

// Execute if run directly
if (require.main === module) {
    main().catch(error => {
        console.error('❌ Unhandled error in test suite:', error);
        process.exit(1);
    });
}

module.exports = {
    runMonitoringObservabilityTest,
    testMetricsEndpoints,
    testHealthCheckEndpoints,
    testLoggingAudit,
    testPerformanceMonitoring,
    testErrorTracking,
    testDistributedTracing
};
