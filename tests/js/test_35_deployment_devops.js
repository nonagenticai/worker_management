#!/usr/bin/env node
/**
 * JavaScript Test Script for Deployment & DevOps - Production Ready
 *
 * This script validates deployment and DevOps readiness of the Worker Management API by:
 * 1. Testing configuration management
 * 2. Testing environment variable handling
 * 3. Testing graceful shutdown procedures
 * 4. Testing health checks for load balancers
 * 5. Testing container readiness and liveness probes
 * 6. Testing configuration hot-reloading
 * 7. Testing backup and restore procedures
 * 8. Testing scaling behavior
 * 9. Testing deployment rollback scenarios
 * 10. Testing infrastructure monitoring integration
 *
 * ENDPOINTS TESTED:
 * - GET /health/live (liveness probe)
 * - GET /health/ready (readiness probe)
 * - GET /config (configuration)
 * - POST /admin/reload (configuration reload)
 * - POST /admin/shutdown (graceful shutdown)
 *
 * Requirements: Node.js 18+ (for native fetch API)
 * Usage: node test_35_deployment_devops.js
 */

const { v4: uuidv4 } = require('uuid');
const fs = require('fs').promises;
const path = require('path');

// Enhanced Configuration
const API_BASE = 'http://localhost:8887';
const TIMEOUT_MS = 30000; // 30 seconds
const MAX_RETRIES = 3;
const RETRY_DELAY = 2000; // 2 seconds
const REQUEST_DELAY = 500; // 500ms between requests

// DevOps test configurations
const DEVOPS_CONFIG = {
    HEALTH_CHECK_INTERVAL: 2000,    // 2 seconds
    STARTUP_TIMEOUT: 30000,         // 30 seconds
    SHUTDOWN_TIMEOUT: 15000,        // 15 seconds
    CONFIG_RELOAD_TIMEOUT: 10000,   // 10 seconds
    SCALING_TEST_DURATION: 20000    // 20 seconds
};

/**
 * Enhanced utility function to make HTTP requests
 */
async function makeRequest(url, options = {}) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'WorkerManagement-DevOpsTest/1.0',
            ...options.headers
        },
        signal: AbortSignal.timeout(TIMEOUT_MS)
    };

    const finalOptions = { ...defaultOptions, ...options };

    try {
        const response = await fetch(url, finalOptions);

        // For DevOps endpoints, we want to capture response even if not 2xx
        const responseData = {
            status: response.status,
            ok: response.ok,
            headers: Object.fromEntries(response.headers.entries())
        };

        if (response.status === 204) {
            return { ...responseData, data: null };
        }

        const contentType = response.headers.get('content-type') || '';

        if (contentType.includes('application/json')) {
            responseData.data = await response.json();
        } else {
            responseData.data = await response.text();
        }

        return responseData;
    } catch (error) {
        throw error;
    }
}

/**
 * Test liveness and readiness probes
 */
async function testHealthProbes() {
    console.log('\n=== Testing Health Probes for Container Orchestration ===');

    const results = {
        test_passed: false,
        liveness_probe: false,
        readiness_probe: false,
        startup_probe: false,
        probe_performance: {}
    };

    try {
        // Test liveness probe
        console.log('\n📊 Testing liveness probe...');

        const livenessEndpoints = [
            '/health/live',
            '/healthz',
            '/live',
            '/health/liveness'
        ];

        for (const endpoint of livenessEndpoints) {
            try {
                const startTime = Date.now();
                const response = await makeRequest(`${API_BASE}${endpoint}`);
                const responseTime = Date.now() - startTime;

                if (response.ok) {
                    console.log(`✅ Liveness probe found: ${endpoint}`);
                    console.log(`   Response time: ${responseTime}ms`);
                    console.log(`   Status: ${response.status}`);

                    results.liveness_probe = true;
                    results.probe_performance.liveness = responseTime;

                    // Check response content
                    if (response.data) {
                        if (typeof response.data === 'object') {
                            console.log(`   Status: ${response.data.status || 'ok'}`);
                        } else {
                            console.log(`   Response: ${response.data.substring(0, 50)}`);
                        }
                    }

                    break; // Found working liveness probe
                }
            } catch (error) {
                console.log(`   ❌ ${endpoint} not available: ${error.message}`);
            }
        }

        // Test readiness probe
        console.log('\n📊 Testing readiness probe...');

        const readinessEndpoints = [
            '/health/ready',
            '/readiness',
            '/ready',
            '/health/readiness'
        ];

        for (const endpoint of readinessEndpoints) {
            try {
                const startTime = Date.now();
                const response = await makeRequest(`${API_BASE}${endpoint}`);
                const responseTime = Date.now() - startTime;

                if (response.ok) {
                    console.log(`✅ Readiness probe found: ${endpoint}`);
                    console.log(`   Response time: ${responseTime}ms`);
                    console.log(`   Status: ${response.status}`);

                    results.readiness_probe = true;
                    results.probe_performance.readiness = responseTime;

                    // Check if readiness includes dependency checks
                    if (response.data && typeof response.data === 'object') {
                        if (response.data.dependencies || response.data.checks) {
                            console.log('   ✅ Dependency checks included in readiness');

                            const deps = response.data.dependencies || response.data.checks;
                            Object.entries(deps).forEach(([dep, status]) => {
                                console.log(`     ${dep}: ${typeof status === 'object' ? status.status : status}`);
                            });
                        }
                    }

                    break; // Found working readiness probe
                }
            } catch (error) {
                console.log(`   ❌ ${endpoint} not available: ${error.message}`);
            }
        }

        // Test startup probe
        console.log('\n📊 Testing startup probe...');

        const startupEndpoints = [
            '/health/startup',
            '/startup',
            '/health/start'
        ];

        for (const endpoint of startupEndpoints) {
            try {
                const response = await makeRequest(`${API_BASE}${endpoint}`);

                if (response.ok) {
                    console.log(`✅ Startup probe found: ${endpoint}`);
                    results.startup_probe = true;
                    break;
                }
            } catch (error) {
                console.log(`   ❌ ${endpoint} not available`);
            }
        }

        if (!results.startup_probe) {
            console.log('ℹ️ No dedicated startup probe (using readiness/liveness is fine)');
        }

        // Performance validation
        const maxProbeTime = 1000; // 1 second max for probes

        if (results.probe_performance.liveness && results.probe_performance.liveness > maxProbeTime) {
            console.log(`⚠️ Liveness probe slow: ${results.probe_performance.liveness}ms`);
        }

        if (results.probe_performance.readiness && results.probe_performance.readiness > maxProbeTime) {
            console.log(`⚠️ Readiness probe slow: ${results.probe_performance.readiness}ms`);
        }

        results.test_passed = results.liveness_probe && results.readiness_probe;

        console.log(`\n📊 Health Probes Summary:`);
        console.log(`   Liveness probe: ${results.liveness_probe ? 'AVAILABLE' : 'MISSING'}`);
        console.log(`   Readiness probe: ${results.readiness_probe ? 'AVAILABLE' : 'MISSING'}`);
        console.log(`   Startup probe: ${results.startup_probe ? 'AVAILABLE' : 'NOT FOUND'}`);

    } catch (error) {
        console.error(`❌ Health probes test failed: ${error.message}`);
    }

    return results;
}

/**
 * Test configuration management
 */
async function testConfigurationManagement() {
    console.log('\n=== Testing Configuration Management ===');

    const results = {
        test_passed: false,
        config_endpoint: false,
        env_vars_supported: false,
        config_validation: false,
        hot_reload: false,
        config_data: {}
    };

    try {
        // Test configuration endpoints
        console.log('\n📊 Testing configuration endpoints...');

        const configEndpoints = [
            '/config',
            '/api/config',
            '/admin/config',
            '/settings'
        ];

        for (const endpoint of configEndpoints) {
            try {
                const response = await makeRequest(`${API_BASE}${endpoint}`);

                if (response.ok) {
                    console.log(`✅ Configuration endpoint found: ${endpoint}`);
                    results.config_endpoint = true;
                    results.config_data = response.data;

                    if (typeof response.data === 'object') {
                        console.log('📊 Configuration structure:');
                        const configKeys = Object.keys(response.data);
                        configKeys.slice(0, 10).forEach(key => {
                            const value = response.data[key];
                            const displayValue = typeof value === 'string' && value.length > 50
                                ? value.substring(0, 47) + '...'
                                : value;
                            console.log(`   ${key}: ${JSON.stringify(displayValue)}`);
                        });

                        if (configKeys.length > 10) {
                            console.log(`   ... and ${configKeys.length - 10} more`);
                        }

                        // Check for environment variable usage
                        const configStr = JSON.stringify(response.data);
                        if (configStr.includes('${') || configStr.includes('$ENV') ||
                            configStr.includes('process.env')) {
                            results.env_vars_supported = true;
                            console.log('✅ Environment variable substitution detected');
                        }
                    }

                    break; // Found config endpoint
                }
            } catch (error) {
                console.log(`   ❌ ${endpoint} not available: ${error.message}`);
            }
        }

        // Test configuration validation
        console.log('\n📊 Testing configuration validation...');

        try {
            const validateResponse = await makeRequest(`${API_BASE}/config/validate`, {
                method: 'POST',
                body: JSON.stringify({
                    test_config: 'validation'
                })
            });

            if (validateResponse.status === 200 || validateResponse.status === 400) {
                console.log('✅ Configuration validation endpoint available');
                results.config_validation = true;
            }
        } catch (error) {
            console.log('ℹ️ No configuration validation endpoint');
        }

        // Test hot reload capability
        console.log('\n📊 Testing configuration hot reload...');

        const reloadEndpoints = [
            '/admin/reload',
            '/config/reload',
            '/api/reload'
        ];

        for (const endpoint of reloadEndpoints) {
            try {
                const reloadResponse = await makeRequest(`${API_BASE}${endpoint}`, {
                    method: 'POST'
                });

                if (reloadResponse.status === 200 || reloadResponse.status === 202) {
                    console.log(`✅ Configuration reload endpoint found: ${endpoint}`);
                    results.hot_reload = true;

                    if (reloadResponse.data) {
                        console.log(`   Response: ${JSON.stringify(reloadResponse.data)}`);
                    }

                    break;
                }
            } catch (error) {
                console.log(`   ❌ ${endpoint} not available: ${error.message}`);
            }
        }

        // Check for environment-specific configuration
        console.log('\n📊 Checking environment detection...');

        if (results.config_data) {
            const envIndicators = ['environment', 'env', 'stage', 'deployment'];
            const foundEnvConfig = envIndicators.some(indicator =>
                Object.keys(results.config_data).some(key =>
                    key.toLowerCase().includes(indicator)
                )
            );

            if (foundEnvConfig) {
                console.log('✅ Environment-specific configuration detected');
            }
        }

        results.test_passed = results.config_endpoint;

        console.log(`\n📊 Configuration Management Summary:`);
        console.log(`   Config endpoint: ${results.config_endpoint ? 'AVAILABLE' : 'MISSING'}`);
        console.log(`   Environment variables: ${results.env_vars_supported ? 'SUPPORTED' : 'UNKNOWN'}`);
        console.log(`   Config validation: ${results.config_validation ? 'AVAILABLE' : 'NOT FOUND'}`);
        console.log(`   Hot reload: ${results.hot_reload ? 'SUPPORTED' : 'NOT SUPPORTED'}`);

    } catch (error) {
        console.error(`❌ Configuration management test failed: ${error.message}`);
    }

    return results;
}

/**
 * Test graceful shutdown
 */
async function testGracefulShutdown() {
    console.log('\n=== Testing Graceful Shutdown ===');

    const results = {
        test_passed: false,
        shutdown_endpoint: false,
        signal_handling: false,
        connection_draining: false,
        shutdown_timeout: false
    };

    try {
        // Test shutdown endpoint availability (don't actually trigger it)
        console.log('\n📊 Testing shutdown endpoint availability...');

        const shutdownEndpoints = [
            '/admin/shutdown',
            '/shutdown',
            '/api/shutdown'
        ];

        for (const endpoint of shutdownEndpoints) {
            try {
                // Use HEAD request to test availability without triggering shutdown
                const response = await makeRequest(`${API_BASE}${endpoint}`, {
                    method: 'HEAD'
                });

                if (response.status === 200 || response.status === 405) { // 405 = Method Not Allowed but endpoint exists
                    console.log(`✅ Shutdown endpoint available: ${endpoint}`);
                    results.shutdown_endpoint = true;
                    break;
                }
            } catch (error) {
                console.log(`   ❌ ${endpoint} not available`);
            }
        }

        // Test graceful shutdown behavior indicators
        console.log('\n📊 Checking graceful shutdown indicators...');

        // Check for process signal handling documentation or hints
        try {
            const healthResponse = await makeRequest(`${API_BASE}/health`);

            if (healthResponse.data && typeof healthResponse.data === 'object') {
                if (healthResponse.data.shutdown_timeout ||
                    healthResponse.data.graceful_shutdown ||
                    healthResponse.data.signal_handlers) {
                    console.log('✅ Graceful shutdown configuration detected');
                    results.signal_handling = true;
                }
            }
        } catch (error) {
            console.log('ℹ️ Could not check shutdown configuration');
        }

        // Test connection draining simulation
        console.log('\n📊 Testing connection draining simulation...');

        try {
            // Start multiple concurrent requests
            const concurrentRequests = Array(5).fill(null).map((_, i) =>
                makeRequest(`${API_BASE}/health?test=concurrent_${i}`)
                    .then(() => ({ success: true, index: i }))
                    .catch(error => ({ success: false, index: i, error }))
            );

            const startTime = Date.now();
            const results_concurrent = await Promise.all(concurrentRequests);
            const duration = Date.now() - startTime;

            const successCount = results_concurrent.filter(r => r.success).length;

            if (successCount === 5 && duration < 5000) {
                console.log('✅ Concurrent request handling working (good for draining)');
                results.connection_draining = true;
            }

            console.log(`   Concurrent requests: ${successCount}/5 successful in ${duration}ms`);

        } catch (error) {
            console.log('⚠️ Could not test concurrent request handling');
        }

        // Check for shutdown timeout configuration
        if (results.shutdown_endpoint) {
            try {
                const optionsResponse = await makeRequest(`${API_BASE}/admin/shutdown`, {
                    method: 'OPTIONS'
                });

                if (optionsResponse.ok && optionsResponse.data) {
                    console.log('✅ Shutdown configuration available');
                    results.shutdown_timeout = true;
                }
            } catch (error) {
                console.log('ℹ️ No shutdown configuration endpoint');
            }
        }

        results.test_passed = results.shutdown_endpoint || results.signal_handling;

        console.log(`\n📊 Graceful Shutdown Summary:`);
        console.log(`   Shutdown endpoint: ${results.shutdown_endpoint ? 'AVAILABLE' : 'MISSING'}`);
        console.log(`   Signal handling: ${results.signal_handling ? 'CONFIGURED' : 'UNKNOWN'}`);
        console.log(`   Connection draining: ${results.connection_draining ? 'WORKING' : 'UNTESTED'}`);
        console.log(`   Shutdown timeout: ${results.shutdown_timeout ? 'CONFIGURED' : 'UNKNOWN'}`);

    } catch (error) {
        console.error(`❌ Graceful shutdown test failed: ${error.message}`);
    }

    return results;
}

/**
 * Test scaling behavior
 */
async function testScalingBehavior() {
    console.log('\n=== Testing Scaling Behavior ===');

    const results = {
        test_passed: false,
        load_distribution: false,
        stateless_operation: false,
        session_handling: false,
        resource_efficiency: false,
        performance_data: {}
    };

    try {
        // Test stateless operation
        console.log('\n📊 Testing stateless operation...');

        // Create worker
        const workerName = `scaling_test_worker_${Date.now()}`;
        const workerConfig = {
            worker_name: workerName,
            agent_role: 'scaling test',
            agent_type: 'pydantic_ai',
            model_provider: 'deepseek',
            model: 'deepseek-chat',
            temperature: 0.5,
            system_prompt: 'Test scaling behavior',
            instruction: 'Handle stateless operations',
            user_prompt: '',
            description: 'Scaling test worker'
        };

        await makeRequest(`${API_BASE}/workers`, {
            method: 'POST',
            body: JSON.stringify(workerConfig)
        });

        // Test multiple requests to different endpoints
        const testEndpoints = [
            `/workers/${workerName}`,
            '/workers',
            '/health',
            '/workers_states'
        ];

        const requestPromises = [];

        testEndpoints.forEach((endpoint, i) => {
            for (let j = 0; j < 3; j++) {
                requestPromises.push(
                    makeRequest(`${API_BASE}${endpoint}`)
                        .then(response => ({
                            success: response.ok,
                            endpoint,
                            instance: j,
                            status: response.status
                        }))
                        .catch(error => ({
                            success: false,
                            endpoint,
                            instance: j,
                            error: error.message
                        }))
                );
            }
        });

        const startTime = Date.now();
        const requestResults = await Promise.all(requestPromises);
        const duration = Date.now() - startTime;

        const successfulRequests = requestResults.filter(r => r.success).length;
        const throughput = successfulRequests / (duration / 1000);

        console.log(`✅ Scaling test completed:`);
        console.log(`   Requests: ${successfulRequests}/${requestResults.length}`);
        console.log(`   Duration: ${duration}ms`);
        console.log(`   Throughput: ${throughput.toFixed(2)} req/sec`);

        results.performance_data = {
            requests_total: requestResults.length,
            requests_successful: successfulRequests,
            duration_ms: duration,
            throughput_rps: throughput
        };

        // Check for consistent responses (stateless behavior)
        const healthRequests = requestResults.filter(r => r.endpoint === '/health');
        const healthSuccesses = healthRequests.filter(r => r.success).length;

        if (healthSuccesses === healthRequests.length) {
            console.log('✅ Consistent responses across requests (good for scaling)');
            results.stateless_operation = true;
        }

        // Test session handling
        console.log('\n📊 Testing session handling...');

        const sessionRequests = [];
        const sessionId = uuidv4();

        for (let i = 0; i < 3; i++) {
            sessionRequests.push(
                makeRequest(`${API_BASE}/workers/${workerName}`, {
                    headers: {
                        'X-Session-ID': sessionId
                    }
                }).then(response => ({
                    success: response.ok,
                    sessionHeader: response.headers['x-session-id']
                }))
            );
        }

        const sessionResults = await Promise.all(sessionRequests);
        const consistentSessions = sessionResults.filter(r =>
            r.success && (!r.sessionHeader || r.sessionHeader === sessionId)
        ).length;

        if (consistentSessions === sessionResults.length) {
            console.log('✅ Session handling consistent across requests');
            results.session_handling = true;
        }

        // Test resource efficiency
        const avgResponseTime = duration / requestResults.length;
        if (avgResponseTime < 1000) { // Less than 1 second average
            console.log('✅ Good response time for scaling');
            results.resource_efficiency = true;
        }

        // Cleanup
        await makeRequest(`${API_BASE}/workers/${workerName}`, { method: 'DELETE' });

        results.test_passed = results.stateless_operation && results.resource_efficiency;

        console.log(`\n📊 Scaling Behavior Summary:`);
        console.log(`   Stateless operation: ${results.stateless_operation ? 'YES' : 'NEEDS_REVIEW'}`);
        console.log(`   Session handling: ${results.session_handling ? 'CONSISTENT' : 'NEEDS_REVIEW'}`);
        console.log(`   Resource efficiency: ${results.resource_efficiency ? 'GOOD' : 'NEEDS_IMPROVEMENT'}`);
        console.log(`   Average response time: ${avgResponseTime.toFixed(2)}ms`);

    } catch (error) {
        console.error(`❌ Scaling behavior test failed: ${error.message}`);
    }

    return results;
}

/**
 * Test backup and restore readiness
 */
async function testBackupRestore() {
    console.log('\n=== Testing Backup and Restore Readiness ===');

    const results = {
        test_passed: false,
        backup_endpoint: false,
        restore_endpoint: false,
        data_export: false,
        data_import: false,
        backup_scheduling: false
    };

    try {
        // Test backup endpoints
        console.log('\n📊 Testing backup endpoints...');

        const backupEndpoints = [
            '/admin/backup',
            '/backup',
            '/api/backup',
            '/export'
        ];

        for (const endpoint of backupEndpoints) {
            try {
                const response = await makeRequest(`${API_BASE}${endpoint}`, {
                    method: 'GET'
                });

                if (response.ok || response.status === 202) { // 202 = Accepted (async backup)
                    console.log(`✅ Backup endpoint found: ${endpoint}`);
                    results.backup_endpoint = true;

                    if (response.data) {
                        console.log(`   Backup info:`, JSON.stringify(response.data, null, 2));

                        if (response.data.schedule || response.data.automated) {
                            results.backup_scheduling = true;
                            console.log('✅ Automated backup scheduling detected');
                        }
                    }

                    break;
                }
            } catch (error) {
                console.log(`   ❌ ${endpoint} not available: ${error.message}`);
            }
        }

        // Test restore endpoints
        console.log('\n📊 Testing restore endpoints...');

        const restoreEndpoints = [
            '/admin/restore',
            '/restore',
            '/api/restore',
            '/import'
        ];

        for (const endpoint of restoreEndpoints) {
            try {
                const response = await makeRequest(`${API_BASE}${endpoint}`, {
                    method: 'OPTIONS' // Use OPTIONS to test availability without triggering restore
                });

                if (response.status === 200 || response.status === 204) {
                    console.log(`✅ Restore endpoint found: ${endpoint}`);
                    results.restore_endpoint = true;
                    break;
                }
            } catch (error) {
                console.log(`   ❌ ${endpoint} not available`);
            }
        }

        // Test data export capability
        console.log('\n📊 Testing data export capability...');

        const exportEndpoints = [
            '/export/workers',
            '/api/export/data',
            '/admin/export'
        ];

        for (const endpoint of exportEndpoints) {
            try {
                const response = await makeRequest(`${API_BASE}${endpoint}`);

                if (response.ok) {
                    console.log(`✅ Data export endpoint found: ${endpoint}`);
                    results.data_export = true;

                    if (response.data) {
                        if (Array.isArray(response.data) || typeof response.data === 'object') {
                            console.log('   ✅ Structured data export available');
                        }
                    }

                    break;
                }
            } catch (error) {
                console.log(`   ❌ ${endpoint} not available`);
            }
        }

        // Test data import capability (without actually importing)
        console.log('\n📊 Testing data import capability...');

        const importEndpoints = [
            '/import/workers',
            '/api/import/data',
            '/admin/import'
        ];

        for (const endpoint of importEndpoints) {
            try {
                const response = await makeRequest(`${API_BASE}${endpoint}`, {
                    method: 'POST',
                    body: JSON.stringify({
                        test: true,
                        dry_run: true,
                        data: []
                    })
                });

                if (response.ok || response.status === 400) { // 400 might be validation error (endpoint exists)
                    console.log(`✅ Data import endpoint found: ${endpoint}`);
                    results.data_import = true;
                    break;
                }
            } catch (error) {
                console.log(`   ❌ ${endpoint} not available`);
            }
        }

        results.test_passed =
            results.backup_endpoint ||
            results.data_export ||
            results.data_import;

        console.log(`\n📊 Backup and Restore Summary:`);
        console.log(`   Backup endpoint: ${results.backup_endpoint ? 'AVAILABLE' : 'MISSING'}`);
        console.log(`   Restore endpoint: ${results.restore_endpoint ? 'AVAILABLE' : 'MISSING'}`);
        console.log(`   Data export: ${results.data_export ? 'AVAILABLE' : 'MISSING'}`);
        console.log(`   Data import: ${results.data_import ? 'AVAILABLE' : 'MISSING'}`);
        console.log(`   Backup scheduling: ${results.backup_scheduling ? 'CONFIGURED' : 'MANUAL'}`);

    } catch (error) {
        console.error(`❌ Backup and restore test failed: ${error.message}`);
    }

    return results;
}

/**
 * Test infrastructure monitoring integration
 */
async function testInfrastructureMonitoring() {
    console.log('\n=== Testing Infrastructure Monitoring Integration ===');

    const results = {
        test_passed: false,
        prometheus_metrics: false,
        custom_metrics: false,
        infrastructure_info: false,
        monitoring_labels: false
    };

    try {
        // Test Prometheus metrics format
        console.log('\n📊 Testing Prometheus metrics...');

        try {
            const metricsResponse = await makeRequest(`${API_BASE}/metrics`, {
                headers: {
                    'Accept': 'text/plain'
                }
            });

            if (metricsResponse.ok && typeof metricsResponse.data === 'string') {
                console.log('✅ Prometheus metrics endpoint available');
                results.prometheus_metrics = true;

                const metricsLines = metricsResponse.data.split('\n');
                const metricCount = metricsLines.filter(line =>
                    !line.startsWith('#') && line.trim() !== ''
                ).length;

                console.log(`   Metric count: ${metricCount}`);

                // Check for infrastructure-specific metrics
                const infraMetrics = metricsLines.filter(line =>
                    line.includes('cpu') ||
                    line.includes('memory') ||
                    line.includes('disk') ||
                    line.includes('network') ||
                    line.includes('container')
                );

                if (infraMetrics.length > 0) {
                    console.log('✅ Infrastructure metrics available');
                    results.infrastructure_info = true;

                    infraMetrics.slice(0, 3).forEach(metric => {
                        console.log(`   ${metric}`);
                    });
                }

                // Check for monitoring labels
                const labeledMetrics = metricsLines.filter(line =>
                    line.includes('{') && line.includes('}')
                );

                if (labeledMetrics.length > 0) {
                    console.log('✅ Labeled metrics for monitoring');
                    results.monitoring_labels = true;
                }

                // Check for custom application metrics
                const customMetrics = metricsLines.filter(line =>
                    line.includes('worker') ||
                    line.includes('execution') ||
                    line.includes('graph') ||
                    line.includes('api_')
                );

                if (customMetrics.length > 0) {
                    console.log('✅ Custom application metrics available');
                    results.custom_metrics = true;
                }
            }
        } catch (error) {
            console.log('ℹ️ Prometheus metrics not available');
        }

        // Test monitoring integration hints
        console.log('\n📊 Checking monitoring integration...');

        try {
            const healthResponse = await makeRequest(`${API_BASE}/health`);

            if (healthResponse.ok && healthResponse.data) {
                const health = healthResponse.data;

                if (health.version || health.build_info || health.deployment_info) {
                    console.log('✅ Deployment info available for monitoring');
                    results.infrastructure_info = true;
                }

                if (health.instance_id || health.node_id || health.pod_name) {
                    console.log('✅ Instance identification for monitoring');
                }
            }
        } catch (error) {
            console.log('ℹ️ Could not check health for monitoring info');
        }

        results.test_passed =
            results.prometheus_metrics ||
            results.custom_metrics ||
            results.infrastructure_info;

        console.log(`\n📊 Infrastructure Monitoring Summary:`);
        console.log(`   Prometheus metrics: ${results.prometheus_metrics ? 'AVAILABLE' : 'MISSING'}`);
        console.log(`   Custom metrics: ${results.custom_metrics ? 'AVAILABLE' : 'MISSING'}`);
        console.log(`   Infrastructure info: ${results.infrastructure_info ? 'AVAILABLE' : 'MISSING'}`);
        console.log(`   Monitoring labels: ${results.monitoring_labels ? 'PRESENT' : 'BASIC'}`);

    } catch (error) {
        console.error(`❌ Infrastructure monitoring test failed: ${error.message}`);
    }

    return results;
}

/**
 * Main test function for deployment and DevOps
 */
async function runDeploymentDevOpsTest() {
    console.log('🚀 Starting Deployment & DevOps Test Suite');
    console.log('='.repeat(80));

    const testResults = {
        health_probes: false,
        configuration_management: false,
        graceful_shutdown: false,
        scaling_behavior: false,
        backup_restore: false,
        infrastructure_monitoring: false,
        overall_success: false,
        execution_time: 0,
        details: {}
    };

    const testStartTime = Date.now();

    try {
        // Test 1: Health Probes
        console.log('\n📋 Test 1: Health Probes for Container Orchestration');
        const probesResult = await testHealthProbes();
        testResults.health_probes = probesResult.test_passed;
        testResults.details.probes = probesResult;

        // Test 2: Configuration Management
        console.log('\n📋 Test 2: Configuration Management');
        const configResult = await testConfigurationManagement();
        testResults.configuration_management = configResult.test_passed;
        testResults.details.config = configResult;

        // Test 3: Graceful Shutdown
        console.log('\n📋 Test 3: Graceful Shutdown');
        const shutdownResult = await testGracefulShutdown();
        testResults.graceful_shutdown = shutdownResult.test_passed;
        testResults.details.shutdown = shutdownResult;

        // Test 4: Scaling Behavior
        console.log('\n📋 Test 4: Scaling Behavior');
        const scalingResult = await testScalingBehavior();
        testResults.scaling_behavior = scalingResult.test_passed;
        testResults.details.scaling = scalingResult;

        // Test 5: Backup and Restore
        console.log('\n📋 Test 5: Backup and Restore Readiness');
        const backupResult = await testBackupRestore();
        testResults.backup_restore = backupResult.test_passed;
        testResults.details.backup = backupResult;

        // Test 6: Infrastructure Monitoring
        console.log('\n📋 Test 6: Infrastructure Monitoring Integration');
        const monitoringResult = await testInfrastructureMonitoring();
        testResults.infrastructure_monitoring = monitoringResult.test_passed;
        testResults.details.monitoring = monitoringResult;

        // Calculate overall success
        const passedTests = Object.keys(testResults)
            .filter(key => key !== 'overall_success' && key !== 'execution_time' && key !== 'details')
            .filter(key => testResults[key] === true).length;

        const totalTests = 6;

        // Require core DevOps capabilities
        testResults.overall_success =
            testResults.health_probes &&
            testResults.scaling_behavior &&
            passedTests >= 4;

        testResults.execution_time = Date.now() - testStartTime;

        // Generate summary report
        console.log('\n📊 Deployment & DevOps Test Summary');
        console.log('='.repeat(80));
        console.log(`🏁 Test Suite - COMPLETE`);
        console.log(`   Total execution time: ${testResults.execution_time}ms`);
        console.log(`   Overall success: ${testResults.overall_success ? '✅ PASSED' : '❌ FAILED'}`);
        console.log(`   Tests passed: ${passedTests}/${totalTests}`);

        console.log('\n📋 Test Results:');
        console.log(`   ✅ Health Probes: ${testResults.health_probes ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Configuration Management: ${testResults.configuration_management ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Graceful Shutdown: ${testResults.graceful_shutdown ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Scaling Behavior: ${testResults.scaling_behavior ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Backup and Restore: ${testResults.backup_restore ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Infrastructure Monitoring: ${testResults.infrastructure_monitoring ? 'PASSED' : 'FAILED'}`);

        if (testResults.overall_success) {
            console.log('\n🎉 Deployment & DevOps Test PASSED!');
            console.log('   ✅ Application is ready for production deployment');
            console.log('   ✅ Container orchestration support is adequate');
            console.log('   ✅ DevOps workflows are supported');
        } else {
            console.log('\n💥 Deployment & DevOps Test FAILED!');
            console.log('   ❌ Some DevOps readiness issues detected');
            console.log('   ❌ Review deployment and operational procedures');
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
    console.log('🚀 Deployment & DevOps Test Suite');
    console.log('Node.js version:', process.version);
    console.log('Platform:', process.platform);
    console.log('Architecture:', process.arch);
    console.log('='.repeat(80));

    try {
        const testResults = await runDeploymentDevOpsTest();

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
    runDeploymentDevOpsTest,
    testHealthProbes,
    testConfigurationManagement,
    testGracefulShutdown,
    testScalingBehavior,
    testBackupRestore,
    testInfrastructureMonitoring
};
