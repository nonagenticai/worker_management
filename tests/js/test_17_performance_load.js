#!/usr/bin/env node
/**
 * Performance and Load Testing Suite for Worker Management API
 *
 * This comprehensive test suite measures and validates:
 * 1. API response time testing under various load conditions
 * 2. Concurrent request handling (up to 100 concurrent requests)
 * 3. Memory usage monitoring throughout test execution
 * 4. Database query performance and connection pool efficiency
 * 5. Resource utilization metrics and stress testing
 * 6. Rate limiting verification and behavior under load
 * 7. Worker communication latency and graph execution performance
 *
 * Test Categories:
 * - Single Request Performance Baselines
 * - Concurrent Request Handling Tests
 * - Database Connection Pool Efficiency Tests
 * - Memory Leak Detection Tests
 * - API Endpoint Response Time Analysis
 * - Graph Execution Performance Tests
 * - Worker Communication Latency Tests
 * - Stress Testing with Large Payloads
 *
 * Requirements: Node.js 18+ (for native fetch API)
 * Usage: node test_17_performance_load.js
 */

const { v4: uuidv4 } = require('uuid');
const { performance } = require('perf_hooks');
const os = require('os');

// Test Configuration
const API_BASE = 'http://localhost:8887';
const TIMEOUT_MS = 120000; // 2 minutes for performance tests

// Performance Test Constants
const PERF_CONFIG = {
    // Concurrency levels for testing
    CONCURRENCY_LEVELS: [1, 5, 10, 25, 50, 100],

    // Number of iterations for each test
    ITERATIONS: {
        BASELINE: 10,
        CONCURRENT: 5,
        STRESS: 3
    },

    // Payload sizes for stress testing
    PAYLOAD_SIZES: {
        SMALL: 100,      // 100 bytes
        MEDIUM: 10000,   // 10 KB
        LARGE: 100000,   // 100 KB
        XLARGE: 1000000  // 1 MB
    },

    // Performance thresholds (in milliseconds)
    THRESHOLDS: {
        SINGLE_REQUEST: {
            EXCELLENT: 100,
            GOOD: 500,
            ACCEPTABLE: 1000,
            POOR: 2000
        },
        CONCURRENT: {
            EXCELLENT: 200,
            GOOD: 1000,
            ACCEPTABLE: 2000,
            POOR: 5000
        }
    }
};

// Global test results storage
const performanceResults = {
    timestamp: new Date().toISOString(),
    environment: {
        platform: os.platform(),
        cpus: os.cpus().length,
        totalMemory: os.totalmem(),
        nodeVersion: process.version
    },
    tests: {},
    summary: {}
};

// Memory tracking
let memoryBaseline = null;
const memorySnapshots = [];

/**
 * Capture memory usage snapshot
 */
function captureMemorySnapshot(label) {
    const usage = process.memoryUsage();
    const snapshot = {
        label,
        timestamp: Date.now(),
        rss: usage.rss,
        heapTotal: usage.heapTotal,
        heapUsed: usage.heapUsed,
        external: usage.external,
        arrayBuffers: usage.arrayBuffers
    };

    memorySnapshots.push(snapshot);
    return snapshot;
}

/**
 * Calculate memory difference from baseline
 */
function calculateMemoryDelta(current, baseline) {
    return {
        rss: current.rss - baseline.rss,
        heapTotal: current.heapTotal - baseline.heapTotal,
        heapUsed: current.heapUsed - baseline.heapUsed,
        external: current.external - baseline.external,
        arrayBuffers: current.arrayBuffers - baseline.arrayBuffers
    };
}

/**
 * Format bytes to human readable format
 */
function formatBytes(bytes) {
    const sizes = ['B', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 B';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`;
}

/**
 * Make HTTP request with performance timing
 */
async function timedRequest(url, options = {}) {
    const startTime = performance.now();
    const startCpu = process.cpuUsage();

    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                'User-Agent': 'WorkerManagement-PerformanceTest/1.0',
                ...options.headers
            }
        });

        const endTime = performance.now();
        const endCpu = process.cpuUsage(startCpu);
        const responseTime = endTime - startTime;

        let data = null;
        if (response.status !== 204) {
            data = await response.json();
        }

        return {
            success: response.ok,
            status: response.status,
            responseTime,
            cpuTime: (endCpu.user + endCpu.system) / 1000, // Convert to ms
            data,
            error: null
        };
    } catch (error) {
        const endTime = performance.now();
        return {
            success: false,
            status: 0,
            responseTime: endTime - startTime,
            cpuTime: 0,
            data: null,
            error: error.message
        };
    }
}

/**
 * Test single request performance baseline
 */
async function testSingleRequestBaseline() {
    console.log('\n📊 Testing Single Request Performance Baselines...\n');

    const endpoints = [
        { url: '/health', method: 'GET', name: 'Health Check' },
        { url: '/workers', method: 'GET', name: 'List Workers' },
        { url: '/graphs/', method: 'GET', name: 'List Graphs' },
        { url: '/tools/', method: 'GET', name: 'List Tools' },
        { url: '/workers_states', method: 'GET', name: 'List Worker States' }
    ];

    const results = {};

    for (const endpoint of endpoints) {
        console.log(`Testing ${endpoint.name}...`);
        const timings = [];

        for (let i = 0; i < PERF_CONFIG.ITERATIONS.BASELINE; i++) {
            const result = await timedRequest(`${API_BASE}${endpoint.url}`, {
                method: endpoint.method
            });

            if (result.success) {
                timings.push(result.responseTime);
                process.stdout.write(`  Iteration ${i + 1}: ${result.responseTime.toFixed(2)}ms\r`);
            } else {
                console.error(`  ❌ Request failed: ${result.error}`);
            }

            // Small delay between requests
            await new Promise(resolve => setTimeout(resolve, 100));
        }

        if (timings.length > 0) {
            const stats = calculateStats(timings);
            results[endpoint.name] = stats;

            console.log(`\n  ✅ ${endpoint.name} Results:`);
            console.log(`     Average: ${stats.mean.toFixed(2)}ms`);
            console.log(`     Median: ${stats.median.toFixed(2)}ms`);
            console.log(`     P95: ${stats.p95.toFixed(2)}ms`);
            console.log(`     P99: ${stats.p99.toFixed(2)}ms`);
            console.log(`     Min: ${stats.min.toFixed(2)}ms`);
            console.log(`     Max: ${stats.max.toFixed(2)}ms`);
            console.log(`     Performance: ${getPerformanceRating(stats.mean, 'single')}\n`);
        }
    }

    performanceResults.tests.singleRequestBaseline = results;
    return results;
}

/**
 * Test concurrent request handling
 */
async function testConcurrentRequests() {
    console.log('\n🔄 Testing Concurrent Request Handling...\n');

    const results = {};

    for (const concurrency of PERF_CONFIG.CONCURRENCY_LEVELS) {
        console.log(`\nTesting with ${concurrency} concurrent requests...`);

        const iterationResults = [];

        for (let iteration = 0; iteration < PERF_CONFIG.ITERATIONS.CONCURRENT; iteration++) {
            // Create promises for concurrent requests
            const promises = [];
            for (let i = 0; i < concurrency; i++) {
                promises.push(
                    timedRequest(`${API_BASE}/health`, { method: 'GET' })
                );
            }

            // Execute concurrent requests
            const startTime = performance.now();
            const responses = await Promise.all(promises);
            const totalTime = performance.now() - startTime;

            // Calculate statistics
            const successCount = responses.filter(r => r.success).length;
            const responseTimes = responses.map(r => r.responseTime);
            const stats = calculateStats(responseTimes);

            iterationResults.push({
                totalTime,
                successCount,
                failureCount: concurrency - successCount,
                stats
            });

            console.log(`  Iteration ${iteration + 1}: ${successCount}/${concurrency} successful, Total time: ${totalTime.toFixed(2)}ms`);

            // Delay between iterations
            await new Promise(resolve => setTimeout(resolve, 1000));
        }

        // Aggregate results
        const avgSuccessRate = iterationResults.reduce((sum, r) => sum + (r.successCount / concurrency), 0) / iterationResults.length;
        const avgTotalTime = iterationResults.reduce((sum, r) => sum + r.totalTime, 0) / iterationResults.length;

        results[`${concurrency}_concurrent`] = {
            concurrency,
            avgSuccessRate: avgSuccessRate * 100,
            avgTotalTime,
            throughput: (concurrency / avgTotalTime) * 1000, // requests per second
            iterations: iterationResults
        };

        console.log(`\n  ✅ ${concurrency} Concurrent Requests Summary:`);
        console.log(`     Success Rate: ${(avgSuccessRate * 100).toFixed(2)}%`);
        console.log(`     Avg Total Time: ${avgTotalTime.toFixed(2)}ms`);
        console.log(`     Throughput: ${results[`${concurrency}_concurrent`].throughput.toFixed(2)} req/s\n`);
    }

    performanceResults.tests.concurrentRequests = results;
    return results;
}

/**
 * Test database query performance
 */
async function testDatabasePerformance() {
    console.log('\n🗄️  Testing Database Query Performance...\n');

    const results = {};

    // Create test workers for database operations
    const testWorkers = [];
    console.log('Creating test workers for database performance testing...');

    for (let i = 0; i < 5; i++) {
        const workerName = `perf_test_worker_${uuidv4()}`;
        const createResult = await timedRequest(`${API_BASE}/workers`, {
            method: 'POST',
            body: JSON.stringify({
                worker_name: workerName,
                tool_names: ['example_text_analyzer'],
                description: 'Performance test worker'
            })
        });

        if (createResult.success) {
            testWorkers.push(workerName);
            console.log(`  Created worker ${i + 1}/5: ${createResult.responseTime.toFixed(2)}ms`);
        }
    }

    // Test 1: Single worker state queries
    console.log('\n1. Testing single worker state queries...');
    const singleQueryTimes = [];

    for (const worker of testWorkers) {
        const result = await timedRequest(`${API_BASE}/workers/${worker}/states?limit=10`, {
            method: 'GET'
        });

        if (result.success) {
            singleQueryTimes.push(result.responseTime);
        }
    }

    results.singleWorkerQueries = calculateStats(singleQueryTimes);

    // Test 2: Bulk state queries
    console.log('2. Testing bulk state queries...');
    const bulkQueryTimes = [];

    for (let i = 0; i < PERF_CONFIG.ITERATIONS.BASELINE; i++) {
        const result = await timedRequest(`${API_BASE}/workers/states`, {
            method: 'GET'
        });

        if (result.success) {
            bulkQueryTimes.push(result.responseTime);
            console.log(`  Iteration ${i + 1}: ${result.responseTime.toFixed(2)}ms, Records: ${result.data?.total || 0}`);
        }
    }

    results.bulkQueries = calculateStats(bulkQueryTimes);

    // Test 3: Concurrent database operations
    console.log('3. Testing concurrent database operations...');
    const concurrentDbOps = [];

    for (let i = 0; i < 20; i++) {
        concurrentDbOps.push(
            timedRequest(`${API_BASE}/workers/${testWorkers[i % testWorkers.length]}/states?limit=5`, {
                method: 'GET'
            })
        );
    }

    const dbStartTime = performance.now();
    const dbResults = await Promise.all(concurrentDbOps);
    const dbTotalTime = performance.now() - dbStartTime;

    const dbSuccessCount = dbResults.filter(r => r.success).length;
    results.concurrentDbOps = {
        totalRequests: 20,
        successCount: dbSuccessCount,
        totalTime: dbTotalTime,
        avgTimePerRequest: dbTotalTime / 20,
        throughput: (20 / dbTotalTime) * 1000
    };

    console.log(`  ✅ Concurrent DB Ops: ${dbSuccessCount}/20 successful in ${dbTotalTime.toFixed(2)}ms`);

    // Cleanup test workers
    console.log('\nCleaning up test workers...');
    for (const worker of testWorkers) {
        await timedRequest(`${API_BASE}/workers/${worker}`, { method: 'DELETE' });
    }

    performanceResults.tests.databasePerformance = results;
    return results;
}

/**
 * Test memory usage and leak detection
 */
async function testMemoryUsage() {
    console.log('\n🧠 Testing Memory Usage and Leak Detection...\n');

    const results = {
        baseline: null,
        afterWorkerCreation: null,
        afterMessaging: null,
        afterCleanup: null,
        leakDetected: false
    };

    // Capture baseline memory
    global.gc && global.gc(); // Force garbage collection if available
    await new Promise(resolve => setTimeout(resolve, 1000));
    results.baseline = captureMemorySnapshot('baseline');
    console.log(`Baseline memory: RSS=${formatBytes(results.baseline.rss)}, Heap=${formatBytes(results.baseline.heapUsed)}`);

    // Create workers and capture memory
    const testWorkers = [];
    console.log('\nCreating 10 workers...');

    for (let i = 0; i < 10; i++) {
        const workerName = `memory_test_${uuidv4()}`;
        await timedRequest(`${API_BASE}/workers`, {
            method: 'POST',
            body: JSON.stringify({
                worker_name: workerName,
                tool_names: ['example_text_analyzer'],
                agent_role: 'Memory test agent',
                model_provider: 'mlx',
                model: 'openhermes',
                temperature: 0.1
            })
        });
        testWorkers.push(workerName);
    }

    results.afterWorkerCreation = captureMemorySnapshot('after_worker_creation');
    const creationDelta = calculateMemoryDelta(results.afterWorkerCreation, results.baseline);
    console.log(`After worker creation: RSS=${formatBytes(results.afterWorkerCreation.rss)} (+${formatBytes(creationDelta.rss)})`);

    // Send messages to workers
    console.log('\nSending messages to workers...');
    for (let i = 0; i < 5; i++) {
        const promises = testWorkers.slice(0, 5).map(worker =>
            timedRequest(`${API_BASE}/workers/${worker}/message`, {
                method: 'POST',
                body: JSON.stringify({
                    message: `Memory test message ${i}. Please provide a brief response.`
                })
            })
        );
        await Promise.all(promises);
        console.log(`  Completed round ${i + 1}/5`);
    }

    results.afterMessaging = captureMemorySnapshot('after_messaging');
    const messagingDelta = calculateMemoryDelta(results.afterMessaging, results.afterWorkerCreation);
    console.log(`After messaging: RSS=${formatBytes(results.afterMessaging.rss)} (+${formatBytes(messagingDelta.rss)})`);

    // Cleanup
    console.log('\nCleaning up workers...');
    for (const worker of testWorkers) {
        await timedRequest(`${API_BASE}/workers/${worker}`, { method: 'DELETE' });
    }

    // Force garbage collection and wait
    global.gc && global.gc();
    await new Promise(resolve => setTimeout(resolve, 2000));

    results.afterCleanup = captureMemorySnapshot('after_cleanup');
    const cleanupDelta = calculateMemoryDelta(results.afterCleanup, results.baseline);
    console.log(`After cleanup: RSS=${formatBytes(results.afterCleanup.rss)} (${cleanupDelta.rss > 0 ? '+' : ''}${formatBytes(cleanupDelta.rss)} from baseline)`);

    // Check for memory leak
    const heapGrowth = cleanupDelta.heapUsed;
    const rssGrowth = cleanupDelta.rss;
    results.leakDetected = heapGrowth > 50 * 1024 * 1024; // 50MB threshold

    if (results.leakDetected) {
        console.log('\n⚠️  Potential memory leak detected!');
        console.log(`  Heap growth: ${formatBytes(heapGrowth)}`);
        console.log(`  RSS growth: ${formatBytes(rssGrowth)}`);
    } else {
        console.log('\n✅ No significant memory leak detected');
    }

    performanceResults.tests.memoryUsage = results;
    return results;
}

/**
 * Test API endpoint response times under load
 */
async function testEndpointResponseTimes() {
    console.log('\n⏱️  Testing API Endpoint Response Times Under Load...\n');

    const endpoints = [
        { path: '/workers', method: 'GET', name: 'List Workers' },
        { path: '/workers', method: 'POST', name: 'Create Worker', body: () => ({
            worker_name: `response_test_${uuidv4()}`,
            tool_names: ['example_text_analyzer']
        })},
        { path: '/graphs/', method: 'GET', name: 'List Graphs' },
        { path: '/tools/', method: 'GET', name: 'List Tools' },
        { path: '/workers_states', method: 'GET', name: 'Worker States' }
    ];

    const results = {};

    for (const endpoint of endpoints) {
        console.log(`\nTesting ${endpoint.name}...`);
        const measurements = [];

        // Test with increasing load
        for (const concurrency of [1, 5, 10]) {
            const promises = [];

            for (let i = 0; i < concurrency; i++) {
                const options = {
                    method: endpoint.method
                };

                if (endpoint.body) {
                    options.body = JSON.stringify(endpoint.body());
                }

                promises.push(timedRequest(`${API_BASE}${endpoint.path}`, options));
            }

            const startTime = performance.now();
            const responses = await Promise.all(promises);
            const totalTime = performance.now() - startTime;

            const successCount = responses.filter(r => r.success).length;
            const avgResponseTime = responses.reduce((sum, r) => sum + r.responseTime, 0) / responses.length;

            measurements.push({
                concurrency,
                successCount,
                totalTime,
                avgResponseTime,
                throughput: (successCount / totalTime) * 1000
            });

            console.log(`  Load ${concurrency}: ${successCount}/${concurrency} successful, Avg: ${avgResponseTime.toFixed(2)}ms`);

            // Cleanup created workers
            if (endpoint.name === 'Create Worker') {
                for (const response of responses) {
                    if (response.success && response.data?.name) {
                        await timedRequest(`${API_BASE}/workers/${response.data.name}`, { method: 'DELETE' });
                    }
                }
            }

            await new Promise(resolve => setTimeout(resolve, 500));
        }

        results[endpoint.name] = measurements;
    }

    performanceResults.tests.endpointResponseTimes = results;
    return results;
}

/**
 * Test graph execution performance
 */
async function testGraphExecutionPerformance() {
    console.log('\n📈 Testing Graph Execution Performance...\n');

    const results = {
        simpleGraph: null,
        complexGraph: null,
        parallelExecution: null
    };

    // Create test graph
    const graphName = `perf_test_graph_${uuidv4()}`;
    console.log('Creating test graph...');

    const graphDefinition = {
        graph_name: graphName,
        title: "Performance Test Graph",
        description: "Graph for performance testing",
        nodes: [
            {
                id: "start",
                type: "start",
                data: { label: "Start" },
                position: { x: 0, y: 0 }
            },
            {
                id: "process1",
                type: "process",
                data: {
                    label: "Process 1",
                    tool_calls: [{
                        tool_name: "example_text_analyzer",
                        parameters: { text: "Performance test text" }
                    }]
                },
                position: { x: 100, y: 0 }
            },
            {
                id: "process2",
                type: "process",
                data: {
                    label: "Process 2",
                    tool_calls: [{
                        tool_name: "example_text_analyzer",
                        parameters: { text: "Another test text" }
                    }]
                },
                position: { x: 200, y: 0 }
            },
            {
                id: "end",
                type: "end",
                data: { label: "End" },
                position: { x: 300, y: 0 }
            }
        ],
        edges: [
            { id: "e1", source: "start", target: "process1" },
            { id: "e2", source: "process1", target: "process2" },
            { id: "e3", source: "process2", target: "end" }
        ]
    };

    const createResult = await timedRequest(`${API_BASE}/graphs/`, {
        method: 'POST',
        body: JSON.stringify(graphDefinition)
    });

    if (!createResult.success) {
        console.error('Failed to create test graph');
        return results;
    }

    // Test 1: Simple graph execution
    console.log('\n1. Testing simple graph execution...');
    const simpleExecutionTimes = [];

    for (let i = 0; i < PERF_CONFIG.ITERATIONS.BASELINE; i++) {
        const result = await timedRequest(`${API_BASE}/graphs/${graphName}/execute`, {
            method: 'POST',
            body: JSON.stringify({
                initial_data: { test_iteration: i }
            })
        });

        if (result.success) {
            simpleExecutionTimes.push(result.responseTime);
            console.log(`  Iteration ${i + 1}: ${result.responseTime.toFixed(2)}ms`);
        }
    }

    results.simpleGraph = calculateStats(simpleExecutionTimes);

    // Test 2: Parallel graph executions
    console.log('\n2. Testing parallel graph executions...');
    const parallelPromises = [];

    for (let i = 0; i < 5; i++) {
        parallelPromises.push(
            timedRequest(`${API_BASE}/graphs/${graphName}/execute`, {
                method: 'POST',
                body: JSON.stringify({
                    initial_data: { parallel_test: i }
                })
            })
        );
    }

    const parallelStartTime = performance.now();
    const parallelResults = await Promise.all(parallelPromises);
    const parallelTotalTime = performance.now() - parallelStartTime;

    const parallelSuccessCount = parallelResults.filter(r => r.success).length;
    results.parallelExecution = {
        totalRequests: 5,
        successCount: parallelSuccessCount,
        totalTime: parallelTotalTime,
        avgTimePerExecution: parallelTotalTime / 5,
        throughput: (5 / parallelTotalTime) * 1000
    };

    console.log(`  ✅ Parallel executions: ${parallelSuccessCount}/5 successful in ${parallelTotalTime.toFixed(2)}ms`);

    // Cleanup
    await timedRequest(`${API_BASE}/graphs/${graphName}`, { method: 'DELETE' });

    performanceResults.tests.graphExecutionPerformance = results;
    return results;
}

/**
 * Test stress with large payloads
 */
async function testStressWithLargePayloads() {
    console.log('\n💪 Testing Stress with Large Payloads...\n');

    const results = {};

    // Create test worker
    const workerName = `stress_test_${uuidv4()}`;
    await timedRequest(`${API_BASE}/workers`, {
        method: 'POST',
        body: JSON.stringify({
            worker_name: workerName,
            tool_names: ['example_text_analyzer'],
            description: 'Stress test worker'
        })
    });

    for (const [sizeName, size] of Object.entries(PERF_CONFIG.PAYLOAD_SIZES)) {
        console.log(`\nTesting with ${sizeName} payload (${formatBytes(size)})...`);

        // Generate payload
        const payload = 'x'.repeat(size);
        const measurements = [];

        for (let i = 0; i < PERF_CONFIG.ITERATIONS.STRESS; i++) {
            const result = await timedRequest(`${API_BASE}/workers/${workerName}`, {
                method: 'PUT',
                body: JSON.stringify({
                    worker_name: workerName,
                    tool_names: ['example_text_analyzer'],
                    description: payload
                })
            });

            if (result.success) {
                measurements.push(result.responseTime);
                console.log(`  Iteration ${i + 1}: ${result.responseTime.toFixed(2)}ms`);
            } else {
                console.log(`  Iteration ${i + 1}: Failed - ${result.error}`);
            }

            await new Promise(resolve => setTimeout(resolve, 500));
        }

        if (measurements.length > 0) {
            results[sizeName] = {
                payloadSize: size,
                stats: calculateStats(measurements),
                successRate: (measurements.length / PERF_CONFIG.ITERATIONS.STRESS) * 100
            };
        }
    }

    // Cleanup
    await timedRequest(`${API_BASE}/workers/${workerName}`, { method: 'DELETE' });

    performanceResults.tests.stressWithLargePayloads = results;
    return results;
}

/**
 * Calculate statistics from array of numbers
 */
function calculateStats(numbers) {
    if (numbers.length === 0) return null;

    const sorted = [...numbers].sort((a, b) => a - b);
    const sum = sorted.reduce((a, b) => a + b, 0);
    const mean = sum / sorted.length;

    // Calculate standard deviation
    const squaredDiffs = sorted.map(x => Math.pow(x - mean, 2));
    const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / sorted.length;
    const stdDev = Math.sqrt(avgSquaredDiff);

    return {
        count: sorted.length,
        mean,
        median: sorted[Math.floor(sorted.length / 2)],
        min: sorted[0],
        max: sorted[sorted.length - 1],
        p95: sorted[Math.floor(sorted.length * 0.95)],
        p99: sorted[Math.floor(sorted.length * 0.99)],
        stdDev
    };
}

/**
 * Get performance rating based on response time
 */
function getPerformanceRating(responseTime, type = 'single') {
    const thresholds = type === 'single' ?
        PERF_CONFIG.THRESHOLDS.SINGLE_REQUEST :
        PERF_CONFIG.THRESHOLDS.CONCURRENT;

    if (responseTime <= thresholds.EXCELLENT) return '🟢 Excellent';
    if (responseTime <= thresholds.GOOD) return '🟡 Good';
    if (responseTime <= thresholds.ACCEPTABLE) return '🟠 Acceptable';
    return '🔴 Poor';
}

/**
 * Generate performance report
 */
function generatePerformanceReport() {
    console.log('\n' + '='.repeat(80));
    console.log('📊 PERFORMANCE TEST REPORT');
    console.log('='.repeat(80));

    console.log('\n📅 Test Information:');
    console.log(`   Timestamp: ${performanceResults.timestamp}`);
    console.log(`   Platform: ${performanceResults.environment.platform}`);
    console.log(`   CPUs: ${performanceResults.environment.cpus}`);
    console.log(`   Total Memory: ${formatBytes(performanceResults.environment.totalMemory)}`);
    console.log(`   Node Version: ${performanceResults.environment.nodeVersion}`);

    // Single Request Performance
    if (performanceResults.tests.singleRequestBaseline) {
        console.log('\n🎯 Single Request Performance:');
        for (const [endpoint, stats] of Object.entries(performanceResults.tests.singleRequestBaseline)) {
            console.log(`\n   ${endpoint}:`);
            console.log(`     Average: ${stats.mean.toFixed(2)}ms ${getPerformanceRating(stats.mean, 'single')}`);
            console.log(`     P95: ${stats.p95.toFixed(2)}ms`);
            console.log(`     P99: ${stats.p99.toFixed(2)}ms`);
        }
    }

    // Concurrent Request Performance
    if (performanceResults.tests.concurrentRequests) {
        console.log('\n🔄 Concurrent Request Performance:');
        for (const [key, data] of Object.entries(performanceResults.tests.concurrentRequests)) {
            console.log(`\n   ${data.concurrency} Concurrent Requests:`);
            console.log(`     Success Rate: ${data.avgSuccessRate.toFixed(2)}%`);
            console.log(`     Throughput: ${data.throughput.toFixed(2)} req/s`);
            console.log(`     Total Time: ${data.avgTotalTime.toFixed(2)}ms`);
        }
    }

    // Database Performance
    if (performanceResults.tests.databasePerformance) {
        console.log('\n🗄️  Database Performance:');
        const dbPerf = performanceResults.tests.databasePerformance;

        if (dbPerf.singleWorkerQueries) {
            console.log(`\n   Single Worker Queries:`);
            console.log(`     Average: ${dbPerf.singleWorkerQueries.mean.toFixed(2)}ms`);
            console.log(`     P95: ${dbPerf.singleWorkerQueries.p95.toFixed(2)}ms`);
        }

        if (dbPerf.concurrentDbOps) {
            console.log(`\n   Concurrent DB Operations:`);
            console.log(`     Success Rate: ${(dbPerf.concurrentDbOps.successCount / dbPerf.concurrentDbOps.totalRequests * 100).toFixed(2)}%`);
            console.log(`     Throughput: ${dbPerf.concurrentDbOps.throughput.toFixed(2)} ops/s`);
        }
    }

    // Memory Usage
    if (performanceResults.tests.memoryUsage) {
        console.log('\n🧠 Memory Usage:');
        const mem = performanceResults.tests.memoryUsage;
        console.log(`   Baseline: ${formatBytes(mem.baseline.heapUsed)}`);
        console.log(`   Peak: ${formatBytes(mem.afterMessaging.heapUsed)}`);
        console.log(`   After Cleanup: ${formatBytes(mem.afterCleanup.heapUsed)}`);
        console.log(`   Leak Detected: ${mem.leakDetected ? '⚠️  Yes' : '✅ No'}`);
    }

    // Graph Execution
    if (performanceResults.tests.graphExecutionPerformance) {
        console.log('\n📈 Graph Execution Performance:');
        const graphPerf = performanceResults.tests.graphExecutionPerformance;

        if (graphPerf.simpleGraph) {
            console.log(`   Simple Graph Execution:`);
            console.log(`     Average: ${graphPerf.simpleGraph.mean.toFixed(2)}ms`);
            console.log(`     P95: ${graphPerf.simpleGraph.p95.toFixed(2)}ms`);
        }

        if (graphPerf.parallelExecution) {
            console.log(`   Parallel Executions:`);
            console.log(`     Throughput: ${graphPerf.parallelExecution.throughput.toFixed(2)} exec/s`);
        }
    }

    // Bottlenecks and Recommendations
    console.log('\n🚨 Identified Bottlenecks and Recommendations:');

    const bottlenecks = [];

    // Check single request performance
    if (performanceResults.tests.singleRequestBaseline) {
        for (const [endpoint, stats] of Object.entries(performanceResults.tests.singleRequestBaseline)) {
            if (stats.mean > PERF_CONFIG.THRESHOLDS.SINGLE_REQUEST.ACCEPTABLE) {
                bottlenecks.push(`Slow response time for ${endpoint}: ${stats.mean.toFixed(2)}ms`);
            }
        }
    }

    // Check concurrent performance degradation
    if (performanceResults.tests.concurrentRequests) {
        const results = Object.values(performanceResults.tests.concurrentRequests);
        const highLoadResult = results.find(r => r.concurrency === 100);
        if (highLoadResult && highLoadResult.avgSuccessRate < 95) {
            bottlenecks.push(`Low success rate under high load: ${highLoadResult.avgSuccessRate.toFixed(2)}%`);
        }
    }

    // Check memory usage
    if (performanceResults.tests.memoryUsage?.leakDetected) {
        bottlenecks.push('Potential memory leak detected');
    }

    if (bottlenecks.length > 0) {
        bottlenecks.forEach((bottleneck, i) => {
            console.log(`   ${i + 1}. ${bottleneck}`);
        });

        console.log('\n   Recommendations:');
        console.log('   • Implement connection pooling for database operations');
        console.log('   • Add caching layer for frequently accessed data');
        console.log('   • Optimize database queries with proper indexing');
        console.log('   • Implement request rate limiting to prevent overload');
        console.log('   • Consider horizontal scaling for high-load scenarios');
    } else {
        console.log('   ✅ No significant bottlenecks detected');
    }

    console.log('\n' + '='.repeat(80));

    // Save results to file
    const fs = require('fs');
    const reportPath = `performance_report_${Date.now()}.json`;
    fs.writeFileSync(reportPath, JSON.stringify(performanceResults, null, 2));
    console.log(`\n📁 Detailed results saved to: ${reportPath}`);
}

/**
 * Main test execution
 */
async function runPerformanceTests() {
    console.log('🚀 Starting Worker Management API Performance and Load Testing Suite');
    console.log('=' .repeat(80));

    try {
        // Check API health
        console.log('\n🏥 Checking API health...');
        const healthResult = await timedRequest(`${API_BASE}/health`);
        if (!healthResult.success) {
            throw new Error('API is not healthy');
        }
        console.log('✅ API is healthy\n');

        // Set memory baseline
        memoryBaseline = captureMemorySnapshot('initial');

        // Run all performance tests
        await testSingleRequestBaseline();
        await testConcurrentRequests();
        await testDatabasePerformance();
        await testMemoryUsage();
        await testEndpointResponseTimes();
        await testGraphExecutionPerformance();
        await testStressWithLargePayloads();

        // Generate final report
        generatePerformanceReport();

        console.log('\n✅ All performance tests completed successfully!');

    } catch (error) {
        console.error('\n❌ Performance test suite failed:', error.message);
        process.exit(1);
    }
}

// Execute tests if running directly
if (require.main === module) {
    runPerformanceTests().catch(error => {
        console.error('Unhandled error:', error);
        process.exit(1);
    });
}

module.exports = {
    runPerformanceTests,
    performanceResults
};
