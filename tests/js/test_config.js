#!/usr/bin/env node
/**
 * Test Configuration and Utilities for Worker Management API Tests
 *
 * This module provides centralized configuration and utilities to address
 * timing issues, environment setup, and test reliability improvements.
 */

// Enhanced Configuration
const TEST_CONFIG = {
    API_BASE: 'http://localhost:8887',

    // Timing Configuration
    TIMEOUTS: {
        REQUEST: 30000,      // 30 seconds for individual requests
        TEST_SUITE: 300000,  // 5 minutes for entire test suite
        HEALTH_CHECK: 10000, // 10 seconds for health check
    },

    // Retry Configuration
    RETRIES: {
        MAX_ATTEMPTS: 5,     // Increased from 3
        INITIAL_DELAY: 1000, // 1 second
        BACKOFF_FACTOR: 1.5, // Exponential backoff
        MAX_DELAY: 10000,    // 10 seconds max delay
    },

    // Delays for Test Stability
    DELAYS: {
        BETWEEN_REQUESTS: 500,    // 500ms between requests
        AFTER_CREATION: 2000,     // 2s after creating resources
        AFTER_UPDATE: 1000,       // 1s after updates
        AFTER_DELETE: 1000,       // 1s after deletions
        API_READY: 5000,          // 5s for API to be ready
        PARALLEL_BATCH: 3,        // Max 3 parallel operations
    },

    // Test Environment
    ENVIRONMENT: {
        MAX_CONCURRENT_TESTS: 3,  // Reduced from 5
        CLEANUP_TIMEOUT: 30000,   // 30s for cleanup
        HEALTH_CHECK_RETRIES: 10, // 10 attempts for health check
    }
};

/**
 * Enhanced HTTP request function with exponential backoff and better error handling
 */
async function makeEnhancedRequest(url, options = {}, retries = TEST_CONFIG.RETRIES.MAX_ATTEMPTS) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'WorkerManagement-EnhancedTest/2.0',
            ...options.headers
        },
        timeout: TEST_CONFIG.TIMEOUTS.REQUEST,
    };

    const finalOptions = { ...defaultOptions, ...options };
    let lastError;

    for (let attempt = 1; attempt <= retries; attempt++) {
        try {
            console.log(`🌐 ${finalOptions.method || 'GET'} ${url} (Attempt ${attempt}/${retries})`);

            // Add timeout wrapper
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), TEST_CONFIG.TIMEOUTS.REQUEST);

            const response = await fetch(url, {
                ...finalOptions,
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            // Handle 204 No Content responses
            if (response.status === 204) {
                console.log(`✅ Response received (${response.status}) - No Content`);
                await sleep(TEST_CONFIG.DELAYS.BETWEEN_REQUESTS);
                return { success: true, status: 204 };
            }

            if (!response.ok) {
                const errorText = await response.text();
                const error = new Error(`HTTP ${response.status}: ${errorText}`);
                error.status = response.status;
                error.response = errorText;
                throw error;
            }

            const data = await response.json();
            console.log(`✅ Response received (${response.status}) - Attempt ${attempt}`);

            // Adaptive delay between requests
            await sleep(TEST_CONFIG.DELAYS.BETWEEN_REQUESTS);

            return data;

        } catch (error) {
            lastError = error;
            console.error(`❌ Request failed (Attempt ${attempt}/${retries}): ${error.message}`);

            // Don't retry on client errors (4xx) except 429 (rate limit)
            if (error.status && error.status >= 400 && error.status < 500 && error.status !== 429) {
                console.log(`🚫 Not retrying client error: ${error.status}`);
                throw error;
            }

            if (attempt === retries) {
                console.error(`💥 All retry attempts exhausted for ${url}`);
                throw lastError;
            }

            // Exponential backoff with jitter
            const delay = Math.min(
                TEST_CONFIG.RETRIES.INITIAL_DELAY * Math.pow(TEST_CONFIG.RETRIES.BACKOFF_FACTOR, attempt - 1),
                TEST_CONFIG.RETRIES.MAX_DELAY
            );
            const jitteredDelay = delay + Math.random() * 1000; // Add up to 1s jitter

            console.log(`⏳ Retrying in ${Math.round(jitteredDelay)}ms...`);
            await sleep(jitteredDelay);
        }
    }
}

/**
 * Enhanced API health check with multiple attempts
 */
async function waitForApiReady(maxAttempts = TEST_CONFIG.ENVIRONMENT.HEALTH_CHECK_RETRIES) {
    console.log('\n🏥 Waiting for API to be ready...');

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            const response = await makeEnhancedRequest(`${TEST_CONFIG.API_BASE}/health`, {}, 1);

            if (response.status === 'healthy') {
                console.log(`✅ API is ready and healthy (attempt ${attempt})`);

                // Additional checks
                await verifyApiEndpoints();

                // Give API extra time to fully initialize
                await sleep(TEST_CONFIG.DELAYS.API_READY);
                return true;
            }
        } catch (error) {
            console.log(`❌ Health check attempt ${attempt}/${maxAttempts} failed: ${error.message}`);
        }

        if (attempt < maxAttempts) {
            const delay = 2000 * attempt; // Increasing delay
            console.log(`⏳ Waiting ${delay}ms before next health check...`);
            await sleep(delay);
        }
    }

    throw new Error('API failed to become ready within timeout period');
}

/**
 * Verify critical API endpoints are responding
 */
async function verifyApiEndpoints() {
    const criticalEndpoints = [
        '/workers',
        '/graphs/',
        '/tools/',
        '/workers_states'
    ];

    console.log('🔍 Verifying critical endpoints...');

    for (const endpoint of criticalEndpoints) {
        try {
            await makeEnhancedRequest(`${TEST_CONFIG.API_BASE}${endpoint}`, {}, 2);
            console.log(`✅ Endpoint verified: ${endpoint}`);
        } catch (error) {
            console.log(`⚠️  Endpoint check failed: ${endpoint} - ${error.message}`);
            // Don't fail completely, just warn
        }
    }
}

/**
 * Smart cleanup with retry logic
 */
async function cleanupWithRetry(cleanupFn, resourceType, resourceId, maxAttempts = 3) {
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            await cleanupFn();
            console.log(`✅ Cleanup successful for ${resourceType}: ${resourceId}`);
            return true;
        } catch (error) {
            if (error.status === 404) {
                console.log(`ℹ️  ${resourceType} ${resourceId} already deleted (404)`);
                return true; // Consider 404 as success for cleanup
            }

            console.log(`❌ Cleanup attempt ${attempt}/${maxAttempts} failed for ${resourceType} ${resourceId}: ${error.message}`);

            if (attempt < maxAttempts) {
                await sleep(1000 * attempt);
            }
        }
    }

    console.error(`💥 Failed to cleanup ${resourceType}: ${resourceId} after ${maxAttempts} attempts`);
    return false;
}

/**
 * Parallel execution with concurrency control
 */
async function executeWithConcurrencyLimit(tasks, limit = TEST_CONFIG.DELAYS.PARALLEL_BATCH) {
    const results = [];

    for (let i = 0; i < tasks.length; i += limit) {
        const batch = tasks.slice(i, i + limit);
        console.log(`📦 Executing batch ${Math.floor(i/limit) + 1} with ${batch.length} tasks`);

        const batchResults = await Promise.allSettled(batch);
        results.push(...batchResults);

        // Delay between batches
        if (i + limit < tasks.length) {
            await sleep(TEST_CONFIG.DELAYS.BETWEEN_REQUESTS * 2);
        }
    }

    return results;
}

/**
 * Enhanced sleep function
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Test result aggregator
 */
function aggregateTestResults(testSuites) {
    const summary = {
        totalTests: 0,
        passedTests: 0,
        failedTests: 0,
        skippedTests: 0,
        suites: {}
    };

    for (const [suiteName, results] of Object.entries(testSuites)) {
        const suiteStats = {
            total: 0,
            passed: 0,
            failed: 0,
            skipped: 0
        };

        if (Array.isArray(results)) {
            suiteStats.total = results.length;
            suiteStats.passed = results.filter(r => r.success === true).length;
            suiteStats.failed = results.filter(r => r.success === false).length;
            suiteStats.skipped = results.filter(r => r.success === undefined).length;
        } else if (typeof results === 'object') {
            const values = Object.values(results);
            suiteStats.total = values.length;
            suiteStats.passed = values.filter(r => r === true).length;
            suiteStats.failed = values.filter(r => r === false).length;
            suiteStats.skipped = values.filter(r => r === undefined || r === null).length;
        }

        summary.suites[suiteName] = suiteStats;
        summary.totalTests += suiteStats.total;
        summary.passedTests += suiteStats.passed;
        summary.failedTests += suiteStats.failed;
        summary.skippedTests += suiteStats.skipped;
    }

    summary.successRate = summary.totalTests > 0 ?
        Math.round((summary.passedTests / summary.totalTests) * 100) : 0;

    return summary;
}

/**
 * Generate unique test identifier
 */
function generateTestId(prefix = 'test') {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(2, 8);
    return `${prefix}_${timestamp}_${random}`;
}

module.exports = {
    TEST_CONFIG,
    makeEnhancedRequest,
    waitForApiReady,
    verifyApiEndpoints,
    cleanupWithRetry,
    executeWithConcurrencyLimit,
    sleep,
    aggregateTestResults,
    generateTestId
};
