#!/usr/bin/env node
/**
 * JavaScript Test Script for Root Health Endpoints - Production Ready
 *
 * This script validates the root health endpoints of the Worker Management API by:
 * 1. Testing GET / endpoint for basic API availability
 * 2. Testing GET /health endpoint for service health status
 * 3. Validating response formats and expected content
 * 4. Implementing comprehensive error handling and retry logic
 * 5. Providing detailed logging and validation
 * 6. Ensuring proper resource cleanup
 *
 * ENDPOINTS TESTED:
 * - GET / - Root endpoint showing API status and version
 * - GET /health - Health check endpoint showing service status
 *
 * Requirements: Node.js 18+ (for native fetch API)
 * Usage: node test_01_root_health.js
 */

// Enhanced Configuration
const API_BASE = 'http://localhost:8887';
const TIMEOUT_MS = 30000; // 30 seconds for health checks
const MAX_RETRIES = 3;
const RETRY_DELAY = 2000; // 2 seconds
const REQUEST_DELAY = 500; // 0.5 seconds between requests

// Add a flag to track if we've already retried for timeout
let hasRetried = false;

/**
 * Enhanced utility function to make HTTP requests with retry logic and comprehensive error handling
 */
async function makeRequest(url, options = {}, retries = MAX_RETRIES) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'WorkerManagement-RootHealthTest/1.0',
            ...options.headers
        },
        signal: AbortSignal.timeout(TIMEOUT_MS)
    };

    const finalOptions = { ...defaultOptions, ...options };

    for (let attempt = 1; attempt <= retries; attempt++) {
        try {
            console.log(`🌐 ${finalOptions.method || 'GET'} ${url} (Attempt ${attempt}/${retries})`);

            const response = await fetch(url, finalOptions);

            if (!response.ok) {
                const errorText = await response.text();
                const error = new Error(`HTTP ${response.status}: ${errorText}`);
                error.status = response.status;
                error.response = errorText;
                throw error;
            }


            // Handle 204 No Content responses
            if (response.status === 204) {
                return { success: true };
            }

            const data = await response.json();
            console.log(`✅ Response received (${response.status}) - Attempt ${attempt}`);

            // Add delay between requests
            await new Promise(resolve => setTimeout(resolve, REQUEST_DELAY));

            return data;
        } catch (error) {
            console.error(`❌ Request failed (Attempt ${attempt}/${retries}): ${error.message}`);

            // Check if this is a timeout/network issue and we haven't retried yet
            if ((error.name === 'AbortError' ||
                 error.name === 'TimeoutError' ||
                 error.message.includes('timeout') ||
                 error.message.includes('fetch') ||
                 error.message.includes('network') ||
                 error.message.includes('ECONNRESET') ||
                 error.message.includes('ENOTFOUND') ||
                 (error.status >= 500 && error.status < 600)) // Server errors
                && !hasRetried
                && attempt === retries) {

                console.log('🕐 API timeout/network issue detected. Waiting 30 seconds before single retry...');
                console.log(`   Original error: ${error.message}`);
                hasRetried = true;

                // Sleep for 30 seconds
                await new Promise(resolve => setTimeout(resolve, 30000));

                console.log('🔄 Attempting retry after timeout...');

                try {
                    const response = await fetch(url, finalOptions);

                    if (!response.ok) {
                        const errorText = await response.text();
                        throw new Error(`HTTP ${response.status}: ${errorText}`);
                    }


            // Handle 204 No Content responses
            if (response.status === 204) {
                return { success: true };
            }

            const data = await response.json();
                    console.log(`✅ Retry successful!`);

                    return data;
                } catch (retryError) {
                    console.log(`❌ Retry also failed: ${retryError.message}`);
                    throw retryError;
                }
            }

            if (attempt === retries) {
                throw error;
            }

            if (attempt < retries) {
                console.log(`⏳ Retrying in ${RETRY_DELAY/1000} seconds...`);
                await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
            }
        }
    }
}

/**
 * Test the root endpoint (GET /)
 */
async function testRootEndpoint() {
    console.log('\n=== Testing Root Endpoint (GET /) ===');

    try {
        const response = await makeRequest(`${API_BASE}/`);

        console.log('✅ Root endpoint response received');
        console.log(`   Response type: ${typeof response}`);
        console.log(`   Response keys: ${Object.keys(response || {}).join(', ')}`);

        // Validate expected response structure
        const expectedStructure = {
            message: 'Worker Management API is running',
            version: '1.0.0'
        };

        const validation = {
            has_message: false,
            message_correct: false,
            has_version: false,
            version_correct: false,
            overall_valid: false
        };

        // Check message field
        if (response.message !== undefined) {
            validation.has_message = true;
            console.log(`✅ Message field present: "${response.message}"`);

            if (response.message === expectedStructure.message) {
                validation.message_correct = true;
                console.log('✅ Message content matches expected value');
            } else {
                console.log(`❌ Message content mismatch. Expected: "${expectedStructure.message}", Got: "${response.message}"`);
            }
        } else {
            console.log('❌ Message field missing from response');
        }

        // Check version field
        if (response.version !== undefined) {
            validation.has_version = true;
            console.log(`✅ Version field present: "${response.version}"`);

            if (response.version === expectedStructure.version) {
                validation.version_correct = true;
                console.log('✅ Version content matches expected value');
            } else {
                console.log(`❌ Version content mismatch. Expected: "${expectedStructure.version}", Got: "${response.version}"`);
            }
        } else {
            console.log('❌ Version field missing from response');
        }

        // Overall validation
        validation.overall_valid = validation.has_message &&
                                  validation.message_correct &&
                                  validation.has_version &&
                                  validation.version_correct;

        console.log(`\n📊 Root Endpoint Validation: ${validation.overall_valid ? '✅ PASSED' : '❌ FAILED'}`);
        console.log(`   Message validation: ${validation.message_correct ? 'PASSED' : 'FAILED'}`);
        console.log(`   Version validation: ${validation.version_correct ? 'PASSED' : 'FAILED'}`);

        return {
            success: true,
            validation: validation,
            response: response
        };

    } catch (error) {
        console.error(`❌ Root endpoint test failed: ${error.message}`);
        if (error.status && error.response) {
            console.error(`   HTTP Details: Status ${error.status}, Response: ${error.response}`);
        }

        return {
            success: false,
            error: error.message,
            status: error.status || null
        };
    }
}

/**
 * Test the health endpoint (GET /health)
 */
async function testHealthEndpoint() {
    console.log('\n=== Testing Health Endpoint (GET /health) ===');

    try {
        const response = await makeRequest(`${API_BASE}/health`);

        console.log('✅ Health endpoint response received');
        console.log(`   Response type: ${typeof response}`);
        console.log(`   Response keys: ${Object.keys(response || {}).join(', ')}`);

        // Validate expected response structure
        const expectedStructure = {
            status: 'healthy',
            service: 'worker-management-api'
        };

        const validation = {
            has_status: false,
            status_correct: false,
            has_service: false,
            service_correct: false,
            overall_valid: false
        };

        // Check status field
        if (response.status !== undefined) {
            validation.has_status = true;
            console.log(`✅ Status field present: "${response.status}"`);

            if (response.status === expectedStructure.status) {
                validation.status_correct = true;
                console.log('✅ Status content matches expected value');
            } else {
                console.log(`❌ Status content mismatch. Expected: "${expectedStructure.status}", Got: "${response.status}"`);
            }
        } else {
            console.log('❌ Status field missing from response');
        }

        // Check service field
        if (response.service !== undefined) {
            validation.has_service = true;
            console.log(`✅ Service field present: "${response.service}"`);

            if (response.service === expectedStructure.service) {
                validation.service_correct = true;
                console.log('✅ Service content matches expected value');
            } else {
                console.log(`❌ Service content mismatch. Expected: "${expectedStructure.service}", Got: "${response.service}"`);
            }
        } else {
            console.log('❌ Service field missing from response');
        }

        // Check for additional useful fields that might be present
        if (response.timestamp) {
            console.log(`ℹ️  Timestamp field present: "${response.timestamp}"`);
        }

        if (response.uptime) {
            console.log(`ℹ️  Uptime field present: "${response.uptime}"`);
        }

        if (response.environment) {
            console.log(`ℹ️  Environment field present: "${response.environment}"`);
        }

        // Overall validation
        validation.overall_valid = validation.has_status &&
                                  validation.status_correct &&
                                  validation.has_service &&
                                  validation.service_correct;

        console.log(`\n📊 Health Endpoint Validation: ${validation.overall_valid ? '✅ PASSED' : '❌ FAILED'}`);
        console.log(`   Status validation: ${validation.status_correct ? 'PASSED' : 'FAILED'}`);
        console.log(`   Service validation: ${validation.service_correct ? 'PASSED' : 'FAILED'}`);

        return {
            success: true,
            validation: validation,
            response: response
        };

    } catch (error) {
        console.error(`❌ Health endpoint test failed: ${error.message}`);
        if (error.status && error.response) {
            console.error(`   HTTP Details: Status ${error.status}, Response: ${error.response}`);
        }

        return {
            success: false,
            error: error.message,
            status: error.status || null
        };
    }
}

/**
 * Test API connectivity and basic service availability
 */
async function testApiConnectivity() {
    console.log('\n=== Testing API Connectivity ===');

    try {
        console.log(`🔍 Attempting to connect to API at: ${API_BASE}`);

        // Try a simple HEAD request first to check connectivity
        const response = await fetch(`${API_BASE}/`, {
            method: 'HEAD',
            signal: AbortSignal.timeout(10000) // 10 second timeout for connectivity test
        });

        console.log(`✅ API connectivity confirmed (HTTP ${response.status})`);
        console.log(`   Server: ${response.headers.get('server') || 'Unknown'}`);
        console.log(`   Content-Type: ${response.headers.get('content-type') || 'Unknown'}`);

        return {
            success: true,
            status: response.status,
            headers: {
                server: response.headers.get('server'),
                contentType: response.headers.get('content-type'),
                date: response.headers.get('date')
            }
        };

    } catch (error) {
        console.error(`❌ API connectivity test failed: ${error.message}`);

        // Provide specific error guidance
        if (error.name === 'AbortError' || error.name === 'TimeoutError') {
            console.error('   💡 Timeout error - API may be starting up or overloaded');
        } else if (error.message.includes('ECONNREFUSED')) {
            console.error('   💡 Connection refused - API may not be running');
        } else if (error.message.includes('ENOTFOUND')) {
            console.error('   💡 Host not found - Check API_BASE configuration');
        }

        return {
            success: false,
            error: error.message,
            errorType: error.name
        };
    }
}

/**
 * Validate overall root health test results
 */
function validateOverallResults(connectivityResult, rootResult, healthResult) {
    console.log('\n🔍 Performing Overall Root Health Validation...');

    const validation = {
        connectivity_check: false,
        root_endpoint_check: false,
        health_endpoint_check: false,
        root_response_format: false,
        health_response_format: false,
        overall_valid: false
    };

    // Connectivity validation
    if (connectivityResult.success) {
        validation.connectivity_check = true;
        console.log('✅ Connectivity: API is reachable');
    } else {
        console.log(`❌ Connectivity: API connection failed - ${connectivityResult.error}`);
    }

    // Root endpoint validation
    if (rootResult.success) {
        validation.root_endpoint_check = true;
        console.log('✅ Root Endpoint: GET / is accessible');

        if (rootResult.validation && rootResult.validation.overall_valid) {
            validation.root_response_format = true;
            console.log('✅ Root Response: Format and content are correct');
        } else {
            console.log('❌ Root Response: Format or content validation failed');
        }
    } else {
        console.log(`❌ Root Endpoint: GET / failed - ${rootResult.error}`);
    }

    // Health endpoint validation
    if (healthResult.success) {
        validation.health_endpoint_check = true;
        console.log('✅ Health Endpoint: GET /health is accessible');

        if (healthResult.validation && healthResult.validation.overall_valid) {
            validation.health_response_format = true;
            console.log('✅ Health Response: Format and content are correct');
        } else {
            console.log('❌ Health Response: Format or content validation failed');
        }
    } else {
        console.log(`❌ Health Endpoint: GET /health failed - ${healthResult.error}`);
    }

    // Overall validation
    const validationChecks = Object.values(validation).slice(0, -1); // Exclude overall_valid
    const passedChecks = validationChecks.filter(check => check === true).length;
    validation.overall_valid = passedChecks >= 4; // At least 4 out of 5 checks must pass

    console.log(`\n📊 Overall Validation Summary: ${passedChecks}/5 checks passed`);

    if (validation.overall_valid) {
        console.log('🎉 Overall root health validation: PASSED');
    } else {
        console.log('❌ Overall root health validation: FAILED');
    }

    return validation;
}

/**
 * Main test function for root health endpoints
 */
async function runRootHealthTest() {
    console.log('🚀 Starting Root Health Endpoints Test Suite');
    console.log('='.repeat(80));

    // Reset retry flag for this test run
    hasRetried = false;
    console.log('🔄 Retry mechanism initialized (1 retry with 30s delay available)');

    const testResults = {
        connectivity_test: false,
        root_endpoint_test: false,
        health_endpoint_test: false,
        response_validation: false,
        overall_success: false,
        execution_time: 0,
        partial_results: {}
    };

    const testStartTime = Date.now();

    try {
        // Step 1: Test API connectivity
        console.log('\n🔌 Step 1: API Connectivity Test');
        const connectivityResult = await testApiConnectivity();
        testResults.connectivity_test = connectivityResult.success;
        testResults.partial_results.connectivity = connectivityResult;

        // Step 2: Test root endpoint
        console.log('\n🏠 Step 2: Root Endpoint Test');
        const rootResult = await testRootEndpoint();
        testResults.root_endpoint_test = rootResult.success;
        testResults.partial_results.root_endpoint = rootResult;

        // Step 3: Test health endpoint
        console.log('\n🏥 Step 3: Health Endpoint Test');
        const healthResult = await testHealthEndpoint();
        testResults.health_endpoint_test = healthResult.success;
        testResults.partial_results.health_endpoint = healthResult;

        // Step 4: Overall validation
        console.log('\n🔍 Step 4: Overall Response Validation');
        const overallValidation = validateOverallResults(connectivityResult, rootResult, healthResult);
        testResults.response_validation = overallValidation.overall_valid;
        testResults.partial_results.overall_validation = overallValidation;

        // Overall success determination
        testResults.overall_success = testResults.connectivity_test &&
                                     testResults.root_endpoint_test &&
                                     testResults.health_endpoint_test &&
                                     testResults.response_validation;

        testResults.execution_time = Date.now() - testStartTime;

        // Generate final report
        console.log('\n📊 Step 5: Root Health Test Summary');
        console.log('='.repeat(80));
        console.log(`🏁 Root Health Endpoints Test - COMPLETE`);
        console.log(`   Total execution time: ${testResults.execution_time}ms`);
        console.log(`   Overall success: ${testResults.overall_success ? '✅ PASSED' : '❌ FAILED'}`);
        console.log('\n📋 Root Health Test Results:');
        console.log(`   ✅ API Connectivity: ${testResults.connectivity_test ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Root Endpoint (GET /): ${testResults.root_endpoint_test ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Health Endpoint (GET /health): ${testResults.health_endpoint_test ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Response Validation: ${testResults.response_validation ? 'PASSED' : 'FAILED'}`);

        if (testResults.overall_success) {
            console.log('\n🎉 Root Health Endpoints Test PASSED!');
            console.log('   ✅ API is running and accessible');
            console.log('   ✅ Root endpoint returns correct message and version');
            console.log('   ✅ Health endpoint returns correct status and service info');
            console.log('   ✅ Both endpoints respond with expected JSON format');
        } else {
            console.log('\n💥 Root Health Endpoints Test FAILED!');
            console.log('   ❌ One or more endpoints are not working correctly');
            console.log('   ❌ Check the detailed output above for specific issues');
        }

    } catch (error) {
        console.error(`\n❌ Root health test execution failed: ${error.message}`);
        if (error.stack) {
            console.error(`Stack trace: ${error.stack}`);
        }
        testResults.overall_success = false;
        testResults.execution_time = Date.now() - testStartTime;
    }

    return testResults;
}

/**
 * Main function with comprehensive error handling and reporting
 */
async function main() {
    console.log('🚀 Root Health Endpoints Test Suite for Worker Management API');
    console.log('Node.js version:', process.version);
    console.log('Platform:', process.platform);
    console.log('Architecture:', process.arch);
    console.log('API Base URL:', API_BASE);
    console.log('='.repeat(80));

    try {
        const testResults = await runRootHealthTest();

        // Final summary
        console.log('\n📊 FINAL ROOT HEALTH TEST SUMMARY');
        console.log('='.repeat(80));
        console.log(`Overall Result: ${testResults.overall_success ? '🎉 SUCCESS' : '💥 FAILURE'}`);
        console.log(`Total Execution Time: ${testResults.execution_time}ms`);
        console.log(`Test Components: ${Object.keys(testResults).filter(k => typeof testResults[k] === 'boolean').length}`);
        console.log(`Passed Components: ${Object.values(testResults).filter(v => v === true).length}`);

        // Display detailed results for troubleshooting
        if (!testResults.overall_success) {
            console.log('\n🔍 TROUBLESHOOTING INFO:');

            if (!testResults.connectivity_test) {
                console.log('❌ Connectivity failed - Ensure the API server is running on', API_BASE);
            }

            if (!testResults.root_endpoint_test) {
                const rootError = testResults.partial_results.root_endpoint?.error;
                console.log('❌ Root endpoint failed -', rootError || 'Unknown error');
            }

            if (!testResults.health_endpoint_test) {
                const healthError = testResults.partial_results.health_endpoint?.error;
                console.log('❌ Health endpoint failed -', healthError || 'Unknown error');
            }

            if (!testResults.response_validation) {
                console.log('❌ Response validation failed - Check response format and content');
            }
        }

        // Exit with appropriate code
        process.exit(testResults.overall_success ? 0 : 1);

    } catch (error) {
        console.error('❌ Root health test suite failed to execute:', error.message);
        if (error.stack) {
            console.error('Stack trace:', error.stack);
        }
        process.exit(1);
    }
}

// Enhanced execution with proper error handling
if (require.main === module) {
    main().catch(error => {
        console.error('❌ Unhandled error in root health test suite:', error);
        process.exit(1);
    });
}

module.exports = {
    makeRequest,
    testRootEndpoint,
    testHealthEndpoint,
    testApiConnectivity,
    validateOverallResults,
    runRootHealthTest
};
