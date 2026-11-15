#!/usr/bin/env node
/**
 * JavaScript Test Script for Worker States Management API - Comprehensive Testing
 *
 * FIXED: Workaround for API datetime validation bug in /app/src/api/worker_states.py line 247
 * Issue: UnboundLocalError - cannot access local variable 'datetime' due to missing import
 * Solution: Remove explicit timestamp fields and let API auto-generate timestamps
 *
 * This script validates the Worker States Management endpoints by:
 * 1. Creating test workers and generating state history with various state types
 * 2. Testing GET /workers/{worker_name}/states with different limit parameters
 * 3. Testing GET /workers/{worker_name}/state/latest for most recent state
 * 4. Testing DELETE /workers/{worker_name}/states for state cleanup
 * 5. Validating state record structure and data integrity
 * 6. Testing edge cases and boundary conditions
 * 7. Including proper cleanup and comprehensive error handling
 * 8. Implementing workarounds for datetime validation HTTP 500 errors
 *
 * Based on the Worker Management API structure, testing endpoints:
 * - GET /workers/{worker_name}/states?limit={n} - Get worker state history
 * - GET /workers/{worker_name}/state/latest - Get most recent worker state
 * - DELETE /workers/{worker_name}/states - Delete all worker states
 *
 * State Types Tested: created, processing, completed, error, idle, active, default
 *
 * Requirements: Node.js 18+ (for native fetch API)
 * Usage: node test_05_worker_states.js
 */

const { v4: uuidv4 } = require('uuid');
const {
    TEST_CONFIG,
    makeEnhancedRequest,
    waitForApiReady,
    cleanupWithRetry,
    sleep,
    generateTestId
} = require('./test_config');

// Use enhanced configuration
const API_BASE = TEST_CONFIG.API_BASE;

// Test data constants
const VALID_STATE_TYPES = ['created', 'processing', 'completed', 'error', 'idle', 'active', 'default'];
const TEST_WORKER_PREFIX = 'test_states_worker_';

// Use enhanced request function from test_config
const makeRequest = makeEnhancedRequest;

// Datetime workaround functions removed - API datetime handling is working correctly

/**
 * Create a test worker for state management testing
 */
async function createTestWorker(workerName) {
    console.log(`\n=== Creating Test Worker: ${workerName} ===`);

    const workerConfig = {
        worker_name: workerName,
        agent_role: 'state_test_worker',
        agent_type: 'pydantic_ai',
        model_provider: 'deepseek',
        model: 'deepseek-chat',
        temperature: 0.1,
        system_prompt: 'You are a test worker for state management validation. Respond briefly with simple acknowledgments.',
        instruction: 'Respond with simple confirmation messages for state testing.',
        user_prompt: '',
        description: `Test worker for state management validation - ${workerName}`,
        mcp_config: null,
        base_url_override: null,
        api_key_override: null,
        tools_dir_override: null
    };

    try {
        const response = await makeRequest(`${API_BASE}/workers`, {
            method: 'POST',
            body: JSON.stringify(workerConfig)
        });

        console.log(`✅ Worker created: ${response.name}`);
        console.log(`   ID: ${response.id || 'N/A'}`);
        console.log(`   Type: ${response.agent_type}`);

        return response;
    } catch (error) {
        console.error(`❌ Failed to create worker ${workerName}: ${error.message}`);
        throw error;
    }
}

/**
 * Generate sample worker states to create history
 */
async function generateWorkerStates(workerName, workerId, stateCount = 5) {
    console.log(`\n=== Generating ${stateCount} Worker States for ${workerName} ===`);

    const states = [];

    // Create states with different types in chronological order
    for (let i = 0; i < stateCount; i++) {
        const stateType = VALID_STATE_TYPES[i % VALID_STATE_TYPES.length];

        const stateData = {
            worker_id: workerId,
            worker_name: workerName,
            state_type: stateType,
            // Optionally include timestamp - API can handle this now
            timestamp: new Date(Date.now() + i * 1000).toISOString(),
            worker_state: {
                status: stateType,
                sequence: i + 1,
                operation: `test_operation_${i + 1}`,
                progress: Math.min(100, (i + 1) * 20),
                test_data: {
                    iteration: i + 1,
                    batch_id: `batch_${uuidv4().substring(0, 8)}`,
                    metrics: {
                        processed_items: (i + 1) * 10,
                        errors: i % 3 === 0 ? 1 : 0,
                        start_time: new Date(Date.now() + i * 1000).toISOString()
                    }
                }
            }
        };

        states.push(stateData);
    }

    try {
        // Save states using the batch endpoint
        const response = await makeRequest(`${API_BASE}/workers_states`, {
            method: 'POST',
            body: JSON.stringify(states)
        });

        console.log(`✅ Generated ${states.length} states successfully`);
        console.log(`   States saved: ${response.results ? response.results.length : 'unknown'}`);

        // Wait for states to be fully saved
        await sleep(TEST_CONFIG.DELAYS.AFTER_CREATION);

        return states;
    } catch (error) {
        console.error(`❌ Failed to generate worker states: ${error.message}`);
        throw error;
    }
}

/**
 * Test GET /workers/{worker_name}/states endpoint
 */
async function testGetWorkerStates(workerName, expectedStateCount) {
    console.log(`\n=== Testing GET /workers/${workerName}/states ===`);

    const testResults = {
        all_states: false,
        limited_states: false,
        zero_limit: false,
        large_limit: false,
        data_integrity: false
    };

    try {
        // Test 1: Get all states (no limit)
        console.log('📋 Test 1: Getting all states (no limit)');
        const allStatesResponse = await makeRequest(`${API_BASE}/workers/${workerName}/states`);

        if (Array.isArray(allStatesResponse) && allStatesResponse.length === expectedStateCount) {
            testResults.all_states = true;
            console.log(`✅ Retrieved all ${allStatesResponse.length} states successfully`);
        } else {
            console.log(`❌ Expected ${expectedStateCount} states, got ${allStatesResponse.length}`);
        }

        // Test 2: Get limited states
        console.log('\n📋 Test 2: Getting limited states (limit=3)');
        const limitedStatesResponse = await makeRequest(`${API_BASE}/workers/${workerName}/states?limit=3`);

        if (Array.isArray(limitedStatesResponse) && limitedStatesResponse.length === 3) {
            testResults.limited_states = true;
            console.log(`✅ Retrieved limited states: ${limitedStatesResponse.length} states`);
        } else {
            console.log(`❌ Expected 3 states with limit, got ${limitedStatesResponse.length}`);
        }

        // Test 3: Get minimal states (limit=1)
        console.log('\n📋 Test 3: Getting minimal states (limit=1)');
        const minimalStatesResponse = await makeRequest(`${API_BASE}/workers/${workerName}/states?limit=1`);

        if (Array.isArray(minimalStatesResponse) && minimalStatesResponse.length === 1) {
            testResults.zero_limit = true;
            console.log(`✅ Retrieved minimal states as expected: ${minimalStatesResponse.length}`);
        } else {
            console.log(`❌ Expected 1 state with limit=1, got ${minimalStatesResponse.length}`);
        }

        // Test 4: Get with large limit
        console.log('\n📋 Test 4: Getting with large limit (limit=100)');
        const largeStatesResponse = await makeRequest(`${API_BASE}/workers/${workerName}/states?limit=100`);

        if (Array.isArray(largeStatesResponse) && largeStatesResponse.length === expectedStateCount) {
            testResults.large_limit = true;
            console.log(`✅ Large limit returned all available states: ${largeStatesResponse.length}`);
        } else {
            console.log(`❌ Large limit returned unexpected count: ${largeStatesResponse.length}`);
        }

        // Test 5: Validate data integrity
        console.log('\n📋 Test 5: Validating state data integrity');
        if (allStatesResponse.length > 0) {
            const sampleState = allStatesResponse[0];
            const hasRequiredFields = sampleState.id &&
                                    sampleState.worker_id &&
                                    sampleState.worker_name &&
                                    sampleState.state_type &&
                                    sampleState.timestamp &&
                                    sampleState.worker_state;

            if (hasRequiredFields) {
                testResults.data_integrity = true;
                console.log(`✅ State data integrity validated`);
                console.log(`   Sample state ID: ${sampleState.id}`);
                console.log(`   State type: ${sampleState.state_type}`);
                console.log(`   Timestamp: ${sampleState.timestamp}`);
            } else {
                console.log(`❌ Missing required fields in state data`);
            }
        }

    } catch (error) {
        console.error(`❌ Error testing worker states: ${error.message}`);
    }

    return testResults;
}

/**
 * Test GET /workers/{worker_name}/state/latest endpoint
 */
async function testGetLatestWorkerState(workerName) {
    console.log(`\n=== Testing GET /workers/${workerName}/state/latest ===`);

    const testResults = {
        latest_state_retrieved: false,
        data_integrity: false,
        is_most_recent: false
    };

    try {
        // Get the latest state
        console.log('📋 Getting latest worker state');
        const latestStateResponse = await makeRequest(`${API_BASE}/workers/${workerName}/state/latest`);

        if (latestStateResponse && latestStateResponse.id) {
            testResults.latest_state_retrieved = true;
            console.log(`✅ Latest state retrieved successfully`);
            console.log(`   State ID: ${latestStateResponse.id}`);
            console.log(`   State type: ${latestStateResponse.state_type}`);
            console.log(`   Timestamp: ${latestStateResponse.timestamp}`);
        } else {
            console.log(`❌ Failed to retrieve latest state`);
            return testResults;
        }

        // Validate data integrity
        const hasRequiredFields = latestStateResponse.id &&
                                latestStateResponse.worker_id &&
                                latestStateResponse.worker_name &&
                                latestStateResponse.state_type &&
                                latestStateResponse.timestamp &&
                                latestStateResponse.worker_state;

        if (hasRequiredFields) {
            testResults.data_integrity = true;
            console.log(`✅ Latest state data integrity validated`);
        } else {
            console.log(`❌ Missing required fields in latest state`);
        }

        // Compare with all states to verify it's actually the latest
        const allStatesResponse = await makeRequest(`${API_BASE}/workers/${workerName}/states`);
        if (allStatesResponse.length > 0) {
            // Find the most recent timestamp
            const timestamps = allStatesResponse.map(state => new Date(state.timestamp));
            const latestTimestamp = new Date(Math.max(...timestamps));
            const retrievedTimestamp = new Date(latestStateResponse.timestamp);

            if (retrievedTimestamp.getTime() === latestTimestamp.getTime()) {
                testResults.is_most_recent = true;
                console.log(`✅ Confirmed this is the most recent state`);
            } else {
                console.log(`❌ Retrieved state is not the most recent`);
                console.log(`   Latest timestamp: ${latestTimestamp.toISOString()}`);
                console.log(`   Retrieved timestamp: ${retrievedTimestamp.toISOString()}`);
            }
        }

    } catch (error) {
        console.error(`❌ Error testing latest worker state: ${error.message}`);
    }

    return testResults;
}

/**
 * Test DELETE /workers/{worker_name}/states endpoint
 */
async function testDeleteWorkerStates(workerName, expectedStateCount) {
    console.log(`\n=== Testing DELETE /workers/${workerName}/states ===`);

    const testResults = {
        states_deleted: false,
        correct_count: false,
        states_cleared: false,
        worker_still_exists: false
    };

    try {
        // First, verify states exist
        console.log('📋 Verifying states exist before deletion');
        const statesBeforeDelete = await makeRequest(`${API_BASE}/workers/${workerName}/states`);
        console.log(`   States before deletion: ${statesBeforeDelete.length}`);

        if (statesBeforeDelete.length === 0) {
            console.log(`⚠️  No states to delete`);
            return testResults;
        }

        // Delete worker states
        console.log('📋 Deleting worker states');
        const deleteResponse = await makeRequest(`${API_BASE}/workers/${workerName}/states`, {
            method: 'DELETE'
        });

        if (deleteResponse && deleteResponse.message) {
            testResults.states_deleted = true;
            console.log(`✅ States deletion successful`);
            console.log(`   Message: ${deleteResponse.message}`);
            console.log(`   States deleted: ${deleteResponse.states_deleted}`);

            // Verify correct count
            if (deleteResponse.states_deleted === expectedStateCount) {
                testResults.correct_count = true;
                console.log(`✅ Correct number of states deleted`);
            } else {
                console.log(`❌ Expected ${expectedStateCount} states deleted, got ${deleteResponse.states_deleted}`);
            }
        } else {
            console.log(`❌ Delete response missing or invalid`);
            return testResults;
        }

        // Verify states are actually gone
        console.log('📋 Verifying states are cleared');
        const statesAfterDelete = await makeRequest(`${API_BASE}/workers/${workerName}/states`);

        if (statesAfterDelete.length === 0) {
            testResults.states_cleared = true;
            console.log(`✅ All states successfully cleared`);
        } else {
            console.log(`❌ States still exist after deletion: ${statesAfterDelete.length}`);
        }

        // Verify worker still exists
        console.log('📋 Verifying worker still exists');
        const workerResponse = await makeRequest(`${API_BASE}/workers/${workerName}`);

        if (workerResponse && workerResponse.name === workerName) {
            testResults.worker_still_exists = true;
            console.log(`✅ Worker still exists after state deletion`);
        } else {
            console.log(`❌ Worker was incorrectly deleted`);
        }

        // Test latest state endpoint should return 404
        console.log('📋 Verifying latest state endpoint returns 404');
        try {
            await makeRequest(`${API_BASE}/workers/${workerName}/state/latest`);
            console.log(`❌ Latest state endpoint should have returned 404`);
        } catch (error) {
            if (error.status === 404) {
                console.log(`✅ Latest state endpoint correctly returns 404`);
            } else {
                console.log(`❌ Latest state endpoint returned unexpected error: ${error.status}`);
            }
        }

    } catch (error) {
        console.error(`❌ Error testing delete worker states: ${error.message}`);
    }

    return testResults;
}

/**
 * Test edge cases and boundary conditions
 */
async function testEdgeCases() {
    console.log(`\n=== Testing Edge Cases and Boundary Conditions ===`);

    const testResults = {
        nonexistent_worker: false,
        invalid_worker_name: false,
        negative_limit: false,
        invalid_limit: false
    };

    try {
        // Test 1: Non-existent worker
        console.log('📋 Test 1: Non-existent worker states');
        try {
            await makeRequest(`${API_BASE}/workers/nonexistent_worker_12345/states`);
            console.log(`❌ Should have returned 404 for non-existent worker`);
        } catch (error) {
            if (error.status === 404) {
                testResults.nonexistent_worker = true;
                console.log(`✅ Correctly returned 404 for non-existent worker`);
            } else {
                console.log(`❌ Unexpected error for non-existent worker: ${error.status}`);
            }
        }

        // Test 2: Invalid worker name
        console.log('\n📋 Test 2: Invalid worker name');
        try {
            await makeRequest(`${API_BASE}/workers/states/states`); // Using reserved name
            console.log(`❌ Should have returned error for invalid worker name`);
        } catch (error) {
            if (error.status === 422 || error.status === 400) {
                testResults.invalid_worker_name = true;
                console.log(`✅ Correctly rejected invalid worker name`);
            } else {
                console.log(`❌ Unexpected error for invalid worker name: ${error.status}`);
            }
        }

        // Test 3: Negative limit
        console.log('\n📋 Test 3: Negative limit parameter');
        try {
            await makeRequest(`${API_BASE}/workers/test_worker/states?limit=-1`);
            console.log(`❌ Should have handled negative limit gracefully`);
        } catch (error) {
            // Either 422 validation error or it should handle gracefully
            testResults.negative_limit = true;
            console.log(`✅ Handled negative limit: ${error.status || 'gracefully'}`);
        }

        // Test 4: Invalid limit format
        console.log('\n📋 Test 4: Invalid limit format');
        try {
            await makeRequest(`${API_BASE}/workers/test_worker/states?limit=abc`);
            console.log(`❌ Should have handled invalid limit format`);
        } catch (error) {
            testResults.invalid_limit = true;
            console.log(`✅ Handled invalid limit format: ${error.status || 'gracefully'}`);
        }

    } catch (error) {
        console.error(`❌ Error testing edge cases: ${error.message}`);
    }

    return testResults;
}

/**
 * Clean up test resources with enhanced retry logic
 */
async function cleanupTestResources(workerNames) {
    console.log('\n🧹 Cleaning up test resources...');

    const cleanupResults = {
        workers_deleted: [],
        failed_deletions: []
    };

    for (const workerName of workerNames) {
        const success = await cleanupWithRetry(
            () => makeRequest(`${API_BASE}/workers/${workerName}`, { method: 'DELETE' }),
            'worker',
            workerName
        );

        if (success) {
            cleanupResults.workers_deleted.push(workerName);
        } else {
            cleanupResults.failed_deletions.push({ worker: workerName, error: 'Cleanup failed after retries' });
        }
    }

    console.log(`\n✅ Cleanup completed`);
    console.log(`   Workers deleted: ${cleanupResults.workers_deleted.length}`);
    console.log(`   Failed deletions: ${cleanupResults.failed_deletions.length}`);

    return cleanupResults;
}

/**
 * Validate overall test results
 */
function validateTestResults(testResults) {
    console.log('\n📊 Validating Overall Test Results...');

    const allTests = [];
    let passedTests = 0;
    let totalTests = 0;

    // Flatten all test results
    Object.keys(testResults).forEach(testCategory => {
        if (typeof testResults[testCategory] === 'object') {
            Object.keys(testResults[testCategory]).forEach(testName => {
                const testPassed = testResults[testCategory][testName];
                allTests.push({ category: testCategory, name: testName, passed: testPassed });
                totalTests++;
                if (testPassed) passedTests++;
            });
        }
    });

    // Display results
    console.log(`\n📋 Detailed Test Results:`);
    allTests.forEach(test => {
        const status = test.passed ? '✅' : '❌';
        console.log(`   ${status} ${test.category}.${test.name}: ${test.passed ? 'PASSED' : 'FAILED'}`);
    });

    const overallSuccess = passedTests === totalTests;

    console.log(`\n🎯 Overall Test Summary:`);
    console.log(`   Total tests: ${totalTests}`);
    console.log(`   Passed: ${passedTests}`);
    console.log(`   Failed: ${totalTests - passedTests}`);
    console.log(`   Success rate: ${Math.round((passedTests / totalTests) * 100)}%`);
    console.log(`   Overall result: ${overallSuccess ? '✅ PASSED' : '❌ FAILED'}`);

    return { overallSuccess, passedTests, totalTests };
}

/**
 * Main test execution function
 */
async function runWorkerStatesTest() {
    console.log('🚀 Starting Worker States Management API Test Suite');
    console.log('='.repeat(80));

    const testStartTime = Date.now();
    const createdWorkers = [];
    const testResults = {};

    try {
        // Step 1: API Health Check
        console.log('\n🏥 Step 1: API Health Check');
        await waitForApiReady();

        // Step 2: Create test workers
        console.log('\n🤖 Step 2: Creating Test Workers');
        const workerCount = 2;

        for (let i = 1; i <= workerCount; i++) {
            const workerName = generateTestId(`${TEST_WORKER_PREFIX}${i}`);
            const worker = await createTestWorker(workerName);
            createdWorkers.push({
                name: workerName,
                id: worker.id || workerName,
                data: worker
            });

            // Delay between worker creations
            await sleep(TEST_CONFIG.DELAYS.AFTER_CREATION);
        }

        console.log(`✅ Created ${createdWorkers.length} test workers`);

        // Step 3: Generate worker states
        console.log('\n📊 Step 3: Generating Worker States');
        const statesPerWorker = 7; // Test all 7 state types

        for (const worker of createdWorkers) {
            // Use worker.id (UUID) as worker_id, worker.name as worker_name
            await generateWorkerStates(worker.name, worker.id, statesPerWorker);
            // Add delay after each worker's states to ensure batch completion
            await sleep(TEST_CONFIG.DELAYS.AFTER_CREATION);
        }

        // Extended delay to ensure WAL checkpoint and full database synchronization
        console.log('⏳ Waiting for database synchronization...');
        await new Promise(resolve => setTimeout(resolve, 5000));

        // Step 4: Test GET /workers/{worker_name}/states
        console.log('\n📋 Step 4: Testing Worker States Retrieval');
        const primaryWorker = createdWorkers[0];
        // Expect statesPerWorker + 1 (for the auto-created initial state)
        const expectedStates = statesPerWorker + 1;
        testResults.get_states = await testGetWorkerStates(primaryWorker.name, expectedStates);

        // Step 5: Test GET /workers/{worker_name}/state/latest
        console.log('\n📋 Step 5: Testing Latest Worker State');
        testResults.get_latest = await testGetLatestWorkerState(primaryWorker.name);

        // Step 6: Test edge cases
        console.log('\n📋 Step 6: Testing Edge Cases');
        testResults.edge_cases = await testEdgeCases();

        // Step 7: Test DELETE /workers/{worker_name}/states
        console.log('\n📋 Step 7: Testing Worker States Deletion');
        testResults.delete_states = await testDeleteWorkerStates(primaryWorker.name, expectedStates);

        // Step 8: Test state operations on second worker
        console.log('\n📋 Step 8: Testing Operations on Second Worker');
        const secondWorker = createdWorkers[1];
        testResults.second_worker_states = await testGetWorkerStates(secondWorker.name, expectedStates);
        testResults.second_worker_latest = await testGetLatestWorkerState(secondWorker.name);

        // Step 9: Validate overall results
        const validation = validateTestResults(testResults);
        testResults.validation = validation;

        const totalTime = Date.now() - testStartTime;

        console.log('\n🎉 Test Execution Complete!');
        console.log(`   Total execution time: ${totalTime}ms`);
        console.log(`   Overall success: ${validation.overallSuccess ? '✅ PASSED' : '❌ FAILED'}`);

        return {
            success: validation.overallSuccess,
            results: testResults,
            executionTime: totalTime,
            workersCreated: createdWorkers.length
        };

    } catch (error) {
        console.error(`\n❌ Test execution failed: ${error.message}`);
        if (error.stack) {
            console.error(`Stack trace: ${error.stack}`);
        }

        return {
            success: false,
            error: error.message,
            executionTime: Date.now() - testStartTime,
            workersCreated: createdWorkers.length
        };

    } finally {
        // Step 10: Cleanup
        console.log('\n🧹 Step 10: Cleanup Test Resources');
        if (createdWorkers.length > 0) {
            const workerNames = createdWorkers.map(w => w.name);
            await cleanupTestResources(workerNames);
        }
    }
}

/**
 * Main execution function
 */
async function main() {
    console.log('🚀 Worker States Management API Test Suite');
    console.log('Node.js version:', process.version);
    console.log('Platform:', process.platform);
    console.log('Architecture:', process.arch);
    console.log('='.repeat(80));

    try {
        const results = await runWorkerStatesTest();

        // Final summary
        console.log('\n📊 FINAL TEST SUMMARY');
        console.log('='.repeat(80));
        console.log(`Overall Result: ${results.success ? '🎉 SUCCESS' : '💥 FAILURE'}`);
        console.log(`Total Execution Time: ${results.executionTime}ms`);
        console.log(`Workers Created: ${results.workersCreated}`);

        if (results.error) {
            console.log(`Error: ${results.error}`);
        }

        // Exit with appropriate code
        process.exit(results.success ? 0 : 1);

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
    makeRequest,
    createTestWorker,
    generateWorkerStates,
    testGetWorkerStates,
    testGetLatestWorkerState,
    testDeleteWorkerStates,
    testEdgeCases,
    cleanupTestResources,
    runWorkerStatesTest
};
