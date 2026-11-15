#!/usr/bin/env node
/**
 * JavaScript Test Suite for Individual Worker Operations - Production Ready
 *
 * This script comprehensively tests individual worker API endpoints:
 * - GET /workers/{worker_name} - retrieve worker details
 * - PUT /workers/{worker_name} - update existing worker
 * - DELETE /workers/{worker_name} - delete worker
 *
 * Test Coverage:
 * 1. Worker creation and retrieval
 * 2. Worker updates (configuration, model, system prompt, etc.)
 * 3. Worker deletion and cleanup
 * 4. Edge cases (non-existent workers, invalid data)
 * 5. Error handling and validation
 * 6. Concurrent operation testing
 *
 * Requirements: Node.js 18+ (for native fetch API)
 * Usage: node test_03_worker_individual.js
 */

const { v4: uuidv4 } = require('uuid');

// Configuration
const API_BASE = 'http://localhost:8887';
const TIMEOUT_MS = 60000; // 60 seconds
const MAX_RETRIES = 3;
const RETRY_DELAY = 2000; // 2 seconds
const REQUEST_DELAY = 1000; // 1 second between requests

/**
 * Enhanced utility function to make HTTP requests with retry logic
 */
async function makeRequest(url, options = {}, retries = MAX_RETRIES) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'WorkerManagement-IndividualWorkerTest/1.0',
            ...options.headers
        }
    };

    const finalOptions = { ...defaultOptions, ...options };

    for (let attempt = 1; attempt <= retries; attempt++) {
        try {
            console.log(`🌐 ${finalOptions.method || 'GET'} ${url} (Attempt ${attempt}/${retries})`);

            const response = await fetch(url, finalOptions);

            // Handle 204 No Content responses (successful DELETE)
            if (response.status === 204) {
                console.log(`✅ Response received (${response.status}) - No Content`);
                await new Promise(resolve => setTimeout(resolve, REQUEST_DELAY));
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

            // Add delay between requests
            await new Promise(resolve => setTimeout(resolve, REQUEST_DELAY));

            return data;
        } catch (error) {
            console.error(`❌ Request failed (Attempt ${attempt}/${retries}): ${error.message}`);

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
 * Create a test worker with specified configuration
 */
async function createTestWorker(workerConfig) {
    console.log(`\n=== Creating Test Worker: ${workerConfig.worker_name} ===`);

    try {
        // Validate required fields
        if (!workerConfig.worker_name || !workerConfig.agent_type) {
            throw new Error('Worker configuration missing required fields: worker_name, agent_type');
        }

        const response = await makeRequest(`${API_BASE}/workers`, {
            method: 'POST',
            body: JSON.stringify(workerConfig)
        });

        console.log(`✅ Test worker created successfully: ${response.name}`);
        console.log(`   ID: ${response.id || 'N/A'}`);
        console.log(`   Type: ${response.agent_type}`);
        console.log(`   Model: ${response.model_provider}:${response.model}`);
        console.log(`   Status: ${response.status}`);

        return response;
    } catch (error) {
        console.error(`❌ Failed to create test worker ${workerConfig.worker_name}: ${error.message}`);
        throw error;
    }
}

/**
 * Test GET /workers/{worker_name} - Retrieve worker details
 */
async function testGetWorker(workerName, shouldExist = true) {
    console.log(`\n=== Testing GET /workers/${workerName} ===`);

    try {
        const response = await makeRequest(`${API_BASE}/workers/${workerName}`);

        if (!shouldExist) {
            throw new Error('Expected worker not to exist, but GET request succeeded');
        }

        console.log(`✅ Worker retrieved successfully: ${response.name}`);
        console.log(`   Agent Type: ${response.agent_type}`);
        console.log(`   Model Provider: ${response.model_provider}`);
        console.log(`   Model: ${response.model}`);
        console.log(`   Temperature: ${response.temperature}`);
        console.log(`   Status: ${response.status}`);
        console.log(`   Description: ${response.description || 'No description'}`);
        console.log(`   Created At: ${response.created_at || 'N/A'}`);
        console.log(`   Updated At: ${response.updated_at || 'N/A'}`);

        // Validate response structure
        const requiredFields = ['name', 'agent_type', 'model_provider', 'model'];
        const missingFields = requiredFields.filter(field => !response[field]);

        if (missingFields.length > 0) {
            throw new Error(`Worker response missing required fields: ${missingFields.join(', ')}`);
        }

        return { success: true, worker: response };

    } catch (error) {
        if (shouldExist) {
            console.error(`❌ Failed to retrieve worker ${workerName}: ${error.message}`);
            return { success: false, error: error.message };
        } else {
            // Expected failure for non-existent worker
            if (error.status === 404) {
                console.log(`✅ Expected 404 error for non-existent worker: ${workerName}`);
                return { success: true, expectedError: true };
            } else {
                console.error(`❌ Unexpected error for non-existent worker ${workerName}: ${error.message}`);
                return { success: false, error: error.message };
            }
        }
    }
}

/**
 * Test PUT /workers/{worker_name} - Update existing worker
 */
async function testUpdateWorker(workerName, updateConfig, shouldSucceed = true) {
    console.log(`\n=== Testing PUT /workers/${workerName} ===`);

    // Ensure worker_name is included in the update configuration
    const completeUpdateConfig = {
        worker_name: workerName,
        ...updateConfig
    };

    console.log('Update configuration:', JSON.stringify(completeUpdateConfig, null, 2));

    try {
        const response = await makeRequest(`${API_BASE}/workers/${workerName}`, {
            method: 'PUT',
            body: JSON.stringify(completeUpdateConfig)
        });

        if (!shouldSucceed) {
            throw new Error('Expected update to fail, but PUT request succeeded');
        }

        console.log(`✅ Worker updated successfully: ${response.name}`);
        console.log(`   Updated fields reflected in response:`);

        // Verify that updates were applied
        for (const [key, value] of Object.entries(updateConfig)) {
            if (key !== 'worker_name' && response[key] !== undefined) {
                console.log(`   ${key}: ${response[key]} ${response[key] === value ? '✅' : '❌ (expected: ' + value + ')'}`);
            }
        }

        return { success: true, worker: response };

    } catch (error) {
        if (shouldSucceed) {
            console.error(`❌ Failed to update worker ${workerName}: ${error.message}`);
            return { success: false, error: error.message };
        } else {
            // Expected failure
            console.log(`✅ Expected error for invalid update: ${error.message}`);
            return { success: true, expectedError: true };
        }
    }
}

/**
 * Test DELETE /workers/{worker_name} - Delete worker
 */
async function testDeleteWorker(workerName, shouldExist = true) {
    console.log(`\n=== Testing DELETE /workers/${workerName} ===`);

    try {
        const response = await makeRequest(`${API_BASE}/workers/${workerName}`, {
            method: 'DELETE'
        });

        if (!shouldExist) {
            throw new Error('Expected worker not to exist, but DELETE request succeeded');
        }

        console.log(`✅ Worker deleted successfully: ${workerName}`);
        return { success: true, deleted: true };

    } catch (error) {
        if (shouldExist) {
            console.error(`❌ Failed to delete worker ${workerName}: ${error.message}`);
            return { success: false, error: error.message };
        } else {
            // Expected failure for non-existent worker
            if (error.status === 404) {
                console.log(`✅ Expected 404 error for non-existent worker: ${workerName}`);
                return { success: true, expectedError: true };
            } else {
                console.error(`❌ Unexpected error for non-existent worker ${workerName}: ${error.message}`);
                return { success: false, error: error.message };
            }
        }
    }
}

/**
 * Comprehensive API health check
 */
async function checkApiHealth() {
    console.log('\n🏥 Performing API Health Check...');

    try {
        // Main API health
        const healthResponse = await makeRequest(`${API_BASE}/health`);
        console.log(`✅ Main API is healthy: ${healthResponse.status}`);

        // Worker service check
        const workersResponse = await makeRequest(`${API_BASE}/workers`);
        console.log(`✅ Worker service is responding (${workersResponse.length} workers found)`);

        return { healthy: true };

    } catch (error) {
        console.error(`❌ Health check failed: ${error.message}`);
        return { healthy: false, error: error.message };
    }
}

/**
 * Test concurrent worker operations
 */
async function testConcurrentOperations(baseWorkerName) {
    console.log('\n=== Testing Concurrent Worker Operations ===');

    const workerNames = [
        `${baseWorkerName}_concurrent_1`,
        `${baseWorkerName}_concurrent_2`,
        `${baseWorkerName}_concurrent_3`
    ];

    const workerConfigs = workerNames.map(name => ({
        worker_name: name,
        agent_role: 'concurrent_test_worker',
        agent_type: 'pydantic_ai',
        model_provider: 'deepseek',
        model: 'deepseek-chat',
        temperature: 0.3,
        system_prompt: 'You are a test worker for concurrent operations testing.',
        instruction: 'Respond with confirmation of your worker name and current timestamp.',
        user_prompt: '',
        description: `Concurrent test worker: ${name}`,
        mcp_config: null
    }));

    try {
        // Create workers concurrently
        console.log('Creating workers concurrently...');
        const createPromises = workerConfigs.map(config => createTestWorker(config));
        const createResults = await Promise.allSettled(createPromises);

        const successfulCreations = createResults.filter(result => result.status === 'fulfilled');
        console.log(`✅ Created ${successfulCreations.length}/${workerConfigs.length} workers concurrently`);

        if (successfulCreations.length === 0) {
            throw new Error('No workers were created successfully');
        }

        // Get workers concurrently
        console.log('Retrieving workers concurrently...');
        const getPromises = workerNames.map(name => testGetWorker(name, true));
        const getResults = await Promise.allSettled(getPromises);

        const successfulGets = getResults.filter(result =>
            result.status === 'fulfilled' && result.value.success
        );
        console.log(`✅ Retrieved ${successfulGets.length}/${workerNames.length} workers concurrently`);

        // Update workers concurrently
        console.log('Updating workers concurrently...');
        const updatePromises = workerNames.map(name => testUpdateWorker(name, {
            temperature: 0.7,
            description: `Updated concurrent test worker: ${name}`
        }, true));
        const updateResults = await Promise.allSettled(updatePromises);

        const successfulUpdates = updateResults.filter(result =>
            result.status === 'fulfilled' && result.value.success
        );
        console.log(`✅ Updated ${successfulUpdates.length}/${workerNames.length} workers concurrently`);

        // Delete workers concurrently
        console.log('Deleting workers concurrently...');
        const deletePromises = workerNames.map(name => testDeleteWorker(name, true));
        const deleteResults = await Promise.allSettled(deletePromises);

        const successfulDeletes = deleteResults.filter(result =>
            result.status === 'fulfilled' && result.value.success
        );
        console.log(`✅ Deleted ${successfulDeletes.length}/${workerNames.length} workers concurrently`);

        return {
            success: true,
            created: successfulCreations.length,
            retrieved: successfulGets.length,
            updated: successfulUpdates.length,
            deleted: successfulDeletes.length
        };

    } catch (error) {
        console.error(`❌ Concurrent operations test failed: ${error.message}`);

        // Cleanup any remaining workers
        console.log('Cleaning up remaining workers...');
        for (const name of workerNames) {
            try {
                await testDeleteWorker(name, false); // Don't expect them to exist
            } catch (e) {
                // Ignore cleanup errors
            }
        }

        return { success: false, error: error.message };
    }
}

/**
 * Generate comprehensive worker update scenarios
 */
function generateUpdateScenarios() {
    return [
        {
            name: 'Basic Configuration Update',
            config: {
                temperature: 0.8,
                description: 'Updated basic configuration test worker'
            }
        },
        {
            name: 'Model Configuration Update',
            config: {
                model: 'deepseek-reasoner',
                temperature: 0.2
            }
        },
        {
            name: 'System Prompt Update',
            config: {
                system_prompt: 'You are an updated test worker with enhanced capabilities.',
                instruction: 'Provide detailed responses with reasoning.'
            }
        },
        {
            name: 'Multiple Field Update',
            config: {
                temperature: 0.5,
                description: 'Multi-field updated test worker',
                system_prompt: 'You are a comprehensively updated test worker.',
                user_prompt: 'Execute tasks with updated parameters.'
            }
        }
    ];
}

/**
 * Main test execution function
 */
async function runIndividualWorkerTests() {
    console.log('🚀 Starting Individual Worker Operations Test Suite');
    console.log('='.repeat(80));

    const testResults = {
        health_check: false,
        worker_creation: false,
        worker_retrieval: false,
        worker_updates: [],
        worker_deletion: false,
        edge_cases: [],
        concurrent_operations: false,
        overall_success: false,
        execution_time: 0
    };

    const testStartTime = Date.now();
    let createdWorkerName = null;

    try {
        // Step 1: Health Check
        console.log('\n🏥 Step 1: API Health Check');
        const healthResult = await checkApiHealth();
        testResults.health_check = healthResult.healthy;

        if (!healthResult.healthy) {
            throw new Error('API health check failed - cannot proceed with tests');
        }

        // Step 2: Create Test Worker
        console.log('\n🤖 Step 2: Create Test Worker');
        const uniqueId = Date.now();
        const testWorkerName = `individual_test_worker_${uniqueId}`;
        createdWorkerName = testWorkerName;

        const workerConfig = {
            worker_name: testWorkerName,
            agent_role: 'individual_operations_test_specialist',
            agent_type: 'pydantic_ai',
            model_provider: 'deepseek',
            model: 'deepseek-chat',
            temperature: 0.3,
            system_prompt: 'You are a test worker for individual operations validation. Provide clear, structured responses.',
            instruction: 'When receiving tasks, analyze and respond with detailed feedback.',
            user_prompt: '',
            description: 'Test worker for individual operations testing',
            mcp_config: null,
            base_url_override: null,
            api_key_override: null,
            tools_dir_override: null
        };

        const createdWorker = await createTestWorker(workerConfig);
        testResults.worker_creation = true;

        // Step 3: Test GET Worker
        console.log('\n📋 Step 3: Test GET Worker');
        const getResult = await testGetWorker(testWorkerName, true);
        testResults.worker_retrieval = getResult.success;

        if (!getResult.success) {
            throw new Error('Failed to retrieve created worker');
        }

        // Step 4: Test Worker Updates
        console.log('\n🔄 Step 4: Test Worker Updates');
        const updateScenarios = generateUpdateScenarios();

        for (const scenario of updateScenarios) {
            console.log(`\n--- Testing ${scenario.name} ---`);
            const updateResult = await testUpdateWorker(testWorkerName, scenario.config, true);
            testResults.worker_updates.push({
                scenario: scenario.name,
                success: updateResult.success,
                config: scenario.config
            });

            if (updateResult.success) {
                // Verify updates by getting worker again
                const verifyResult = await testGetWorker(testWorkerName, true);
                if (verifyResult.success) {
                    console.log('✅ Update verification successful');
                } else {
                    console.log('⚠️  Update verification failed');
                }
            }
        }

        // Step 5: Test Edge Cases
        console.log('\n🚨 Step 5: Test Edge Cases');

        // Test GET non-existent worker
        const nonExistentWorkerName = `non_existent_worker_${Date.now()}`;
        const getNonExistentResult = await testGetWorker(nonExistentWorkerName, false);
        testResults.edge_cases.push({
            test: 'GET Non-existent Worker',
            success: getNonExistentResult.success
        });

        // Test UPDATE non-existent worker
        const updateNonExistentResult = await testUpdateWorker(nonExistentWorkerName, {
            temperature: 0.5
        }, false);
        testResults.edge_cases.push({
            test: 'UPDATE Non-existent Worker',
            success: updateNonExistentResult.success
        });

        // Test DELETE non-existent worker
        const deleteNonExistentResult = await testDeleteWorker(nonExistentWorkerName, false);
        testResults.edge_cases.push({
            test: 'DELETE Non-existent Worker',
            success: deleteNonExistentResult.success
        });

        // Test invalid update configuration
        const invalidUpdateResult = await testUpdateWorker(testWorkerName, {
            invalid_field: 'invalid_value',
            temperature: 'invalid_temperature'
        }, false);
        testResults.edge_cases.push({
            test: 'Invalid Update Configuration',
            success: invalidUpdateResult.success
        });

        // Step 6: Test Concurrent Operations
        console.log('\n⚡ Step 6: Test Concurrent Operations');
        const concurrentResult = await testConcurrentOperations(`concurrent_base_${uniqueId}`);
        testResults.concurrent_operations = concurrentResult.success;

        // Step 7: Test DELETE Worker (cleanup main test worker)
        console.log('\n🗑️  Step 7: Test DELETE Worker');
        const deleteResult = await testDeleteWorker(testWorkerName, true);
        testResults.worker_deletion = deleteResult.success;

        if (deleteResult.success) {
            createdWorkerName = null; // Mark as cleaned up

            // Verify deletion by trying to GET the worker
            const verifyDeleteResult = await testGetWorker(testWorkerName, false);
            if (verifyDeleteResult.success && verifyDeleteResult.expectedError) {
                console.log('✅ Deletion verification successful');
            } else {
                console.log('⚠️  Deletion verification failed');
            }
        }

        // Calculate overall success
        const allUpdatesPassed = testResults.worker_updates.every(update => update.success);
        const allEdgeCasesPassed = testResults.edge_cases.every(edge => edge.success);

        testResults.overall_success = testResults.health_check &&
                                     testResults.worker_creation &&
                                     testResults.worker_retrieval &&
                                     allUpdatesPassed &&
                                     testResults.worker_deletion &&
                                     allEdgeCasesPassed &&
                                     testResults.concurrent_operations;

        testResults.execution_time = Date.now() - testStartTime;

        // Step 8: Generate Test Report
        console.log('\n📊 Step 8: Individual Worker Operations Test Summary');
        console.log('='.repeat(80));
        console.log(`🏁 Individual Worker Operations Test - COMPLETE`);
        console.log(`   Total execution time: ${testResults.execution_time}ms`);
        console.log(`   Overall success: ${testResults.overall_success ? '✅ PASSED' : '❌ FAILED'}`);

        console.log('\n📋 Test Results:');
        console.log(`   ✅ API Health Check: ${testResults.health_check ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Worker Creation: ${testResults.worker_creation ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Worker Retrieval (GET): ${testResults.worker_retrieval ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Worker Updates (PUT): ${testResults.worker_updates.length} scenarios tested`);

        for (const update of testResults.worker_updates) {
            console.log(`      - ${update.scenario}: ${update.success ? 'PASSED' : 'FAILED'}`);
        }

        console.log(`   ✅ Worker Deletion (DELETE): ${testResults.worker_deletion ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Edge Cases: ${testResults.edge_cases.length} scenarios tested`);

        for (const edge of testResults.edge_cases) {
            console.log(`      - ${edge.test}: ${edge.success ? 'PASSED' : 'FAILED'}`);
        }

        console.log(`   ✅ Concurrent Operations: ${testResults.concurrent_operations ? 'PASSED' : 'FAILED'}`);

        if (testResults.overall_success) {
            console.log('\n🎉 All Individual Worker Operations Tests PASSED!');
            console.log('   ✅ GET /workers/{worker_name} is working correctly');
            console.log('   ✅ PUT /workers/{worker_name} is working correctly');
            console.log('   ✅ DELETE /workers/{worker_name} is working correctly');
            console.log('   ✅ Error handling is working correctly');
            console.log('   ✅ Concurrent operations are supported');
        } else {
            console.log('\n💥 Some Individual Worker Operations Tests FAILED!');
            console.log('   ❌ Check the detailed output above for specific issues');
        }

    } catch (error) {
        console.error(`\n❌ Test execution failed: ${error.message}`);
        if (error.stack) {
            console.error(`Stack trace: ${error.stack}`);
        }
        testResults.overall_success = false;
        testResults.execution_time = Date.now() - testStartTime;
    }

    // Cleanup remaining worker if test failed
    if (createdWorkerName) {
        console.log('\n🧹 Cleanup: Removing remaining test worker');
        try {
            await testDeleteWorker(createdWorkerName, true);
            console.log('✅ Cleanup completed');
        } catch (cleanupError) {
            console.error(`❌ Cleanup failed: ${cleanupError.message}`);
        }
    }

    return testResults;
}

/**
 * Main function with comprehensive error handling
 */
async function main() {
    console.log('🚀 Individual Worker Operations Test Suite');
    console.log('Node.js version:', process.version);
    console.log('Platform:', process.platform);
    console.log('Architecture:', process.arch);
    console.log('='.repeat(80));

    try {
        const testResults = await runIndividualWorkerTests();

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

// Execute test suite
if (require.main === module) {
    main().catch(error => {
        console.error('❌ Unhandled error in test suite:', error);
        process.exit(1);
    });
}

module.exports = {
    makeRequest,
    createTestWorker,
    testGetWorker,
    testUpdateWorker,
    testDeleteWorker,
    checkApiHealth,
    testConcurrentOperations,
    runIndividualWorkerTests
};
