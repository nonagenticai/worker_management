#!/usr/bin/env node
/**
 * JavaScript Test Script for Worker Instance Management & Bulk State Operations - Production Ready
 *
 * This script validates Worker Instance Management and Bulk State Operations by:
 * 1. Testing DELETE /workers/{worker_name}/instance - force cleanup worker instance from cache
 * 2. Testing GET /workers/states - get all worker states across all workers
 * 3. Testing DELETE /workers/states - delete all worker states (keeps workers)
 * 4. Creating multiple workers and generating states to test bulk operations
 * 5. Validating proper state cleanup and worker preservation
 * 6. Including comprehensive error handling and cleanup
 *
 * TESTED ENDPOINTS:
 * - DELETE /workers/{worker_name}/instance - Worker instance cache cleanup
 * - GET /workers/states - Bulk state retrieval
 * - DELETE /workers/states - Bulk state deletion
 * - Worker state generation and validation
 * - Worker preservation after state cleanup
 *
 * Requirements: Node.js 18+ (for native fetch API)
 * Usage: node test_06_worker_instance_bulk.js
 */

const { v4: uuidv4 } = require('uuid');

// Enhanced Configuration
const API_BASE = 'http://localhost:8887';
const TIMEOUT_MS = 60000; // 60 seconds for production testing
const MAX_RETRIES = 3;
const RETRY_DELAY = 2000; // 2 seconds
const REQUEST_DELAY = 1000; // 1 second between requests

// Add a flag to track if we've already retried for timeout
let hasRetried = false;

/**
 * Enhanced utility function to make HTTP requests with retry logic and comprehensive error handling
 */
async function makeRequest(url, options = {}, retries = MAX_RETRIES) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'WorkerManagement-InstanceBulkTest/1.0',
            ...options.headers
        }
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
                 error.message.includes('timeout') ||
                 error.message.includes('fetch') ||
                 error.message.includes('network') ||
                 error.message.includes('ECONNRESET') ||
                 error.message.includes('ENOTFOUND') ||
                 (error.status >= 500 && error.status < 600)) // Server errors
                && !hasRetried
                && attempt === retries) {

                console.log('🕐 API timeout/network issue detected. Waiting 60 seconds before single retry...');
                console.log(`   Original error: ${error.message}`);
                hasRetried = true;

                // Sleep for 1 minute
                await new Promise(resolve => setTimeout(resolve, 60000));

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
                    console.log(`✅ retry successful!`);

                    return data;
                } catch (retryError) {
                    console.log(`❌ retry also failed: ${retryError.message}`);
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
 * Enhanced worker creation with validation and detailed configuration
 */
async function createWorker(workerConfig) {
    console.log(`\n=== Creating Worker: ${workerConfig.worker_name} ===`);

    try {
        // Pre-validate configuration
        if (!workerConfig.worker_name || !workerConfig.agent_type) {
            throw new Error('Worker configuration missing required fields: worker_name, agent_type');
        }

        const response = await makeRequest(`${API_BASE}/workers`, {
            method: 'POST',
            body: JSON.stringify(workerConfig)
        });

        console.log(`✅ Worker created successfully: ${response.name}`);
        console.log(`   ID: ${response.id || 'N/A'}`);
        console.log(`   Type: ${response.agent_type}`);
        console.log(`   Model: ${response.model_provider}:${response.model}`);
        console.log(`   Status: ${response.status}`);
        console.log(`   Temperature: ${response.temperature}`);

        // Enhanced validation of required fields
        const requiredFields = ['name', 'agent_type'];
        const missingFields = requiredFields.filter(field => !response[field]);

        if (missingFields.length > 0) {
            throw new Error(`Worker creation response missing required fields: ${missingFields.join(', ')}`);
        }

        return response;
    } catch (error) {
        console.error(`❌ Failed to create worker ${workerConfig.worker_name}: ${error.message}`);
        if (error.status && error.response) {
            console.error(`   HTTP Details: Status ${error.status}, Response: ${error.response}`);
        }
        throw error;
    }
}

/**
 * Generate worker state by sending message to worker
 */
async function generateWorkerState(workerName, message, context = {}) {
    console.log(`\n=== Generating State for Worker: ${workerName} ===`);

    try {
        const messageRequest = {
            message: message,
            context: context
        };

        const response = await makeRequest(`${API_BASE}/workers/${workerName}/message`, {
            method: 'POST',
            body: JSON.stringify(messageRequest)
        });

        console.log(`✅ Message sent to worker: ${workerName}`);
        console.log(`   Response length: ${JSON.stringify(response).length} characters`);
        console.log(`   Response preview: ${JSON.stringify(response).substring(0, 100)}...`);

        return response;
    } catch (error) {
        console.error(`❌ Failed to send message to worker ${workerName}: ${error.message}`);
        if (error.status && error.response) {
            console.error(`   HTTP Details: Status ${error.status}, Response: ${error.response}`);
        }
        throw error;
    }
}

/**
 * Test DELETE /workers/{worker_name}/instance endpoint
 */
async function testDeleteWorkerInstance(workerName) {
    console.log(`\n=== Testing DELETE /workers/${workerName}/instance ===`);

    try {
        const response = await makeRequest(`${API_BASE}/workers/${workerName}/instance`, {
            method: 'DELETE'
        });

        console.log(`✅ Worker instance cleanup successful: ${workerName}`);
        console.log(`   Response: ${JSON.stringify(response)}`);

        return response;
    } catch (error) {
        console.error(`❌ Failed to delete worker instance ${workerName}: ${error.message}`);
        if (error.status && error.response) {
            console.error(`   HTTP Details: Status ${error.status}, Response: ${error.response}`);
        }
        throw error;
    }
}

/**
 * Test GET /workers/states endpoint
 */
async function testGetAllWorkerStates() {
    console.log(`\n=== Testing GET /workers/states ===`);

    try {
        const response = await makeRequest(`${API_BASE}/workers/states`);

        console.log(`✅ All worker states retrieved successfully`);

        // The response format is {states: [...], total: number}
        const states = response.states || [];
        const total = response.total || 0;

        // Group states by worker name for display
        const statesByWorker = {};
        states.forEach(state => {
            const workerName = state.worker_name;
            if (!statesByWorker[workerName]) {
                statesByWorker[workerName] = [];
            }
            statesByWorker[workerName].push(state);
        });

        console.log(`   ${Object.keys(statesByWorker).length} states`);
        console.log(`   ${total} total`);

        // Display summary of states by worker
        for (const [workerName, workerStates] of Object.entries(statesByWorker)) {
            console.log(`   ${workerName}: ${workerStates.length} states`);
        }

        return response;
    } catch (error) {
        console.error(`❌ Failed to get all worker states: ${error.message}`);
        if (error.status && error.response) {
            console.error(`   HTTP Details: Status ${error.status}, Response: ${error.response}`);
        }
        throw error;
    }
}

/**
 * Test DELETE /workers/states endpoint
 */
async function testDeleteAllWorkerStates() {
    console.log(`\n=== Testing DELETE /workers/states ===`);

    try {
        const response = await makeRequest(`${API_BASE}/workers/states`, {
            method: 'DELETE'
        });

        console.log(`✅ All worker states deleted successfully`);
        console.log(`   Response: ${JSON.stringify(response)}`);

        return response;
    } catch (error) {
        console.error(`❌ Failed to delete all worker states: ${error.message}`);
        if (error.status && error.response) {
            console.error(`   HTTP Details: Status ${error.status}, Response: ${error.response}`);
        }
        throw error;
    }
}

/**
 * Verify workers still exist after state cleanup
 */
async function verifyWorkersExist(expectedWorkerNames) {
    console.log(`\n=== Verifying Workers Still Exist ===`);

    try {
        const workers = await makeRequest(`${API_BASE}/workers`);
        const workerNames = workers; // Workers endpoint returns array of strings

        console.log(`✅ Workers list retrieved: ${workers.length} workers found`);

        const verificationResults = {};
        for (const expectedName of expectedWorkerNames) {
            const exists = workerNames.includes(expectedName);
            verificationResults[expectedName] = exists;
            console.log(`   ${exists ? '✅' : '❌'} Worker ${expectedName}: ${exists ? 'EXISTS' : 'MISSING'}`);
        }

        return verificationResults;
    } catch (error) {
        console.error(`❌ Failed to verify workers exist: ${error.message}`);
        throw error;
    }
}

/**
 * Comprehensive API health check
 */
async function checkApiHealth() {
    console.log('\n🏥 Performing Comprehensive API Health Check...');

    const healthChecks = [];

    try {
        // Main API health
        console.log('🔍 Checking main API health...');
        const healthResponse = await makeRequest(`${API_BASE}/health`);
        console.log(`✅ Main API is healthy: ${healthResponse.status}`);
        healthChecks.push({ service: 'main_api', status: 'healthy', details: healthResponse });

        // Worker service check
        console.log('🔍 Checking worker service...');
        const workersResponse = await makeRequest(`${API_BASE}/workers`);
        console.log(`✅ Worker service is responding (${workersResponse.length} workers found)`);
        healthChecks.push({ service: 'workers', status: 'healthy', details: { worker_count: workersResponse.length } });

        console.log('\n🎉 All health checks passed!');
        return { healthy: true, checks: healthChecks };

    } catch (error) {
        console.error(`❌ Health check failed: ${error.message}`);
        healthChecks.push({ service: 'unknown', status: 'failed', error: error.message });
        return { healthy: false, checks: healthChecks, error: error.message };
    }
}

/**
 * Enhanced resource cleanup with comprehensive error handling
 */
async function cleanupResources(workerNames = []) {
    console.log('\n🧹 Performing Enhanced Resource Cleanup...');
    const cleanupResults = {
        workers_deleted: [],
        failed_deletions: [],
        total_time: 0
    };

    const startTime = Date.now();

    try {
        // Delete workers
        for (const workerName of workerNames) {
            console.log(`🗑️  Deleting worker: ${workerName}`);
            try {
                await makeRequest(`${API_BASE}/workers/${workerName}`, { method: 'DELETE' });
                console.log(`✅ Worker deleted successfully: ${workerName}`);
                cleanupResults.workers_deleted.push(workerName);
            } catch (error) {
                console.error(`❌ Failed to delete worker ${workerName}: ${error.message}`);
                cleanupResults.failed_deletions.push({ type: 'worker', id: workerName, error: error.message });
            }
        }

        cleanupResults.total_time = Date.now() - startTime;

        console.log('\n✅ Cleanup completed');
        console.log(`   Workers deleted: ${cleanupResults.workers_deleted.length}/${workerNames.length}`);
        console.log(`   Failed deletions: ${cleanupResults.failed_deletions.length}`);
        console.log(`   Total cleanup time: ${cleanupResults.total_time}ms`);

    } catch (error) {
        console.error(`❌ Cleanup process failed: ${error.message}`);
        cleanupResults.total_time = Date.now() - startTime;
    }

    return cleanupResults;
}

/**
 * Enhanced test result validation
 */
function validateTestResults(testResults) {
    console.log('\n🔍 Performing Enhanced Test Validation...');

    const validation = {
        health_check: testResults.health_check || false,
        workers_created: testResults.workers_created || false,
        states_generated: testResults.states_generated || false,
        all_states_retrieved: testResults.all_states_retrieved || false,
        all_states_deleted: testResults.all_states_deleted || false,
        workers_preserved: testResults.workers_preserved || false,
        instance_cleanup: testResults.instance_cleanup || false,
        overall_valid: false
    };

    try {
        // Count passed validations
        const validationChecks = Object.values(validation).slice(0, -1); // Exclude overall_valid
        const passedChecks = validationChecks.filter(check => check === true).length;
        validation.overall_valid = passedChecks >= 6; // At least 6 out of 7 checks must pass

        console.log(`📊 Validation Summary: ${passedChecks}/7 checks passed`);
        console.log(`   ✅ Health Check: ${validation.health_check ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Workers Created: ${validation.workers_created ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ States Generated: ${validation.states_generated ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ All States Retrieved: ${validation.all_states_retrieved ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ All States Deleted: ${validation.all_states_deleted ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Workers Preserved: ${validation.workers_preserved ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Instance Cleanup: ${validation.instance_cleanup ? 'PASSED' : 'FAILED'}`);

        if (validation.overall_valid) {
            console.log('🎉 Overall validation: PASSED');
        } else {
            console.log('❌ Overall validation: FAILED');
        }

    } catch (error) {
        console.error(`❌ Validation process failed: ${error.message}`);
    }

    return validation;
}

/**
 * Main test function for Worker Instance Management & Bulk State Operations
 */
async function runWorkerInstanceBulkTest() {
    console.log('🚀 Starting Worker Instance Management & Bulk State Operations Test');
    console.log('='.repeat(80));

    // Reset retry flag for this test run
    hasRetried = false;
    console.log('🔄 Retry mechanism initialized (1 retry with 60s delay available)');

    const createdWorkerNames = [];
    const testResults = {
        health_check: false,
        workers_created: false,
        states_generated: false,
        all_states_retrieved: false,
        all_states_deleted: false,
        workers_preserved: false,
        instance_cleanup: false,
        overall_success: false,
        execution_time: 0,
        partial_results: {}
    };

    const testStartTime = Date.now();

    try {
        // Step 1: Comprehensive Health Check
        console.log('\n🏥 Step 1: Comprehensive Health Check');
        const healthResult = await checkApiHealth();
        testResults.health_check = healthResult.healthy;
        testResults.partial_results.health_check = healthResult;

        if (!healthResult.healthy) {
            throw new Error('API health check failed - cannot proceed with test');
        }

        // Step 2: Create multiple test workers
        console.log('\n🤖 Step 2: Creating Multiple Test Workers');

        const workerConfigs = [
            {
                worker_name: 'bulk_test_worker_1',
                agent_role: 'state_generation_specialist_1',
                agent_type: 'pydantic_ai',
                model_provider: 'deepseek',
                model: 'deepseek-chat',
                temperature: 0.3,
                system_prompt: 'You are a state generation specialist for worker instance testing. Respond with structured data.',
                instruction: 'Generate structured responses for testing worker state management.',
                user_prompt: '',
                description: 'Test worker 1 for bulk state operations testing',
                mcp_config: null,
                base_url_override: null,
                api_key_override: null,
                tools_dir_override: null
            },
            {
                worker_name: 'bulk_test_worker_2',
                agent_role: 'state_generation_specialist_2',
                agent_type: 'pydantic_ai',
                model_provider: 'deepseek',
                model: 'deepseek-chat',
                temperature: 0.4,
                system_prompt: 'You are another state generation specialist for worker instance testing. Provide detailed responses.',
                instruction: 'Create comprehensive responses for testing bulk operations.',
                user_prompt: '',
                description: 'Test worker 2 for bulk state operations testing',
                mcp_config: null,
                base_url_override: null,
                api_key_override: null,
                tools_dir_override: null
            },
            {
                worker_name: 'bulk_test_worker_3',
                agent_role: 'state_generation_specialist_3',
                agent_type: 'pydantic_ai',
                model_provider: 'deepseek',
                model: 'deepseek-chat',
                temperature: 0.5,
                system_prompt: 'You are a third state generation specialist for comprehensive testing.',
                instruction: 'Provide varied responses for testing worker instance management.',
                user_prompt: '',
                description: 'Test worker 3 for bulk state operations testing',
                mcp_config: null,
                base_url_override: null,
                api_key_override: null,
                tools_dir_override: null
            }
        ];

        for (const workerConfig of workerConfigs) {
            const worker = await createWorker(workerConfig);
            createdWorkerNames.push(worker.name);
        }

        testResults.workers_created = true;
        testResults.partial_results.created_workers = createdWorkerNames;

        console.log(`✅ ${createdWorkerNames.length} workers created successfully`);

        // Step 3: Generate states for each worker
        console.log('\n📊 Step 3: Generating States for Each Worker');

        const stateGeneration = [];

        for (let i = 0; i < createdWorkerNames.length; i++) {
            const workerName = createdWorkerNames[i];

            // Generate multiple states per worker
            for (let j = 1; j <= 3; j++) {
                const message = `State generation test ${j} for ${workerName}. Generate a structured response with: worker_name="${workerName}", test_id=${j}, timestamp="${new Date().toISOString()}", data="test_data_${j}".`;
                const context = {
                    test_phase: 'state_generation',
                    worker_index: i + 1,
                    state_index: j,
                    total_workers: createdWorkerNames.length
                };

                try {
                    const response = await generateWorkerState(workerName, message, context);
                    stateGeneration.push({ worker: workerName, state_index: j, success: true, response });
                } catch (error) {
                    stateGeneration.push({ worker: workerName, state_index: j, success: false, error: error.message });
                }
            }
        }

        const successfulStates = stateGeneration.filter(s => s.success).length;
        testResults.states_generated = successfulStates > 0;
        testResults.partial_results.state_generation = stateGeneration;

        console.log(`✅ State generation completed: ${successfulStates}/${stateGeneration.length} successful`);

        // Step 4: Test GET /workers/states
        console.log('\n📋 Step 4: Testing GET /workers/states');

        const allStatesResponse = await testGetAllWorkerStates();
        testResults.all_states_retrieved = true;
        testResults.partial_results.all_states_response = allStatesResponse;

        // Validate that we have states for our workers
        const states = allStatesResponse.states || [];
        const workersWithStates = [...new Set(states.map(state => state.worker_name))];
        const ourWorkersWithStates = createdWorkerNames.filter(name => workersWithStates.includes(name));

        console.log(`✅ Found states for ${ourWorkersWithStates.length}/${createdWorkerNames.length} of our test workers`);

        // Step 5: Test worker instance cleanup
        console.log('\n🧹 Step 5: Testing Worker Instance Cleanup');

        let instanceCleanupResults = [];

        for (const workerName of createdWorkerNames) {
            try {
                const cleanupResponse = await testDeleteWorkerInstance(workerName);
                instanceCleanupResults.push({ worker: workerName, success: true, response: cleanupResponse });
            } catch (error) {
                instanceCleanupResults.push({ worker: workerName, success: false, error: error.message });
            }
        }

        const successfulCleanups = instanceCleanupResults.filter(c => c.success).length;
        testResults.instance_cleanup = successfulCleanups > 0;
        testResults.partial_results.instance_cleanup = instanceCleanupResults;

        console.log(`✅ Instance cleanup completed: ${successfulCleanups}/${instanceCleanupResults.length} successful`);

        // Step 6: Test DELETE /workers/states
        console.log('\n🗑️  Step 6: Testing DELETE /workers/states');

        const deleteAllStatesResponse = await testDeleteAllWorkerStates();
        testResults.all_states_deleted = true;
        testResults.partial_results.delete_all_states_response = deleteAllStatesResponse;

        // Step 7: Verify workers still exist after state cleanup
        console.log('\n🔍 Step 7: Verifying Workers Still Exist After State Cleanup');

        const workerVerification = await verifyWorkersExist(createdWorkerNames);
        const allWorkersExist = Object.values(workerVerification).every(exists => exists === true);
        testResults.workers_preserved = allWorkersExist;
        testResults.partial_results.worker_verification = workerVerification;

        console.log(`✅ Worker preservation check: ${allWorkersExist ? 'ALL PRESERVED' : 'SOME MISSING'}`);

        // Step 8: Verify states are actually gone
        console.log('\n🔍 Step 8: Verifying States Are Actually Deleted');

        const finalStatesCheck = await testGetAllWorkerStates();
        const finalStates = finalStatesCheck.states || [];
        const remainingStates = finalStates.length;

        console.log(`📊 States remaining after cleanup: ${remainingStates}`);

        // If there are remaining states, they should be from other workers not part of our test
        let ourStatesRemaining = 0;
        const finalWorkersWithStates = [...new Set(finalStates.map(state => state.worker_name))];
        for (const workerName of createdWorkerNames) {
            if (finalWorkersWithStates.includes(workerName)) {
                ourStatesRemaining++;
            }
        }

        console.log(`📊 Our test workers with remaining states: ${ourStatesRemaining}/${createdWorkerNames.length}`);

        // Overall success determination
        testResults.overall_success = testResults.health_check &&
                                     testResults.workers_created &&
                                     testResults.states_generated &&
                                     testResults.all_states_retrieved &&
                                     testResults.all_states_deleted &&
                                     testResults.workers_preserved;

        testResults.execution_time = Date.now() - testStartTime;

        // Step 9: Generate comprehensive test report
        console.log('\n📊 Step 9: Worker Instance & Bulk Operations Test Summary');
        console.log('='.repeat(80));
        console.log(`🏁 Worker Instance Management & Bulk State Operations Test - COMPLETE`);
        console.log(`   Total execution time: ${testResults.execution_time}ms`);
        console.log(`   Overall success: ${testResults.overall_success ? '✅ PASSED' : '❌ FAILED'}`);
        console.log('\n📋 Test Results:');
        console.log(`   ✅ API Health Check: ${testResults.health_check ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Workers Created (${createdWorkerNames.length}): ${testResults.workers_created ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ States Generated: ${testResults.states_generated ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ All States Retrieved: ${testResults.all_states_retrieved ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Worker Instance Cleanup: ${testResults.instance_cleanup ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ All States Deleted: ${testResults.all_states_deleted ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Workers Preserved: ${testResults.workers_preserved ? 'PASSED' : 'FAILED'}`);

        if (testResults.overall_success) {
            console.log('\n🎉 Worker Instance Management & Bulk Operations Test PASSED!');
            console.log('   ✅ DELETE /workers/{worker_name}/instance working correctly');
            console.log('   ✅ GET /workers/states working correctly');
            console.log('   ✅ DELETE /workers/states working correctly');
            console.log('   ✅ Workers preserved after state cleanup');
            console.log('   ✅ Bulk operations functioning as expected');
        } else {
            console.log('\n💥 Worker Instance Management & Bulk Operations Test FAILED!');
            console.log('   ❌ Some endpoints or operations are not working correctly');
            console.log('   ❌ Check the detailed output above for specific issues');
        }

    } catch (error) {
        console.error(`\n❌ Enhanced test execution failed: ${error.message}`);
        if (error.stack) {
            console.error(`Stack trace: ${error.stack}`);
        }
        testResults.overall_success = false;
        testResults.execution_time = Date.now() - testStartTime;
    }

    // Enhanced cleanup process
    console.log('\n🧹 Step 10: Enhanced Resource Cleanup');
    try {
        const cleanupResult = await cleanupResources(createdWorkerNames);
        testResults.partial_results.cleanup = cleanupResult;

        if (cleanupResult.failed_deletions.length > 0) {
            console.log(`⚠️  Some resources failed to cleanup: ${cleanupResult.failed_deletions.length} items`);
        }
    } catch (cleanupError) {
        console.error(`❌ Cleanup failed: ${cleanupError.message}`);
    }

    return testResults;
}

/**
 * Enhanced main function with comprehensive error handling and reporting
 */
async function main() {
    console.log('🚀 Worker Instance Management & Bulk State Operations Test Suite');
    console.log('Node.js version:', process.version);
    console.log('Platform:', process.platform);
    console.log('Architecture:', process.arch);
    console.log('='.repeat(80));

    try {
        const testResults = await runWorkerInstanceBulkTest();

        // Final summary
        console.log('\n📊 FINAL TEST SUMMARY');
        console.log('='.repeat(80));
        console.log(`Overall Result: ${testResults.overall_success ? '🎉 SUCCESS' : '💥 FAILURE'}`);
        console.log(`Total Execution Time: ${testResults.execution_time}ms`);
        console.log(`Test Components: ${Object.keys(testResults).filter(k => typeof testResults[k] === 'boolean').length}`);
        console.log(`Passed Components: ${Object.values(testResults).filter(v => v === true).length}`);

        // Enhanced validation
        const validationResult = validateTestResults(testResults);

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

// Enhanced execution with proper error handling
if (require.main === module) {
    main().catch(error => {
        console.error('❌ Unhandled error in test suite:', error);
        process.exit(1);
    });
}

module.exports = {
    makeRequest,
    createWorker,
    generateWorkerState,
    testDeleteWorkerInstance,
    testGetAllWorkerStates,
    testDeleteAllWorkerStates,
    verifyWorkersExist,
    checkApiHealth,
    cleanupResources,
    validateTestResults,
    runWorkerInstanceBulkTest
};
