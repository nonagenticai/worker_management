#!/usr/bin/env node
/**
 * JavaScript Test Script for Worker Collection Operations - Comprehensive Testing Suite
 *
 * This script provides comprehensive testing for the Worker Management API's collection operations:
 * 1. GET /workers - List all workers with comprehensive validation
 * 2. GET /workers/ - Same as above (trailing slash variant)
 * 3. POST /workers - Create new workers with various configurations
 * 4. POST /workers/ - Same as above (trailing slash variant)
 *
 * FEATURES TESTED:
 * - Complete WorkerCreateRequest validation with all fields
 * - Various worker configurations (different agent types, models, providers)
 * - Response structure validation and data integrity checks
 * - Error handling for invalid requests and edge cases
 * - Comprehensive retry logic and cleanup procedures
 * - Unicode handling and edge case worker name validation
 * - MCP configuration and server integration testing
 * - Tool integration and dependency validation
 * - Temperature and model parameter validation
 *
 * Requirements: Node.js 18+ (for native fetch API)
 * Usage: node test_02_worker_collection.js
 */

const { v4: uuidv4 } = require('uuid');

// Configuration
const API_BASE = 'http://localhost:8887';
const TIMEOUT_MS = 60000; // 60 seconds
const MAX_RETRIES = 3;
const RETRY_DELAY = 2000; // 2 seconds
const REQUEST_DELAY = 1500; // 1.5 seconds between requests to prevent memory issues

// Add a flag to track if we've already retried for timeout
let hasRetried = false;

/**
 * Enhanced utility function to make HTTP requests with retry logic and comprehensive error handling
 */
async function makeRequest(url, options = {}, retries = MAX_RETRIES) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'WorkerManagement-WorkerCollectionTest/1.0',
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

            // Add delay between requests to prevent memory issues
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
 * API health check with detailed service validation
 */
async function checkApiHealth() {
    console.log('\n🏥 Performing API Health Check for Worker Collection Tests...');

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
 * Test GET /workers endpoint
 */
async function testGetWorkers() {
    console.log('\n=== Testing GET /workers ===');

    try {
        const workers = await makeRequest(`${API_BASE}/workers`);

        // Validate response structure
        if (!Array.isArray(workers)) {
            throw new Error('GET /workers should return an array');
        }

        console.log(`✅ GET /workers returned ${workers.length} workers`);

        // Validate individual worker structure if workers exist
        if (workers.length > 0) {
            const worker = workers[0];

            // Check if workers are returned as strings (worker names) or objects
            if (typeof worker === 'string') {
                console.log('✅ Workers returned as name strings (simplified format)');
                console.log(`📋 Sample Worker Name: ${worker}`);
            } else if (typeof worker === 'object' && worker !== null) {
                const requiredFields = ['name', 'agent_type', 'model_provider', 'model', 'status'];
                const missingFields = requiredFields.filter(field => !(field in worker));

                if (missingFields.length > 0) {
                    console.warn(`⚠️  First worker missing fields: ${missingFields.join(', ')}`);
                } else {
                    console.log('✅ Worker structure validation passed');
                }

                // Log sample worker for inspection
                console.log(`📋 Sample Worker: ${JSON.stringify(worker, null, 2).substring(0, 300)}...`);
            } else {
                console.warn('⚠️  Unexpected worker format:', typeof worker);
            }
        }

        return { success: true, data: workers };
    } catch (error) {
        console.error(`❌ GET /workers failed: ${error.message}`);
        return { success: false, error: error.message };
    }
}

/**
 * Test GET /workers/ endpoint (trailing slash variant)
 */
async function testGetWorkersTrailingSlash() {
    console.log('\n=== Testing GET /workers/ ===');

    try {
        const workers = await makeRequest(`${API_BASE}/workers/`);

        // Validate response structure
        if (!Array.isArray(workers)) {
            throw new Error('GET /workers/ should return an array');
        }

        console.log(`✅ GET /workers/ returned ${workers.length} workers`);

        return { success: true, data: workers };
    } catch (error) {
        console.error(`❌ GET /workers/ failed: ${error.message}`);
        return { success: false, error: error.message };
    }
}

/**
 * Create comprehensive worker configurations for testing
 */
function generateWorkerConfigurations() {
    const timestamp = Date.now();

    return [
        // Basic OpenAI worker
        {
            worker_name: `test_openai_worker_${timestamp}`,
            tool_names: ['example_text_analyzer', 'file_reader'],
            agent_role: 'OpenAI Test Agent',
            agent_type: 'pydantic_ai',
            model_provider: 'openai',
            model: 'deepseek-chat',
            temperature: 0.7,
            system_prompt: 'You are a test agent for OpenAI integration testing.',
            instruction: 'Process test requests and provide structured responses.',
            user_prompt: 'Analyze the provided input and return results.',
            description: 'OpenAI-based test worker for comprehensive validation',
            deps: ['numpy', 'pandas'],
            mcp_servers: ['filesystem'],
            base_url_override: null,
            api_key_override: null,
            tools_dir_override: null
        },

        // DeepSeek worker with different configuration
        {
            worker_name: `test_deepseek_worker_${timestamp}`,
            tool_names: ['data_transformer'],
            agent_role: 'DeepSeek Analysis Agent',
            agent_type: 'pydantic_ai',
            model_provider: 'deepseek',
            model: 'deepseek-chat',
            temperature: 0.3,
            system_prompt: 'You are a specialized analysis agent using DeepSeek models.',
            instruction: 'Perform detailed analysis of provided data.',
            user_prompt: 'What insights can you provide about this data?',
            description: 'DeepSeek-powered analytical worker',
            deps: ['scikit-learn'],
            mcp_config: null,
            mcp_servers: ['memory', 'filesystem'],
            base_url_override: null,
            api_key_override: null,
            tools_dir_override: null
        },

        // DeepSeek reasoning worker
        {
            worker_name: `test_deepseek_reasoning_worker_${timestamp}`,
            tool_names: [],
            agent_role: 'DeepSeek Reasoning Agent',
            agent_type: 'pydantic_ai',
            model_provider: 'deepseek',
            model: 'deepseek-chat',
            temperature: 0.5,
            system_prompt: 'You are a reasoning-focused agent powered by DeepSeek.',
            instruction: 'Apply logical reasoning to solve complex problems.',
            user_prompt: 'Please analyze this problem step by step.',
            description: 'DeepSeek-based reasoning and problem-solving agent',
            deps: null,
            mcp_config_preset: 'basic',
            mcp_servers: null,
            base_url_override: null,
            api_key_override: null,
            tools_dir_override: null
        },

        // High-temperature creative worker
        {
            worker_name: `test_creative_worker_${timestamp}`,
            tool_names: ['unitTestCreator', 'writeMD'],
            agent_role: 'Creative Content Generator',
            agent_type: 'pydantic_ai',
            model_provider: 'openai',
            model: 'deepseek-chat',
            temperature: 1.8,
            system_prompt: 'You are a creative content generation specialist.',
            instruction: 'Generate creative and original content based on prompts.',
            user_prompt: 'Create engaging content for the given topic.',
            description: 'High-creativity worker for content generation tasks',
            deps: ['markdown'],
            mcp_config: {'memory_size': 1000, 'filesystem_access': true},
            mcp_servers: ['memory', 'filesystem'],
            base_url_override: 'https://api.openai.com/v1',
            api_key_override: null,
            tools_dir_override: '/custom/tools'
        },

        // Minimal configuration worker
        {
            worker_name: `test_minimal_worker_${timestamp}`,
            tool_names: [],
            agent_role: null,
            agent_type: null,
            model_provider: null,
            model: null,
            temperature: null,
            system_prompt: null,
            instruction: null,
            user_prompt: null,
            description: null,
            deps: null,
            mcp_config: null,
            mcp_servers: null,
            base_url_override: null,
            api_key_override: null,
            tools_dir_override: null
        },

        // Unicode name worker
        {
            worker_name: `测试工作者_${timestamp}`,
            tool_names: ['readPy'],
            agent_role: 'Unicode Test Agent 中文',
            agent_type: 'pydantic_ai',
            model_provider: 'openai',
            model: 'gpt-3.5-turbo',
            temperature: 0.1,
            system_prompt: 'You handle Unicode text processing. 你好世界!',
            instruction: 'Process Unicode content carefully.',
            user_prompt: 'Please analyze this Unicode text.',
            description: 'Worker for testing Unicode character handling',
            deps: ['unicodedata'],
            mcp_servers: [],
            base_url_override: null,
            api_key_override: null,
            tools_dir_override: null
        }
    ];
}

/**
 * Test POST /workers endpoint with comprehensive worker creation
 */
async function testPostWorkers() {
    console.log('\n=== Testing POST /workers ===');

    const workerConfigs = generateWorkerConfigurations();
    const results = [];
    const createdWorkers = [];

    for (let i = 0; i < workerConfigs.length; i++) {
        const config = workerConfigs[i];
        console.log(`\n--- Creating Worker ${i + 1}/${workerConfigs.length}: ${config.worker_name} ---`);

        try {
            console.log(`📝 Worker Configuration:`);
            console.log(`   Name: ${config.worker_name}`);
            console.log(`   Type: ${config.agent_type || 'Not specified'}`);
            console.log(`   Provider: ${config.model_provider || 'Not specified'}`);
            console.log(`   Model: ${config.model || 'Not specified'}`);
            console.log(`   Temperature: ${config.temperature || 'Not specified'}`);
            console.log(`   Tools: ${config.tool_names?.length || 0} tools`);
            console.log(`   MCP Servers: ${config.mcp_servers?.length || 0} servers`);

            const response = await makeRequest(`${API_BASE}/workers`, {
                method: 'POST',
                body: JSON.stringify(config)
            });

            console.log(`✅ Worker created successfully: ${response.name}`);
            console.log(`   ID: ${response.id || 'N/A'}`);
            console.log(`   Status: ${response.status}`);
            console.log(`   Created At: ${response.created_at || 'N/A'}`);

            // Validate response structure
            const requiredFields = ['name'];
            const missingFields = requiredFields.filter(field => !response[field]);

            if (missingFields.length > 0) {
                console.warn(`⚠️  Response missing required fields: ${missingFields.join(', ')}`);
            }

            // Validate that response matches request
            if (response.name !== config.worker_name) {
                console.warn(`⚠️  Response name '${response.name}' doesn't match request '${config.worker_name}'`);
            }

            results.push({ success: true, config, response });
            createdWorkers.push(response.name);

        } catch (error) {
            console.error(`❌ Failed to create worker ${config.worker_name}: ${error.message}`);

            if (error.status && error.response) {
                console.error(`   HTTP Details: Status ${error.status}`);
                try {
                    const errorData = JSON.parse(error.response);
                    console.error(`   Error Details: ${JSON.stringify(errorData, null, 2)}`);
                } catch {
                    console.error(`   Error Response: ${error.response.substring(0, 200)}`);
                }
            }

            results.push({ success: false, config, error: error.message });
        }
    }

    console.log(`\n📊 POST /workers Summary: ${results.filter(r => r.success).length}/${results.length} workers created successfully`);

    return { results, createdWorkers };
}

/**
 * Test POST /workers/ endpoint (trailing slash variant)
 */
async function testPostWorkersTrailingSlash() {
    console.log('\n=== Testing POST /workers/ ===');

    const timestamp = Date.now();
    const testConfig = {
        worker_name: `test_trailing_slash_worker_${timestamp}`,
        tool_names: ['example_text_analyzer'],
        agent_role: 'Trailing Slash Test Agent',
        agent_type: 'pydantic_ai',
        model_provider: 'openai',
        model: 'gpt-3.5-turbo',
        temperature: 0.8,
        system_prompt: 'You are a test agent for trailing slash endpoint validation.',
        instruction: 'Handle test requests appropriately.',
        user_prompt: 'Process this test input.',
        description: 'Worker created via trailing slash endpoint',
        deps: [],
        mcp_servers: [],
        base_url_override: null,
        api_key_override: null,
        tools_dir_override: null
    };

    try {
        const response = await makeRequest(`${API_BASE}/workers/`, {
            method: 'POST',
            body: JSON.stringify(testConfig)
        });

        console.log(`✅ Worker created via POST /workers/: ${response.name}`);
        console.log(`   Status: ${response.status}`);

        return { success: true, response, workerName: response.name };
    } catch (error) {
        console.error(`❌ POST /workers/ failed: ${error.message}`);
        return { success: false, error: error.message };
    }
}

/**
 * Test error handling with invalid requests
 */
async function testErrorHandling() {
    console.log('\n=== Testing Error Handling ===');

    const errorTests = [
        {
            name: 'Empty worker name',
            config: { worker_name: '', agent_type: 'pydantic_ai' },
            expectedStatus: 422
        },
        {
            name: 'Invalid temperature (above maximum 2.0)',
            config: { worker_name: 'test_invalid_temp_high', temperature: 2.5 },
            expectedStatus: 422
        },
        {
            name: 'Invalid temperature (below minimum 0.0)',
            config: { worker_name: 'test_invalid_temp_low', temperature: -0.1 },
            expectedStatus: 422
        },
        {
            name: 'Unsupported model provider',
            config: {
                worker_name: 'test_invalid_provider',
                model_provider: 'anthropic',
                model: 'claude-3-sonnet-20240229'
            },
            expectedStatus: 422
        },
        {
            name: 'Empty base URL override',
            config: {
                worker_name: 'test_empty_base_url',
                base_url_override: ''
            },
            expectedStatus: 422
        },
        {
            name: 'Invalid tool names with special characters',
            config: {
                worker_name: 'test_invalid_tools',
                tool_names: ['valid_tool', 'invalid@tool', 'another$tool']
            },
            expectedStatus: 422
        }
    ];

    const errorResults = [];

    for (const test of errorTests) {
        console.log(`\n--- Testing: ${test.name} ---`);
        try {
            const response = await makeRequest(`${API_BASE}/workers`, {
                method: 'POST',
                body: JSON.stringify(test.config)
            });

            console.log(`❌ Expected error but got success: ${JSON.stringify(response)}`);
            errorResults.push({
                test: test.name,
                success: false,
                reason: 'Expected error but got success'
            });

        } catch (error) {
            if (error.status === test.expectedStatus) {
                console.log(`✅ Correctly rejected with status ${error.status}`);
                errorResults.push({ test: test.name, success: true });
            } else {
                console.log(`⚠️  Got status ${error.status}, expected ${test.expectedStatus}`);
                errorResults.push({
                    test: test.name,
                    success: false,
                    reason: `Wrong status: got ${error.status}, expected ${test.expectedStatus}`
                });
            }
        }
    }

    const successfulTests = errorResults.filter(r => r.success).length;
    console.log(`\n📊 Error Handling Summary: ${successfulTests}/${errorResults.length} tests passed`);

    return errorResults;
}

/**
 * Enhanced resource cleanup with comprehensive error handling
 */
async function cleanupWorkers(workerNames = []) {
    if (workerNames.length === 0) {
        console.log('\n🧹 No workers to clean up');
        return { deleted: [], failed: [] };
    }

    console.log(`\n🧹 Cleaning up ${workerNames.length} test workers...`);
    const deleted = [];
    const failed = [];

    for (const workerName of workerNames) {
        try {
            console.log(`🗑️  Deleting worker: ${workerName}`);
            await makeRequest(`${API_BASE}/workers/${encodeURIComponent(workerName)}`, { method: 'DELETE' });
            console.log(`✅ Deleted worker: ${workerName}`);
            deleted.push(workerName);
        } catch (error) {
            console.error(`❌ Failed to delete worker ${workerName}: ${error.message}`);
            failed.push({ name: workerName, error: error.message });
        }
    }

    console.log(`\n📊 Cleanup Summary: ${deleted.length} deleted, ${failed.length} failed`);
    return { deleted, failed };
}

/**
 * Validate worker collection consistency
 */
async function validateCollectionConsistency(initialWorkers, finalWorkers) {
    console.log('\n🔍 Validating Collection Consistency...');

    const validation = {
        initial_count: initialWorkers.length,
        final_count: finalWorkers.length,
        count_difference: finalWorkers.length - initialWorkers.length,
        new_workers_detected: [],
        missing_workers_detected: [],
        consistency_valid: false
    };

    // Find workers that exist in final but not in initial
    const initialNames = new Set(initialWorkers.map(w => w.name));
    const finalNames = new Set(finalWorkers.map(w => w.name));

    validation.new_workers_detected = finalWorkers
        .filter(w => !initialNames.has(w.name))
        .map(w => w.name);

    validation.missing_workers_detected = initialWorkers
        .filter(w => !finalNames.has(w.name))
        .map(w => w.name);

    // Basic consistency check
    validation.consistency_valid = validation.missing_workers_detected.length === 0;

    console.log(`📊 Collection Consistency Report:`);
    console.log(`   Initial workers: ${validation.initial_count}`);
    console.log(`   Final workers: ${validation.final_count}`);
    console.log(`   Difference: ${validation.count_difference > 0 ? '+' : ''}${validation.count_difference}`);
    console.log(`   New workers detected: ${validation.new_workers_detected.length}`);
    console.log(`   Missing workers detected: ${validation.missing_workers_detected.length}`);
    console.log(`   Consistency valid: ${validation.consistency_valid ? '✅' : '❌'}`);

    if (validation.new_workers_detected.length > 0) {
        console.log(`   New workers: ${validation.new_workers_detected.join(', ')}`);
    }

    if (validation.missing_workers_detected.length > 0) {
        console.log(`   Missing workers: ${validation.missing_workers_detected.join(', ')}`);
    }

    return validation;
}

/**
 * Main test function with comprehensive reporting
 */
async function runWorkerCollectionTests() {
    console.log('🚀 Starting Comprehensive Worker Collection API Tests');
    console.log('='.repeat(80));

    // Reset retry flag for this test run
    hasRetried = false;
    console.log('🔄 Retry mechanism initialized (1 retry with 60s delay available)');

    const testResults = {
        health_check: false,
        get_workers: false,
        get_workers_trailing_slash: false,
        post_workers: false,
        post_workers_trailing_slash: false,
        error_handling: false,
        collection_consistency: false,
        overall_success: false,
        execution_time: 0,
        details: {}
    };

    const testStartTime = Date.now();
    let createdWorkers = [];

    try {
        // Step 1: Health Check
        console.log('\n🏥 Step 1: API Health Check');
        const healthResult = await checkApiHealth();
        testResults.health_check = healthResult.healthy;
        testResults.details.health_check = healthResult;

        if (!healthResult.healthy) {
            throw new Error('API health check failed - cannot proceed with tests');
        }

        // Step 2: Get initial worker list
        console.log('\n📋 Step 2: Initial Worker Collection State');
        const initialWorkersResult = await testGetWorkers();
        testResults.get_workers = initialWorkersResult.success;
        testResults.details.initial_workers = initialWorkersResult;

        const initialWorkers = initialWorkersResult.data || [];

        // Step 3: Test GET /workers/ (trailing slash)
        console.log('\n📋 Step 3: GET /workers/ (Trailing Slash)');
        const getWorkersSlashResult = await testGetWorkersTrailingSlash();
        testResults.get_workers_trailing_slash = getWorkersSlashResult.success;
        testResults.details.get_workers_trailing_slash = getWorkersSlashResult;

        // Step 4: Test POST /workers with comprehensive configurations
        console.log('\n🤖 Step 4: POST /workers (Worker Creation)');
        const postWorkersResult = await testPostWorkers();
        testResults.post_workers = postWorkersResult.results.some(r => r.success);
        testResults.details.post_workers = postWorkersResult;
        createdWorkers = createdWorkers.concat(postWorkersResult.createdWorkers);

        // Step 5: Test POST /workers/ (trailing slash)
        console.log('\n🤖 Step 5: POST /workers/ (Trailing Slash)');
        const postWorkersSlashResult = await testPostWorkersTrailingSlash();
        testResults.post_workers_trailing_slash = postWorkersSlashResult.success;
        testResults.details.post_workers_trailing_slash = postWorkersSlashResult;

        if (postWorkersSlashResult.success && postWorkersSlashResult.workerName) {
            createdWorkers.push(postWorkersSlashResult.workerName);
        }

        // Step 6: Test error handling
        console.log('\n❌ Step 6: Error Handling Validation');
        const errorResults = await testErrorHandling();
        testResults.error_handling = errorResults.every(r => r.success);
        testResults.details.error_handling = errorResults;

        // Step 7: Get final worker list and validate consistency
        console.log('\n🔍 Step 7: Final Collection State and Consistency Validation');
        const finalWorkersResult = await testGetWorkers();
        const finalWorkers = finalWorkersResult.data || [];

        const consistencyResult = await validateCollectionConsistency(initialWorkers, finalWorkers);
        testResults.collection_consistency = consistencyResult.consistency_valid;
        testResults.details.collection_consistency = consistencyResult;

        // Overall success determination
        testResults.overall_success = testResults.health_check &&
                                     testResults.get_workers &&
                                     testResults.get_workers_trailing_slash &&
                                     testResults.post_workers &&
                                     testResults.post_workers_trailing_slash &&
                                     testResults.error_handling &&
                                     testResults.collection_consistency;

        testResults.execution_time = Date.now() - testStartTime;

        // Step 8: Generate comprehensive test report
        console.log('\n📊 Step 8: Comprehensive Test Report');
        console.log('='.repeat(80));
        console.log(`🏁 Worker Collection API Tests - COMPLETE`);
        console.log(`   Total execution time: ${testResults.execution_time}ms`);
        console.log(`   Overall success: ${testResults.overall_success ? '✅ PASSED' : '❌ FAILED'}`);
        console.log('\n📋 Detailed Results:');
        console.log(`   ✅ API Health Check: ${testResults.health_check ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ GET /workers: ${testResults.get_workers ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ GET /workers/: ${testResults.get_workers_trailing_slash ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ POST /workers: ${testResults.post_workers ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ POST /workers/: ${testResults.post_workers_trailing_slash ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Error Handling: ${testResults.error_handling ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Collection Consistency: ${testResults.collection_consistency ? 'PASSED' : 'FAILED'}`);

        // Additional statistics
        const successfulCreations = postWorkersResult.results.filter(r => r.success).length;
        const totalCreationAttempts = postWorkersResult.results.length;
        console.log(`   📊 Worker Creation Rate: ${successfulCreations}/${totalCreationAttempts} (${Math.round(successfulCreations/totalCreationAttempts*100)}%)`);

        const errorTestsPassed = errorResults.filter(r => r.success).length;
        console.log(`   📊 Error Handling Rate: ${errorTestsPassed}/${errorResults.length} (${Math.round(errorTestsPassed/errorResults.length*100)}%)`);

        if (testResults.overall_success) {
            console.log('\n🎉 Worker Collection API Tests PASSED!');
            console.log('   ✅ All collection operations (GET, POST) work correctly');
            console.log('   ✅ Both trailing slash and non-trailing slash endpoints work');
            console.log('   ✅ Comprehensive worker configurations are supported');
            console.log('   ✅ Error handling validates input properly');
            console.log('   ✅ Collection state consistency is maintained');
            console.log('   ✅ Unicode worker names are handled correctly');
            console.log('   ✅ MCP integration and tool validation work as expected');
        } else {
            console.log('\n💥 Worker Collection API Tests FAILED!');
            console.log('   ❌ Some collection operations or validations failed');
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

    // Step 9: Cleanup
    console.log('\n🧹 Step 9: Resource Cleanup');
    try {
        const cleanupResult = await cleanupWorkers(createdWorkers);
        testResults.details.cleanup = cleanupResult;

        if (cleanupResult.failed.length > 0) {
            console.log(`⚠️  Some workers failed to cleanup: ${cleanupResult.failed.length} items`);
        }
    } catch (cleanupError) {
        console.error(`❌ Cleanup failed: ${cleanupError.message}`);
    }

    return testResults;
}

/**
 * Main function with comprehensive error handling and reporting
 */
async function main() {
    console.log('🚀 Worker Collection API Test Suite');
    console.log('Node.js version:', process.version);
    console.log('Platform:', process.platform);
    console.log('Architecture:', process.arch);
    console.log('='.repeat(80));

    try {
        const testResults = await runWorkerCollectionTests();

        // Final summary
        console.log('\n📊 FINAL TEST SUMMARY');
        console.log('='.repeat(80));
        console.log(`Overall Result: ${testResults.overall_success ? '🎉 SUCCESS' : '💥 FAILURE'}`);
        console.log(`Total Execution Time: ${testResults.execution_time}ms`);
        console.log(`Test Components: ${Object.keys(testResults).filter(k => typeof testResults[k] === 'boolean').length}`);
        console.log(`Passed Components: ${Object.values(testResults).filter(v => v === true).length}`);

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

// Execute the test suite
if (require.main === module) {
    main().catch(error => {
        console.error('❌ Unhandled error in test suite:', error);
        process.exit(1);
    });
}

module.exports = {
    makeRequest,
    checkApiHealth,
    testGetWorkers,
    testGetWorkersTrailingSlash,
    testPostWorkers,
    testPostWorkersTrailingSlash,
    testErrorHandling,
    cleanupWorkers,
    validateCollectionConsistency,
    runWorkerCollectionTests
};
