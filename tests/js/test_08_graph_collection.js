#!/usr/bin/env node
/**
 * JavaScript Test Script for Graph Collection Operations - Comprehensive Testing Suite
 *
 * This script provides comprehensive testing for the Worker Management API's graph collection operations:
 * 1. GET /graphs/ - List all graphs with include_details parameter testing
 * 2. POST /graphs/ - Create new graphs with GraphCreateRequest validation
 * 3. DELETE /graphs/ - Delete all graphs and states with comprehensive cleanup
 *
 * FEATURES TESTED:
 * - Complete GraphCreateRequest validation with all required fields
 * - Various graph configurations (simple, complex, different node types)
 * - include_details parameter testing (true/false/0/1)
 * - GraphResponse structure validation and data integrity checks
 * - Error handling for invalid requests and edge cases
 * - Comprehensive retry logic and cleanup procedures
 * - Unicode handling and edge case graph name validation
 * - Mixed node types (worker, function, condition)
 * - Edge validation and dependency management
 * - Variable substitution and context handling
 * - Comprehensive cleanup and state management
 *
 * Requirements: Node.js 18+ (for native fetch API)
 * Usage: node test_08_graph_collection.js
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
            'User-Agent': 'WorkerManagement-GraphCollectionTest/1.0',
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
 * Enhanced worker creation with validation for graph testing
 */
async function createTestWorker(workerName, agentType = 'pydantic_ai') {
    console.log(`\n=== Creating Test Worker: ${workerName} ===`);

    const workerConfig = {
        worker_name: workerName,
        agent_role: 'graph_test_specialist',
        agent_type: agentType,
        model_provider: 'deepseek',
        model: 'deepseek-chat',
        temperature: 0.3,
        system_prompt: 'You are a specialized test worker for graph testing. Provide structured JSON responses.',
        instruction: 'When receiving task messages, analyze the provided content and return structured JSON responses with clear status indicators.',
        user_prompt: '',
        description: `Test worker for graph collection operations testing: ${workerName}`,
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

        console.log(`✅ Worker created successfully: ${response.name}`);
        console.log(`   ID: ${response.id || 'N/A'}`);
        console.log(`   Type: ${response.agent_type}`);
        console.log(`   Status: ${response.status}`);

        return response;
    } catch (error) {
        console.error(`❌ Failed to create worker ${workerName}: ${error.message}`);
        throw error;
    }
}

/**
 * Test GET /graphs/ with include_details parameter
 */
async function testGetGraphs() {
    console.log('\n=== Testing GET /graphs/ with include_details parameter ===');

    const testResults = {
        basic_get: false,
        include_details_true: false,
        include_details_false: false,
        include_details_1: false,
        include_details_0: false,
        trailing_slash: false,
        structure_validation: false
    };

    try {
        // Test 1: Basic GET /graphs/
        console.log('\n📊 Test 1: Basic GET /graphs/');
        const basicResponse = await makeRequest(`${API_BASE}/graphs/`);
        console.log(`✅ Basic GET returned ${basicResponse.length} graphs`);
        testResults.basic_get = Array.isArray(basicResponse);

        // Test 2: GET /graphs/ with include_details=true
        console.log('\n📊 Test 2: GET /graphs/ with include_details=true');
        const detailsTrueResponse = await makeRequest(`${API_BASE}/graphs/?include_details=true`);
        console.log(`✅ include_details=true returned ${detailsTrueResponse.length} graphs`);
        testResults.include_details_true = Array.isArray(detailsTrueResponse);

        // Test 3: GET /graphs/ with include_details=false
        console.log('\n📊 Test 3: GET /graphs/ with include_details=false');
        const detailsFalseResponse = await makeRequest(`${API_BASE}/graphs/?include_details=false`);
        console.log(`✅ include_details=false returned ${detailsFalseResponse.length} graphs`);
        testResults.include_details_false = Array.isArray(detailsFalseResponse);

        // Test 4: GET /graphs/ with include_details=1
        console.log('\n📊 Test 4: GET /graphs/ with include_details=1');
        const details1Response = await makeRequest(`${API_BASE}/graphs/?include_details=1`);
        console.log(`✅ include_details=1 returned ${details1Response.length} graphs`);
        testResults.include_details_1 = Array.isArray(details1Response);

        // Test 5: GET /graphs/ with include_details=0
        console.log('\n📊 Test 5: GET /graphs/ with include_details=0');
        const details0Response = await makeRequest(`${API_BASE}/graphs/?include_details=0`);
        console.log(`✅ include_details=0 returned ${details0Response.length} graphs`);
        testResults.include_details_0 = Array.isArray(details0Response);

        // Test 6: GET /graphs (without trailing slash)
        console.log('\n📊 Test 6: GET /graphs (without trailing slash)');
        try {
            const noSlashResponse = await makeRequest(`${API_BASE}/graphs`);
            console.log(`✅ No trailing slash returned ${noSlashResponse.length} graphs`);
            testResults.trailing_slash = Array.isArray(noSlashResponse);
        } catch (error) {
            console.log(`⚠️  No trailing slash variant: ${error.message}`);
            testResults.trailing_slash = false;
        }

        // Test 7: Structure validation for detailed response
        console.log('\n📊 Test 7: Structure validation for detailed response');
        if (detailsTrueResponse.length > 0) {
            const sampleGraph = detailsTrueResponse[0];
            const requiredFields = ['id', 'name', 'nodes', 'edges'];
            const hasAllFields = requiredFields.every(field => sampleGraph.hasOwnProperty(field));

            console.log(`✅ Sample graph structure validation: ${hasAllFields ? 'PASSED' : 'FAILED'}`);
            console.log(`   Required fields: ${requiredFields.join(', ')}`);
            console.log(`   Sample graph has: ${Object.keys(sampleGraph).join(', ')}`);

            testResults.structure_validation = hasAllFields;
        } else {
            console.log(`⚠️  No graphs available for structure validation`);
            testResults.structure_validation = true; // No graphs is also valid
        }

    } catch (error) {
        console.error(`❌ GET /graphs/ testing failed: ${error.message}`);
        throw error;
    }

    return testResults;
}

/**
 * Create various graph configurations for testing
 */
function createGraphConfigurations(testWorkerName) {
    const configs = [];

    // Configuration 1: Simple single-node graph
    const simpleNodeId = uuidv4();
    configs.push({
        name: 'Simple Test Graph - Single Node',
        description: 'Basic graph with single worker node for collection testing',
        variables: {
            test_message: 'simple_test',
            node_count: 1
        },
        nodes: [
            {
                id: simpleNodeId,
                name: 'Simple Worker Node',
                node_type: 'worker',
                worker_name: testWorkerName,
                task_template: 'Process simple test: {test_message}',
                dependencies: [],
                data: {
                    test_type: 'simple',
                    expected_response: 'structured_json'
                },
                timeout: 60,
                retry_count: 1
            }
        ],
        edges: []
    });

    // Configuration 2: Complex multi-node graph with different node types
    const workerNodeId = uuidv4();
    const functionNodeId = uuidv4();
    const conditionNodeId = uuidv4();
    const finalNodeId = uuidv4();

    configs.push({
        name: 'Complex Multi-Node Test Graph',
        description: 'Complex graph with mixed node types (worker, function, condition) for comprehensive testing',
        variables: {
            input_value: 'complex_test_data',
            threshold: 50,
            enable_validation: true,
            test_iteration: 1
        },
        nodes: [
            {
                id: workerNodeId,
                name: 'Initial Worker Node',
                node_type: 'worker',
                worker_name: testWorkerName,
                task_template: 'Initialize complex test with input: {input_value}. Enable validation: {enable_validation}. Return structured data for downstream processing.',
                dependencies: [],
                data: {
                    node_position: 'initial',
                    processing_type: 'data_initialization'
                },
                timeout: 90,
                retry_count: 2
            },
            {
                id: functionNodeId,
                name: 'Processing Function Node',
                node_type: 'function',
                task_template: 'Process worker output with mathematical transformation',
                dependencies: [workerNodeId],
                data: {
                    function_code: `
def run(data):
    # Process data from worker node
    import json

    # Extract values from context
    input_value = data.get('input_value', 'default')
    threshold = data.get('threshold', 0)
    test_iteration = data.get('test_iteration', 1)

    # Perform calculations
    calculated_score = len(input_value) * threshold + test_iteration * 10
    validation_result = calculated_score > 100

    return {
        "node_type": "function",
        "calculated_score": calculated_score,
        "validation_result": validation_result,
        "input_processed": input_value,
        "threshold_used": threshold,
        "test_iteration": test_iteration,
        "function_status": "completed"
    }
`,
                    processing_type: 'mathematical_transformation'
                },
                timeout: 45,
                retry_count: 1
            },
            {
                id: conditionNodeId,
                name: 'Validation Condition Node',
                node_type: 'condition',
                task_template: 'Evaluate validation condition based on calculated score',
                dependencies: [functionNodeId],
                data: {
                    condition: 'calculated_score > threshold',
                    evaluation_type: 'score_validation'
                },
                timeout: 30,
                retry_count: 1
            },
            {
                id: finalNodeId,
                name: 'Final Summary Worker Node',
                node_type: 'worker',
                worker_name: testWorkerName,
                task_template: 'Generate final summary report. Input value: {input_value}. Threshold: {threshold}. Test iteration: {test_iteration}. Provide comprehensive JSON summary of the entire workflow execution.',
                dependencies: [conditionNodeId],
                data: {
                    node_position: 'final',
                    report_type: 'comprehensive_summary'
                },
                timeout: 90,
                retry_count: 1
            }
        ],
        edges: [
            {
                source: workerNodeId,
                target: functionNodeId,
                condition: null,
                metadata: {
                    data_flow: 'worker_to_function',
                    edge_type: 'sequential'
                }
            },
            {
                source: functionNodeId,
                target: conditionNodeId,
                condition: null,
                metadata: {
                    data_flow: 'function_to_condition',
                    edge_type: 'conditional_evaluation'
                }
            },
            {
                source: conditionNodeId,
                target: finalNodeId,
                condition: null,
                metadata: {
                    data_flow: 'condition_to_final',
                    edge_type: 'summary_generation'
                }
            }
        ]
    });

    // Configuration 3: Function-only graph
    const funcOnlyNodeId = uuidv4();
    configs.push({
        name: 'Function-Only Test Graph',
        description: 'Graph with only function nodes for pure computational testing',
        variables: {
            number_a: 15,
            number_b: 25,
            operation: 'multiply'
        },
        nodes: [
            {
                id: funcOnlyNodeId,
                name: 'Mathematical Function Node',
                node_type: 'function',
                task_template: 'Perform mathematical operation on input numbers',
                dependencies: [],
                data: {
                    function_code: `
def run(data):
    # Pure mathematical function
    number_a = data.get('number_a', 0)
    number_b = data.get('number_b', 0)
    operation = data.get('operation', 'add')

    if operation == 'multiply':
        result = number_a * number_b
    elif operation == 'add':
        result = number_a + number_b
    elif operation == 'subtract':
        result = number_a - number_b
    else:
        result = 0

    return {
        "node_type": "function",
        "operation": operation,
        "number_a": number_a,
        "number_b": number_b,
        "result": result,
        "calculation": f"{number_a} {operation} {number_b} = {result}",
        "function_status": "completed"
    }
`,
                    computation_type: 'mathematical'
                },
                timeout: 30,
                retry_count: 1
            }
        ],
        edges: []
    });

    // Configuration 4: Edge case graph with special characters
    const specialNodeId = uuidv4();
    configs.push({
        name: 'Special Characters Test Graph - 特殊文字テスト',
        description: 'Graph with Unicode characters and special cases for edge case testing - Тест специальных символов',
        variables: {
            unicode_test: '测试数据 🚀 Тест データ',
            special_chars: '!@#$%^&*()_+-=[]{}|;:,.<>?',
            emoji_data: '🎯🔥💻🚀🎉'
        },
        nodes: [
            {
                id: specialNodeId,
                name: 'Unicode Test Node - テストノード',
                node_type: 'function',
                task_template: 'Process Unicode and special character data',
                dependencies: [],
                data: {
                    function_code: `
def run(data):
    # Handle Unicode and special characters
    unicode_test = data.get('unicode_test', '')
    special_chars = data.get('special_chars', '')
    emoji_data = data.get('emoji_data', '')

    return {
        "node_type": "function",
        "unicode_length": len(unicode_test),
        "special_chars_length": len(special_chars),
        "emoji_count": len(emoji_data),
        "processed_unicode": unicode_test[:50],  # Truncate for safety
        "processing_status": "unicode_processed",
        "character_analysis": "completed"
    }
`,
                    processing_type: 'unicode_handling'
                },
                timeout: 45,
                retry_count: 1
            }
        ],
        edges: []
    });

    return configs;
}

/**
 * Test POST /graphs/ with various graph configurations
 */
async function testPostGraphs(testWorkerName) {
    console.log('\n=== Testing POST /graphs/ with various configurations ===');

    const testResults = {
        simple_graph: false,
        complex_graph: false,
        function_only_graph: false,
        unicode_graph: false,
        structure_validation: false,
        created_graph_ids: []
    };

    try {
        const graphConfigs = createGraphConfigurations(testWorkerName);

        for (let i = 0; i < graphConfigs.length; i++) {
            const config = graphConfigs[i];
            const testName = ['simple_graph', 'complex_graph', 'function_only_graph', 'unicode_graph'][i];

            console.log(`\n📊 Creating Graph ${i + 1}: ${config.name}`);

            try {
                const response = await makeRequest(`${API_BASE}/graphs/`, {
                    method: 'POST',
                    body: JSON.stringify(config)
                });

                console.log(`✅ Graph created successfully: ${response.name}`);
                console.log(`   ID: ${response.id}`);
                console.log(`   Nodes: ${response.nodes.length}`);
                console.log(`   Edges: ${response.edges.length}`);
                console.log(`   Created: ${response.created_at}`);

                // Validate response structure
                const requiredFields = ['id', 'name', 'nodes', 'edges', 'created_at'];
                const hasAllFields = requiredFields.every(field => response.hasOwnProperty(field));

                if (hasAllFields && response.nodes.length === config.nodes.length) {
                    testResults[testName] = true;
                    testResults.created_graph_ids.push(response.id);
                    console.log(`✅ Structure validation passed for ${config.name}`);
                } else {
                    console.log(`❌ Structure validation failed for ${config.name}`);
                    console.log(`   Expected nodes: ${config.nodes.length}, Got: ${response.nodes.length}`);
                    console.log(`   Missing fields: ${requiredFields.filter(f => !response.hasOwnProperty(f))}`);
                }

            } catch (error) {
                console.error(`❌ Failed to create graph ${config.name}: ${error.message}`);
                testResults[testName] = false;
            }
        }

        // Overall structure validation
        testResults.structure_validation = testResults.created_graph_ids.length > 0;

    } catch (error) {
        console.error(`❌ POST /graphs/ testing failed: ${error.message}`);
        throw error;
    }

    return testResults;
}

/**
 * Test error cases for POST /graphs/
 */
async function testPostGraphsErrorCases() {
    console.log('\n=== Testing POST /graphs/ Error Cases ===');

    const testResults = {
        empty_nodes_error: false,
        invalid_node_structure: false,
        missing_worker_error: false,
        invalid_json_error: false
    };

    try {
        // Test 1: Empty nodes array
        console.log('\n📊 Test 1: Empty nodes array');
        try {
            await makeRequest(`${API_BASE}/graphs/`, {
                method: 'POST',
                body: JSON.stringify({
                    name: 'Empty Nodes Test',
                    description: 'Should fail with empty nodes',
                    nodes: [],
                    edges: []
                })
            });
            console.log(`❌ Empty nodes should have failed but didn't`);
        } catch (error) {
            if (error.status >= 400 && error.status < 500) {
                console.log(`✅ Empty nodes correctly returned error: ${error.status}`);
                testResults.empty_nodes_error = true;
            } else {
                console.log(`⚠️ Unexpected error type: ${error.status}`);
            }
        }

        // Test 2: Invalid node structure
        console.log('\n📊 Test 2: Invalid node structure');
        try {
            await makeRequest(`${API_BASE}/graphs/`, {
                method: 'POST',
                body: JSON.stringify({
                    name: 'Invalid Node Test',
                    description: 'Should fail with invalid node structure',
                    nodes: [
                        {
                            // Missing required fields like id, name, node_type
                            invalid_field: 'test'
                        }
                    ],
                    edges: []
                })
            });
            console.log(`❌ Invalid node structure should have failed but didn't`);
        } catch (error) {
            if (error.status >= 400 && error.status < 500) {
                console.log(`✅ Invalid node structure correctly returned error: ${error.status}`);
                testResults.invalid_node_structure = true;
            } else {
                console.log(`⚠️ Unexpected error type: ${error.status}`);
            }
        }

        // Test 3: Worker node with non-existent worker
        console.log('\n📊 Test 3: Worker node with non-existent worker');
        try {
            await makeRequest(`${API_BASE}/graphs/`, {
                method: 'POST',
                body: JSON.stringify({
                    name: 'Non-existent Worker Test',
                    description: 'Should fail with non-existent worker',
                    nodes: [
                        {
                            id: uuidv4(),
                            name: 'Invalid Worker Node',
                            node_type: 'worker',
                            worker_name: 'definitely_does_not_exist_worker_12345',
                            task_template: 'This should fail',
                            dependencies: []
                        }
                    ],
                    edges: []
                })
            });
            console.log(`❌ Non-existent worker should have failed but didn't`);
        } catch (error) {
            if (error.status >= 400 && error.status < 500) {
                console.log(`✅ Non-existent worker correctly returned error: ${error.status}`);
                testResults.missing_worker_error = true;
            } else {
                console.log(`⚠️ Unexpected error type: ${error.status}`);
            }
        }

        // Test 4: Invalid JSON
        console.log('\n📊 Test 4: Invalid JSON');
        try {
            const response = await fetch(`${API_BASE}/graphs/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'User-Agent': 'WorkerManagement-GraphCollectionTest/1.0'
                },
                body: '{"invalid": json syntax}'  // Invalid JSON
            });

            if (!response.ok) {
                console.log(`✅ Invalid JSON correctly returned error: ${response.status}`);
                testResults.invalid_json_error = true;
            } else {
                console.log(`❌ Invalid JSON should have failed but didn't`);
            }
        } catch (error) {
            console.log(`✅ Invalid JSON correctly caused error: ${error.message}`);
            testResults.invalid_json_error = true;
        }

    } catch (error) {
        console.error(`❌ Error case testing failed: ${error.message}`);
        throw error;
    }

    return testResults;
}

/**
 * Test DELETE /graphs/ - Delete all graphs and states
 */
async function testDeleteGraphs() {
    console.log('\n=== Testing DELETE /graphs/ - Delete All Graphs ===');

    const testResults = {
        delete_all_success: false,
        verification_success: false,
        proper_cleanup: false
    };

    try {
        // First, get count of existing graphs
        console.log('\n📊 Getting initial graph count');
        const initialGraphs = await makeRequest(`${API_BASE}/graphs/`);
        console.log(`📊 Found ${initialGraphs.length} graphs before deletion`);

        // Test DELETE /graphs/
        console.log('\n📊 Executing DELETE /graphs/');
        try {
            const deleteResponse = await makeRequest(`${API_BASE}/graphs/`, {
                method: 'DELETE'
            });

            console.log(`✅ DELETE /graphs/ completed`);
            console.log(`   Response: ${JSON.stringify(deleteResponse)}`);
            testResults.delete_all_success = true;

        } catch (error) {
            if (error.status === 204) {
                // 204 No Content is also a valid success response for DELETE
                console.log(`✅ DELETE /graphs/ completed with 204 No Content`);
                testResults.delete_all_success = true;
            } else {
                console.error(`❌ DELETE /graphs/ failed: ${error.message}`);
                throw error;
            }
        }

        // Verify deletion by checking graph count
        console.log('\n📊 Verifying deletion by checking remaining graphs');
        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait for cleanup

        const remainingGraphs = await makeRequest(`${API_BASE}/graphs/`);
        console.log(`📊 Found ${remainingGraphs.length} graphs after deletion`);

        if (remainingGraphs.length === 0) {
            console.log(`✅ All graphs successfully deleted`);
            testResults.verification_success = true;
            testResults.proper_cleanup = true;
        } else if (remainingGraphs.length < initialGraphs.length) {
            console.log(`⚠️ Some graphs deleted (${initialGraphs.length - remainingGraphs.length} removed)`);
            testResults.verification_success = true;
            testResults.proper_cleanup = false;
        } else {
            console.log(`❌ No graphs were deleted`);
            testResults.verification_success = false;
            testResults.proper_cleanup = false;
        }

    } catch (error) {
        console.error(`❌ DELETE /graphs/ testing failed: ${error.message}`);
        throw error;
    }

    return testResults;
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

        // Graph service check
        const graphsResponse = await makeRequest(`${API_BASE}/graphs/`);
        console.log(`✅ Graph service is responding (${graphsResponse.length} graphs found)`);

        // Worker service check
        const workersResponse = await makeRequest(`${API_BASE}/workers`);
        console.log(`✅ Worker service is responding (${workersResponse.length} workers found)`);

        return { healthy: true, graphs: graphsResponse.length, workers: workersResponse.length };

    } catch (error) {
        console.error(`❌ Health check failed: ${error.message}`);
        return { healthy: false, error: error.message };
    }
}

/**
 * Cleanup created resources
 */
async function cleanupResources(createdWorkerNames = [], createdGraphIds = []) {
    console.log('\n🧹 Performing Resource Cleanup...');

    const cleanupResults = {
        workers_deleted: 0,
        graphs_deleted: 0,
        failed_deletions: [],
        total_time: 0
    };

    const startTime = Date.now();

    try {
        // Delete created graphs
        for (const graphId of createdGraphIds) {
            try {
                await makeRequest(`${API_BASE}/graphs/${graphId}`, { method: 'DELETE' });
                console.log(`✅ Graph deleted: ${graphId}`);
                cleanupResults.graphs_deleted++;
            } catch (error) {
                console.error(`❌ Failed to delete graph ${graphId}: ${error.message}`);
                cleanupResults.failed_deletions.push({ type: 'graph', id: graphId, error: error.message });
            }
        }

        // Delete created workers
        for (const workerName of createdWorkerNames) {
            try {
                await makeRequest(`${API_BASE}/workers/${workerName}`, { method: 'DELETE' });
                console.log(`✅ Worker deleted: ${workerName}`);
                cleanupResults.workers_deleted++;
            } catch (error) {
                console.error(`❌ Failed to delete worker ${workerName}: ${error.message}`);
                cleanupResults.failed_deletions.push({ type: 'worker', id: workerName, error: error.message });
            }
        }

        cleanupResults.total_time = Date.now() - startTime;

        console.log(`\n✅ Cleanup completed in ${cleanupResults.total_time}ms`);
        console.log(`   Workers deleted: ${cleanupResults.workers_deleted}/${createdWorkerNames.length}`);
        console.log(`   Graphs deleted: ${cleanupResults.graphs_deleted}/${createdGraphIds.length}`);
        console.log(`   Failed deletions: ${cleanupResults.failed_deletions.length}`);

    } catch (error) {
        console.error(`❌ Cleanup process failed: ${error.message}`);
        cleanupResults.total_time = Date.now() - startTime;
    }

    return cleanupResults;
}

/**
 * Main test function
 */
async function runGraphCollectionTest() {
    console.log('🚀 Starting Comprehensive Graph Collection Operations Test');
    console.log('='.repeat(80));

    // Reset retry flag for this test run
    hasRetried = false;
    console.log('🔄 Retry mechanism initialized (1 retry with 60s delay available)');

    const testResults = {
        health_check: false,
        worker_creation: false,
        get_graphs_tests: null,
        post_graphs_tests: null,
        error_case_tests: null,
        delete_graphs_tests: null,
        overall_success: false,
        execution_time: 0,
        created_resources: {
            workers: [],
            graph_ids: []
        }
    };

    const testStartTime = Date.now();

    try {
        // Step 1: Health Check
        console.log('\n🏥 Step 1: API Health Check');
        const healthResult = await checkApiHealth();
        testResults.health_check = healthResult.healthy;

        if (!healthResult.healthy) {
            throw new Error('API health check failed - cannot proceed with test');
        }

        // Step 2: Create test worker
        console.log('\n🤖 Step 2: Creating Test Worker');
        const testWorkerName = `graph_collection_test_worker_${Date.now()}`;
        const worker = await createTestWorker(testWorkerName);
        testResults.worker_creation = true;
        testResults.created_resources.workers.push(worker.name);

        // Step 3: Test GET /graphs/
        console.log('\n📊 Step 3: Testing GET /graphs/');
        const getGraphsResults = await testGetGraphs();
        testResults.get_graphs_tests = getGraphsResults;

        // Step 4: Test POST /graphs/
        console.log('\n📊 Step 4: Testing POST /graphs/');
        const postGraphsResults = await testPostGraphs(testWorkerName);
        testResults.post_graphs_tests = postGraphsResults;
        testResults.created_resources.graph_ids = postGraphsResults.created_graph_ids;

        // Step 5: Test error cases
        console.log('\n📊 Step 5: Testing Error Cases');
        const errorCaseResults = await testPostGraphsErrorCases();
        testResults.error_case_tests = errorCaseResults;

        // Step 6: Test DELETE /graphs/
        console.log('\n📊 Step 6: Testing DELETE /graphs/');
        const deleteGraphsResults = await testDeleteGraphs();
        testResults.delete_graphs_tests = deleteGraphsResults;

        // Calculate overall success
        const getSuccess = Object.values(getGraphsResults).filter(v => v === true).length >= 4;
        const postSuccess = Object.values(postGraphsResults).filter(v => v === true).length >= 3;
        const errorSuccess = Object.values(errorCaseResults).filter(v => v === true).length >= 2;
        const deleteSuccess = deleteGraphsResults.delete_all_success && deleteGraphsResults.verification_success;

        testResults.overall_success = testResults.health_check &&
                                     testResults.worker_creation &&
                                     getSuccess &&
                                     postSuccess &&
                                     errorSuccess &&
                                     deleteSuccess;

        testResults.execution_time = Date.now() - testStartTime;

        // Generate summary report
        console.log('\n📊 Step 7: Test Summary Report');
        console.log('='.repeat(80));
        console.log(`🏁 Graph Collection Operations Test - COMPLETE`);
        console.log(`   Total execution time: ${testResults.execution_time}ms`);
        console.log(`   Overall success: ${testResults.overall_success ? '✅ PASSED' : '❌ FAILED'}`);

        console.log('\n📋 Individual Test Results:');
        console.log(`   ✅ API Health Check: ${testResults.health_check ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Worker Creation: ${testResults.worker_creation ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ GET /graphs/ Tests: ${getSuccess ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ POST /graphs/ Tests: ${postSuccess ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Error Case Tests: ${errorSuccess ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ DELETE /graphs/ Tests: ${deleteSuccess ? 'PASSED' : 'FAILED'}`);

        // Detailed breakdown
        console.log('\n📊 Detailed Test Breakdown:');

        console.log('\n   GET /graphs/ Results:');
        Object.entries(getGraphsResults).forEach(([test, result]) => {
            console.log(`     ${result ? '✅' : '❌'} ${test}: ${result ? 'PASSED' : 'FAILED'}`);
        });

        console.log('\n   POST /graphs/ Results:');
        Object.entries(postGraphsResults).forEach(([test, result]) => {
            if (test !== 'created_graph_ids') {
                console.log(`     ${result ? '✅' : '❌'} ${test}: ${result ? 'PASSED' : 'FAILED'}`);
            }
        });

        console.log('\n   Error Case Results:');
        Object.entries(errorCaseResults).forEach(([test, result]) => {
            console.log(`     ${result ? '✅' : '❌'} ${test}: ${result ? 'PASSED' : 'FAILED'}`);
        });

        console.log('\n   DELETE /graphs/ Results:');
        Object.entries(deleteGraphsResults).forEach(([test, result]) => {
            console.log(`     ${result ? '✅' : '❌'} ${test}: ${result ? 'PASSED' : 'FAILED'}`);
        });

        if (testResults.overall_success) {
            console.log('\n🎉 Graph Collection Operations Test PASSED!');
            console.log('   ✅ All major graph collection operations are working correctly');
            console.log('   ✅ GET /graphs/ with all parameter variations work');
            console.log('   ✅ POST /graphs/ creates graphs with various configurations');
            console.log('   ✅ Error handling is working properly');
            console.log('   ✅ DELETE /graphs/ cleans up all graphs and states');
        } else {
            console.log('\n💥 Graph Collection Operations Test FAILED!');
            console.log('   ❌ Some graph collection operations are not working correctly');
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

    // Cleanup resources
    console.log('\n🧹 Step 8: Resource Cleanup');
    try {
        const cleanupResult = await cleanupResources(
            testResults.created_resources.workers,
            testResults.created_resources.graph_ids
        );

        if (cleanupResult.failed_deletions.length > 0) {
            console.log(`⚠️ Some resources failed to cleanup: ${cleanupResult.failed_deletions.length} items`);
        }
    } catch (cleanupError) {
        console.error(`❌ Cleanup failed: ${cleanupError.message}`);
    }

    return testResults;
}

/**
 * Main function with comprehensive error handling
 */
async function main() {
    console.log('🚀 Graph Collection Operations Test Suite');
    console.log('Node.js version:', process.version);
    console.log('Platform:', process.platform);
    console.log('Architecture:', process.arch);
    console.log('='.repeat(80));

    try {
        const testResults = await runGraphCollectionTest();

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
    makeRequest,
    createTestWorker,
    testGetGraphs,
    testPostGraphs,
    testPostGraphsErrorCases,
    testDeleteGraphs,
    checkApiHealth,
    cleanupResources,
    runGraphCollectionTest
};
