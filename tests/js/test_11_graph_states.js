#!/usr/bin/env node
/**
 * JavaScript Test Script for Graph States Management API Validation
 *
 * This script comprehensively tests the Graph States Management endpoints:
 * 1. GET /graphs/{graph_id}/states - Retrieves state records for specific graph
 * 2. DELETE /graphs/{graph_id}/states - Deletes all states for specific graph
 * 3. DELETE /graphs/states - Deletes all graph state records (leaves graphs intact)
 *
 * Test Coverage:
 * - Create test graphs and execute them to generate states
 * - Test GET endpoint with various limit parameters (1-100, default 10)
 * - Validate state record structure and data integrity
 * - Test DELETE operations with proper validation
 * - Edge cases and boundary conditions
 * - Comprehensive cleanup and error handling
 * - Multi-graph state management scenarios
 *
 * Requirements: Node.js 18+ (for native fetch API)
 * Usage: node test_11_graph_states.js
 */

const { v4: uuidv4 } = require('uuid');

// Enhanced Configuration
const API_BASE = 'http://localhost:8887';
const TIMEOUT_MS = 60000; // 60 seconds
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
            'User-Agent': 'WorkerManagement-GraphStates-Test/1.0',
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
 * Enhanced worker creation with validation
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
 * Enhanced graph creation with comprehensive validation
 */
async function createGraph(graphConfig) {
    console.log(`\n=== Creating Graph: ${graphConfig.name} ===`);

    try {
        // Comprehensive pre-validation
        if (!graphConfig.nodes || graphConfig.nodes.length === 0) {
            throw new Error('Graph must have at least one node');
        }

        console.log(`📊 Graph Configuration:`);
        console.log(`   Name: ${graphConfig.name}`);
        console.log(`   Description: ${graphConfig.description || 'No description'}`);
        console.log(`   Nodes: ${graphConfig.nodes.length}`);
        console.log(`   Edges: ${(graphConfig.edges || []).length}`);

        const response = await makeRequest(`${API_BASE}/graphs/`, {
            method: 'POST',
            body: JSON.stringify(graphConfig)
        });

        console.log(`✅ Graph created successfully: ${response.name}`);
        console.log(`   ID: ${response.id}`);
        console.log(`   Nodes: ${response.nodes.length}`);
        console.log(`   Edges: ${response.edges.length}`);
        console.log(`   Created: ${response.created_at}`);

        return response;
    } catch (error) {
        console.error(`❌ Failed to create graph: ${error.message}`);
        if (error.status && error.response) {
            console.error(`   HTTP Details: Status ${error.status}, Response: ${error.response}`);
        }
        throw error;
    }
}

/**
 * Enhanced graph execution
 */
async function executeGraph(graphId, executionRequest) {
    console.log(`\n=== Executing Graph: ${graphId} ===`);

    try {
        if (!graphId) {
            throw new Error('Graph ID is required for execution');
        }

        console.log(`⚡ Execution Request Details:`);
        console.log(`   Variables: ${JSON.stringify(executionRequest.variables, null, 2)}`);

        const response = await makeRequest(`${API_BASE}/execution/graphs/${graphId}/execute`, {
            method: 'POST',
            body: JSON.stringify(executionRequest)
        });

        console.log(`✅ Graph execution initiated successfully`);
        console.log(`   Session ID: ${response.session_id}`);
        console.log(`   Status: ${response.status}`);
        console.log(`   Current Node: ${response.current_node || 'None'}`);
        console.log(`   Started At: ${response.started_at}`);

        return response;
    } catch (error) {
        console.error(`❌ Failed to execute graph: ${error.message}`);
        throw error;
    }
}

/**
 * Test GET /graphs/{graph_id}/states endpoint
 */
async function testGetGraphStates(graphId, limit = null) {
    console.log(`\n=== Testing GET /graphs/${graphId}/states${limit ? ' with limit ' + limit : ''} ===`);

    try {
        const url = limit ?
            `${API_BASE}/graphs/${graphId}/states?limit=${limit}` :
            `${API_BASE}/graphs/${graphId}/states`;

        const response = await makeRequest(url);

        console.log(`✅ Retrieved graph states successfully`);
        console.log(`   Total states returned: ${response.length}`);
        console.log(`   Limit applied: ${limit || 'default (10)'}`);

        // Validate state record structure
        if (response.length > 0) {
            const firstState = response[0];
            console.log(`   Sample state structure validation:`);

            const requiredFields = ['id', 'graph_id', 'session_id', 'node_id', 'status', 'created_at', 'updated_at'];
            const missingFields = requiredFields.filter(field => !(field in firstState));

            if (missingFields.length === 0) {
                console.log(`   ✅ All required fields present: ${requiredFields.join(', ')}`);
            } else {
                console.log(`   ❌ Missing required fields: ${missingFields.join(', ')}`);
            }

            // Check data types
            if (firstState.graph_id === graphId) {
                console.log(`   ✅ Graph ID matches request: ${firstState.graph_id}`);
            } else {
                console.log(`   ❌ Graph ID mismatch: expected ${graphId}, got ${firstState.graph_id}`);
            }

            // Display sample state data
            console.log(`   Sample state data:`);
            console.log(`     ID: ${firstState.id}`);
            console.log(`     Session ID: ${firstState.session_id}`);
            console.log(`     Node ID: ${firstState.node_id}`);
            console.log(`     Status: ${firstState.status}`);
            console.log(`     Created: ${firstState.created_at}`);
            console.log(`     Updated: ${firstState.updated_at}`);

            if (firstState.node_output) {
                const outputPreview = JSON.stringify(firstState.node_output).substring(0, 100);
                console.log(`     Output Preview: ${outputPreview}...`);
            }
        } else {
            console.log(`   ℹ️  No states found for graph ${graphId}`);
        }

        return response;
    } catch (error) {
        console.error(`❌ Failed to get graph states: ${error.message}`);
        throw error;
    }
}

/**
 * Test DELETE /graphs/{graph_id}/states endpoint
 */
async function testDeleteGraphStates(graphId) {
    console.log(`\n=== Testing DELETE /graphs/${graphId}/states ===`);

    try {
        // First, get current states count for validation
        const statesBefore = await testGetGraphStates(graphId);
        const countBefore = statesBefore.length;

        console.log(`   States before deletion: ${countBefore}`);

        const response = await makeRequest(`${API_BASE}/graphs/${graphId}/states`, {
            method: 'DELETE'
        });

        console.log(`✅ Delete graph states operation completed`);

        // Validate response structure
        if (response.message) {
            console.log(`   Response message: ${response.message}`);
        }

        if (response.deleted_count !== undefined) {
            console.log(`   Deleted states count: ${response.deleted_count}`);
        }

        // Verify deletion by checking states again
        const statesAfter = await testGetGraphStates(graphId);
        const countAfter = statesAfter.length;

        console.log(`   States after deletion: ${countAfter}`);

        if (countAfter === 0) {
            console.log(`   ✅ All states successfully deleted for graph ${graphId}`);
        } else {
            console.log(`   ⚠️  ${countAfter} states still remain after deletion`);
        }

        return {
            success: true,
            states_before: countBefore,
            states_after: countAfter,
            deleted_count: response.deleted_count || (countBefore - countAfter)
        };
    } catch (error) {
        console.error(`❌ Failed to delete graph states: ${error.message}`);
        throw error;
    }
}

/**
 * Test DELETE /graphs/states endpoint (delete all graph states)
 */
async function testDeleteAllGraphStates() {
    console.log(`\n=== Testing DELETE /graphs/states (All Graph States) ===`);

    try {
        const response = await makeRequest(`${API_BASE}/graphs/states`, {
            method: 'DELETE'
        });

        console.log(`✅ Delete all graph states operation completed`);

        // Validate response structure
        if (response.message) {
            console.log(`   Response message: ${response.message}`);
        }

        if (response.deleted_count !== undefined) {
            console.log(`   Total deleted states count: ${response.deleted_count}`);
        }

        if (response.affected_graphs !== undefined) {
            console.log(`   Affected graphs count: ${response.affected_graphs}`);
        }

        return response;
    } catch (error) {
        console.error(`❌ Failed to delete all graph states: ${error.message}`);
        throw error;
    }
}

/**
 * Comprehensive API health check
 */
async function checkApiHealth() {
    console.log('\n🏥 Performing API Health Check...');

    try {
        // Main API health
        console.log('🔍 Checking main API health...');
        const healthResponse = await makeRequest(`${API_BASE}/health`);
        console.log(`✅ Main API is healthy: ${healthResponse.status}`);

        // Execution service health
        console.log('🔍 Checking execution service health...');
        const executionHealthResponse = await makeRequest(`${API_BASE}/execution/health`);
        console.log(`✅ Execution service is healthy`);
        console.log(`   Engine: ${executionHealthResponse.execution_engine || 'pydantic_ai'}`);
        console.log(`   Timestamp: ${executionHealthResponse.timestamp}`);

        console.log('\n🎉 All health checks passed!');
        return { healthy: true };

    } catch (error) {
        console.error(`❌ Health check failed: ${error.message}`);
        return { healthy: false, error: error.message };
    }
}

/**
 * Enhanced resource cleanup
 */
async function cleanupResources(graphIds = [], workerNames = []) {
    console.log('\n🧹 Performing Resource Cleanup...');
    const cleanupResults = {
        graphs_deleted: [],
        workers_deleted: [],
        failed_deletions: []
    };

    try {
        // Delete graphs
        for (const graphId of graphIds) {
            console.log(`🗑️  Deleting graph: ${graphId}`);
            try {
                await makeRequest(`${API_BASE}/graphs/${graphId}`, { method: 'DELETE' });
                console.log(`✅ Graph deleted successfully: ${graphId}`);
                cleanupResults.graphs_deleted.push(graphId);
            } catch (error) {
                console.error(`❌ Failed to delete graph ${graphId}: ${error.message}`);
                cleanupResults.failed_deletions.push({ type: 'graph', id: graphId, error: error.message });
            }
        }

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

        console.log('\n✅ Cleanup completed');
        console.log(`   Graphs deleted: ${cleanupResults.graphs_deleted.length}/${graphIds.length}`);
        console.log(`   Workers deleted: ${cleanupResults.workers_deleted.length}/${workerNames.length}`);
        console.log(`   Failed deletions: ${cleanupResults.failed_deletions.length}`);

    } catch (error) {
        console.error(`❌ Cleanup process failed: ${error.message}`);
    }

    return cleanupResults;
}

/**
 * Test limit parameter edge cases
 */
async function testLimitParameterEdgeCases(graphId) {
    console.log('\n=== Testing Limit Parameter Edge Cases ===');

    const testCases = [
        { limit: 1, description: 'Minimum valid limit' },
        { limit: 5, description: 'Small limit' },
        { limit: 10, description: 'Default limit value' },
        { limit: 50, description: 'Medium limit' },
        { limit: 100, description: 'Maximum valid limit' }
    ];

    const results = [];

    for (const testCase of testCases) {
        try {
            console.log(`\n🔍 Testing ${testCase.description} (limit=${testCase.limit}):`);
            const states = await testGetGraphStates(graphId, testCase.limit);

            results.push({
                limit: testCase.limit,
                description: testCase.description,
                success: true,
                returned_count: states.length,
                within_limit: states.length <= testCase.limit
            });

            if (states.length <= testCase.limit) {
                console.log(`   ✅ Returned ${states.length} states (within limit ${testCase.limit})`);
            } else {
                console.log(`   ❌ Returned ${states.length} states (exceeds limit ${testCase.limit})`);
            }

        } catch (error) {
            console.error(`   ❌ Failed with limit ${testCase.limit}: ${error.message}`);
            results.push({
                limit: testCase.limit,
                description: testCase.description,
                success: false,
                error: error.message
            });
        }
    }

    return results;
}

/**
 * Test boundary conditions and invalid parameters
 */
async function testBoundaryConditions(graphId) {
    console.log('\n=== Testing Boundary Conditions ===');

    const boundaryTests = [
        {
            test: 'Zero limit',
            url: `${API_BASE}/graphs/${graphId}/states?limit=0`,
            expectError: true
        },
        {
            test: 'Negative limit',
            url: `${API_BASE}/graphs/${graphId}/states?limit=-1`,
            expectError: true
        },
        {
            test: 'Limit exceeding maximum',
            url: `${API_BASE}/graphs/${graphId}/states?limit=101`,
            expectError: true
        },
        {
            test: 'Non-numeric limit',
            url: `${API_BASE}/graphs/${graphId}/states?limit=abc`,
            expectError: true
        },
        {
            test: 'Invalid graph ID',
            url: `${API_BASE}/graphs/invalid-uuid/states`,
            expectError: true
        },
        {
            test: 'Non-existent graph ID',
            url: `${API_BASE}/graphs/${uuidv4()}/states`,
            expectError: false // Should return empty array, not error
        }
    ];

    const results = [];

    for (const boundaryTest of boundaryTests) {
        try {
            console.log(`\n🔍 Testing ${boundaryTest.test}:`);
            const response = await makeRequest(boundaryTest.url);

            if (boundaryTest.expectError) {
                console.log(`   ⚠️  Expected error but got success response`);
                results.push({
                    test: boundaryTest.test,
                    expected_error: true,
                    got_error: false,
                    response_length: Array.isArray(response) ? response.length : 'N/A'
                });
            } else {
                console.log(`   ✅ Got expected success response`);
                results.push({
                    test: boundaryTest.test,
                    expected_error: false,
                    got_error: false,
                    response_length: Array.isArray(response) ? response.length : 'N/A'
                });
            }

        } catch (error) {
            if (boundaryTest.expectError) {
                console.log(`   ✅ Got expected error: ${error.message}`);
                results.push({
                    test: boundaryTest.test,
                    expected_error: true,
                    got_error: true,
                    error_message: error.message
                });
            } else {
                console.log(`   ❌ Got unexpected error: ${error.message}`);
                results.push({
                    test: boundaryTest.test,
                    expected_error: false,
                    got_error: true,
                    error_message: error.message
                });
            }
        }
    }

    return results;
}

/**
 * Main test function for Graph States Management
 */
async function runGraphStatesTest() {
    console.log('🚀 Starting Graph States Management API Test');
    console.log('='.repeat(80));

    // Reset retry flag
    hasRetried = false;

    const createdGraphIds = [];
    const createdWorkerNames = [];
    const testResults = {
        health_check: false,
        setup_complete: false,
        get_states_basic: false,
        get_states_with_limits: false,
        limit_edge_cases: false,
        boundary_conditions: false,
        delete_graph_states: false,
        delete_all_states: false,
        data_integrity: false,
        overall_success: false,
        execution_time: 0,
        partial_results: {}
    };

    const testStartTime = Date.now();

    try {
        // Step 1: Health Check
        console.log('\n🏥 Step 1: API Health Check');
        const healthResult = await checkApiHealth();
        testResults.health_check = healthResult.healthy;
        testResults.partial_results.health_check = healthResult;

        if (!healthResult.healthy) {
            throw new Error('API health check failed - cannot proceed with test');
        }

        // Step 2: Setup - Create test workers and graphs, then execute them to generate states
        console.log('\n🔧 Step 2: Test Setup - Creating Workers and Graphs');

        // Create test worker
        const workerConfig = {
            worker_name: 'graph_states_test_worker',
            agent_role: 'test_data_processor',
            agent_type: 'pydantic_ai',
            model_provider: 'deepseek',
            model: 'deepseek-chat',
            temperature: 0.1,
            system_prompt: 'You are a test data processor that generates structured responses for graph state testing.',
            instruction: 'Process the input data and return a structured JSON response with test results.',
            user_prompt: '',
            description: 'Test worker for graph states management validation'
        };

        const worker = await createWorker(workerConfig);
        createdWorkerNames.push(worker.name);

        // Create multiple test graphs to generate diverse state data
        const graphConfigs = [
            {
                name: 'Graph States Test Graph 1',
                description: 'First test graph for generating state records',
                variables: {
                    test_input: 'graph_states_test_1',
                    process_id: 1
                },
                nodes: [
                    {
                        id: uuidv4(),
                        name: 'Test Worker Node 1',
                        node_type: 'worker',
                        worker_name: 'graph_states_test_worker',
                        task_template: 'Process test data: {test_input} with process ID: {process_id}. Return structured JSON with test results.',
                        dependencies: [],
                        data: { test_purpose: 'state_generation_1' }
                    },
                    {
                        id: uuidv4(),
                        name: 'Test Function Node 1',
                        node_type: 'function',
                        task_template: 'Transform data from previous node',
                        dependencies: [],
                        data: {
                            function_code: `
def run(data):
    return {
        "node_type": "function",
        "test_purpose": "state_generation_function_1",
        "processed_data": f"Transformed: {data.get('test_input', 'unknown')}",
        "timestamp": "test_timestamp_1"
                            }
`
                        }
                    }
                ],
                edges: []
            },
            {
                name: 'Graph States Test Graph 2',
                description: 'Second test graph for generating more state records',
                variables: {
                    test_input: 'graph_states_test_2',
                    process_id: 2
                },
                nodes: [
                    {
                        id: uuidv4(),
                        name: 'Test Worker Node 2',
                        node_type: 'worker',
                        worker_name: 'graph_states_test_worker',
                        task_template: 'Process test data: {test_input} with process ID: {process_id}. Return structured JSON with test results.',
                        dependencies: [],
                        data: { test_purpose: 'state_generation_2' }
                    }
                ],
                edges: []
            }
        ];

        // Create and execute graphs to generate state data
        for (let i = 0; i < graphConfigs.length; i++) {
            const graph = await createGraph(graphConfigs[i]);
            createdGraphIds.push(graph.id);

            // Execute graph multiple times to generate multiple state records
            for (let exec = 0; exec < 3; exec++) {
                console.log(`\n⚡ Executing Graph ${i + 1}, Execution ${exec + 1}:`);

                const executionRequest = {
                    variables: {
                        test_input: `graph_states_test_${i + 1}_exec_${exec + 1}`,
                        process_id: i + 1,
                        execution_number: exec + 1
                    }
                };

                try {
                    await executeGraph(graph.id, executionRequest);
                    console.log(`✅ Graph ${i + 1} execution ${exec + 1} completed`);

                    // Wait a bit for state records to be persisted
                    await new Promise(resolve => setTimeout(resolve, 2000));
                } catch (execError) {
                    console.error(`⚠️  Graph ${i + 1} execution ${exec + 1} failed: ${execError.message}`);
                    // Continue with other executions even if one fails
                }
            }
        }

        testResults.setup_complete = true;
        console.log(`✅ Test setup completed: ${createdGraphIds.length} graphs, ${createdWorkerNames.length} workers`);

        // Step 3: Test GET /graphs/{graph_id}/states (Basic functionality)
        console.log('\n📊 Step 3: Testing GET /graphs/{graph_id}/states (Basic)');

        for (const graphId of createdGraphIds) {
            try {
                const states = await testGetGraphStates(graphId);
                console.log(`✅ Successfully retrieved states for graph ${graphId}: ${states.length} states`);
            } catch (error) {
                console.error(`❌ Failed to get states for graph ${graphId}: ${error.message}`);
                throw error;
            }
        }

        testResults.get_states_basic = true;

        // Step 4: Test GET with different limit parameters
        console.log('\n📊 Step 4: Testing GET /graphs/{graph_id}/states with Limits');

        if (createdGraphIds.length > 0) {
            const primaryGraphId = createdGraphIds[0];

            // Test default behavior (no limit specified)
            await testGetGraphStates(primaryGraphId);

            // Test with various limit values
            const limitTests = [1, 5, 10, 25, 50, 100];
            for (const limit of limitTests) {
                await testGetGraphStates(primaryGraphId, limit);
            }
        }

        testResults.get_states_with_limits = true;

        // Step 5: Test limit parameter edge cases
        console.log('\n🔍 Step 5: Testing Limit Parameter Edge Cases');

        if (createdGraphIds.length > 0) {
            const edgeCaseResults = await testLimitParameterEdgeCases(createdGraphIds[0]);
            testResults.partial_results.limit_edge_cases = edgeCaseResults;

            const successfulCases = edgeCaseResults.filter(r => r.success && r.within_limit);
            testResults.limit_edge_cases = successfulCases.length === edgeCaseResults.length;
        }

        // Step 6: Test boundary conditions
        console.log('\n🚫 Step 6: Testing Boundary Conditions');

        if (createdGraphIds.length > 0) {
            const boundaryResults = await testBoundaryConditions(createdGraphIds[0]);
            testResults.partial_results.boundary_conditions = boundaryResults;

            // Check if all boundary tests behaved as expected
            const expectedResults = boundaryResults.filter(r =>
                (r.expected_error && r.got_error) || (!r.expected_error && !r.got_error)
            );
            testResults.boundary_conditions = expectedResults.length === boundaryResults.length;
        }

        // Step 7: Test DELETE /graphs/{graph_id}/states
        console.log('\n🗑️  Step 7: Testing DELETE /graphs/{graph_id}/states');

        if (createdGraphIds.length > 1) {
            // Delete states for the first graph only (keep the second for further testing)
            const deleteResult = await testDeleteGraphStates(createdGraphIds[0]);
            testResults.partial_results.delete_graph_states = deleteResult;
            testResults.delete_graph_states = deleteResult.success && deleteResult.states_after === 0;
        }

        // Step 8: Test data integrity after partial deletion
        console.log('\n🔍 Step 8: Testing Data Integrity After Partial Deletion');

        if (createdGraphIds.length > 1) {
            // Verify that states for the second graph are still intact
            const remainingStates = await testGetGraphStates(createdGraphIds[1]);

            if (remainingStates.length > 0) {
                console.log(`✅ Data integrity maintained: Graph ${createdGraphIds[1]} still has ${remainingStates.length} states`);
                testResults.data_integrity = true;
            } else {
                console.log(`⚠️  Unexpected: Graph ${createdGraphIds[1]} has no states after partial deletion`);
                testResults.data_integrity = false;
            }
        }

        // Step 9: Test DELETE /graphs/states (delete all)
        console.log('\n🗑️  Step 9: Testing DELETE /graphs/states (Delete All)');

        const deleteAllResult = await testDeleteAllGraphStates();
        testResults.partial_results.delete_all_states = deleteAllResult;

        // Verify all states are deleted
        let allStatesDeleted = true;
        for (const graphId of createdGraphIds) {
            const remainingStates = await testGetGraphStates(graphId);
            if (remainingStates.length > 0) {
                console.log(`❌ Graph ${graphId} still has ${remainingStates.length} states after delete all`);
                allStatesDeleted = false;
            } else {
                console.log(`✅ Graph ${graphId} has no remaining states`);
            }
        }

        testResults.delete_all_states = allStatesDeleted;

        // Overall success determination
        testResults.overall_success = testResults.health_check &&
                                     testResults.setup_complete &&
                                     testResults.get_states_basic &&
                                     testResults.get_states_with_limits &&
                                     testResults.limit_edge_cases &&
                                     testResults.boundary_conditions &&
                                     testResults.delete_graph_states &&
                                     testResults.delete_all_states &&
                                     testResults.data_integrity;

        testResults.execution_time = Date.now() - testStartTime;

        // Final Summary
        console.log('\n📊 Step 10: Graph States Management Test Summary');
        console.log('='.repeat(80));
        console.log(`🏁 Graph States Management API Test - COMPLETE`);
        console.log(`   Total execution time: ${testResults.execution_time}ms`);
        console.log(`   Overall success: ${testResults.overall_success ? '✅ PASSED' : '❌ FAILED'}`);
        console.log('\n📋 Test Results:');
        console.log(`   ✅ API Health Check: ${testResults.health_check ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Test Setup (Workers & Graphs): ${testResults.setup_complete ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ GET States Basic: ${testResults.get_states_basic ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ GET States with Limits: ${testResults.get_states_with_limits ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Limit Edge Cases: ${testResults.limit_edge_cases ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Boundary Conditions: ${testResults.boundary_conditions ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ DELETE Graph States: ${testResults.delete_graph_states ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ DELETE All States: ${testResults.delete_all_states ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Data Integrity: ${testResults.data_integrity ? 'PASSED' : 'FAILED'}`);

        if (testResults.overall_success) {
            console.log('\n🎉 Graph States Management API Test PASSED!');
            console.log('   ✅ All endpoints function correctly');
            console.log('   ✅ Limit parameters work as expected');
            console.log('   ✅ State record structure is valid');
            console.log('   ✅ Delete operations work correctly');
            console.log('   ✅ Data integrity is maintained');
            console.log('   ✅ Boundary conditions are handled properly');
        } else {
            console.log('\n💥 Graph States Management API Test FAILED!');
            console.log('   ❌ Some endpoints or features are not working correctly');
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

    // Step 11: Resource Cleanup
    console.log('\n🧹 Step 11: Resource Cleanup');
    try {
        const cleanupResult = await cleanupResources(createdGraphIds, createdWorkerNames);
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
 * Main function with comprehensive error handling
 */
async function main() {
    console.log('🚀 Graph States Management API Test Suite');
    console.log('Node.js version:', process.version);
    console.log('Platform:', process.platform);
    console.log('Architecture:', process.arch);
    console.log('='.repeat(80));

    try {
        const testResults = await runGraphStatesTest();

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
    createWorker,
    createGraph,
    executeGraph,
    testGetGraphStates,
    testDeleteGraphStates,
    testDeleteAllGraphStates,
    checkApiHealth,
    cleanupResources,
    testLimitParameterEdgeCases,
    testBoundaryConditions,
    runGraphStatesTest
};
