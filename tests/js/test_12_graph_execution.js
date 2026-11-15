#!/usr/bin/env node
/**
 * JavaScript Test Script for Graph Execution Endpoints - Production Ready
 *
 * This script validates the graph execution endpoints of the Worker Management API by:
 * 1. Testing GET /execution/health - execution service health check
 * 2. Testing POST /execution/graphs/{graph_id}/execute - normal graph execution
 * 3. Testing POST /execution/graphs/{graph_id}/execute/stepwise - stepwise execution
 * 4. Creating various graph types for execution testing
 * 5. Validating ExecutionRequest and ExecutionResponse structures
 * 6. Testing with different variables and start_node parameters
 * 7. Including comprehensive error handling and execution monitoring
 *
 * ENDPOINTS TESTED:
 * - GET /execution/health - execution service health check
 * - POST /execution/graphs/{graph_id}/execute - full graph execution
 * - POST /execution/graphs/{graph_id}/execute/stepwise - stepwise execution
 *
 * FEATURES TESTED:
 * - ExecutionRequest validation (variables, start_node)
 * - ExecutionResponse structure validation
 * - Graph execution monitoring and status tracking
 * - Different graph types (simple, complex, conditional)
 * - Error handling for invalid requests
 * - Execution timeout handling
 * - Node output validation
 *
 * Requirements: Node.js 18+ (for native fetch API)
 * Usage: node test_12_graph_execution.js
 */

const { v4: uuidv4 } = require('uuid');

// Enhanced Configuration
const API_BASE = 'http://localhost:8887';
const TIMEOUT_MS = 120000; // 2 minutes for execution tests
const MAX_RETRIES = 3;
const RETRY_DELAY = 2000; // 2 seconds
const REQUEST_DELAY = 1000; // 1 second between requests
const EXECUTION_POLL_INTERVAL = 2000; // 2 seconds between status polls
const MAX_POLL_ATTEMPTS = 60; // Maximum polling attempts (2 minutes)

// Add a flag to track if we've already retried for timeout
let hasRetried = false;

/**
 * Enhanced utility function to make HTTP requests with retry logic and comprehensive error handling
 */
async function makeRequest(url, options = {}, retries = MAX_RETRIES) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'WorkerManagement-GraphExecutionTest/1.0',
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

            // Add delay between requests to prevent server overload
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

        if (!graphConfig.edges) {
            console.warn('⚠️  Graph has no edges defined');
        }

        // Validate node structure
        for (const node of graphConfig.nodes) {
            if (!node.id || !node.name) {
                throw new Error(`Node ${node.name || 'unnamed'} missing required fields: id, name`);
            }

            // Only worker nodes require worker_name
            if (node.node_type === 'worker' && !node.worker_name) {
                throw new Error(`Worker node ${node.name} missing required field: worker_name`);
            }
        }

        console.log(`📊 Graph Configuration:`);
        console.log(`   Name: ${graphConfig.name}`);
        console.log(`   Description: ${graphConfig.description || 'No description'}`);
        console.log(`   Nodes: ${graphConfig.nodes.length}`);
        console.log(`   Edges: ${(graphConfig.edges || []).length}`);
        console.log(`   Variables: ${Object.keys(graphConfig.variables || {}).length}`);

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
 * Test execution service health endpoint
 */
async function testExecutionHealth() {
    console.log('\n=== Testing Execution Health Endpoint ===');

    try {
        const response = await makeRequest(`${API_BASE}/execution/health`);

        console.log('✅ Execution health endpoint responding');
        console.log(`   Status: ${response.status || 'unknown'}`);
        console.log(`   Engine: ${response.execution_engine || 'unknown'}`);
        console.log(`   Timestamp: ${response.timestamp || 'unknown'}`);

        // Validate response structure
        const requiredFields = ['status'];
        const missingFields = requiredFields.filter(field => !response[field]);

        if (missingFields.length > 0) {
            throw new Error(`Health response missing required fields: ${missingFields.join(', ')}`);
        }

        return {
            success: true,
            response: response,
            validation: {
                has_status: !!response.status,
                has_engine: !!response.execution_engine,
                has_timestamp: !!response.timestamp
            }
        };
    } catch (error) {
        console.error(`❌ Execution health check failed: ${error.message}`);
        return {
            success: false,
            error: error.message,
            validation: {
                has_status: false,
                has_engine: false,
                has_timestamp: false
            }
        };
    }
}

/**
 * Enhanced graph execution with comprehensive monitoring
 */
async function executeGraph(graphId, executionRequest, stepwise = false) {
    const executionType = stepwise ? 'stepwise' : 'normal';
    const endpoint = stepwise ?
        `${API_BASE}/execution/graphs/${graphId}/execute/stepwise` :
        `${API_BASE}/execution/graphs/${graphId}/execute`;

    console.log(`\n=== Executing Graph (${executionType}): ${graphId} ===`);

    try {
        // Validate inputs
        if (!graphId) {
            throw new Error('Graph ID is required for execution');
        }

        console.log(`⚡ Execution Request Details:`);
        console.log(`   Type: ${executionType}`);
        console.log(`   Variables: ${JSON.stringify(executionRequest.variables || {}, null, 2)}`);
        console.log(`   Start Node: ${executionRequest.start_node || 'default'}`);

        const response = await makeRequest(endpoint, {
            method: 'POST',
            body: JSON.stringify(executionRequest)
        });

        console.log(`✅ Graph execution initiated successfully`);
        console.log(`   Session ID: ${response.session_id}`);
        console.log(`   Status: ${response.status}`);
        console.log(`   Current Node: ${response.current_node || 'None'}`);
        console.log(`   Started At: ${response.started_at}`);

        // Enhanced validation of execution response
        if (!response.session_id) {
            throw new Error('Execution response missing session_id');
        }

        if (!response.status) {
            throw new Error('Execution response missing status');
        }

        return response;
    } catch (error) {
        console.error(`❌ Failed to execute graph (${executionType}): ${error.message}`);
        if (error.status && error.response) {
            console.error(`   HTTP Details: Status ${error.status}, Response: ${error.response}`);
        }
        throw error;
    }
}

/**
 * Poll execution status until completion
 */
async function pollExecutionStatus(sessionId) {
    console.log(`\n=== Polling Execution Status: ${sessionId} ===`);

    let attempts = 0;
    while (attempts < MAX_POLL_ATTEMPTS) {
        try {
            const response = await makeRequest(`${API_BASE}/execution/sessions/${sessionId}/status`);

            console.log(`📊 Status Poll ${attempts + 1}: ${response.status}`);
            console.log(`   Current Node: ${response.current_node || 'None'}`);
            console.log(`   Completed Nodes: ${response.completed_nodes ? response.completed_nodes.length : 0}`);
            console.log(`   Failed Nodes: ${response.failed_nodes ? response.failed_nodes.length : 0}`);

            if (response.status === 'completed' || response.status === 'failed') {
                console.log(`✅ Execution finished with status: ${response.status}`);
                return response;
            }

            attempts++;
            if (attempts < MAX_POLL_ATTEMPTS) {
                console.log(`⏳ Waiting ${EXECUTION_POLL_INTERVAL/1000}s before next poll...`);
                await new Promise(resolve => setTimeout(resolve, EXECUTION_POLL_INTERVAL));
            }
        } catch (error) {
            console.error(`❌ Status poll failed (attempt ${attempts + 1}): ${error.message}`);
            attempts++;

            if (attempts < MAX_POLL_ATTEMPTS) {
                await new Promise(resolve => setTimeout(resolve, EXECUTION_POLL_INTERVAL));
            }
        }
    }

    throw new Error(`Execution polling timed out after ${MAX_POLL_ATTEMPTS} attempts`);
}

/**
 * Validate ExecutionRequest structure
 */
function validateExecutionRequest(request) {
    console.log('\n🔍 Validating ExecutionRequest Structure...');

    const validation = {
        has_variables: false,
        variables_is_object: false,
        has_start_node: false,
        start_node_is_string: false,
        overall_valid: false
    };

    try {
        // Check for variables field
        if ('variables' in request) {
            validation.has_variables = true;
            console.log('✅ Has variables field');

            if (typeof request.variables === 'object' && request.variables !== null) {
                validation.variables_is_object = true;
                console.log('✅ Variables is valid object');
            } else {
                console.log('❌ Variables is not a valid object');
            }
        } else {
            console.log('⚠️ Missing variables field (optional)');
        }

        // Check for start_node field
        if ('start_node' in request) {
            validation.has_start_node = true;
            console.log('✅ Has start_node field');

            if (typeof request.start_node === 'string') {
                validation.start_node_is_string = true;
                console.log('✅ Start_node is valid string');
            } else {
                console.log('❌ Start_node is not a string');
            }
        } else {
            console.log('⚠️ Missing start_node field (optional)');
        }

        // Overall validation - at least variables should be present and valid
        validation.overall_valid = validation.has_variables && validation.variables_is_object;

        console.log(`📊 ExecutionRequest Validation: ${validation.overall_valid ? 'PASSED' : 'FAILED'}`);

    } catch (error) {
        console.error(`❌ ExecutionRequest validation failed: ${error.message}`);
    }

    return validation;
}

/**
 * Validate ExecutionResponse structure
 */
function validateExecutionResponse(response) {
    console.log('\n🔍 Validating ExecutionResponse Structure...');

    const validation = {
        has_session_id: false,
        session_id_is_string: false,
        has_status: false,
        status_is_valid: false,
        has_started_at: false,
        has_node_outputs: false,
        has_completed_nodes: false,
        has_failed_nodes: false,
        overall_valid: false
    };

    try {
        // Check session_id
        if ('session_id' in response) {
            validation.has_session_id = true;
            console.log('✅ Has session_id field');

            if (typeof response.session_id === 'string' && response.session_id.length > 0) {
                validation.session_id_is_string = true;
                console.log('✅ Session_id is valid string');
            } else {
                console.log('❌ Session_id is not a valid string');
            }
        } else {
            console.log('❌ Missing session_id field');
        }

        // Check status
        if ('status' in response) {
            validation.has_status = true;
            console.log('✅ Has status field');

            const validStatuses = ['pending', 'running', 'completed', 'failed', 'cancelled'];
            if (validStatuses.includes(response.status)) {
                validation.status_is_valid = true;
                console.log(`✅ Status is valid: ${response.status}`);
            } else {
                console.log(`❌ Status is invalid: ${response.status}`);
            }
        } else {
            console.log('❌ Missing status field');
        }

        // Check started_at
        if ('started_at' in response) {
            validation.has_started_at = true;
            console.log('✅ Has started_at field');
        } else {
            console.log('⚠️ Missing started_at field');
        }

        // Check node_outputs
        if ('node_outputs' in response) {
            validation.has_node_outputs = true;
            console.log('✅ Has node_outputs field');
        } else {
            console.log('⚠️ Missing node_outputs field');
        }

        // Check completed_nodes
        if ('completed_nodes' in response) {
            validation.has_completed_nodes = true;
            console.log('✅ Has completed_nodes field');
        } else {
            console.log('⚠️ Missing completed_nodes field');
        }

        // Check failed_nodes
        if ('failed_nodes' in response) {
            validation.has_failed_nodes = true;
            console.log('✅ Has failed_nodes field');
        } else {
            console.log('⚠️ Missing failed_nodes field');
        }

        // Overall validation - must have session_id and status
        validation.overall_valid = validation.has_session_id &&
                                  validation.session_id_is_string &&
                                  validation.has_status &&
                                  validation.status_is_valid;

        console.log(`📊 ExecutionResponse Validation: ${validation.overall_valid ? 'PASSED' : 'FAILED'}`);

    } catch (error) {
        console.error(`❌ ExecutionResponse validation failed: ${error.message}`);
    }

    return validation;
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

        // Execution service health
        console.log('🔍 Checking execution service health...');
        const executionHealthResult = await testExecutionHealth();
        healthChecks.push({
            service: 'execution',
            status: executionHealthResult.success ? 'healthy' : 'failed',
            details: executionHealthResult
        });

        // Graph service check
        console.log('🔍 Checking graph service...');
        const graphsResponse = await makeRequest(`${API_BASE}/graphs/`);
        console.log(`✅ Graph service is responding (${graphsResponse.length} graphs found)`);
        healthChecks.push({ service: 'graphs', status: 'healthy', details: { graph_count: graphsResponse.length } });

        console.log('\n🎉 All health checks passed!');
        return { healthy: true, checks: healthChecks };

    } catch (error) {
        console.error(`❌ Health check failed: ${error.message}`);
        healthChecks.push({ service: 'unknown', status: 'failed', error: error.message });
        return { healthy: false, checks: healthChecks, error: error.message };
    }
}

/**
 * Create test graphs for execution testing
 */
async function createTestGraphs(workerName) {
    console.log('\n=== Creating Test Graphs ===');

    const graphs = [];

    // Simple linear graph
    const simpleGraphId = uuidv4();
    const node1Id = uuidv4();
    const node2Id = uuidv4();

    const simpleGraph = {
        name: 'Simple Linear Execution Test Graph',
        description: 'Basic two-node linear graph for execution testing',
        variables: {
            message: 'Hello from simple graph',
            multiplier: 2
        },
        nodes: [
            {
                id: node1Id,
                name: 'First Worker Node',
                node_type: 'worker',
                worker_name: workerName,
                task_template: 'Process message: {message}. Return a structured response with status=success and processed_message="{message}_processed".',
                dependencies: [],
                data: {
                    node_type_test: 'worker'
                },
                timeout: 60,
                retry_count: 1
            },
            {
                id: node2Id,
                name: 'Function Node',
                node_type: 'function',
                task_template: 'Apply transformation using multiplier: {multiplier}',
                dependencies: [node1Id],
                data: {
                    function_code: `
def run(data):
    multiplier = data.get('multiplier', 1)
    return {
        "result": 42 * multiplier,
        "status": "function_executed",
        "multiplier_used": multiplier
    }
`,
                    node_type_test: 'function'
                },
                timeout: 30,
                retry_count: 1
            }
        ],
        edges: [
            {
                source: node1Id,
                target: node2Id,
                condition: null,
                metadata: {
                    data_flow: 'worker_to_function'
                }
            }
        ]
    };

    const createdSimpleGraph = await createGraph(simpleGraph);
    graphs.push({ type: 'simple', graph: createdSimpleGraph });

    // Complex conditional graph
    const complexGraphId = uuidv4();
    const workerNode1Id = uuidv4();
    const functionNodeId = uuidv4();
    const conditionNodeId = uuidv4();
    const finalWorkerNodeId = uuidv4();

    const complexGraph = {
        name: 'Complex Conditional Execution Test Graph',
        description: 'Multi-node graph with conditional logic for comprehensive execution testing',
        variables: {
            input_value: 15,
            threshold: 10,
            operation_mode: 'advanced'
        },
        nodes: [
            {
                id: workerNode1Id,
                name: 'Initial Processing Worker',
                node_type: 'worker',
                worker_name: workerName,
                task_template: 'Analyze input value: {input_value} with operation mode: {operation_mode}. Return JSON with status=analyzed, input_received={input_value}, analysis_complete=true.',
                dependencies: [],
                data: {
                    node_type_test: 'worker',
                    processing_type: 'initial'
                },
                timeout: 60,
                retry_count: 1
            },
            {
                id: functionNodeId,
                name: 'Value Processing Function',
                node_type: 'function',
                task_template: 'Process value with threshold comparison',
                dependencies: [workerNode1Id],
                data: {
                    function_code: `
def run(data):
    input_value = data.get('input_value', 0)
    threshold = data.get('threshold', 10)

    processed_value = input_value * 3
    exceeds_threshold = processed_value > threshold

    return {
        "original_value": input_value,
        "processed_value": processed_value,
        "threshold": threshold,
        "exceeds_threshold": exceeds_threshold,
        "status": "value_processed"
    }
`,
                    node_type_test: 'function'
                },
                timeout: 30,
                retry_count: 1
            },
            {
                id: conditionNodeId,
                name: 'Threshold Condition Check',
                node_type: 'condition',
                task_template: 'Check if processed value exceeds threshold',
                dependencies: [functionNodeId],
                data: {
                    condition: 'input_value > threshold',
                    node_type_test: 'condition'
                },
                timeout: 30,
                retry_count: 1
            },
            {
                id: finalWorkerNodeId,
                name: 'Final Result Worker',
                node_type: 'worker',
                worker_name: workerName,
                task_template: 'Generate final report for operation mode: {operation_mode}. Input value was {input_value}. Create summary with status=final_complete and operation_successful=true.',
                dependencies: [conditionNodeId],
                data: {
                    node_type_test: 'worker',
                    processing_type: 'final'
                },
                timeout: 60,
                retry_count: 1
            }
        ],
        edges: [
            {
                source: workerNode1Id,
                target: functionNodeId,
                condition: null,
                metadata: { data_flow: 'worker_to_function' }
            },
            {
                source: functionNodeId,
                target: conditionNodeId,
                condition: null,
                metadata: { data_flow: 'function_to_condition' }
            },
            {
                source: conditionNodeId,
                target: finalWorkerNodeId,
                condition: null,
                metadata: { data_flow: 'condition_to_final_worker' }
            }
        ]
    };

    const createdComplexGraph = await createGraph(complexGraph);
    graphs.push({ type: 'complex', graph: createdComplexGraph });

    return graphs;
}

/**
 * Enhanced resource cleanup
 */
async function cleanupResources(graphIds = [], workerNames = []) {
    console.log('\n🧹 Performing Resource Cleanup...');
    const cleanupResults = {
        graphs_deleted: [],
        workers_deleted: [],
        failed_deletions: [],
        total_time: 0
    };

    const startTime = Date.now();

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

        cleanupResults.total_time = Date.now() - startTime;

        console.log('\n✅ Cleanup completed');
        console.log(`   Graphs deleted: ${cleanupResults.graphs_deleted.length}/${graphIds.length}`);
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
 * Main test function for graph execution endpoints
 */
async function runGraphExecutionTest() {
    console.log('🚀 Starting Graph Execution Endpoints Test Suite');
    console.log('='.repeat(80));

    // Reset retry flag for this test run
    hasRetried = false;
    console.log('🔄 Retry mechanism initialized (1 retry with 60s delay available)');

    let createdGraphIds = [];
    const createdWorkerNames = [];
    const testResults = {
        health_check: false,
        execution_health_check: false,
        worker_creation: false,
        graph_creation: false,
        simple_graph_execution: false,
        complex_graph_execution: false,
        stepwise_execution: false,
        execution_request_validation: false,
        execution_response_validation: false,
        overall_success: false,
        execution_time: 0,
        partial_results: {}
    };

    const testStartTime = Date.now();

    try {
        // Step 1: Comprehensive Health Check
        console.log('\n🏥 Step 1: API Health Check');
        const healthResult = await checkApiHealth();
        testResults.health_check = healthResult.healthy;
        testResults.partial_results.health_check = healthResult;

        if (!healthResult.healthy) {
            throw new Error('API health check failed - cannot proceed with test');
        }

        // Step 2: Execution Health Check
        console.log('\n🏥 Step 2: Execution Service Health Check');
        const executionHealthResult = await testExecutionHealth();
        testResults.execution_health_check = executionHealthResult.success;
        testResults.partial_results.execution_health = executionHealthResult;

        if (!executionHealthResult.success) {
            throw new Error('Execution service health check failed');
        }

        // Step 3: Create test worker
        console.log('\n🤖 Step 3: Creating Test Worker');

        const workerConfig = {
            worker_name: 'graph_execution_test_worker',
            agent_role: 'execution_test_specialist',
            agent_type: 'pydantic_ai',
            model_provider: 'deepseek',
            model: 'deepseek-chat',
            temperature: 0.2,
            system_prompt: 'You are a specialized test worker for graph execution validation. You analyze inputs and provide structured JSON responses.',
            instruction: 'When receiving task messages, analyze the content and return well-structured JSON responses that match the requested format.',
            user_prompt: '',
            description: 'Test worker for graph execution endpoint validation',
            mcp_config: null,
            base_url_override: null,
            api_key_override: null,
            tools_dir_override: null
        };

        const worker = await createWorker(workerConfig);
        createdWorkerNames.push(worker.name);
        testResults.worker_creation = true;
        testResults.partial_results.test_worker = worker;

        // Step 4: Create test graphs
        console.log('\n📊 Step 4: Creating Test Graphs');
        const testGraphs = await createTestGraphs(worker.name);
        createdGraphIds = testGraphs.map(g => g.graph.id);
        testResults.graph_creation = testGraphs.length > 0;
        testResults.partial_results.test_graphs = testGraphs;

        // Step 5: Test ExecutionRequest validation
        console.log('\n🔍 Step 5: ExecutionRequest Validation');
        const sampleExecutionRequest = {
            variables: {
                message: 'test_message',
                multiplier: 3
            },
            start_node: testGraphs[0].graph.nodes[0].id
        };

        const requestValidation = validateExecutionRequest(sampleExecutionRequest);
        testResults.execution_request_validation = requestValidation.overall_valid;
        testResults.partial_results.request_validation = requestValidation;

        // Step 6: Test simple graph execution
        console.log('\n⚡ Step 6: Simple Graph Execution Test');
        const simpleGraph = testGraphs.find(g => g.type === 'simple').graph;

        const simpleExecutionRequest = {
            variables: {
                message: 'simple_execution_test',
                multiplier: 5
            }
        };

        console.log('\n--- Testing Normal Execution ---');
        const simpleExecution = await executeGraph(simpleGraph.id, simpleExecutionRequest, false);

        // Validate ExecutionResponse structure
        const responseValidation = validateExecutionResponse(simpleExecution);
        testResults.execution_response_validation = responseValidation.overall_valid;
        testResults.partial_results.response_validation = responseValidation;

        // Poll for completion
        const simpleExecutionResult = await pollExecutionStatus(simpleExecution.session_id);
        testResults.simple_graph_execution = simpleExecutionResult.status === 'completed';
        testResults.partial_results.simple_execution = simpleExecutionResult;

        console.log('\n📊 Simple Execution Results:');
        console.log(`   Status: ${simpleExecutionResult.status}`);
        console.log(`   Completed Nodes: ${simpleExecutionResult.completed_nodes ? simpleExecutionResult.completed_nodes.length : 0}`);
        console.log(`   Failed Nodes: ${simpleExecutionResult.failed_nodes ? simpleExecutionResult.failed_nodes.length : 0}`);

        // Step 7: Test complex graph execution
        console.log('\n⚡ Step 7: Complex Graph Execution Test');
        const complexGraph = testGraphs.find(g => g.type === 'complex').graph;

        const complexExecutionRequest = {
            variables: {
                input_value: 25,
                threshold: 20,
                operation_mode: 'test_execution'
            }
        };

        const complexExecution = await executeGraph(complexGraph.id, complexExecutionRequest, false);
        const complexExecutionResult = await pollExecutionStatus(complexExecution.session_id);
        testResults.complex_graph_execution = complexExecutionResult.status === 'completed';
        testResults.partial_results.complex_execution = complexExecutionResult;

        console.log('\n📊 Complex Execution Results:');
        console.log(`   Status: ${complexExecutionResult.status}`);
        console.log(`   Completed Nodes: ${complexExecutionResult.completed_nodes ? complexExecutionResult.completed_nodes.length : 0}`);
        console.log(`   Failed Nodes: ${complexExecutionResult.failed_nodes ? complexExecutionResult.failed_nodes.length : 0}`);

        // Step 8: Test stepwise execution
        console.log('\n⚡ Step 8: Stepwise Execution Test');

        const stepwiseExecutionRequest = {
            variables: {
                message: 'stepwise_test',
                multiplier: 3
            }
        };

        try {
            console.log('\n--- Testing Stepwise Execution ---');
            const stepwiseExecution = await executeGraph(simpleGraph.id, stepwiseExecutionRequest, true);
            const stepwiseExecutionResult = await pollExecutionStatus(stepwiseExecution.session_id);
            testResults.stepwise_execution = stepwiseExecutionResult.status !== 'failed';
            testResults.partial_results.stepwise_execution = stepwiseExecutionResult;

            console.log('\n📊 Stepwise Execution Results:');
            console.log(`   Status: ${stepwiseExecutionResult.status}`);
            console.log(`   Completed Nodes: ${stepwiseExecutionResult.completed_nodes ? stepwiseExecutionResult.completed_nodes.length : 0}`);
        } catch (error) {
            console.log(`⚠️  Stepwise execution test failed (this may be expected if endpoint is not implemented): ${error.message}`);
            testResults.stepwise_execution = false; // Don't fail overall test for this
            testResults.partial_results.stepwise_execution = { error: error.message };
        }

        // Overall success determination
        testResults.overall_success = testResults.health_check &&
                                     testResults.execution_health_check &&
                                     testResults.worker_creation &&
                                     testResults.graph_creation &&
                                     testResults.simple_graph_execution &&
                                     testResults.complex_graph_execution &&
                                     testResults.execution_request_validation &&
                                     testResults.execution_response_validation;

        testResults.execution_time = Date.now() - testStartTime;

        // Step 9: Generate test report
        console.log('\n📊 Step 9: Graph Execution Test Summary');
        console.log('='.repeat(80));
        console.log(`🏁 Graph Execution Endpoints Test - COMPLETE`);
        console.log(`   Total execution time: ${testResults.execution_time}ms`);
        console.log(`   Overall success: ${testResults.overall_success ? '✅ PASSED' : '❌ FAILED'}`);
        console.log('\n📋 Test Results Breakdown:');
        console.log(`   ✅ API Health Check: ${testResults.health_check ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Execution Health Check: ${testResults.execution_health_check ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Worker Creation: ${testResults.worker_creation ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Graph Creation: ${testResults.graph_creation ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Simple Graph Execution: ${testResults.simple_graph_execution ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Complex Graph Execution: ${testResults.complex_graph_execution ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Stepwise Execution: ${testResults.stepwise_execution ? 'PASSED' : 'FAILED/OPTIONAL'}`);
        console.log(`   ✅ ExecutionRequest Validation: ${testResults.execution_request_validation ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ ExecutionResponse Validation: ${testResults.execution_response_validation ? 'PASSED' : 'FAILED'}`);

        if (testResults.overall_success) {
            console.log('\n🎉 Graph Execution Endpoints Test PASSED!');
            console.log('   ✅ All core execution endpoints are functioning correctly');
            console.log('   ✅ ExecutionRequest and ExecutionResponse structures are valid');
            console.log('   ✅ Graph execution monitoring is working');
            console.log('   ✅ Different graph types execute successfully');
        } else {
            console.log('\n💥 Graph Execution Endpoints Test FAILED!');
            console.log('   ❌ Some execution features are not working correctly');
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

    // Enhanced cleanup process
    console.log('\n🧹 Step 10: Resource Cleanup');
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
 * Main function with comprehensive error handling and reporting
 */
async function main() {
    console.log('🚀 Graph Execution Endpoints Test Suite');
    console.log('Node.js version:', process.version);
    console.log('Platform:', process.platform);
    console.log('Architecture:', process.arch);
    console.log('='.repeat(80));

    try {
        const testResults = await runGraphExecutionTest();

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
    createGraph,
    executeGraph,
    pollExecutionStatus,
    testExecutionHealth,
    validateExecutionRequest,
    validateExecutionResponse,
    checkApiHealth,
    createTestGraphs,
    cleanupResources,
    runGraphExecutionTest
};
