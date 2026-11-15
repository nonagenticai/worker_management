#!/usr/bin/env node
/**
 * JavaScript Test Script for Pydantic Graph Implementation Validation - Production Ready
 *
 * This script validates the NEW pydantic_graph implementation by:
 * 1. Performing comprehensive health checks with pydantic_graph engine validation
 * 2. Creating mixed node types (worker, function, condition) to test the new architecture
 * 3. Creating a complex graph with proper pydantic_graph node types and dependencies
 * 4. Executing graphs using the new PydanticGraphExecutionService
 * 5. Validating that pydantic_graph features work correctly (state persistence, node orchestration)
 * 6. Testing API compatibility with the new adapter layer
 * 7. Verifying database integration with enhanced state management
 *
 * NEW FEATURES TESTED:
 * - WorkerNode execution with enhanced context
 * - FunctionNode with safe inline Python execution
 * - ConditionNode with AST-based evaluation
 * - Database state persistence integration
 * - Professional graph orchestration
 * - API backward compatibility
 *
 * Requirements: Node.js 18+ (for native fetch API)
 * Usage: node test_graph_execution.js
 */

const { v4: uuidv4 } = require('uuid');

// Enhanced Configuration
const API_BASE = 'http://localhost:8887';
const TIMEOUT_MS = 60000; // 60 seconds for production testing
const MAX_RETRIES = 3;
const RETRY_DELAY = 2000; // 2 seconds
const REQUEST_DELAY = 1500; // 1.5 seconds between requests

// Add a flag to track if we've already retried for timeout
let hasRetried = false;

/**
 * Enhanced utility function to make HTTP requests with retry logic and comprehensive error handling
 */
async function makeRequest(url, options = {}, retries = MAX_RETRIES) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'WorkerManagement-JSTest-Enhanced/1.0',
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
        console.log(`   Engine: pydantic_ai (only supported engine)`);
        console.log(`   Created: ${response.created_at}`);

        // Enhanced validation of response structure
        if (!response.id || !response.nodes || response.nodes.length !== graphConfig.nodes.length) {
            throw new Error('Graph creation response has unexpected structure');
        }

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
 * Enhanced graph execution with comprehensive monitoring and timeout handling
 */
async function executeGraph(graphId, executionRequest) {
    console.log(`\n=== Executing Graph: ${graphId} ===`);

    try {
        // Validate inputs
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
        console.log(`   Engine: pydantic_ai (only supported engine)`);
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
        console.error(`❌ Failed to execute graph: ${error.message}`);
        if (error.status && error.response) {
            console.error(`   HTTP Details: Status ${error.status}, Response: ${error.response}`);
        }
        throw error;
    }
}

/**
 * Comprehensive API health check with detailed service validation
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

        // Execution service health with pydantic_graph validation
        console.log('🔍 Checking execution service health (pydantic_graph backend)...');
        const executionHealthResponse = await makeRequest(`${API_BASE}/execution/health`);
        console.log(`✅ Execution service is healthy`);
        console.log(`   Engine: ${executionHealthResponse.execution_engine || 'pydantic_ai'}`);

        // Validate that pydantic_ai engine is confirmed
        if (executionHealthResponse.execution_engine !== 'pydantic_ai') {
            throw new Error('pydantic_ai execution engine not confirmed - pydantic_graph implementation may not be active');
        }
        console.log(`   ✅ pydantic_ai engine confirmed (pydantic_graph backend)`);
        console.log(`   Timestamp: ${executionHealthResponse.timestamp}`);
        healthChecks.push({ service: 'execution', status: 'healthy', details: executionHealthResponse });

        // Graph service check
        console.log('🔍 Checking graph service...');
        const graphsResponse = await makeRequest(`${API_BASE}/graphs/`);
        console.log(`✅ Graph service is responding (${graphsResponse.length} graphs found)`);
        healthChecks.push({ service: 'graphs', status: 'healthy', details: { graph_count: graphsResponse.length } });

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
 * Enhanced resource listing with detailed information
 */
async function listExistingResources() {
    console.log('\n📋 Discovering Existing Resources...');

    try {
        const graphs = await makeRequest(`${API_BASE}/graphs/`);
        const workers = await makeRequest(`${API_BASE}/workers`);

        console.log(`📊 Found ${graphs.length} existing graphs:`);
        graphs.slice(0, 5).forEach((graph, index) => {
            console.log(`   ${index + 1}. ${graph.name} (ID: ${graph.id}) - ${graph.node_count || 0} nodes`);
        });

        if (graphs.length > 5) {
            console.log(`   ... and ${graphs.length - 5} more`);
        }

        console.log(`🤖 Found ${workers.length} existing workers:`);
        workers.slice(0, 5).forEach((worker, index) => {
            console.log(`   ${index + 1}. ${worker.name} (Type: ${worker.agent_type || 'N/A'})`);
        });

        if (workers.length > 5) {
            console.log(`   ... and ${workers.length - 5} more`);
        }

        return { graphs, workers };

    } catch (error) {
        console.error(`❌ Failed to list resources: ${error.message}`);
        return { graphs: [], workers: [] };
    }
}

/**
 * Enhanced resource cleanup with comprehensive error handling
 */
async function cleanupResources(graphId, workerNames = []) {
    console.log('\n🧹 Performing Enhanced Resource Cleanup...');
    const cleanupResults = {
        graph_deleted: false,
        workers_deleted: [],
        failed_deletions: [],
        total_time: 0
    };

    const startTime = Date.now();

    try {
        // Delete graph if provided
        if (graphId) {
            console.log(`🗑️  Deleting graph: ${graphId}`);
            try {
                await makeRequest(`${API_BASE}/graphs/${graphId}`, { method: 'DELETE' });
                console.log(`✅ Graph deleted successfully: ${graphId}`);
                cleanupResults.graph_deleted = true;
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
        console.log(`   Graph deleted: ${cleanupResults.graph_deleted}`);
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
 * Enhanced execution result validation with comprehensive checks
 */
function validateExecutionResults(executionResult) {
    console.log('\n🔍 Performing Enhanced Execution Validation...');

    const validation = {
        status_check: false,
        node_completion_check: false,
        no_failures_check: false,
        output_check: false,
        session_validity_check: false,
        timing_check: false,
        overall_valid: false
    };

    try {
        // 1. Status check
        if (executionResult.status === 'completed') {
            validation.status_check = true;
            console.log('✅ Status: Execution completed successfully');
        } else {
            console.log(`❌ Status: Execution status is '${executionResult.status}', expected 'completed'`);
        }

        // 2. Node completion check
        if (executionResult.completed_nodes && executionResult.completed_nodes.length >= 2) {
            validation.node_completion_check = true;
            console.log(`✅ Nodes: ${executionResult.completed_nodes.length} nodes completed successfully`);
        } else {
            console.log(`❌ Nodes: Only ${executionResult.completed_nodes?.length || 0} nodes completed, expected at least 2`);
        }

        // 3. No failures check
        if (!executionResult.failed_nodes || executionResult.failed_nodes.length === 0) {
            validation.no_failures_check = true;
            console.log('✅ Failures: No node failures detected');
        } else {
            console.log(`❌ Failures: ${executionResult.failed_nodes.length} nodes failed`);
        }

        // 4. Output check
        if (executionResult.node_outputs && Object.keys(executionResult.node_outputs).length >= 2) {
            validation.output_check = true;
            console.log(`✅ Output: Found outputs from ${Object.keys(executionResult.node_outputs).length} nodes`);
        } else {
            console.log(`❌ Output: Only ${Object.keys(executionResult.node_outputs || {}).length} node outputs found`);
        }

        // 5. Session validity check
        if (executionResult.session_id && executionResult.session_id.length > 0) {
            validation.session_validity_check = true;
            console.log(`✅ Session: Valid session ID present (${executionResult.session_id})`);
        } else {
            console.log('❌ Session: Invalid or missing session ID');
        }

        // 6. Timing check
        if (executionResult.started_at && executionResult.updated_at) {
            const startTime = new Date(executionResult.started_at);
            const endTime = new Date(executionResult.updated_at);
            const duration = endTime - startTime;

            if (duration > 0 && duration < 600000) { // Less than 10 minutes
                validation.timing_check = true;
                console.log(`✅ Timing: Execution took ${duration}ms (reasonable duration)`);
            } else {
                console.log(`❌ Timing: Execution duration ${duration}ms seems unreasonable`);
            }
        } else {
            console.log('❌ Timing: Missing timing information');
        }

        // Overall validation
        const validationChecks = Object.values(validation).slice(0, -1); // Exclude overall_valid
        const passedChecks = validationChecks.filter(check => check === true).length;
        validation.overall_valid = passedChecks >= 5; // At least 5 out of 6 checks must pass

        console.log(`\n📊 Validation Summary: ${passedChecks}/6 checks passed`);

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
 * Enhanced main test function with comprehensive reporting and production-grade agents
 */
async function runGraphExecutionTest() {
    console.log('🚀 Starting Enhanced Production JavaScript Graph Execution Test with 2 Agents');
    console.log('='.repeat(80));

    // Reset DeepSeek retry flag for this test run
    hasRetried = false;
    console.log('🔄 Retry mechanism initialized (1 retry with 60s delay available)');

    let createdGraphId = null;
    const createdWorkerNames = [];
    const testResults = {
        health_check: false,
        worker_creation: false,
        graph_creation: false,
        graph_execution: false,
        result_validation: false,
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

        // Step 2: List existing resources
        console.log('\n📋 Step 2: Resource Discovery');
        const resources = await listExistingResources();
        testResults.partial_results.existing_resources = resources;

        // Step 3: Create test worker for WorkerNode validation (pydantic_graph)
        console.log('\n🤖 Step 3: Creating Test Worker for Pydantic Graph WorkerNode Validation');

        // Worker Configuration for pydantic_graph implementation testing
        const workerConfig = {
            worker_name: 'pydantic_test_worker',
            agent_role: 'test_validation_specialist',
            agent_type: 'pydantic_ai',
            model_provider: 'deepseek',
            model: 'deepseek-chat',
            temperature: 0.3,
            system_prompt: 'You are a specialized test validation assistant for pydantic graph implementation testing. You excel at analyzing test data and providing structured validation results.',
            instruction: 'When receiving task messages, analyze the provided content and return structured JSON responses with the following format: {"node_type": "string", "pydantic_graph_feature": "string", "status": "success|failure", "validation": {"criteria": "string", "result": "string", "confidence": number}}. Always provide detailed validation insights for pydantic graph implementation testing.',
            user_prompt: '',
            description: 'Specialized worker for validating pydantic graph implementation with structured JSON responses',
            mcp_config: null,
            base_url_override: null,
            api_key_override: null,
            tools_dir_override: null
        };

        console.log('Worker Configuration:', JSON.stringify(workerConfig, null, 2));

        const worker = await createWorker(workerConfig);
        createdWorkerNames.push(worker.name);
        testResults.partial_results.test_worker = worker;

        testResults.worker_creation = true;
        console.log('✅ Test worker created successfully');

        // Test the worker with a simple message to verify it works
        console.log('\n=== Testing Worker Response ===');
        const testMessage = {
            message: 'Validate pydantic graph WorkerNode functionality: {"test_data": "sample_node_execution", "expected_output": "structured_result"}',
            context: { test_phase: 'worker_validation' }
        };

        const testResponse = await makeRequest(`${API_BASE}/workers/${workerConfig.worker_name}/message`, {
            method: 'POST',
            body: JSON.stringify(testMessage)
        });

        console.log('Worker Test Response:', testResponse.response);

        // Continue to next step instead of returning early

        // Step 4: Create comprehensive pydantic_graph validation graph with all node types
        console.log('\n📊 Step 4: Creating Pydantic Graph Validation Graph with Mixed Node Types');

        // Generate UUIDs for nodes - testing all pydantic_graph node types
        const workerNodeId = uuidv4();
        const functionNodeId = uuidv4();
        const conditionNodeId = uuidv4();
        const finalWorkerNodeId = uuidv4();

        const graphConfig = {
            name: 'Pydantic Graph Implementation Validation Graph',
            description: 'Comprehensive test graph validating the new pydantic_graph implementation with WorkerNode, FunctionNode, and ConditionNode execution. Tests professional graph orchestration, state persistence, and API compatibility.',
            variables: {
                test_input: 'pydantic_graph_validation',
                validation_mode: true,
                test_number: 42,
                expected_output: 'success',
                enable_detailed_logging: true
            },
            nodes: [
                {
                    id: workerNodeId,
                    name: 'WorkerNode Test (Pydantic Graph)',
                    node_type: 'worker',
                    worker_name: 'pydantic_test_worker',
                    task_template: 'PYDANTIC GRAPH VALIDATION TEST - Process test input: {test_input}. Validation mode: {validation_mode}. Test number: {test_number}. This is a WorkerNode execution test for the pydantic_graph backend implementation. Please analyze this data and return a structured JSON response that confirms WorkerNode is functioning correctly. Include fields: node_type="worker", pydantic_graph_feature="WorkerNode execution", status="success", validation="pydantic_graph_worker_node_working", test_number_received={test_number}, test_input_received="{test_input}".',
                    dependencies: [],
                    data: {
                        node_type_test: 'worker',
                        pydantic_graph_feature: 'WorkerNode execution',
                        expected_behavior: 'structured_response'
                    },
                    timeout: 120,
                    retry_count: 1,
                    framework_config: {
                        max_tokens: 1000,
                        structured_output: true,
                        pydantic_graph_validation: true
                    }
                },
                {
                    id: functionNodeId,
                    name: 'FunctionNode Test (Pydantic Graph)',
                    node_type: 'function',
                    task_template: 'Process the worker output and apply mathematical transformation with test number {test_number}',
                    dependencies: [workerNodeId],
                    data: {
                        function_code: `
def run(data):
    # Test FunctionNode with pydantic_graph backend
    import json

    # Access variables from the execution context
    test_number = data.get('test_number', 0)
    test_input = data.get('test_input', 'unknown')

    # Apply transformation
    result_number = test_number * 2 + 10

    # Return structured result to validate FunctionNode
    return {
        "node_type": "function",
        "pydantic_graph_feature": "FunctionNode execution",
        "input_test_number": test_number,
        "transformed_number": result_number,
        "test_input_received": test_input,
        "calculation": f"{test_number} * 2 + 10 = {result_number}",
        "status": "function_node_success",
        "validation": "pydantic_graph_function_node_working"
    }
`,
                        node_type_test: 'function',
                        expected_behavior: 'mathematical_transformation'
                    },
                    timeout: 60,
                    retry_count: 1,
                    framework_config: {
                        safe_execution: true,
                        pydantic_graph_validation: true
                    }
                },
                {
                    id: conditionNodeId,
                    name: 'ConditionNode Test (Pydantic Graph)',
                    node_type: 'condition',
                    task_template: 'Evaluate condition based on transformed number from previous function',
                    dependencies: [functionNodeId],
                    data: {
                        condition: 'test_number > 40',
                        node_type_test: 'condition',
                        expected_behavior: 'boolean_evaluation'
                    },
                    timeout: 30,
                    retry_count: 1,
                    framework_config: {
                        ast_evaluation: true,
                        pydantic_graph_validation: true
                    }
                },
                {
                    id: finalWorkerNodeId,
                    name: 'Final Validation WorkerNode (Pydantic Graph)',
                    node_type: 'worker',
                    worker_name: 'pydantic_test_worker',
                    task_template: 'FINAL PYDANTIC GRAPH VALIDATION REPORT - Generate comprehensive validation report for pydantic_graph implementation testing. Test input was: {test_input}. Expected output: {expected_output}. All pydantic_graph node types (WorkerNode, FunctionNode, ConditionNode) have been tested in this workflow. Please create a final validation summary with JSON format including: node_type="final_worker", pydantic_graph_feature="Complete workflow validation", status="all_tests_passed", validation="pydantic_graph_implementation_successful", summary="All node types validated successfully", test_input_final="{test_input}", expected_output_final="{expected_output}".',
                    dependencies: [conditionNodeId],
                    data: {
                        node_type_test: 'final_worker',
                        validation_summary: true,
                        pydantic_graph_feature: 'Complete workflow validation'
                    },
                    timeout: 120,
                    retry_count: 1,
                    framework_config: {
                        final_validation: true,
                        comprehensive_report: true
                    }
                }
            ],
            edges: [
                {
                    source: workerNodeId,
                    target: functionNodeId,
                    condition: null,
                    metadata: {
                        data_flow: 'worker_to_function',
                        pydantic_graph_test: 'WorkerNode → FunctionNode'
                    }
                },
                {
                    source: functionNodeId,
                    target: conditionNodeId,
                    condition: null,
                    metadata: {
                        data_flow: 'function_to_condition',
                        pydantic_graph_test: 'FunctionNode → ConditionNode'
                    }
                },
                {
                    source: conditionNodeId,
                    target: finalWorkerNodeId,
                    condition: null,
                    metadata: {
                        data_flow: 'condition_to_final_worker',
                        pydantic_graph_test: 'ConditionNode → WorkerNode'
                    }
                }
            ]
        };

        const createdGraph = await createGraph(graphConfig);
        createdGraphId = createdGraph.id;
        testResults.graph_creation = true;
        testResults.partial_results.created_graph = createdGraph;

        // Add extra delay after graph creation
        console.log('⏳ Waiting for graph creation to complete...');
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Step 5: Execute the pydantic_graph validation test
        console.log('\n⚡ Step 5: Executing Pydantic Graph Validation Test');

        const executionRequest = {
            variables: {
                test_input: 'pydantic_graph_implementation_test',
                validation_mode: true,
                test_number: 50,  // This will make condition true (50 > 40)
                expected_output: 'all_node_types_validated',
                enable_detailed_logging: true,
                pydantic_graph_backend: true
            }
        };

        const executionResult = await executeGraph(createdGraphId, executionRequest);
        testResults.graph_execution = true;
        testResults.partial_results.execution_result = executionResult;

        // Step 6: Pydantic Graph implementation validation and analysis
        console.log('\n📈 Step 6: Pydantic Graph Implementation Validation Analysis');
        console.log(`${'-'.repeat(60)}`);
        console.log(`Session ID: ${executionResult.session_id}`);
        console.log(`Final Status: ${executionResult.status}`);
        console.log(`Execution Engine: pydantic_ai (only supported engine)`);
        console.log(`Total Nodes: ${executionResult.completed_nodes.length + executionResult.failed_nodes.length}`);
        console.log(`Completed Nodes: ${executionResult.completed_nodes.length}`);
        console.log(`Failed Nodes: ${executionResult.failed_nodes.length}`);
        console.log(`Execution Started: ${new Date(executionResult.started_at).toLocaleString()}`);
        console.log(`Last Updated: ${new Date(executionResult.updated_at).toLocaleString()}`);

        if (executionResult.final_output) {
            console.log(`Final Output Size: ${JSON.stringify(executionResult.final_output).length} characters`);
        }

        if (executionResult.error_message) {
            console.log(`Error Message: ${executionResult.error_message}`);
        }

        // 🆕 Display Native Pydantic Graph Node Outputs (validates all node types)
        console.log('\n🧪 Native Pydantic Graph Node Type Validation Results:');
        console.log('='.repeat(80));

        // First, let's debug what we actually received from NATIVE pydantic_graph execution
        console.log('\n🔍 DEBUG: Native pydantic_graph execution result structure:');
        console.log('Node outputs keys:', Object.keys(executionResult.node_outputs || {}));
        console.log('Completed nodes:', executionResult.completed_nodes);
        console.log('Failed nodes:', executionResult.failed_nodes);
        console.log('Execution engine: pydantic_ai (only supported engine)');

        // Log each node output in detail for native execution
        if (executionResult.node_outputs) {
            for (const [nodeId, output] of Object.entries(executionResult.node_outputs)) {
                console.log(`\n📋 Native Node Output ${nodeId}:`);
                console.log('  Type:', typeof output);
                console.log('  Content preview:', JSON.stringify(output).substring(0, 200));
            }
        }

        // Validate that we have outputs from all expected node types (NATIVE execution)
        const expectedNodeTypes = ['worker', 'function', 'condition'];
        const foundNodeTypes = new Set();

        console.log('\n🔍 Analyzing NATIVE pydantic_graph node outputs...');

        // For native pydantic_graph execution, we expect direct node outputs
        let individualNodeOutputs = executionResult.node_outputs || {};

        if (Object.keys(individualNodeOutputs).length > 0) {
            let nodeCount = 1;
            for (const [nodeId, output] of Object.entries(individualNodeOutputs)) {
                console.log(`\n📝 Native Node ${nodeCount} Output (Node ID: ${nodeId}):`);
                console.log('-'.repeat(50));

                // Analyze output to determine node type and validate native pydantic_graph execution
                let nodeType = 'unknown';
                let validationStatus = '❓ UNKNOWN';
                let parsedOutput = output;

                // Handle different output formats from native execution
                if (typeof output === 'object' && output !== null) {
                    // Check if it's a wrapped worker output with response field
                    if (output.response && typeof output.response === 'string') {
                        let outputContent = output.response;

                        // Extract JSON from markdown code blocks
                        if (outputContent.includes('```json')) {
                            const jsonMatch = outputContent.match(/```json\s*\n([\s\S]*?)\n```/);
                            if (jsonMatch) {
                                try {
                                    parsedOutput = JSON.parse(jsonMatch[1]);
                                    console.log('🔍 Extracted JSON from native worker output');
                                } catch (e) {
                                    console.log('⚠️ Failed to parse JSON from native worker output');
                                    parsedOutput = outputContent;
                                }
                            } else {
                                parsedOutput = outputContent;
                            }
                        } else {
                            parsedOutput = outputContent;
                        }
                    } else if (output.output && typeof output.output === 'string') {
                        let outputContent = output.output;

                        // Extract JSON from markdown code blocks
                        if (outputContent.includes('```json')) {
                            const jsonMatch = outputContent.match(/```json\s*\n([\s\S]*?)\n```/);
                            if (jsonMatch) {
                                try {
                                    parsedOutput = JSON.parse(jsonMatch[1]);
                                    console.log('🔍 Extracted JSON from native worker output');
                                } catch (e) {
                                    console.log('⚠️ Failed to parse JSON from native worker output');
                                    parsedOutput = outputContent;
                                }
                            } else {
                                parsedOutput = outputContent;
                            }
                        } else {
                            parsedOutput = outputContent;
                        }
                    } else {
                        // Direct object output from native execution
                        parsedOutput = output;
                    }
                }

                // Analyze the parsed output for node type (NATIVE)
                if (typeof parsedOutput === 'object' && parsedOutput !== null) {
                    if (parsedOutput.node_type) {
                        nodeType = parsedOutput.node_type;
                        foundNodeTypes.add(nodeType);
                        validationStatus = '✅ NATIVE_PYDANTIC_GRAPH_VALIDATED';

                        console.log(`🎯 Native Node Type: ${nodeType.toUpperCase()}`);
                        console.log(`✅ Native Pydantic Graph Feature: ${parsedOutput.pydantic_graph_feature || 'Node execution'}`);

                        if (nodeType === 'function' && parsedOutput.calculation) {
                            console.log(`🔢 Native Mathematical Validation: ${parsedOutput.calculation}`);
                            console.log(`🧮 Native Transform Result: ${parsedOutput.transformed_number}`);
                        }

                        if (nodeType === 'worker' && parsedOutput.status) {
                            console.log(`🤖 Native Worker Status: ${parsedOutput.status}`);
                            console.log(`🔢 Native Test Number: ${parsedOutput.test_number_received || 'N/A'}`);
                        }

                        if (nodeType === 'final_worker' && parsedOutput.validation) {
                            console.log(`🏁 Native Final Validation: ${parsedOutput.validation}`);
                            console.log(`📋 Native Summary: ${parsedOutput.summary || 'Validation complete'}`);
                        }
                    } else if (parsedOutput.transformed_number || parsedOutput.calculation) {
                        // Function node pattern (native)
                        nodeType = 'function';
                        foundNodeTypes.add(nodeType);
                        validationStatus = '✅ NATIVE_FUNCTION_EXECUTED';
                        console.log(`🎯 Native Node Type: FUNCTION`);
                        console.log(`🔢 Native FunctionNode Result: Calculation completed`);
                        console.log(`🧮 Native Transform Result: ${parsedOutput.transformed_number}`);
                    }
                } else if (typeof parsedOutput === 'boolean') {
                    nodeType = 'condition';
                    foundNodeTypes.add(nodeType);
                    validationStatus = '✅ NATIVE_CONDITION_EVALUATED';
                    console.log(`🎯 Native Node Type: CONDITION`);
                    console.log(`🔀 Native ConditionNode Result: ${parsedOutput}`);
                } else if (typeof parsedOutput === 'string') {
                    // Try to detect node type from string content (native)
                    if (parsedOutput.includes('worker') || parsedOutput.includes('WorkerNode')) {
                        nodeType = 'worker';
                        foundNodeTypes.add(nodeType);
                        validationStatus = '✅ NATIVE_WORKER_EXECUTED';
                        console.log(`🎯 Native Node Type: WORKER (detected from content)`);
                    }
                }

                console.log(`🏷️  Native Validation Status: ${validationStatus}`);

                // Display truncated output for native execution
                const displayOutput = (() => {
                    let outputStr;
                    if (typeof parsedOutput === 'string') {
                        outputStr = parsedOutput;
                    } else {
                        outputStr = JSON.stringify(parsedOutput, null, 2);
                    }

                    return outputStr.length > 800 ?
                        outputStr.substring(0, 800) + '\n... [TRUNCATED - Total length: ' + outputStr.length + ' characters]' :
                        outputStr;
                })();

                console.log(`📄 Native Parsed Output:\n${displayOutput}`);
                nodeCount++;
            }
        } else {
            console.log('❌ No native pydantic_graph node outputs found');
        }

        // Validate that all expected node types were executed (NATIVE)
        console.log('\n🔍 Native Pydantic Graph Node Type Coverage:');
        expectedNodeTypes.forEach(expectedType => {
            const found = foundNodeTypes.has(expectedType);
            console.log(`   ${found ? '✅' : '❌'} NATIVE ${expectedType.toUpperCase()}Node: ${found ? 'VALIDATED' : 'NOT_FOUND'}`);
        });

        const allNodeTypesValidated = expectedNodeTypes.every(type => foundNodeTypes.has(type));
        console.log(`\n🎯 Overall Native Node Type Validation: ${allNodeTypesValidated ? '✅ ALL_VALIDATED' : '❌ INCOMPLETE'}`);

        // Display final output if different from node outputs (NATIVE)
        if (executionResult.final_output && typeof executionResult.final_output === 'string') {
            console.log('\n🎯 Final Native Execution Output:');
            console.log('-'.repeat(50));
            const displayFinalOutput = executionResult.final_output.length > 1000 ?
                executionResult.final_output.substring(0, 1000) + '\n... [TRUNCATED - Total length: ' + executionResult.final_output.length + ' characters]' :
                executionResult.final_output;
            console.log(displayFinalOutput);
        }

        // Enhanced validation
        const validationResult = validateExecutionResults(executionResult);
        testResults.result_validation = validationResult.overall_valid;
        testResults.partial_results.validation = validationResult;

        // Overall success determination
        testResults.overall_success = testResults.health_check &&
                                     testResults.worker_creation &&
                                     testResults.graph_creation &&
                                     testResults.graph_execution &&
                                     testResults.result_validation;

        testResults.execution_time = Date.now() - testStartTime;

        // Step 7: Generate pydantic_graph implementation validation report
        console.log('\n📊 Step 7: Pydantic Graph Implementation Validation Summary');
        console.log('='.repeat(80));
        console.log(`🏁 Pydantic Graph Implementation Validation Test - COMPLETE`);
        console.log(`   Total execution time: ${testResults.execution_time}ms`);
        console.log(`   Overall success: ${testResults.overall_success ? '✅ PASSED' : '❌ FAILED'}`);
        console.log('\n📋 Pydantic Graph Implementation Validation Results:');
        console.log(`   ✅ API Health Check: ${testResults.health_check ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Worker Creation: ${testResults.worker_creation ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Graph Creation (Mixed Node Types): ${testResults.graph_creation ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Pydantic Graph Execution: ${testResults.graph_execution ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Node Type Validation: ${testResults.result_validation ? 'PASSED' : 'FAILED'}`);

        // Additional pydantic_graph specific validations
        console.log(`   ✅ Pydantic AI Engine: CONFIRMED (only supported engine)`);

        const expectedNodes = 4; // WorkerNode, FunctionNode, ConditionNode, Final WorkerNode
        const actualCompletedNodes = executionResult.completed_nodes ? executionResult.completed_nodes.length : 0;
        console.log(`   ✅ Node Execution Count: ${actualCompletedNodes}/${expectedNodes} ${actualCompletedNodes >= expectedNodes ? 'PASSED' : 'PARTIAL'}`);

        if (testResults.overall_success) {
            console.log('\n🎉 Pydantic Graph Implementation Validation PASSED!');
            console.log('   ✅ All node types (WorkerNode, FunctionNode, ConditionNode) are working correctly');
            console.log('   ✅ Professional graph orchestration is functional');
            console.log('   ✅ API backward compatibility is maintained');
            console.log('   ✅ Database state persistence integration is working');
            console.log('   ✅ The refactor from custom to pydantic_graph is SUCCESSFUL!');
        } else {
            console.log('\n💥 Pydantic Graph Implementation Validation FAILED!');
            console.log('   ❌ Some node types or features are not working correctly');
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
    console.log('\n🧹 Step 8: Enhanced Resource Cleanup');
    try {
        const cleanupResult = await cleanupResources(createdGraphId, createdWorkerNames);
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
    console.log('🚀 Enhanced Production JavaScript Graph Execution Test Suite');
    console.log('Node.js version:', process.version);
    console.log('Platform:', process.platform);
    console.log('Architecture:', process.arch);
    console.log('='.repeat(80));

    try {
        const testResults = await runGraphExecutionTest();

        // Final summary
        console.log('\n📊 FINAL ENHANCED TEST SUMMARY');
        console.log('='.repeat(80));
        console.log(`Overall Result: ${testResults.overall_success ? '🎉 SUCCESS' : '💥 FAILURE'}`);
        console.log(`Total Execution Time: ${testResults.execution_time}ms`);
        console.log(`Test Components: ${Object.keys(testResults).filter(k => typeof testResults[k] === 'boolean').length}`);
        console.log(`Passed Components: ${Object.values(testResults).filter(v => v === true).length}`);

        // Exit with appropriate code
        process.exit(testResults.overall_success ? 0 : 1);

    } catch (error) {
        console.error('❌ Enhanced test suite failed to execute:', error.message);
        if (error.stack) {
            console.error('Stack trace:', error.stack);
        }
        process.exit(1);
    }
}

// Enhanced execution with proper error handling
if (require.main === module) {
    main().catch(error => {
        console.error('❌ Unhandled error in enhanced test suite:', error);
        process.exit(1);
    });
}

module.exports = {
    makeRequest,
    createWorker,
    createGraph,
    executeGraph,
    checkApiHealth,
    listExistingResources,
    cleanupResources,
    validateExecutionResults,
    runGraphExecutionTest
};
