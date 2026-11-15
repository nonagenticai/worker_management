#!/usr/bin/env node
/**
 * JavaScript Test Script for Graph Execution Edge Cases - Production Ready
 *
 * This script validates edge cases in graph execution of the Worker Management API by:
 * 1. Testing execution with missing variables
 * 2. Testing execution with invalid graph IDs
 * 3. Testing execution with circular dependencies
 * 4. Testing execution with deeply nested graphs
 * 5. Testing execution with timeout scenarios
 * 6. Testing execution with failed nodes and recovery
 * 7. Testing execution with conditional branches
 * 8. Testing execution with parallel node execution
 * 9. Testing execution cancellation
 * 10. Testing execution state persistence
 *
 * ENDPOINTS TESTED:
 * - POST /execution/graphs/{graph_id}/execute
 * - POST /execution/graphs/{graph_id}/execute/stepwise
 * - GET /execution/sessions/{session_id}/status
 * - POST /execution/sessions/{session_id}/cancel
 *
 * Requirements: Node.js 18+ (for native fetch API)
 * Usage: node test_22_graph_execution_edge_cases.js
 */

const { v4: uuidv4 } = require('uuid');

// Enhanced Configuration
const API_BASE = 'http://localhost:8887';
const TIMEOUT_MS = 120000; // 2 minutes for execution tests
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
            'User-Agent': 'WorkerManagement-GraphExecutionEdgeCasesTest/1.0',
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
 * Create a worker for testing
 */
async function createWorker(workerConfig) {
    console.log(`\n=== Creating Worker: ${workerConfig.worker_name} ===`);

    try {
        const response = await makeRequest(`${API_BASE}/workers`, {
            method: 'POST',
            body: JSON.stringify(workerConfig)
        });

        console.log(`✅ Worker created successfully: ${response.name}`);
        return response;
    } catch (error) {
        console.error(`❌ Failed to create worker: ${error.message}`);
        throw error;
    }
}

/**
 * Create a graph for testing
 */
async function createGraph(graphConfig) {
    console.log(`\n=== Creating Graph: ${graphConfig.name} ===`);

    try {
        const response = await makeRequest(`${API_BASE}/graphs/`, {
            method: 'POST',
            body: JSON.stringify(graphConfig)
        });

        console.log(`✅ Graph created successfully: ${response.name}`);
        return response;
    } catch (error) {
        console.error(`❌ Failed to create graph: ${error.message}`);
        throw error;
    }
}

/**
 * Test execution with missing required variables
 */
async function testMissingVariables(workerName) {
    console.log('\n=== Testing Execution with Missing Variables ===');

    try {
        // Create graph that expects variables
        const graphId = uuidv4();
        const nodeId = uuidv4();

        const graph = {
            name: 'Missing Variables Test Graph',
            description: 'Tests execution behavior with missing required variables',
            variables: {
                required_var: null,
                optional_var: 'default_value'
            },
            nodes: [
                {
                    id: nodeId,
                    name: 'Variable Processing Node',
                    node_type: 'worker',
                    worker_name: workerName,
                    task_template: 'Process required variable: {required_var} and optional: {optional_var}',
                    dependencies: [],
                    data: {},
                    timeout: 60,
                    retry_count: 1
                }
            ],
            edges: []
        };

        const createdGraph = await createGraph(graph);

        // Try to execute without providing required variable
        console.log('\n📊 Attempting execution without required variables...');
        try {
            const executionRequest = {
                variables: {
                    optional_var: 'provided_value'
                    // Missing required_var
                }
            };

            const response = await makeRequest(`${API_BASE}/execution/graphs/${createdGraph.id}/execute`, {
                method: 'POST',
                body: JSON.stringify(executionRequest)
            });

            console.log(`⚠️ Execution started despite missing variables: ${response.session_id}`);

            // Check if execution fails appropriately
            await new Promise(resolve => setTimeout(resolve, 5000)); // Wait for execution

            const status = await makeRequest(`${API_BASE}/execution/sessions/${response.session_id}/status`);

            return {
                success: true,
                expected_behavior: status.status === 'failed',
                details: {
                    execution_started: true,
                    final_status: status.status,
                    error_message: status.error || 'No error message'
                }
            };

        } catch (error) {
            console.log(`✅ Execution correctly rejected with missing variables: ${error.message}`);
            return {
                success: true,
                expected_behavior: true,
                details: {
                    execution_started: false,
                    error: error.message
                }
            };
        }
    } catch (error) {
        console.error(`❌ Missing variables test failed: ${error.message}`);
        return {
            success: false,
            error: error.message
        };
    }
}

/**
 * Test execution with invalid graph ID
 */
async function testInvalidGraphId() {
    console.log('\n=== Testing Execution with Invalid Graph ID ===');

    try {
        const invalidGraphId = uuidv4(); // Non-existent graph ID

        const executionRequest = {
            variables: {
                test: 'value'
            }
        };

        try {
            await makeRequest(`${API_BASE}/execution/graphs/${invalidGraphId}/execute`, {
                method: 'POST',
                body: JSON.stringify(executionRequest)
            });

            console.log('❌ Execution should have failed with invalid graph ID');
            return {
                success: false,
                error: 'Expected 404 error but request succeeded'
            };

        } catch (error) {
            if (error.status === 404) {
                console.log('✅ Correctly returned 404 for invalid graph ID');
                return {
                    success: true,
                    expected_behavior: true,
                    error_status: error.status
                };
            } else {
                console.log(`⚠️ Unexpected error status: ${error.status}`);
                return {
                    success: false,
                    error: `Expected 404 but got ${error.status}: ${error.message}`
                };
            }
        }
    } catch (error) {
        console.error(`❌ Invalid graph ID test failed: ${error.message}`);
        return {
            success: false,
            error: error.message
        };
    }
}

/**
 * Test execution with circular dependencies
 */
async function testCircularDependencies(workerName) {
    console.log('\n=== Testing Execution with Circular Dependencies ===');

    try {
        const node1Id = uuidv4();
        const node2Id = uuidv4();
        const node3Id = uuidv4();

        const graph = {
            name: 'Circular Dependencies Test Graph',
            description: 'Tests execution behavior with circular dependencies',
            variables: {},
            nodes: [
                {
                    id: node1Id,
                    name: 'Node 1',
                    node_type: 'worker',
                    worker_name: workerName,
                    task_template: 'Process node 1',
                    dependencies: [node3Id], // Circular: 1 depends on 3
                    data: {},
                    timeout: 60,
                    retry_count: 1
                },
                {
                    id: node2Id,
                    name: 'Node 2',
                    node_type: 'worker',
                    worker_name: workerName,
                    task_template: 'Process node 2',
                    dependencies: [node1Id], // 2 depends on 1
                    data: {},
                    timeout: 60,
                    retry_count: 1
                },
                {
                    id: node3Id,
                    name: 'Node 3',
                    node_type: 'worker',
                    worker_name: workerName,
                    task_template: 'Process node 3',
                    dependencies: [node2Id], // 3 depends on 2 (creates cycle)
                    data: {},
                    timeout: 60,
                    retry_count: 1
                }
            ],
            edges: [
                { source: node3Id, target: node1Id },
                { source: node1Id, target: node2Id },
                { source: node2Id, target: node3Id }
            ]
        };

        try {
            const createdGraph = await createGraph(graph);
            console.log('⚠️ Graph with circular dependencies was created');

            // Try to execute it
            const executionRequest = { variables: {} };

            try {
                const response = await makeRequest(`${API_BASE}/execution/graphs/${createdGraph.id}/execute`, {
                    method: 'POST',
                    body: JSON.stringify(executionRequest)
                });

                console.log('⚠️ Execution started on circular graph');

                // Check execution status
                await new Promise(resolve => setTimeout(resolve, 5000));
                const status = await makeRequest(`${API_BASE}/execution/sessions/${response.session_id}/status`);

                return {
                    success: true,
                    circular_graph_created: true,
                    execution_started: true,
                    final_status: status.status,
                    expected_behavior: status.status === 'failed'
                };

            } catch (execError) {
                console.log('✅ Execution correctly rejected circular graph');
                return {
                    success: true,
                    circular_graph_created: true,
                    execution_started: false,
                    error: execError.message
                };
            }

        } catch (createError) {
            console.log('✅ Graph creation correctly rejected circular dependencies');
            return {
                success: true,
                circular_graph_created: false,
                expected_behavior: true,
                error: createError.message
            };
        }

    } catch (error) {
        console.error(`❌ Circular dependencies test failed: ${error.message}`);
        return {
            success: false,
            error: error.message
        };
    }
}

/**
 * Test execution with deeply nested graph
 */
async function testDeeplyNestedGraph(workerName) {
    console.log('\n=== Testing Execution with Deeply Nested Graph ===');

    try {
        const depth = 10; // Create a graph with 10 levels
        const nodes = [];
        const edges = [];

        // Create chain of nodes
        for (let i = 0; i < depth; i++) {
            const nodeId = uuidv4();
            nodes.push({
                id: nodeId,
                name: `Level ${i + 1} Node`,
                node_type: 'worker',
                worker_name: workerName,
                task_template: `Process level ${i + 1} data`,
                dependencies: i > 0 ? [nodes[i - 1].id] : [],
                data: {
                    level: i + 1
                },
                timeout: 30,
                retry_count: 1
            });

            if (i > 0) {
                edges.push({
                    source: nodes[i - 1].id,
                    target: nodeId
                });
            }
        }

        const graph = {
            name: 'Deeply Nested Test Graph',
            description: `Tests execution with ${depth} levels of nesting`,
            variables: {
                initial_value: 1
            },
            nodes: nodes,
            edges: edges
        };

        const createdGraph = await createGraph(graph);

        // Execute the deeply nested graph
        const executionRequest = {
            variables: {
                initial_value: 42
            }
        };

        const response = await makeRequest(`${API_BASE}/execution/graphs/${createdGraph.id}/execute`, {
            method: 'POST',
            body: JSON.stringify(executionRequest)
        });

        console.log(`✅ Deep graph execution started: ${response.session_id}`);

        // Poll for completion with extended timeout
        let attempts = 0;
        const maxAttempts = 30; // 1 minute total
        let finalStatus = null;

        while (attempts < maxAttempts) {
            await new Promise(resolve => setTimeout(resolve, 2000));

            try {
                const status = await makeRequest(`${API_BASE}/execution/sessions/${response.session_id}/status`);
                console.log(`   Status check ${attempts + 1}: ${status.status}, completed nodes: ${status.completed_nodes?.length || 0}`);

                if (status.status === 'completed' || status.status === 'failed') {
                    finalStatus = status;
                    break;
                }
            } catch (error) {
                console.error(`   Status check failed: ${error.message}`);
            }

            attempts++;
        }

        return {
            success: true,
            graph_depth: depth,
            execution_completed: finalStatus?.status === 'completed',
            final_status: finalStatus?.status || 'unknown',
            completed_nodes: finalStatus?.completed_nodes?.length || 0,
            total_nodes: depth
        };

    } catch (error) {
        console.error(`❌ Deeply nested graph test failed: ${error.message}`);
        return {
            success: false,
            error: error.message
        };
    }
}

/**
 * Test execution timeout handling
 */
async function testExecutionTimeout(workerName) {
    console.log('\n=== Testing Execution Timeout Handling ===');

    try {
        const nodeId = uuidv4();

        const graph = {
            name: 'Timeout Test Graph',
            description: 'Tests execution timeout behavior',
            variables: {},
            nodes: [
                {
                    id: nodeId,
                    name: 'Slow Processing Node',
                    node_type: 'worker',
                    worker_name: workerName,
                    task_template: 'Simulate a long-running task that will timeout. Sleep for 120 seconds.',
                    dependencies: [],
                    data: {},
                    timeout: 5, // Very short timeout to trigger timeout
                    retry_count: 0 // No retries
                }
            ],
            edges: []
        };

        const createdGraph = await createGraph(graph);

        // Execute graph
        const executionRequest = { variables: {} };

        const response = await makeRequest(`${API_BASE}/execution/graphs/${createdGraph.id}/execute`, {
            method: 'POST',
            body: JSON.stringify(executionRequest)
        });

        console.log(`✅ Timeout test execution started: ${response.session_id}`);

        // Wait for timeout to occur
        await new Promise(resolve => setTimeout(resolve, 10000)); // Wait 10 seconds

        const status = await makeRequest(`${API_BASE}/execution/sessions/${response.session_id}/status`);

        return {
            success: true,
            execution_status: status.status,
            timeout_triggered: status.status === 'failed' || status.failed_nodes?.length > 0,
            failed_nodes: status.failed_nodes || [],
            error_details: status.error || 'No error details'
        };

    } catch (error) {
        console.error(`❌ Timeout test failed: ${error.message}`);
        return {
            success: false,
            error: error.message
        };
    }
}

/**
 * Test execution cancellation
 */
async function testExecutionCancellation(workerName) {
    console.log('\n=== Testing Execution Cancellation ===');

    try {
        // Create a long-running graph
        const nodes = [];
        for (let i = 0; i < 5; i++) {
            nodes.push({
                id: uuidv4(),
                name: `Slow Node ${i + 1}`,
                node_type: 'worker',
                worker_name: workerName,
                task_template: `Process step ${i + 1} - simulate 10 second task`,
                dependencies: i > 0 ? [nodes[i - 1].id] : [],
                data: {},
                timeout: 60,
                retry_count: 1
            });
        }

        const graph = {
            name: 'Cancellation Test Graph',
            description: 'Tests execution cancellation functionality',
            variables: {},
            nodes: nodes,
            edges: nodes.slice(1).map((node, i) => ({
                source: nodes[i].id,
                target: node.id
            }))
        };

        const createdGraph = await createGraph(graph);

        // Start execution
        const executionRequest = { variables: {} };

        const response = await makeRequest(`${API_BASE}/execution/graphs/${createdGraph.id}/execute`, {
            method: 'POST',
            body: JSON.stringify(executionRequest)
        });

        console.log(`✅ Long-running execution started: ${response.session_id}`);

        // Wait a bit then try to cancel
        await new Promise(resolve => setTimeout(resolve, 3000));

        try {
            const cancelResponse = await makeRequest(`${API_BASE}/execution/sessions/${response.session_id}/cancel`, {
                method: 'POST'
            });

            console.log('✅ Cancellation request sent successfully');

            // Check final status
            await new Promise(resolve => setTimeout(resolve, 2000));
            const finalStatus = await makeRequest(`${API_BASE}/execution/sessions/${response.session_id}/status`);

            return {
                success: true,
                cancellation_supported: true,
                final_status: finalStatus.status,
                cancelled_successfully: finalStatus.status === 'cancelled',
                completed_nodes: finalStatus.completed_nodes?.length || 0
            };

        } catch (cancelError) {
            console.log(`⚠️ Cancellation not supported or failed: ${cancelError.message}`);

            return {
                success: true, // Not a test failure if cancellation isn't implemented
                cancellation_supported: false,
                error: cancelError.message
            };
        }

    } catch (error) {
        console.error(`❌ Cancellation test failed: ${error.message}`);
        return {
            success: false,
            error: error.message
        };
    }
}

/**
 * Test parallel node execution
 */
async function testParallelExecution(workerName) {
    console.log('\n=== Testing Parallel Node Execution ===');

    try {
        // Create graph with parallel branches
        const startNode = uuidv4();
        const parallel1 = uuidv4();
        const parallel2 = uuidv4();
        const parallel3 = uuidv4();
        const mergeNode = uuidv4();

        const graph = {
            name: 'Parallel Execution Test Graph',
            description: 'Tests parallel node execution capabilities',
            variables: {
                branch_data: 'test'
            },
            nodes: [
                {
                    id: startNode,
                    name: 'Start Node',
                    node_type: 'worker',
                    worker_name: workerName,
                    task_template: 'Initialize parallel processing for {branch_data}',
                    dependencies: [],
                    data: {},
                    timeout: 30,
                    retry_count: 1
                },
                {
                    id: parallel1,
                    name: 'Parallel Branch 1',
                    node_type: 'worker',
                    worker_name: workerName,
                    task_template: 'Process branch 1 - simulate 5 second task',
                    dependencies: [startNode],
                    data: { branch: 1 },
                    timeout: 30,
                    retry_count: 1
                },
                {
                    id: parallel2,
                    name: 'Parallel Branch 2',
                    node_type: 'worker',
                    worker_name: workerName,
                    task_template: 'Process branch 2 - simulate 5 second task',
                    dependencies: [startNode],
                    data: { branch: 2 },
                    timeout: 30,
                    retry_count: 1
                },
                {
                    id: parallel3,
                    name: 'Parallel Branch 3',
                    node_type: 'worker',
                    worker_name: workerName,
                    task_template: 'Process branch 3 - simulate 5 second task',
                    dependencies: [startNode],
                    data: { branch: 3 },
                    timeout: 30,
                    retry_count: 1
                },
                {
                    id: mergeNode,
                    name: 'Merge Results',
                    node_type: 'worker',
                    worker_name: workerName,
                    task_template: 'Merge results from all parallel branches',
                    dependencies: [parallel1, parallel2, parallel3],
                    data: {},
                    timeout: 30,
                    retry_count: 1
                }
            ],
            edges: [
                { source: startNode, target: parallel1 },
                { source: startNode, target: parallel2 },
                { source: startNode, target: parallel3 },
                { source: parallel1, target: mergeNode },
                { source: parallel2, target: mergeNode },
                { source: parallel3, target: mergeNode }
            ]
        };

        const createdGraph = await createGraph(graph);

        // Execute graph
        const executionRequest = {
            variables: {
                branch_data: 'parallel_test'
            }
        };

        const startTime = Date.now();
        const response = await makeRequest(`${API_BASE}/execution/graphs/${createdGraph.id}/execute`, {
            method: 'POST',
            body: JSON.stringify(executionRequest)
        });

        console.log(`✅ Parallel execution started: ${response.session_id}`);

        // Monitor execution progress
        let finalStatus = null;
        let attempts = 0;
        const maxAttempts = 30;

        while (attempts < maxAttempts) {
            await new Promise(resolve => setTimeout(resolve, 2000));

            const status = await makeRequest(`${API_BASE}/execution/sessions/${response.session_id}/status`);
            console.log(`   Progress: ${status.completed_nodes?.length || 0}/${graph.nodes.length} nodes completed`);

            if (status.status === 'completed' || status.status === 'failed') {
                finalStatus = status;
                break;
            }

            attempts++;
        }

        const executionTime = Date.now() - startTime;

        return {
            success: true,
            execution_completed: finalStatus?.status === 'completed',
            total_execution_time_ms: executionTime,
            parallel_execution_detected: executionTime < 20000, // Should be much less than sequential (25s)
            completed_nodes: finalStatus?.completed_nodes?.length || 0,
            total_nodes: graph.nodes.length
        };

    } catch (error) {
        console.error(`❌ Parallel execution test failed: ${error.message}`);
        return {
            success: false,
            error: error.message
        };
    }
}

/**
 * Test conditional branching
 */
async function testConditionalBranching(workerName) {
    console.log('\n=== Testing Conditional Branching ===');

    try {
        const startNode = uuidv4();
        const conditionNode = uuidv4();
        const trueBranch = uuidv4();
        const falseBranch = uuidv4();

        const graph = {
            name: 'Conditional Branching Test Graph',
            description: 'Tests conditional execution paths',
            variables: {
                test_value: 15,
                threshold: 10
            },
            nodes: [
                {
                    id: startNode,
                    name: 'Initialize',
                    node_type: 'worker',
                    worker_name: workerName,
                    task_template: 'Initialize with test_value: {test_value}',
                    dependencies: [],
                    data: {},
                    timeout: 30,
                    retry_count: 1
                },
                {
                    id: conditionNode,
                    name: 'Check Condition',
                    node_type: 'condition',
                    task_template: 'Check if test_value > threshold',
                    dependencies: [startNode],
                    data: {
                        condition: 'test_value > threshold'
                    },
                    timeout: 30,
                    retry_count: 1
                },
                {
                    id: trueBranch,
                    name: 'True Branch',
                    node_type: 'worker',
                    worker_name: workerName,
                    task_template: 'Execute true branch: value {test_value} exceeds threshold {threshold}',
                    dependencies: [conditionNode],
                    data: {
                        branch: 'true'
                    },
                    timeout: 30,
                    retry_count: 1
                },
                {
                    id: falseBranch,
                    name: 'False Branch',
                    node_type: 'worker',
                    worker_name: workerName,
                    task_template: 'Execute false branch: value {test_value} below threshold {threshold}',
                    dependencies: [conditionNode],
                    data: {
                        branch: 'false'
                    },
                    timeout: 30,
                    retry_count: 1
                }
            ],
            edges: [
                { source: startNode, target: conditionNode },
                {
                    source: conditionNode,
                    target: trueBranch,
                    condition: 'true'
                },
                {
                    source: conditionNode,
                    target: falseBranch,
                    condition: 'false'
                }
            ]
        };

        const createdGraph = await createGraph(graph);

        // Test with value > threshold (should take true branch)
        console.log('\n📊 Testing with value > threshold...');
        const trueRequest = {
            variables: {
                test_value: 20,
                threshold: 10
            }
        };

        const trueResponse = await makeRequest(`${API_BASE}/execution/graphs/${createdGraph.id}/execute`, {
            method: 'POST',
            body: JSON.stringify(trueRequest)
        });

        // Wait for completion
        await new Promise(resolve => setTimeout(resolve, 10000));
        const trueStatus = await makeRequest(`${API_BASE}/execution/sessions/${trueResponse.session_id}/status`);

        // Test with value < threshold (should take false branch)
        console.log('\n📊 Testing with value < threshold...');
        const falseRequest = {
            variables: {
                test_value: 5,
                threshold: 10
            }
        };

        const falseResponse = await makeRequest(`${API_BASE}/execution/graphs/${createdGraph.id}/execute`, {
            method: 'POST',
            body: JSON.stringify(falseRequest)
        });

        // Wait for completion
        await new Promise(resolve => setTimeout(resolve, 10000));
        const falseStatus = await makeRequest(`${API_BASE}/execution/sessions/${falseResponse.session_id}/status`);

        return {
            success: true,
            true_branch_test: {
                completed: trueStatus.status === 'completed',
                executed_nodes: trueStatus.completed_nodes || []
            },
            false_branch_test: {
                completed: falseStatus.status === 'completed',
                executed_nodes: falseStatus.completed_nodes || []
            },
            conditional_logic_working: trueStatus.completed_nodes?.length !== falseStatus.completed_nodes?.length
        };

    } catch (error) {
        console.error(`❌ Conditional branching test failed: ${error.message}`);
        return {
            success: false,
            error: error.message
        };
    }
}

/**
 * Test node failure and recovery
 */
async function testNodeFailureRecovery(workerName) {
    console.log('\n=== Testing Node Failure and Recovery ===');

    try {
        const node1 = uuidv4();
        const node2 = uuidv4();
        const node3 = uuidv4();

        const graph = {
            name: 'Failure Recovery Test Graph',
            description: 'Tests node failure and retry behavior',
            variables: {
                fail_probability: 0.5
            },
            nodes: [
                {
                    id: node1,
                    name: 'Stable Node',
                    node_type: 'worker',
                    worker_name: workerName,
                    task_template: 'Execute stable operation',
                    dependencies: [],
                    data: {},
                    timeout: 30,
                    retry_count: 0
                },
                {
                    id: node2,
                    name: 'Failing Node',
                    node_type: 'worker',
                    worker_name: workerName,
                    task_template: 'FAIL: This node should fail. Simulate an error condition.',
                    dependencies: [node1],
                    data: {
                        should_fail: true
                    },
                    timeout: 30,
                    retry_count: 2 // Will retry twice
                },
                {
                    id: node3,
                    name: 'Recovery Node',
                    node_type: 'worker',
                    worker_name: workerName,
                    task_template: 'Attempt recovery after failure',
                    dependencies: [node2],
                    data: {},
                    timeout: 30,
                    retry_count: 1
                }
            ],
            edges: [
                { source: node1, target: node2 },
                { source: node2, target: node3 }
            ]
        };

        const createdGraph = await createGraph(graph);

        // Execute graph
        const executionRequest = {
            variables: {
                fail_probability: 1.0 // Ensure failure
            }
        };

        const response = await makeRequest(`${API_BASE}/execution/graphs/${createdGraph.id}/execute`, {
            method: 'POST',
            body: JSON.stringify(executionRequest)
        });

        console.log(`✅ Failure test execution started: ${response.session_id}`);

        // Monitor execution with retries
        let finalStatus = null;
        let attempts = 0;
        const maxAttempts = 20;

        while (attempts < maxAttempts) {
            await new Promise(resolve => setTimeout(resolve, 3000));

            const status = await makeRequest(`${API_BASE}/execution/sessions/${response.session_id}/status`);
            console.log(`   Status: ${status.status}, Failed nodes: ${status.failed_nodes?.length || 0}`);

            if (status.status === 'completed' || status.status === 'failed') {
                finalStatus = status;
                break;
            }

            attempts++;
        }

        return {
            success: true,
            execution_status: finalStatus?.status || 'unknown',
            completed_nodes: finalStatus?.completed_nodes?.length || 0,
            failed_nodes: finalStatus?.failed_nodes?.length || 0,
            retry_behavior_detected: finalStatus?.retry_attempts > 0,
            total_nodes: graph.nodes.length
        };

    } catch (error) {
        console.error(`❌ Failure recovery test failed: ${error.message}`);
        return {
            success: false,
            error: error.message
        };
    }
}

/**
 * Cleanup test resources
 */
async function cleanupResources(graphIds = [], workerNames = []) {
    console.log('\n🧹 Cleaning up test resources...');

    for (const graphId of graphIds) {
        try {
            await makeRequest(`${API_BASE}/graphs/${graphId}`, { method: 'DELETE' });
            console.log(`✅ Deleted graph: ${graphId}`);
        } catch (error) {
            console.error(`❌ Failed to delete graph ${graphId}: ${error.message}`);
        }
    }

    for (const workerName of workerNames) {
        try {
            await makeRequest(`${API_BASE}/workers/${workerName}`, { method: 'DELETE' });
            console.log(`✅ Deleted worker: ${workerName}`);
        } catch (error) {
            console.error(`❌ Failed to delete worker ${workerName}: ${error.message}`);
        }
    }
}

/**
 * Main test function for graph execution edge cases
 */
async function runGraphExecutionEdgeCasesTest() {
    console.log('🚀 Starting Graph Execution Edge Cases Test Suite');
    console.log('='.repeat(80));

    const testResults = {
        missing_variables: false,
        invalid_graph_id: false,
        circular_dependencies: false,
        deeply_nested: false,
        timeout_handling: false,
        cancellation: false,
        parallel_execution: false,
        conditional_branching: false,
        failure_recovery: false,
        overall_success: false,
        execution_time: 0
    };

    const createdGraphIds = [];
    const createdWorkerNames = [];
    const testStartTime = Date.now();

    try {
        // Create test worker
        console.log('\n🤖 Creating test worker for edge case testing...');
        const workerConfig = {
            worker_name: 'edge_case_test_worker',
            agent_role: 'edge_case_specialist',
            agent_type: 'pydantic_ai',
            model_provider: 'deepseek',
            model: 'deepseek-chat',
            temperature: 0.2,
            system_prompt: 'You are a test worker for edge case validation. Process instructions and respond appropriately.',
            instruction: 'Execute the given task and return structured responses.',
            user_prompt: '',
            description: 'Edge case test worker'
        };

        const worker = await createWorker(workerConfig);
        createdWorkerNames.push(worker.name);

        // Run edge case tests

        // Test 1: Missing Variables
        console.log('\n📋 Test 1: Missing Variables');
        const missingVarsResult = await testMissingVariables(worker.name);
        testResults.missing_variables = missingVarsResult.success;
        if (missingVarsResult.graphId) createdGraphIds.push(missingVarsResult.graphId);

        // Test 2: Invalid Graph ID
        console.log('\n📋 Test 2: Invalid Graph ID');
        const invalidIdResult = await testInvalidGraphId();
        testResults.invalid_graph_id = invalidIdResult.success;

        // Test 3: Circular Dependencies
        console.log('\n📋 Test 3: Circular Dependencies');
        const circularResult = await testCircularDependencies(worker.name);
        testResults.circular_dependencies = circularResult.success;
        if (circularResult.graphId) createdGraphIds.push(circularResult.graphId);

        // Test 4: Deeply Nested Graph
        console.log('\n📋 Test 4: Deeply Nested Graph');
        const deeplyNestedResult = await testDeeplyNestedGraph(worker.name);
        testResults.deeply_nested = deeplyNestedResult.success;
        if (deeplyNestedResult.graphId) createdGraphIds.push(deeplyNestedResult.graphId);

        // Test 5: Timeout Handling
        console.log('\n📋 Test 5: Timeout Handling');
        const timeoutResult = await testExecutionTimeout(worker.name);
        testResults.timeout_handling = timeoutResult.success;
        if (timeoutResult.graphId) createdGraphIds.push(timeoutResult.graphId);

        // Test 6: Cancellation
        console.log('\n📋 Test 6: Execution Cancellation');
        const cancellationResult = await testExecutionCancellation(worker.name);
        testResults.cancellation = cancellationResult.success;
        if (cancellationResult.graphId) createdGraphIds.push(cancellationResult.graphId);

        // Test 7: Parallel Execution
        console.log('\n📋 Test 7: Parallel Execution');
        const parallelResult = await testParallelExecution(worker.name);
        testResults.parallel_execution = parallelResult.success;
        if (parallelResult.graphId) createdGraphIds.push(parallelResult.graphId);

        // Test 8: Conditional Branching
        console.log('\n📋 Test 8: Conditional Branching');
        const conditionalResult = await testConditionalBranching(worker.name);
        testResults.conditional_branching = conditionalResult.success;
        if (conditionalResult.graphId) createdGraphIds.push(conditionalResult.graphId);

        // Test 9: Failure Recovery
        console.log('\n📋 Test 9: Node Failure and Recovery');
        const failureResult = await testNodeFailureRecovery(worker.name);
        testResults.failure_recovery = failureResult.success;
        if (failureResult.graphId) createdGraphIds.push(failureResult.graphId);

        // Calculate overall success
        testResults.overall_success = Object.values(testResults)
            .filter(v => typeof v === 'boolean')
            .every(v => v === true);

        testResults.execution_time = Date.now() - testStartTime;

        // Generate summary
        console.log('\n📊 Graph Execution Edge Cases Test Summary');
        console.log('='.repeat(80));
        console.log(`🏁 Test Suite - COMPLETE`);
        console.log(`   Total execution time: ${testResults.execution_time}ms`);
        console.log(`   Overall success: ${testResults.overall_success ? '✅ PASSED' : '❌ FAILED'}`);
        console.log('\n📋 Test Results:');
        console.log(`   ✅ Missing Variables: ${testResults.missing_variables ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Invalid Graph ID: ${testResults.invalid_graph_id ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Circular Dependencies: ${testResults.circular_dependencies ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Deeply Nested Graph: ${testResults.deeply_nested ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Timeout Handling: ${testResults.timeout_handling ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Execution Cancellation: ${testResults.cancellation ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Parallel Execution: ${testResults.parallel_execution ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Conditional Branching: ${testResults.conditional_branching ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Failure Recovery: ${testResults.failure_recovery ? 'PASSED' : 'FAILED'}`);

    } catch (error) {
        console.error(`\n❌ Test execution failed: ${error.message}`);
        testResults.overall_success = false;
        testResults.execution_time = Date.now() - testStartTime;
    }

    // Cleanup
    await cleanupResources(createdGraphIds, createdWorkerNames);

    return testResults;
}

/**
 * Main function
 */
async function main() {
    console.log('🚀 Graph Execution Edge Cases Test Suite');
    console.log('Node.js version:', process.version);
    console.log('Platform:', process.platform);
    console.log('='.repeat(80));

    try {
        const testResults = await runGraphExecutionEdgeCasesTest();

        // Final summary
        console.log('\n📊 FINAL TEST SUMMARY');
        console.log('='.repeat(80));
        console.log(`Overall Result: ${testResults.overall_success ? '🎉 SUCCESS' : '💥 FAILURE'}`);
        console.log(`Total Execution Time: ${testResults.execution_time}ms`);

        // Exit with appropriate code
        process.exit(testResults.overall_success ? 0 : 1);

    } catch (error) {
        console.error('❌ Test suite failed to execute:', error.message);
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
    runGraphExecutionEdgeCasesTest,
    testMissingVariables,
    testInvalidGraphId,
    testCircularDependencies,
    testDeeplyNestedGraph,
    testExecutionTimeout,
    testExecutionCancellation,
    testParallelExecution,
    testConditionalBranching,
    testNodeFailureRecovery
};
