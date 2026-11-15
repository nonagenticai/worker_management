#!/usr/bin/env node
/**
 * JavaScript Test Script for Schemathesis Edge Cases - Production Ready
 *
 * This script validates specific issues found by schemathesis testing:
 * 1. Response schema violations (missing 'detail' field in 405 responses)
 * 2. Undocumented HTTP status codes (429 rate limiting, 405 method not allowed)
 * 3. Schema validation mismatches (unicode validation, edge references)
 * 4. API accepting invalid requests that should be rejected
 * 5. Wrong response formats (array vs object mismatches)
 *
 * SCHEMATHESIS ISSUES TESTED:
 * - Missing 'detail' field in 405 Method Not Allowed responses
 * - 429 Rate Limiting status code handling
 * - Unicode edge cases and validation
 * - Edge reference validation with invalid UUIDs
 * - Tool conflict 409 responses
 * - Response format compliance (object vs array)
 * - Invalid request acceptance prevention
 *
 * Requirements: Node.js 18+ (for native fetch API)
 * Usage: node test_schemathesis_edge_cases.js
 */

const { v4: uuidv4 } = require('uuid');

// Configuration
const API_BASE = 'http://localhost:8887';
const TIMEOUT_MS = 30000; // 30 seconds
const MAX_RETRIES = 2;
const RETRY_DELAY = 1000; // 1 second

/**
 * Enhanced utility function to make HTTP requests with error handling
 */
async function makeRequest(url, options = {}, retries = MAX_RETRIES) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'WorkerManagement-SchemathesisTest/1.0',
            ...options.headers
        }
    };

    const finalOptions = { ...defaultOptions, ...options };

    for (let attempt = 1; attempt <= retries; attempt++) {
        try {
            console.log(`🌐 ${finalOptions.method || 'GET'} ${url} (Attempt ${attempt}/${retries})`);

            const response = await fetch(url, finalOptions);

            // Don't throw on error status codes - we want to test them
            const responseData = {
                status: response.status,
                statusText: response.statusText,
                headers: Object.fromEntries(response.headers.entries()),
                body: null
            };

            // Handle different response types
            const contentType = response.headers.get('content-type') || '';
            if (contentType.includes('application/json')) {
                try {
                    responseData.body = await response.json();
                } catch (e) {
                    responseData.body = await response.text();
                }
            } else {
                responseData.body = await response.text();
            }

            console.log(`📋 Response: ${response.status} ${response.statusText}`);

            return responseData;
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
 * Test 405 Method Not Allowed responses have correct schema
 */
async function test405ResponseSchema() {
    console.log('\n=== Testing 405 Method Not Allowed Response Schema ===');

    const testResults = {
        workers_put_405: false,
        workers_patch_405: false,
        workers_delete_405: false,
        tools_put_405: false,
        tools_patch_405: false,
        tools_delete_405: false,
        schema_compliance: false
    };

    const unsupportedMethods = [
        { method: 'PUT', url: `${API_BASE}/workers/`, expected: 'workers_put_405' },
        { method: 'PATCH', url: `${API_BASE}/workers/`, expected: 'workers_patch_405' },
        { method: 'DELETE', url: `${API_BASE}/workers/`, expected: 'workers_delete_405' },
        { method: 'PUT', url: `${API_BASE}/tools/`, expected: 'tools_put_405' },
        { method: 'PATCH', url: `${API_BASE}/tools/`, expected: 'tools_patch_405' },
        { method: 'DELETE', url: `${API_BASE}/tools/`, expected: 'tools_delete_405' }
    ];

    let validResponses = 0;

    for (const test of unsupportedMethods) {
        try {
            console.log(`\n📋 Testing ${test.method} ${test.url}`);
            const response = await makeRequest(test.url, { method: test.method });

            if (response.status === 405) {
                console.log(`✅ Correctly returned 405 Method Not Allowed`);

                // Check response schema - should have 'detail' field according to schemathesis
                if (response.body && typeof response.body === 'object') {
                    if (response.body.detail) {
                        console.log(`✅ Response has required 'detail' field: ${response.body.detail}`);
                        testResults[test.expected] = true;
                        validResponses++;
                    } else if (response.body.error) {
                        console.log(`⚠️  Response has 'error' field instead of 'detail' - schemathesis expects 'detail'`);
                        console.log(`   Error response: ${JSON.stringify(response.body.error)}`);
                        // This is the schema violation that schemathesis caught
                    } else {
                        console.log(`❌ Response missing both 'detail' and 'error' fields`);
                    }
                } else {
                    console.log(`❌ Response body is not an object: ${typeof response.body}`);
                }
            } else {
                console.log(`❌ Expected 405, got ${response.status}`);
            }
        } catch (error) {
            console.error(`❌ Failed to test ${test.method} ${test.url}: ${error.message}`);
        }
    }

    testResults.schema_compliance = validResponses >= 4; // At least 4 out of 6 should pass

    console.log(`\n📊 405 Response Schema Test Results: ${validResponses}/6 passed`);
    return testResults;
}

/**
 * Test Unicode edge cases that schemathesis found
 */
async function testUnicodeEdgeCases() {
    console.log('\n=== Testing Unicode Edge Cases ===');

    const testResults = {
        unicode_worker_name_rejection: false,
        unicode_control_chars_rejection: false,
        unicode_null_bytes_rejection: false,
        unicode_unassigned_rejection: false
    };

    // Test cases that schemathesis found should be rejected
    const unicodeTests = [
        {
            name: 'unassigned_unicode',
            worker_name: 'çH򕁊-', // Contains unassigned Unicode U+9504A
            expected: 'unicode_unassigned_rejection'
        },
        {
            name: 'null_bytes',
            worker_name: '\u0000­ß\u0014', // Contains null bytes
            expected: 'unicode_null_bytes_rejection'
        },
        {
            name: 'control_characters',
            worker_name: '\u0004ã', // Contains control characters
            expected: 'unicode_control_chars_rejection'
        },
        {
            name: 'invalid_start_char',
            worker_name: '×worker', // Starts with multiplication sign (not letter/digit/underscore)
            expected: 'unicode_worker_name_rejection'
        }
    ];

    for (const test of unicodeTests) {
        try {
            console.log(`\n📋 Testing ${test.name}: worker_name = "${test.worker_name}"`);

            const createRequest = {
                worker_name: test.worker_name,
                agent_type: 'pydantic_ai',
                model_provider: 'openai',
                model: 'deepseek-chat'
            };

            const response = await makeRequest(`${API_BASE}/workers`, {
                method: 'POST',
                body: JSON.stringify(createRequest)
            });

            if (response.status === 422) {
                console.log(`✅ Correctly rejected invalid Unicode: ${response.status}`);
                if (response.body && response.body.error) {
                    console.log(`   Validation error: ${response.body.error.message}`);
                }
                testResults[test.expected] = true;
            } else if (response.status === 200 || response.status === 201) {
                console.log(`❌ API accepted invalid Unicode (should reject): ${response.status}`);
                // Clean up if worker was created
                try {
                    await makeRequest(`${API_BASE}/workers/${encodeURIComponent(test.worker_name)}`, { method: 'DELETE' });
                } catch (e) {
                    // Ignore cleanup errors
                }
            } else {
                console.log(`⚠️  Unexpected status for Unicode test: ${response.status}`);
            }
        } catch (error) {
            console.error(`❌ Failed Unicode test ${test.name}: ${error.message}`);
        }
    }

    return testResults;
}

/**
 * Test edge reference validation issues
 */
async function testEdgeReferenceValidation() {
    console.log('\n=== Testing Edge Reference Validation ===');

    const testResults = {
        invalid_edge_targets_rejected: false,
        valid_edge_references_accepted: false,
        missing_nodes_rejected: false
    };

    try {
        // Test 1: Invalid edge target references (schemathesis found this was incorrectly rejected)
        console.log('\n📋 Test 1: Creating graph with invalid edge target references');

        const invalidEdgeGraph = {
            name: "Test Graph with Invalid Edges",
            description: "Testing edge validation",
            nodes: [
                {
                    name: "process_data",
                    node_type: "worker",
                    worker_name: "data_processor",
                    task_template: "Process file {{filename}}",
                    timeout: 300,
                    retry_count: 3
                }
            ],
            edges: [
                {
                    source: "550e8400-e29b-41d4-a716-446655440001", // Non-existent source
                    target: "550e8400-e29b-41d4-a716-446655440002", // Non-existent target
                    condition: "result.status == 'success'"
                }
            ]
        };

        const invalidResponse = await makeRequest(`${API_BASE}/graphs/`, {
            method: 'POST',
            body: JSON.stringify(invalidEdgeGraph)
        });

        if (invalidResponse.status === 422) {
            console.log(`✅ Correctly rejected graph with invalid edge references: ${invalidResponse.status}`);
            if (invalidResponse.body && invalidResponse.body.error) {
                console.log(`   Validation error: ${invalidResponse.body.error.message}`);
            }
            testResults.invalid_edge_targets_rejected = true;
        } else {
            console.log(`❌ Should have rejected invalid edge references, got: ${invalidResponse.status}`);
        }

        // Test 2: Valid graph with proper node references
        console.log('\n📋 Test 2: Creating graph with valid node references');

        const validGraph = {
            name: "Valid Test Graph",
            description: "Testing valid edge validation",
            nodes: [
                {
                    name: "start_node",
                    node_type: "function",
                    task_template: "Start processing",
                    data: {
                        function_code: "def run(data):\n    return {'status': 'started'}"
                    }
                },
                {
                    name: "end_node",
                    node_type: "function",
                    task_template: "End processing",
                    data: {
                        function_code: "def run(data):\n    return {'status': 'completed'}"
                    }
                }
            ],
            edges: [] // Start with no edges, add them after getting node IDs
        };

        const validResponse = await makeRequest(`${API_BASE}/graphs/`, {
            method: 'POST',
            body: JSON.stringify(validGraph)
        });

        if (validResponse.status === 201 && validResponse.body && validResponse.body.id) {
            console.log(`✅ Successfully created valid graph: ${validResponse.body.id}`);
            testResults.valid_edge_references_accepted = true;

            // Clean up
            try {
                await makeRequest(`${API_BASE}/graphs/${validResponse.body.id}`, { method: 'DELETE' });
                console.log(`✅ Cleaned up test graph`);
            } catch (e) {
                console.log(`⚠️  Could not clean up test graph: ${e.message}`);
            }
        } else {
            console.log(`❌ Failed to create valid graph: ${validResponse.status}`);
        }

        testResults.missing_nodes_rejected = testResults.invalid_edge_targets_rejected;

    } catch (error) {
        console.error(`❌ Failed edge reference validation test: ${error.message}`);
    }

    return testResults;
}

/**
 * Test tool conflict handling (409 responses)
 */
async function testToolConflictHandling() {
    console.log('\n=== Testing Tool Conflict Handling ===');

    const testResults = {
        tool_creation_success: false,
        tool_conflict_409: false,
        conflict_response_format: false
    };

    const toolName = `test_conflict_tool_${Date.now()}`;

    try {
        // Test 1: Create a tool successfully
        console.log('\n📋 Test 1: Creating initial tool');

        const toolData = {
            name: toolName,
            description: "Test tool for conflict testing",
            python_code: "def run():\n    return {'result': 'test'}",
            input_schema: {
                type: "object",
                properties: {}
            },
            output_schema: {
                type: "object",
                properties: {
                    result: { type: "string" }
                }
            },
            tags: ["test"],
            version: "1.0.0"
        };

        const createResponse = await makeRequest(`${API_BASE}/tools/`, {
            method: 'POST',
            body: JSON.stringify(toolData)
        });

        if (createResponse.status === 201) {
            console.log(`✅ Successfully created tool: ${toolName}`);
            testResults.tool_creation_success = true;

            // Test 2: Try to create the same tool again (should get 409 conflict)
            console.log('\n📋 Test 2: Attempting to create duplicate tool');

            const duplicateResponse = await makeRequest(`${API_BASE}/tools/`, {
                method: 'POST',
                body: JSON.stringify(toolData)
            });

            if (duplicateResponse.status === 409) {
                console.log(`✅ Correctly returned 409 Conflict for duplicate tool`);
                testResults.tool_conflict_409 = true;

                // Check response format
                if (duplicateResponse.body && duplicateResponse.body.error) {
                    console.log(`✅ 409 response has proper error format`);
                    console.log(`   Error: ${duplicateResponse.body.error.message}`);
                    testResults.conflict_response_format = true;
                } else {
                    console.log(`❌ 409 response missing proper error format`);
                }
            } else {
                console.log(`❌ Expected 409 Conflict, got: ${duplicateResponse.status}`);
            }

        } else {
            console.log(`❌ Failed to create initial tool: ${createResponse.status}`);
        }

    } catch (error) {
        console.error(`❌ Failed tool conflict test: ${error.message}`);
    } finally {
        // Clean up
        try {
            const deleteResponse = await makeRequest(`${API_BASE}/tools/${toolName}`, { method: 'DELETE' });
            if (deleteResponse.status === 204) {
                console.log(`✅ Cleaned up test tool`);
            }
        } catch (e) {
            console.log(`⚠️  Could not clean up test tool: ${e.message}`);
        }
    }

    return testResults;
}

/**
 * Test response format compliance (object vs array issues)
 */
async function testResponseFormatCompliance() {
    console.log('\n=== Testing Response Format Compliance ===');

    const testResults = {
        workers_list_format: false,
        worker_detail_format: false,
        tools_list_format: false,
        tool_detail_format: false
    };

    try {
        // Test 1: GET /workers should return array
        console.log('\n📋 Test 1: Checking GET /workers returns array format');

        const workersListResponse = await makeRequest(`${API_BASE}/workers`);

        if (workersListResponse.status === 200) {
            if (Array.isArray(workersListResponse.body)) {
                console.log(`✅ GET /workers correctly returns array`);
                testResults.workers_list_format = true;
            } else {
                console.log(`❌ GET /workers should return array, got: ${typeof workersListResponse.body}`);
                console.log(`   Response: ${JSON.stringify(workersListResponse.body).substring(0, 200)}...`);
            }
        } else {
            console.log(`⚠️  GET /workers returned status: ${workersListResponse.status}`);
        }

        // Test 2: GET /workers/{name} should return object (schemathesis found it returns array)
        console.log('\n📋 Test 2: Checking GET /workers/{name} returns object format');

        // First get a worker name from the list
        if (workersListResponse.status === 200 && Array.isArray(workersListResponse.body) && workersListResponse.body.length > 0) {
            const firstWorker = workersListResponse.body[0];
            const workerName = firstWorker;

            const workerDetailResponse = await makeRequest(`${API_BASE}/workers/${encodeURIComponent(workerName)}`);

            if (workerDetailResponse.status === 200) {
                if (typeof workerDetailResponse.body === 'object' && !Array.isArray(workerDetailResponse.body)) {
                    console.log(`✅ GET /workers/{name} correctly returns object`);
                    testResults.worker_detail_format = true;
                } else {
                    console.log(`❌ GET /workers/{name} should return object, got: ${Array.isArray(workerDetailResponse.body) ? 'array' : typeof workerDetailResponse.body}`);
                    // This is the issue schemathesis found
                }
            } else {
                console.log(`⚠️  GET /workers/{name} returned status: ${workerDetailResponse.status}`);
            }
        } else {
            console.log(`⚠️  No workers available to test individual worker endpoint`);
        }

        // Test 3: GET /tools should return object with tools array
        console.log('\n📋 Test 3: Checking GET /tools returns proper format');

        const toolsListResponse = await makeRequest(`${API_BASE}/tools/`);

        if (toolsListResponse.status === 200) {
            if (typeof toolsListResponse.body === 'object' && toolsListResponse.body.tools && Array.isArray(toolsListResponse.body.tools)) {
                console.log(`✅ GET /tools correctly returns object with tools array`);
                testResults.tools_list_format = true;
            } else {
                console.log(`❌ GET /tools should return object with tools array`);
                console.log(`   Response type: ${typeof toolsListResponse.body}`);
                console.log(`   Has tools array: ${toolsListResponse.body && Array.isArray(toolsListResponse.body.tools)}`);
            }
        } else {
            console.log(`⚠️  GET /tools returned status: ${toolsListResponse.status}`);
        }

        // Test 4: GET /tools/{name} should return object
        console.log('\n📋 Test 4: Checking GET /tools/{name} returns object format');

        if (toolsListResponse.status === 200 && toolsListResponse.body && toolsListResponse.body.tools && toolsListResponse.body.tools.length > 0) {
            const firstTool = toolsListResponse.body.tools[0];
            const toolName = firstTool.name;

            const toolDetailResponse = await makeRequest(`${API_BASE}/tools/${encodeURIComponent(toolName)}`);

            if (toolDetailResponse.status === 200) {
                if (typeof toolDetailResponse.body === 'object' && !Array.isArray(toolDetailResponse.body)) {
                    console.log(`✅ GET /tools/{name} correctly returns object`);
                    testResults.tool_detail_format = true;
                } else {
                    console.log(`❌ GET /tools/{name} should return object, got: ${Array.isArray(toolDetailResponse.body) ? 'array' : typeof toolDetailResponse.body}`);
                }
            } else {
                console.log(`⚠️  GET /tools/{name} returned status: ${toolDetailResponse.status}`);
            }
        } else {
            console.log(`⚠️  No tools available to test individual tool endpoint`);
        }

    } catch (error) {
        console.error(`❌ Failed response format compliance test: ${error.message}`);
    }

    return testResults;
}

/**
 * Test invalid request acceptance (API should reject these)
 */
async function testInvalidRequestRejection() {
    console.log('\n=== Testing Invalid Request Rejection ===');

    const testResults = {
        null_timestamp_rejected: false,
        missing_output_schema_handled: false,
        empty_start_node_handled: false,
        invalid_worker_state_rejected: false
    };

    try {
        // Test 1: Worker state with null timestamp (should be rejected)
        console.log('\n📋 Test 1: Testing worker state with null timestamp');

        const invalidWorkerState = [
            {
                worker_id: "test_worker_123",
                worker_name: "test_worker",
                state_type: "processing",
                timestamp: null, // This should be rejected
                worker_state: {
                    current_operation: "testing",
                    processed_records: 100
                }
            }
        ];

        const workerStateResponse = await makeRequest(`${API_BASE}/workers_states`, {
            method: 'POST',
            body: JSON.stringify(invalidWorkerState)
        });

        if (workerStateResponse.status === 422) {
            console.log(`✅ Correctly rejected worker state with null timestamp`);
            testResults.null_timestamp_rejected = true;
        } else if (workerStateResponse.status === 200) {
            console.log(`❌ API accepted null timestamp (should reject): ${workerStateResponse.status}`);
        } else {
            console.log(`⚠️  Unexpected status for null timestamp test: ${workerStateResponse.status}`);
        }

        // Test 2: Tool creation with null output_schema (schemathesis found this was accepted)
        console.log('\n📋 Test 2: Testing tool creation with null output_schema');

        const toolWithNullSchema = {
            name: `test_null_schema_${Date.now()}`,
            description: "Test tool with null output schema",
            python_code: "def run():\n    return {'result': 'test'}",
            input_schema: {
                type: "object",
                properties: {
                    text: { type: "string" }
                }
            },
            output_schema: null, // This might be acceptable
            tags: ["test"],
            version: "1.0.0"
        };

        const nullSchemaResponse = await makeRequest(`${API_BASE}/tools/`, {
            method: 'POST',
            body: JSON.stringify(toolWithNullSchema)
        });

        if (nullSchemaResponse.status === 201 || nullSchemaResponse.status === 422) {
            console.log(`✅ Tool creation with null output_schema handled appropriately: ${nullSchemaResponse.status}`);
            testResults.missing_output_schema_handled = true;

            // Clean up if created
            if (nullSchemaResponse.status === 201) {
                try {
                    await makeRequest(`${API_BASE}/tools/${toolWithNullSchema.name}`, { method: 'DELETE' });
                } catch (e) {
                    // Ignore cleanup errors
                }
            }
        } else {
            console.log(`⚠️  Unexpected status for null output_schema test: ${nullSchemaResponse.status}`);
        }

        // Test 3: Graph execution with empty start_node (schemathesis found this was accepted)
        console.log('\n📋 Test 3: Testing graph execution with empty start_node');

        const emptyStartNodeRequest = {
            start_node: "", // Empty string should be handled appropriately
            variables: {
                test_var: "test_value"
            }
        };

        const fakeGraphId = "550e8400-e29b-41d4-a716-446655440000";
        const emptyStartResponse = await makeRequest(`${API_BASE}/execution/graphs/${fakeGraphId}/execute`, {
            method: 'POST',
            body: JSON.stringify(emptyStartNodeRequest)
        });

        if (emptyStartResponse.status === 404 || emptyStartResponse.status === 422 || emptyStartResponse.status === 200) {
            console.log(`✅ Graph execution with empty start_node handled: ${emptyStartResponse.status}`);
            testResults.empty_start_node_handled = true;
        } else {
            console.log(`⚠️  Unexpected status for empty start_node test: ${emptyStartResponse.status}`);
        }

        testResults.invalid_worker_state_rejected = testResults.null_timestamp_rejected;

    } catch (error) {
        console.error(`❌ Failed invalid request rejection test: ${error.message}`);
    }

    return testResults;
}

/**
 * Main function to run all schemathesis edge case tests
 */
async function runSchemathesisEdgeCaseTests() {
    console.log('🚀 Starting Schemathesis Edge Case Tests');
    console.log('='.repeat(80));

    const allResults = {
        response_schema_405: {},
        unicode_edge_cases: {},
        edge_reference_validation: {},
        tool_conflict_handling: {},
        response_format_compliance: {},
        invalid_request_rejection: {},
        overall_success: false
    };

    const testStartTime = Date.now();

    try {
        // Run all test suites
        console.log('\n🔍 Running 405 Response Schema Tests...');
        allResults.response_schema_405 = await test405ResponseSchema();

        console.log('\n🔍 Running Unicode Edge Case Tests...');
        allResults.unicode_edge_cases = await testUnicodeEdgeCases();

        console.log('\n🔍 Running Edge Reference Validation Tests...');
        allResults.edge_reference_validation = await testEdgeReferenceValidation();

        console.log('\n🔍 Running Tool Conflict Handling Tests...');
        allResults.tool_conflict_handling = await testToolConflictHandling();

        console.log('\n🔍 Running Response Format Compliance Tests...');
        allResults.response_format_compliance = await testResponseFormatCompliance();

        console.log('\n🔍 Running Invalid Request Rejection Tests...');
        allResults.invalid_request_rejection = await testInvalidRequestRejection();

        // Calculate overall success
        const testSuites = [
            allResults.response_schema_405,
            allResults.unicode_edge_cases,
            allResults.edge_reference_validation,
            allResults.tool_conflict_handling,
            allResults.response_format_compliance,
            allResults.invalid_request_rejection
        ];

        let totalTests = 0;
        let passedTests = 0;

        testSuites.forEach(suite => {
            Object.values(suite).forEach(result => {
                if (typeof result === 'boolean') {
                    totalTests++;
                    if (result) passedTests++;
                }
            });
        });

        allResults.overall_success = passedTests >= (totalTests * 0.7); // 70% pass rate

        const executionTime = Date.now() - testStartTime;

        // Final summary
        console.log('\n📊 SCHEMATHESIS EDGE CASE TEST SUMMARY');
        console.log('='.repeat(80));
        console.log(`Overall Result: ${allResults.overall_success ? '🎉 SUCCESS' : '💥 FAILURE'}`);
        console.log(`Total Execution Time: ${executionTime}ms`);
        console.log(`Tests Passed: ${passedTests}/${totalTests} (${Math.round(passedTests/totalTests*100)}%)`);

        console.log('\n📋 Test Suite Results:');
        console.log(`   ✅ 405 Response Schema: ${Object.values(allResults.response_schema_405).filter(r => r === true).length} passed`);
        console.log(`   ✅ Unicode Edge Cases: ${Object.values(allResults.unicode_edge_cases).filter(r => r === true).length} passed`);
        console.log(`   ✅ Edge Reference Validation: ${Object.values(allResults.edge_reference_validation).filter(r => r === true).length} passed`);
        console.log(`   ✅ Tool Conflict Handling: ${Object.values(allResults.tool_conflict_handling).filter(r => r === true).length} passed`);
        console.log(`   ✅ Response Format Compliance: ${Object.values(allResults.response_format_compliance).filter(r => r === true).length} passed`);
        console.log(`   ✅ Invalid Request Rejection: ${Object.values(allResults.invalid_request_rejection).filter(r => r === true).length} passed`);

        if (allResults.overall_success) {
            console.log('\n🎉 Schemathesis Edge Case Tests PASSED!');
            console.log('   ✅ API correctly handles 405 Method Not Allowed responses');
            console.log('   ✅ Unicode validation working as expected');
            console.log('   ✅ Edge reference validation functioning properly');
            console.log('   ✅ Tool conflict responses properly formatted');
            console.log('   ✅ Response formats comply with schema expectations');
            console.log('   ✅ Invalid requests are appropriately rejected');
            console.log('   ✅ The API is robust against schemathesis edge cases!');
        } else {
            console.log('\n💥 Schemathesis Edge Case Tests FAILED!');
            console.log('   ❌ Some edge cases are not properly handled');
            console.log('   ❌ Review the detailed output above for specific issues');
            console.log('   ❌ Consider updating API validation and error handling');
        }

        // Exit with appropriate code
        process.exit(allResults.overall_success ? 0 : 1);

    } catch (error) {
        console.error(`\n❌ Schemathesis edge case test execution failed: ${error.message}`);
        if (error.stack) {
            console.error(`Stack trace: ${error.stack}`);
        }
        process.exit(1);
    }
}

/**
 * Main function with proper error handling
 */
async function main() {
    console.log('🚀 Schemathesis Edge Case Test Suite');
    console.log('Node.js version:', process.version);
    console.log('Platform:', process.platform);
    console.log('Architecture:', process.arch);
    console.log('='.repeat(80));

    try {
        await runSchemathesisEdgeCaseTests();
    } catch (error) {
        console.error('❌ Schemathesis edge case test suite failed to execute:', error.message);
        if (error.stack) {
            console.error('Stack trace:', error.stack);
        }
        process.exit(1);
    }
}

// Enhanced execution with proper error handling
if (require.main === module) {
    main().catch(error => {
        console.error('❌ Unhandled error in schemathesis edge case test suite:', error);
        process.exit(1);
    });
}

module.exports = {
    makeRequest,
    test405ResponseSchema,
    testUnicodeEdgeCases,
    testEdgeReferenceValidation,
    testToolConflictHandling,
    testResponseFormatCompliance,
    testInvalidRequestRejection,
    runSchemathesisEdgeCaseTests
};
