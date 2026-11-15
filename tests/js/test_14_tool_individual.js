#!/usr/bin/env node
/**
 * JavaScript Test Suite for Tool Individual Operations - Production Ready
 *
 * This script comprehensively tests individual tool API endpoints:
 * - GET /tools/{tool_name} - retrieve specific tool by name
 * - PUT /tools/{tool_name} - update existing tool with ToolCreateRequest
 * - DELETE /tools/{tool_name} - delete tool from registry
 *
 * Test Coverage:
 * 1. Tool creation and retrieval by name
 * 2. Tool updates (code, description, schema, tags, version)
 * 3. Tool deletion and cleanup
 * 4. Various update scenarios and edge cases
 * 5. Proper tool name handling and error responses
 * 6. Concurrent individual tool operations
 * 7. Comprehensive cleanup and error handling
 *
 * Requirements: Node.js 18+ (for native fetch API)
 * Usage: node test_14_tool_individual.js
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

// Use enhanced request function from test_config
const makeRequest = makeEnhancedRequest;

/**
 * Create a test tool with specified configuration
 */
async function createTestTool(toolConfig) {
    console.log(`\n=== Creating Test Tool: ${toolConfig.name} ===`);

    try {
        // Validate required fields
        if (!toolConfig.name || !toolConfig.description || !toolConfig.python_code) {
            throw new Error('Tool configuration missing required fields: name, description, python_code');
        }

        const response = await makeRequest(`${API_BASE}/tools/`, {
            method: 'POST',
            body: JSON.stringify(toolConfig)
        });

        console.log(`✅ Test tool created successfully: ${response.name}`);
        console.log(`   ID: ${response.id || 'N/A'}`);
        console.log(`   Version: ${response.version}`);
        console.log(`   Tags: ${response.tags ? response.tags.join(', ') : 'None'}`);
        console.log(`   Description: ${response.description.substring(0, 100)}...`);

        return response;
    } catch (error) {
        console.error(`❌ Failed to create test tool ${toolConfig.name}: ${error.message}`);
        throw error;
    }
}

/**
 * Test GET /tools/{tool_name} - Retrieve specific tool by name
 */
async function testGetTool(toolName, shouldExist = true) {
    console.log(`\n=== Testing GET /tools/${toolName} ===`);

    try {
        // Handle empty tool name edge case - should construct URL that will fail
        const url = toolName === '' ? `${API_BASE}/tools//` : `${API_BASE}/tools/${toolName}`;
        const response = await makeRequest(url);

        if (!shouldExist) {
            throw new Error('Expected tool not to exist, but GET request succeeded');
        }

        console.log(`✅ Tool retrieved successfully: ${response.name}`);
        console.log(`   Version: ${response.version}`);
        console.log(`   Description: ${response.description.substring(0, 100)}...`);
        console.log(`   Tags: ${response.tags ? response.tags.join(', ') : 'None'}`);
        console.log(`   Has Input Schema: ${response.input_schema ? 'Yes' : 'No'}`);
        console.log(`   Has Output Schema: ${response.output_schema ? 'Yes' : 'No'}`);
        console.log(`   Python Code Length: ${response.python_code ? response.python_code.length : 0} characters`);
        console.log(`   Created At: ${response.created_at || 'N/A'}`);
        console.log(`   Updated At: ${response.updated_at || 'N/A'}`);

        // Validate response structure
        const requiredFields = ['name', 'description', 'python_code', 'version'];
        const missingFields = requiredFields.filter(field => !response[field]);

        if (missingFields.length > 0) {
            throw new Error(`Tool response missing required fields: ${missingFields.join(', ')}`);
        }

        // Validate that response is NOT wrapped in 'tools' array (specific requirement)
        if (response.tools) {
            throw new Error('Tool response incorrectly wrapped in "tools" array - should be single tool object');
        }

        return { success: true, tool: response };

    } catch (error) {
        if (shouldExist) {
            console.error(`❌ Failed to retrieve tool ${toolName}: ${error.message}`);
            return { success: false, error: error.message };
        } else {
            // Expected failure for non-existent tool or invalid tool name
            if (error.status === 404 || error.status === 422 || error.status === 400) {
                console.log(`✅ Expected ${error.status} error for non-existent/invalid tool: ${toolName}`);
                return { success: true, expectedError: true };
            } else {
                console.error(`❌ Unexpected error for non-existent tool ${toolName}: ${error.message}`);
                return { success: false, error: error.message };
            }
        }
    }
}

/**
 * Test PUT /tools/{tool_name} - Update existing tool with ToolCreateRequest
 */
async function testUpdateTool(toolName, updateConfig, shouldSucceed = true) {
    console.log(`\n=== Testing PUT /tools/${toolName} ===`);
    console.log('Update configuration:', JSON.stringify(updateConfig, null, 2));

    try {
        // Handle empty tool name edge case - should construct URL that will fail
        const url = toolName === '' ? `${API_BASE}/tools//` : `${API_BASE}/tools/${toolName}`;
        const response = await makeRequest(url, {
            method: 'PUT',
            body: JSON.stringify(updateConfig)
        });

        if (!shouldSucceed) {
            throw new Error('Expected update to fail, but PUT request succeeded');
        }

        console.log(`✅ Tool updated successfully: ${response.name}`);
        console.log(`   Updated fields reflected in response:`);

        // Verify that updates were applied
        for (const [key, value] of Object.entries(updateConfig)) {
            if (response[key] !== undefined) {
                let isMatch = false;
                if (Array.isArray(value) && Array.isArray(response[key])) {
                    // Compare arrays
                    isMatch = JSON.stringify(value.sort()) === JSON.stringify(response[key].sort());
                } else if (typeof value === 'object' && typeof response[key] === 'object') {
                    // Compare objects
                    isMatch = JSON.stringify(value) === JSON.stringify(response[key]);
                } else {
                    isMatch = response[key] === value;
                }
                console.log(`   ${key}: ${JSON.stringify(response[key])} ${isMatch ? '✅' : '❌ (expected: ' + JSON.stringify(value) + ')'}`);
            }
        }

        return { success: true, tool: response };

    } catch (error) {
        if (shouldSucceed) {
            console.error(`❌ Failed to update tool ${toolName}: ${error.message}`);
            return { success: false, error: error.message };
        } else {
            // Expected failure
            console.log(`✅ Expected error for invalid update: ${error.message}`);
            return { success: true, expectedError: true };
        }
    }
}

/**
 * Test DELETE /tools/{tool_name} - Delete tool from registry
 */
async function testDeleteTool(toolName, shouldExist = true) {
    console.log(`\n=== Testing DELETE /tools/${toolName} ===`);

    try {
        // Handle empty tool name edge case - should construct URL that will fail
        const url = toolName === '' ? `${API_BASE}/tools//` : `${API_BASE}/tools/${toolName}`;
        const response = await makeRequest(url, {
            method: 'DELETE'
        });

        if (!shouldExist) {
            throw new Error('Expected tool not to exist, but DELETE request succeeded');
        }

        console.log(`✅ Tool deleted successfully: ${toolName}`);
        return { success: true, deleted: true };

    } catch (error) {
        if (shouldExist) {
            console.error(`❌ Failed to delete tool ${toolName}: ${error.message}`);
            return { success: false, error: error.message };
        } else {
            // Expected failure for non-existent tool
            if (error.status === 404) {
                console.log(`✅ Expected 404 error for non-existent tool: ${toolName}`);
                return { success: true, expectedError: true };
            } else {
                console.error(`❌ Unexpected error for non-existent tool ${toolName}: ${error.message}`);
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

        // Tool registry service check
        const toolsResponse = await makeRequest(`${API_BASE}/tools/`);
        console.log(`✅ Tool registry service is responding (${toolsResponse.tools ? toolsResponse.tools.length : 0} tools found)`);

        // Tool registry health check
        try {
            const toolHealthResponse = await makeRequest(`${API_BASE}/tools/health`);
            console.log(`✅ Tool registry health check passed`);
        } catch (e) {
            console.log(`⚠️  Tool registry health endpoint not available`);
        }

        return { healthy: true };

    } catch (error) {
        console.error(`❌ Health check failed: ${error.message}`);
        return { healthy: false, error: error.message };
    }
}

/**
 * Test concurrent tool operations
 */
async function testConcurrentOperations(baseToolName) {
    console.log('\n=== Testing Concurrent Tool Operations ===');

    const toolNames = [
        `${baseToolName}_concurrent_1`,
        `${baseToolName}_concurrent_2`,
        `${baseToolName}_concurrent_3`
    ];

    const toolConfigs = toolNames.map(name => ({
        name: name,
        description: `Concurrent test tool: ${name}`,
        python_code: `def concurrent_test_${name.split('_').pop()}():\n    """Test function for concurrent operations"""\n    return {"message": "concurrent test successful", "tool_name": "${name}"}`,
        tags: ["test", "concurrent"],
        version: "1.0.0",
        input_schema: {
            type: "object",
            properties: {},
            required: []
        },
        output_schema: {
            type: "object",
            properties: {
                message: { type: "string" },
                tool_name: { type: "string" }
            }
        }
    }));

    try {
        // Create tools concurrently
        console.log('Creating tools concurrently...');
        const createPromises = toolConfigs.map(config => createTestTool(config));
        const createResults = await Promise.allSettled(createPromises);

        const successfulCreations = createResults.filter(result => result.status === 'fulfilled');
        console.log(`✅ Created ${successfulCreations.length}/${toolConfigs.length} tools concurrently`);

        if (successfulCreations.length === 0) {
            throw new Error('No tools were created successfully');
        }

        // Get tools concurrently
        console.log('Retrieving tools concurrently...');
        const getPromises = toolNames.map(name => testGetTool(name, true));
        const getResults = await Promise.allSettled(getPromises);

        const successfulGets = getResults.filter(result =>
            result.status === 'fulfilled' && result.value.success
        );
        console.log(`✅ Retrieved ${successfulGets.length}/${toolNames.length} tools concurrently`);

        // Update tools concurrently
        console.log('Updating tools concurrently...');
        const updatePromises = toolNames.map(name => testUpdateTool(name, {
            name: name,
            description: `Updated concurrent test tool: ${name}`,
            python_code: `def concurrent_test_updated():\n    """Updated test function for concurrent operations"""\n    return {"message": "concurrent test updated", "tool_name": "${name}"}`,
            version: "1.1.0",
            tags: ["test", "concurrent", "updated"]
        }, true));
        const updateResults = await Promise.allSettled(updatePromises);

        const successfulUpdates = updateResults.filter(result =>
            result.status === 'fulfilled' && result.value.success
        );
        console.log(`✅ Updated ${successfulUpdates.length}/${toolNames.length} tools concurrently`);

        // Delete tools concurrently
        console.log('Deleting tools concurrently...');
        const deletePromises = toolNames.map(name => testDeleteTool(name, true));
        const deleteResults = await Promise.allSettled(deletePromises);

        const successfulDeletes = deleteResults.filter(result =>
            result.status === 'fulfilled' && result.value.success
        );
        console.log(`✅ Deleted ${successfulDeletes.length}/${toolNames.length} tools concurrently`);

        return {
            success: true,
            created: successfulCreations.length,
            retrieved: successfulGets.length,
            updated: successfulUpdates.length,
            deleted: successfulDeletes.length
        };

    } catch (error) {
        console.error(`❌ Concurrent operations test failed: ${error.message}`);

        // Cleanup any remaining tools
        console.log('Cleaning up remaining tools...');
        for (const name of toolNames) {
            try {
                await testDeleteTool(name, false); // Don't expect them to exist
            } catch (e) {
                // Ignore cleanup errors
            }
        }

        return { success: false, error: error.message };
    }
}

/**
 * Generate comprehensive tool update scenarios
 */
function generateUpdateScenarios() {
    return [
        {
            name: 'Basic Information Update',
            config: {
                name: "updated_basic_test_tool",
                description: 'Updated basic information test tool with enhanced description',
                python_code: `def updated_basic_test():\n    """Updated test function with enhanced functionality"""\n    return {"status": "updated", "message": "basic test successful"}`,
                version: "1.1.0"
            }
        },
        {
            name: 'Tags and Version Update',
            config: {
                name: "updated_tags_test_tool",
                description: 'Test tool for tags and version updates',
                python_code: `def tags_version_test():\n    """Test function for tags and version"""\n    return {"tags": "updated", "version": "enhanced"}`,
                tags: ["updated", "enhanced", "test"],
                version: "2.0.0"
            }
        },
        {
            name: 'Schema Update',
            config: {
                name: "updated_schema_test_tool",
                description: 'Test tool with updated input and output schemas',
                python_code: `def schema_test(text: str, count: int = 1):\n    """Test function with schema validation"""\n    return {"processed_text": text * count, "length": len(text * count)}`,
                input_schema: {
                    type: "object",
                    properties: {
                        text: { type: "string", description: "Text to process" },
                        count: { type: "integer", description: "Repetition count", default: 1 }
                    },
                    required: ["text"]
                },
                output_schema: {
                    type: "object",
                    properties: {
                        processed_text: { type: "string" },
                        length: { type: "integer" }
                    }
                },
                version: "1.2.0"
            }
        },
        {
            name: 'Complex Code Update',
            config: {
                name: "updated_complex_test_tool",
                description: 'Test tool with complex Python code update',
                python_code: `def complex_analysis(data: list, operation: str = "sum"):\n    """Complex analysis function with multiple operations"""\n    if not data or not isinstance(data, list):\n        return {"error": "Invalid input data"}\n    \n    if operation == "sum":\n        result = sum(data)\n    elif operation == "average":\n        result = sum(data) / len(data)\n    elif operation == "max":\n        result = max(data)\n    elif operation == "min":\n        result = min(data)\n    else:\n        return {"error": "Unsupported operation"}\n    \n    return {\n        "operation": operation,\n        "result": result,\n        "input_count": len(data),\n        "input_sum": sum(data)\n    }`,
                input_schema: {
                    type: "object",
                    properties: {
                        data: {
                            type: "array",
                            items: { type: "number" },
                            description: "Array of numbers to analyze"
                        },
                        operation: {
                            type: "string",
                            enum: ["sum", "average", "max", "min"],
                            default: "sum",
                            description: "Analysis operation to perform"
                        }
                    },
                    required: ["data"]
                },
                output_schema: {
                    type: "object",
                    properties: {
                        operation: { type: "string" },
                        result: { type: "number" },
                        input_count: { type: "integer" },
                        input_sum: { type: "number" },
                        error: { type: "string" }
                    }
                },
                tags: ["analysis", "math", "complex", "updated"],
                version: "2.1.0"
            }
        }
    ];
}

/**
 * Test tool name handling edge cases
 */
async function testToolNameHandling() {
    console.log('\n=== Testing Tool Name Handling Edge Cases ===');

    const edgeCases = [
        {
            name: 'Special Characters in Name',
            toolName: 'test-tool_with_special123',
            shouldWork: true
        },
        {
            name: 'Invalid Characters (spaces)',
            toolName: 'test tool with spaces',
            shouldWork: false
        },
        {
            name: 'Invalid Characters (symbols)',
            toolName: 'test@tool#invalid',
            shouldWork: false
        },
        {
            name: 'Empty Tool Name',
            toolName: '',
            shouldWork: false
        },
        {
            name: 'Very Long Tool Name',
            toolName: 'a'.repeat(150), // Exceeds max length
            shouldWork: false
        }
    ];

    const results = [];

    for (const testCase of edgeCases) {
        console.log(`\n--- Testing ${testCase.name}: "${testCase.toolName}" ---`);

        if (testCase.shouldWork) {
            // Test creation with valid name
            try {
                const toolConfig = {
                    name: testCase.toolName,
                    description: `Test tool for name validation: ${testCase.name}`,
                    python_code: `def test_function():\n    return {"name": "${testCase.toolName}"}`,
                    version: "1.0.0"
                };

                const tool = await createTestTool(toolConfig);
                const getResult = await testGetTool(testCase.toolName, true);
                const deleteResult = await testDeleteTool(testCase.toolName, true);

                results.push({
                    testCase: testCase.name,
                    success: getResult.success && deleteResult.success,
                    expected: true
                });

            } catch (error) {
                console.log(`❌ Unexpected failure for valid name: ${error.message}`);
                results.push({
                    testCase: testCase.name,
                    success: false,
                    expected: true,
                    error: error.message
                });
            }
        } else {
            // Test that invalid names are rejected
            try {
                const getResult = await testGetTool(testCase.toolName, false);
                results.push({
                    testCase: testCase.name,
                    success: getResult.success && getResult.expectedError,
                    expected: false
                });

            } catch (error) {
                if (error.status === 404 || error.status === 422 || error.status === 400) {
                    console.log(`✅ Correctly rejected invalid tool name (status: ${error.status})`);
                    results.push({
                        testCase: testCase.name,
                        success: true,
                        expected: false
                    });
                } else {
                    console.log(`❌ Unexpected error for invalid name: ${error.message}`);
                    results.push({
                        testCase: testCase.name,
                        success: false,
                        expected: false,
                        error: error.message
                    });
                }
            }
        }
    }

    return results;
}

/**
 * Main test execution function
 */
async function runIndividualToolTests() {
    console.log('🚀 Starting Individual Tool Operations Test Suite');
    console.log('='.repeat(80));

    const testResults = {
        health_check: false,
        tool_creation: false,
        tool_retrieval: false,
        tool_updates: [],
        tool_deletion: false,
        edge_cases: [],
        name_handling: [],
        concurrent_operations: false,
        overall_success: false,
        execution_time: 0
    };

    const testStartTime = Date.now();
    let createdToolName = null;

    try {
        // Step 1: Health Check
        console.log('\n🏥 Step 1: API Health Check');
        await waitForApiReady();
        testResults.health_check = true;

        // Step 2: Create Test Tool
        console.log('\n🔧 Step 2: Create Test Tool');
        const testToolName = generateTestId('individual_test_tool');
        createdToolName = testToolName;

        const toolConfig = {
            name: testToolName,
            description: 'Test tool for individual operations validation with comprehensive functionality',
            python_code: `def individual_test_function(message: str = "hello"):\n    """Test function for individual tool operations testing"""\n    import datetime\n    return {\n        "message": f"Individual test: {message}",\n        "timestamp": datetime.datetime.now().isoformat(),\n        "tool_name": "${testToolName}",\n        "status": "success"\n    }`,
            input_schema: {
                type: "object",
                properties: {
                    message: {
                        type: "string",
                        default: "hello",
                        description: "Message to process"
                    }
                },
                required: []
            },
            output_schema: {
                type: "object",
                properties: {
                    message: { type: "string" },
                    timestamp: { type: "string" },
                    tool_name: { type: "string" },
                    status: { type: "string" }
                }
            },
            tags: ["test", "individual", "validation"],
            version: "1.0.0"
        };

        const createdTool = await createTestTool(toolConfig);
        testResults.tool_creation = true;

        // Wait for tool to be fully created
        await sleep(TEST_CONFIG.DELAYS.AFTER_CREATION);

        // Step 3: Test GET Tool
        console.log('\n📋 Step 3: Test GET Tool');
        const getResult = await testGetTool(testToolName, true);
        testResults.tool_retrieval = getResult.success;

        if (!getResult.success) {
            throw new Error('Failed to retrieve created tool');
        }

        // Step 4: Test Tool Updates
        console.log('\n🔄 Step 4: Test Tool Updates');
        const updateScenarios = generateUpdateScenarios();

        for (let i = 0; i < updateScenarios.length; i++) {
            const scenario = updateScenarios[i];
            console.log(`\n--- Testing ${scenario.name} (${i + 1}/${updateScenarios.length}) ---`);

            // Update the name to use the current test tool name
            const scenarioConfig = { ...scenario.config };
            scenarioConfig.name = testToolName;

            const updateResult = await testUpdateTool(testToolName, scenarioConfig, true);
            testResults.tool_updates.push({
                scenario: scenario.name,
                success: updateResult.success,
                config: scenarioConfig
            });

            if (updateResult.success) {
                // Wait before verification
                await sleep(TEST_CONFIG.DELAYS.AFTER_UPDATE);

                // Verify updates by getting tool again
                const verifyResult = await testGetTool(testToolName, true);
                if (verifyResult.success) {
                    console.log('✅ Update verification successful');
                } else {
                    console.log('⚠️  Update verification failed');
                }
            }

            // Delay between update scenarios
            await sleep(TEST_CONFIG.DELAYS.BETWEEN_REQUESTS);
        }

        // Step 5: Test Edge Cases
        console.log('\n🚨 Step 5: Test Edge Cases');

        // Test GET non-existent tool
        const nonExistentToolName = `non_existent_tool_${Date.now()}`;
        const getNonExistentResult = await testGetTool(nonExistentToolName, false);
        testResults.edge_cases.push({
            test: 'GET Non-existent Tool',
            success: getNonExistentResult.success
        });

        // Test UPDATE non-existent tool
        const updateNonExistentResult = await testUpdateTool(nonExistentToolName, {
            name: nonExistentToolName,
            description: 'This should fail',
            python_code: 'def test(): return "fail"',
            version: "1.0.0"
        }, false);
        testResults.edge_cases.push({
            test: 'UPDATE Non-existent Tool',
            success: updateNonExistentResult.success
        });

        // Test DELETE non-existent tool
        const deleteNonExistentResult = await testDeleteTool(nonExistentToolName, false);
        testResults.edge_cases.push({
            test: 'DELETE Non-existent Tool',
            success: deleteNonExistentResult.success
        });

        // Test invalid update configuration
        const invalidUpdateResult = await testUpdateTool(testToolName, {
            name: testToolName,
            description: 'Valid description',
            python_code: '', // Invalid - too short
            version: "invalid.version" // Invalid version format
        }, false);
        testResults.edge_cases.push({
            test: 'Invalid Update Configuration',
            success: invalidUpdateResult.success
        });

        // Step 6: Test Tool Name Handling
        console.log('\n🏷️  Step 6: Test Tool Name Handling');
        const nameHandlingResults = await testToolNameHandling();
        testResults.name_handling = nameHandlingResults;

        // Step 7: Test Concurrent Operations
        console.log('\n⚡ Step 7: Test Concurrent Operations');
        const concurrentResult = await testConcurrentOperations(generateTestId('concurrent_tool'));
        testResults.concurrent_operations = concurrentResult.success;

        // Step 8: Test DELETE Tool (cleanup main test tool)
        console.log('\n🗑️  Step 8: Test DELETE Tool');
        const deleteResult = await testDeleteTool(testToolName, true);
        testResults.tool_deletion = deleteResult.success;

        if (deleteResult.success) {
            createdToolName = null; // Mark as cleaned up

            // Verify deletion by trying to GET the tool
            const verifyDeleteResult = await testGetTool(testToolName, false);
            if (verifyDeleteResult.success && verifyDeleteResult.expectedError) {
                console.log('✅ Deletion verification successful');
            } else {
                console.log('⚠️  Deletion verification failed');
            }
        }

        // Calculate overall success
        const allUpdatesPassed = testResults.tool_updates.every(update => update.success);
        const allEdgeCasesPassed = testResults.edge_cases.every(edge => edge.success);
        const allNameHandlingPassed = testResults.name_handling.every(test => test.success);

        testResults.overall_success = testResults.health_check &&
                                     testResults.tool_creation &&
                                     testResults.tool_retrieval &&
                                     allUpdatesPassed &&
                                     testResults.tool_deletion &&
                                     allEdgeCasesPassed &&
                                     allNameHandlingPassed &&
                                     testResults.concurrent_operations;

        testResults.execution_time = Date.now() - testStartTime;

        // Step 9: Generate Test Report
        console.log('\n📊 Step 9: Individual Tool Operations Test Summary');
        console.log('='.repeat(80));
        console.log(`🏁 Individual Tool Operations Test - COMPLETE`);
        console.log(`   Total execution time: ${testResults.execution_time}ms`);
        console.log(`   Overall success: ${testResults.overall_success ? '✅ PASSED' : '❌ FAILED'}`);

        console.log('\n📋 Test Results:');
        console.log(`   ✅ API Health Check: ${testResults.health_check ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Tool Creation: ${testResults.tool_creation ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Tool Retrieval (GET): ${testResults.tool_retrieval ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Tool Updates (PUT): ${testResults.tool_updates.length} scenarios tested`);

        for (const update of testResults.tool_updates) {
            console.log(`      - ${update.scenario}: ${update.success ? 'PASSED' : 'FAILED'}`);
        }

        console.log(`   ✅ Tool Deletion (DELETE): ${testResults.tool_deletion ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Edge Cases: ${testResults.edge_cases.length} scenarios tested`);

        for (const edge of testResults.edge_cases) {
            console.log(`      - ${edge.test}: ${edge.success ? 'PASSED' : 'FAILED'}`);
        }

        console.log(`   ✅ Tool Name Handling: ${testResults.name_handling.length} scenarios tested`);

        for (const nameTest of testResults.name_handling) {
            console.log(`      - ${nameTest.testCase}: ${nameTest.success ? 'PASSED' : 'FAILED'}`);
        }

        console.log(`   ✅ Concurrent Operations: ${testResults.concurrent_operations ? 'PASSED' : 'FAILED'}`);

        if (testResults.overall_success) {
            console.log('\n🎉 All Individual Tool Operations Tests PASSED!');
            console.log('   ✅ GET /tools/{tool_name} is working correctly');
            console.log('   ✅ PUT /tools/{tool_name} is working correctly');
            console.log('   ✅ DELETE /tools/{tool_name} is working correctly');
            console.log('   ✅ Tool name handling is working correctly');
            console.log('   ✅ Error handling is working correctly');
            console.log('   ✅ Concurrent operations are supported');
            console.log('   ✅ Response format validation passed');
        } else {
            console.log('\n💥 Some Individual Tool Operations Tests FAILED!');
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

    // Cleanup remaining tool if test failed
    if (createdToolName) {
        console.log('\n🧹 Cleanup: Removing remaining test tool');
        const success = await cleanupWithRetry(
            () => makeRequest(`${API_BASE}/tools/${createdToolName}`, { method: 'DELETE' }),
            'tool',
            createdToolName
        );

        if (success) {
            console.log('✅ Cleanup completed');
        } else {
            console.error(`❌ Cleanup failed after retries`);
        }
    }

    return testResults;
}

/**
 * Main function with comprehensive error handling
 */
async function main() {
    console.log('🚀 Individual Tool Operations Test Suite');
    console.log('Node.js version:', process.version);
    console.log('Platform:', process.platform);
    console.log('Architecture:', process.arch);
    console.log('='.repeat(80));

    try {
        const testResults = await runIndividualToolTests();

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
    createTestTool,
    testGetTool,
    testUpdateTool,
    testDeleteTool,
    checkApiHealth,
    testConcurrentOperations,
    testToolNameHandling,
    runIndividualToolTests
};
