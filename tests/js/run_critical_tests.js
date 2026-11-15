#\!/usr/bin/env node
/**
 * Critical Production Readiness Tests
 * Tests core functionality and error handling
 */

const CONFIG = {
    API_BASE_URL: 'http://localhost:8887',
    TEST_TIMEOUT: 30000,
};

async function apiCall(endpoint, method = 'GET', body = null) {
    const options = {
        method,
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(CONFIG.TEST_TIMEOUT),
    };
    if (body) options.body = JSON.stringify(body);

    const response = await fetch(`${CONFIG.API_BASE_URL}${endpoint}`, options);
    const text = await response.text();

    return {
        status: response.status,
        data: text ? JSON.parse(text) : null,
        ok: response.ok
    };
}

async function runTests() {
    console.log('🚀 Production Readiness Test Suite');
    console.log('==================================\n');

    const results = { passed: 0, failed: 0, tests: [] };

    // Test 1: Health Check
    console.log('1️⃣  Testing Health Endpoint...');
    try {
        const health = await apiCall('/health');
        if (health.ok && health.data.status === 'healthy') {
            console.log('✅ Health check passed\n');
            results.passed++;
        } else {
            throw new Error('Health check failed');
        }
    } catch (e) {
        console.log('❌ Health check failed:', e.message, '\n');
        results.failed++;
    }

    // Test 2: Error Format Standardization
    console.log('2️⃣  Testing Error Response Standardization...');
    try {
        const notFound = await apiCall('/workers/non_existent');
        if (\!notFound.ok && notFound.data.error && notFound.data.error.code === 'NOT_FOUND') {
            console.log('✅ Error format standardized\n');
            results.passed++;
        } else {
            throw new Error('Error format not standardized');
        }
    } catch (e) {
        console.log('❌ Error format test failed:', e.message, '\n');
        results.failed++;
    }

    // Test 3: Worker Creation with Empty Tools
    console.log('3️⃣  Testing Worker Creation (empty tool_names)...');
    const workerName = `test_prod_${Date.now()}`;
    try {
        const worker = await apiCall('/workers', 'POST', {
            worker_name: workerName,
            agent_type: 'pydantic_ai',
            model_provider: 'deepseek',
            model: 'deepseek-chat',
            agent_role: 'general_assistant',
            instruction: 'You are a helpful assistant',
            tool_names: []
        });

        if (worker.ok && worker.data.name === workerName) {
            console.log('✅ Worker creation with empty tools passed\n');
            results.passed++;

            // Cleanup
            await apiCall(`/workers/${workerName}`, 'DELETE');
        } else {
            throw new Error('Worker creation failed');
        }
    } catch (e) {
        console.log('❌ Worker creation failed:', e.message, '\n');
        results.failed++;
    }

    // Test 4: Validation Error Handling
    console.log('4️⃣  Testing Validation Error Handling...');
    try {
        const invalid = await apiCall('/workers', 'POST', {
            worker_name: 'test@invalid\!name'
        });

        if (\!invalid.ok && invalid.data.error && invalid.data.error.code === 'VALIDATION_ERROR') {
            console.log('✅ Validation error handling passed\n');
            results.passed++;
        } else {
            throw new Error('Validation should have failed');
        }
    } catch (e) {
        console.log('❌ Validation error test failed:', e.message, '\n');
        results.failed++;
    }

    // Test 5: Graph Creation
    console.log('5️⃣  Testing Graph Creation...');
    try {
        const graph = await apiCall('/graphs/', 'POST', {
            name: `test_graph_${Date.now()}`,
            description: 'Production readiness test graph',
            nodes: [{
                id: 'start',
                name: 'start',
                node_type: 'function',
                function_name: 'test',
                data: {},
                timeout: 30
            }],
            edges: []
        });

        if (graph.ok && graph.data.id) {
            console.log('✅ Graph creation passed\n');
            results.passed++;

            // Cleanup
            await apiCall(`/graphs/${graph.data.id}`, 'DELETE');
        } else {
            throw new Error('Graph creation failed');
        }
    } catch (e) {
        console.log('❌ Graph creation failed:', e.message, '\n');
        results.failed++;
    }

    // Summary
    console.log('\n📊 Test Summary');
    console.log('===============');
    console.log(`Total tests: ${results.passed + results.failed}`);
    console.log(`✅ Passed: ${results.passed}`);
    console.log(`❌ Failed: ${results.failed}`);
    console.log(`Success rate: ${((results.passed / (results.passed + results.failed)) * 100).toFixed(1)}%`);

    if (results.failed === 0) {
        console.log('\n🎉 All critical tests passed\! API is production ready.');
    } else {
        console.log('\n⚠️  Some tests failed. API needs attention.');
        process.exit(1);
    }
}

runTests().catch(console.error);
