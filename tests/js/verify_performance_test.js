#!/usr/bin/env node
/**
 * Quick verification script for performance test setup
 */

const API_BASE = 'http://localhost:8887';

async function verifySetup() {
    console.log('🔍 Verifying Performance Test Setup\n');

    try {
        // Check API health
        console.log('1. Checking API health...');
        const healthResponse = await fetch(`${API_BASE}/health`);
        const healthData = await healthResponse.json();
        console.log(`   ✅ API Status: ${healthData.status}`);

        // Check endpoints
        const endpoints = ['/workers', '/graphs/', '/tools/', '/workers_states'];
        console.log('\n2. Checking API endpoints...');

        for (const endpoint of endpoints) {
            try {
                const response = await fetch(`${API_BASE}${endpoint}`);
                console.log(`   ✅ ${endpoint} - Status: ${response.status}`);
            } catch (error) {
                console.log(`   ❌ ${endpoint} - Error: ${error.message}`);
            }
        }

        // Check worker creation capability
        console.log('\n3. Testing worker creation...');
        const testWorker = `verify_test_${Date.now()}`;
        const createResponse = await fetch(`${API_BASE}/workers`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                worker_name: testWorker,
                tool_names: ['example_text_analyzer'],
                description: 'Verification test worker'
            })
        });

        if (createResponse.ok) {
            console.log('   ✅ Worker creation successful');

            // Cleanup
            await fetch(`${API_BASE}/workers/${testWorker}`, { method: 'DELETE' });
            console.log('   ✅ Worker cleanup successful');
        } else {
            const error = await createResponse.text();
            console.log(`   ❌ Worker creation failed: ${error}`);
        }

        console.log('\n✅ Setup verification complete!');
        console.log('\nYou can now run the performance test:');
        console.log('   node test_17_performance_load.js');

    } catch (error) {
        console.error('\n❌ Setup verification failed:', error.message);
        console.log('\nPlease ensure the API is running on port 8887');
    }
}

verifySetup();
