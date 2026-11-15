#!/usr/bin/env node
/**
 * Cleanup script to remove test workers and problematic entries from the API
 */

const API_BASE = 'http://localhost:8887';

async function makeRequest(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });

        if (response.status === 204) {
            return { success: true };
        }

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        return await response.json();
    } catch (error) {
        console.error(`Request failed: ${error.message}`);
        throw error;
    }
}

async function cleanup() {
    console.log('🧹 Starting cleanup of test workers and problematic entries...');

    try {
        // Get all workers
        const workers = await makeRequest(`${API_BASE}/workers`);
        console.log(`Found ${workers.length} workers`);

        let deletedCount = 0;
        let errorCount = 0;

        // Delete problematic workers - be more aggressive
        const allowedWorkers = ['data_processor']; // Keep only known good workers

        for (const workerName of workers) {
            try {
                const workerStr = workerName.toString();

                // Keep only explicitly allowed workers that are normal names
                const isAllowed = allowedWorkers.includes(workerStr) &&
                                 workerStr.length >= 3 &&
                                 /^[a-zA-Z0-9_-]+$/.test(workerStr);

                if (!isAllowed) {
                    console.log(`🗑️  Deleting problematic worker: ${workerName.toString().substring(0, 50)}...`);

                    const encodedName = encodeURIComponent(workerName);
                    await makeRequest(`${API_BASE}/workers/${encodedName}`, { method: 'DELETE' });
                    deletedCount++;
                }
            } catch (error) {
                console.error(`❌ Failed to delete worker: ${error.message}`);
                errorCount++;
            }
        }

        console.log(`✅ Cleanup completed: ${deletedCount} workers deleted, ${errorCount} errors`);

        // Verify cleanup
        const remainingWorkers = await makeRequest(`${API_BASE}/workers`);
        console.log(`📊 Remaining workers: ${remainingWorkers.length}`);

    } catch (error) {
        console.error(`❌ Cleanup failed: ${error.message}`);
        process.exit(1);
    }
}

cleanup().catch(error => {
    console.error('❌ Unhandled error:', error);
    process.exit(1);
});
