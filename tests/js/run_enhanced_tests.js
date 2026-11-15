#!/usr/bin/env node
/**
 * Enhanced Test Runner for Worker Management API
 *
 * This script runs the enhanced test suites with improved timing,
 * error handling, and debugging capabilities.
 */

const { execSync } = require('child_process');
const path = require('path');
const {
    TEST_CONFIG,
    waitForApiReady,
    aggregateTestResults
} = require('./test_config');

// Test configuration
const TESTS_TO_RUN = [
    {
        name: 'Test 05 - Worker States Management',
        file: 'test_05_worker_states.js',
        timeout: 180000, // 3 minutes
        critical: true
    },
    {
        name: 'Test 07 - Worker States API',
        file: 'test_07_worker_states_api.js',
        timeout: 180000, // 3 minutes
        critical: true
    },
    {
        name: 'Test 09 - Individual Graph Operations',
        file: 'test_09_graph_individual.js',
        timeout: 240000, // 4 minutes
        critical: true
    },
    {
        name: 'Test 10 - Graph Structure & Analysis',
        file: 'test_10_graph_structure.js',
        timeout: 300000, // 5 minutes
        critical: true
    },
    {
        name: 'Test 14 - Individual Tool Operations',
        file: 'test_14_tool_individual.js',
        timeout: 180000, // 3 minutes
        critical: true
    }
];

/**
 * Run a single test with timeout and error handling
 */
async function runSingleTest(test) {
    const testPath = path.join(__dirname, test.file);
    const startTime = Date.now();

    console.log(`\n${'='.repeat(80)}`);
    console.log(`🧪 Running ${test.name}`);
    console.log(`📁 File: ${test.file}`);
    console.log(`⏱️  Timeout: ${test.timeout / 1000}s`);
    console.log(`${'='.repeat(80)}\n`);

    try {
        // Run the test with timeout
        const result = execSync(`node "${testPath}"`, {
            timeout: test.timeout,
            encoding: 'utf8',
            stdio: 'pipe'
        });

        const executionTime = Date.now() - startTime;

        console.log(`\n✅ ${test.name} PASSED`);
        console.log(`⏱️  Execution time: ${executionTime}ms`);

        return {
            name: test.name,
            file: test.file,
            status: 'passed',
            executionTime,
            output: result
        };

    } catch (error) {
        const executionTime = Date.now() - startTime;

        console.error(`\n❌ ${test.name} FAILED`);
        console.error(`⏱️  Execution time: ${executionTime}ms`);
        console.error(`💥 Error: ${error.message}`);

        if (error.stdout) {
            console.error(`📤 STDOUT:\n${error.stdout}`);
        }
        if (error.stderr) {
            console.error(`📥 STDERR:\n${error.stderr}`);
        }

        return {
            name: test.name,
            file: test.file,
            status: 'failed',
            executionTime,
            error: error.message,
            stdout: error.stdout,
            stderr: error.stderr
        };
    }
}

/**
 * Run all tests sequentially with delays
 */
async function runAllTests() {
    console.log('🚀 Enhanced Worker Management API Test Suite');
    console.log(`📅 Started at: ${new Date().toLocaleString()}`);
    console.log(`🔧 API Base: ${TEST_CONFIG.API_BASE}`);
    console.log(`📊 Tests to run: ${TESTS_TO_RUN.length}`);
    console.log(`${'='.repeat(80)}\n`);

    const overallStartTime = Date.now();
    const results = [];

    try {
        // Initial API health check
        console.log('🏥 Performing initial API health check...');
        await waitForApiReady();
        console.log('✅ API is ready for testing\n');

        // Run tests sequentially with delays
        for (let i = 0; i < TESTS_TO_RUN.length; i++) {
            const test = TESTS_TO_RUN[i];

            console.log(`📋 Running test ${i + 1}/${TESTS_TO_RUN.length}`);
            const result = await runSingleTest(test);
            results.push(result);

            // Add delay between tests to prevent resource conflicts
            if (i < TESTS_TO_RUN.length - 1) {
                console.log(`\n⏳ Waiting ${TEST_CONFIG.DELAYS.BETWEEN_REQUESTS * 3}ms before next test...`);
                await new Promise(resolve => setTimeout(resolve, TEST_CONFIG.DELAYS.BETWEEN_REQUESTS * 3));
            }
        }

    } catch (error) {
        console.error(`\n💥 Test suite failed: ${error.message}`);
        results.push({
            name: 'Test Suite',
            status: 'failed',
            error: error.message
        });
    }

    // Generate summary report
    const overallExecutionTime = Date.now() - overallStartTime;
    generateSummaryReport(results, overallExecutionTime);

    // Return exit code
    const hasFailures = results.some(r => r.status === 'failed');
    const hasCriticalFailures = results.some(r => r.status === 'failed' &&
        TESTS_TO_RUN.find(t => t.file === r.file)?.critical);

    if (hasCriticalFailures) {
        console.log(`\n💥 CRITICAL TEST FAILURES - Exiting with code 1`);
        process.exit(1);
    } else if (hasFailures) {
        console.log(`\n⚠️  Some tests failed but no critical failures - Exiting with code 0`);
        process.exit(0);
    } else {
        console.log(`\n🎉 ALL TESTS PASSED - Exiting with code 0`);
        process.exit(0);
    }
}

/**
 * Generate summary report
 */
function generateSummaryReport(results, totalTime) {
    console.log(`\n${'='.repeat(80)}`);
    console.log('📊 ENHANCED TEST SUITE SUMMARY REPORT');
    console.log(`📅 Completed at: ${new Date().toLocaleString()}`);
    console.log(`⏱️  Total execution time: ${Math.round(totalTime / 1000)}s`);
    console.log(`${'='.repeat(80)}\n`);

    // Overall statistics
    const passed = results.filter(r => r.status === 'passed').length;
    const failed = results.filter(r => r.status === 'failed').length;
    const total = results.length;
    const successRate = total > 0 ? Math.round((passed / total) * 100) : 0;

    console.log(`📈 Overall Statistics:`);
    console.log(`   Total tests: ${total}`);
    console.log(`   Passed: ${passed}`);
    console.log(`   Failed: ${failed}`);
    console.log(`   Success rate: ${successRate}%`);
    console.log(`   Status: ${successRate >= 80 ? '✅ GOOD' : successRate >= 60 ? '⚠️  FAIR' : '❌ POOR'}\n`);

    // Individual test results
    console.log(`📋 Individual Test Results:`);
    results.forEach((result, index) => {
        const status = result.status === 'passed' ? '✅' : '❌';
        const time = result.executionTime ? `${Math.round(result.executionTime / 1000)}s` : 'N/A';
        console.log(`   ${index + 1}. ${status} ${result.name} (${time})`);

        if (result.status === 'failed' && result.error) {
            console.log(`      🔍 Error: ${result.error.substring(0, 100)}...`);
        }
    });

    // Critical failures
    const criticalFailures = results.filter(r =>
        r.status === 'failed' && TESTS_TO_RUN.find(t => t.file === r.file)?.critical
    );

    if (criticalFailures.length > 0) {
        console.log(`\n💥 Critical Failures:`);
        criticalFailures.forEach(failure => {
            console.log(`   - ${failure.name}: ${failure.error}`);
        });
    }

    // Performance analysis
    const avgTime = total > 0 ? Math.round(results.reduce((sum, r) => sum + (r.executionTime || 0), 0) / total) : 0;
    const slowTests = results.filter(r => r.executionTime > avgTime * 1.5);

    console.log(`\n⚡ Performance Analysis:`);
    console.log(`   Average test time: ${Math.round(avgTime / 1000)}s`);

    if (slowTests.length > 0) {
        console.log(`   Slow tests (>${Math.round(avgTime * 1.5 / 1000)}s):`);
        slowTests.forEach(test => {
            console.log(`     - ${test.name}: ${Math.round(test.executionTime / 1000)}s`);
        });
    }

    // Recommendations
    console.log(`\n💡 Recommendations:`);
    if (failed > 0) {
        console.log(`   - Review failed test logs for specific issues`);
        console.log(`   - Check API connectivity and database state`);
        console.log(`   - Verify test environment setup`);
    }
    if (slowTests.length > 0) {
        console.log(`   - Investigate slow tests for optimization opportunities`);
        console.log(`   - Consider increasing timeouts for complex operations`);
    }
    if (successRate >= 80) {
        console.log(`   - Great job! Tests are performing well`);
        console.log(`   - Consider adding more edge case coverage`);
    }
}

// Helper function for delays
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Run the test suite
if (require.main === module) {
    runAllTests().catch(error => {
        console.error('❌ Test runner failed:', error.message);
        process.exit(1);
    });
}

module.exports = {
    runAllTests,
    runSingleTest,
    generateSummaryReport
};
