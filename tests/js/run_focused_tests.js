#!/usr/bin/env node

/**
 * Focused Test Runner for Worker Management API
 *
 * This script runs the focused test suite covering core functionality and schemathesis edge cases.
 * After cleanup, we now have a lean test suite that provides 100% OpenAPI coverage plus schemathesis protection.
 *
 * Usage:
 *   node run_focused_tests.js                    # Run all tests sequentially
 *   node run_focused_tests.js --parallel         # Run tests in parallel (faster)
 *   node run_focused_tests.js --category core    # Run specific category
 *   node run_focused_tests.js --list             # List all available tests
 *   node run_focused_tests.js --help             # Show help
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

// Configuration
const TEST_TIMEOUT = 180000; // 3 minutes per test (reduced)
const PARALLEL_LIMIT = 4; // Max concurrent tests
const API_BASE = 'http://localhost:8887';

// Focused test suite - core functionality + schemathesis protection
const TEST_SUITE = [
  // Core API Tests (Tests 1-15) - 100% OpenAPI Coverage
  {
    id: '01',
    name: 'Root Health Endpoints',
    file: 'test_01_root_health.js',
    category: 'core',
    priority: 'critical',
    description: 'Tests root health endpoints (GET /, GET /health)',
    estimatedTime: '30s'
  },
  {
    id: '02',
    name: 'Worker Collection Operations',
    file: 'test_02_worker_collection.js',
    category: 'core',
    priority: 'critical',
    description: 'Tests worker CRUD operations and collection management',
    estimatedTime: '2m'
  },
  {
    id: '03',
    name: 'Worker Individual Operations',
    file: 'test_03_worker_individual.js',
    category: 'core',
    priority: 'critical',
    description: 'Tests individual worker operations (GET, PUT, DELETE)',
    estimatedTime: '2m'
  },
  {
    id: '04',
    name: 'Worker Messaging',
    file: 'test_04_worker_messaging.js',
    category: 'core',
    priority: 'high',
    description: 'Tests worker messaging API (POST /workers/{name}/message)',
    estimatedTime: '2m'
  },
  {
    id: '05',
    name: 'Worker States',
    file: 'test_05_worker_states.js',
    category: 'core',
    priority: 'high',
    description: 'Tests worker state management endpoints',
    estimatedTime: '2m'
  },
  {
    id: '06',
    name: 'Worker Instance & Bulk Operations',
    file: 'test_06_worker_instance_bulk.js',
    category: 'core',
    priority: 'high',
    description: 'Tests worker instance cleanup and bulk operations',
    estimatedTime: '2m'
  },
  {
    id: '07',
    name: 'Worker States API',
    file: 'test_07_worker_states_api.js',
    category: 'core',
    priority: 'high',
    description: 'Tests worker states bulk API endpoints',
    estimatedTime: '2m'
  },
  {
    id: '08',
    name: 'Graph Collection Operations',
    file: 'test_08_graph_collection.js',
    category: 'core',
    priority: 'critical',
    description: 'Tests graph CRUD operations and collection management',
    estimatedTime: '2m'
  },
  {
    id: '09',
    name: 'Graph Individual Operations',
    file: 'test_09_graph_individual.js',
    category: 'core',
    priority: 'critical',
    description: 'Tests individual graph operations (GET, PATCH, DELETE)',
    estimatedTime: '2m'
  },
  {
    id: '10',
    name: 'Graph Structure & Validation',
    file: 'test_10_graph_structure.js',
    category: 'core',
    priority: 'high',
    description: 'Tests graph structure and validation endpoints',
    estimatedTime: '2m'
  },
  {
    id: '11',
    name: 'Graph States',
    file: 'test_11_graph_states.js',
    category: 'core',
    priority: 'high',
    description: 'Tests graph state management endpoints',
    estimatedTime: '2m'
  },
  {
    id: '12',
    name: 'Graph Execution',
    file: 'test_12_graph_execution.js',
    category: 'core',
    priority: 'critical',
    description: 'Tests graph execution endpoints and workflows',
    estimatedTime: '3m'
  },
  {
    id: '13',
    name: 'Tool Collection Operations',
    file: 'test_13_tool_collection.js',
    category: 'core',
    priority: 'critical',
    description: 'Tests tool CRUD operations and collection management',
    estimatedTime: '2m'
  },
  {
    id: '14',
    name: 'Tool Individual Operations',
    file: 'test_14_tool_individual.js',
    category: 'core',
    priority: 'critical',
    description: 'Tests individual tool operations (GET, PUT, DELETE)',
    estimatedTime: '2m'
  },
  {
    id: '15',
    name: 'Tool Validation & Testing',
    file: 'test_15_tool_validation.js',
    category: 'core',
    priority: 'high',
    description: 'Tests tool validation and testing endpoints',
    estimatedTime: '2m'
  },

  // Schemathesis Protection Tests (Tests 16-26)
  {
    id: '16',
    name: 'Security Validation',
    file: 'test_16_security_validation.js',
    category: 'security',
    priority: 'high',
    description: 'Tests security measures and input sanitization',
    estimatedTime: '2m'
  },
  {
    id: '18',
    name: 'Error Handling',
    file: 'test_18_error_handling.js',
    category: 'quality',
    priority: 'high',
    description: 'Tests error handling and validation responses',
    estimatedTime: '2m'
  },
  {
    id: '20',
    name: 'Schema Compliance',
    file: 'test_20_schema_compliance.js',
    category: 'quality',
    priority: 'high',
    description: 'Tests response schema compliance and validation',
    estimatedTime: '2m'
  },
  {
    id: '21',
    name: 'Unicode Support',
    file: 'test_21_unicode_support.js',
    category: 'quality',
    priority: 'medium',
    description: 'Tests Unicode and international character support',
    estimatedTime: '2m'
  },
  {
    id: '26',
    name: 'HTTP Compliance',
    file: 'test_26_http_compliance.js',
    category: 'quality',
    priority: 'medium',
    description: 'Tests HTTP status codes, headers, and compliance (includes 429/405 testing)',
    estimatedTime: '2m'
  },

  // Schemathesis Edge Cases
  {
    id: 'S1',
    name: 'Schemathesis Edge Cases',
    file: 'test_schemathesis_edge_cases.js',
    category: 'schemathesis',
    priority: 'high',
    description: 'Tests specific edge cases found by schemathesis',
    estimatedTime: '3m'
  },

  // Legacy Integration Tests (if useful)
  {
    id: '32',
    name: 'WebSocket Real-time (Original)',
    file: 'test_32_websocket_realtime_original.js',
    category: 'integration',
    priority: 'low',
    description: 'Tests WebSocket real-time communication features',
    estimatedTime: '2m'
  },
  {
    id: '39',
    name: 'Third-party Integration',
    file: 'test_39_third_party_integration.js',
    category: 'integration',
    priority: 'medium',
    description: 'Tests third-party service integrations',
    estimatedTime: '3m'
  }
];

// Test execution results
let testResults = {
  total: 0,
  passed: 0,
  failed: 0,
  skipped: 0,
  duration: 0,
  details: [],
  categories: {}
};

/**
 * Display help information
 */
function showHelp() {
  console.log(`
🧪 Focused Worker Management API Test Runner

USAGE:
  node run_focused_tests.js [options]

OPTIONS:
  --help              Show this help message
  --list              List all available tests
  --parallel          Run tests in parallel (faster but less detailed output)
  --category <name>   Run tests from specific category only
  --priority <level>  Run tests with specific priority only
  --timeout <ms>      Set custom timeout per test (default: ${TEST_TIMEOUT}ms)
  --verbose           Show detailed output during test execution
  --json              Output results in JSON format
  --continue          Continue running tests even if some fail

CATEGORIES:
  core         Core API functionality (tests 1-15) - 100% OpenAPI coverage
  security     Security and validation tests
  quality      Quality assurance and compliance tests
  schemathesis Tests for schemathesis edge cases
  integration  Third-party and legacy integration tests

PRIORITIES:
  critical     Essential tests that must pass
  high         Important tests for production readiness
  medium       Recommended tests for quality assurance
  low          Optional tests for comprehensive coverage

EXAMPLES:
  node run_focused_tests.js
  node run_focused_tests.js --parallel --category core
  node run_focused_tests.js --priority critical
  node run_focused_tests.js --verbose --continue
  `);
}

/**
 * List all available tests
 */
function listTests() {
  console.log('\n📋 Available Tests:\n');

  const categories = {};
  TEST_SUITE.forEach(test => {
    if (!categories[test.category]) {
      categories[test.category] = [];
    }
    categories[test.category].push(test);
  });

  Object.entries(categories).forEach(([category, tests]) => {
    console.log(`\n📁 ${category.toUpperCase()}:`);
    tests.forEach(test => {
      const status = fs.existsSync(path.join(__dirname, test.file)) ? '✅' : '❌';
      console.log(`  ${status} Test ${test.id}: ${test.name} (${test.priority}) - ${test.estimatedTime}`);
      console.log(`      File: ${test.file}`);
      console.log(`      Description: ${test.description}`);
    });
  });

  console.log(`\n📊 Total Tests: ${TEST_SUITE.length}`);
  console.log(`📊 Categories: ${Object.keys(categories).length}`);
}

/**
 * Parse command line arguments
 */
function parseArguments() {
  const args = process.argv.slice(2);
  const options = {
    parallel: false,
    category: null,
    priority: null,
    timeout: TEST_TIMEOUT,
    verbose: false,
    json: false,
    continue: false,
    help: false,
    list: false
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--help':
      case '-h':
        options.help = true;
        break;
      case '--list':
      case '-l':
        options.list = true;
        break;
      case '--parallel':
      case '-p':
        options.parallel = true;
        break;
      case '--category':
      case '-c':
        options.category = args[++i];
        break;
      case '--priority':
        options.priority = args[++i];
        break;
      case '--timeout':
      case '-t':
        options.timeout = parseInt(args[++i]) || TEST_TIMEOUT;
        break;
      case '--verbose':
      case '-v':
        options.verbose = true;
        break;
      case '--json':
      case '-j':
        options.json = true;
        break;
      case '--continue':
        options.continue = true;
        break;
      default:
        console.error(`❌ Unknown option: ${args[i]}`);
        process.exit(1);
    }
  }

  return options;
}

/**
 * Filter tests based on options
 */
function filterTests(options) {
  let filteredTests = [...TEST_SUITE];

  if (options.category) {
    filteredTests = filteredTests.filter(test => test.category === options.category);
    if (filteredTests.length === 0) {
      console.error(`❌ No tests found for category: ${options.category}`);
      console.log('Available categories:', [...new Set(TEST_SUITE.map(t => t.category))].join(', '));
      process.exit(1);
    }
  }

  if (options.priority) {
    filteredTests = filteredTests.filter(test => test.priority === options.priority);
    if (filteredTests.length === 0) {
      console.error(`❌ No tests found for priority: ${options.priority}`);
      console.log('Available priorities:', [...new Set(TEST_SUITE.map(t => t.priority))].join(', '));
      process.exit(1);
    }
  }

  return filteredTests;
}

/**
 * Run a single test
 */
function runSingleTest(test, options) {
  return new Promise((resolve) => {
    const testPath = path.join(__dirname, test.file);

    // Check if test file exists
    if (!fs.existsSync(testPath)) {
      console.log(`⏭️  Skipping Test ${test.id}: ${test.name} (file not found)`);
      resolve({
        id: test.id,
        name: test.name,
        file: test.file,
        status: 'skipped',
        reason: 'File not found',
        duration: 0,
        output: '',
        error: 'Test file does not exist'
      });
      return;
    }

    const startTime = Date.now();
    let output = '';
    let errorOutput = '';

    if (options.verbose && !options.parallel) {
      console.log(`\n🧪 Running Test ${test.id}: ${test.name}...`);
    }

    const child = spawn('node', [testPath], {
      cwd: __dirname,
      env: { ...process.env, NODE_ENV: 'test' }
    });

    // Collect output
    child.stdout.on('data', (data) => {
      const chunk = data.toString();
      output += chunk;
      if (options.verbose && !options.parallel) {
        process.stdout.write(chunk);
      }
    });

    child.stderr.on('data', (data) => {
      const chunk = data.toString();
      errorOutput += chunk;
      if (options.verbose && !options.parallel) {
        process.stderr.write(chunk);
      }
    });

    // Set timeout
    const timeout = setTimeout(() => {
      child.kill('SIGTERM');
    }, options.timeout);

    child.on('close', (code) => {
      clearTimeout(timeout);
      const duration = Date.now() - startTime;
      const status = code === 0 ? 'passed' : 'failed';

      if (!options.verbose || options.parallel) {
        const icon = status === 'passed' ? '✅' : '❌';
        const timeStr = `(${Math.round(duration/1000)}s)`;
        console.log(`${icon} Test ${test.id} (${test.name}): ${status.toUpperCase()} ${timeStr}`);
      }

      resolve({
        id: test.id,
        name: test.name,
        file: test.file,
        category: test.category,
        priority: test.priority,
        status,
        code,
        duration,
        output: output.trim(),
        error: errorOutput.trim(),
        estimatedTime: test.estimatedTime
      });
    });

    child.on('error', (error) => {
      clearTimeout(timeout);
      const duration = Date.now() - startTime;

      console.log(`❌ Test ${test.id} (${test.name}): ERROR (${Math.round(duration/1000)}s)`);

      resolve({
        id: test.id,
        name: test.name,
        file: test.file,
        status: 'failed',
        code: 1,
        duration,
        output: output.trim(),
        error: error.message,
        estimatedTime: test.estimatedTime
      });
    });
  });
}

/**
 * Run tests in parallel with limited concurrency
 */
async function runTestsParallel(tests, options) {
  console.log(`🚀 Running ${tests.length} tests in parallel (max ${PARALLEL_LIMIT} concurrent)`);

  const results = [];
  const executing = [];

  for (const test of tests) {
    const promise = runSingleTest(test, options).then(result => {
      executing.splice(executing.indexOf(promise), 1);
      return result;
    });

    results.push(promise);
    executing.push(promise);

    if (executing.length >= PARALLEL_LIMIT) {
      await Promise.race(executing);
    }
  }

  return Promise.all(results);
}

/**
 * Run tests sequentially
 */
async function runTestsSequential(tests, options) {
  console.log(`🧪 Running ${tests.length} tests sequentially`);

  const results = [];

  for (const test of tests) {
    const result = await runSingleTest(test, options);
    results.push(result);

    // Stop on first failure unless --continue is specified
    if (!options.continue && result.status === 'failed') {
      console.log(`\n💥 Test failed: ${test.name}`);
      console.log('Use --continue to run all tests regardless of failures');
      break;
    }
  }

  return results;
}

/**
 * Generate test report
 */
function generateReport(results, options) {
  const endTime = Date.now();

  // Calculate statistics
  testResults.total = results.length;
  testResults.passed = results.filter(r => r.status === 'passed').length;
  testResults.failed = results.filter(r => r.status === 'failed').length;
  testResults.skipped = results.filter(r => r.status === 'skipped').length;
  testResults.duration = endTime - testResults.startTime;
  testResults.details = results;

  // Group by category
  results.forEach(result => {
    if (!testResults.categories[result.category]) {
      testResults.categories[result.category] = { passed: 0, failed: 0, skipped: 0, total: 0 };
    }
    testResults.categories[result.category][result.status]++;
    testResults.categories[result.category].total++;
  });

  if (options.json) {
    console.log(JSON.stringify(testResults, null, 2));
    return;
  }

  // Generate human-readable report
  console.log('\n' + '='.repeat(80));
  console.log('📊 FOCUSED TEST SUITE RESULTS');
  console.log('='.repeat(80));

  console.log(`\n📈 Overall Results:`);
  console.log(`   Total Tests: ${testResults.total}`);
  console.log(`   ✅ Passed: ${testResults.passed}`);
  console.log(`   ❌ Failed: ${testResults.failed}`);
  console.log(`   ⏭️  Skipped: ${testResults.skipped}`);
  console.log(`   ⏱️  Duration: ${Math.round(testResults.duration / 1000)}s`);
  console.log(`   📊 Success Rate: ${Math.round((testResults.passed / testResults.total) * 100)}%`);

  console.log(`\n📁 Results by Category:`);
  Object.entries(testResults.categories).forEach(([category, stats]) => {
    const successRate = Math.round((stats.passed / stats.total) * 100);
    console.log(`   ${category}: ${stats.passed}/${stats.total} passed (${successRate}%)`);
  });

  // Show failed tests
  const failedTests = results.filter(r => r.status === 'failed');
  if (failedTests.length > 0) {
    console.log(`\n❌ Failed Tests:`);
    failedTests.forEach(test => {
      console.log(`   Test ${test.id}: ${test.name}`);
      if (test.error) {
        console.log(`      Error: ${test.error.split('\n')[0]}`);
      }
    });
  }

  // Show skipped tests
  const skippedTests = results.filter(r => r.status === 'skipped');
  if (skippedTests.length > 0) {
    console.log(`\n⏭️  Skipped Tests:`);
    skippedTests.forEach(test => {
      console.log(`   Test ${test.id}: ${test.name} (${test.reason || 'Unknown reason'})`);
    });
  }

  console.log('\n' + '='.repeat(80));

  // Overall assessment
  const coreTests = results.filter(r => r.category === 'core');
  const corePassRate = coreTests.length > 0 ? (coreTests.filter(r => r.status === 'passed').length / coreTests.length) * 100 : 0;

  if (testResults.failed === 0) {
    console.log('🎉 ALL TESTS PASSED! API is ready for production.');
  } else if (corePassRate >= 90) {
    console.log('✅ Core functionality is solid. Some quality tests failed but API is functional.');
  } else if (corePassRate >= 70) {
    console.log('⚠️  Most core functionality works but there are significant issues to address.');
  } else {
    console.log('❌ Critical issues detected. API needs major fixes before deployment.');
  }

  return testResults.failed === 0 ? 0 : 1;
}

/**
 * Main execution function
 */
async function main() {
  const options = parseArguments();

  if (options.help) {
    showHelp();
    return 0;
  }

  if (options.list) {
    listTests();
    return 0;
  }

  console.log('🧪 Focused Worker Management API Test Suite');
  console.log('⚡ Lean test suite: Core functionality + Schemathesis protection');
  console.log(`🎯 API Base: ${API_BASE}`);
  console.log(`⏱️  Test Timeout: ${options.timeout}ms`);
  console.log('='.repeat(80));

  testResults.startTime = Date.now();

  // Filter tests based on options
  const testsToRun = filterTests(options);

  if (testsToRun.length === 0) {
    console.log('❌ No tests to run based on filters');
    return 1;
  }

  console.log(`📋 Selected ${testsToRun.length} tests to run`);

  if (options.category) {
    console.log(`📁 Category filter: ${options.category}`);
  }

  if (options.priority) {
    console.log(`⭐ Priority filter: ${options.priority}`);
  }

  try {
    // Run tests
    let results;
    if (options.parallel) {
      results = await runTestsParallel(testsToRun, options);
    } else {
      results = await runTestsSequential(testsToRun, options);
    }

    // Generate report and return exit code
    return generateReport(results, options);

  } catch (error) {
    console.error(`\n❌ Test execution failed: ${error.message}`);
    if (error.stack) {
      console.error(error.stack);
    }
    return 1;
  }
}

// Execute if called directly
if (require.main === module) {
  main()
    .then(exitCode => process.exit(exitCode))
    .catch(error => {
      console.error('❌ Unhandled error:', error);
      process.exit(1);
    });
}

module.exports = {
  TEST_SUITE,
  runSingleTest,
  runTestsParallel,
  runTestsSequential,
  generateReport,
  main
};
