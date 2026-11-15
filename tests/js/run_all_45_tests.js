#!/usr/bin/env node

/**
 * Comprehensive Test Runner for Worker Management API
 *
 * This script runs all 45 specialized test files covering every aspect of the Worker Management API.
 * It provides detailed reporting, parallel execution options, and comprehensive test management.
 *
 * Created by 45 specialized subagents to provide complete API test coverage.
 *
 * Usage:
 *   node run_all_45_tests.js                    # Run all tests sequentially
 *   node run_all_45_tests.js --parallel         # Run tests in parallel (faster)
 *   node run_all_45_tests.js --category core    # Run specific category
 *   node run_all_45_tests.js --list             # List all available tests
 *   node run_all_45_tests.js --help             # Show help
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

// Configuration
const TEST_TIMEOUT = 300000; // 5 minutes per test
const PARALLEL_LIMIT = 5; // Max concurrent tests
const API_BASE = 'http://localhost:8887';

// Complete test suite with 45 specialized test files
const TEST_SUITE = [
  // Core API Tests (Tests 1-15)
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
    name: 'Individual Worker Operations',
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
    priority: 'critical',
    description: 'Tests worker messaging and communication',
    estimatedTime: '3m'
  },
  {
    id: '05',
    name: 'Worker States Management',
    file: 'test_05_worker_states.js',
    category: 'core',
    priority: 'high',
    description: 'Tests worker state management and history',
    estimatedTime: '2m'
  },
  {
    id: '06',
    name: 'Worker Instance & Bulk Operations',
    file: 'test_06_worker_instance_bulk.js',
    category: 'core',
    priority: 'high',
    description: 'Tests worker instance management and bulk operations',
    estimatedTime: '2m'
  },
  {
    id: '07',
    name: 'Worker States API',
    file: 'test_07_worker_states_api.js',
    category: 'core',
    priority: 'high',
    description: 'Tests worker states API endpoints',
    estimatedTime: '2m'
  },
  {
    id: '08',
    name: 'Graph Collection Operations',
    file: 'test_08_graph_collection.js',
    category: 'core',
    priority: 'critical',
    description: 'Tests graph CRUD operations and collection management',
    estimatedTime: '3m'
  },
  {
    id: '09',
    name: 'Individual Graph Operations',
    file: 'test_09_graph_individual.js',
    category: 'core',
    priority: 'critical',
    description: 'Tests individual graph operations (GET, PATCH, DELETE)',
    estimatedTime: '2m'
  },
  {
    id: '10',
    name: 'Graph Structure & Analysis',
    file: 'test_10_graph_structure.js',
    category: 'core',
    priority: 'high',
    description: 'Tests graph structure analysis and validation',
    estimatedTime: '2m'
  },
  {
    id: '11',
    name: 'Graph States Management',
    file: 'test_11_graph_states.js',
    category: 'core',
    priority: 'high',
    description: 'Tests graph state management and history',
    estimatedTime: '3m'
  },
  {
    id: '12',
    name: 'Graph Execution',
    file: 'test_12_graph_execution.js',
    category: 'core',
    priority: 'critical',
    description: 'Tests graph execution engine and monitoring',
    estimatedTime: '4m'
  },
  {
    id: '13',
    name: 'Tool Collection Operations',
    file: 'test_13_tool_collection.js',
    category: 'core',
    priority: 'high',
    description: 'Tests tool registry collection operations',
    estimatedTime: '2m'
  },
  {
    id: '14',
    name: 'Individual Tool Operations',
    file: 'test_14_tool_individual.js',
    category: 'core',
    priority: 'high',
    description: 'Tests individual tool operations',
    estimatedTime: '2m'
  },
  {
    id: '15',
    name: 'Tool Validation & Testing',
    file: 'test_15_tool_validation.js',
    category: 'core',
    priority: 'high',
    description: 'Tests tool validation and testing capabilities',
    estimatedTime: '2m'
  },

  // Quality & Reliability Tests (Tests 16-30)
  {
    id: '16',
    name: 'Security & Input Validation',
    file: 'test_16_security_validation.js',
    category: 'quality',
    priority: 'critical',
    description: 'Tests security, SQL injection, XSS, and input validation',
    estimatedTime: '4m'
  },
  {
    id: '17',
    name: 'Performance & Load Testing',
    file: 'test_17_performance_load.js',
    category: 'quality',
    priority: 'high',
    description: 'Tests performance under load and stress conditions',
    estimatedTime: '5m'
  },
  {
    id: '18',
    name: 'Error Handling & Edge Cases',
    file: 'test_18_error_handling.js',
    category: 'quality',
    priority: 'high',
    description: 'Tests comprehensive error handling and edge cases',
    estimatedTime: '3m'
  },
  {
    id: '19',
    name: 'Data Consistency & Integrity',
    file: 'test_19_data_consistency.js',
    category: 'quality',
    priority: 'high',
    description: 'Tests data consistency and integrity across operations',
    estimatedTime: '3m'
  },
  {
    id: '20',
    name: 'Schema Compliance',
    file: 'test_20_schema_compliance.js',
    category: 'quality',
    priority: 'high',
    description: 'Tests API schema compliance and documentation alignment',
    estimatedTime: '2m'
  },
  {
    id: '21',
    name: 'Unicode & International Support',
    file: 'test_21_unicode_support.js',
    category: 'quality',
    priority: 'medium',
    description: 'Tests Unicode and international character support',
    estimatedTime: '2m'
  },
  {
    id: '22',
    name: 'Graph Execution Edge Cases',
    file: 'test_22_graph_execution_edge_cases.js',
    category: 'quality',
    priority: 'high',
    description: 'Tests complex graph execution scenarios and edge cases',
    estimatedTime: '4m'
  },
  {
    id: '23',
    name: 'Model Provider Coverage',
    file: 'test_23_model_provider_coverage.js',
    category: 'quality',
    priority: 'high',
    description: 'Tests all AI model providers and configurations',
    estimatedTime: '5m'
  },
  {
    id: '24',
    name: 'Tool Integration',
    file: 'test_24_tool_integration.js',
    category: 'quality',
    priority: 'high',
    description: 'Tests tool system integration and custom tool execution',
    estimatedTime: '3m'
  },
  {
    id: '25',
    name: 'MCP Integration',
    file: 'test_25_mcp_integration.js',
    category: 'quality',
    priority: 'medium',
    description: 'Tests Model Control Protocol server integration',
    estimatedTime: '3m'
  },
  {
    id: '26',
    name: 'HTTP Compliance',
    file: 'test_26_http_compliance.js',
    category: 'quality',
    priority: 'medium',
    description: 'Tests HTTP status codes, headers, and compliance',
    estimatedTime: '2m'
  },
  {
    id: '27',
    name: 'Concurrency & Race Conditions',
    file: 'test_27_concurrency_race_conditions.js',
    category: 'quality',
    priority: 'high',
    description: 'Tests concurrent operations and race condition prevention',
    estimatedTime: '4m'
  },
  {
    id: '28',
    name: 'Memory Management',
    file: 'test_28_memory_management.js',
    category: 'quality',
    priority: 'medium',
    description: 'Tests memory management and resource leak detection',
    estimatedTime: '3m'
  },
  {
    id: '29',
    name: 'Rate Limiting',
    file: 'test_29_rate_limiting.js',
    category: 'quality',
    priority: 'medium',
    description: 'Tests API rate limiting and throttling mechanisms',
    estimatedTime: '3m'
  },
  {
    id: '30',
    name: 'Database Operations',
    file: 'test_30_database_operations.js',
    category: 'quality',
    priority: 'high',
    description: 'Tests database operations and SQL injection prevention',
    estimatedTime: '3m'
  },

  // Infrastructure & Operations Tests (Tests 31-40)
  {
    id: '31',
    name: 'API Versioning',
    file: 'test_31_api_versioning.js',
    category: 'infrastructure',
    priority: 'medium',
    description: 'Tests API versioning and backward compatibility',
    estimatedTime: '2m'
  },
  {
    id: '32',
    name: 'WebSocket & Real-time',
    file: 'test_32_websocket_realtime.js',
    category: 'infrastructure',
    priority: 'medium',
    description: 'Tests WebSocket and real-time communication features',
    estimatedTime: '3m'
  },
  {
    id: '33',
    name: 'Monitoring & Observability',
    file: 'test_33_monitoring_observability.js',
    category: 'infrastructure',
    priority: 'high',
    description: 'Tests monitoring, logging, and observability features',
    estimatedTime: '3m'
  },
  {
    id: '34',
    name: 'Configuration Management',
    file: 'test_34_configuration_management.js',
    category: 'infrastructure',
    priority: 'medium',
    description: 'Tests configuration management and environment handling',
    estimatedTime: '2m'
  },
  {
    id: '35',
    name: 'Deployment & DevOps',
    file: 'test_35_deployment_devops.js',
    category: 'infrastructure',
    priority: 'high',
    description: 'Tests deployment readiness and DevOps integration',
    estimatedTime: '3m'
  },
  {
    id: '36',
    name: 'Compliance & Regulatory',
    file: 'test_36_compliance_regulatory.js',
    category: 'infrastructure',
    priority: 'medium',
    description: 'Tests GDPR, HIPAA, SOC 2 compliance requirements',
    estimatedTime: '4m'
  },
  {
    id: '37',
    name: 'Accessibility & Inclusivity',
    file: 'test_37_accessibility_inclusivity.js',
    category: 'infrastructure',
    priority: 'low',
    description: 'Tests accessibility and inclusivity features',
    estimatedTime: '3m'
  },
  {
    id: '38',
    name: 'AI/ML Governance',
    file: 'test_38_ai_ml_governance.js',
    category: 'infrastructure',
    priority: 'high',
    description: 'Tests AI/ML model integration and governance',
    estimatedTime: '4m'
  },
  {
    id: '39',
    name: 'Third-party Integration',
    file: 'test_39_third_party_integration.js',
    category: 'infrastructure',
    priority: 'medium',
    description: 'Tests external service integration and webhooks',
    estimatedTime: '3m'
  },
  {
    id: '40',
    name: 'Cost Optimization',
    file: 'test_40_cost_optimization.js',
    category: 'infrastructure',
    priority: 'medium',
    description: 'Tests cost optimization and resource efficiency',
    estimatedTime: '3m'
  },

  // Advanced & Future Technologies Tests (Tests 41-45)
  {
    id: '41',
    name: 'Blockchain Integration',
    file: 'test_41_blockchain_integration.js',
    category: 'advanced',
    priority: 'low',
    description: 'Tests blockchain and distributed ledger integration',
    estimatedTime: '4m'
  },
  {
    id: '42',
    name: 'Edge & IoT Integration',
    file: 'test_42_edge_iot_integration.js',
    category: 'advanced',
    priority: 'low',
    description: 'Tests edge computing and IoT device integration',
    estimatedTime: '3m'
  },
  {
    id: '43',
    name: 'Multi-tenancy',
    file: 'test_43_multi_tenancy.js',
    category: 'advanced',
    priority: 'medium',
    description: 'Tests multi-tenant isolation and resource management',
    estimatedTime: '3m'
  },
  {
    id: '44',
    name: 'Analytics & Business Intelligence',
    file: 'test_44_analytics_business_intelligence.js',
    category: 'advanced',
    priority: 'medium',
    description: 'Tests advanced analytics and BI capabilities',
    estimatedTime: '4m'
  },
  {
    id: '45',
    name: 'Future Technologies',
    file: 'test_45_future_technologies.js',
    category: 'advanced',
    priority: 'low',
    description: 'Tests quantum computing, AR/VR, and emerging technologies',
    estimatedTime: '4m'
  }
];

// Test results tracking
const results = {
  total: 0,
  passed: 0,
  failed: 0,
  skipped: 0,
  errors: [],
  details: [],
  startTime: null,
  endTime: null,
  duration: 0
};

// Helper functions
function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    parallel: false,
    category: null,
    priority: null,
    list: false,
    help: false,
    verbose: false,
    tests: []
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    switch (arg) {
      case '--parallel':
      case '-p':
        config.parallel = true;
        break;
      case '--category':
      case '-c':
        config.category = args[++i];
        break;
      case '--priority':
        config.priority = args[++i];
        break;
      case '--list':
      case '-l':
        config.list = true;
        break;
      case '--help':
      case '-h':
        config.help = true;
        break;
      case '--verbose':
      case '-v':
        config.verbose = true;
        break;
      default:
        if (arg.startsWith('test_')) {
          config.tests.push(arg);
        }
        break;
    }
  }

  return config;
}

function showHelp() {
  console.log(`
🚀 Worker Management API - Comprehensive Test Suite Runner

Usage: node run_all_45_tests.js [options]

Options:
  --parallel, -p         Run tests in parallel (faster, up to ${PARALLEL_LIMIT} concurrent)
  --category, -c <cat>   Run tests from specific category (core, quality, infrastructure, advanced)
  --priority <level>     Run tests with specific priority (critical, high, medium, low)
  --list, -l             List all available tests without running them
  --verbose, -v          Enable verbose output
  --help, -h             Show this help message

Categories:
  core            Core API functionality (tests 1-15)
  quality         Quality & reliability testing (tests 16-30)
  infrastructure  Infrastructure & operations (tests 31-40)
  advanced        Advanced & future technologies (tests 41-45)

Priority Levels:
  critical        Must-pass tests for basic API functionality
  high            Important tests for production readiness
  medium          Recommended tests for full coverage
  low             Optional tests for comprehensive validation

Examples:
  node run_all_45_tests.js                    # Run all tests sequentially
  node run_all_45_tests.js --parallel         # Run all tests in parallel
  node run_all_45_tests.js --category core    # Run only core API tests
  node run_all_45_tests.js --priority critical # Run only critical tests
  node run_all_45_tests.js --list             # List all available tests

Total Tests: ${TEST_SUITE.length}
Estimated Total Time: ~3-4 hours sequential, ~45-60 minutes parallel
`);
}

function listTests(config) {
  let testsToShow = TEST_SUITE;

  if (config.category) {
    testsToShow = testsToShow.filter(test => test.category === config.category);
  }

  if (config.priority) {
    testsToShow = testsToShow.filter(test => test.priority === config.priority);
  }

  console.log(`\n📋 Available Tests${config.category ? ` (${config.category})` : ''}${config.priority ? ` (${config.priority} priority)` : ''}:\n`);

  const categories = ['core', 'quality', 'infrastructure', 'advanced'];

  categories.forEach(category => {
    const categoryTests = testsToShow.filter(test => test.category === category);
    if (categoryTests.length === 0) return;

    console.log(`\n🔧 ${category.toUpperCase()} (${categoryTests.length} tests):`);
    categoryTests.forEach(test => {
      const priority = test.priority === 'critical' ? '🔴' :
                      test.priority === 'high' ? '🟡' :
                      test.priority === 'medium' ? '🟢' : '🔵';
      console.log(`  ${priority} Test ${test.id}: ${test.name} (${test.estimatedTime})`);
      if (config.verbose) {
        console.log(`      File: ${test.file}`);
        console.log(`      Description: ${test.description}\n`);
      }
    });
  });

  console.log(`\nTotal: ${testsToShow.length} tests`);

  const totalEstimatedMinutes = testsToShow.reduce((total, test) => {
    const timeStr = test.estimatedTime;
    const minutes = timeStr.includes('m') ? parseInt(timeStr) :
                   timeStr.includes('s') ? parseInt(timeStr) / 60 : 1;
    return total + minutes;
  }, 0);

  console.log(`Estimated time: ${Math.ceil(totalEstimatedMinutes)} minutes sequential`);
  if (testsToShow.length > PARALLEL_LIMIT) {
    console.log(`                ${Math.ceil(totalEstimatedMinutes / PARALLEL_LIMIT)} minutes parallel`);
  }
}

function getTestsToRun(config) {
  let testsToRun = [...TEST_SUITE];

  if (config.tests.length > 0) {
    testsToRun = testsToRun.filter(test => config.tests.includes(test.file));
  }

  if (config.category) {
    testsToRun = testsToRun.filter(test => test.category === config.category);
  }

  if (config.priority) {
    testsToRun = testsToRun.filter(test => test.priority === config.priority);
  }

  return testsToRun;
}

function runTest(test) {
  return new Promise((resolve) => {
    const testPath = path.join(__dirname, test.file);

    // Check if test file exists
    if (!fs.existsSync(testPath)) {
      console.log(`❌ Test ${test.id} (${test.name}): File not found - ${test.file}`);
      resolve({
        test,
        success: false,
        error: `Test file not found: ${test.file}`,
        output: '',
        duration: 0
      });
      return;
    }

    console.log(`🧪 Running Test ${test.id}: ${test.name}...`);
    const startTime = Date.now();

    const child = spawn('node', [testPath], {
      stdio: 'pipe',
      timeout: TEST_TIMEOUT,
      cwd: __dirname
    });

    let output = '';
    let errorOutput = '';

    child.stdout.on('data', (data) => {
      output += data.toString();
    });

    child.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    child.on('close', (code) => {
      const duration = Date.now() - startTime;
      const success = code === 0;

      if (success) {
        console.log(`✅ Test ${test.id} (${test.name}): PASSED (${Math.round(duration/1000)}s)`);
      } else {
        console.log(`❌ Test ${test.id} (${test.name}): FAILED (code ${code}) (${Math.round(duration/1000)}s)`);
      }

      resolve({
        test,
        success,
        code,
        output,
        errorOutput,
        duration,
        error: success ? null : (errorOutput || `Exit code: ${code}`)
      });
    });

    child.on('error', (error) => {
      const duration = Date.now() - startTime;
      console.log(`❌ Test ${test.id} (${test.name}): ERROR - ${error.message}`);

      resolve({
        test,
        success: false,
        error: error.message,
        output,
        errorOutput,
        duration
      });
    });
  });
}

async function runTestsSequentially(tests) {
  console.log(`\n🔄 Running ${tests.length} tests sequentially...\n`);

  for (const test of tests) {
    const result = await runTest(test);
    processTestResult(result);

    // Small delay between tests
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}

async function runTestsInParallel(tests) {
  console.log(`\n🔄 Running ${tests.length} tests in parallel (limit: ${PARALLEL_LIMIT})...\n`);

  const chunks = [];
  for (let i = 0; i < tests.length; i += PARALLEL_LIMIT) {
    chunks.push(tests.slice(i, i + PARALLEL_LIMIT));
  }

  for (const chunk of chunks) {
    const promises = chunk.map(test => runTest(test));
    const results = await Promise.all(promises);

    results.forEach(result => processTestResult(result));

    // Small delay between batches
    if (chunks.indexOf(chunk) < chunks.length - 1) {
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }
}

function processTestResult(result) {
  results.total++;
  results.details.push(result);

  if (result.success) {
    results.passed++;
  } else {
    results.failed++;
    results.errors.push({
      test: result.test,
      error: result.error,
      output: result.errorOutput || result.output
    });
  }
}

function generateReport() {
  results.endTime = Date.now();
  results.duration = results.endTime - results.startTime;

  console.log('\n' + '='.repeat(80));
  console.log('🎯 COMPREHENSIVE API TEST RESULTS SUMMARY');
  console.log('='.repeat(80));

  // Overall statistics
  const passRate = results.total > 0 ? ((results.passed / results.total) * 100).toFixed(1) : 0;
  const avgDuration = results.total > 0 ? (results.duration / results.total / 1000).toFixed(1) : 0;

  console.log(`\n📊 OVERALL STATISTICS:`);
  console.log(`   Total Tests:     ${results.total}`);
  console.log(`   Passed:          ${results.passed} ✅`);
  console.log(`   Failed:          ${results.failed} ❌`);
  console.log(`   Pass Rate:       ${passRate}%`);
  console.log(`   Total Duration:  ${Math.round(results.duration / 1000)}s`);
  console.log(`   Avg Per Test:    ${avgDuration}s`);

  // Category breakdown
  const categories = ['core', 'quality', 'infrastructure', 'advanced'];
  console.log(`\n📋 RESULTS BY CATEGORY:`);

  categories.forEach(category => {
    const categoryTests = results.details.filter(r => r.test.category === category);
    if (categoryTests.length === 0) return;

    const categoryPassed = categoryTests.filter(r => r.success).length;
    const categoryRate = ((categoryPassed / categoryTests.length) * 100).toFixed(1);

    console.log(`   ${category.toUpperCase().padEnd(15)} ${categoryPassed}/${categoryTests.length} (${categoryRate}%)`);
  });

  // Priority breakdown
  console.log(`\n⚡ RESULTS BY PRIORITY:`);
  ['critical', 'high', 'medium', 'low'].forEach(priority => {
    const priorityTests = results.details.filter(r => r.test.priority === priority);
    if (priorityTests.length === 0) return;

    const priorityPassed = priorityTests.filter(r => r.success).length;
    const priorityRate = ((priorityPassed / priorityTests.length) * 100).toFixed(1);
    const emoji = priority === 'critical' ? '🔴' : priority === 'high' ? '🟡' : priority === 'medium' ? '🟢' : '🔵';

    console.log(`   ${emoji} ${priority.toUpperCase().padEnd(12)} ${priorityPassed}/${priorityTests.length} (${priorityRate}%)`);
  });

  // Failed tests details
  if (results.failed > 0) {
    console.log(`\n❌ FAILED TESTS DETAILS:`);
    results.errors.forEach((error, index) => {
      console.log(`\n   ${index + 1}. Test ${error.test.id}: ${error.test.name}`);
      console.log(`      File: ${error.test.file}`);
      console.log(`      Error: ${error.error}`);
      if (error.output && error.output.trim()) {
        const outputLines = error.output.trim().split('\n');
        const lastFewLines = outputLines.slice(-3).join('\n      ');
        console.log(`      Output: ${lastFewLines}`);
      }
    });
  }

  // Recommendations
  console.log(`\n💡 RECOMMENDATIONS:`);
  if (results.failed === 0) {
    console.log(`   🎉 Excellent! All tests passed. Your API is in great shape!`);
  } else {
    const criticalFailed = results.errors.filter(e => e.test.priority === 'critical').length;
    const highFailed = results.errors.filter(e => e.test.priority === 'high').length;

    if (criticalFailed > 0) {
      console.log(`   🚨 ${criticalFailed} critical tests failed. Address these immediately!`);
    }
    if (highFailed > 0) {
      console.log(`   ⚠️  ${highFailed} high-priority tests failed. These should be fixed soon.`);
    }

    console.log(`   📝 Review failed test details above for specific issues.`);
    console.log(`   🔧 Run individual tests with more verbose output for debugging.`);
  }

  // API Health Assessment
  console.log(`\n🏥 API HEALTH ASSESSMENT:`);
  if (passRate >= 95) {
    console.log(`   🌟 EXCELLENT (${passRate}%) - Production ready!`);
  } else if (passRate >= 85) {
    console.log(`   ✅ GOOD (${passRate}%) - Minor issues to address`);
  } else if (passRate >= 70) {
    console.log(`   ⚠️  FAIR (${passRate}%) - Several issues need attention`);
  } else {
    console.log(`   ❌ POOR (${passRate}%) - Significant issues require immediate attention`);
  }

  console.log('\n' + '='.repeat(80));

  // Save detailed report to file
  const reportData = {
    timestamp: new Date().toISOString(),
    summary: {
      total: results.total,
      passed: results.passed,
      failed: results.failed,
      passRate: parseFloat(passRate),
      duration: results.duration
    },
    details: results.details.map(r => ({
      testId: r.test.id,
      testName: r.test.name,
      category: r.test.category,
      priority: r.test.priority,
      success: r.success,
      duration: r.duration,
      error: r.error
    })),
    errors: results.errors.map(e => ({
      testId: e.test.id,
      testName: e.test.name,
      error: e.error
    }))
  };

  const reportFile = `test-results-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
  fs.writeFileSync(reportFile, JSON.stringify(reportData, null, 2));
  console.log(`📄 Detailed report saved to: ${reportFile}`);

  return results.failed === 0;
}

async function checkApiHealth() {
  console.log(`🏥 Checking API health at ${API_BASE}...`);

  try {
    const response = await fetch(`${API_BASE}/health`);
    if (response.ok) {
      console.log(`✅ API is healthy and ready for testing`);
      return true;
    } else {
      console.log(`❌ API health check failed (${response.status})`);
      return false;
    }
  } catch (error) {
    console.log(`❌ Cannot connect to API at ${API_BASE}`);
    console.log(`   Error: ${error.message}`);
    console.log(`   Make sure the API server is running!`);
    return false;
  }
}

// Main execution
async function main() {
  const config = parseArgs();

  console.log('🚀 Worker Management API - Comprehensive Test Suite');
  console.log(`📅 Started at: ${new Date().toLocaleString()}`);

  if (config.help) {
    showHelp();
    return;
  }

  if (config.list) {
    listTests(config);
    return;
  }

  // Check API health before running tests
  const apiHealthy = await checkApiHealth();
  if (!apiHealthy) {
    console.log('\n❌ Cannot proceed with tests - API is not available');
    process.exit(1);
  }

  const testsToRun = getTestsToRun(config);

  if (testsToRun.length === 0) {
    console.log('❌ No tests match the specified criteria');
    return;
  }

  console.log(`\n📋 Running ${testsToRun.length} tests...`);
  if (config.category) console.log(`   Category: ${config.category}`);
  if (config.priority) console.log(`   Priority: ${config.priority}`);
  if (config.parallel) console.log(`   Mode: Parallel (max ${PARALLEL_LIMIT} concurrent)`);

  results.startTime = Date.now();

  try {
    if (config.parallel) {
      await runTestsInParallel(testsToRun);
    } else {
      await runTestsSequentially(testsToRun);
    }

    const success = generateReport();
    process.exit(success ? 0 : 1);

  } catch (error) {
    console.error(`\n❌ Test runner error: ${error.message}`);
    process.exit(1);
  }
}

// Handle Ctrl+C gracefully
process.on('SIGINT', () => {
  console.log('\n\n⚠️  Test run interrupted by user');
  if (results.total > 0) {
    generateReport();
  }
  process.exit(1);
});

// Run if called directly
if (require.main === module) {
  main().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}

module.exports = {
  TEST_SUITE,
  runTest,
  generateReport
};
