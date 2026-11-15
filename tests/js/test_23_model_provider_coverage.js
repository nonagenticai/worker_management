#!/usr/bin/env node
/**
 * JavaScript Test Script for Model Provider Coverage - Production Ready
 *
 * This script validates model provider functionality of the Worker Management API by:
 * 1. Testing all supported model providers (OpenAI, Anthropic, DeepSeek, etc.)
 * 2. Testing model-specific configurations and parameters
 * 3. Testing provider authentication and API key handling
 * 4. Testing provider-specific features and limitations
 * 5. Testing fallback and error handling for providers
 * 6. Testing temperature and parameter variations
 * 7. Testing base URL overrides for custom deployments
 * 8. Testing provider response format compatibility
 * 9. Testing concurrent requests across providers
 * 10. Testing provider availability and health checks
 *
 * ENDPOINTS TESTED:
 * - POST /workers (with different model providers)
 * - GET /workers/{worker_name}
 * - POST /workers/{worker_name}/message
 * - GET /providers/health (if available)
 *
 * Requirements: Node.js 18+ (for native fetch API)
 * Usage: node test_23_model_provider_coverage.js
 */

const { v4: uuidv4 } = require('uuid');

// Enhanced Configuration
const API_BASE = 'http://localhost:8887';
const TIMEOUT_MS = 60000; // 1 minute for provider tests
const MAX_RETRIES = 3;
const RETRY_DELAY = 2000; // 2 seconds
const REQUEST_DELAY = 1000; // 1 second between requests

// Model provider configurations
const MODEL_PROVIDERS = [
    {
        provider: 'openai',
        models: ['deepseek-chat'],
        defaultModel: 'gpt-3.5-turbo',
        supportedFeatures: ['temperature', 'max_tokens', 'tools', 'streaming'],
        requiresApiKey: true,
        baseUrlConfigurable: true
    },
    {
        provider: 'anthropic',
        models: ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
        defaultModel: 'claude-3-haiku-20240307',
        supportedFeatures: ['temperature', 'max_tokens', 'tools'],
        requiresApiKey: true,
        baseUrlConfigurable: false
    },
    {
        provider: 'deepseek',
        models: ['deepseek-chat', 'deepseek-coder'],
        defaultModel: 'deepseek-chat',
        supportedFeatures: ['temperature', 'max_tokens'],
        requiresApiKey: true,
        baseUrlConfigurable: true
    },
    {
        provider: 'groq',
        models: ['llama2-70b-4096', 'mixtral-8x7b-32768'],
        defaultModel: 'llama2-70b-4096',
        supportedFeatures: ['temperature', 'max_tokens', 'streaming'],
        requiresApiKey: true,
        baseUrlConfigurable: false
    },
    {
        provider: 'mlx',
        models: ['openhermes', 'mistral', 'llama'],
        defaultModel: 'openhermes',
        supportedFeatures: ['temperature', 'streaming'],
        requiresApiKey: false,
        baseUrlConfigurable: true
    }
];

// Add a flag to track if we've already retried for timeout
let hasRetried = false;

/**
 * Enhanced utility function to make HTTP requests with retry logic and comprehensive error handling
 */
async function makeRequest(url, options = {}, retries = MAX_RETRIES) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'WorkerManagement-ModelProviderTest/1.0',
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

                console.log('🕐 API timeout/network issue detected. Waiting 30 seconds before single retry...');
                console.log(`   Original error: ${error.message}`);
                hasRetried = true;

                // Sleep for 30 seconds
                await new Promise(resolve => setTimeout(resolve, 30000));

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
 * Test basic model provider functionality
 */
async function testBasicProviderFunctionality(providerConfig) {
    console.log(`\n=== Testing Basic Functionality for ${providerConfig.provider} ===`);

    const results = {
        provider: providerConfig.provider,
        worker_creation: false,
        worker_retrieval: false,
        message_sending: false,
        model_switching: false,
        error: null
    };

    try {
        // Test 1: Create worker with default model
        const workerName = `test_${providerConfig.provider}_${Date.now()}`;
        const workerConfig = {
            worker_name: workerName,
            agent_role: `${providerConfig.provider} test agent`,
            agent_type: 'pydantic_ai',
            model_provider: providerConfig.provider,
            model: providerConfig.defaultModel,
            temperature: 0.5,
            system_prompt: `You are a test agent using ${providerConfig.provider} provider.`,
            instruction: 'Respond to test messages appropriately.',
            user_prompt: '',
            description: `Test worker for ${providerConfig.provider} provider`
        };

        console.log(`📊 Creating worker with ${providerConfig.provider}/${providerConfig.defaultModel}...`);

        try {
            const createResponse = await makeRequest(`${API_BASE}/workers`, {
                method: 'POST',
                body: JSON.stringify(workerConfig)
            });

            console.log(`✅ Worker created successfully: ${createResponse.name}`);
            results.worker_creation = true;

            // Test 2: Retrieve worker
            const getResponse = await makeRequest(`${API_BASE}/workers/${workerName}`);
            console.log(`✅ Worker retrieved successfully`);
            results.worker_retrieval = true;

            // Test 3: Send message
            console.log(`📨 Sending test message to ${providerConfig.provider} worker...`);
            const messageRequest = {
                message: `Hello! Please confirm you are using the ${providerConfig.provider} provider with model ${providerConfig.defaultModel}.`,
                thread_id: uuidv4()
            };

            try {
                const messageResponse = await makeRequest(`${API_BASE}/workers/${workerName}/message`, {
                    method: 'POST',
                    body: JSON.stringify(messageRequest)
                });

                console.log(`✅ Message sent successfully`);
                console.log(`   Response preview: ${messageResponse.response?.substring(0, 100)}...`);
                results.message_sending = true;
            } catch (msgError) {
                console.log(`⚠️ Message sending failed: ${msgError.message}`);
                results.message_sending = false;
            }

            // Test 4: Model switching (if multiple models available)
            if (providerConfig.models.length > 1) {
                console.log(`🔄 Testing model switching...`);
                const alternateModel = providerConfig.models.find(m => m !== providerConfig.defaultModel);

                const updateConfig = {
                    ...workerConfig,
                    model: alternateModel
                };

                try {
                    await makeRequest(`${API_BASE}/workers/${workerName}`, {
                        method: 'PUT',
                        body: JSON.stringify(updateConfig)
                    });

                    console.log(`✅ Successfully switched to model: ${alternateModel}`);
                    results.model_switching = true;
                } catch (updateError) {
                    console.log(`⚠️ Model switching failed: ${updateError.message}`);
                    results.model_switching = false;
                }
            } else {
                console.log(`ℹ️ Only one model available for ${providerConfig.provider}, skipping model switch test`);
                results.model_switching = true; // Not applicable
            }

            // Cleanup
            try {
                await makeRequest(`${API_BASE}/workers/${workerName}`, { method: 'DELETE' });
                console.log(`🧹 Worker cleaned up successfully`);
            } catch (cleanupError) {
                console.log(`⚠️ Cleanup failed: ${cleanupError.message}`);
            }

        } catch (error) {
            console.error(`❌ Provider test failed: ${error.message}`);
            results.error = error.message;

            // Check if it's an API key issue
            if (error.message.includes('401') || error.message.includes('api_key') || error.message.includes('unauthorized')) {
                console.log(`🔑 This appears to be an API key issue for ${providerConfig.provider}`);
                results.error = 'API_KEY_REQUIRED';
            }
        }

    } catch (error) {
        console.error(`❌ Test setup failed: ${error.message}`);
        results.error = error.message;
    }

    return results;
}

/**
 * Test temperature parameter variations
 */
async function testTemperatureVariations(providerConfig) {
    console.log(`\n=== Testing Temperature Variations for ${providerConfig.provider} ===`);

    if (!providerConfig.supportedFeatures.includes('temperature')) {
        console.log(`ℹ️ ${providerConfig.provider} does not support temperature parameter`);
        return {
            provider: providerConfig.provider,
            supported: false,
            skipped: true
        };
    }

    const temperatures = [0.0, 0.5, 1.0, 1.5];
    const results = {
        provider: providerConfig.provider,
        supported: true,
        temperature_tests: {}
    };

    for (const temp of temperatures) {
        console.log(`\n📊 Testing temperature: ${temp}`);
        const workerName = `temp_test_${providerConfig.provider}_${temp}_${Date.now()}`;

        try {
            // Create worker with specific temperature
            const workerConfig = {
                worker_name: workerName,
                agent_role: 'temperature test agent',
                agent_type: 'pydantic_ai',
                model_provider: providerConfig.provider,
                model: providerConfig.defaultModel,
                temperature: temp,
                system_prompt: 'You are testing temperature settings. Be creative at high temperatures, precise at low.',
                instruction: 'Respond based on temperature setting.',
                user_prompt: '',
                description: `Temperature ${temp} test`
            };

            await makeRequest(`${API_BASE}/workers`, {
                method: 'POST',
                body: JSON.stringify(workerConfig)
            });

            // Send test message
            const messageRequest = {
                message: 'Generate a creative story title.',
                thread_id: uuidv4()
            };

            const response = await makeRequest(`${API_BASE}/workers/${workerName}/message`, {
                method: 'POST',
                body: JSON.stringify(messageRequest)
            });

            results.temperature_tests[temp] = {
                success: true,
                response_length: response.response?.length || 0,
                response_preview: response.response?.substring(0, 50) || ''
            };

            console.log(`✅ Temperature ${temp} test successful`);

            // Cleanup
            await makeRequest(`${API_BASE}/workers/${workerName}`, { method: 'DELETE' });

        } catch (error) {
            console.error(`❌ Temperature ${temp} test failed: ${error.message}`);
            results.temperature_tests[temp] = {
                success: false,
                error: error.message
            };
        }
    }

    return results;
}

/**
 * Test base URL override functionality
 */
async function testBaseUrlOverride(providerConfig) {
    console.log(`\n=== Testing Base URL Override for ${providerConfig.provider} ===`);

    if (!providerConfig.baseUrlConfigurable) {
        console.log(`ℹ️ ${providerConfig.provider} does not support base URL override`);
        return {
            provider: providerConfig.provider,
            supported: false,
            skipped: true
        };
    }

    const results = {
        provider: providerConfig.provider,
        supported: true,
        override_accepted: false,
        override_functional: false
    };

    const workerName = `baseurl_test_${providerConfig.provider}_${Date.now()}`;

    try {
        // Create worker with custom base URL
        const customBaseUrl = providerConfig.provider === 'mlx'
            ? 'http://localhost:8080/v1'
            : `https://custom-${providerConfig.provider}.example.com/v1`;

        const workerConfig = {
            worker_name: workerName,
            agent_role: 'base URL test agent',
            agent_type: 'pydantic_ai',
            model_provider: providerConfig.provider,
            model: providerConfig.defaultModel,
            temperature: 0.5,
            system_prompt: 'You are testing custom base URL configuration.',
            instruction: 'Respond to test messages.',
            user_prompt: '',
            description: 'Base URL override test',
            base_url_override: customBaseUrl
        };

        console.log(`📊 Creating worker with custom base URL: ${customBaseUrl}`);

        const createResponse = await makeRequest(`${API_BASE}/workers`, {
            method: 'POST',
            body: JSON.stringify(workerConfig)
        });

        console.log(`✅ Worker created with base URL override`);
        results.override_accepted = true;

        // Verify the override was saved
        const getResponse = await makeRequest(`${API_BASE}/workers/${workerName}`);

        if (getResponse.base_url_override === customBaseUrl) {
            console.log(`✅ Base URL override correctly saved`);
            results.override_functional = true;
        } else {
            console.log(`⚠️ Base URL override not saved correctly`);
            results.override_functional = false;
        }

        // Cleanup
        await makeRequest(`${API_BASE}/workers/${workerName}`, { method: 'DELETE' });

    } catch (error) {
        console.error(`❌ Base URL override test failed: ${error.message}`);
        results.error = error.message;
    }

    return results;
}

/**
 * Test concurrent requests across providers
 */
async function testConcurrentRequests(providerConfigs) {
    console.log('\n=== Testing Concurrent Requests Across Providers ===');

    const workers = [];
    const results = {
        workers_created: 0,
        concurrent_messages_sent: 0,
        responses_received: 0,
        errors: []
    };

    try {
        // Create workers for each provider
        console.log('\n📊 Creating workers for concurrent testing...');

        for (const provider of providerConfigs.slice(0, 3)) { // Test first 3 providers
            const workerName = `concurrent_${provider.provider}_${Date.now()}`;
            const workerConfig = {
                worker_name: workerName,
                agent_role: `${provider.provider} concurrent test`,
                agent_type: 'pydantic_ai',
                model_provider: provider.provider,
                model: provider.defaultModel,
                temperature: 0.5,
                system_prompt: `You are a ${provider.provider} worker for concurrent testing.`,
                instruction: 'Respond quickly to test messages.',
                user_prompt: '',
                description: 'Concurrent test worker'
            };

            try {
                await makeRequest(`${API_BASE}/workers`, {
                    method: 'POST',
                    body: JSON.stringify(workerConfig)
                });

                workers.push({
                    name: workerName,
                    provider: provider.provider
                });

                results.workers_created++;
                console.log(`✅ Created worker for ${provider.provider}`);
            } catch (error) {
                console.log(`⚠️ Failed to create worker for ${provider.provider}: ${error.message}`);
                results.errors.push(`${provider.provider}: ${error.message}`);
            }
        }

        // Send concurrent messages
        console.log('\n📨 Sending concurrent messages...');
        const messagePromises = workers.map(worker => {
            const messageRequest = {
                message: `Concurrent test for ${worker.provider}. Respond with your provider name.`,
                thread_id: uuidv4()
            };

            return makeRequest(`${API_BASE}/workers/${worker.name}/message`, {
                method: 'POST',
                body: JSON.stringify(messageRequest)
            }).then(response => {
                results.responses_received++;
                return { worker: worker.name, provider: worker.provider, success: true, response };
            }).catch(error => {
                return { worker: worker.name, provider: worker.provider, success: false, error: error.message };
            });
        });

        results.concurrent_messages_sent = messagePromises.length;

        const responses = await Promise.allSettled(messagePromises);

        console.log('\n📊 Concurrent request results:');
        responses.forEach(result => {
            if (result.status === 'fulfilled' && result.value.success) {
                console.log(`✅ ${result.value.provider}: Message processed successfully`);
            } else {
                console.log(`❌ ${result.value?.provider || 'Unknown'}: Failed`);
            }
        });

        // Cleanup workers
        console.log('\n🧹 Cleaning up workers...');
        for (const worker of workers) {
            try {
                await makeRequest(`${API_BASE}/workers/${worker.name}`, { method: 'DELETE' });
            } catch (error) {
                console.log(`⚠️ Failed to cleanup ${worker.name}`);
            }
        }

    } catch (error) {
        console.error(`❌ Concurrent test failed: ${error.message}`);
        results.errors.push(error.message);
    }

    return results;
}

/**
 * Test provider-specific features
 */
async function testProviderSpecificFeatures(providerConfig) {
    console.log(`\n=== Testing Provider-Specific Features for ${providerConfig.provider} ===`);

    const results = {
        provider: providerConfig.provider,
        features_tested: {},
        overall_success: true
    };

    // Test tools/function calling if supported
    if (providerConfig.supportedFeatures.includes('tools')) {
        console.log('\n📊 Testing tools/function calling...');

        const workerName = `tools_test_${providerConfig.provider}_${Date.now()}`;

        try {
            const workerConfig = {
                worker_name: workerName,
                agent_role: 'tools test agent',
                agent_type: 'pydantic_ai',
                model_provider: providerConfig.provider,
                model: providerConfig.defaultModel,
                temperature: 0.3,
                system_prompt: 'You are an agent that can use tools and functions.',
                instruction: 'Use available tools when appropriate.',
                user_prompt: '',
                description: 'Tools feature test',
                tools_enabled: true
            };

            await makeRequest(`${API_BASE}/workers`, {
                method: 'POST',
                body: JSON.stringify(workerConfig)
            });

            // Send message that might trigger tool use
            const messageRequest = {
                message: 'What is the current weather in New York?',
                thread_id: uuidv4()
            };

            const response = await makeRequest(`${API_BASE}/workers/${workerName}/message`, {
                method: 'POST',
                body: JSON.stringify(messageRequest)
            });

            results.features_tested.tools = {
                supported: true,
                success: true,
                response_includes_tool_mention: response.response?.includes('tool') || response.response?.includes('function')
            };

            // Cleanup
            await makeRequest(`${API_BASE}/workers/${workerName}`, { method: 'DELETE' });

        } catch (error) {
            results.features_tested.tools = {
                supported: true,
                success: false,
                error: error.message
            };
            results.overall_success = false;
        }
    }

    // Test streaming if supported
    if (providerConfig.supportedFeatures.includes('streaming')) {
        console.log('\n📊 Testing streaming capability...');
        results.features_tested.streaming = {
            supported: true,
            tested: false,
            note: 'Streaming tests would require WebSocket or SSE support'
        };
    }

    // Test max_tokens if supported
    if (providerConfig.supportedFeatures.includes('max_tokens')) {
        console.log('\n📊 Testing max_tokens parameter...');

        const workerName = `tokens_test_${providerConfig.provider}_${Date.now()}`;

        try {
            const workerConfig = {
                worker_name: workerName,
                agent_role: 'max tokens test',
                agent_type: 'pydantic_ai',
                model_provider: providerConfig.provider,
                model: providerConfig.defaultModel,
                temperature: 0.5,
                max_tokens: 50, // Very limited response
                system_prompt: 'You must provide concise responses.',
                instruction: 'Keep responses very brief.',
                user_prompt: '',
                description: 'Max tokens test'
            };

            await makeRequest(`${API_BASE}/workers`, {
                method: 'POST',
                body: JSON.stringify(workerConfig)
            });

            const messageRequest = {
                message: 'Explain quantum computing in detail.',
                thread_id: uuidv4()
            };

            const response = await makeRequest(`${API_BASE}/workers/${workerName}/message`, {
                method: 'POST',
                body: JSON.stringify(messageRequest)
            });

            results.features_tested.max_tokens = {
                supported: true,
                success: true,
                response_length: response.response?.length || 0,
                likely_truncated: response.response?.length < 200 // Should be truncated with 50 tokens
            };

            // Cleanup
            await makeRequest(`${API_BASE}/workers/${workerName}`, { method: 'DELETE' });

        } catch (error) {
            results.features_tested.max_tokens = {
                supported: true,
                success: false,
                error: error.message
            };
            results.overall_success = false;
        }
    }

    return results;
}

/**
 * Test provider health and availability
 */
async function testProviderHealth() {
    console.log('\n=== Testing Provider Health Endpoint ===');

    try {
        // Check if provider health endpoint exists
        const response = await makeRequest(`${API_BASE}/providers/health`, {}, 1);

        console.log('✅ Provider health endpoint available');
        console.log(`   Providers status:`, JSON.stringify(response, null, 2));

        return {
            endpoint_available: true,
            provider_statuses: response
        };

    } catch (error) {
        if (error.status === 404) {
            console.log('ℹ️ Provider health endpoint not implemented');
            return {
                endpoint_available: false,
                note: 'Endpoint not implemented'
            };
        } else {
            console.error(`❌ Provider health check failed: ${error.message}`);
            return {
                endpoint_available: false,
                error: error.message
            };
        }
    }
}

/**
 * Main test function for model provider coverage
 */
async function runModelProviderCoverageTest() {
    console.log('🚀 Starting Model Provider Coverage Test Suite');
    console.log('='.repeat(80));

    // Reset retry flag for this test run
    hasRetried = false;

    const testResults = {
        provider_tests: {},
        temperature_tests: {},
        base_url_tests: {},
        concurrent_test: null,
        provider_health: null,
        overall_success: false,
        execution_time: 0,
        summary: {
            total_providers: MODEL_PROVIDERS.length,
            providers_tested: 0,
            providers_passed: 0,
            providers_failed: 0,
            providers_skipped: 0
        }
    };

    const testStartTime = Date.now();

    try {
        // Test 1: Provider Health Check
        console.log('\n🏥 Step 1: Provider Health Check');
        testResults.provider_health = await testProviderHealth();

        // Test 2: Basic Provider Functionality
        console.log('\n📋 Step 2: Testing Basic Provider Functionality');

        for (const provider of MODEL_PROVIDERS) {
            console.log('\n' + '='.repeat(60));
            const providerResult = await testBasicProviderFunctionality(provider);
            testResults.provider_tests[provider.provider] = providerResult;

            testResults.summary.providers_tested++;

            if (providerResult.error === 'API_KEY_REQUIRED') {
                console.log(`⚠️ ${provider.provider} skipped - API key required`);
                testResults.summary.providers_skipped++;
            } else if (providerResult.worker_creation && providerResult.message_sending) {
                testResults.summary.providers_passed++;
            } else {
                testResults.summary.providers_failed++;
            }
        }

        // Test 3: Temperature Variations (for working providers)
        console.log('\n📋 Step 3: Testing Temperature Variations');

        const workingProviders = MODEL_PROVIDERS.filter(p => {
            const result = testResults.provider_tests[p.provider];
            return result && result.worker_creation && !result.error;
        });

        for (const provider of workingProviders.slice(0, 2)) { // Test first 2 working providers
            const tempResult = await testTemperatureVariations(provider);
            testResults.temperature_tests[provider.provider] = tempResult;
        }

        // Test 4: Base URL Override
        console.log('\n📋 Step 4: Testing Base URL Override');

        for (const provider of MODEL_PROVIDERS.filter(p => p.baseUrlConfigurable).slice(0, 2)) {
            const baseUrlResult = await testBaseUrlOverride(provider);
            testResults.base_url_tests[provider.provider] = baseUrlResult;
        }

        // Test 5: Concurrent Requests
        console.log('\n📋 Step 5: Testing Concurrent Requests');
        if (workingProviders.length >= 2) {
            testResults.concurrent_test = await testConcurrentRequests(workingProviders);
        } else {
            console.log('⚠️ Not enough working providers for concurrent test');
            testResults.concurrent_test = { skipped: true, reason: 'Insufficient working providers' };
        }

        // Test 6: Provider-Specific Features
        console.log('\n📋 Step 6: Testing Provider-Specific Features');

        for (const provider of workingProviders.slice(0, 2)) {
            const featureResult = await testProviderSpecificFeatures(provider);
            if (!testResults.provider_tests[provider.provider].features) {
                testResults.provider_tests[provider.provider].features = {};
            }
            testResults.provider_tests[provider.provider].features = featureResult.features_tested;
        }

        // Calculate overall success
        testResults.overall_success = testResults.summary.providers_passed > 0 &&
                                     testResults.summary.providers_failed === 0;

        testResults.execution_time = Date.now() - testStartTime;

        // Generate summary report
        console.log('\n📊 Model Provider Coverage Test Summary');
        console.log('='.repeat(80));
        console.log(`🏁 Test Suite - COMPLETE`);
        console.log(`   Total execution time: ${testResults.execution_time}ms`);
        console.log(`   Overall success: ${testResults.overall_success ? '✅ PASSED' : '❌ FAILED'}`);

        console.log('\n📋 Provider Summary:');
        console.log(`   Total Providers: ${testResults.summary.total_providers}`);
        console.log(`   Providers Tested: ${testResults.summary.providers_tested}`);
        console.log(`   Providers Passed: ${testResults.summary.providers_passed}`);
        console.log(`   Providers Failed: ${testResults.summary.providers_failed}`);
        console.log(`   Providers Skipped: ${testResults.summary.providers_skipped}`);

        console.log('\n📋 Individual Provider Results:');
        for (const [provider, result] of Object.entries(testResults.provider_tests)) {
            const status = result.error === 'API_KEY_REQUIRED' ? 'SKIPPED' :
                          (result.worker_creation && result.message_sending ? 'PASSED' : 'FAILED');
            console.log(`   ${provider}: ${status}`);

            if (result.error && result.error !== 'API_KEY_REQUIRED') {
                console.log(`      Error: ${result.error}`);
            }
        }

        if (testResults.overall_success) {
            console.log('\n🎉 Model Provider Coverage Test PASSED!');
            console.log('   ✅ At least one provider is fully functional');
            console.log('   ✅ Basic operations work across providers');
            console.log('   ✅ Provider-specific features tested where applicable');
        } else {
            console.log('\n💥 Model Provider Coverage Test FAILED!');
            console.log('   ❌ No providers are fully functional');
            console.log('   ❌ Check API keys and provider configurations');
        }

    } catch (error) {
        console.error(`\n❌ Test execution failed: ${error.message}`);
        if (error.stack) {
            console.error(`Stack trace: ${error.stack}`);
        }
        testResults.overall_success = false;
        testResults.execution_time = Date.now() - testStartTime;
    }

    return testResults;
}

/**
 * Main function
 */
async function main() {
    console.log('🚀 Model Provider Coverage Test Suite');
    console.log('Node.js version:', process.version);
    console.log('Platform:', process.platform);
    console.log('Architecture:', process.arch);
    console.log('='.repeat(80));

    try {
        const testResults = await runModelProviderCoverageTest();

        // Final summary
        console.log('\n📊 FINAL TEST SUMMARY');
        console.log('='.repeat(80));
        console.log(`Overall Result: ${testResults.overall_success ? '🎉 SUCCESS' : '💥 FAILURE'}`);
        console.log(`Total Execution Time: ${testResults.execution_time}ms`);
        console.log(`Providers Functional: ${testResults.summary.providers_passed}/${testResults.summary.total_providers}`);

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

// Execute if run directly
if (require.main === module) {
    main().catch(error => {
        console.error('❌ Unhandled error in test suite:', error);
        process.exit(1);
    });
}

module.exports = {
    runModelProviderCoverageTest,
    testBasicProviderFunctionality,
    testTemperatureVariations,
    testBaseUrlOverride,
    testConcurrentRequests,
    testProviderSpecificFeatures,
    testProviderHealth,
    MODEL_PROVIDERS
};
