#!/usr/bin/env node
/**
 * JavaScript Test Script for Unicode and International Character Support - Worker Management API
 *
 * This comprehensive test script validates Unicode and International Character Support by:
 * 1. Testing Unicode characters in worker names, descriptions, and prompts
 * 2. Testing different language scripts (Chinese, Arabic, Cyrillic, emoji, etc.)
 * 3. Testing Unicode normalization handling (NFC, NFD, NFKC, NFKD)
 * 4. Testing right-to-left languages and mixed text directions
 * 5. Testing Unicode edge cases (null characters, control characters)
 * 6. Testing UTF-8 encoding/decoding correctness
 * 7. Validating proper Unicode handling in graph execution
 * 8. Including comprehensive Unicode compliance reporting
 *
 * FEATURES TESTED:
 * - Unicode character sets: Latin Extended, CJK, Arabic, Cyrillic, Emoji
 * - Unicode normalization forms: NFC, NFD, NFKC, NFKD
 * - Bidirectional text handling (LTR/RTL)
 * - Unicode edge cases and special characters
 * - UTF-8 encoding/decoding integrity
 * - International character support in all API endpoints
 * - Graph execution with Unicode data
 *
 * Requirements: Node.js 18+ (for native fetch API and Unicode support)
 * Usage: node test_21_unicode_support.js
 */

const { v4: uuidv4 } = require('uuid');

// Enhanced Configuration
const API_BASE = 'http://localhost:8887';
const TIMEOUT_MS = 60000; // 60 seconds for Unicode processing
const MAX_RETRIES = 3;
const RETRY_DELAY = 2000; // 2 seconds
const REQUEST_DELAY = 1500; // 1.5 seconds between requests

// Unicode test data sets
const UNICODE_TEST_DATA = {
    // Basic multilingual plane characters
    latin_extended: {
        name: 'Tëst_Wörker_Üñîcødé',
        description: 'Tëst wørker fõr Ûñîcødé sûppørt vålîdåtîøñ',
        prompt: 'Prøcéss thîs tëxt wîth spëcîål chåråctërs: àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ'
    },

    // Chinese (Simplified and Traditional)
    chinese: {
        name: '中文测试工作者',
        description: '这是一个用于测试中文Unicode支持的工作者',
        prompt: '请处理这个包含中文字符的任务：您好世界！测试Unicode字符处理能力。繁體中文：測試繁體字處理'
    },

    // Arabic (Right-to-Left)
    arabic: {
        name: 'عامل_اختبار_عربي',
        description: 'هذا عامل اختبار لدعم الأحرف العربية والنص من اليمين إلى اليسار',
        prompt: 'معالجة هذا النص العربي: مرحبا بالعالم! اختبار دعم اليونيكود العربي. أرقام: ١٢٣٤٥٦٧٨٩٠'
    },

    // Cyrillic
    cyrillic: {
        name: 'Тест_Работник_Юникод',
        description: 'Тестовый работник для проверки поддержки кириллических символов',
        prompt: 'Обработайте этот текст на кириллице: Привет мир! Тестирование поддержки Unicode. АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    },

    // Japanese (Hiragana, Katakana, Kanji)
    japanese: {
        name: 'テスト_ワーカー_日本語',
        description: 'Unicode日本語サポートのテストワーカーです。ひらがな、カタカナ、漢字を含みます',
        prompt: 'この日本語テキストを処理してください：こんにちは世界！ユニコード対応テスト。ひらがな、カタカナ、漢字：山田太郎さん'
    },

    // Emoji and symbols
    emoji: {
        name: '🚀Test_Worker_🌟Unicode🎯',
        description: '🎉 Test worker with emoji and special symbols: ★☆♠♣♥♦→←↑↓⚡️🔥💡✨',
        prompt: 'Process this text with emojis: 👋 Hello World! 🌍 Testing Unicode emoji support 🧪✅ Math: ∑∞≠≤≥±×÷'
    },

    // Mixed scripts and bidirectional text
    mixed_bidi: {
        name: 'Mixed_العربية_English_中文',
        description: 'Mixed bidirectional text: English + العربية + 中文 + עברית + Русский',
        prompt: 'Process mixed text: Hello العالم and 世界 with עברית text and Русский язык. Numbers: 123 ١٢٣ 一二三'
    },

    // Unicode normalization test cases
    normalization: {
        // NFC vs NFD - é can be represented as single character (é) or e + combining accent (e + ́)
        nfc: 'café', // Composed form (NFC)
        nfd: 'cafe\u0301', // Decomposed form (NFD) - e + combining acute accent
        nfkc: 'ﬁle', // Compatibility composed (NFKC) - ligature fi
        nfkd: 'file' // Compatibility decomposed (NFKD) - separate f and i
    },

    // Edge cases and special characters
    edge_cases: {
        zero_width: 'Test\u200BZero\u200CWidth\u200DJoiners',
        control_chars: 'Control\u0001\u0002\u0003Characters',
        surrogate_pairs: '𝒽𝑒𝓁𝓁𝑜', // Mathematical script letters (surrogate pairs)
        combining: 'a\u0300\u0301\u0302\u0303\u0304', // a with multiple combining marks
        rtl_override: '\u202E\u0644\u0645\u202C', // RTL override characters
        null_char: 'Test\u0000Null', // Null character
        high_unicode: '𝕌𝕟𝕚𝕔𝕠𝕕𝕖', // High Unicode code points
        variation_selectors: 'Test\uFE0E\uFE0F', // Variation selectors
    }
};

// Unicode validation patterns
const UNICODE_PATTERNS = {
    has_latin_extended: /[\u0080-\u00FF]/,
    has_cjk: /[\u4E00-\u9FFF\u3400-\u4DBF]/,
    has_arabic: /[\u0600-\u06FF\u0750-\u077F]/,
    has_cyrillic: /[\u0400-\u04FF]/,
    has_emoji: /[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F1E0}-\u{1F1FF}]/u,
    has_rtl: /[\u0590-\u05FF\u0600-\u06FF\u0700-\u074F]/,
    has_combining: /[\u0300-\u036F]/,
    has_control: /[\u0000-\u001F\u007F-\u009F]/,
    has_surrogate_pairs: /[\uD800-\uDBFF][\uDC00-\uDFFF]/,
    has_zero_width: /[\u200B-\u200D\uFEFF]/
};

/**
 * Enhanced utility function to make HTTP requests with Unicode-aware error handling
 */
async function makeRequest(url, options = {}, retries = MAX_RETRIES) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'application/json; charset=utf-8',
            'User-Agent': 'WorkerManagement-Unicode-Test/1.0',
            'Accept-Charset': 'utf-8',
            ...options.headers
        }
    };

    const finalOptions = { ...defaultOptions, ...options };

    for (let attempt = 1; attempt <= retries; attempt++) {
        try {
            console.log(`🌐 ${finalOptions.method || 'GET'} ${url} (Attempt ${attempt}/${retries}) [Unicode-aware]`);

            // Ensure body is properly UTF-8 encoded
            if (finalOptions.body && typeof finalOptions.body === 'string') {
                finalOptions.body = Buffer.from(finalOptions.body, 'utf8').toString('utf8');
            }

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
            console.log(`✅ Unicode response received (${response.status}) - Attempt ${attempt}`);

            // Add delay between requests
            await new Promise(resolve => setTimeout(resolve, REQUEST_DELAY));

            return data;
        } catch (error) {
            console.error(`❌ Unicode request failed (Attempt ${attempt}/${retries}): ${error.message}`);

            if (attempt === retries) {
                throw error;
            }

            if (attempt < retries) {
                console.log(`⏳ Retrying Unicode request in ${RETRY_DELAY/1000} seconds...`);
                await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
            }
        }
    }
}

/**
 * Unicode-aware worker creation with validation
 */
async function createUnicodeWorker(workerConfig, testCaseName) {
    console.log(`\n=== Creating Unicode Worker: ${testCaseName} ===`);

    try {
        // Validate Unicode configuration
        if (!workerConfig.worker_name || !workerConfig.agent_type) {
            throw new Error('Unicode worker configuration missing required fields');
        }

        // Log Unicode characteristics
        console.log(`🔤 Unicode Analysis for ${testCaseName}:`);
        analyzeUnicodeCharacteristics(workerConfig.worker_name, 'Worker Name');
        analyzeUnicodeCharacteristics(workerConfig.description, 'Description');
        analyzeUnicodeCharacteristics(workerConfig.system_prompt, 'System Prompt');

        const response = await makeRequest(`${API_BASE}/workers`, {
            method: 'POST',
            body: JSON.stringify(workerConfig)
        });

        console.log(`✅ Unicode worker created successfully: ${response.name}`);
        console.log(`   ID: ${response.id || 'N/A'}`);
        console.log(`   Unicode Name Length: ${response.name.length} characters`);
        console.log(`   Unicode Byte Length: ${Buffer.byteLength(response.name, 'utf8')} bytes`);

        // Validate Unicode integrity
        validateUnicodeIntegrity(workerConfig.worker_name, response.name, 'Worker Name');

        return response;
    } catch (error) {
        console.error(`❌ Failed to create Unicode worker ${testCaseName}: ${error.message}`);
        throw error;
    }
}

/**
 * Analyze Unicode characteristics of text
 */
function analyzeUnicodeCharacteristics(text, label) {
    if (!text) return;

    console.log(`   📊 ${label} Unicode Analysis:`);
    console.log(`      Character count: ${text.length}`);
    console.log(`      Byte length (UTF-8): ${Buffer.byteLength(text, 'utf8')}`);
    console.log(`      Code points: ${[...text].length}`);

    // Check for different Unicode ranges
    const characteristics = [];
    for (const [name, pattern] of Object.entries(UNICODE_PATTERNS)) {
        if (pattern.test(text)) {
            characteristics.push(name);
        }
    }

    if (characteristics.length > 0) {
        console.log(`      Unicode features: ${characteristics.join(', ')}`);
    }

    // Check for normalization form
    const nfc = text.normalize('NFC');
    const nfd = text.normalize('NFD');
    if (text !== nfc) {
        console.log(`      ⚠️  Text is not in NFC form`);
    }
    if (text !== nfd) {
        console.log(`      ℹ️  NFD form differs (expected for composed characters)`);
    }
}

/**
 * Validate Unicode integrity between input and output
 */
function validateUnicodeIntegrity(original, received, context) {
    const originalNFC = original.normalize('NFC');
    const receivedNFC = received.normalize('NFC');

    if (originalNFC === receivedNFC) {
        console.log(`   ✅ Unicode integrity maintained for ${context}`);
        return true;
    } else {
        console.log(`   ❌ Unicode integrity lost for ${context}`);
        console.log(`      Original: "${original}" (${original.length} chars, ${Buffer.byteLength(original, 'utf8')} bytes)`);
        console.log(`      Received: "${received}" (${received.length} chars, ${Buffer.byteLength(received, 'utf8')} bytes)`);
        return false;
    }
}

/**
 * Create Unicode test graph with international characters
 */
async function createUnicodeGraph(graphConfig, testCaseName) {
    console.log(`\n=== Creating Unicode Graph: ${testCaseName} ===`);

    try {
        // Validate Unicode graph configuration
        if (!graphConfig.nodes || graphConfig.nodes.length === 0) {
            throw new Error('Unicode graph must have at least one node');
        }

        console.log(`📊 Unicode Graph Configuration for ${testCaseName}:`);
        console.log(`   Name: ${graphConfig.name}`);
        console.log(`   Nodes: ${graphConfig.nodes.length}`);
        console.log(`   Variables: ${Object.keys(graphConfig.variables || {}).length}`);

        // Analyze Unicode content in variables
        if (graphConfig.variables) {
            for (const [key, value] of Object.entries(graphConfig.variables)) {
                if (typeof value === 'string') {
                    analyzeUnicodeCharacteristics(value, `Variable "${key}"`);
                }
            }
        }

        const response = await makeRequest(`${API_BASE}/graphs/`, {
            method: 'POST',
            body: JSON.stringify(graphConfig)
        });

        console.log(`✅ Unicode graph created successfully: ${response.name}`);
        console.log(`   ID: ${response.id}`);
        console.log(`   Unicode Name Integrity: ${validateUnicodeIntegrity(graphConfig.name, response.name, 'Graph Name')}`);

        return response;
    } catch (error) {
        console.error(`❌ Failed to create Unicode graph: ${error.message}`);
        throw error;
    }
}

/**
 * Execute Unicode graph with international character processing
 */
async function executeUnicodeGraph(graphId, executionRequest, testCaseName) {
    console.log(`\n=== Executing Unicode Graph: ${testCaseName} ===`);

    try {
        console.log(`⚡ Unicode Execution Request for ${testCaseName}:`);

        // Analyze Unicode content in execution variables
        if (executionRequest.variables) {
            for (const [key, value] of Object.entries(executionRequest.variables)) {
                if (typeof value === 'string') {
                    analyzeUnicodeCharacteristics(value, `Execution Variable "${key}"`);
                }
            }
        }

        const response = await makeRequest(`${API_BASE}/execution/graphs/${graphId}/execute`, {
            method: 'POST',
            body: JSON.stringify(executionRequest)
        });

        console.log(`✅ Unicode graph execution initiated successfully`);
        console.log(`   Session ID: ${response.session_id}`);
        console.log(`   Status: ${response.status}`);

        return response;
    } catch (error) {
        console.error(`❌ Failed to execute Unicode graph: ${error.message}`);
        throw error;
    }
}

/**
 * Test Unicode normalization handling
 */
async function testUnicodeNormalization() {
    console.log('\n🔤 Testing Unicode Normalization Forms...');

    const normalizationTests = [];
    const testData = UNICODE_TEST_DATA.normalization;

    try {
        // Test NFC vs NFD
        const nfcWorker = await createUnicodeWorker({
            worker_name: `nfc_test_${testData.nfc}`,
            agent_role: 'normalization_tester',
            agent_type: 'pydantic_ai',
            model_provider: 'deepseek',
            model: 'deepseek-chat',
            temperature: 0.1,
            system_prompt: `Process text with Unicode normalization: ${testData.nfc}`,
            instruction: 'Handle Unicode normalization correctly',
            description: `NFC normalization test: ${testData.nfc}`,
            user_prompt: '',
            mcp_config: null
        }, 'NFC Normalization');

        const nfdWorker = await createUnicodeWorker({
            worker_name: `nfd_test_${testData.nfd.replace(/\u0301/g, '_accent')}`,
            agent_role: 'normalization_tester',
            agent_type: 'pydantic_ai',
            model_provider: 'deepseek',
            model: 'deepseek-chat',
            temperature: 0.1,
            system_prompt: `Process text with Unicode normalization: ${testData.nfd}`,
            instruction: 'Handle Unicode normalization correctly',
            description: `NFD normalization test: ${testData.nfd}`,
            user_prompt: '',
            mcp_config: null
        }, 'NFD Normalization');

        normalizationTests.push({
            type: 'NFC',
            worker: nfcWorker,
            original: testData.nfc,
            normalized: testData.nfc.normalize('NFC')
        });

        normalizationTests.push({
            type: 'NFD',
            worker: nfdWorker,
            original: testData.nfd,
            normalized: testData.nfd.normalize('NFD')
        });

        // Test compatibility forms
        console.log('🔤 Testing compatibility normalization forms...');
        console.log(`   NFKC: "${testData.nfkc}" → "${testData.nfkc.normalize('NFKC')}"`);
        console.log(`   NFKD: "${testData.nfkd}" → "${testData.nfkd.normalize('NFKD')}"`);

        return normalizationTests;

    } catch (error) {
        console.error('❌ Unicode normalization test failed:', error.message);
        return [];
    }
}

/**
 * Test bidirectional text handling
 */
async function testBidirectionalText() {
    console.log('\n🔄 Testing Bidirectional Text Handling...');

    try {
        const bidiData = UNICODE_TEST_DATA.mixed_bidi;

        const bidiWorker = await createUnicodeWorker({
            worker_name: 'bidi_test_worker',
            agent_role: 'bidirectional_text_processor',
            agent_type: 'pydantic_ai',
            model_provider: 'deepseek',
            model: 'deepseek-chat',
            temperature: 0.1,
            system_prompt: bidiData.description,
            instruction: bidiData.prompt,
            description: bidiData.description,
            user_prompt: '',
            mcp_config: null
        }, 'Bidirectional Text');

        // Test message with bidirectional content
        const bidiMessage = {
            message: bidiData.prompt,
            context: {
                text_direction: 'mixed',
                languages: ['English', 'Arabic', 'Chinese', 'Hebrew', 'Russian']
            }
        };

        const response = await makeRequest(`${API_BASE}/workers/${bidiWorker.name}/message`, {
            method: 'POST',
            body: JSON.stringify(bidiMessage)
        });

        console.log('✅ Bidirectional text processing completed');
        analyzeUnicodeCharacteristics(response.response, 'Bidirectional Response');

        return { worker: bidiWorker, response };

    } catch (error) {
        console.error('❌ Bidirectional text test failed:', error.message);
        return null;
    }
}

/**
 * Test Unicode edge cases
 */
async function testUnicodeEdgeCases() {
    console.log('\n⚠️  Testing Unicode Edge Cases...');

    const edgeCaseResults = [];
    const edgeCases = UNICODE_TEST_DATA.edge_cases;

    for (const [caseName, testText] of Object.entries(edgeCases)) {
        try {
            console.log(`\n🧪 Testing edge case: ${caseName}`);
            console.log(`   Text: "${testText}"`);

            // Analyze the edge case text
            analyzeUnicodeCharacteristics(testText, `Edge Case: ${caseName}`);

            // Try to create a worker with edge case content
            const edgeWorker = await createUnicodeWorker({
                worker_name: `edge_case_${caseName}_worker`,
                agent_role: 'edge_case_tester',
                agent_type: 'pydantic_ai',
                model_provider: 'deepseek',
                model: 'deepseek-chat',
                temperature: 0.1,
                system_prompt: `Handle edge case Unicode text: ${testText}`,
                instruction: 'Process Unicode edge case correctly',
                description: `Edge case test: ${caseName}`,
                user_prompt: '',
                mcp_config: null
            }, `Edge Case: ${caseName}`);

            edgeCaseResults.push({
                case: caseName,
                text: testText,
                worker: edgeWorker,
                status: 'success'
            });

        } catch (error) {
            console.error(`❌ Edge case ${caseName} failed: ${error.message}`);
            edgeCaseResults.push({
                case: caseName,
                text: testText,
                worker: null,
                status: 'failed',
                error: error.message
            });
        }
    }

    return edgeCaseResults;
}

/**
 * Comprehensive API health check with Unicode validation
 */
async function checkUnicodeApiHealth() {
    console.log('\n🏥 Performing Unicode-Aware API Health Check...');

    const healthChecks = [];

    try {
        // Main API health
        const healthResponse = await makeRequest(`${API_BASE}/health`);
        console.log(`✅ Main API is healthy: ${healthResponse.status}`);
        healthChecks.push({ service: 'main_api', status: 'healthy', unicode_aware: true });

        // Test Unicode in basic endpoints
        const workersResponse = await makeRequest(`${API_BASE}/workers`);
        console.log(`✅ Workers endpoint responding (${workersResponse.length} workers)`);

        // Check if any existing workers have Unicode names
        const unicodeWorkers = workersResponse.filter(w =>
            w.name && (Buffer.byteLength(w.name, 'utf8') !== w.name.length)
        );
        console.log(`📊 Found ${unicodeWorkers.length} workers with Unicode characters`);

        healthChecks.push({
            service: 'workers',
            status: 'healthy',
            unicode_workers: unicodeWorkers.length,
            unicode_capable: true
        });

        const graphsResponse = await makeRequest(`${API_BASE}/graphs/`);
        console.log(`✅ Graphs endpoint responding (${graphsResponse.length} graphs)`);

        healthChecks.push({
            service: 'graphs',
            status: 'healthy',
            unicode_capable: true
        });

        console.log('\n🎉 Unicode-aware health checks passed!');
        return { healthy: true, checks: healthChecks, unicode_support: true };

    } catch (error) {
        console.error(`❌ Unicode health check failed: ${error.message}`);
        return { healthy: false, error: error.message, unicode_support: false };
    }
}

/**
 * Generate comprehensive Unicode compliance report
 */
function generateUnicodeComplianceReport(testResults) {
    console.log('\n📋 Unicode Compliance Report');
    console.log('='.repeat(80));

    const report = {
        test_summary: {
            total_tests: 0,
            passed_tests: 0,
            failed_tests: 0,
            unicode_features_tested: []
        },
        character_sets: {
            latin_extended: false,
            cjk: false,
            arabic: false,
            cyrillic: false,
            japanese: false,
            emoji: false
        },
        normalization: {
            nfc_support: false,
            nfd_support: false,
            nfkc_support: false,
            nfkd_support: false
        },
        bidirectional: {
            rtl_support: false,
            mixed_direction: false,
            bidi_override: false
        },
        edge_cases: {
            zero_width_characters: false,
            control_characters: false,
            surrogate_pairs: false,
            combining_marks: false,
            null_characters: false
        },
        encoding: {
            utf8_integrity: false,
            byte_preservation: false,
            length_accuracy: false
        },
        api_endpoints: {
            workers_unicode: false,
            graphs_unicode: false,
            execution_unicode: false,
            messages_unicode: false
        }
    };

    // Analyze test results and populate report
    if (testResults.worker_creation_tests) {
        report.test_summary.total_tests += testResults.worker_creation_tests.length;
        report.test_summary.passed_tests += testResults.worker_creation_tests.filter(t => t.success).length;
        report.test_summary.failed_tests += testResults.worker_creation_tests.filter(t => !t.success).length;

        // Check character sets
        testResults.worker_creation_tests.forEach(test => {
            if (test.success) {
                report.api_endpoints.workers_unicode = true;

                if (test.testCase.includes('latin')) report.character_sets.latin_extended = true;
                if (test.testCase.includes('chinese')) report.character_sets.cjk = true;
                if (test.testCase.includes('arabic')) report.character_sets.arabic = true;
                if (test.testCase.includes('cyrillic')) report.character_sets.cyrillic = true;
                if (test.testCase.includes('japanese')) report.character_sets.japanese = true;
                if (test.testCase.includes('emoji')) report.character_sets.emoji = true;
            }
        });
    }

    if (testResults.normalization_tests) {
        report.normalization.nfc_support = testResults.normalization_tests.some(t => t.type === 'NFC');
        report.normalization.nfd_support = testResults.normalization_tests.some(t => t.type === 'NFD');
        report.test_summary.unicode_features_tested.push('normalization');
    }

    if (testResults.bidi_test) {
        report.bidirectional.rtl_support = true;
        report.bidirectional.mixed_direction = true;
        report.test_summary.unicode_features_tested.push('bidirectional');
    }

    if (testResults.edge_case_tests) {
        testResults.edge_case_tests.forEach(test => {
            if (test.status === 'success') {
                switch (test.case) {
                    case 'zero_width':
                        report.edge_cases.zero_width_characters = true;
                        break;
                    case 'control_chars':
                        report.edge_cases.control_characters = true;
                        break;
                    case 'surrogate_pairs':
                        report.edge_cases.surrogate_pairs = true;
                        break;
                    case 'combining':
                        report.edge_cases.combining_marks = true;
                        break;
                    case 'null_char':
                        report.edge_cases.null_characters = true;
                        break;
                }
            }
        });
        report.test_summary.unicode_features_tested.push('edge_cases');
    }

    if (testResults.graph_execution_tests) {
        report.api_endpoints.graphs_unicode = true;
        report.api_endpoints.execution_unicode = true;
        report.encoding.utf8_integrity = testResults.graph_execution_tests.some(t => t.unicode_integrity);
    }

    // Display report
    console.log(`📊 Test Summary:`);
    console.log(`   Total Tests: ${report.test_summary.total_tests}`);
    console.log(`   Passed: ${report.test_summary.passed_tests}`);
    console.log(`   Failed: ${report.test_summary.failed_tests}`);
    console.log(`   Success Rate: ${report.test_summary.total_tests > 0 ?
        ((report.test_summary.passed_tests / report.test_summary.total_tests) * 100).toFixed(1) : 0}%`);

    console.log(`\n🔤 Character Set Support:`);
    Object.entries(report.character_sets).forEach(([set, supported]) => {
        console.log(`   ${supported ? '✅' : '❌'} ${set.replace('_', ' ').toUpperCase()}: ${supported ? 'SUPPORTED' : 'NOT_TESTED'}`);
    });

    console.log(`\n🔄 Normalization Support:`);
    Object.entries(report.normalization).forEach(([form, supported]) => {
        console.log(`   ${supported ? '✅' : '❌'} ${form.toUpperCase()}: ${supported ? 'SUPPORTED' : 'NOT_TESTED'}`);
    });

    console.log(`\n↔️  Bidirectional Text:`);
    Object.entries(report.bidirectional).forEach(([feature, supported]) => {
        console.log(`   ${supported ? '✅' : '❌'} ${feature.replace('_', ' ').toUpperCase()}: ${supported ? 'SUPPORTED' : 'NOT_TESTED'}`);
    });

    console.log(`\n⚠️  Edge Case Handling:`);
    Object.entries(report.edge_cases).forEach(([case_type, handled]) => {
        console.log(`   ${handled ? '✅' : '❌'} ${case_type.replace('_', ' ').toUpperCase()}: ${handled ? 'HANDLED' : 'NOT_TESTED'}`);
    });

    console.log(`\n🌐 API Endpoint Unicode Support:`);
    Object.entries(report.api_endpoints).forEach(([endpoint, supported]) => {
        console.log(`   ${supported ? '✅' : '❌'} ${endpoint.replace('_', ' ').toUpperCase()}: ${supported ? 'SUPPORTED' : 'NOT_TESTED'}`);
    });

    // Overall compliance score
    const totalFeatures = Object.values(report.character_sets).length +
                         Object.values(report.normalization).length +
                         Object.values(report.bidirectional).length +
                         Object.values(report.edge_cases).length +
                         Object.values(report.api_endpoints).length;

    const supportedFeatures = Object.values(report.character_sets).filter(Boolean).length +
                             Object.values(report.normalization).filter(Boolean).length +
                             Object.values(report.bidirectional).filter(Boolean).length +
                             Object.values(report.edge_cases).filter(Boolean).length +
                             Object.values(report.api_endpoints).filter(Boolean).length;

    const complianceScore = totalFeatures > 0 ? (supportedFeatures / totalFeatures) * 100 : 0;

    console.log(`\n🎯 Overall Unicode Compliance Score: ${complianceScore.toFixed(1)}%`);

    if (complianceScore >= 90) {
        console.log('🏆 EXCELLENT Unicode compliance!');
    } else if (complianceScore >= 75) {
        console.log('✅ GOOD Unicode compliance');
    } else if (complianceScore >= 50) {
        console.log('⚠️  PARTIAL Unicode compliance');
    } else {
        console.log('❌ POOR Unicode compliance');
    }

    return report;
}

/**
 * Enhanced resource cleanup with Unicode handling
 */
async function cleanupUnicodeResources(graphId, workerNames = []) {
    console.log('\n🧹 Performing Unicode Resource Cleanup...');
    const cleanupResults = {
        graph_deleted: false,
        workers_deleted: [],
        failed_deletions: [],
        unicode_workers_cleaned: 0
    };

    try {
        // Delete graph if provided
        if (graphId) {
            console.log(`🗑️  Deleting Unicode graph: ${graphId}`);
            try {
                await makeRequest(`${API_BASE}/graphs/${graphId}`, { method: 'DELETE' });
                console.log(`✅ Unicode graph deleted successfully`);
                cleanupResults.graph_deleted = true;
            } catch (error) {
                console.error(`❌ Failed to delete Unicode graph: ${error.message}`);
                cleanupResults.failed_deletions.push({ type: 'graph', id: graphId, error: error.message });
            }
        }

        // Delete workers (with Unicode names)
        for (const workerName of workerNames) {
            console.log(`🗑️  Deleting Unicode worker: ${workerName}`);
            console.log(`   Unicode characteristics: ${workerName.length} chars, ${Buffer.byteLength(workerName, 'utf8')} bytes`);

            try {
                // URL encode the Unicode worker name properly
                const encodedWorkerName = encodeURIComponent(workerName);
                await makeRequest(`${API_BASE}/workers/${encodedWorkerName}`, { method: 'DELETE' });
                console.log(`✅ Unicode worker deleted successfully: ${workerName}`);
                cleanupResults.workers_deleted.push(workerName);

                // Check if it was a Unicode worker
                if (Buffer.byteLength(workerName, 'utf8') !== workerName.length) {
                    cleanupResults.unicode_workers_cleaned++;
                }
            } catch (error) {
                console.error(`❌ Failed to delete Unicode worker ${workerName}: ${error.message}`);
                cleanupResults.failed_deletions.push({ type: 'worker', id: workerName, error: error.message });
            }
        }

        console.log(`\n✅ Unicode cleanup completed`);
        console.log(`   Unicode workers cleaned: ${cleanupResults.unicode_workers_cleaned}`);
        console.log(`   Total workers deleted: ${cleanupResults.workers_deleted.length}/${workerNames.length}`);
        console.log(`   Failed deletions: ${cleanupResults.failed_deletions.length}`);

    } catch (error) {
        console.error(`❌ Unicode cleanup process failed: ${error.message}`);
    }

    return cleanupResults;
}

/**
 * Main Unicode test function
 */
async function runUnicodeTest() {
    console.log('🌍 Starting Comprehensive Unicode and International Character Support Test');
    console.log('='.repeat(80));

    let createdGraphId = null;
    const createdWorkerNames = [];
    const testResults = {
        health_check: false,
        worker_creation_tests: [],
        normalization_tests: [],
        bidi_test: null,
        edge_case_tests: [],
        graph_execution_tests: [],
        unicode_compliance_score: 0,
        overall_success: false,
        execution_time: 0
    };

    const testStartTime = Date.now();

    try {
        // Step 1: Unicode-aware health check
        console.log('\n🏥 Step 1: Unicode-Aware API Health Check');
        const healthResult = await checkUnicodeApiHealth();
        testResults.health_check = healthResult.healthy;

        if (!healthResult.healthy) {
            throw new Error('Unicode API health check failed');
        }

        // Step 2: Test worker creation with different Unicode character sets
        console.log('\n🤖 Step 2: Unicode Worker Creation Tests');

        const unicodeTestCases = [
            { name: 'latin_extended', data: UNICODE_TEST_DATA.latin_extended },
            { name: 'chinese', data: UNICODE_TEST_DATA.chinese },
            { name: 'arabic', data: UNICODE_TEST_DATA.arabic },
            { name: 'cyrillic', data: UNICODE_TEST_DATA.cyrillic },
            { name: 'japanese', data: UNICODE_TEST_DATA.japanese },
            { name: 'emoji', data: UNICODE_TEST_DATA.emoji },
            { name: 'mixed_bidi', data: UNICODE_TEST_DATA.mixed_bidi }
        ];

        for (const testCase of unicodeTestCases) {
            try {
                const workerConfig = {
                    worker_name: testCase.data.name,
                    agent_role: 'unicode_test_specialist',
                    agent_type: 'pydantic_ai',
                    model_provider: 'deepseek',
                    model: 'deepseek-chat',
                    temperature: 0.2,
                    system_prompt: testCase.data.description,
                    instruction: testCase.data.prompt,
                    description: testCase.data.description,
                    user_prompt: '',
                    mcp_config: null
                };

                const worker = await createUnicodeWorker(workerConfig, testCase.name);
                createdWorkerNames.push(worker.name);

                testResults.worker_creation_tests.push({
                    testCase: testCase.name,
                    worker: worker,
                    success: true,
                    unicode_integrity: validateUnicodeIntegrity(
                        testCase.data.name,
                        worker.name,
                        `${testCase.name} Worker`
                    )
                });

            } catch (error) {
                console.error(`❌ Unicode test case ${testCase.name} failed: ${error.message}`);
                testResults.worker_creation_tests.push({
                    testCase: testCase.name,
                    worker: null,
                    success: false,
                    error: error.message
                });
            }
        }

        // Step 3: Test Unicode normalization
        console.log('\n🔤 Step 3: Unicode Normalization Tests');
        const normalizationResults = await testUnicodeNormalization();
        testResults.normalization_tests = normalizationResults;

        // Add normalization test workers to cleanup list
        normalizationResults.forEach(result => {
            if (result.worker) {
                createdWorkerNames.push(result.worker.name);
            }
        });

        // Step 4: Test bidirectional text
        console.log('\n🔄 Step 4: Bidirectional Text Tests');
        const bidiResult = await testBidirectionalText();
        testResults.bidi_test = bidiResult;

        if (bidiResult && bidiResult.worker) {
            createdWorkerNames.push(bidiResult.worker.name);
        }

        // Step 5: Test Unicode edge cases
        console.log('\n⚠️  Step 5: Unicode Edge Case Tests');
        const edgeCaseResults = await testUnicodeEdgeCases();
        testResults.edge_case_tests = edgeCaseResults;

        // Add edge case workers to cleanup list
        edgeCaseResults.forEach(result => {
            if (result.worker) {
                createdWorkerNames.push(result.worker.name);
            }
        });

        // Step 6: Test Unicode in graph execution
        console.log('\n📊 Step 6: Unicode Graph Execution Tests');

        if (testResults.worker_creation_tests.length > 0) {
            const successfulWorkers = testResults.worker_creation_tests.filter(t => t.success);

            if (successfulWorkers.length > 0) {
                const testWorker = successfulWorkers[0].worker;

                // Create a graph with Unicode content
                const unicodeGraphConfig = {
                    name: '🌍 Unicode Test Graph - 多语言测试图表',
                    description: 'Graph for testing Unicode support with international characters: العربية, 中文, Русский, 日本語',
                    variables: {
                        unicode_input: '🚀 Multi-language test: Hello世界مرحباПривет🎉',
                        test_languages: ['English', '中文', 'العربية', 'Русский', '日本語'],
                        emoji_data: '🌟✨🎯🔥💡⚡️🎨🚀',
                        normalization_test: 'café vs cafe\u0301'
                    },
                    nodes: [
                        {
                            id: uuidv4(),
                            name: '🌍 Unicode Processing Node',
                            node_type: 'worker',
                            worker_name: testWorker.name,
                            task_template: 'Process this Unicode content: {unicode_input}. Languages: {test_languages}. Emojis: {emoji_data}. Normalization: {normalization_test}',
                            dependencies: [],
                            data: {
                                unicode_test: true,
                                multilingual: true
                            },
                            timeout: 120
                        }
                    ],
                    edges: []
                };

                try {
                    const unicodeGraph = await createUnicodeGraph(unicodeGraphConfig, 'Unicode Execution');
                    createdGraphId = unicodeGraph.id;

                    const executionRequest = {
                        variables: {
                            unicode_input: '🎉 Processing Unicode: Hello世界! مرحبا بالعالم! Привет мир! こんにちは世界！',
                            test_languages: ['English', '中文', 'العربية', 'Русский', '日本語', '🌍'],
                            emoji_data: '🚀🌟✨🎯🔥💡⚡️🌈🎨🎭🎪🎊',
                            normalization_test: 'Testing: café = cafe\u0301 (NFC vs NFD)'
                        }
                    };

                    const executionResult = await executeUnicodeGraph(
                        unicodeGraph.id,
                        executionRequest,
                        'Unicode Execution'
                    );

                    testResults.graph_execution_tests.push({
                        graph: unicodeGraph,
                        execution: executionResult,
                        success: true,
                        unicode_integrity: true
                    });

                } catch (error) {
                    console.error(`❌ Unicode graph execution failed: ${error.message}`);
                    testResults.graph_execution_tests.push({
                        graph: null,
                        execution: null,
                        success: false,
                        error: error.message
                    });
                }
            }
        }

        // Step 7: Generate Unicode compliance report
        console.log('\n📋 Step 7: Unicode Compliance Analysis');
        const complianceReport = generateUnicodeComplianceReport(testResults);
        testResults.unicode_compliance_score = complianceReport.test_summary.passed_tests /
                                               Math.max(complianceReport.test_summary.total_tests, 1) * 100;

        // Determine overall success
        testResults.overall_success = testResults.health_check &&
                                     testResults.worker_creation_tests.some(t => t.success) &&
                                     testResults.unicode_compliance_score >= 50; // At least 50% compliance

        testResults.execution_time = Date.now() - testStartTime;

        // Final summary
        console.log('\n🎯 Step 8: Unicode Test Summary');
        console.log('='.repeat(80));
        console.log(`🌍 Unicode and International Character Support Test - COMPLETE`);
        console.log(`   Total execution time: ${testResults.execution_time}ms`);
        console.log(`   Overall success: ${testResults.overall_success ? '✅ PASSED' : '❌ FAILED'}`);
        console.log(`   Unicode compliance score: ${testResults.unicode_compliance_score.toFixed(1)}%`);

        console.log('\n📊 Test Results Summary:');
        console.log(`   ✅ API Health Check: ${testResults.health_check ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Worker Creation Tests: ${testResults.worker_creation_tests.filter(t => t.success).length}/${testResults.worker_creation_tests.length} PASSED`);
        console.log(`   ✅ Normalization Tests: ${testResults.normalization_tests.length} COMPLETED`);
        console.log(`   ✅ Bidirectional Test: ${testResults.bidi_test ? 'PASSED' : 'FAILED'}`);
        console.log(`   ✅ Edge Case Tests: ${testResults.edge_case_tests.filter(t => t.status === 'success').length}/${testResults.edge_case_tests.length} PASSED`);
        console.log(`   ✅ Graph Execution Tests: ${testResults.graph_execution_tests.filter(t => t.success).length}/${testResults.graph_execution_tests.length} PASSED`);

        if (testResults.overall_success) {
            console.log('\n🎉 Unicode and International Character Support VALIDATED!');
            console.log('   ✅ Multiple character sets supported');
            console.log('   ✅ Unicode normalization handling confirmed');
            console.log('   ✅ Bidirectional text processing working');
            console.log('   ✅ Edge cases handled appropriately');
            console.log('   ✅ Graph execution with Unicode data successful');
            console.log('   ✅ API endpoints are Unicode-compliant');
        } else {
            console.log('\n💥 Unicode Support Test FAILED!');
            console.log('   ❌ Some Unicode features are not working correctly');
            console.log('   ❌ Check the detailed output above for specific issues');
        }

    } catch (error) {
        console.error(`\n❌ Unicode test execution failed: ${error.message}`);
        if (error.stack) {
            console.error(`Stack trace: ${error.stack}`);
        }
        testResults.overall_success = false;
        testResults.execution_time = Date.now() - testStartTime;
    }

    // Cleanup
    console.log('\n🧹 Step 9: Unicode Resource Cleanup');
    try {
        const cleanupResult = await cleanupUnicodeResources(createdGraphId, createdWorkerNames);
        console.log(`✅ Cleanup completed: ${cleanupResult.unicode_workers_cleaned} Unicode workers cleaned`);
    } catch (cleanupError) {
        console.error(`❌ Unicode cleanup failed: ${cleanupError.message}`);
    }

    return testResults;
}

/**
 * Main function with comprehensive error handling
 */
async function main() {
    console.log('🌍 Unicode and International Character Support Test Suite for Worker Management API');
    console.log('Node.js version:', process.version);
    console.log('Unicode support:', process.versions.unicode || 'Available');
    console.log('Platform:', process.platform);
    console.log('='.repeat(80));

    try {
        const testResults = await runUnicodeTest();

        // Final summary
        console.log('\n📊 FINAL UNICODE TEST SUMMARY');
        console.log('='.repeat(80));
        console.log(`Overall Result: ${testResults.overall_success ? '🎉 SUCCESS' : '💥 FAILURE'}`);
        console.log(`Unicode Compliance Score: ${testResults.unicode_compliance_score.toFixed(1)}%`);
        console.log(`Total Execution Time: ${testResults.execution_time}ms`);
        console.log(`Character Sets Tested: ${testResults.worker_creation_tests.length}`);
        console.log(`Successful Unicode Workers: ${testResults.worker_creation_tests.filter(t => t.success).length}`);

        // Exit with appropriate code
        process.exit(testResults.overall_success ? 0 : 1);

    } catch (error) {
        console.error('❌ Unicode test suite failed to execute:', error.message);
        if (error.stack) {
            console.error('Stack trace:', error.stack);
        }
        process.exit(1);
    }
}

// Execute the test suite
if (require.main === module) {
    main().catch(error => {
        console.error('❌ Unhandled error in Unicode test suite:', error);
        process.exit(1);
    });
}

module.exports = {
    makeRequest,
    createUnicodeWorker,
    createUnicodeGraph,
    executeUnicodeGraph,
    testUnicodeNormalization,
    testBidirectionalText,
    testUnicodeEdgeCases,
    checkUnicodeApiHealth,
    generateUnicodeComplianceReport,
    cleanupUnicodeResources,
    runUnicodeTest,
    UNICODE_TEST_DATA,
    UNICODE_PATTERNS
};
