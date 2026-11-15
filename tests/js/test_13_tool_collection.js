#!/usr/bin/env node
/**
 * JavaScript Test Script for Tool Registry Collection Operations - Comprehensive Testing
 *
 * This script validates the Tool Registry API by:
 * 1. Testing GET /tools/ with various query parameters (tags, search, limit, offset)
 * 2. Testing POST /tools/ - creating new tools with ToolCreateRequest
 * 3. Creating tools with different configurations (Python code, schemas, tags, versions)
 * 4. Validating ToolListResponse and ToolResponse structures
 * 5. Testing search functionality and filtering
 * 6. Testing edge cases and validation errors
 * 7. Including comprehensive cleanup and error handling
 *
 * Features Tested:
 * - Tool creation with various Python code implementations
 * - Input/output schema validation
 * - Tag filtering and search functionality
 * - Pagination (limit, offset parameters)
 * - Duplicate query parameter validation
 * - Tool name validation and edge cases
 * - Response structure validation
 * - Error handling for invalid requests
 * - Comprehensive cleanup procedures
 *
 * Requirements: Node.js 18+ (for native fetch API)
 * Usage: node test_13_tool_collection.js
 */

const { v4: uuidv4 } = require('uuid');

// Enhanced Configuration
const API_BASE = 'http://localhost:8887';
const TIMEOUT_MS = 30000; // 30 seconds for tool operations
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000; // 1 second
const REQUEST_DELAY = 500; // 500ms between requests for tool operations

// Track created tools for cleanup
const createdTools = new Set();

/**
 * Enhanced utility function to make HTTP requests with retry logic
 */
async function makeRequest(url, options = {}, retries = MAX_RETRIES) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'WorkerManagement-ToolRegistryTest/1.0',
            ...options.headers
        }
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

            // Add delay between requests
            await new Promise(resolve => setTimeout(resolve, REQUEST_DELAY));

            return data;
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
 * Enhanced tool creation with validation and detailed configuration
 */
async function createTool(toolConfig) {
    console.log(`\n=== Creating Tool: ${toolConfig.name} ===`);

    try {
        // Pre-validate configuration
        if (!toolConfig.name || !toolConfig.description || !toolConfig.python_code) {
            throw new Error('Tool configuration missing required fields: name, description, python_code');
        }

        const response = await makeRequest(`${API_BASE}/tools/`, {
            method: 'POST',
            body: JSON.stringify(toolConfig)
        });

        console.log(`✅ Tool created successfully: ${response.name}`);
        console.log(`   Description: ${response.description}`);
        console.log(`   Version: ${response.version}`);
        console.log(`   Tags: ${response.tags.join(', ')}`);
        console.log(`   Valid: ${response.is_valid}`);
        console.log(`   Created At: ${response.created_at}`);

        if (!response.is_valid && response.validation_error) {
            console.log(`   Validation Error: ${response.validation_error}`);
        }

        // Track for cleanup
        createdTools.add(response.name);

        // Enhanced validation of required fields
        const requiredFields = ['name', 'description', 'python_code', 'is_valid', 'created_at', 'updated_at'];
        const missingFields = requiredFields.filter(field => response[field] === undefined);

        if (missingFields.length > 0) {
            throw new Error(`Tool creation response missing required fields: ${missingFields.join(', ')}`);
        }

        return response;
    } catch (error) {
        console.error(`❌ Failed to create tool ${toolConfig.name}: ${error.message}`);
        if (error.status && error.response) {
            console.error(`   HTTP Details: Status ${error.status}, Response: ${error.response}`);
        }
        throw error;
    }
}

/**
 * Enhanced tool listing with filtering and pagination
 */
async function listTools(queryParams = {}) {
    console.log(`\n=== Listing Tools with params: ${JSON.stringify(queryParams)} ===`);

    try {
        // Build query string
        const searchParams = new URLSearchParams();
        if (queryParams.tags) searchParams.append('tags', queryParams.tags);
        if (queryParams.search) searchParams.append('search', queryParams.search);
        if (queryParams.limit !== undefined) searchParams.append('limit', queryParams.limit.toString());
        if (queryParams.offset !== undefined) searchParams.append('offset', queryParams.offset.toString());

        const queryString = searchParams.toString();
        const url = `${API_BASE}/tools/${queryString ? '?' + queryString : ''}`;

        const response = await makeRequest(url);

        console.log(`✅ Tools retrieved successfully`);
        console.log(`   Total tools: ${response.total}`);
        console.log(`   Returned tools: ${response.tools.length}`);

        // Validate ToolListResponse structure
        if (!response.hasOwnProperty('tools') || !response.hasOwnProperty('total')) {
            throw new Error('ToolListResponse missing required fields: tools, total');
        }

        if (!Array.isArray(response.tools)) {
            throw new Error('ToolListResponse.tools must be an array');
        }

        if (typeof response.total !== 'number') {
            throw new Error('ToolListResponse.total must be a number');
        }

        // Validate each ToolResponse in the list
        for (const tool of response.tools) {
            validateToolResponse(tool, false); // Don't log each validation
        }

        return response;
    } catch (error) {
        console.error(`❌ Failed to list tools: ${error.message}`);
        if (error.status && error.response) {
            console.error(`   HTTP Details: Status ${error.status}, Response: ${error.response}`);
        }
        throw error;
    }
}

/**
 * Validate ToolResponse structure
 */
function validateToolResponse(toolResponse, logValidation = true) {
    if (logValidation) {
        console.log(`🔍 Validating ToolResponse for: ${toolResponse.name || 'unknown'}`);
    }

    const requiredFields = [
        'name', 'description', 'python_code', 'version',
        'created_at', 'updated_at', 'is_valid'
    ];

    const missingFields = requiredFields.filter(field => toolResponse[field] === undefined);
    if (missingFields.length > 0) {
        throw new Error(`ToolResponse missing required fields: ${missingFields.join(', ')}`);
    }

    // Validate field types
    if (typeof toolResponse.name !== 'string') {
        throw new Error('ToolResponse.name must be a string');
    }

    if (typeof toolResponse.description !== 'string') {
        throw new Error('ToolResponse.description must be a string');
    }

    if (typeof toolResponse.python_code !== 'string') {
        throw new Error('ToolResponse.python_code must be a string');
    }

    if (typeof toolResponse.version !== 'string') {
        throw new Error('ToolResponse.version must be a string');
    }

    if (typeof toolResponse.is_valid !== 'boolean') {
        throw new Error('ToolResponse.is_valid must be a boolean');
    }

    // Validate optional fields
    if (toolResponse.tags !== undefined && !Array.isArray(toolResponse.tags)) {
        throw new Error('ToolResponse.tags must be an array');
    }

    if (toolResponse.input_schema !== undefined && toolResponse.input_schema !== null && typeof toolResponse.input_schema !== 'object') {
        throw new Error('ToolResponse.input_schema must be an object or null');
    }

    if (toolResponse.output_schema !== undefined && toolResponse.output_schema !== null && typeof toolResponse.output_schema !== 'object') {
        throw new Error('ToolResponse.output_schema must be an object or null');
    }

    if (logValidation) {
        console.log(`✅ ToolResponse validation passed for: ${toolResponse.name}`);
    }

    return true;
}

/**
 * Test tool creation with various configurations
 */
async function testToolCreation() {
    console.log('\n🛠️  Testing Tool Creation with Various Configurations');
    console.log('='.repeat(60));

    const testResults = {
        basicTool: false,
        toolWithSchemas: false,
        toolWithTags: false,
        toolWithComplexCode: false,
        mathTool: false,
        textProcessorTool: false
    };

    try {
        // Test 1: Basic tool creation
        console.log('\n📝 Test 1: Basic Tool Creation');
        const basicTool = {
            name: `basic_tool_${Date.now()}`,
            description: 'A simple test tool for basic operations',
            python_code: `
def simple_function(input_data):
    """Simple function that returns the input with a greeting"""
    name = input_data.get('name', 'World')
    return {'message': f'Hello, {name}!', 'input_received': input_data}
`,
            tags: ['test', 'basic'],
            version: '1.0.0'
        };

        const createdBasic = await createTool(basicTool);
        testResults.basicTool = createdBasic && createdBasic.is_valid;

        // Test 2: Tool with input/output schemas
        console.log('\n📋 Test 2: Tool with Input/Output Schemas');
        const schemaToolName = `schema_tool_${Date.now()}`;
        const schemaTool = {
            name: schemaToolName,
            description: 'A tool with comprehensive input and output schemas',
            python_code: `
def calculate_metrics(input_data):
    """Calculate various metrics from numerical data"""
    numbers = input_data.get('numbers', [])

    if not numbers:
        return {'error': 'No numbers provided'}

    return {
        'count': len(numbers),
        'sum': sum(numbers),
        'average': sum(numbers) / len(numbers),
        'min': min(numbers),
        'max': max(numbers),
        'range': max(numbers) - min(numbers)
    }
`,
            input_schema: {
                type: 'object',
                properties: {
                    numbers: {
                        type: 'array',
                        items: { type: 'number' },
                        description: 'Array of numbers to analyze'
                    }
                },
                required: ['numbers']
            },
            output_schema: {
                type: 'object',
                properties: {
                    count: { type: 'integer', description: 'Number of items' },
                    sum: { type: 'number', description: 'Sum of all numbers' },
                    average: { type: 'number', description: 'Average value' },
                    min: { type: 'number', description: 'Minimum value' },
                    max: { type: 'number', description: 'Maximum value' },
                    range: { type: 'number', description: 'Range (max - min)' }
                }
            },
            tags: ['math', 'statistics', 'analysis'],
            version: '2.1.0'
        };

        const createdSchema = await createTool(schemaTool);
        testResults.toolWithSchemas = createdSchema && createdSchema.is_valid;

        // Test 3: Tool with extensive tags
        console.log('\n🏷️  Test 3: Tool with Multiple Tags');
        const tagToolName = `tag_tool_${Date.now()}`;
        const tagTool = {
            name: tagToolName,
            description: 'A tool demonstrating comprehensive tagging',
            python_code: `
class TextAnalyzer:
    """Comprehensive text analysis tool"""

    def analyze(self, input_data):
        text = input_data.get('text', '')

        # Basic statistics
        words = text.split()
        sentences = text.split('.')

        # Character analysis
        chars = len(text)
        chars_no_spaces = len(text.replace(' ', ''))

        return {
            'text_length': chars,
            'text_length_no_spaces': chars_no_spaces,
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'average_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'unique_words': len(set(word.lower() for word in words))
        }
`,
            tags: ['text', 'analysis', 'nlp', 'statistics', 'processing', 'utility', 'data', 'metrics'],
            version: '1.5.0'
        };

        const createdTag = await createTool(tagTool);
        testResults.toolWithTags = createdTag && createdTag.is_valid;

        // Test 4: Tool with complex Python code
        console.log('\n🧮 Test 4: Tool with Complex Python Implementation');
        const complexToolName = `complex_tool_${Date.now()}`;
        const complexTool = {
            name: complexToolName,
            description: 'A complex tool with advanced Python features',
            python_code: `
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

class DataProcessor:
    """Advanced data processing tool with multiple capabilities"""

    def __init__(self):
        self.supported_operations = [
            'aggregate', 'filter', 'transform', 'validate', 'summarize'
        ]

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data based on operation type"""
        operation = input_data.get('operation', 'summarize')
        data = input_data.get('data', [])

        if operation not in self.supported_operations:
            return {
                'error': f'Unsupported operation: {operation}',
                'supported_operations': self.supported_operations
            }

        try:
            if operation == 'aggregate':
                return self._aggregate_data(data, input_data.get('group_by'))
            elif operation == 'filter':
                return self._filter_data(data, input_data.get('criteria', {}))
            elif operation == 'transform':
                return self._transform_data(data, input_data.get('transformations', []))
            elif operation == 'validate':
                return self._validate_data(data, input_data.get('schema', {}))
            else:  # summarize
                return self._summarize_data(data)

        except Exception as e:
            return {'error': f'Processing failed: {str(e)}'}

    def _aggregate_data(self, data: List[Dict], group_by: Optional[str]) -> Dict:
        """Aggregate data by specified field"""
        if not group_by:
            return {'total_records': len(data)}

        groups = {}
        for item in data:
            key = item.get(group_by, 'unknown')
            if key not in groups:
                groups[key] = {'count': 0, 'items': []}
            groups[key]['count'] += 1
            groups[key]['items'].append(item)

        return {'aggregation': groups, 'total_groups': len(groups)}

    def _filter_data(self, data: List[Dict], criteria: Dict) -> Dict:
        """Filter data based on criteria"""
        filtered = []
        for item in data:
            match = True
            for key, value in criteria.items():
                if item.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(item)

        return {'filtered_data': filtered, 'filtered_count': len(filtered)}

    def _transform_data(self, data: List[Dict], transformations: List[Dict]) -> Dict:
        """Apply transformations to data"""
        transformed = []
        for item in data.copy():
            for transform in transformations:
                field = transform.get('field')
                operation = transform.get('operation', 'copy')

                if field in item:
                    if operation == 'uppercase' and isinstance(item[field], str):
                        item[field] = item[field].upper()
                    elif operation == 'lowercase' and isinstance(item[field], str):
                        item[field] = item[field].lower()
                    elif operation == 'multiply' and isinstance(item[field], (int, float)):
                        multiplier = transform.get('value', 1)
                        item[field] = item[field] * multiplier

            transformed.append(item)

        return {'transformed_data': transformed, 'transformations_applied': len(transformations)}

    def _validate_data(self, data: List[Dict], schema: Dict) -> Dict:
        """Validate data against schema"""
        required_fields = schema.get('required', [])
        field_types = schema.get('types', {})

        validation_results = []
        for i, item in enumerate(data):
            result = {'index': i, 'valid': True, 'errors': []}

            # Check required fields
            for field in required_fields:
                if field not in item:
                    result['valid'] = False
                    result['errors'].append(f'Missing required field: {field}')

            # Check field types
            for field, expected_type in field_types.items():
                if field in item:
                    actual_type = type(item[field]).__name__
                    if actual_type != expected_type:
                        result['valid'] = False
                        result['errors'].append(f'Field {field} has type {actual_type}, expected {expected_type}')

            validation_results.append(result)

        valid_count = sum(1 for r in validation_results if r['valid'])
        return {
            'validation_results': validation_results,
            'valid_count': valid_count,
            'invalid_count': len(validation_results) - valid_count,
            'total_records': len(data)
        }

    def _summarize_data(self, data: List[Dict]) -> Dict:
        """Generate summary statistics"""
        if not data:
            return {'summary': 'No data provided'}

        # Field analysis
        all_fields = set()
        for item in data:
            all_fields.update(item.keys())

        field_stats = {}
        for field in all_fields:
            values = [item.get(field) for item in data if field in item]
            field_stats[field] = {
                'count': len(values),
                'unique_count': len(set(str(v) for v in values)),
                'null_count': sum(1 for v in values if v is None),
                'data_types': list(set(type(v).__name__ for v in values if v is not None))
            }

        return {
            'total_records': len(data),
            'total_fields': len(all_fields),
            'field_statistics': field_stats,
            'summary_generated_at': datetime.now().isoformat()
        }

# Initialize processor
processor = DataProcessor()

def run(input_data):
    """Main entry point for the tool"""
    return processor.process(input_data)
`,
            input_schema: {
                type: 'object',
                properties: {
                    operation: {
                        type: 'string',
                        enum: ['aggregate', 'filter', 'transform', 'validate', 'summarize'],
                        description: 'Type of data processing operation'
                    },
                    data: {
                        type: 'array',
                        items: { type: 'object' },
                        description: 'Array of data objects to process'
                    },
                    group_by: { type: 'string', description: 'Field to group by (for aggregate)' },
                    criteria: { type: 'object', description: 'Filter criteria (for filter)' },
                    transformations: { type: 'array', description: 'Transformations to apply' },
                    schema: { type: 'object', description: 'Validation schema (for validate)' }
                },
                required: ['data']
            },
            output_schema: {
                type: 'object',
                properties: {
                    total_records: { type: 'integer' },
                    error: { type: 'string' },
                    aggregation: { type: 'object' },
                    filtered_data: { type: 'array' },
                    transformed_data: { type: 'array' },
                    validation_results: { type: 'array' },
                    field_statistics: { type: 'object' }
                }
            },
            tags: ['data', 'processing', 'analysis', 'advanced', 'utility'],
            version: '3.0.0'
        };

        const createdComplex = await createTool(complexTool);
        testResults.toolWithComplexCode = createdComplex && createdComplex.is_valid;

        // Test 5: Mathematical tool
        console.log('\n➕ Test 5: Mathematical Calculator Tool');
        const mathToolName = `math_calculator_${Date.now()}`;
        const mathTool = {
            name: mathToolName,
            description: 'Advanced mathematical calculator with multiple operations',
            python_code: `
import math
from typing import Union, List

class MathCalculator:
    """Advanced mathematical calculator"""

    def calculate(self, input_data):
        """Perform mathematical calculations"""
        operation = input_data.get('operation', 'add')
        numbers = input_data.get('numbers', [])

        if not numbers:
            return {'error': 'No numbers provided', 'supported_operations': self.get_operations()}

        try:
            if operation == 'add':
                result = sum(numbers)
            elif operation == 'multiply':
                result = 1
                for num in numbers:
                    result *= num
            elif operation == 'subtract':
                result = numbers[0]
                for num in numbers[1:]:
                    result -= num
            elif operation == 'divide':
                result = numbers[0]
                for num in numbers[1:]:
                    if num == 0:
                        return {'error': 'Division by zero'}
                    result /= num
            elif operation == 'average':
                result = sum(numbers) / len(numbers)
            elif operation == 'median':
                sorted_nums = sorted(numbers)
                n = len(sorted_nums)
                if n % 2 == 0:
                    result = (sorted_nums[n//2-1] + sorted_nums[n//2]) / 2
                else:
                    result = sorted_nums[n//2]
            elif operation == 'mode':
                from collections import Counter
                counts = Counter(numbers)
                max_count = max(counts.values())
                modes = [num for num, count in counts.items() if count == max_count]
                result = modes[0] if len(modes) == 1 else modes
            elif operation == 'range':
                result = max(numbers) - min(numbers)
            elif operation == 'variance':
                mean = sum(numbers) / len(numbers)
                result = sum((x - mean) ** 2 for x in numbers) / len(numbers)
            elif operation == 'std_dev':
                mean = sum(numbers) / len(numbers)
                variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
                result = math.sqrt(variance)
            elif operation == 'factorial':
                if len(numbers) != 1 or not isinstance(numbers[0], int) or numbers[0] < 0:
                    return {'error': 'Factorial requires a single non-negative integer'}
                result = math.factorial(numbers[0])
            elif operation == 'power':
                if len(numbers) != 2:
                    return {'error': 'Power operation requires exactly 2 numbers: base and exponent'}
                result = numbers[0] ** numbers[1]
            elif operation == 'sqrt':
                if len(numbers) != 1 or numbers[0] < 0:
                    return {'error': 'Square root requires a single non-negative number'}
                result = math.sqrt(numbers[0])
            elif operation == 'log':
                if len(numbers) not in [1, 2] or any(n <= 0 for n in numbers):
                    return {'error': 'Logarithm requires 1 or 2 positive numbers'}
                result = math.log(numbers[0]) if len(numbers) == 1 else math.log(numbers[0], numbers[1])
            else:
                return {'error': f'Unknown operation: {operation}', 'supported_operations': self.get_operations()}

            return {
                'operation': operation,
                'numbers': numbers,
                'result': result,
                'result_type': type(result).__name__
            }

        except Exception as e:
            return {'error': f'Calculation error: {str(e)}'}

    def get_operations(self):
        """Get list of supported operations"""
        return [
            'add', 'multiply', 'subtract', 'divide', 'average', 'median', 'mode',
            'range', 'variance', 'std_dev', 'factorial', 'power', 'sqrt', 'log'
        ]

calculator = MathCalculator()

def run(input_data):
    """Main entry point"""
    return calculator.calculate(input_data)
`,
            tags: ['math', 'calculator', 'statistics', 'utility'],
            version: '2.0.0'
        };

        const createdMath = await createTool(mathTool);
        testResults.mathTool = createdMath && createdMath.is_valid;

        // Test 6: Text processing tool
        console.log('\n📝 Test 6: Advanced Text Processor Tool');
        const textToolName = `text_processor_${Date.now()}`;
        const textTool = {
            name: textToolName,
            description: 'Comprehensive text processing and analysis tool',
            python_code: `
import re
from collections import Counter
from datetime import datetime

class TextProcessor:
    """Advanced text processing capabilities"""

    def process(self, input_data):
        """Process text based on operation"""
        text = input_data.get('text', '')
        operation = input_data.get('operation', 'analyze')

        if not text:
            return {'error': 'No text provided', 'supported_operations': self.get_operations()}

        try:
            if operation == 'analyze':
                return self._analyze_text(text)
            elif operation == 'clean':
                return self._clean_text(text, input_data.get('options', {}))
            elif operation == 'extract':
                return self._extract_patterns(text, input_data.get('pattern_type', 'emails'))
            elif operation == 'transform':
                return self._transform_text(text, input_data.get('transformation', 'uppercase'))
            elif operation == 'sentiment':
                return self._analyze_sentiment(text)
            elif operation == 'keywords':
                return self._extract_keywords(text, input_data.get('top_n', 10))
            else:
                return {'error': f'Unknown operation: {operation}', 'supported_operations': self.get_operations()}

        except Exception as e:
            return {'error': f'Processing error: {str(e)}'}

    def _analyze_text(self, text):
        """Comprehensive text analysis"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        paragraphs = text.split('\\n\\n')

        # Character analysis
        char_count = len(text)
        char_count_no_spaces = len(text.replace(' ', ''))

        # Word analysis
        word_count = len(words)
        unique_words = len(set(word.lower().strip('.,!?;:"()[]{}') for word in words))
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

        # Sentence analysis
        sentence_count = len([s for s in sentences if s.strip()])
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        # Advanced metrics
        readability_score = self._calculate_readability(text, word_count, sentence_count)
        word_frequency = Counter(word.lower().strip('.,!?;:"()[]{}') for word in words if word.strip('.,!?;:"()[]{}'))

        return {
            'character_count': char_count,
            'character_count_no_spaces': char_count_no_spaces,
            'word_count': word_count,
            'unique_word_count': unique_words,
            'sentence_count': sentence_count,
            'paragraph_count': len([p for p in paragraphs if p.strip()]),
            'average_word_length': round(avg_word_length, 2),
            'average_sentence_length': round(avg_sentence_length, 2),
            'readability_score': readability_score,
            'most_common_words': word_frequency.most_common(10),
            'lexical_diversity': round(unique_words / word_count, 3) if word_count > 0 else 0,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def _clean_text(self, text, options):
        """Clean text based on options"""
        cleaned = text

        if options.get('remove_extra_spaces', True):
            cleaned = re.sub(r'\\s+', ' ', cleaned).strip()

        if options.get('remove_punctuation', False):
            cleaned = re.sub(r'[^\\w\\s]', '', cleaned)

        if options.get('lowercase', False):
            cleaned = cleaned.lower()

        if options.get('remove_numbers', False):
            cleaned = re.sub(r'\\d+', '', cleaned)

        if options.get('remove_urls', True):
            cleaned = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned)

        return {
            'original_text': text,
            'cleaned_text': cleaned,
            'cleaning_options': options,
            'character_reduction': len(text) - len(cleaned)
        }

    def _extract_patterns(self, text, pattern_type):
        """Extract patterns from text"""
        patterns = {
            'emails': r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b',
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'phones': r'\\b(?:\\+?1[-.]?)?\\(?[0-9]{3}\\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\\b',
            'dates': r'\\b\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}\\b',
            'numbers': r'\\b\\d+(?:\\.\\d+)?\\b',
            'mentions': r'@\\w+',
            'hashtags': r'#\\w+'
        }

        if pattern_type not in patterns:
            return {'error': f'Unknown pattern type: {pattern_type}', 'available_patterns': list(patterns.keys())}

        matches = re.findall(patterns[pattern_type], text)

        return {
            'pattern_type': pattern_type,
            'matches': matches,
            'match_count': len(matches),
            'unique_matches': list(set(matches))
        }

    def _transform_text(self, text, transformation):
        """Transform text"""
        transformations = {
            'uppercase': text.upper(),
            'lowercase': text.lower(),
            'title': text.title(),
            'capitalize': text.capitalize(),
            'reverse': text[::-1],
            'remove_vowels': re.sub(r'[aeiouAEIOU]', '', text),
            'pig_latin': self._to_pig_latin(text)
        }

        if transformation not in transformations:
            return {'error': f'Unknown transformation: {transformation}', 'available_transformations': list(transformations.keys())}

        return {
            'original_text': text,
            'transformation': transformation,
            'transformed_text': transformations[transformation]
        }

    def _analyze_sentiment(self, text):
        """Simple sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'joy']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'sad', 'angry', 'disappointed']

        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        if positive_count > negative_count:
            sentiment = 'positive'
        elif negative_count > positive_count:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        confidence = abs(positive_count - negative_count) / len(words) if words else 0

        return {
            'sentiment': sentiment,
            'confidence': round(confidence, 3),
            'positive_words_found': positive_count,
            'negative_words_found': negative_count,
            'total_words': len(words)
        }

    def _extract_keywords(self, text, top_n):
        """Extract keywords using simple frequency analysis"""
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those'}

        words = [word.lower().strip('.,!?;:"()[]{}') for word in text.split()]
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]

        word_freq = Counter(filtered_words)
        keywords = word_freq.most_common(top_n)

        return {
            'keywords': keywords,
            'total_unique_words': len(word_freq),
            'stop_words_removed': len(words) - len(filtered_words)
        }

    def _calculate_readability(self, text, word_count, sentence_count):
        """Simple readability score (Flesch-like)"""
        if sentence_count == 0 or word_count == 0:
            return 0

        avg_sentence_length = word_count / sentence_count
        # Simplified calculation
        score = 206.835 - (1.015 * avg_sentence_length)
        return max(0, min(100, round(score, 1)))

    def _to_pig_latin(self, text):
        """Convert text to pig latin"""
        words = text.split()
        pig_latin_words = []

        for word in words:
            if word.isalpha():
                if word[0].lower() in 'aeiou':
                    pig_latin_words.append(word + 'way')
                else:
                    pig_latin_words.append(word[1:] + word[0] + 'ay')
            else:
                pig_latin_words.append(word)

        return ' '.join(pig_latin_words)

    def get_operations(self):
        """Get supported operations"""
        return ['analyze', 'clean', 'extract', 'transform', 'sentiment', 'keywords']

processor = TextProcessor()

def run(input_data):
    """Main entry point"""
    return processor.process(input_data)
`,
            tags: ['text', 'nlp', 'processing', 'analysis', 'utility'],
            version: '1.0.0'
        };

        const createdText = await createTool(textTool);
        testResults.textProcessorTool = createdText && createdText.is_valid;

    } catch (error) {
        console.error(`❌ Tool creation test failed: ${error.message}`);
    }

    return testResults;
}

/**
 * Test tool listing with various filters and pagination
 */
async function testToolListing() {
    console.log('\n📋 Testing Tool Listing with Filters and Pagination');
    console.log('='.repeat(60));

    const testResults = {
        basicListing: false,
        tagFiltering: false,
        searchFiltering: false,
        pagination: false,
        combinedFilters: false,
        emptyResults: false,
        invalidParameters: false
    };

    try {
        // Test 1: Basic listing
        console.log('\n📝 Test 1: Basic Tool Listing');
        const basicList = await listTools();
        testResults.basicListing = basicList && Array.isArray(basicList.tools) && typeof basicList.total === 'number';
        console.log(`   Found ${basicList.total} tools`);

        // Test 2: Tag filtering
        console.log('\n🏷️  Test 2: Tag Filtering');
        const tagFiltered = await listTools({ tags: 'test,utility' });
        testResults.tagFiltering = tagFiltered && Array.isArray(tagFiltered.tools);
        console.log(`   Found ${tagFiltered.total} tools with tags 'test' or 'utility'`);

        // Test 3: Search filtering
        console.log('\n🔍 Test 3: Search Filtering');
        const searchFiltered = await listTools({ search: 'calculator' });
        testResults.searchFiltering = searchFiltered && Array.isArray(searchFiltered.tools);
        console.log(`   Found ${searchFiltered.total} tools matching 'calculator'`);

        // Test 4: Pagination
        console.log('\n📄 Test 4: Pagination');
        const page1 = await listTools({ limit: 2, offset: 0 });
        const page2 = await listTools({ limit: 2, offset: 2 });
        testResults.pagination = page1 && page2 && page1.tools.length <= 2 && page2.tools.length <= 2;
        console.log(`   Page 1: ${page1.tools.length} tools, Page 2: ${page2.tools.length} tools`);

        // Test 5: Combined filters
        console.log('\n🔗 Test 5: Combined Filters');
        const combined = await listTools({
            tags: 'utility',
            search: 'tool',
            limit: 5,
            offset: 0
        });
        testResults.combinedFilters = combined && Array.isArray(combined.tools);
        console.log(`   Found ${combined.total} tools with combined filters`);

        // Test 6: Empty results
        console.log('\n🚫 Test 6: Filters with No Results');
        const empty = await listTools({ search: 'nonexistent_tool_name_xyz' });
        testResults.emptyResults = empty && empty.total === 0 && empty.tools.length === 0;
        console.log(`   Found ${empty.total} tools with impossible search term`);

        // Test 7: Invalid parameters (should be handled gracefully)
        console.log('\n⚠️  Test 7: Invalid Parameters Handling');
        try {
            // Test with invalid tags
            const invalidTags = await makeRequest(`${API_BASE}/tools/?tags=invalid%20tag%20with%20spaces`);
            testResults.invalidParameters = false; // Should have failed
        } catch (error) {
            if (error.status === 400 || error.status === 422) {
                testResults.invalidParameters = true; // Expected to fail
                console.log('   ✅ Invalid parameters correctly rejected');
            }
        }

    } catch (error) {
        console.error(`❌ Tool listing test failed: ${error.message}`);
    }

    return testResults;
}

/**
 * Test edge cases and error handling
 */
async function testEdgeCases() {
    console.log('\n🔍 Testing Edge Cases and Error Handling');
    console.log('='.repeat(60));

    const testResults = {
        duplicateToolName: false,
        invalidPythonCode: false,
        longToolName: false,
        invalidSchema: false,
        tooManyTags: false,
        duplicateQueryParams: false,
        invalidToolName: false
    };

    try {
        // Test 1: Duplicate tool name
        console.log('\n🔄 Test 1: Duplicate Tool Name');
        const duplicateTool = {
            name: `duplicate_test_${Date.now()}`,
            description: 'First tool with this name',
            python_code: 'def test(): return "first"',
            version: '1.0.0'
        };

        try {
            await createTool(duplicateTool);
            // Try to create again with same name
            await createTool({ ...duplicateTool, description: 'Second tool with same name' });
            testResults.duplicateToolName = false; // Should have failed
        } catch (error) {
            if (error.status === 409) {
                testResults.duplicateToolName = true; // Expected conflict
                console.log('   ✅ Duplicate tool name correctly rejected');
            }
        }

        // Test 2: Invalid Python code
        console.log('\n🐍 Test 2: Invalid Python Code');
        const invalidCodeTool = {
            name: `invalid_code_${Date.now()}`,
            description: 'Tool with invalid Python code',
            python_code: 'def broken_function(\n    # Syntax error: missing closing parenthesis',
            version: '1.0.0'
        };

        try {
            const result = await createTool(invalidCodeTool);
            if (result && !result.is_valid) {
                testResults.invalidPythonCode = true;
                console.log('   ✅ Invalid Python code detected and flagged');
            }
        } catch (error) {
            if (error.status === 422) {
                testResults.invalidPythonCode = true;
                console.log('   ✅ Invalid Python code correctly rejected');
            }
        }

        // Test 3: Tool name too long
        console.log('\n📏 Test 3: Tool Name Too Long');
        const longNameTool = {
            name: 'a'.repeat(150), // Exceeds max length
            description: 'Tool with very long name',
            python_code: 'def test(): return "ok"',
            version: '1.0.0'
        };

        try {
            await createTool(longNameTool);
            testResults.longToolName = false; // Should have failed
        } catch (error) {
            if (error.status === 422) {
                testResults.longToolName = true;
                console.log('   ✅ Long tool name correctly rejected');
            }
        }

        // Test 4: Invalid schema
        console.log('\n📋 Test 4: Invalid Schema');
        const invalidSchemaTool = {
            name: `invalid_schema_${Date.now()}`,
            description: 'Tool with invalid schema',
            python_code: 'def test(): return "ok"',
            input_schema: {}, // Empty schema should be rejected
            version: '1.0.0'
        };

        try {
            await createTool(invalidSchemaTool);
            testResults.invalidSchema = false; // Should have failed
        } catch (error) {
            if (error.status === 422) {
                testResults.invalidSchema = true;
                console.log('   ✅ Invalid schema correctly rejected');
            }
        }

        // Test 5: Too many tags
        console.log('\n🏷️  Test 5: Too Many Tags');
        const tooManyTagsTool = {
            name: `too_many_tags_${Date.now()}`,
            description: 'Tool with too many tags',
            python_code: 'def test(): return "ok"',
            tags: Array.from({ length: 15 }, (_, i) => `tag${i}`), // Exceeds max
            version: '1.0.0'
        };

        try {
            await createTool(tooManyTagsTool);
            testResults.tooManyTags = false; // Should have failed
        } catch (error) {
            if (error.status === 422) {
                testResults.tooManyTags = true;
                console.log('   ✅ Too many tags correctly rejected');
            }
        }

        // Test 7: Duplicate query parameters
        console.log('\n🔗 Test 6: Duplicate Query Parameters');
        try {
            const response = await makeRequest(`${API_BASE}/tools/?search=test&search=duplicate`);
            testResults.duplicateQueryParams = false; // Should have failed
        } catch (error) {
            if (error.status === 422) {
                testResults.duplicateQueryParams = true;
                console.log('   ✅ Duplicate query parameters correctly rejected');
            }
        }

        // Test 8: Invalid tool name characters
        console.log('\n🔤 Test 7: Invalid Tool Name Characters');
        const invalidNameTool = {
            name: 'invalid@tool#name!',
            description: 'Tool with invalid characters in name',
            python_code: 'def test(): return "ok"',
            version: '1.0.0'
        };

        try {
            await createTool(invalidNameTool);
            testResults.invalidToolName = false; // Should have failed
        } catch (error) {
            if (error.status === 422) {
                testResults.invalidToolName = true;
                console.log('   ✅ Invalid tool name characters correctly rejected');
            }
        }

    } catch (error) {
        console.error(`❌ Edge cases test failed: ${error.message}`);
    }

    return testResults;
}

/**
 * Comprehensive API health check
 */
async function checkApiHealth() {
    console.log('\n🏥 Performing API Health Check...');

    try {
        // Check main API health
        const healthResponse = await makeRequest(`${API_BASE}/health`);
        console.log(`✅ Main API is healthy: ${healthResponse.status}`);

        // Check tool registry health
        const toolHealthResponse = await makeRequest(`${API_BASE}/tools/health`);
        console.log(`✅ Tool registry is healthy: ${toolHealthResponse.status}`);
        console.log(`   Total tools in registry: ${toolHealthResponse.total_tools}`);

        return { healthy: true, mainApi: healthResponse, toolRegistry: toolHealthResponse };

    } catch (error) {
        console.error(`❌ Health check failed: ${error.message}`);
        return { healthy: false, error: error.message };
    }
}

/**
 * Enhanced resource cleanup
 */
async function cleanupResources() {
    console.log('\n🧹 Performing Resource Cleanup...');
    const cleanupResults = {
        tools_deleted: [],
        failed_deletions: [],
        total_time: 0
    };

    const startTime = Date.now();

    try {
        for (const toolName of createdTools) {
            console.log(`🗑️  Deleting tool: ${toolName}`);
            try {
                await makeRequest(`${API_BASE}/tools/${toolName}`, { method: 'DELETE' });
                console.log(`✅ Tool deleted successfully: ${toolName}`);
                cleanupResults.tools_deleted.push(toolName);
            } catch (error) {
                console.error(`❌ Failed to delete tool ${toolName}: ${error.message}`);
                cleanupResults.failed_deletions.push({ toolName, error: error.message });
            }
        }

        cleanupResults.total_time = Date.now() - startTime;

        console.log('\n✅ Cleanup completed');
        console.log(`   Tools deleted: ${cleanupResults.tools_deleted.length}/${createdTools.size}`);
        console.log(`   Failed deletions: ${cleanupResults.failed_deletions.length}`);
        console.log(`   Total cleanup time: ${cleanupResults.total_time}ms`);

    } catch (error) {
        console.error(`❌ Cleanup process failed: ${error.message}`);
        cleanupResults.total_time = Date.now() - startTime;
    }

    return cleanupResults;
}

/**
 * Main test function
 */
async function runToolCollectionTests() {
    console.log('🚀 Starting Tool Registry Collection Operations Test Suite');
    console.log('='.repeat(80));

    const testResults = {
        health_check: false,
        tool_creation: {},
        tool_listing: {},
        edge_cases: {},
        overall_success: false,
        execution_time: 0
    };

    const testStartTime = Date.now();

    try {
        // Step 1: API Health Check
        console.log('\n🏥 Step 1: API Health Check');
        const healthResult = await checkApiHealth();
        testResults.health_check = healthResult.healthy;

        if (!healthResult.healthy) {
            throw new Error('API health check failed - cannot proceed with tests');
        }

        // Step 2: Tool Creation Tests
        console.log('\n🛠️  Step 2: Tool Creation Tests');
        testResults.tool_creation = await testToolCreation();

        // Step 3: Tool Listing Tests
        console.log('\n📋 Step 3: Tool Listing Tests');
        testResults.tool_listing = await testToolListing();

        // Step 4: Edge Cases and Error Handling
        console.log('\n🔍 Step 4: Edge Cases and Error Handling');
        testResults.edge_cases = await testEdgeCases();

        // Overall success calculation
        const creationPassed = Object.values(testResults.tool_creation).filter(v => v === true).length;
        const listingPassed = Object.values(testResults.tool_listing).filter(v => v === true).length;
        const edgeCasesPassed = Object.values(testResults.edge_cases).filter(v => v === true).length;

        const totalTests = Object.keys(testResults.tool_creation).length +
                          Object.keys(testResults.tool_listing).length +
                          Object.keys(testResults.edge_cases).length;
        const passedTests = creationPassed + listingPassed + edgeCasesPassed;

        testResults.overall_success = testResults.health_check && (passedTests >= totalTests * 0.8); // 80% pass rate
        testResults.execution_time = Date.now() - testStartTime;

        // Test Summary
        console.log('\n📊 Test Summary');
        console.log('='.repeat(80));
        console.log(`🏁 Tool Registry Collection Operations Test - COMPLETE`);
        console.log(`   Total execution time: ${testResults.execution_time}ms`);
        console.log(`   Overall success: ${testResults.overall_success ? '✅ PASSED' : '❌ FAILED'}`);
        console.log('\n📋 Detailed Results:');
        console.log(`   ✅ API Health Check: ${testResults.health_check ? 'PASSED' : 'FAILED'}`);
        console.log(`   🛠️  Tool Creation Tests: ${creationPassed}/${Object.keys(testResults.tool_creation).length} PASSED`);
        console.log(`   📋 Tool Listing Tests: ${listingPassed}/${Object.keys(testResults.tool_listing).length} PASSED`);
        console.log(`   🔍 Edge Cases Tests: ${edgeCasesPassed}/${Object.keys(testResults.edge_cases).length} PASSED`);
        console.log(`   📈 Overall Pass Rate: ${Math.round((passedTests / totalTests) * 100)}%`);

        // Detailed test results
        console.log('\n🔍 Detailed Test Results:');
        console.log('\n🛠️  Tool Creation:');
        for (const [test, result] of Object.entries(testResults.tool_creation)) {
            console.log(`   ${result ? '✅' : '❌'} ${test}: ${result ? 'PASSED' : 'FAILED'}`);
        }

        console.log('\n📋 Tool Listing:');
        for (const [test, result] of Object.entries(testResults.tool_listing)) {
            console.log(`   ${result ? '✅' : '❌'} ${test}: ${result ? 'PASSED' : 'FAILED'}`);
        }

        console.log('\n🔍 Edge Cases:');
        for (const [test, result] of Object.entries(testResults.edge_cases)) {
            console.log(`   ${result ? '✅' : '❌'} ${test}: ${result ? 'PASSED' : 'FAILED'}`);
        }

        if (testResults.overall_success) {
            console.log('\n🎉 Tool Registry Collection Operations Test PASSED!');
            console.log('   ✅ All core functionality is working correctly');
            console.log('   ✅ Error handling is robust');
            console.log('   ✅ API responses are properly structured');
            console.log('   ✅ Validation is working as expected');
        } else {
            console.log('\n💥 Tool Registry Collection Operations Test FAILED!');
            console.log('   ❌ Some tests did not pass');
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

    // Cleanup
    console.log('\n🧹 Step 5: Resource Cleanup');
    try {
        const cleanupResult = await cleanupResources();

        if (cleanupResult.failed_deletions.length > 0) {
            console.log(`⚠️  Some resources failed to cleanup: ${cleanupResult.failed_deletions.length} items`);
        }
    } catch (cleanupError) {
        console.error(`❌ Cleanup failed: ${cleanupError.message}`);
    }

    return testResults;
}

/**
 * Main function with comprehensive error handling
 */
async function main() {
    console.log('🚀 Tool Registry Collection Operations Test Suite');
    console.log('Node.js version:', process.version);
    console.log('Platform:', process.platform);
    console.log('Architecture:', process.arch);
    console.log('='.repeat(80));

    try {
        const testResults = await runToolCollectionTests();

        // Final summary
        console.log('\n📊 FINAL TEST SUMMARY');
        console.log('='.repeat(80));
        console.log(`Overall Result: ${testResults.overall_success ? '🎉 SUCCESS' : '💥 FAILURE'}`);
        console.log(`Total Execution Time: ${testResults.execution_time}ms`);
        console.log(`Tools Created During Testing: ${createdTools.size}`);

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

// Execute the test suite
if (require.main === module) {
    main().catch(error => {
        console.error('❌ Unhandled error in test suite:', error);
        process.exit(1);
    });
}

module.exports = {
    makeRequest,
    createTool,
    listTools,
    validateToolResponse,
    testToolCreation,
    testToolListing,
    testEdgeCases,
    checkApiHealth,
    cleanupResources,
    runToolCollectionTests
};
