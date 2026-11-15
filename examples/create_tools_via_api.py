#!/usr/bin/env python3
"""
Example: Creating tools via the Worker Management API by providing Python code
"""

import asyncio
from typing import Any

import httpx

# Tool creation examples
MATH_CALCULATOR_TOOL = {
    "name": "math_calculator",
    "description": "Performs basic mathematical calculations",
    "python_code": '''
class MathCalculator:
    """A tool for performing mathematical calculations"""

    def run(self, input_data):
        """Perform calculation based on input"""
        operation = input_data.get("operation", "add")
        numbers = input_data.get("numbers", [])

        if not numbers:
            return {"error": "No numbers provided"}

        if operation == "add":
            result = sum(numbers)
        elif operation == "multiply":
            result = 1
            for num in numbers:
                result *= num
        elif operation == "average":
            result = sum(numbers) / len(numbers)
        elif operation == "max":
            result = max(numbers)
        elif operation == "min":
            result = min(numbers)
        else:
            return {"error": f"Unknown operation: {operation}"}

        return {
            "operation": operation,
            "numbers": numbers,
            "result": result
        }
''',
    "input_schema": {
        "operation": {"type": "string", "description": "Mathematical operation"},
        "numbers": {"type": "array", "description": "List of numbers"},
    },
    "output_schema": {
        "result": {"type": "number", "description": "Calculation result"},
        "operation": {"type": "string", "description": "Operation performed"},
    },
    "tags": ["math", "calculator", "utility"],
    "version": "1.0.0",
}

TEXT_PROCESSOR_TOOL = {
    "name": "text_processor",
    "description": "Processes text with various operations",
    "python_code": '''
import re

class TextProcessor:
    """A tool for text processing operations"""

    def run(self, input_data):
        """Process text based on operation"""
        text = input_data.get("text", "")
        operation = input_data.get("operation", "uppercase")

        if operation == "uppercase":
            result = text.upper()
        elif operation == "lowercase":
            result = text.lower()
        elif operation == "word_count":
            result = len(text.split())
        elif operation == "char_count":
            result = len(text)
        elif operation == "reverse":
            result = text[::-1]
        elif operation == "extract_emails":
            email_pattern = r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
            result = re.findall(email_pattern, text)
        elif operation == "clean_whitespace":
            result = re.sub(r'\\s+', ' ', text.strip())
        else:
            return {"error": f"Unknown operation: {operation}"}

        return {
            "original_text": text,
            "operation": operation,
            "result": result
        }
''',
    "input_schema": {
        "text": {"type": "string", "description": "Text to process"},
        "operation": {"type": "string", "description": "Processing operation"},
    },
    "output_schema": {
        "result": {"type": "any", "description": "Processed result"},
        "operation": {"type": "string", "description": "Operation performed"},
    },
    "tags": ["text", "processing", "utility"],
    "version": "1.0.0",
}

DATA_VALIDATOR_TOOL = {
    "name": "data_validator",
    "description": "Validates data structures and formats",
    "python_code": '''
import json
import re
from datetime import datetime

class DataValidator:
    """A tool for data validation"""

    def run(self, input_data):
        """Validate data based on validation type"""
        data = input_data.get("data")
        validation_type = input_data.get("validation_type", "json")

        result = {
            "validation_type": validation_type,
            "is_valid": False,
            "errors": [],
            "warnings": []
        }

        try:
            if validation_type == "json":
                if isinstance(data, str):
                    json.loads(data)
                result["is_valid"] = True

            elif validation_type == "email":
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
                if isinstance(data, str) and re.match(email_pattern, data):
                    result["is_valid"] = True
                else:
                    result["errors"].append("Invalid email format")

            elif validation_type == "phone":
                # Simple phone validation
                phone_pattern = r'^[\\+]?[1-9]?[0-9]{7,15}$'
                clean_phone = re.sub(r'[\\s\\-\\(\\)]', '', str(data))
                if re.match(phone_pattern, clean_phone):
                    result["is_valid"] = True
                else:
                    result["errors"].append("Invalid phone number format")

            elif validation_type == "required_fields":
                required_fields = input_data.get("required_fields", [])
                if isinstance(data, dict):
                    missing_fields = [field for field in required_fields if field not in data]
                    if not missing_fields:
                        result["is_valid"] = True
                    else:
                        result["errors"].extend([f"Missing required field: {field}" for field in missing_fields])
                else:
                    result["errors"].append("Data must be a dictionary for required_fields validation")

            else:
                result["errors"].append(f"Unknown validation type: {validation_type}")

        except Exception as e:
            result["errors"].append(f"Validation error: {str(e)}")

        return result
''',
    "input_schema": {
        "data": {"type": "any", "description": "Data to validate"},
        "validation_type": {"type": "string", "description": "Type of validation"},
        "required_fields": {
            "type": "array",
            "description": "Required fields (for required_fields validation)",
        },
    },
    "output_schema": {
        "is_valid": {"type": "boolean", "description": "Whether data is valid"},
        "errors": {"type": "array", "description": "Validation errors"},
        "warnings": {"type": "array", "description": "Validation warnings"},
    },
    "tags": ["validation", "data", "utility"],
    "version": "1.0.0",
}


async def create_tool_via_api(
    tool_definition: dict[str, Any], base_url: str = "http://localhost:8887"
):
    """Create a tool via the API"""

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{base_url}/tools/",
                json=tool_definition,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 201:
                tool_data = response.json()
                print(f"✅ Tool '{tool_data['name']}' created successfully!")
                print(f"   Description: {tool_data['description']}")
                print(f"   Version: {tool_data['version']}")
                print(f"   Tags: {', '.join(tool_data['tags'])}")
                print(f"   Valid: {tool_data['is_valid']}")
                if not tool_data["is_valid"]:
                    print(f"   Validation Error: {tool_data['validation_error']}")
                return tool_data
            else:
                print(
                    f"❌ Failed to create tool '{tool_definition['name']}': {response.status_code}"
                )
                print(f"   Error: {response.text}")
                return None

        except Exception as e:
            print(f"❌ Error creating tool '{tool_definition['name']}': {e}")
            return None


async def test_tool_via_api(
    tool_name: str, test_input: dict[str, Any], base_url: str = "http://localhost:8887"
):
    """Test a tool via the API"""

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{base_url}/tools/{tool_name}/test",
                json=test_input,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.json()
                print(f"🧪 Test result for tool '{tool_name}':")
                print(f"   Success: {result['success']}")
                print(f"   Input: {result['input']}")
                print(f"   Output: {result['output']}")
                if not result["success"]:
                    print(f"   Error: {result.get('error')}")
                return result
            else:
                print(f"❌ Failed to test tool '{tool_name}': {response.status_code}")
                print(f"   Error: {response.text}")
                return None

        except Exception as e:
            print(f"❌ Error testing tool '{tool_name}': {e}")
            return None


async def list_tools_via_api(base_url: str = "http://localhost:8887"):
    """List all tools via the API"""

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{base_url}/tools/")

            if response.status_code == 200:
                result = response.json()
                print(f"📋 Available tools ({result['total']}):")
                for tool in result["tools"]:
                    print(f"   • {tool['name']} v{tool['version']} - {tool['description']}")
                    print(f"     Tags: {', '.join(tool['tags'])}")
                    print(f"     Valid: {tool['is_valid']}")
                return result
            else:
                print(f"❌ Failed to list tools: {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ Error listing tools: {e}")
            return None


async def create_worker_with_tools(
    worker_name: str, tool_names: list, base_url: str = "http://localhost:8887"
):
    """Create a worker with specific tools"""

    worker_data = {
        "worker_name": worker_name,
        "tool_names": tool_names,
        "description": f"Worker with tools: {', '.join(tool_names)}",
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{base_url}/workers",
                json=worker_data,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code in [200, 201]:
                result = response.json()
                print(f"👤 Worker '{worker_name}' created with tools: {tool_names}")
                return result
            else:
                print(f"❌ Failed to create worker: {response.status_code}")
                print(f"   Error: {response.text}")
                return None

        except Exception as e:
            print(f"❌ Error creating worker: {e}")
            return None


async def main():
    """Demonstrate tool creation via API"""

    print("🛠️  Tool Creation via Worker Management API Demo")
    print("=" * 50)

    # Step 1: Create tools
    print("\n1. Creating tools via API...")

    tools_to_create = [MATH_CALCULATOR_TOOL, TEXT_PROCESSOR_TOOL, DATA_VALIDATOR_TOOL]

    created_tools = []
    for tool_def in tools_to_create:
        result = await create_tool_via_api(tool_def)
        if result:
            created_tools.append(result["name"])

    # Step 2: List all tools
    print("\n2. Listing all available tools...")
    await list_tools_via_api()

    # Step 3: Test the tools
    print("\n3. Testing created tools...")

    # Test math calculator
    if "math_calculator" in created_tools:
        await test_tool_via_api(
            "math_calculator", {"operation": "add", "numbers": [10, 20, 30, 40]}
        )

        await test_tool_via_api(
            "math_calculator", {"operation": "average", "numbers": [1, 2, 3, 4, 5]}
        )

    # Test text processor
    if "text_processor" in created_tools:
        await test_tool_via_api(
            "text_processor",
            {
                "text": "Hello World! This is a test email: user@example.com",
                "operation": "extract_emails",
            },
        )

        await test_tool_via_api(
            "text_processor",
            {"text": "Convert This Text To Lowercase", "operation": "lowercase"},
        )

    # Test data validator
    if "data_validator" in created_tools:
        await test_tool_via_api(
            "data_validator", {"data": "user@example.com", "validation_type": "email"}
        )

        await test_tool_via_api(
            "data_validator",
            {
                "data": {"name": "John", "age": 30},
                "validation_type": "required_fields",
                "required_fields": ["name", "email"],
            },
        )

    # Step 4: Create a worker with tools
    print("\n4. Creating worker with tools...")
    if created_tools:
        await create_worker_with_tools("tool_demo_worker", created_tools[:2])

    print("\n✅ Tool creation demo completed!")
    print(f"   Created {len(created_tools)} tools")
    print("   Tools are now available for use by workers")


if __name__ == "__main__":
    asyncio.run(main())
