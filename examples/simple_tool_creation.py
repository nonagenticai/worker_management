#!/usr/bin/env python3
"""
Simple example: Create a tool by providing Python code
"""

import requests

# Define your tool with Python code
tool_definition = {
    "name": "simple_calculator",
    "description": "A simple calculator that can add, subtract, multiply, and divide",
    "python_code": '''
class SimpleCalculator:
    """A simple calculator tool"""

    def run(self, input_data):
        """Perform calculation"""
        operation = input_data.get("operation")
        a = input_data.get("a", 0)
        b = input_data.get("b", 0)

        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b != 0:
                result = a / b
            else:
                return {"error": "Cannot divide by zero"}
        else:
            return {"error": f"Unknown operation: {operation}"}

        return {
            "operation": operation,
            "a": a,
            "b": b,
            "result": result
        }
''',
    "input_schema": {
        "operation": {
            "type": "string",
            "description": "Operation to perform (add, subtract, multiply, divide)",
        },
        "a": {"type": "number", "description": "First number"},
        "b": {"type": "number", "description": "Second number"},
    },
    "output_schema": {"result": {"type": "number", "description": "Calculation result"}},
    "tags": ["math", "calculator"],
    "version": "1.0.0",
}


def create_tool():
    """Create the tool via API"""

    try:
        response = requests.post(
            "http://localhost:8887/tools/",
            json=tool_definition,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 201:
            tool_data = response.json()
            print("✅ Tool created successfully!")
            print(f"   Name: {tool_data['name']}")
            print(f"   Valid: {tool_data['is_valid']}")
            return tool_data
        else:
            print(f"❌ Failed to create tool: {response.status_code}")
            print(f"   Error: {response.text}")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_tool():
    """Test the created tool"""

    test_cases = [
        {"operation": "add", "a": 10, "b": 5},
        {"operation": "multiply", "a": 7, "b": 8},
        {"operation": "divide", "a": 20, "b": 4},
        {"operation": "divide", "a": 10, "b": 0},  # Test error case
    ]

    for test_input in test_cases:
        try:
            response = requests.post(
                "http://localhost:8887/tools/simple_calculator/test",
                json=test_input,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.json()
                print(f"🧪 Test: {test_input}")
                print(f"   Result: {result['output']}")
                print(f"   Success: {result['success']}")
            else:
                print(f"❌ Test failed: {response.status_code}")

        except Exception as e:
            print(f"❌ Test error: {e}")


if __name__ == "__main__":
    print("Creating a simple calculator tool...")

    # Create the tool
    tool = create_tool()

    if tool:
        print("\nTesting the tool...")
        test_tool()

        print("\n✅ Tool is now available!")
        print("   You can use it in workers by specifying tool_names=['simple_calculator']")
    else:
        print("\n❌ Tool creation failed")
