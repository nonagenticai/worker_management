# Structured Output Guide

## Overview

The Worker Management API supports **structured outputs** for AI agents using JSON Schema validation and optional custom Python validators. This allows you to enforce specific output formats and data structures from your agents, ensuring consistent, type-safe responses.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Creating Workers with Structured Output](#creating-workers-with-structured-output)
3. [Updating Worker Structured Output](#updating-worker-structured-output)
4. [Sending Messages with Structured Output](#sending-messages-with-structured-output)
5. [Output Types](#output-types)
6. [JSON Schema Examples](#json-schema-examples)
7. [Custom Validators](#custom-validators)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

Create an agent with structured output that extracts contact information:

```bash
curl -X POST http://localhost:8000/workers \
  -H "Content-Type: application/json" \
  -d '{
    "worker_name": "contact_extractor",
    "agent_role": "Contact Information Extractor",
    "agent_type": "pydantic_ai",
    "model_provider": "inference_router",
    "model": "local-llm",
    "system_prompt": "Extract contact information from text.",
    "output_schema": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "email": {"type": "string", "format": "email"},
        "phone": {"type": "string"},
        "company": {"type": "string"}
      },
      "required": ["name", "email"]
    }
  }'
```

Send a message and get structured output:

```bash
curl -X POST http://localhost:8000/workers/contact_extractor/message \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hi, I am John Doe from Acme Corp. You can reach me at john.doe@acme.com or 555-1234."
  }'
```

Response:

```json
{
  "output": {
    "name": "John Doe",
    "email": "john.doe@acme.com",
    "phone": "555-1234",
    "company": "Acme Corp"
  },
  "output_type": "object",
  "output_schema_used": {
    "type": "object",
    "properties": {
      "name": {"type": "string"},
      "email": {"type": "string", "format": "email"},
      "phone": {"type": "string"},
      "company": {"type": "string"}
    },
    "required": ["name", "email"]
  },
  "metadata": null,
  "usage_metrics": {
    "prompt_tokens": 45,
    "completion_tokens": 23,
    "total_tokens": 68
  }
}
```

---

## Creating Workers with Structured Output

### Endpoint

```http
POST /workers
```

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `worker_name` | string | Yes | Unique identifier for the worker |
| `agent_role` | string | Yes | Role description for the agent |
| `agent_type` | string | No | Defaults to "pydantic_ai" |
| `model_provider` | string | Yes | "deepseek", "anthropic", "openai", "groq", "cohere", "mlx" |
| `model` | string | Yes | Model name — `local-llm` for every platform worker (the one pgx served-model-name) |
| `system_prompt` | string | No | System instructions for the agent |
| `output_schema` | object | No | JSON Schema for structured output |
| `output_validator` | string | No | Python code for custom validation |
| `temperature` | float | No | Model temperature (0.0-2.0, default: 0.0) |
| `tool_names` | array | No | List of tools the worker can use |
| `instruction` | string | No | Additional instructions |
| `user_prompt` | string | No | Default user prompt |
| `description` | string | No | Worker description |
| `mcp_config` | string/object | No | MCP configuration |
| `mcp_servers` | array | No | List of MCP server names |
| `base_url_override` | string | No | Custom API base URL |
| `api_key_override` | string | No | Custom API key |
| `tools_dir_override` | string | No | Custom tools directory |

### Output Schema Format

The `output_schema` parameter accepts a JSON Schema (Draft 7+) object. It **must** include a `type` field.

**Basic Requirements:**
- Must be a valid JSON object (dictionary)
- Must contain a `"type"` field
- Supports all standard JSON Schema features: properties, required, enum, format, etc.

### Examples

#### Simple String Output with Enum

```json
{
  "worker_name": "sentiment_analyzer",
  "agent_role": "Sentiment Analyzer",
  "model_provider": "inference_router",
  "model": "local-llm",
  "system_prompt": "Analyze sentiment of text.",
  "output_schema": {
    "type": "string",
    "enum": ["positive", "negative", "neutral"]
  }
}
```

#### Boolean Output

```json
{
  "worker_name": "spam_detector",
  "agent_role": "Spam Detector",
  "model_provider": "inference_router",
  "model": "local-llm",
  "system_prompt": "Determine if text is spam.",
  "output_schema": {
    "type": "boolean"
  }
}
```

#### Array Output

```json
{
  "worker_name": "keyword_extractor",
  "agent_role": "Keyword Extractor",
  "model_provider": "inference_router",
  "model": "local-llm",
  "system_prompt": "Extract keywords from text.",
  "output_schema": {
    "type": "array",
    "items": {
      "type": "string"
    },
    "minItems": 1,
    "maxItems": 10
  }
}
```

#### Complex Nested Object

```json
{
  "worker_name": "resume_parser",
  "agent_role": "Resume Parser",
  "model_provider": "inference_router",
  "model": "local-llm",
  "system_prompt": "Parse resume information.",
  "output_schema": {
    "type": "object",
    "properties": {
      "personal_info": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "email": {"type": "string", "format": "email"},
          "phone": {"type": "string"}
        },
        "required": ["name", "email"]
      },
      "experience": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "company": {"type": "string"},
            "position": {"type": "string"},
            "start_date": {"type": "string", "format": "date"},
            "end_date": {"type": "string", "format": "date"},
            "description": {"type": "string"}
          },
          "required": ["company", "position"]
        }
      },
      "skills": {
        "type": "array",
        "items": {"type": "string"}
      }
    },
    "required": ["personal_info", "skills"]
  }
}
```

---

## Updating Worker Structured Output

You can update an existing worker's structured output configuration, including changing or removing the schema.

### Endpoint

```http
PUT /workers/{worker_name}
```

### Important Notes

1. **Worker name must match**: The `worker_name` in the path must match `worker_name` in the request body
2. **Cache is cleared**: The worker instance is removed from cache, forcing recreation with the new configuration
3. **All validation rules apply**: Same validation as worker creation

### Update Examples

#### Add Structured Output to Existing Worker

```bash
curl -X PUT http://localhost:8000/workers/data_analyst \
  -H "Content-Type: application/json" \
  -d '{
    "worker_name": "data_analyst",
    "agent_role": "Data Analyst",
    "model_provider": "inference_router",
    "model": "local-llm",
    "tool_names": ["query_database"],
    "system_prompt": "You are a data analyst.",
    "output_schema": {
      "type": "object",
      "properties": {
        "summary": {"type": "string"},
        "metrics": {
          "type": "object",
          "properties": {
            "total_records": {"type": "integer"},
            "average_value": {"type": "number"}
          }
        }
      },
      "required": ["summary", "metrics"]
    }
  }'
```

#### Change Output Schema

```bash
curl -X PUT http://localhost:8000/workers/contact_extractor \
  -H "Content-Type: application/json" \
  -d '{
    "worker_name": "contact_extractor",
    "agent_role": "Contact Information Extractor",
    "model_provider": "inference_router",
    "model": "local-llm",
    "output_schema": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "email": {"type": "string", "format": "email"},
        "phone": {"type": "string"},
        "company": {"type": "string"},
        "job_title": {"type": "string"},
        "linkedin": {"type": "string", "format": "uri"}
      },
      "required": ["name"]
    }
  }'
```

#### Remove Structured Output (Return to Plain Text)

Simply omit the `output_schema` field or set it to `null`:

```bash
curl -X PUT http://localhost:8000/workers/contact_extractor \
  -H "Content-Type: application/json" \
  -d '{
    "worker_name": "contact_extractor",
    "agent_role": "Contact Information Extractor",
    "model_provider": "inference_router",
    "model": "local-llm",
    "system_prompt": "Extract contact information from text.",
    "output_schema": null
  }'
```

---

## Sending Messages with Structured Output

### Endpoint

```http
POST /workers/{worker_name}/message
```

### Request Body

```json
{
  "message": "string (required, 1-10000 chars)",
  "context": {
    "key": "value"
  },
  "system_prompt": "optional override",
  "instruction": "optional override"
}
```

### Response Structure

```json
{
  "output": "string | object | array | boolean | other",
  "output_type": "string|object|array|boolean|enum|other",
  "output_schema_used": {
    "type": "object",
    "properties": {}
  },
  "metadata": {
    "context": {}
  },
  "usage_metrics": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `output` | any | The agent's output (type depends on schema) |
| `output_type` | string | Type classification of the output |
| `output_schema_used` | object/null | The JSON schema applied (if any) |
| `metadata` | object/null | Optional metadata including context |
| `usage_metrics` | object/null | Token usage from the model provider |

---

## Output Types

The `output_type` field helps clients understand the structure of the response:

| Type | Description | Example |
|------|-------------|---------|
| `string` | Plain text or enum string | `"positive"`, `"Hello world"` |
| `object` | JSON object/dictionary | `{"name": "John", "age": 30}` |
| `array` | List/array | `["apple", "banana", "cherry"]` |
| `boolean` | Boolean value | `true`, `false` |
| `enum` | String from predefined set | `"positive"` (from enum: ["positive", "negative", "neutral"]) |
| `other` | Any other type | Complex types not covered above |

### Output Type Detection

The system automatically determines the output type based on:

1. **Boolean**: Direct boolean values
2. **Enum**: String values when schema contains an `"enum"` field
3. **String**: String values without enum constraint
4. **Object**: Dictionary/object values
5. **Array**: List values
6. **Other**: Anything else

---

## JSON Schema Examples

### Email Validator

```json
{
  "type": "object",
  "properties": {
    "email": {
      "type": "string",
      "format": "email"
    },
    "is_valid": {
      "type": "boolean"
    },
    "domain": {
      "type": "string"
    }
  },
  "required": ["email", "is_valid"]
}
```

### Multi-Choice Question

```json
{
  "type": "object",
  "properties": {
    "question": {
      "type": "string"
    },
    "answer": {
      "type": "string",
      "enum": ["A", "B", "C", "D"]
    },
    "confidence": {
      "type": "number",
      "minimum": 0,
      "maximum": 1
    }
  },
  "required": ["answer"]
}
```

### Product Recommendation

```json
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "product_id": {
        "type": "string"
      },
      "product_name": {
        "type": "string"
      },
      "relevance_score": {
        "type": "number",
        "minimum": 0,
        "maximum": 10
      },
      "reason": {
        "type": "string"
      }
    },
    "required": ["product_id", "product_name", "relevance_score"]
  },
  "minItems": 1,
  "maxItems": 5
}
```

### Classification with Confidence

```json
{
  "type": "object",
  "properties": {
    "category": {
      "type": "string",
      "enum": ["technology", "business", "entertainment", "sports", "health"]
    },
    "subcategory": {
      "type": "string"
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    },
    "keywords": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "maxItems": 10
    }
  },
  "required": ["category", "confidence"]
}
```

---

## Custom Validators

In addition to JSON Schema validation, you can provide custom Python validation logic using the `output_validator` parameter.

### Format

The `output_validator` is a string containing Python code that **must** define a `validate(ctx, output)` function.

### Function Signature

```python
def validate(ctx, output):
    """
    Custom validation function.

    Args:
        ctx: Context object (contains agent context, tools, etc.)
        output: The output from the agent (after JSON Schema validation)

    Returns:
        The validated output (can be modified)

    Raises:
        ValueError: If validation fails
    """
    # Your validation logic here
    return output
```

### Example: Email Domain Validator

```json
{
  "worker_name": "email_extractor",
  "agent_role": "Email Extractor",
  "model_provider": "inference_router",
  "model": "local-llm",
  "output_schema": {
    "type": "object",
    "properties": {
      "email": {"type": "string", "format": "email"}
    },
    "required": ["email"]
  },
  "output_validator": "def validate(ctx, output):\n    email = output.get('email', '')\n    if not email.endswith('@company.com'):\n        raise ValueError('Email must be from company.com domain')\n    return output"
}
```

### Example: Business Hours Validator

```json
{
  "worker_name": "appointment_scheduler",
  "agent_role": "Appointment Scheduler",
  "model_provider": "inference_router",
  "model": "local-llm",
  "output_schema": {
    "type": "object",
    "properties": {
      "date": {"type": "string", "format": "date"},
      "time": {"type": "string", "pattern": "^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$"}
    },
    "required": ["date", "time"]
  },
  "output_validator": "from datetime import datetime\n\ndef validate(ctx, output):\n    time_str = output['time']\n    hour = int(time_str.split(':')[0])\n    if hour < 9 or hour > 17:\n        raise ValueError('Appointments must be between 9 AM and 5 PM')\n    return output"
}
```

### Example: Budget Constraint Validator

```json
{
  "worker_name": "budget_calculator",
  "agent_role": "Budget Calculator",
  "model_provider": "inference_router",
  "model": "local-llm",
  "output_schema": {
    "type": "object",
    "properties": {
      "total": {"type": "number"},
      "items": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "cost": {"type": "number"}
          }
        }
      }
    },
    "required": ["total", "items"]
  },
  "output_validator": "def validate(ctx, output):\n    max_budget = ctx.get('max_budget', 10000)\n    if output['total'] > max_budget:\n        raise ValueError(f'Total ${output[\"total\"]} exceeds budget ${max_budget}')\n    \n    # Verify sum of items matches total\n    calculated_total = sum(item['cost'] for item in output['items'])\n    if abs(calculated_total - output['total']) > 0.01:\n        raise ValueError('Item costs do not match total')\n    \n    return output"
}
```

### Validator Notes

1. **Must define `validate` function**: The code must include `def validate(ctx, output):`
2. **Syntax validation**: Code is checked for syntax errors during worker creation
3. **Execution order**: JSON Schema validation runs first, then custom validator
4. **Error handling**: Raise `ValueError` with a descriptive message for validation failures
5. **Modification allowed**: You can modify and return the output if needed

---

## Best Practices

### 1. Start Simple

Begin with a basic schema and add complexity as needed:

```json
{
  "type": "object",
  "properties": {
    "result": {"type": "string"}
  }
}
```

### 2. Use Required Fields Sparingly

Only mark fields as required if they are absolutely necessary:

```json
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "email": {"type": "string"},
    "phone": {"type": "string"}
  },
  "required": ["email"]
}
```

### 3. Provide Clear System Prompts

Help the model understand what structure to produce:

```json
{
  "system_prompt": "Extract contact information and return it as a JSON object with fields: name, email, phone, and company. Always include at least the email address."
}
```

### 4. Use Format Validators

Leverage built-in JSON Schema formats:

- `"format": "email"` - Email addresses
- `"format": "uri"` - URLs
- `"format": "date"` - ISO 8601 dates (YYYY-MM-DD)
- `"format": "date-time"` - ISO 8601 timestamps
- `"format": "ipv4"` - IPv4 addresses
- `"format": "uuid"` - UUIDs

### 5. Set Reasonable Limits

Use `minItems`, `maxItems`, `minimum`, `maximum` to constrain outputs:

```json
{
  "type": "array",
  "items": {"type": "string"},
  "minItems": 1,
  "maxItems": 10
}
```

### 6. Test with Edge Cases

Test your schema with:
- Empty inputs
- Minimal valid inputs
- Maximum complexity inputs
- Invalid formats
- Missing required fields

### 7. Handle Validation Errors Gracefully

Structure your client code to handle validation failures:

```python
try:
    response = await client.post(f"/workers/{worker_name}/message", json=request)
    data = response.json()
    # Handle structured output
    structured_data = data["output"]
except HTTPException as e:
    if e.status_code == 500:
        # Validation or processing error
        logger.error(f"Worker failed: {e.detail}")
```

### 8. Version Your Schemas

When updating schemas, consider backward compatibility:

```json
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "email": {"type": "string"},
    "phone": {"type": "string"}
  },
  "required": ["name"]
}
```

Later version (adding optional field):

```json
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "email": {"type": "string"},
    "phone": {"type": "string"},
    "address": {"type": "string"}
  },
  "required": ["name"]
}
```

### 9. Document Your Workers

Keep track of what each worker's schema expects:

```python
WORKER_SCHEMAS = {
    "contact_extractor": {
        "description": "Extracts contact information from text",
        "input": "Free-form text containing contact details",
        "output": "Object with name, email, phone, company fields",
    }
}
```

---

## Troubleshooting

### Issue: Worker Returns String Instead of Object

**Cause**: No `output_schema` defined or schema not applied correctly.

**Solution**: Verify the schema was set during worker creation:

```bash
curl -X GET http://localhost:8000/workers/my_worker
```

Check that the response includes `output_schema` in the config.

### Issue: Validation Errors in Output

**Cause**: Model output doesn't match schema requirements.

**Solution**:
1. Review your `system_prompt` - make it more explicit about required format
2. Use examples in the prompt
3. Check if required fields are too strict
4. Consider using a more capable model (e.g., gpt-4o instead of gpt-3.5-turbo)

### Issue: Output Schema Not Being Applied

**Cause**: Worker cache contains old configuration.

**Solution**: Update the worker configuration to clear cache:

```bash
curl -X PUT http://localhost:8000/workers/{worker_name} \
  -H "Content-Type: application/json" \
  -d '{
    "worker_name": "my_worker",
    ...
    "output_schema": {...}
  }'
```

### Issue: Custom Validator Errors

**Cause**: Syntax errors or missing `validate` function.

**Solution**:
1. Verify Python syntax: `python -m py_compile validator.py`
2. Ensure function signature is correct: `def validate(ctx, output):`
3. Test validator logic independently before adding to worker

### Issue: Output Type Shows "other"

**Cause**: Output doesn't match standard types or schema is complex.

**Solution**: This is informational and doesn't affect functionality. The actual output structure is in the `output` field.

### Issue: Schema Too Restrictive

**Cause**: Model cannot consistently produce the required structure.

**Solution**:
1. Simplify the schema
2. Make fewer fields required
3. Use more flexible types
4. Provide clearer instructions in `system_prompt`

---

## API Reference Summary

### Create Worker with Structured Output

```http
POST /workers
Content-Type: application/json

{
  "worker_name": "string",
  "agent_role": "string",
  "model_provider": "openai|anthropic|deepseek|groq|cohere|mlx",
  "model": "string",
  "output_schema": {
    "type": "object|array|string|boolean|number",
    "properties": {...},
    "required": [...]
  },
  "output_validator": "def validate(ctx, output): ..."
}
```

### Update Worker Structured Output

```http
PUT /workers/{worker_name}
Content-Type: application/json

{
  "worker_name": "{worker_name}",
  "output_schema": {...}
}
```

### Send Message to Worker

```http
POST /workers/{worker_name}/message
Content-Type: application/json

{
  "message": "string"
}
```

**Response:**

```json
{
  "output": "any",
  "output_type": "string|object|array|boolean|enum|other",
  "output_schema_used": {...},
  "metadata": {...},
  "usage_metrics": {...}
}
```

---

## Related Documentation

- [Main README](README.md) - Complete API reference
- [Graph Execution Guide](README_GRAPH_EXECUTION.md) - Workflow graphs
- OpenAPI Specification: `GET /openapi.json`

---

## Support

For issues or questions:
- Check the [troubleshooting section](#troubleshooting)
- Review the OpenAPI documentation at `/docs`
- Examine worker state history: `GET /workers/{worker_name}/states`
- Check application logs for detailed error messages
