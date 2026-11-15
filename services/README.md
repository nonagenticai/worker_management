# Worker Context Enrichment System

## Overview

The Worker Context Enrichment System provides structured parsing and enrichment of worker execution context with metadata including procedures, requirements, principles, code snippets, entities, and semantic relationships.

## Architecture

### Components

1. **Parsed Data Models** (`models/parsed_data_models.py`)
   - `CodeSnippet`: Code with metadata (language, file path, confidence)
   - `Entity`: Semantic entities (functions, classes, concepts)
   - `SemanticRelationship`: Relationships between entities
   - `ParsedProcedure`: Structured procedure data with steps
   - `ParsedRequirement`: Requirements with acceptance criteria
   - `ParsedPrinciple`: Design principles and guidelines

2. **Worker Context** (`models/worker_context.py`)
   - Central data structure for enriched context
   - Contains both raw (backward compatible) and parsed data
   - Provides utility methods for querying and filtering
   - Tracks enrichment metrics and performance

3. **Context Orchestrator** (`services/orchestrator.py`)
   - Main service for parsing and enrichment
   - `AssetParser`: Utility for extracting structured data
   - `ContextOrchestrator`: Manages enrichment workflow
   - Performance monitoring and caching

## Key Features

### 1. Structured Metadata Extraction

The system automatically extracts:
- **Code Snippets**: From markdown code blocks or inline code
- **Entities**: Functions, classes, and concepts from text
- **Relationships**: Semantic connections between entities
- **Steps**: Numbered or bulleted procedure steps
- **Acceptance Criteria**: From requirement documents
- **Examples**: From principle descriptions

### 2. Confidence Scoring

All parsed items include confidence scores (0.0-1.0):
- High confidence (>0.9): Explicit code blocks, clear structures
- Medium confidence (0.7-0.9): Inferred structures, patterns
- Low confidence (<0.7): Ambiguous or uncertain extractions

### 3. Performance Monitoring

Enrichment metrics track:
- Duration in milliseconds
- Number of items parsed per category
- Parsing errors encountered
- Cache statistics

### 4. Backward Compatibility

The system maintains both raw and parsed data:
- **Raw data**: Original strings (for legacy code)
- **Parsed data**: Structured models (for new code)
- Existing systems continue to work unchanged

## Usage

### Basic Enrichment

```python
from worker_management.services.orchestrator import ContextOrchestrator

# Initialize orchestrator
orchestrator = ContextOrchestrator(enable_async=True)

# Prepare raw assets
raw_procedures = [
    \"\"\"
    1. Initialize database connection
    2. Execute query
    3. Return results

    ```python
    def execute_query(conn, sql):
        return conn.execute(sql).fetchall()
    ```
    \"\"\"
]

raw_requirements = [
    \"\"\"
    High Priority: API Response Time

    All API endpoints must respond within 200ms.

    Acceptance Criteria:
    - 95th percentile < 200ms
    - 99th percentile < 500ms
    \"\"\"
]

# Enrich context
context = await orchestrator.enrich_context(
    context_id="ctx_001",
    worker_id="worker_123",
    raw_procedures=raw_procedures,
    raw_requirements=raw_requirements,
    session_id="session_001",
)

# Access enriched data
print(f"Parsed {len(context.parsed_procedures)} procedures")
print(f"Extracted {len(context.extracted_code)} code snippets")
print(f"Found {len(context.semantic_entities)} entities")
print(f"Enrichment took {context.enrichment_metrics.enrichment_duration_ms}ms")
```

### Querying Context

```python
# Get all code snippets
all_code = context.get_all_code_snippets()

# Get entities by type
functions = context.get_entities_by_type("function")
classes = context.get_entities_by_type("class")

# Get high confidence items
high_conf = context.get_high_confidence_items(min_confidence=0.8)

# Access decision confidence
if "procedure_completeness" in context.decision_confidence:
    confidence = context.decision_confidence["procedure_completeness"]
    print(f"Procedure completeness: {confidence:.2%}")

# Get enrichment summary
summary = context.get_enrichment_summary()
```

### Performance Tracking

```python
# Check enrichment metrics
metrics = context.enrichment_metrics
print(f"Duration: {metrics.enrichment_duration_ms}ms")
print(f"Procedures parsed: {metrics.procedures_parsed}")
print(f"Errors: {len(metrics.parsing_errors)}")

# Cache management
cached = orchestrator.get_cached_context("ctx_001")
stats = orchestrator.get_cache_stats()
orchestrator.clear_cache()
```

## Data Models

### WorkerContext

```python
class WorkerContext(BaseModel):
    # Identification
    context_id: str
    worker_id: str
    session_id: Optional[str]
    created_at: datetime

    # Raw data (backward compatible)
    raw_procedures: List[str]
    raw_requirements: List[str]
    raw_principles: List[str]

    # Parsed structured data
    parsed_procedures: List[ParsedProcedure]
    parsed_requirements: List[ParsedRequirement]
    parsed_principles: List[ParsedPrinciple]
    extracted_code: List[CodeSnippet]
    semantic_entities: List[Entity]
    semantic_relationships: List[SemanticRelationship]

    # Decision support
    decision_confidence: Dict[str, float]
    recommended_actions: List[str]

    # Enrichment metadata
    enrichment_metrics: ContextEnrichmentMetrics
    enrichment_version: str
```

### ParsedProcedure

```python
class ParsedProcedure(BaseModel):
    procedure_id: str
    title: str
    description: Optional[str]
    steps: List[str]
    preconditions: List[str]
    postconditions: List[str]
    code_snippets: List[CodeSnippet]
    entities: List[Entity]
    tags: List[str]
    confidence: float
    source_file: Optional[str]
    metadata: Dict[str, Any]
    parsed_at: datetime
```

### ParsedRequirement

```python
class ParsedRequirement(BaseModel):
    requirement_id: str
    title: str
    description: str
    requirement_type: str  # functional, non-functional, constraint
    priority: str  # high, medium, low
    status: str
    acceptance_criteria: List[str]
    dependencies: List[str]
    code_snippets: List[CodeSnippet]
    entities: List[Entity]
    tags: List[str]
    confidence: float
    source_file: Optional[str]
    metadata: Dict[str, Any]
    parsed_at: datetime
```

### CodeSnippet

```python
class CodeSnippet(BaseModel):
    content: str
    language: str
    file_path: Optional[str]
    line_start: Optional[int]
    line_end: Optional[int]
    function_name: Optional[str]
    class_name: Optional[str]
    confidence: float  # 0.0 - 1.0
    metadata: Dict[str, Any]
    extracted_at: datetime
```

### Entity

```python
class Entity(BaseModel):
    entity_id: str
    entity_type: str  # function, class, module, concept
    name: str
    description: Optional[str]
    source: str
    attributes: Dict[str, Any]
    confidence: float
    tags: List[str]
    extracted_at: datetime
```

## Enrichment Flow

```
Raw Assets → AssetParser → Structured Data → WorkerContext
     ↓            ↓              ↓                ↓
 Procedures   Extract       ParsedProcedure    Enriched
 Requirements Code/Entities ParsedRequirement  Context
 Principles   Relationships CodeSnippet        with
                            Entity             Metadata
```

### Enrichment Steps

1. **Initialize Context**: Create WorkerContext with raw data
2. **Parse Procedures**: Extract steps, code, entities → ParsedProcedure
3. **Parse Requirements**: Extract criteria, priority → ParsedRequirement
4. **Parse Principles**: Extract examples, rationale → ParsedPrinciple
5. **Extract Metadata**: Aggregate code snippets and entities
6. **Identify Relationships**: Find semantic connections
7. **Calculate Confidence**: Score each decision type
8. **Index Terms**: Build search index
9. **Track Metrics**: Record performance data

## Performance Characteristics

### Benchmark Results (on test data)

| Dataset Size | Procedures | Duration | Items/sec |
|-------------|-----------|----------|-----------|
| Small       | 2         | ~5ms     | 400       |
| Medium      | 10        | ~15ms    | 667       |
| Large       | 100       | ~80ms    | 1,250     |

### Performance Tips

1. **Use Caching**: Reuse enriched contexts when possible
2. **Batch Processing**: Enrich multiple contexts in parallel
3. **Async Mode**: Enable async processing for better throughput
4. **Confidence Filtering**: Use high-confidence items for critical decisions

## Testing

### Run Unit Tests

```bash
# Run all tests
python -m pytest services/test_orchestrator.py -v

# Run specific test class
python -m pytest services/test_orchestrator.py::TestContextOrchestrator -v

# Run with coverage
python -m pytest services/test_orchestrator.py --cov=services --cov-report=html
```

### Run Examples

```bash
# Run usage examples
python services/example_usage.py
```

## Integration Guide

### Integration with Worker Factory

```python
from worker_management.services.orchestrator import ContextOrchestrator
from worker_management.models.worker_context import WorkerContext

# In worker creation
orchestrator = ContextOrchestrator()

# Enrich context before execution
context = await orchestrator.enrich_context(
    context_id=f"ctx_{worker_id}_{session_id}",
    worker_id=worker_id,
    raw_procedures=procedures,
    raw_requirements=requirements,
)

# Pass to worker
worker.execute_with_context(context)
```

### Integration with Graph Execution

```python
# In graph node execution
async def execute_with_enriched_context(node_id: str, state: WorkflowState):
    orchestrator = ContextOrchestrator()

    # Enrich context from state
    context = await orchestrator.enrich_context(
        context_id=f"ctx_{node_id}",
        worker_id=state.worker_id,
        raw_procedures=state.get("procedures", []),
        session_id=state.session_id,
    )

    # Use enriched context for decisions
    high_conf_procedures = context.get_high_confidence_items()["procedures"]

    # Execute with confidence-based routing
    if context.decision_confidence.get("procedure_completeness", 0) > 0.8:
        return await execute_high_confidence_path(context)
    else:
        return await execute_fallback_path(context)
```

## Error Handling

The system is designed to be resilient:

- **Parsing Errors**: Logged but don't stop enrichment
- **Validation Errors**: Captured in metrics for analysis
- **Missing Data**: Gracefully handled with defaults
- **Performance Issues**: Tracked and reported in metrics

```python
# Check for errors
if context.enrichment_metrics.parsing_errors:
    for error in context.enrichment_metrics.parsing_errors:
        logger.warning(f"Parsing error: {error}")

# Fallback to raw data if needed
if not context.parsed_procedures and context.raw_procedures:
    # Use raw procedures as fallback
    procedures = context.raw_procedures
```

## Future Enhancements

Potential improvements:

1. **ML-Based Parsing**: Use LLMs for better entity extraction
2. **Semantic Search**: Vector embeddings for context search
3. **Real-time Updates**: Stream parsing results as they're available
4. **Graph Visualization**: Visual representation of entity relationships
5. **Auto-Tagging**: Automatic categorization of assets
6. **Quality Scoring**: Overall quality metrics for context
7. **Version History**: Track context changes over time
8. **Cross-Reference**: Link related items across contexts

## API Reference

### ContextOrchestrator

#### Methods

- `enrich_context(...)`: Main enrichment method
- `enrich_with_procedures(...)`: Parse procedures
- `enrich_with_requirements(...)`: Parse requirements
- `enrich_with_principles(...)`: Parse principles
- `enrich_with_metadata(...)`: Extract metadata
- `get_cached_context(context_id)`: Retrieve from cache
- `clear_cache()`: Clear enrichment cache
- `get_cache_stats()`: Get cache statistics

### WorkerContext

#### Methods

- `get_all_code_snippets()`: Aggregate all code
- `get_all_entities()`: Aggregate all entities
- `get_entities_by_type(type)`: Filter entities
- `get_high_confidence_items(min_confidence)`: Filter by confidence
- `get_enrichment_summary()`: Get summary dict
- `add_decision_confidence(key, score)`: Add confidence score
- `index_terms()`: Build search index

## License

Part of the Worker Management System.

## Support

For issues or questions:
- Check the test files for usage examples
- Review the example_usage.py for common patterns
- Consult the inline documentation in the code
