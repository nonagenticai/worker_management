"""Example usage of the ContextOrchestrator for enriching worker context.

This script demonstrates how to use the ContextOrchestrator to parse raw assets
and create enriched WorkerContext with structured metadata.
"""

import asyncio

from services.orchestrator import ContextOrchestrator


async def example_basic_enrichment():
    """Demonstrate basic context enrichment."""
    print("\n=== Basic Context Enrichment Example ===\n")

    # Sample raw procedures
    raw_procedures = [
        """
        1. Initialize the database connection
        2. Validate user credentials
        3. Execute the query
        4. Return results

        ```python
        def execute_query(connection, query):
            cursor = connection.cursor()
            cursor.execute(query)
            return cursor.fetchall()
        ```
        """,
        """
        - Check system requirements
        - Install dependencies
        - Configure environment variables
        - Run tests
        """,
    ]

    # Sample raw requirements
    raw_requirements = [
        """
        High Priority Requirement: User Authentication

        The system must provide secure authentication for all users.

        Acceptance Criteria:
        - Users can log in with email and password
        - Passwords are hashed using bcrypt
        - Failed login attempts are logged
        - Session timeout after 30 minutes
        """,
        """
        Performance Requirement: Response Time

        API endpoints must respond within 200ms for 95% of requests
        under normal load conditions.
        """,
    ]

    # Sample raw principles
    raw_principles = [
        """
        Design Pattern: Repository Pattern

        Use the repository pattern to abstract data access logic from business logic.

        Example:
        The UserRepository class handles all database operations for User entities,
        keeping the UserService focused on business rules.

        ```python
        class UserRepository:
            def get_by_id(self, user_id: int):
                return self.session.query(User).filter_by(id=user_id).first()
        ```
        """,
    ]

    # Initialize orchestrator
    orchestrator = ContextOrchestrator(enable_async=True)

    # Enrich context
    context = await orchestrator.enrich_context(
        context_id="ctx_example_001",
        worker_id="worker_demo",
        raw_procedures=raw_procedures,
        raw_requirements=raw_requirements,
        raw_principles=raw_principles,
        session_id="session_demo_001",
        metadata={"example": True, "demo_version": "1.0"},
    )

    # Display results
    print(f"Context ID: {context.context_id}")
    print(f"Worker ID: {context.worker_id}")
    print(f"Created at: {context.created_at}")
    print("\nEnrichment Summary:")
    summary = context.get_enrichment_summary()
    print(f"  Parsed Procedures: {summary['parsed_counts']['procedures']}")
    print(f"  Parsed Requirements: {summary['parsed_counts']['requirements']}")
    print(f"  Parsed Principles: {summary['parsed_counts']['principles']}")
    print(f"  Code Snippets: {summary['parsed_counts']['code_snippets']}")
    print(f"  Entities: {summary['parsed_counts']['entities']}")
    print(f"  Relationships: {summary['parsed_counts']['relationships']}")

    # Display enrichment metrics
    metrics = context.enrichment_metrics
    print("\nEnrichment Metrics:")
    print(f"  Duration: {metrics.enrichment_duration_ms:.2f}ms")
    print(f"  Procedures Parsed: {metrics.procedures_parsed}")
    print(f"  Requirements Parsed: {metrics.requirements_parsed}")
    print(f"  Principles Parsed: {metrics.principles_parsed}")
    print(f"  Code Snippets Extracted: {metrics.code_snippets_extracted}")
    print(f"  Entities Extracted: {metrics.entities_extracted}")
    print(f"  Errors: {len(metrics.parsing_errors)}")

    # Display confidence scores
    print("\nDecision Confidence Scores:")
    for decision, confidence in context.decision_confidence.items():
        print(f"  {decision}: {confidence:.2%}")

    # Display some parsed items
    if context.parsed_procedures:
        print("\nFirst Parsed Procedure:")
        proc = context.parsed_procedures[0]
        print(f"  ID: {proc.procedure_id}")
        print(f"  Title: {proc.title}")
        print(f"  Steps: {len(proc.steps)}")
        print(f"  Code Snippets: {len(proc.code_snippets)}")
        print(f"  Entities: {len(proc.entities)}")
        print(f"  Confidence: {proc.confidence:.2%}")

    if context.parsed_requirements:
        print("\nFirst Parsed Requirement:")
        req = context.parsed_requirements[0]
        print(f"  ID: {req.requirement_id}")
        print(f"  Title: {req.title}")
        print(f"  Type: {req.requirement_type}")
        print(f"  Priority: {req.priority}")
        print(f"  Acceptance Criteria: {len(req.acceptance_criteria)}")
        print(f"  Confidence: {req.confidence:.2%}")

    # Display entities
    if context.semantic_entities:
        print("\nExtracted Entities:")
        for entity in context.semantic_entities[:5]:  # Show first 5
            print(f"  - {entity.name} ({entity.entity_type}) [confidence: {entity.confidence:.2%}]")

    # Display high confidence items
    print("\nHigh Confidence Items (>80%):")
    high_conf = context.get_high_confidence_items(min_confidence=0.8)
    for item_type, items in high_conf.items():
        print(f"  {item_type}: {len(items)}")

    return context


async def example_performance_comparison() -> None:
    """Demonstrate performance monitoring."""
    print("\n\n=== Performance Comparison Example ===\n")

    orchestrator = ContextOrchestrator(enable_async=True)

    # Test with small dataset
    small_procedures = ["Step 1: Do something", "Step 2: Do something else"]
    context_small = await orchestrator.enrich_context(
        context_id="ctx_perf_small",
        worker_id="worker_perf",
        raw_procedures=small_procedures,
    )

    # Test with larger dataset
    large_procedures = [f"Step {i}: Perform operation {i}" for i in range(100)]
    context_large = await orchestrator.enrich_context(
        context_id="ctx_perf_large",
        worker_id="worker_perf",
        raw_procedures=large_procedures,
    )

    print("Small Dataset:")
    print(f"  Items: {len(small_procedures)}")
    print(f"  Duration: {context_small.enrichment_metrics.enrichment_duration_ms:.2f}ms")

    print("\nLarge Dataset:")
    print(f"  Items: {len(large_procedures)}")
    print(f"  Duration: {context_large.enrichment_metrics.enrichment_duration_ms:.2f}ms")

    print("\nCache Stats:")
    stats = orchestrator.get_cache_stats()
    print(f"  Cached Contexts: {stats['cached_contexts']}")


async def example_backward_compatibility() -> None:
    """Demonstrate backward compatibility with raw data."""
    print("\n\n=== Backward Compatibility Example ===\n")

    orchestrator = ContextOrchestrator(enable_async=True)

    # Create context with only raw data (no parsing)
    context = await orchestrator.enrich_context(
        context_id="ctx_compat",
        worker_id="worker_compat",
        raw_procedures=["Raw procedure 1", "Raw procedure 2"],
        raw_requirements=["Raw requirement 1"],
    )

    print("Raw Data (Backward Compatible):")
    print(f"  Raw Procedures: {len(context.raw_procedures)}")
    print(f"  Raw Requirements: {len(context.raw_requirements)}")
    print(f"  Raw Principles: {len(context.raw_principles)}")

    print("\nParsed Data:")
    print(f"  Parsed Procedures: {len(context.parsed_procedures)}")
    print(f"  Parsed Requirements: {len(context.parsed_requirements)}")
    print(f"  Parsed Principles: {len(context.parsed_principles)}")

    print("\nBoth raw and parsed data are available!")
    print("Context can be used by legacy code (raw data) and new code (parsed data)")


async def main() -> None:
    """Run all examples."""
    print("=" * 70)
    print("ContextOrchestrator Usage Examples")
    print("=" * 70)

    await example_basic_enrichment()
    await example_performance_comparison()
    await example_backward_compatibility()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
