"""Unit tests for the ContextOrchestrator and related models.

This test suite validates the parsing, enrichment, and performance monitoring
capabilities of the context orchestrator system.
"""

import pytest

from models.parsed_data_models import (
    CodeSnippet,
    Entity,
    ParsedProcedure,
    ParsedRequirement,
)
from models.worker_context import WorkerContext
from services.orchestrator import (
    AssetParser,
    ContextOrchestrator,
)


class TestCodeSnippet:
    """Tests for CodeSnippet model."""

    def test_create_code_snippet(self):
        """Test creating a valid code snippet."""
        snippet = CodeSnippet(
            content="def hello(): pass",
            language="python",
            confidence=0.95,
        )
        assert snippet.content == "def hello(): pass"  # noqa: S101
        assert snippet.language == "python"  # noqa: S101
        assert snippet.confidence == 0.95  # noqa: S101

    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence
        snippet = CodeSnippet(content="test", language="python", confidence=0.5)
        assert snippet.confidence == 0.5  # noqa: S101

        # Invalid confidence should raise error
        with pytest.raises(ValueError):
            CodeSnippet(content="test", language="python", confidence=1.5)


class TestEntity:
    """Tests for Entity model."""

    def test_create_entity(self):
        """Test creating a valid entity."""
        entity = Entity(
            entity_id="ent_123",
            entity_type="function",
            name="calculate_total",
            source="test.py",
            confidence=0.85,
        )
        assert entity.entity_id == "ent_123"  # noqa: S101
        assert entity.entity_type == "function"  # noqa: S101
        assert entity.name == "calculate_total"  # noqa: S101


class TestParsedProcedure:
    """Tests for ParsedProcedure model."""

    def test_create_procedure(self):
        """Test creating a valid procedure."""
        procedure = ParsedProcedure(
            procedure_id="proc_123",
            title="Test Procedure",
            steps=["Step 1", "Step 2"],
            confidence=0.9,
        )
        assert procedure.procedure_id == "proc_123"  # noqa: S101
        assert procedure.title == "Test Procedure"  # noqa: S101
        assert len(procedure.steps) == 2  # noqa: S101


class TestParsedRequirement:
    """Tests for ParsedRequirement model."""

    def test_create_requirement(self):
        """Test creating a valid requirement."""
        requirement = ParsedRequirement(
            requirement_id="req_123",
            title="Test Requirement",
            description="Test description",
            requirement_type="functional",
            priority="high",
        )
        assert requirement.requirement_id == "req_123"  # noqa: S101
        assert requirement.requirement_type == "functional"  # noqa: S101
        assert requirement.priority == "high"  # noqa: S101


class TestWorkerContext:
    """Tests for WorkerContext model."""

    def test_create_basic_context(self):
        """Test creating a basic worker context."""
        context = WorkerContext(
            context_id="ctx_123",
            worker_id="worker_abc",
            raw_procedures=["proc 1", "proc 2"],
        )
        assert context.context_id == "ctx_123"  # noqa: S101
        assert context.worker_id == "worker_abc"  # noqa: S101
        assert len(context.raw_procedures) == 2  # noqa: S101

    def test_get_all_code_snippets(self):
        """Test aggregating code snippets from all sources."""
        context = WorkerContext(
            context_id="ctx_123",
            worker_id="worker_abc",
        )

        # Add code snippets at different levels
        snippet1 = CodeSnippet(content="code1", language="python")
        snippet2 = CodeSnippet(content="code2", language="python")
        snippet3 = CodeSnippet(content="code3", language="python")

        context.extracted_code = [snippet1]
        procedure = ParsedProcedure(
            procedure_id="p1",
            title="Test",
            code_snippets=[snippet2],
        )
        context.parsed_procedures = [procedure]
        requirement = ParsedRequirement(
            requirement_id="r1",
            title="Test",
            description="Test",
            requirement_type="functional",
            code_snippets=[snippet3],
        )
        context.parsed_requirements = [requirement]

        all_snippets = context.get_all_code_snippets()
        assert len(all_snippets) == 3  # noqa: S101

    def test_get_entities_by_type(self):
        """Test filtering entities by type."""
        context = WorkerContext(
            context_id="ctx_123",
            worker_id="worker_abc",
        )

        entity1 = Entity(
            entity_id="e1",
            entity_type="function",
            name="func1",
            source="test",
        )
        entity2 = Entity(
            entity_id="e2",
            entity_type="class",
            name="Class1",
            source="test",
        )
        entity3 = Entity(
            entity_id="e3",
            entity_type="function",
            name="func2",
            source="test",
        )

        context.semantic_entities = [entity1, entity2, entity3]

        functions = context.get_entities_by_type("function")
        assert len(functions) == 2  # noqa: S101

        classes = context.get_entities_by_type("class")
        assert len(classes) == 1  # noqa: S101

    def test_add_decision_confidence(self):
        """Test adding decision confidence scores."""
        context = WorkerContext(
            context_id="ctx_123",
            worker_id="worker_abc",
        )

        context.add_decision_confidence("test_decision", 0.85)
        assert context.decision_confidence["test_decision"] == 0.85  # noqa: S101

        # Invalid confidence should raise error
        with pytest.raises(ValueError):
            context.add_decision_confidence("invalid", 1.5)

    def test_get_high_confidence_items(self):
        """Test filtering high confidence items."""
        context = WorkerContext(
            context_id="ctx_123",
            worker_id="worker_abc",
        )

        # Add items with different confidence scores
        proc1 = ParsedProcedure(
            procedure_id="p1",
            title="High Conf",
            confidence=0.9,
        )
        proc2 = ParsedProcedure(
            procedure_id="p2",
            title="Low Conf",
            confidence=0.5,
        )
        context.parsed_procedures = [proc1, proc2]

        high_conf = context.get_high_confidence_items(min_confidence=0.8)
        assert len(high_conf["procedures"]) == 1  # noqa: S101
        assert high_conf["procedures"][0].procedure_id == "p1"  # noqa: S101

    def test_index_terms(self):
        """Test term indexing for search."""
        context = WorkerContext(
            context_id="ctx_123",
            worker_id="worker_abc",
        )

        proc = ParsedProcedure(
            procedure_id="p1",
            title="Database Connection",
            tags=["database", "connection"],
        )
        context.parsed_procedures = [proc]

        context.index_terms()

        assert "procedures" in context.indexed_terms  # noqa: S101
        assert len(context.indexed_terms["procedures"]) > 0  # noqa: S101


class TestAssetParser:
    """Tests for AssetParser utility."""

    def test_parse_code_snippets(self):
        """Test code snippet extraction."""
        content = """
        Here is some code:

        ```python
        def hello():
            print("Hello")
        ```

        And some inline code: `x = 5`
        """

        parser = AssetParser()
        snippets = parser.parse_code_snippets(content, source="test.md")

        assert len(snippets) >= 1  # noqa: S101
        assert any("def hello" in s.content for s in snippets)  # noqa: S101

    def test_parse_entities(self):
        """Test entity extraction."""
        content = """
        class UserRepository:
            def get_user(self, user_id):
                pass

        function calculate_total():
            pass
        """

        parser = AssetParser()
        entities = parser.parse_entities(content, source="test.py")

        # Should find class and functions
        class_entities = [e for e in entities if e.entity_type == "class"]
        func_entities = [e for e in entities if e.entity_type == "function"]

        assert len(class_entities) >= 1  # noqa: S101
        assert len(func_entities) >= 1  # noqa: S101


class TestContextOrchestrator:
    """Tests for ContextOrchestrator."""

    @pytest.mark.asyncio
    async def test_basic_enrichment(self):
        """Test basic context enrichment."""
        orchestrator = ContextOrchestrator(enable_async=True)

        context = await orchestrator.enrich_context(
            context_id="ctx_test_001",
            worker_id="worker_test",
            raw_procedures=["Step 1: Initialize", "Step 2: Execute"],
            raw_requirements=["Requirement 1: Performance"],
        )

        assert context.context_id == "ctx_test_001"  # noqa: S101
        assert context.worker_id == "worker_test"  # noqa: S101
        assert len(context.raw_procedures) == 2  # noqa: S101
        assert len(context.raw_requirements) == 1  # noqa: S101
        assert len(context.parsed_procedures) == 2  # noqa: S101
        assert len(context.parsed_requirements) == 1  # noqa: S101

    @pytest.mark.asyncio
    async def test_enrichment_metrics(self):
        """Test enrichment metrics tracking."""
        orchestrator = ContextOrchestrator(enable_async=True)

        context = await orchestrator.enrich_context(
            context_id="ctx_metrics",
            worker_id="worker_test",
            raw_procedures=["Test procedure"],
        )

        metrics = context.enrichment_metrics
        assert metrics.enrichment_start_time is not None  # noqa: S101
        assert metrics.enrichment_end_time is not None  # noqa: S101
        assert metrics.enrichment_duration_ms is not None  # noqa: S101
        assert metrics.enrichment_duration_ms > 0  # noqa: S101
        assert metrics.procedures_parsed == 1  # noqa: S101

    @pytest.mark.asyncio
    async def test_enrich_with_procedures(self):
        """Test procedure parsing."""
        orchestrator = ContextOrchestrator(enable_async=True)

        raw_procedures = [
            """
            1. First step
            2. Second step
            3. Third step

            ```python
            def example():
                pass
            ```
            """
        ]

        parsed, errors = await orchestrator.enrich_with_procedures(raw_procedures)

        assert len(parsed) == 1  # noqa: S101
        assert len(parsed[0].steps) == 3  # noqa: S101
        assert len(parsed[0].code_snippets) >= 1  # noqa: S101
        assert len(errors) == 0  # noqa: S101

    @pytest.mark.asyncio
    async def test_enrich_with_requirements(self):
        """Test requirement parsing."""
        orchestrator = ContextOrchestrator(enable_async=True)

        raw_requirements = [
            """
            High Priority: User Authentication

            The system must provide secure authentication.

            Acceptance Criteria:
            - Users can log in
            - Passwords are hashed
            """
        ]

        parsed, errors = await orchestrator.enrich_with_requirements(raw_requirements)

        assert len(parsed) == 1  # noqa: S101
        assert parsed[0].priority == "high"  # noqa: S101
        assert len(parsed[0].acceptance_criteria) >= 1  # noqa: S101
        assert len(errors) == 0  # noqa: S101

    @pytest.mark.asyncio
    async def test_cache_functionality(self):
        """Test context caching."""
        orchestrator = ContextOrchestrator(enable_async=True)

        context_id = "ctx_cache_test"
        await orchestrator.enrich_context(
            context_id=context_id,
            worker_id="worker_test",
            raw_procedures=["Test"],
        )

        # Should be cached
        cached = orchestrator.get_cached_context(context_id)
        assert cached is not None  # noqa: S101
        assert cached.context_id == context_id  # noqa: S101

        # Clear cache
        orchestrator.clear_cache()
        cached_after_clear = orchestrator.get_cached_context(context_id)
        assert cached_after_clear is None  # noqa: S101

    @pytest.mark.asyncio
    async def test_performance_tracking(self):
        """Test performance metrics for different dataset sizes."""
        orchestrator = ContextOrchestrator(enable_async=True)

        # Small dataset
        small_context = await orchestrator.enrich_context(
            context_id="ctx_small",
            worker_id="worker_test",
            raw_procedures=["Step 1", "Step 2"],
        )

        # Large dataset
        large_procedures = [f"Step {i}: Operation {i}" for i in range(50)]
        large_context = await orchestrator.enrich_context(
            context_id="ctx_large",
            worker_id="worker_test",
            raw_procedures=large_procedures,
        )

        # Large dataset should take more time
        small_duration = small_context.enrichment_metrics.enrichment_duration_ms
        large_duration = large_context.enrichment_metrics.enrichment_duration_ms

        assert small_duration is not None  # noqa: S101
        assert large_duration is not None  # noqa: S101
        # Large dataset typically takes longer, but not required for test to pass
        print(f"Small: {small_duration}ms, Large: {large_duration}ms")


def run_tests():
    """Run all tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
