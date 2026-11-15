"""Context orchestrator service for enriching worker context with parsed metadata.

This module provides the ContextOrchestrator class which handles parsing assets
and enriching WorkerContext with structured metadata including procedures,
requirements, principles, code snippets, entities, and semantic relationships.
"""

import logging
import re
import time
from datetime import datetime
from typing import Any
from uuid import uuid7

from pydantic import ValidationError

# NOTE: Import from local models package to avoid relying on a
# distribution-time package name. These modules live at repo root
# under `models/` and are available on the test path.
from models.parsed_data_models import (
    CodeSnippet,
    Entity,
    ParsedPrinciple,
    ParsedProcedure,
    ParsedRequirement,
    SemanticRelationship,
)
from models.worker_context import (
    ContextEnrichmentMetrics,
    WorkerContext,
)

logger = logging.getLogger(__name__)


class AssetParser:
    """Parser for extracting structured data from raw assets."""

    @staticmethod
    def parse_code_snippets(content: str, source: str = "") -> list[CodeSnippet]:
        """Extract code snippets from content with metadata."""
        snippets = []
        # Match code blocks (markdown style or indented)
        code_block_pattern = r"```(\w+)?\n(.*?)```|`([^`]+)`"
        matches = re.finditer(code_block_pattern, content, re.DOTALL)

        for idx, match in enumerate(matches):
            language = match.group(1) or "unknown"
            code_content = match.group(2) or match.group(3) or ""

            if code_content.strip():
                snippet = CodeSnippet(
                    content=code_content.strip(),
                    language=language,
                    file_path=source if source else None,
                    confidence=0.9,  # High confidence for explicit code blocks
                    metadata={"extraction_method": "regex_pattern", "match_index": idx},
                    extracted_at=datetime.now(),
                )
                snippets.append(snippet)

        return snippets

    @staticmethod
    def parse_entities(content: str, source: str = "") -> list[Entity]:
        """Extract entities from content."""
        entities = []

        # Extract function/method names
        function_pattern = r"\b(?:def|function|func|async def)\s+(\w+)\s*\("
        for _idx, match in enumerate(re.finditer(function_pattern, content)):
            entity = Entity(
                entity_id=f"entity_func_{uuid7().hex[:8]}",
                entity_type="function",
                name=match.group(1),
                source=source,
                confidence=0.85,
                tags=["function", "code"],
                extracted_at=datetime.now(),
            )
            entities.append(entity)

        # Extract class names
        class_pattern = r"\bclass\s+(\w+)(?:\(|\:)"
        for _idx, match in enumerate(re.finditer(class_pattern, content)):
            entity = Entity(
                entity_id=f"entity_class_{uuid7().hex[:8]}",
                entity_type="class",
                name=match.group(1),
                source=source,
                confidence=0.85,
                tags=["class", "code"],
                extracted_at=datetime.now(),
            )
            entities.append(entity)

        # Extract important concepts (capitalized phrases)
        concept_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
        seen_concepts = set()
        for match in re.finditer(concept_pattern, content):
            concept = match.group(1)
            if concept not in seen_concepts and len(concept.split()) <= 4:
                seen_concepts.add(concept)
                entity = Entity(
                    entity_id=f"entity_concept_{uuid7().hex[:8]}",
                    entity_type="concept",
                    name=concept,
                    source=source,
                    confidence=0.7,
                    tags=["concept"],
                    extracted_at=datetime.now(),
                )
                entities.append(entity)

        return entities

    @staticmethod
    def identify_relationships(entities: list[Entity]) -> list[SemanticRelationship]:
        """Identify relationships between entities."""
        relationships = []

        # Simple heuristic: functions and classes in same context are related
        functions = [e for e in entities if e.entity_type == "function"]
        classes = [e for e in entities if e.entity_type == "class"]

        for func in functions:
            for cls in classes:
                # If function name suggests it belongs to class
                if cls.name.lower() in func.name.lower():
                    relationship = SemanticRelationship(
                        relationship_id=f"rel_{uuid7().hex[:8]}",
                        source_entity_id=func.entity_id,
                        target_entity_id=cls.entity_id,
                        relationship_type="belongs_to",
                        confidence=0.75,
                        metadata={"inferred": True, "method": "name_similarity"},
                    )
                    relationships.append(relationship)

        return relationships


class ContextOrchestrator:
    """Orchestrator for enriching worker context with parsed metadata.

    This class manages the parsing and enrichment of worker context,
    transforming raw assets into structured, searchable metadata.
    """

    def __init__(self, enable_async: bool = True) -> None:  # noqa: FBT001, FBT002
        """Initialize the context orchestrator.

        Args:
            enable_async: Whether to use async processing for enrichment
        """
        self.enable_async = enable_async
        self.parser = AssetParser()
        self._enrichment_cache: dict[str, WorkerContext] = {}
        logger.info("ContextOrchestrator initialized (async=%s)", enable_async)

    async def enrich_context(
        self,
        context_id: str,
        worker_id: str,
        raw_procedures: list[str] | None = None,
        raw_requirements: list[str] | None = None,
        raw_principles: list[str] | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkerContext:
        """Enrich context with parsed metadata from raw assets.

        Args:
            context_id: Unique context identifier
            worker_id: Worker ID using this context
            raw_procedures: Raw procedure content
            raw_requirements: Raw requirement content
            raw_principles: Raw principle content
            session_id: Optional session ID
            metadata: Additional metadata

        Returns:
            Enriched WorkerContext with parsed data
        """
        start_time = time.time()
        metrics = ContextEnrichmentMetrics(enrichment_start_time=datetime.now())

        logger.info(
            "Starting context enrichment for context_id=%s, worker_id=%s",
            context_id,
            worker_id,
        )

        # Sanitize raw inputs to be robust to invalid types (e.g., None)
        def _sanitize_list(name: str, value: list[Any] | None) -> tuple[list[str], list[str]]:
            cleaned: list[str] = []
            errs: list[str] = []
            if not value:
                return cleaned, errs
            for idx, v in enumerate(value):
                if isinstance(v, str):
                    cleaned.append(v)
                elif v is None:
                    errs.append(f"{name}[{idx}] is None; skipped")
                else:
                    try:
                        cleaned.append(str(v))
                        errs.append(f"{name}[{idx}] coerced to str")
                    except Exception:
                        errs.append(f"{name}[{idx}] invalid type {type(v).__name__}; skipped")
            return cleaned, errs

        proc_list, proc_errs = _sanitize_list("raw_procedures", raw_procedures)
        req_list, req_errs = _sanitize_list("raw_requirements", raw_requirements)
        prin_list, prin_errs = _sanitize_list("raw_principles", raw_principles)

        # Initialize context with sanitized raw data (backward compatible)
        context = WorkerContext(
            context_id=context_id,
            worker_id=worker_id,
            session_id=session_id,
            raw_procedures=proc_list,
            raw_requirements=req_list,
            raw_principles=prin_list,
            metadata=metadata or {},
        )

        try:
            # Record any input sanitation errors
            metrics.parsing_errors.extend(proc_errs + req_errs + prin_errs)

            # Enrich with parsed procedures
            if raw_procedures:
                parsed_procs, proc_errors = await self.enrich_with_procedures(raw_procedures)
                context.parsed_procedures = parsed_procs
                metrics.procedures_parsed = len(parsed_procs)
                metrics.parsing_errors.extend(proc_errors)

            # Enrich with parsed requirements
            if raw_requirements:
                parsed_reqs, req_errors = await self.enrich_with_requirements(raw_requirements)
                context.parsed_requirements = parsed_reqs
                metrics.requirements_parsed = len(parsed_reqs)
                metrics.parsing_errors.extend(req_errors)

            # Enrich with parsed principles
            if raw_principles:
                parsed_prins, prin_errors = await self.enrich_with_principles(raw_principles)
                context.parsed_principles = parsed_prins
                metrics.principles_parsed = len(parsed_prins)
                metrics.parsing_errors.extend(prin_errors)

            # Enrich with extracted metadata (code, entities, relationships)
            await self.enrich_with_metadata(context)
            metrics.code_snippets_extracted = len(context.extracted_code)
            metrics.entities_extracted = len(context.semantic_entities)
            metrics.relationships_identified = len(context.semantic_relationships)

            # Index terms for fast search
            context.index_terms()

            # Calculate enrichment metrics
            end_time = time.time()
            metrics.enrichment_end_time = datetime.now()
            metrics.enrichment_duration_ms = (end_time - start_time) * 1000

            context.enrichment_metrics = metrics

            logger.info(
                "Context enrichment completed in %.2fms: %d procedures, %d requirements, %d principles, %d code snippets, %d entities",
                metrics.enrichment_duration_ms,
                metrics.procedures_parsed,
                metrics.requirements_parsed,
                metrics.principles_parsed,
                metrics.code_snippets_extracted,
                metrics.entities_extracted,
            )

            # Cache the enriched context
            self._enrichment_cache[context_id] = context

            return context

        except Exception as e:
            logger.error("Context enrichment failed: %s", str(e), exc_info=True)
            metrics.parsing_errors.append(f"Critical error: {e!s}")
            metrics.enrichment_end_time = datetime.now()
            metrics.enrichment_duration_ms = (time.time() - start_time) * 1000
            context.enrichment_metrics = metrics
            return context

    async def enrich_with_procedures(
        self, raw_procedures: list[str]
    ) -> tuple[list[ParsedProcedure], list[str]]:
        """Parse and enrich procedures with structured metadata.

        Args:
            raw_procedures: Raw procedure content

        Returns:
            Tuple of (parsed procedures, error messages)
        """
        parsed_procedures = []
        errors = []

        for idx, raw_content in enumerate(raw_procedures):
            try:
                # Extract steps (lines starting with numbers or bullets)
                step_pattern = r"(?:^|\n)\s*(?:\d+\.|\-|\*)\s*(.+)"
                steps = [m.group(1).strip() for m in re.finditer(step_pattern, raw_content)]

                # Extract code snippets
                code_snippets = self.parser.parse_code_snippets(
                    raw_content, source=f"procedure_{idx}"
                )

                # Extract entities
                entities = self.parser.parse_entities(raw_content, source=f"procedure_{idx}")

                # Create parsed procedure
                procedure = ParsedProcedure(
                    procedure_id=f"proc_{uuid7().hex[:12]}",
                    title=f"Procedure {idx + 1}",
                    description=(
                        raw_content[:200] + "..." if len(raw_content) > 200 else raw_content
                    ),
                    steps=steps,
                    code_snippets=code_snippets,
                    entities=entities,
                    confidence=0.9 if steps else 0.7,
                    source_file=None,
                    parsed_at=datetime.now(),
                )
                parsed_procedures.append(procedure)

            except ValidationError as e:
                error_msg = f"Validation error parsing procedure {idx}: {e!s}"
                logger.warning(error_msg)
                errors.append(error_msg)
            except Exception as e:
                error_msg = f"Error parsing procedure {idx}: {e!s}"
                logger.exception(error_msg)
                errors.append(error_msg)

        return parsed_procedures, errors

    async def enrich_with_requirements(
        self, raw_requirements: list[str]
    ) -> tuple[list[ParsedRequirement], list[str]]:
        """Parse and enrich requirements with structured metadata.

        Args:
            raw_requirements: Raw requirement content

        Returns:
            Tuple of (parsed requirements, error messages)
        """
        parsed_requirements = []
        errors = []

        for idx, raw_content in enumerate(raw_requirements):
            try:
                # Extract requirement type from content
                req_type = "functional"
                if any(
                    keyword in raw_content.lower()
                    for keyword in ["performance", "scalability", "security"]
                ):
                    req_type = "non-functional"
                elif any(
                    keyword in raw_content.lower()
                    for keyword in ["must", "constraint", "limitation"]
                ):
                    req_type = "constraint"

                # Extract priority
                priority = "medium"
                if any(
                    keyword in raw_content.lower()
                    for keyword in ["critical", "high priority", "urgent"]
                ):
                    priority = "high"
                elif any(
                    keyword in raw_content.lower()
                    for keyword in ["low priority", "nice to have", "optional"]
                ):
                    priority = "low"

                # Extract acceptance criteria
                criteria_pattern = (
                    r"(?:acceptance criteria|criteria|requirements?):\s*(.+?)(?:\n\n|\Z)"
                )
                criteria_match = re.search(criteria_pattern, raw_content, re.IGNORECASE | re.DOTALL)
                acceptance_criteria = []
                if criteria_match:
                    criteria_text = criteria_match.group(1)
                    acceptance_criteria = [
                        c.strip() for c in re.split(r"[\n\-\*]", criteria_text) if c.strip()
                    ]

                # Extract code snippets
                code_snippets = self.parser.parse_code_snippets(
                    raw_content, source=f"requirement_{idx}"
                )

                # Extract entities
                entities = self.parser.parse_entities(raw_content, source=f"requirement_{idx}")

                # Create parsed requirement
                requirement = ParsedRequirement(
                    requirement_id=f"req_{uuid7().hex[:12]}",
                    title=f"Requirement {idx + 1}",
                    description=(
                        raw_content[:500] + "..." if len(raw_content) > 500 else raw_content
                    ),
                    requirement_type=req_type,
                    priority=priority,
                    acceptance_criteria=acceptance_criteria,
                    code_snippets=code_snippets,
                    entities=entities,
                    confidence=0.85,
                    source_file=None,
                    parsed_at=datetime.now(),
                )
                parsed_requirements.append(requirement)

            except ValidationError as e:
                error_msg = f"Validation error parsing requirement {idx}: {e!s}"
                logger.warning(error_msg)
                errors.append(error_msg)
            except Exception as e:
                error_msg = f"Error parsing requirement {idx}: {e!s}"
                logger.exception(error_msg)
                errors.append(error_msg)

        return parsed_requirements, errors

    async def enrich_with_principles(
        self, raw_principles: list[str]
    ) -> tuple[list[ParsedPrinciple], list[str]]:
        """Parse and enrich principles with structured metadata.

        Args:
            raw_principles: Raw principle content

        Returns:
            Tuple of (parsed principles, error messages)
        """
        parsed_principles = []
        errors = []

        for idx, raw_content in enumerate(raw_principles):
            try:
                # Extract category
                category = "general"
                if any(
                    keyword in raw_content.lower() for keyword in ["design pattern", "architecture"]
                ):
                    category = "architecture"
                elif any(
                    keyword in raw_content.lower()
                    for keyword in ["code quality", "coding standard"]
                ):
                    category = "coding"
                elif any(keyword in raw_content.lower() for keyword in ["test", "testing"]):
                    category = "testing"

                # Extract examples
                example_pattern = r"(?:example|for example|e\.g\.):\s*(.+?)(?:\n\n|\Z)"
                example_matches = re.finditer(
                    example_pattern, raw_content, re.IGNORECASE | re.DOTALL
                )
                examples = [m.group(1).strip() for m in example_matches]

                # Extract code snippets
                code_snippets = self.parser.parse_code_snippets(
                    raw_content, source=f"principle_{idx}"
                )

                # Extract entities
                entities = self.parser.parse_entities(raw_content, source=f"principle_{idx}")

                # Create parsed principle
                principle = ParsedPrinciple(
                    principle_id=f"prin_{uuid7().hex[:12]}",
                    title=f"Principle {idx + 1}",
                    description=(
                        raw_content[:500] + "..." if len(raw_content) > 500 else raw_content
                    ),
                    category=category,
                    examples=examples,
                    code_snippets=code_snippets,
                    entities=entities,
                    confidence=0.8,
                    source_file=None,
                    parsed_at=datetime.now(),
                )
                parsed_principles.append(principle)

            except ValidationError as e:
                error_msg = f"Validation error parsing principle {idx}: {e!s}"
                logger.warning(error_msg)
                errors.append(error_msg)
            except Exception as e:
                error_msg = f"Error parsing principle {idx}: {e!s}"
                logger.exception(error_msg)
                errors.append(error_msg)

        return parsed_principles, errors

    async def enrich_with_metadata(self, context: WorkerContext) -> None:
        """Enrich context with extracted metadata (code, entities, relationships).

        This method aggregates all code snippets and entities from parsed assets
        and identifies semantic relationships between them.

        Args:
            context: WorkerContext to enrich
        """
        try:
            # Aggregate all code snippets
            all_code_snippets = context.get_all_code_snippets()
            context.extracted_code = all_code_snippets

            # Aggregate all entities
            all_entities = context.get_all_entities()
            context.semantic_entities = all_entities

            # Identify relationships between entities
            relationships = self.parser.identify_relationships(all_entities)
            context.semantic_relationships = relationships

            # Add confidence scores for decisions
            if context.parsed_procedures:
                context.add_decision_confidence(
                    "procedure_completeness",
                    sum(p.confidence for p in context.parsed_procedures)
                    / len(context.parsed_procedures),
                )

            if context.parsed_requirements:
                context.add_decision_confidence(
                    "requirement_clarity",
                    sum(r.confidence for r in context.parsed_requirements)
                    / len(context.parsed_requirements),
                )

            if all_entities:
                context.add_decision_confidence(
                    "entity_extraction",
                    sum(e.confidence for e in all_entities) / len(all_entities),
                )

            logger.debug(
                "Metadata enrichment completed: %d code snippets, %d entities, %d relationships",
                len(all_code_snippets),
                len(all_entities),
                len(relationships),
            )

        except Exception as e:
            logger.error("Error during metadata enrichment: %s", str(e), exc_info=True)

    def get_cached_context(self, context_id: str) -> WorkerContext | None:
        """Retrieve cached enriched context.

        Args:
            context_id: Context identifier

        Returns:
            Cached WorkerContext or None if not found
        """
        return self._enrichment_cache.get(context_id)

    def clear_cache(self) -> None:
        """Clear the enrichment cache."""
        self._enrichment_cache.clear()
        logger.info("Enrichment cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return {
            "cached_contexts": len(self._enrichment_cache),
            "context_ids": list(self._enrichment_cache.keys()),
        }
