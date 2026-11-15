"""WorkerContext model for enriched worker execution context.

This module defines the WorkerContext model which holds both raw and parsed
metadata for worker execution, including procedures, requirements, principles,
code snippets, entities, and semantic relationships.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .parsed_data_models import (
    CodeSnippet,
    Entity,
    ParsedPrinciple,
    ParsedProcedure,
    ParsedRequirement,
    SemanticRelationship,
)


class ContextEnrichmentMetrics(BaseModel):
    """Metrics for tracking context enrichment performance."""

    enrichment_start_time: datetime | None = Field(
        None, description="Start time of enrichment process"
    )
    enrichment_end_time: datetime | None = Field(None, description="End time of enrichment process")
    enrichment_duration_ms: float | None = Field(
        None, description="Duration of enrichment in milliseconds"
    )
    procedures_parsed: int = Field(default=0, description="Number of procedures parsed")
    requirements_parsed: int = Field(default=0, description="Number of requirements parsed")
    principles_parsed: int = Field(default=0, description="Number of principles parsed")
    code_snippets_extracted: int = Field(default=0, description="Number of code snippets extracted")
    entities_extracted: int = Field(default=0, description="Number of entities extracted")
    relationships_identified: int = Field(
        default=0, description="Number of relationships identified"
    )
    parsing_errors: list[str] = Field(
        default_factory=list, description="Errors encountered during parsing"
    )


class WorkerContext(BaseModel):
    """Enriched context for worker execution with parsed metadata.

    This model serves as the central data structure for passing enriched
    context to workers, including both raw content and structured parsed data.
    """

    # Core identification
    context_id: str = Field(..., description="Unique identifier for this context")
    worker_id: str = Field(..., description="ID of the worker using this context")
    session_id: str | None = Field(None, description="Execution session ID")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Context creation timestamp"
    )

    # Raw content (backward compatible)
    raw_procedures: list[str] = Field(
        default_factory=list, description="Raw procedure content as strings"
    )
    raw_requirements: list[str] = Field(
        default_factory=list, description="Raw requirement content as strings"
    )
    raw_principles: list[str] = Field(
        default_factory=list, description="Raw principle content as strings"
    )
    raw_context_data: dict[str, Any] = Field(
        default_factory=dict, description="Additional raw context data"
    )

    # Parsed structured data (new enriched fields)
    parsed_procedures: list[ParsedProcedure] = Field(
        default_factory=list,
        description="Structured parsed procedure data with metadata",
    )
    parsed_requirements: list[ParsedRequirement] = Field(
        default_factory=list,
        description="Structured parsed requirement data with metadata",
    )
    parsed_principles: list[ParsedPrinciple] = Field(
        default_factory=list,
        description="Structured parsed principle data with metadata",
    )
    extracted_code: list[CodeSnippet] = Field(
        default_factory=list,
        description="Code snippets extracted from all assets with metadata",
    )
    semantic_entities: list[Entity] = Field(
        default_factory=list,
        description="Entities extracted from assets with semantic information",
    )
    semantic_relationships: list[SemanticRelationship] = Field(
        default_factory=list, description="Semantic relationships between entities"
    )

    # Decision support
    decision_confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Confidence scores for various decisions (0.0-1.0)",
    )
    recommended_actions: list[str] = Field(
        default_factory=list, description="Recommended actions based on parsed context"
    )

    # Enrichment metadata
    enrichment_metrics: ContextEnrichmentMetrics = Field(
        default_factory=ContextEnrichmentMetrics,
        description="Metrics tracking enrichment performance",
    )
    enrichment_version: str = Field(default="1.0.0", description="Version of enrichment logic used")

    # Search and indexing
    indexed_terms: dict[str, list[str]] = Field(
        default_factory=dict, description="Indexed terms for fast context search"
    )
    tags: list[str] = Field(default_factory=list, description="Tags for categorizing context")

    # Additional metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata")

    def get_all_code_snippets(self) -> list[CodeSnippet]:
        """Get all code snippets from all parsed assets."""
        snippets = list(self.extracted_code)
        for procedure in self.parsed_procedures:
            snippets.extend(procedure.code_snippets)
        for requirement in self.parsed_requirements:
            snippets.extend(requirement.code_snippets)
        for principle in self.parsed_principles:
            snippets.extend(principle.code_snippets)
        return snippets

    def get_all_entities(self) -> list[Entity]:
        """Get all entities from all parsed assets."""
        entities = list(self.semantic_entities)
        for procedure in self.parsed_procedures:
            entities.extend(procedure.entities)
        for requirement in self.parsed_requirements:
            entities.extend(requirement.entities)
        for principle in self.parsed_principles:
            entities.extend(principle.entities)
        return entities

    def get_entities_by_type(self, entity_type: str) -> list[Entity]:
        """Get entities filtered by type."""
        return [e for e in self.get_all_entities() if e.entity_type == entity_type]

    def get_high_confidence_items(self, min_confidence: float = 0.8) -> dict[str, list[Any]]:
        """Get all high-confidence parsed items."""
        return {
            "procedures": [p for p in self.parsed_procedures if p.confidence >= min_confidence],
            "requirements": [r for r in self.parsed_requirements if r.confidence >= min_confidence],
            "principles": [p for p in self.parsed_principles if p.confidence >= min_confidence],
            "code_snippets": [c for c in self.extracted_code if c.confidence >= min_confidence],
            "entities": [e for e in self.semantic_entities if e.confidence >= min_confidence],
        }

    def get_enrichment_summary(self) -> dict[str, Any]:
        """Get a summary of enrichment results."""
        return {
            "context_id": self.context_id,
            "worker_id": self.worker_id,
            "created_at": self.created_at,
            "enrichment_version": self.enrichment_version,
            "parsed_counts": {
                "procedures": len(self.parsed_procedures),
                "requirements": len(self.parsed_requirements),
                "principles": len(self.parsed_principles),
                "code_snippets": len(self.extracted_code),
                "entities": len(self.semantic_entities),
                "relationships": len(self.semantic_relationships),
            },
            "raw_counts": {
                "procedures": len(self.raw_procedures),
                "requirements": len(self.raw_requirements),
                "principles": len(self.raw_principles),
            },
            "metrics": (self.enrichment_metrics.model_dump() if self.enrichment_metrics else {}),
            "tags": self.tags,
        }

    def add_decision_confidence(self, decision_key: str, confidence: float) -> None:
        """Add or update a decision confidence score."""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        self.decision_confidence[decision_key] = confidence

    def index_terms(self) -> None:
        """Index all terms from parsed content for fast search."""
        self.indexed_terms.clear()

        # Index procedure terms
        for proc in self.parsed_procedures:
            terms = proc.title.lower().split() + proc.tags
            self.indexed_terms.setdefault("procedures", []).extend(terms)

        # Index requirement terms
        for req in self.parsed_requirements:
            terms = req.title.lower().split() + req.tags
            self.indexed_terms.setdefault("requirements", []).extend(terms)

        # Index principle terms
        for prin in self.parsed_principles:
            terms = prin.title.lower().split() + prin.tags
            self.indexed_terms.setdefault("principles", []).extend(terms)

        # Index entity terms
        for entity in self.semantic_entities:
            terms = entity.name.lower().split() + entity.tags
            self.indexed_terms.setdefault("entities", []).extend(terms)

    def search(self, query: str) -> list[dict[str, Any]]:
        """Simple search across indexed terms and parsed items.

        Args:
            query: Case-insensitive search string

        Returns:
            List of matches with category and matched term.
        """
        q = (query or "").strip().lower()
        if not q:
            return []

        results: list[dict[str, Any]] = []

        # Ensure we have up-to-date index
        if not self.indexed_terms:
            self.index_terms()

        for category, terms in self.indexed_terms.items():
            for term in terms:
                if q in term:
                    results.append({"category": category, "term": term})

        return results

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "context_id": "ctx_123456",
                "worker_id": "worker_abc",
                "session_id": "session_789",
                "raw_procedures": ["Step 1: Initialize...", "Step 2: Execute..."],
                "parsed_procedures": [],
                "extracted_code": [],
                "semantic_entities": [],
                "decision_confidence": {
                    "procedure_selection": 0.95,
                    "requirement_matching": 0.88,
                },
                "tags": ["backend", "api", "database"],
            }
        }
    )
