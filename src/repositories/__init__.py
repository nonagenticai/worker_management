"""Repository helpers — visibility / scoping primitives."""

from src.repositories.visibility import can_mutate, scope_clause

__all__ = ["can_mutate", "scope_clause"]
