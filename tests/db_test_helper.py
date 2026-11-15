"""In-memory database mock for unit testing DB modules.

Provides a fake psycopg-compatible connection that stores data in Python dicts,
allowing tests to verify real data flow (insert then read back) without
hitting a real PostgreSQL server.
"""

from __future__ import annotations

import copy
import re
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch


class _FakeRecord(dict):
    """Dict that also supports attribute and .get() access like psycopg row dicts.

    Missing keys return None (simulating SQL NULL for columns with no explicit value).
    """

    def __getitem__(self, key: str) -> Any:
        try:
            return super().__getitem__(key)
        except KeyError:
            return None

    def __getattr__(self, key: str) -> Any:
        return self[key]


class _FakeTransaction:
    """Async context manager that does nothing (no real transaction)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: object):
        return False


class InMemoryDB:
    """Fake psycopg-compatible connection backed by in-memory dicts.

    Tables are keyed by name.  Each table is a list of _FakeRecord dicts.
    An auto-incrementing ``_sequences`` dict tracks RETURNING id values.
    """

    def __init__(self) -> None:
        self.tables: dict[str, list[_FakeRecord]] = {}
        self._sequences: dict[str, int] = {}

    # -- helpers ---------------------------------------------------------------

    def _table(self, name: str) -> list[_FakeRecord]:
        if name not in self.tables:
            self.tables[name] = []
        return self.tables[name]

    def _next_id(self, table: str) -> int:
        self._sequences.setdefault(table, 0)
        self._sequences[table] += 1
        return self._sequences[table]

    @staticmethod
    def _extract_table(sql: str) -> str:
        """Pull the table name from INSERT/UPDATE/DELETE/SELECT SQL."""
        sql_norm = " ".join(sql.split())  # collapse whitespace

        # Use regex anchored to statement keywords (not substring matches)
        m = re.match(r"\s*INSERT\s+INTO\s+(\w+)", sql_norm, re.IGNORECASE)
        if m:
            return m.group(1)

        m = re.match(r"\s*UPDATE\s+(\w+)", sql_norm, re.IGNORECASE)
        if m:
            return m.group(1)

        m = re.match(r"\s*DELETE\s+FROM\s+(\w+)", sql_norm, re.IGNORECASE)
        if m:
            return m.group(1)

        # SELECT ... FROM table
        m = re.search(r"\bFROM\s+(\w+)", sql_norm, re.IGNORECASE)
        if m:
            return m.group(1)

        return "unknown"

    @staticmethod
    def _extract_columns(sql: str) -> list[str]:
        """Extract column names from an INSERT ... (col1, col2, ...) VALUES ..."""
        m = re.search(r"INSERT\s+INTO\s+\w+\s*\(([^)]+)\)", sql, re.IGNORECASE)
        if not m:
            return []
        return [c.strip() for c in m.group(1).split(",")]

    # A SET assignment this fake can represent, in one of four forms:
    #   col = $N                                       -> plain assign
    #   col = COALESCE($N::type, col)                  -> assign unless the arg is None
    #   col = COALESCE(col, '{}'::jsonb) || $N::jsonb  -> shallow dict merge
    #   col = jsonb_set(col, '{key}', to_jsonb($N::t)) -> set one top-level key
    _SET_PLAIN_RE = re.compile(r"^(\w+)\s*=\s*\$(\d+)(?:::[\w\[\] ]+)?$")
    _SET_COALESCE_RE = re.compile(
        r"^(\w+)\s*=\s*COALESCE\(\s*\$(\d+)(?:::[\w ]+)?\s*,\s*\1\s*\)$", re.I
    )
    _SET_MERGE_RE = re.compile(
        r"^(\w+)\s*=\s*COALESCE\(\s*\1\s*,\s*'\{\}'::jsonb\s*\)\s*\|\|\s*\$(\d+)(?:::jsonb)?$",
        re.IGNORECASE,
    )
    # ``swap_prompt`` (worker_registry_db.py) is the one live user of this shape.
    _SET_JSONB_SET_RE = re.compile(
        r"^(\w+)\s*=\s*jsonb_set\(\s*(?:COALESCE\(\s*\1\s*,\s*'\{\}'::jsonb\s*\)|\1)\s*,"
        r"\s*'\{(\w+)\}'\s*,\s*to_jsonb\(\s*\$(\d+)(?:::[\w ]+)?\s*\)\s*\)$",
        re.IGNORECASE,
    )

    @classmethod
    def _extract_set_assignments(cls, sql: str) -> list[tuple[str, str, int]]:
        """Parse an UPDATE's SET clause into ``(column, kind, param_index)``.

        Raises on an assignment it cannot represent, deliberately. The previous
        implementation scanned for a bare ``col = $N`` pattern and silently
        skipped anything else, so when ``register_worker`` moved its config
        merge into the statement (``config = COALESCE(config, '{}'::jsonb) ||
        $4::jsonb``) the fake applied *some* of the SET clause and dropped the
        rest — the row kept its old config and the test failed pointing at the
        production code rather than at the fake. Failing loudly means a future
        SQL change cannot quietly turn these tests into no-ops.

        This models assignment shape only. Anything whose subject is what
        Postgres actually does with a statement — jsonb operator semantics,
        transaction and locking behaviour, row shape — belongs in a ``-m pg``
        test against a real server, not here.

        Measured 2026-08-28 by running every ``UPDATE ... SET`` literal in the
        repo through this parser: **8 statements do not parse** — the arithmetic
        + ``GREATEST(...)`` forms in ``src/db/skills_db.py``, and the
        column-to-column / ``= NULL`` / ``make_interval(...)`` forms in
        ``src/api/workflow_routing/repository.py``. That is a known boundary,
        not an oversight: none of those modules is driven through ``InMemoryDB``
        today, and the four supported shapes are the ones that are. If you are
        the first to need one of them, add it here only if its meaning is
        unambiguous — otherwise the assertion you are reaching for is about
        Postgres, and belongs in ``tests/integration`` under ``-m pg``.
        """
        set_match = re.search(r"SET\s+(.+?)\s+WHERE", sql, re.IGNORECASE | re.DOTALL)
        if not set_match:
            return []

        assignments: list[tuple[str, str, int]] = []
        depth = 0
        current = ""
        parts: list[str] = []
        for ch in set_match.group(1):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(current)
                current = ""
            else:
                current += ch
        parts.append(current)

        for part in parts:
            piece = " ".join(part.split())
            if not piece:
                continue
            m = cls._SET_JSONB_SET_RE.match(piece)
            if m:
                assignments.append((f"{m.group(1)}.{m.group(2)}", "jsonb_key", int(m.group(3))))
                continue
            for kind, pattern in (
                ("assign", cls._SET_PLAIN_RE),
                ("coalesce", cls._SET_COALESCE_RE),
                ("merge", cls._SET_MERGE_RE),
            ):
                m = pattern.match(piece)
                if m:
                    assignments.append((m.group(1), kind, int(m.group(2))))
                    break
            else:
                raise ValueError(
                    f"InMemoryDB cannot represent the SET assignment {piece!r}. "
                    "Add a form here only if its meaning is unambiguous; if the "
                    "assertion is really about Postgres semantics, write it as a "
                    "-m pg test instead."
                )
        return assignments

    @staticmethod
    def _apply_set_assignments(
        row: Any, assignments: list[tuple[str, str, int]], args: tuple[Any, ...]
    ) -> None:
        """Apply parsed SET assignments to one row."""
        for col, kind, idx in assignments:
            value = copy.deepcopy(args[idx - 1])
            if kind == "jsonb_key":
                column, key = col.split(".", 1)
                current = row.get(column) or {}
                row[column] = {**current, key: value}
            elif kind == "coalesce":
                if value is not None:
                    row[col] = value
            elif kind == "merge":
                current = row.get(col) or {}
                row[col] = {**current, **(value or {})}
            else:
                row[col] = value

    @staticmethod
    def _parse_col(raw_col: str) -> str:
        """Normalize a column expression to a canonical key."""
        # LOWER(col) wrapper
        lower_m = re.match(r"LOWER\((\w+)\)", raw_col, re.IGNORECASE)
        if lower_m:
            return lower_m.group(1).lower() + "__lower"
        # JSONB arrow operators (e.g. state_data->>'session_id')
        arrow_m = re.match(r"(\w+)->>?'(\w+)'", raw_col)
        if arrow_m:
            return f"{arrow_m.group(1)}__{arrow_m.group(2)}"
        return raw_col

    @staticmethod
    def _extract_where_clauses(sql: str) -> list:
        """Extract structured WHERE conditions.

        Returns a list of "clause groups".  Each group is either:
        - ("AND", col, param_idx)   — must all match
        - ("OR", [(col, param_idx), ...])  — at least one must match

        This handles: ``col1 = $1 AND (col2 = $2 OR col3 = $3) AND col4 = $4``
        """
        where_match = re.search(
            r"WHERE\s+(.+?)(?:\s+ORDER|\s+LIMIT|\s+RETURNING|\s*$)",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        if not where_match:
            return []
        where_clause = where_match.group(1).strip()

        clauses: list = []

        # Split by top-level AND (not inside parentheses)
        # Only extract paren groups that contain OR (not function calls like LOWER(col))
        paren_groups: dict[str, str] = {}
        counter = 0

        def replace_paren(m: re.Match) -> str:
            nonlocal counter
            inner = m.group(1)
            if re.search(r"\bOR\b", inner, re.IGNORECASE):
                key = f"__PAREN{counter}__"
                paren_groups[key] = inner
                counter += 1
                return key
            # Not an OR group — keep it as-is (e.g. LOWER(col))
            return m.group(0)

        flat = re.sub(r"\(([^)]+)\)", replace_paren, where_clause)

        parts = re.split(r"\s+AND\s+", flat, flags=re.IGNORECASE)

        for part in parts:
            part = part.strip()
            if part in paren_groups:
                # This is an OR group
                or_body = paren_groups[part]
                or_parts = re.split(r"\s+OR\s+", or_body, flags=re.IGNORECASE)
                or_pairs = []
                for op in or_parts:
                    m = re.search(r"([\w.>'\"-]+(?:\(\w+\))?)\s*=\s*\$(\d+)", op)
                    if m:
                        col = InMemoryDB._parse_col(m.group(1))
                        or_pairs.append((col, int(m.group(2))))
                if or_pairs:
                    clauses.append(("OR", or_pairs))
            else:
                m = re.search(r"([\w.>'\"-]+(?:\(\w+\))?)\s*=\s*\$(\d+)", part)
                if m:
                    col = InMemoryDB._parse_col(m.group(1))
                    clauses.append(("AND", col, int(m.group(2))))

        return clauses

    def _match_row(
        self,
        row: _FakeRecord,
        clauses: list,
        params: tuple[Any, ...],
    ) -> bool:
        """Return True if row matches all WHERE conditions."""
        for clause in clauses:
            if clause[0] == "AND":
                _, col, idx = clause
                if not self._match_col(row, col, params[idx - 1]):
                    return False
            elif clause[0] == "OR":
                or_pairs = clause[1]
                if not any(self._match_col(row, col, params[idx - 1]) for col, idx in or_pairs):
                    return False
        return True

    @staticmethod
    def _match_col(row: _FakeRecord, col: str, val: Any) -> bool:
        """Check if a single column matches a value."""
        if "__lower" in col:
            real_col = col.replace("__lower", "")
            row_val = row.get(real_col, None)
            if row_val is None or str(row_val).lower() != str(val).lower():
                return False
        elif "__" in col:
            # JSONB path like state_data__session_id
            parts = col.split("__")
            obj = row.get(parts[0])
            if isinstance(obj, dict):
                if str(obj.get(parts[1])) != str(val):
                    return False
            else:
                return False
        elif row.get(col) != val:
            return False
        return True

    # -- psycopg-compatible methods --------------------------------------------

    async def execute(self, sql: str, *args: Any) -> str:
        """Handle INSERT / UPDATE / DELETE and return status strings."""
        sql_up = sql.upper()
        table_name = self._extract_table(sql)
        rows = self._table(table_name)

        if "INSERT INTO" in sql_up:
            cols = self._extract_columns(sql)
            record = _FakeRecord()
            for i, col in enumerate(cols):
                if i < len(args):
                    record[col] = copy.deepcopy(args[i])
            # Auto-generate id if not in columns
            if "id" not in record:
                record["id"] = self._next_id(table_name)
            rows.append(record)
            return "INSERT 0 1"

        elif "UPDATE" in sql_up:
            where_pairs = self._extract_where_clauses(sql)
            assignments = self._extract_set_assignments(sql)

            count = 0
            for row in rows:
                if self._match_row(row, where_pairs, args):
                    self._apply_set_assignments(row, assignments, args)
                    count += 1
            return f"UPDATE {count}"

        elif "DELETE" in sql_up:
            where_pairs = self._extract_where_clauses(sql)
            to_remove = [r for r in rows if self._match_row(r, where_pairs, args)]
            for r in to_remove:
                rows.remove(r)
            return f"DELETE {len(to_remove)}"

        return "OK 0"

    async def fetchrow(self, sql: str, *args: Any) -> _FakeRecord | None:
        """Handle SELECT ... or INSERT ... RETURNING *."""
        sql_up = sql.upper()
        table_name = self._extract_table(sql)

        if "INSERT" in sql_up and "RETURNING" in sql_up:
            # Do the insert, then return the record
            cols = self._extract_columns(sql)
            record = _FakeRecord()
            for i, col in enumerate(cols):
                if i < len(args):
                    record[col] = copy.deepcopy(args[i])
            if "id" not in record:
                record["id"] = self._next_id(table_name)
            self._table(table_name).append(record)
            return record

        if "UPDATE" in sql_up and "RETURNING" in sql_up:
            # Do the update, then return the matching record
            rows = self._table(table_name)
            where_pairs = self._extract_where_clauses(sql)
            assignments = self._extract_set_assignments(sql)

            for row in rows:
                if self._match_row(row, where_pairs, args):
                    self._apply_set_assignments(row, assignments, args)
                    return row
            return None

        # Plain SELECT
        rows = self._table(table_name)
        where_pairs = self._extract_where_clauses(sql)
        for row in rows:
            if self._match_row(row, where_pairs, args):
                return row
        return None

    async def fetchval(self, sql: str, *args: Any) -> Any:
        """Handle SELECT returning a single scalar, including ``COUNT(*)``."""
        table_name = self._extract_table(sql)
        rows = self._table(table_name)
        where_pairs = self._extract_where_clauses(sql)
        matched = [r for r in rows if self._match_row(r, where_pairs, args)]

        if re.search(r"COUNT\s*\(", sql, re.IGNORECASE):
            return len(matched)

        if not matched:
            return None
        columns = self._extract_columns(sql)
        first = matched[0]
        return first.get(columns[0]) if columns else None

    async def fetch(self, sql: str, *args: Any) -> list[_FakeRecord]:
        """Handle SELECT ... returning multiple rows."""
        table_name = self._extract_table(sql)
        rows = self._table(table_name)
        where_pairs = self._extract_where_clauses(sql)

        matched = [r for r in rows if self._match_row(r, where_pairs, args)]

        # Handle ORDER BY ... DESC
        if re.search(r"ORDER\s+BY\s+\w+\s+DESC", sql, re.IGNORECASE):
            order_m = re.search(r"ORDER\s+BY\s+(\w+)\s+DESC", sql, re.IGNORECASE)
            if order_m:
                col = order_m.group(1)
                matched.sort(key=lambda r: r.get(col, 0) or 0, reverse=True)
        elif re.search(r"ORDER\s+BY\s+\w+\s+ASC", sql, re.IGNORECASE):
            order_m = re.search(r"ORDER\s+BY\s+(\w+)\s+ASC", sql, re.IGNORECASE)
            if order_m:
                col = order_m.group(1)
                matched.sort(key=lambda r: r.get(col, 0) or 0)
        elif re.search(r"ORDER\s+BY\s+(\w+)", sql, re.IGNORECASE):
            order_m = re.search(r"ORDER\s+BY\s+(\w+)", sql, re.IGNORECASE)
            if order_m:
                col = order_m.group(1)
                matched.sort(key=lambda r: r.get(col, "") or "")

        # Handle OFFSET before LIMIT (SQL semantics: OFFSET first, then LIMIT)
        offset_m = re.search(r"OFFSET\s+\$(\d+)", sql, re.IGNORECASE)
        if offset_m:
            offset_idx = int(offset_m.group(1))
            offset_val = args[offset_idx - 1] if offset_idx <= len(args) else 0
            if offset_val:
                matched = matched[offset_val:]

        # Handle LIMIT $N
        limit_m = re.search(r"LIMIT\s+\$(\d+)", sql, re.IGNORECASE)
        if limit_m:
            limit_idx = int(limit_m.group(1))
            limit_val = args[limit_idx - 1] if limit_idx <= len(args) else None
            if limit_val is not None:
                matched = matched[:limit_val]

        return matched

    def transaction(self) -> _FakeTransaction:
        """Return a no-op async context manager."""
        return _FakeTransaction()


@asynccontextmanager
async def mock_pg_connection(mem_db: InMemoryDB):
    """Async context manager that yields an InMemoryDB as if it were a psycopg connection."""
    yield mem_db


def patch_get_pg_connection(module_path: str, mem_db: InMemoryDB):
    """Return a unittest.mock.patch that replaces get_pg_connection in the given module.

    Usage::

        db = InMemoryDB()
        with patch_get_pg_connection("src.db.graph_state_db", db):
            ...
    """
    return patch(f"{module_path}.get_pg_connection", lambda: mock_pg_connection(mem_db))
