"""Unit tests for db/db_config.py."""

import os
from unittest.mock import AsyncMock, patch

import pytest
from psycopg.types.json import Jsonb

import src.db.db_config as db_config_module


class TestGetPgPool:
    def test_returns_none_when_not_initialized(self):
        original = db_config_module._pg_pool
        db_config_module._pg_pool = None
        assert db_config_module.get_pg_pool() is None
        db_config_module._pg_pool = original


class TestClosePgPool:
    @pytest.mark.asyncio
    async def test_close_handles_exception(self):
        original = db_config_module._pg_pool
        mock_pool = AsyncMock()
        mock_pool.close = AsyncMock(side_effect=Exception("close error"))
        db_config_module._pg_pool = mock_pool
        await db_config_module.close_pg_pool()  # Should not raise
        assert db_config_module._pg_pool is None
        db_config_module._pg_pool = original


class TestTestPgConnection:
    @pytest.mark.asyncio
    async def test_failure(self):
        original = db_config_module._pg_pool
        db_config_module._pg_pool = None

        result = await db_config_module.test_pg_connection()
        assert result is False
        db_config_module._pg_pool = original


class TestGetPostgresConfig:
    def test_success(self):
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://user:pass@host:5432/mydb",
            },
        ):
            config = db_config_module.get_postgres_config()
            assert config["host"] == "host"
            assert config["port"] == 5432
            assert config["database"] == "mydb"
            assert config["user"] == "user"

    def test_missing_database_url(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DATABASE_URL", None)
            with pytest.raises(ValueError, match="DATABASE_URL"):
                db_config_module.get_postgres_config()

    def test_defaults(self):
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://localhost",
            },
        ):
            config = db_config_module.get_postgres_config()
            assert config["host"] == "localhost"
            assert config["port"] == 5432


class TestInitPgPool:
    @pytest.mark.asyncio
    async def test_missing_database_url(self):
        original = db_config_module._pg_pool
        db_config_module._pg_pool = None

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DATABASE_URL", None)
            with pytest.raises(ValueError, match="DATABASE_URL"):
                await db_config_module.init_pg_pool()

        db_config_module._pg_pool = original


class TestTranslatePlaceholders:
    """``$N`` -> ``%(pN)s`` rewrite, preserving asyncpg numbering semantics."""

    def _translate(self, sql):
        return db_config_module._PsycopgConnectionWrapper._translate_placeholders(sql)

    def test_simple_sequential(self):
        assert self._translate("SELECT * FROM t WHERE a = $1 AND b = $2") == (
            "SELECT * FROM t WHERE a = %(p1)s AND b = %(p2)s"
        )

    def test_out_of_order_and_reused(self):
        # $3 appears before $1 and is reused; each occurrence binds the same arg.
        assert self._translate("SET x = $3 WHERE y = $1 AND z <= $3") == (
            "SET x = %(p3)s WHERE y = %(p1)s AND z <= %(p3)s"
        )

    def test_no_placeholders(self):
        assert self._translate("SELECT 1") == "SELECT 1"


class TestArrayParamIndices:
    """Detection of placeholders explicitly cast to a PG array type."""

    def _indices(self, sql):
        return db_config_module._PsycopgConnectionWrapper._array_param_indices(sql)

    def test_detects_any_text_array(self):
        assert self._indices("WHERE col = ANY($1::text[])") == {1}

    def test_detects_multiple_and_various_types(self):
        sql = "SELECT $1::int[], $2::uuid[] WHERE x = ANY($3::text[])"
        assert self._indices(sql) == {1, 2, 3}

    def test_tolerates_whitespace_in_cast(self):
        assert self._indices("WHERE c = ANY($2 :: text [ ])") == {2}

    def test_ignores_non_array_casts(self):
        # A scalar ::text or ::jsonb cast is NOT an array param.
        assert self._indices("SELECT $1::text, $2::jsonb") == set()

    def test_no_casts(self):
        assert self._indices("INSERT INTO t (a, b) VALUES ($1, $2)") == set()


class TestAdaptArgs:
    """Positional args -> ``%(pN)s`` dict, with Jsonb/native-array disambiguation."""

    def _adapt(self, sql, *args):
        return db_config_module._PsycopgConnectionWrapper._adapt_args(sql, args)

    def test_empty_args_returns_none(self):
        assert self._adapt("SELECT 1") is None

    def test_scalars_pass_through(self):
        result = self._adapt("WHERE a = $1 AND b = $2", "x", 7)
        assert result == {"p1": "x", "p2": 7}

    def test_dict_is_jsonb_wrapped(self):
        result = self._adapt("INSERT INTO t (v) VALUES ($1)", {"k": "v"})
        assert isinstance(result["p1"], Jsonb)

    def test_uncast_list_is_jsonb_wrapped(self):
        # JSONB column value such as graphs.nodes = $1 (a bare []).
        result = self._adapt("INSERT INTO graphs (nodes) VALUES ($1)", [])
        assert isinstance(result["p1"], Jsonb)

    def test_array_cast_list_is_native(self):
        # = ANY($1::text[]) must receive a native list, not Jsonb (the bug).
        result = self._adapt("WHERE c = ANY($1::text[])", ["a", "b"])
        assert result["p1"] == ["a", "b"]
        assert not isinstance(result["p1"], Jsonb)

    def test_mixed_jsonb_and_array_in_one_statement(self):
        # $1 jsonb column value (list), $2 native text[] param.
        sql = "INSERT INTO t (meta) SELECT $1 WHERE k = ANY($2::text[])"
        result = self._adapt(sql, ["json", "list"], ["arr", "elems"])
        assert isinstance(result["p1"], Jsonb)
        assert result["p2"] == ["arr", "elems"]
        assert not isinstance(result["p2"], Jsonb)
