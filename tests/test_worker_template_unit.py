"""Unit tests for worker/worker_template.py."""

import pytest

from src.worker.worker_template import WorkerRunResult, WorkerTemplate


class TestWorkerRunResult:
    def test_success_result(self):
        r = WorkerRunResult(
            success=True,
            data={"key": "value"},
            message="Done",
            execution_time=1.5,
            tools_used=["tool1"],
        )
        assert r.success is True
        assert r.error is None

    def test_failure_result(self):
        r = WorkerRunResult(
            success=False,
            data=None,
            message="Failed",
            error="Something broke",
        )
        assert r.success is False
        assert r.error == "Something broke"


class TestWorkerTemplate:
    @pytest.fixture
    def worker(self):
        return WorkerTemplate(
            name="test_worker",
            description="A test worker",
            tools=[],
            db_path="/tmp/test.db",
        )

    def test_creation(self, worker):
        assert worker.name == "test_worker"
        assert worker.retries == 3

    def test_parse_task_json(self, worker):
        result = worker._parse_task('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_task_json_array(self, worker):
        result = worker._parse_task("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_parse_task_plain_text(self, worker):
        result = worker._parse_task("do something")
        assert result == {"task": "do something", "type": "text"}

    def test_parse_task_invalid_json(self, worker):
        result = worker._parse_task("{invalid json")
        assert result["type"] == "text"

    @pytest.mark.asyncio
    async def test_run_no_tools(self, worker):
        result = await worker.run("process this")
        assert result.success is True
        assert result.execution_time is not None

    @pytest.mark.asyncio
    async def test_run_with_callable_tool(self):
        def my_tool(task):
            return f"processed: {task}"

        worker = WorkerTemplate(
            name="w",
            description="d",
            tools=[my_tool],
            db_path="/tmp/t.db",
        )
        result = await worker.run("hello")
        assert result.success is True
        assert "processed:" in str(result.data)

    @pytest.mark.asyncio
    async def test_run_with_async_callable_tool(self):
        async def my_async_tool(task):
            return f"async: {task}"

        worker = WorkerTemplate(
            name="w",
            description="d",
            tools=[my_async_tool],
            db_path="/tmp/t.db",
        )
        result = await worker.run("hello")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_with_tool_having_run_method(self):
        class MyTool:
            def run(self, task):
                return f"tool: {task}"

        worker = WorkerTemplate(
            name="w",
            description="d",
            tools=[MyTool()],
            db_path="/tmp/t.db",
        )
        result = await worker.run("hello")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_with_async_run_method(self):
        class MyAsyncTool:
            async def run(self, task):
                return f"async_tool: {task}"

        worker = WorkerTemplate(
            name="w",
            description="d",
            tools=[MyAsyncTool()],
            db_path="/tmp/t.db",
        )
        result = await worker.run("hello")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_failure(self):
        def bad_tool(task):
            raise RuntimeError("tool broke")

        worker = WorkerTemplate(
            name="w",
            description="d",
            tools=[bad_tool],
            db_path="/tmp/t.db",
        )
        result = await worker.run("hello")
        assert result.success is False
        assert result.error is not None


class TestSelectBestTool:
    def test_no_tools_raises(self):
        worker = WorkerTemplate(
            name="w",
            description="d",
            tools=[],
            db_path="/tmp/t.db",
        )
        with pytest.raises(ValueError, match="No tools available"):
            worker._select_best_tool({"task": "test", "type": "text"})

    def test_selects_matching_tool(self):
        class FileTool:
            """A file processing tool for reading files."""

        class ApiTool:
            """An API tool for HTTP requests."""

        worker = WorkerTemplate(
            name="w",
            description="d",
            tools=[FileTool(), ApiTool()],
            db_path="/tmp/t.db",
        )
        result = worker._select_best_tool({"task": "read the file data", "type": "text"})
        # FileTool should score higher for "file" keywords
        assert result.__class__.__name__ == "FileTool"

    def test_falls_back_to_first_tool(self):
        class ToolA:
            pass

        class ToolB:
            pass

        worker = WorkerTemplate(
            name="w",
            description="d",
            tools=[ToolA(), ToolB()],
            db_path="/tmp/t.db",
        )
        # Generic task shouldn't strongly prefer either
        result = worker._select_best_tool({"task": "xyz", "type": "text"})
        assert result is not None


class TestExecuteSimpleTask:
    @pytest.mark.asyncio
    async def test_returns_dict(self):
        worker = WorkerTemplate(
            name="simple",
            description="d",
            tools=[],
            db_path="/tmp/t.db",
        )
        result = await worker._execute_simple_task({"task": "hello", "type": "text"})
        assert result["worker_name"] == "simple"
        assert "hello" in result["processed_task"]
        assert result["processing_type"] == "simple_worker"
