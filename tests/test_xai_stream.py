"""WM#337: the xAI SDK's stream span must attach and detach in ONE Context.

``xai_sdk``'s ``Chat.stream()`` holds an OTel span open across its yields.
pydantic_ai takes the first chunk inside a task it creates and the caller
drains the rest in its own task, so the span's ``context.detach`` ran in a
different Context and OTel logged ``ERROR opentelemetry.context Failed to
detach context`` on every successful xAI chat.

The first test is the CONTROL: it reproduces that error with a bare generator
under the same task split, so the harness can see the defect. The rest show
:class:`SingleTaskStream` removes it without changing what is streamed.
"""

from __future__ import annotations

import asyncio
import inspect
import logging

import pytest
from opentelemetry.sdk.trace import TracerProvider
from pydantic_ai.models.xai import XaiModel

from src.worker.agents.xai_stream import PinnedContextXaiModel, SingleTaskStream, _ChatProxy


@pytest.fixture(autouse=True)
def _sdk_tracing_enabled(monkeypatch):
    """Many test modules set ``OTEL_SDK_DISABLED=true`` at import, and the SDK
    ``TracerProvider`` reads it at construction: its tracers are then no-ops, no
    span is attached, and every "no detach error" assertion here passes
    VACUOUSLY. Measured: in a full run the CONTROL failed for exactly this
    reason. Clear it, and build the provider per test, after the clear."""
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)


def _tracer():
    return TracerProvider().get_tracer("wm337-test")


async def _span_across_yields(n: int = 3):
    """The xai_sdk shape: a span held open across every yield."""
    with _tracer().start_as_current_span("chat.stream grok"):
        for i in range(n):
            await asyncio.sleep(0)
            yield i


async def _first_in_task_then_drain(stream) -> list[int]:
    """pydantic_ai's split: first anext in a created task, the rest in the caller."""
    first = await asyncio.create_task(stream.__anext__())
    rest = [item async for item in stream]
    return [first, *rest]


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def caplog():
    """Capture ``opentelemetry.context`` with a handler on that logger itself.

    Defensive: other tests reconfigure logging (``dictConfig``), and a handler on
    the logger, with the logger forced enabled, does not depend on propagation
    to pytest's root capture.
    """
    log = logging.getLogger("opentelemetry.context")
    saved = (log.disabled, log.level)
    log.disabled = False
    log.setLevel(logging.ERROR)
    handler = _Records()
    log.addHandler(handler)
    try:
        yield handler
    finally:
        log.removeHandler(handler)
        log.disabled, log.level = saved


def _detach_errors(capture: _Records) -> list[logging.LogRecord]:
    return [r for r in capture.records if "Failed to detach context" in r.getMessage()]


@pytest.mark.asyncio
async def test_control_bare_generator_reproduces_the_detach_error(caplog):

    items = await _first_in_task_then_drain(_span_across_yields())

    assert items == [0, 1, 2]
    assert _detach_errors(caplog), "harness must reproduce WM#337, or the next test proves nothing"


@pytest.mark.asyncio
async def test_single_task_stream_attaches_and_detaches_in_one_context(caplog):

    items = await _first_in_task_then_drain(SingleTaskStream(_span_across_yields))

    assert items == [0, 1, 2]
    assert _detach_errors(caplog) == []


@pytest.mark.asyncio
async def test_early_aclose_closes_the_generator_in_its_own_task(caplog):
    closed = asyncio.Event()

    async def _gen():
        try:
            async for i in _span_across_yields(100):
                yield i
        finally:
            closed.set()

    stream = SingleTaskStream(_gen)
    assert await asyncio.create_task(stream.__anext__()) == 0
    await stream.aclose()

    assert closed.is_set()
    assert _detach_errors(caplog) == []
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


@pytest.mark.asyncio
async def test_an_upstream_error_reaches_the_consumer():
    async def _boom():
        yield 1
        raise RuntimeError("grpc stream failed")

    stream = SingleTaskStream(_boom)
    assert await stream.__anext__() == 1
    with pytest.raises(RuntimeError, match="grpc stream failed"):
        await stream.__anext__()


@pytest.mark.asyncio
async def test_pinned_model_wraps_the_chat_stream(monkeypatch, caplog):

    class _FakeChat:
        proto = "kept"

        def stream(self):
            return _span_across_yields()

    async def _fake_create_chat(self, *args, **kwargs):
        return _FakeChat()

    monkeypatch.setattr(XaiModel, "_create_chat", _fake_create_chat)
    model = PinnedContextXaiModel.__new__(PinnedContextXaiModel)

    chat = await model._create_chat([], {}, None)

    assert isinstance(chat, _ChatProxy)
    assert chat.proto == "kept", "everything but stream() must delegate"
    stream = chat.stream()
    assert isinstance(stream, SingleTaskStream)
    assert await _first_in_task_then_drain(stream) == [0, 1, 2]
    assert _detach_errors(caplog) == []


def test_the_private_hook_still_exists_and_feeds_the_stream():
    """``_create_chat`` is private pydantic_ai API. If an upgrade renames it or
    stops streaming through it, the override silently stops applying -- fail here."""
    assert inspect.iscoroutinefunction(XaiModel._create_chat)
    assert "self._create_chat(" in inspect.getsource(XaiModel.request_stream)
    assert "chat.stream()" in inspect.getsource(XaiModel.request_stream)


def test_the_factory_builds_the_pinned_model():
    from src.worker.agents import pydantic_ai_agent

    src = inspect.getsource(pydantic_ai_agent.ModelFactory._create_xai_model)
    assert "PinnedContextXaiModel(" in src


# --- Paths where the pump ends WITHOUT delivering another item. A stream that
# waits on its queue alone hangs on every one of these; each is bounded by a
# timeout so a hang fails the test instead of the suite.

_HANG = 2.0


@pytest.mark.asyncio
async def test_an_upstream_cancelled_error_reaches_the_consumer_not_a_hang():
    """grpc.aio raises CancelledError when a call is cancelled locally."""

    async def _grpc_cancelled():
        yield 1
        raise asyncio.CancelledError

    stream = SingleTaskStream(_grpc_cancelled)
    assert await asyncio.wait_for(stream.__anext__(), _HANG) == 1
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(stream.__anext__(), _HANG)


@pytest.mark.asyncio
async def test_aclose_while_a_consumer_waits_ends_that_consumer():
    """pydantic_ai's StreamedResponse.cancel() -> close_stream() -> aclose()
    can race a consumer parked in __anext__."""
    gate = asyncio.Event()

    async def _slow():
        yield 1
        await gate.wait()
        yield 2

    stream = SingleTaskStream(_slow)
    assert await stream.__anext__() == 1
    waiting = asyncio.create_task(stream.__anext__())
    await asyncio.sleep(0.05)
    await stream.aclose()
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(waiting, _HANG)


@pytest.mark.asyncio
async def test_a_cancellation_aimed_at_the_aclose_caller_is_not_swallowed():
    stop = asyncio.Event()

    async def _stubborn():
        try:
            yield 1
            await asyncio.sleep(3600)
        finally:
            await stop.wait()  # closing takes a while

    stream = SingleTaskStream(_stubborn)
    assert await stream.__anext__() == 1
    closer = asyncio.create_task(stream.aclose())
    await asyncio.sleep(0.05)
    closer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(closer, _HANG)
    stop.set()


@pytest.mark.asyncio
async def test_an_error_while_closing_the_generator_is_raised_not_a_hang():
    class _BadClose:
        def __init__(self):
            self._n = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self._n += 1
            if self._n > 1:
                raise StopAsyncIteration
            return 1

        async def aclose(self):
            raise RuntimeError("close failed")

    stream = SingleTaskStream(_BadClose)
    assert await asyncio.wait_for(stream.__anext__(), _HANG) == 1
    with pytest.raises(RuntimeError, match="close failed"):
        await asyncio.wait_for(stream.__anext__(), _HANG)
