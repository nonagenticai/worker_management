"""Run the xAI SDK's chat stream in ONE task, so its OTel span attaches and detaches in one Context (WM#337).

``xai_sdk``'s ``Chat.stream()`` is an async generator that holds
``tracer.start_as_current_span("chat.stream …")`` open ACROSS its yields
(``xai_sdk/aio/chat.py``). pydantic_ai pulls the first chunk inside a task it
creates (``PeekableAsyncStream.peek()`` under ``_agent_graph``'s
``asyncio.create_task``), so the span's context token is attached in THAT
task's copied Context; the caller then drains the rest of the stream in its own
task, and the generator's final ``with`` exit runs ``context.detach`` in a
different Context. OTel logs ``ERROR opentelemetry.context Failed to detach
context`` (ValueError: token created in a different Context) once per
successful xAI chat. The request itself succeeds.

:class:`SingleTaskStream` fixes the cause, not the log: one dedicated task
drives the generator from first ``anext`` to close, pushing items through a
small bounded queue. Attach and detach therefore both happen in that task. The
task is created on the first ``__anext__``, so it copies the caller's Context
and the span keeps its parent. ``aclose()`` cancels and awaits the task, so the
generator's ``finally``/``with`` exits also run there.

:class:`PinnedContextXaiModel` wires it in by wrapping the chat object
``XaiModel._create_chat`` returns. ``_create_chat`` is PRIVATE pydantic_ai API
(verified against pydantic_ai 1.107.0); ``test_xai_stream.py`` fails loudly if
it disappears.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pydantic_ai.models.xai import XaiModel

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

_QUEUE_MAX = 16


class SingleTaskStream:
    """An async iterator that drains ``factory()`` entirely inside one task.

    The consumer never waits on the queue alone: it waits on the queue AND the
    pump task, so a pump that ends without delivering another item (upstream
    error, an upstream ``CancelledError`` such as grpc.aio cancelling the call,
    or :meth:`aclose` racing a waiting consumer) always wakes it. A stream that
    waits only on its queue hangs forever on those paths.
    """

    def __init__(self, factory: Callable[[], AsyncIterator[Any]]) -> None:
        self._factory = factory
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._task: asyncio.Task[None] | None = None
        self._error: BaseException | None = None
        self._closing = False
        self._done = False

    def _record(self, exc: BaseException) -> None:
        if self._error is None and not self._closing:
            self._error = exc

    async def _pump(self) -> None:
        agen = self._factory()
        try:
            async for item in agen:
                await self._queue.put(item)
        except asyncio.CancelledError as exc:
            if self._closing:
                raise
            # Not our cancel: the upstream raised it (e.g. grpc.aio). It is the
            # stream's terminal error and must reach the consumer.
            self._record(exc)
        except BaseException as exc:  # handed to the consumer's task
            self._record(exc)
        finally:
            aclose = getattr(agen, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except BaseException as exc:
                    self._record(exc)
                    if isinstance(exc, asyncio.CancelledError) and self._closing:
                        raise

    def __aiter__(self) -> SingleTaskStream:
        """Return self (single-pass iterator)."""
        return self

    def _finish(self) -> None:
        """The pump is done and the queue is drained: end or raise."""
        self._done = True
        task = self._task
        if task is not None and task.done() and not task.cancelled():
            exc = task.exception()
            if exc is not None:
                self._record(exc)
        if self._error is not None:
            raise self._error
        raise StopAsyncIteration

    async def __anext__(self) -> Any:
        """Start the pump on first call, then hand back its next item."""
        if self._done:
            raise StopAsyncIteration
        if self._task is None:
            self._task = asyncio.create_task(self._pump(), name="xai-chat-stream")
        task = self._task
        while True:
            if not self._queue.empty():
                return self._queue.get_nowait()
            if task.done():
                self._finish()
            getter = asyncio.ensure_future(self._queue.get())
            try:
                await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
            except BaseException:
                getter.cancel()
                raise
            if getter.done() and not getter.cancelled():
                return getter.result()
            getter.cancel()

    async def aclose(self) -> None:
        """Cancel the pump and wait for it, so the generator closes in its own task.

        ``asyncio.wait`` (not ``await task``) so the pump's own cancellation is
        not re-raised here, while a cancellation aimed at the CALLER of
        ``aclose`` still propagates instead of being swallowed.
        """
        self._closing = True
        self._done = True
        task = self._task
        if task is None:
            return
        if not task.done():
            task.cancel()
            await asyncio.wait({task})
        if not task.cancelled():
            task.exception()  # retrieved: never "exception was never retrieved"


class _ChatProxy:
    """Delegates everything to the SDK chat object except ``stream()``."""

    def __init__(self, chat: Any) -> None:
        self._chat = chat

    def stream(self, *args: Any, **kwargs: Any) -> SingleTaskStream:
        return SingleTaskStream(lambda: self._chat.stream(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class PinnedContextXaiModel(XaiModel):
    """``XaiModel`` whose streamed chat runs in one task (see module docstring)."""

    async def _create_chat(self, *args: Any, **kwargs: Any) -> Any:
        chat = await super()._create_chat(*args, **kwargs)
        return _ChatProxy(chat)
