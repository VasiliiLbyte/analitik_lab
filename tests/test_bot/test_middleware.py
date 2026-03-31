"""Тесты для middleware: логирование, rate-limiting, обработка ошибок."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bot.middleware import (
    ErrorHandlerMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
)


def _make_message_event(chat_id: int = 1, text: str = "Тест") -> MagicMock:
    event = MagicMock()
    event.text = text
    event.chat = MagicMock(id=chat_id)
    event.answer = AsyncMock()
    # isinstance check uses __class__
    from aiogram.types import Message
    event.__class__ = Message
    return event


class TestLoggingMiddleware:
    @pytest.mark.asyncio
    async def test_calls_handler(self) -> None:
        mw = LoggingMiddleware()
        handler = AsyncMock(return_value="ok")
        event = _make_message_event()

        result = await mw(handler, event, {})

        handler.assert_called_once()
        assert result == "ok"


class TestRateLimitMiddleware:
    @pytest.mark.asyncio
    async def test_allows_normal_traffic(self) -> None:
        mw = RateLimitMiddleware(max_per_minute=5)
        handler = AsyncMock(return_value="ok")
        event = _make_message_event()

        for _ in range(5):
            result = await mw(handler, event, {})
            assert result == "ok"

    @pytest.mark.asyncio
    async def test_blocks_excessive_traffic(self) -> None:
        mw = RateLimitMiddleware(max_per_minute=3)
        handler = AsyncMock(return_value="ok")
        event = _make_message_event()

        for _ in range(3):
            await mw(handler, event, {})

        result = await mw(handler, event, {})
        assert result is None
        event.answer.assert_called()
        assert "подождите" in event.answer.call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_different_users_independent(self) -> None:
        mw = RateLimitMiddleware(max_per_minute=2)
        handler = AsyncMock(return_value="ok")

        event1 = _make_message_event(chat_id=1)
        event2 = _make_message_event(chat_id=2)

        await mw(handler, event1, {})
        await mw(handler, event1, {})
        await mw(handler, event1, {})

        result = await mw(handler, event2, {})
        assert result == "ok"


class TestErrorHandlerMiddleware:
    @pytest.mark.asyncio
    async def test_passes_through_on_success(self) -> None:
        mw = ErrorHandlerMiddleware()
        handler = AsyncMock(return_value="ok")
        event = _make_message_event()

        result = await mw(handler, event, {})
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_catches_exception(self) -> None:
        mw = ErrorHandlerMiddleware()
        handler = AsyncMock(side_effect=RuntimeError("boom"))
        event = _make_message_event()

        result = await mw(handler, event, {})

        assert result is None
        event.answer.assert_called_once()
        assert "ошибка" in event.answer.call_args[0][0].lower()
