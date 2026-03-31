"""Middleware для Telegram-бота: логирование, rate-limiting, обработка ошибок."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject
from loguru import logger

MAX_MESSAGES_PER_MINUTE = 10


class LoggingMiddleware(BaseMiddleware):
    """Логирует каждое входящее сообщение (без PII — только chat_id и длину текста)."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if isinstance(event, Message) and event.text:
            logger.info(
                "Входящее сообщение | chat_id={} | len={}",
                event.chat.id,
                len(event.text),
            )
        return await handler(event, data)


class RateLimitMiddleware(BaseMiddleware):
    """Простой in-memory rate limiter — не более N сообщений в минуту на пользователя."""

    def __init__(self, max_per_minute: int = MAX_MESSAGES_PER_MINUTE) -> None:
        super().__init__()
        self._max = max_per_minute
        self._timestamps: dict[int, list[float]] = defaultdict(list)

    def _clean_old(self, chat_id: int) -> None:
        cutoff = time.monotonic() - 60
        self._timestamps[chat_id] = [
            ts for ts in self._timestamps[chat_id] if ts > cutoff
        ]

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if not isinstance(event, Message):
            return await handler(event, data)

        chat_id = event.chat.id
        self._clean_old(chat_id)

        if len(self._timestamps[chat_id]) >= self._max:
            logger.warning("Rate limit exceeded | chat_id={}", chat_id)
            await event.answer(
                "Слишком много сообщений. Пожалуйста, подождите минуту."
            )
            return None

        self._timestamps[chat_id].append(time.monotonic())
        return await handler(event, data)


class ErrorHandlerMiddleware(BaseMiddleware):
    """Перехватывает исключения в хэндлерах и отправляет пользователю безопасное сообщение."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        try:
            return await handler(event, data)
        except Exception:
            logger.exception("Необработанная ошибка в хэндлере")
            if isinstance(event, Message):
                await event.answer(
                    "Произошла внутренняя ошибка. Мы уже разбираемся. "
                    "Попробуйте ещё раз через пару минут."
                )
            return None
