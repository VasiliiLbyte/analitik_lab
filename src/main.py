"""Точка входа: запуск Telegram-бота с LangGraph-пайплайном."""

from __future__ import annotations

import asyncio
import sys

from aiogram import Bot, Dispatcher
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from src.agents.intake import create_intake_llm
from src.agents.proposal import create_proposal_llm
from src.agents.supervisor import create_supervisor_llm
from src.bot.handlers import router, set_graph
from src.bot.middleware import (
    ErrorHandlerMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
)
from src.config import get_settings
from src.graphs.main_graph import build_graph


def _configure_logging(level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )


async def main() -> None:
    settings = get_settings()
    _configure_logging(settings.log_level)
    logger.info("Запуск Analitik.Lab AI Bot")

    supervisor_llm = create_supervisor_llm(
        credentials=settings.gigachat_credentials,
        scope=settings.gigachat_scope,
    )
    intake_llm = create_intake_llm(
        credentials=settings.gigachat_credentials,
        scope=settings.gigachat_scope,
    )
    proposal_llm = create_proposal_llm(
        credentials=settings.gigachat_credentials,
        scope=settings.gigachat_scope,
    )

    graph = build_graph(
        supervisor_llm=supervisor_llm,
        intake_llm=intake_llm,
        proposal_llm=proposal_llm,
        checkpointer=MemorySaver(),
    )
    set_graph(graph)
    logger.info("LangGraph граф собран и зарегистрирован")

    bot = Bot(token=settings.telegram_bot_token)
    dp = Dispatcher()

    dp.message.middleware(LoggingMiddleware())
    dp.message.middleware(RateLimitMiddleware())
    dp.message.middleware(ErrorHandlerMiddleware())
    dp.include_router(router)

    logger.info("Бот запущен, ожидаем сообщения…")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
