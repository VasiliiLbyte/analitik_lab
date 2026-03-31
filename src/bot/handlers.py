"""Aiogram-хэндлеры: /start и обработка текстовых сообщений через LangGraph."""

from __future__ import annotations

from pathlib import Path

from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import FSInputFile, Message
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

router = Router(name="main")

_graph: CompiledStateGraph | None = None


def set_graph(graph: CompiledStateGraph) -> None:
    """Регистрирует скомпилированный LangGraph для использования хэндлерами."""
    global _graph  # noqa: PLW0603
    _graph = graph


def _get_graph() -> CompiledStateGraph:
    if _graph is None:
        raise RuntimeError("LangGraph не инициализирован — вызовите set_graph()")
    return _graph


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    """Обработка команды /start."""
    await message.answer(
        "Здравствуйте! Я AI-ассистент лаборатории «Аналитик.Лаб».\n\n"
        "Помогу подготовить коммерческое предложение на экологические анализы "
        "(вода, почва, воздух, отходы).\n\n"
        "Расскажите, какой анализ вам нужен?"
    )


@router.message()
async def handle_message(message: Message) -> None:
    """Обработка любого текстового сообщения — прогоняет через LangGraph."""
    if not message.text:
        await message.answer("Пожалуйста, отправьте текстовое сообщение.")
        return

    text = message.text.strip()
    if len(text) > 4000:
        await message.answer("Сообщение слишком длинное. Пожалуйста, сократите до 4000 символов.")
        return

    graph = _get_graph()
    chat_id = message.chat.id
    config = {"configurable": {"thread_id": str(chat_id)}}

    logger.info("Запуск графа | chat_id={}", chat_id)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=text)], "chat_id": chat_id},
        config=config,
    )

    ai_messages = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]
    if not ai_messages:
        await message.answer("Не удалось получить ответ. Попробуйте ещё раз.")
        return

    last_reply = ai_messages[-1].content
    await message.answer(last_reply)

    file_path = result.get("proposal_file_path")
    if file_path and Path(file_path).exists():
        doc = FSInputFile(file_path)
        await message.answer_document(doc, caption="Ваше коммерческое предложение")
        logger.info("КП отправлено | chat_id={} | file={}", chat_id, file_path)
