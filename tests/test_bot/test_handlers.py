"""Тесты для Telegram-бот хэндлеров."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.bot.handlers import _get_graph, cmd_start, handle_message, set_graph


@pytest.fixture(autouse=True)
def _reset_graph():
    """Сбрасываем глобальный граф перед каждым тестом."""
    set_graph(None)
    yield
    set_graph(None)


def _make_message(text: str = "Тест", chat_id: int = 42) -> MagicMock:
    msg = MagicMock()
    msg.text = text
    msg.chat = MagicMock(id=chat_id)
    msg.answer = AsyncMock()
    msg.answer_document = AsyncMock()
    return msg


class TestCmdStart:
    @pytest.mark.asyncio
    async def test_sends_welcome(self) -> None:
        msg = _make_message()
        await cmd_start(msg)
        msg.answer.assert_called_once()
        text = msg.answer.call_args[0][0]
        assert "Аналитик.Лаб" in text


class TestHandleMessage:
    @pytest.mark.asyncio
    async def test_raises_without_graph(self) -> None:
        msg = _make_message("Привет")
        with pytest.raises(RuntimeError, match="LangGraph не инициализирован"):
            await handle_message(msg)

    @pytest.mark.asyncio
    async def test_rejects_empty_text(self) -> None:
        msg = _make_message()
        msg.text = None
        set_graph(MagicMock())
        await handle_message(msg)
        msg.answer.assert_called_once()
        assert "текстовое" in msg.answer.call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_rejects_too_long_text(self) -> None:
        msg = _make_message("a" * 5000)
        set_graph(MagicMock())
        await handle_message(msg)
        msg.answer.assert_called_once()
        assert "длинное" in msg.answer.call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_processes_message_through_graph(self) -> None:
        ai_reply = AIMessage(content="Какой тип анализа?")
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={
            "messages": [HumanMessage(content="Тест"), ai_reply],
        })
        set_graph(graph)

        msg = _make_message("Нужен анализ")
        await handle_message(msg)

        graph.ainvoke.assert_called_once()
        msg.answer.assert_called_once_with("Какой тип анализа?")

    @pytest.mark.asyncio
    async def test_sends_document_when_file_exists(self, tmp_path: Path) -> None:
        docx_file = tmp_path / "test.docx"
        docx_file.write_text("fake docx content")

        ai_reply = AIMessage(content="КП готово!")
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={
            "messages": [ai_reply],
            "proposal_file_path": str(docx_file),
        })
        set_graph(graph)

        msg = _make_message("Готов")
        await handle_message(msg)

        msg.answer.assert_called_once_with("КП готово!")
        msg.answer_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_document_when_no_file(self) -> None:
        ai_reply = AIMessage(content="Уточните адрес.")
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={
            "messages": [ai_reply],
        })
        set_graph(graph)

        msg = _make_message("Тест")
        await handle_message(msg)

        msg.answer_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_empty_ai_response(self) -> None:
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={"messages": []})
        set_graph(graph)

        msg = _make_message("Тест")
        await handle_message(msg)

        assert "не удалось" in msg.answer.call_args[0][0].lower()
