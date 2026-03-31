"""Тесты для Intake & Clarification Agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.intake import DATA_COMPLETE_MARKER, intake_node
from src.schemas.state import AgentState, IntakeData


def _make_llm_mock(response_text: str) -> MagicMock:
    """Создаёт мок GigaChat, возвращающий заданный текст."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(content=response_text)
    )
    return llm


def _make_state(**overrides) -> AgentState:
    base: AgentState = {
        "messages": [],
        "chat_id": 1,
        "current_agent": "intake",
        "is_complete": False,
    }
    base.update(overrides)
    return base


class TestIntakeNode:
    @pytest.mark.asyncio
    async def test_asks_question_on_empty_state(self) -> None:
        llm = _make_llm_mock("Какой тип анализа вас интересует?")
        state = _make_state(messages=[HumanMessage(content="Нужен анализ воды")])

        result = await intake_node(state, llm=llm)

        assert len(result["messages"]) == 1
        assert "анализ" in result["messages"][0].content.lower()
        assert result["current_agent"] == "intake"
        assert result["is_complete"] is False

    @pytest.mark.asyncio
    async def test_asks_about_missing_fields(self) -> None:
        llm = _make_llm_mock("Укажите адрес объекта и количество точек отбора.")
        intake = IntakeData(analysis_type="вода", purpose="питьевая")
        state = _make_state(
            intake_data=intake,
            messages=[HumanMessage(content="Питьевая вода")],
        )

        result = await intake_node(state, llm=llm)

        assert result["current_agent"] == "intake"
        assert result["is_complete"] is False
        assert result["intake_data"] is not None

    @pytest.mark.asyncio
    async def test_signals_completion_when_data_ready(self) -> None:
        response = "Спасибо! Все данные собраны, формирую коммерческое предложение."
        llm = _make_llm_mock(response)
        intake = IntakeData(
            analysis_type="вода",
            purpose="питьевая",
            address="СПб",
            num_points=1,
            deadlines="срочно",
        )
        state = _make_state(
            intake_data=intake,
            messages=[HumanMessage(content="Срочно, 1 точка, СПб")],
        )

        result = await intake_node(state, llm=llm)

        assert result["is_complete"] is True
        assert result["current_agent"] == "supervisor"
        assert DATA_COMPLETE_MARKER in result["messages"][0].content.lower()

    @pytest.mark.asyncio
    async def test_handles_llm_error_gracefully(self) -> None:
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("API unavailable"))
        state = _make_state(messages=[HumanMessage(content="Привет")])

        result = await intake_node(state, llm=llm)

        assert "ошибка" in result["messages"][0].content.lower()
        assert result["current_agent"] == "intake"

    @pytest.mark.asyncio
    async def test_raises_without_llm(self) -> None:
        state = _make_state(messages=[HumanMessage(content="Тест")])

        with pytest.raises(RuntimeError, match="LLM не передан"):
            await intake_node(state, llm=None)

    @pytest.mark.asyncio
    async def test_preserves_existing_intake_data(self) -> None:
        llm = _make_llm_mock("Уточните адрес объекта.")
        intake = IntakeData(analysis_type="почва")
        state = _make_state(
            intake_data=intake,
            messages=[HumanMessage(content="Анализ почвы")],
        )

        result = await intake_node(state, llm=llm)

        assert result["intake_data"].analysis_type == "почва"
