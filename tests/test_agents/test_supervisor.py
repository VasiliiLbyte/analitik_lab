"""Тесты для Supervisor Agent — маршрутизация intent-ов."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.agents.supervisor import (
    GREETING_RESPONSE,
    UNKNOWN_RESPONSE,
    _classify_intent,
    supervisor_node,
)
from src.schemas.state import AgentState, IntakeData


def _make_llm_mock(response_text: str) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content=response_text))
    return llm


def _make_state(**overrides) -> AgentState:
    base: AgentState = {
        "messages": [HumanMessage(content="Нужен анализ воды")],
        "chat_id": 1,
        "current_agent": "supervisor",
        "is_complete": False,
    }
    base.update(overrides)
    return base


class TestClassifyIntent:
    def test_intake(self) -> None:
        assert _classify_intent("INTAKE") == "INTAKE"

    def test_proposal(self) -> None:
        assert _classify_intent("PROPOSAL") == "PROPOSAL"

    def test_greeting(self) -> None:
        assert _classify_intent("GREETING") == "GREETING"

    def test_unknown_fallback(self) -> None:
        assert _classify_intent("что-то непонятное") == "UNKNOWN"

    def test_case_insensitive(self) -> None:
        assert _classify_intent("intake") == "INTAKE"

    def test_with_extra_text(self) -> None:
        assert _classify_intent("Я думаю, что PROPOSAL подойдёт") == "PROPOSAL"


class TestSupervisorNode:
    @pytest.mark.asyncio
    async def test_routes_to_intake_on_new_request(self) -> None:
        llm = _make_llm_mock("INTAKE")
        result = await supervisor_node(_make_state(), llm=llm)

        assert isinstance(result, Command)
        assert result.goto == "intake"

    @pytest.mark.asyncio
    async def test_routes_to_proposal_when_data_complete(self) -> None:
        complete_intake = IntakeData(
            analysis_type="вода",
            purpose="питьевая",
            address="СПб",
            num_points=1,
            deadlines="3 дня",
        )
        state = _make_state(intake_data=complete_intake)
        llm = _make_llm_mock("PROPOSAL")

        result = await supervisor_node(state, llm=llm)

        assert isinstance(result, Command)
        assert result.goto == "proposal"

    @pytest.mark.asyncio
    async def test_shortcut_to_proposal_when_intake_complete(self) -> None:
        """Если intake_data.is_complete — идём в proposal без вызова LLM."""
        complete_intake = IntakeData(
            analysis_type="вода",
            purpose="питьевая",
            address="СПб",
            num_points=1,
            deadlines="3 дня",
        )
        llm = _make_llm_mock("INTAKE")
        state = _make_state(intake_data=complete_intake)

        result = await supervisor_node(state, llm=llm)

        assert result.goto == "proposal"
        llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_greeting_response(self) -> None:
        llm = _make_llm_mock("GREETING")
        state = _make_state(messages=[HumanMessage(content="Привет!")])

        result = await supervisor_node(state, llm=llm)

        assert result.goto == "__end__"
        assert GREETING_RESPONSE in result.update["messages"][0].content

    @pytest.mark.asyncio
    async def test_unknown_response(self) -> None:
        llm = _make_llm_mock("UNKNOWN")
        state = _make_state(messages=[HumanMessage(content="asdfqwerty")])

        result = await supervisor_node(state, llm=llm)

        assert result.goto == "__end__"
        assert UNKNOWN_RESPONSE in result.update["messages"][0].content

    @pytest.mark.asyncio
    async def test_fallback_to_intake_on_llm_error(self) -> None:
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("API down"))

        result = await supervisor_node(_make_state(), llm=llm)

        assert result.goto == "intake"

    @pytest.mark.asyncio
    async def test_raises_without_llm(self) -> None:
        with pytest.raises(RuntimeError, match="LLM не передан"):
            await supervisor_node(_make_state(), llm=None)
