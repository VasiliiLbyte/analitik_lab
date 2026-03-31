"""Интеграционный тест: полный цикл от сообщения до генерации КП."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.graphs.main_graph import build_graph
from src.schemas.state import IntakeData


def _llm_proposal_json() -> str:
    return json.dumps(
        {
            "proposal_number": "АЛ-2026-INT",
            "client_name": "Тест Интеграционный",
            "items": [{"name": "Анализ воды", "params_count": 14, "price": 8500}],
            "total_price": 8500,
            "address": "СПб, Невский 1",
            "deadlines": "3 дня",
        },
        ensure_ascii=False,
    )


class TestGreetingFlow:
    """Пользователь здоровается → бот отвечает приветствием."""

    @pytest.mark.asyncio
    async def test_greeting_ends_immediately(self) -> None:
        supervisor_llm = MagicMock()
        supervisor_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="GREETING")
        )
        intake_llm = MagicMock()
        intake_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Тест"))
        proposal_llm = MagicMock()
        proposal_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content=_llm_proposal_json())
        )

        graph = build_graph(
            supervisor_llm=supervisor_llm,
            intake_llm=intake_llm,
            proposal_llm=proposal_llm,
            checkpointer=MemorySaver(),
        )
        config = {"configurable": {"thread_id": "greet-1"}}

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Привет!")]},
            config=config,
        )

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) >= 1
        assert "Аналитик.Лаб" in ai_msgs[-1].content
        intake_llm.ainvoke.assert_not_called()


class TestIntakeToProposalFlow:
    """Полный цикл: supervisor → intake → (данные собраны) → supervisor → proposal."""

    @pytest.mark.asyncio
    async def test_single_intake_cycle_then_proposal(self) -> None:
        """Один проход intake → данные собраны → proposal генерирует КП."""
        supervisor_call_count = 0

        async def supervisor_side_effect(_messages):
            nonlocal supervisor_call_count
            supervisor_call_count += 1
            if supervisor_call_count == 1:
                return MagicMock(content="INTAKE")
            return MagicMock(content="PROPOSAL")

        supervisor_llm = MagicMock()
        supervisor_llm.ainvoke = AsyncMock(side_effect=supervisor_side_effect)

        intake_llm = MagicMock()
        intake_llm.ainvoke = AsyncMock(
            return_value=MagicMock(
                content="Спасибо! Все данные собраны, формирую коммерческое предложение."
            )
        )

        proposal_llm = MagicMock()
        proposal_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content=_llm_proposal_json())
        )

        graph = build_graph(
            supervisor_llm=supervisor_llm,
            intake_llm=intake_llm,
            proposal_llm=proposal_llm,
            checkpointer=MemorySaver(),
        )
        config = {"configurable": {"thread_id": "intake-proposal-1"}}

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Питьевая вода, СПб, 1 точка, срочно")]},
            config=config,
        )

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) >= 1
        assert result.get("proposal_data") is not None
        assert result["proposal_data"].proposal_number == "АЛ-2026-INT"


class TestStateIsolation:
    """Проверяем, что разные thread_id изолированы друг от друга."""

    @pytest.mark.asyncio
    async def test_different_threads_independent(self) -> None:
        supervisor_llm = MagicMock()
        supervisor_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="GREETING")
        )
        intake_llm = MagicMock()
        intake_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Тест"))
        proposal_llm = MagicMock()
        proposal_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content=_llm_proposal_json())
        )

        saver = MemorySaver()
        graph = build_graph(
            supervisor_llm=supervisor_llm,
            intake_llm=intake_llm,
            proposal_llm=proposal_llm,
            checkpointer=saver,
        )

        r_a = await graph.ainvoke(
            {"messages": [HumanMessage(content="Привет от A")]},
            config={"configurable": {"thread_id": "user-A"}},
        )
        r_b = await graph.ainvoke(
            {"messages": [HumanMessage(content="Привет от B")]},
            config={"configurable": {"thread_id": "user-B"}},
        )

        msgs_a = [m.content for m in r_a["messages"]]
        msgs_b = [m.content for m in r_b["messages"]]
        assert "Привет от A" in msgs_a[0]
        assert "Привет от B" in msgs_b[0]
