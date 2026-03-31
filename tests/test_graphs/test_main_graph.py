"""Тесты для LangGraph — сборка и базовая работа графа."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.graphs.main_graph import _proposal_node_with_state, build_graph
from src.schemas.state import IntakeData


def _make_mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="INTAKE"))
    return llm


class TestBuildGraph:
    def test_graph_compiles(self) -> None:
        llm = _make_mock_llm()
        graph = build_graph(
            supervisor_llm=llm,
            intake_llm=llm,
            proposal_llm=llm,
        )
        assert graph is not None

    @pytest.mark.asyncio
    async def test_proposal_wrapper_sets_few_shot_default(self) -> None:
        llm = _make_mock_llm()
        state = {"messages": [], "chat_id": 1, "current_agent": "proposal", "is_complete": False}
        with patch("src.graphs.main_graph.proposal_node", AsyncMock(return_value={"ok": True})) as mock_node:
            result = await _proposal_node_with_state(state, llm=llm)

        assert result == {"ok": True}
        passed_state = mock_node.call_args.args[0]
        assert passed_state["few_shot_examples"] is None

    def test_graph_has_expected_nodes(self) -> None:
        llm = _make_mock_llm()
        graph = build_graph(
            supervisor_llm=llm,
            intake_llm=llm,
            proposal_llm=llm,
        )
        node_names = set(graph.get_graph().nodes.keys())
        assert "supervisor" in node_names
        assert "intake" in node_names
        assert "proposal" in node_names

    def test_custom_checkpointer(self) -> None:
        llm = _make_mock_llm()
        saver = MemorySaver()
        graph = build_graph(
            supervisor_llm=llm,
            intake_llm=llm,
            proposal_llm=llm,
            checkpointer=saver,
        )
        assert graph is not None


class TestGraphFlow:
    @pytest.mark.asyncio
    async def test_greeting_flow(self) -> None:
        """Supervisor возвращает GREETING → граф завершается с приветствием."""
        supervisor_llm = MagicMock()
        supervisor_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="GREETING")
        )
        intake_llm = _make_mock_llm()
        proposal_llm = _make_mock_llm()

        graph = build_graph(
            supervisor_llm=supervisor_llm,
            intake_llm=intake_llm,
            proposal_llm=proposal_llm,
        )

        config = {"configurable": {"thread_id": "test-greeting"}}
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Привет!")]},
            config=config,
        )

        assert len(result["messages"]) >= 2
        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "Аналитик.Лаб" in last_msg.content

    @pytest.mark.asyncio
    async def test_intake_flow(self) -> None:
        """Supervisor → INTAKE → intake_node задаёт вопрос → возвращается в supervisor."""
        supervisor_responses = iter([
            MagicMock(content="INTAKE"),
            MagicMock(content="GREETING"),
        ])
        supervisor_llm = MagicMock()
        supervisor_llm.ainvoke = AsyncMock(side_effect=lambda _: next(supervisor_responses))

        intake_llm = MagicMock()
        intake_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="Какой тип анализа вас интересует?")
        )
        proposal_llm = _make_mock_llm()

        graph = build_graph(
            supervisor_llm=supervisor_llm,
            intake_llm=intake_llm,
            proposal_llm=proposal_llm,
        )

        config = {"configurable": {"thread_id": "test-intake"}}
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Нужен анализ")]},
            config=config,
        )

        all_text = " ".join(m.content for m in result["messages"] if isinstance(m, AIMessage))
        assert "анализ" in all_text.lower() or "Аналитик" in all_text
