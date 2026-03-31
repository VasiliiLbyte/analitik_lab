"""Тесты для Proposal Agent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.agents.proposal import _parse_proposal_json, proposal_node
from src.schemas.state import AgentState, IntakeData, ProposalData


VALID_LLM_RESPONSE = json.dumps(
    {
        "proposal_number": "АЛ-2026-042",
        "client_name": "Иван Иванов",
        "items": [
            {"name": "Анализ питьевой воды", "params_count": 14, "price": 8500},
        ],
        "total_price": 8500,
        "address": "СПб, Невский 1",
        "deadlines": "3 рабочих дня",
    },
    ensure_ascii=False,
)


def _make_llm_mock(response_text: str) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content=response_text))
    return llm


def _make_state(**overrides) -> AgentState:
    base: AgentState = {
        "messages": [HumanMessage(content="Готов")],
        "intake_data": IntakeData(
            analysis_type="вода",
            purpose="питьевая",
            address="СПб, Невский 1",
            num_points=1,
            deadlines="3 дня",
        ),
        "chat_id": 1,
        "current_agent": "proposal",
        "is_complete": False,
    }
    base.update(overrides)
    return base


class TestParseProposalJson:
    def test_parses_valid_json(self) -> None:
        result = _parse_proposal_json(VALID_LLM_RESPONSE)
        assert isinstance(result, ProposalData)
        assert result.proposal_number == "АЛ-2026-042"
        assert len(result.items) == 1
        assert result.total_price == 8500

    def test_strips_markdown_fences(self) -> None:
        wrapped = f"```json\n{VALID_LLM_RESPONSE}\n```"
        result = _parse_proposal_json(wrapped)
        assert result.proposal_number == "АЛ-2026-042"

    def test_raises_on_invalid_json(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _parse_proposal_json("not json at all")


class TestProposalNode:
    @pytest.mark.asyncio
    async def test_generates_proposal(self) -> None:
        llm = _make_llm_mock(VALID_LLM_RESPONSE)

        with patch("src.agents.proposal.generate_proposal_docx") as mock_gen:
            mock_gen.return_value = MagicMock(__str__=lambda s: "/tmp/test.docx")
            result = await proposal_node(_make_state(), llm=llm)

        assert result["is_complete"] is True
        assert result["proposal_data"].proposal_number == "АЛ-2026-042"
        assert "АЛ-2026-042" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_returns_to_intake_without_data(self) -> None:
        llm = _make_llm_mock(VALID_LLM_RESPONSE)
        state = _make_state(intake_data=None)

        result = await proposal_node(state, llm=llm)

        assert result["current_agent"] == "intake"
        assert "недостаточно" in result["messages"][0].content.lower()

    @pytest.mark.asyncio
    async def test_handles_invalid_llm_response(self) -> None:
        llm = _make_llm_mock("это не JSON ответ")

        result = await proposal_node(_make_state(), llm=llm)

        assert result["current_agent"] == "supervisor"
        assert "не удалось" in result["messages"][0].content.lower()

    @pytest.mark.asyncio
    async def test_handles_llm_error(self) -> None:
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("API down"))

        result = await proposal_node(_make_state(), llm=llm)

        assert "ошибка" in result["messages"][0].content.lower()

    @pytest.mark.asyncio
    async def test_raises_without_llm(self) -> None:
        with pytest.raises(RuntimeError, match="LLM не передан"):
            await proposal_node(_make_state(), llm=None)

    @pytest.mark.asyncio
    async def test_handles_docx_generation_error(self) -> None:
        llm = _make_llm_mock(VALID_LLM_RESPONSE)

        with patch("src.agents.proposal.generate_proposal_docx") as mock_gen:
            mock_gen.side_effect = RuntimeError("Template error")
            result = await proposal_node(_make_state(), llm=llm)

        assert result["proposal_file_path"] is None
        assert result["is_complete"] is True
