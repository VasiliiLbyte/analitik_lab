"""Тесты для src/schemas/state.py — Pydantic-модели и AgentState."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from src.schemas.state import (
    AgentState,
    IntakeData,
    ProposalData,
    ProposalItem,
)


class TestIntakeData:
    def test_empty_intake_has_all_missing(self) -> None:
        data = IntakeData()
        assert set(data.missing_fields()) == {
            "analysis_type",
            "purpose",
            "address",
            "num_points",
            "deadlines",
        }
        assert data.is_complete is False

    def test_partial_intake_reports_missing(self) -> None:
        data = IntakeData(analysis_type="вода", purpose="питьевая")
        missing = data.missing_fields()
        assert "analysis_type" not in missing
        assert "purpose" not in missing
        assert "address" in missing
        assert data.is_complete is False

    def test_complete_intake(self) -> None:
        data = IntakeData(
            analysis_type="вода",
            purpose="питьевая",
            address="СПб, Невский 1",
            num_points=2,
            deadlines="срочно",
        )
        assert data.is_complete is True
        assert data.missing_fields() == []

    def test_num_points_must_be_positive(self) -> None:
        with pytest.raises(ValidationError, match="num_points"):
            IntakeData(num_points=0)

    def test_serialization_roundtrip(self) -> None:
        data = IntakeData(analysis_type="почва", address="Москва")
        restored = IntakeData.model_validate_json(data.model_dump_json())
        assert restored == data


class TestProposalItem:
    def test_valid_item(self) -> None:
        item = ProposalItem(name="Анализ воды", params_count=14, price=8500)
        assert item.name == "Анализ воды"
        assert item.price == 8500

    def test_negative_price_raises(self) -> None:
        with pytest.raises(ValidationError, match="price"):
            ProposalItem(name="test", price=-1)


class TestProposalData:
    def test_defaults(self) -> None:
        data = ProposalData(proposal_number="АЛ-2026-001")
        assert data.proposal_date == date.today()
        assert data.validity_days == 30
        assert data.items == []
        assert data.total_price == 0

    def test_with_items(self) -> None:
        item = ProposalItem(name="Анализ воды", params_count=14, price=8500)
        data = ProposalData(
            proposal_number="АЛ-2026-002",
            items=[item],
            total_price=8500,
        )
        assert len(data.items) == 1
        assert data.total_price == 8500

    def test_serialization_roundtrip(self) -> None:
        data = ProposalData(
            proposal_number="АЛ-2026-003",
            client_name="Иван Иванов",
            address="СПб",
            deadlines="3 дня",
        )
        restored = ProposalData.model_validate_json(data.model_dump_json())
        assert restored == data


class TestAgentState:
    def test_can_create_minimal_state(self) -> None:
        state: AgentState = {
            "messages": [],
            "chat_id": 12345,
            "current_agent": "supervisor",
            "is_complete": False,
        }
        assert state["messages"] == []
        assert state["chat_id"] == 12345

    def test_state_with_intake_data(self) -> None:
        intake = IntakeData(analysis_type="вода")
        state: AgentState = {
            "messages": [],
            "intake_data": intake,
            "few_shot_examples": None,
            "chat_id": 1,
            "current_agent": "intake",
            "is_complete": False,
        }
        assert state["intake_data"] is not None
        assert state["intake_data"].analysis_type == "вода"

    def test_state_can_store_few_shot_examples(self) -> None:
        state: AgentState = {
            "messages": [],
            "chat_id": 1,
            "current_agent": "proposal",
            "is_complete": False,
            "few_shot_examples": "### Example 1 (water.pdf)\n...",
        }
        assert state["few_shot_examples"] is not None
