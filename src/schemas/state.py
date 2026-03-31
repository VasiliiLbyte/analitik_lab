"""Pydantic-модели и LangGraph-состояние для пайплайна агентов."""

from __future__ import annotations

import operator
from datetime import date
from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Pydantic-модели данных, заполняемых агентами
# ---------------------------------------------------------------------------

class IntakeData(BaseModel):
    """Данные, собранные Intake-агентом в ходе уточняющего диалога."""

    analysis_type: str | None = Field(default=None, description="Тип анализа (вода, почва, воздух …)")
    purpose: str | None = Field(default=None, description="Цель анализа (питьевая, техническая …)")
    address: str | None = Field(default=None, description="Адрес объекта")
    num_points: int | None = Field(default=None, ge=1, description="Количество точек отбора")
    deadlines: str | None = Field(default=None, description="Желаемые сроки")
    extra_notes: str | None = Field(default=None, description="Дополнительные комментарии")

    def missing_fields(self) -> list[str]:
        """Возвращает список обязательных полей, которые ещё не заполнены."""
        required = ("analysis_type", "purpose", "address", "num_points", "deadlines")
        return [f for f in required if getattr(self, f) is None]

    @property
    def is_complete(self) -> bool:
        return len(self.missing_fields()) == 0


class ProposalItem(BaseModel):
    """Одна позиция коммерческого предложения."""

    name: str = Field(description="Наименование услуги")
    params_count: int = Field(default=0, ge=0, description="Количество параметров анализа")
    price: float = Field(ge=0, description="Стоимость, ₽")


class ProposalData(BaseModel):
    """Структурированные данные для генерации КП."""

    proposal_number: str = Field(description="Номер КП, например АЛ-2026-078")
    client_name: str = Field(default="", description="Имя клиента")
    proposal_date: date = Field(default_factory=date.today)
    items: list[ProposalItem] = Field(default_factory=list)
    total_price: float = Field(default=0, ge=0, description="Итого, ₽")
    validity_days: int = Field(default=30, ge=1, description="Срок действия КП в днях")
    address: str = Field(default="", description="Адрес объекта")
    deadlines: str = Field(default="", description="Сроки выполнения")


# ---------------------------------------------------------------------------
# LangGraph AgentState
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """Состояние графа LangGraph — единый словарь для всех агентов.

    `messages` использует reducer `operator.add` для накопления истории.
    """

    messages: Annotated[list[AnyMessage], operator.add]
    intake_data: IntakeData | None
    proposal_data: ProposalData | None
    current_agent: str
    chat_id: int
    proposal_file_path: str | None
    is_complete: bool
    metadata: dict[str, Any]
