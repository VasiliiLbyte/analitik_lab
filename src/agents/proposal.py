"""Proposal Agent — формирует коммерческое предложение на основе собранных данных.

Использует GigaChat 2 Lite для структурирования данных в ProposalData,
затем генерирует DOCX через pdf_generator.
"""

from __future__ import annotations

import json
from datetime import date

from langchain_core.messages import AIMessage, SystemMessage
from langchain_gigachat import GigaChat
from loguru import logger

from src.schemas.state import AgentState, ProposalData, ProposalItem
from src.tools.pdf_generator import generate_proposal_docx

# TODO(Phase 2): подгружать few-shot примеры из knowledge/examples/kp/
# и добавлять в system prompt как образцы стиля и структуры КП.
# Это позволит LLM генерировать КП максимально близкие к реальным.

# TODO(Phase 2): подключить RAG (Qdrant) для получения актуальных цен
# из knowledge/prices/ вместо статического списка в промпте.

PROPOSAL_SYSTEM_PROMPT = """\
Ты — агент формирования коммерческих предложений лаборатории «Аналитик.Лаб».

На входе ты получаешь собранные данные о запросе клиента (intake_data).
Твоя задача — сформировать JSON-объект для генерации КП.

Ответь СТРОГО в формате JSON (без markdown-обрамления):
{
  "proposal_number": "АЛ-2026-XXX",
  "client_name": "Имя клиента если известно",
  "items": [
    {"name": "Название услуги", "params_count": 14, "price": 8500}
  ],
  "total_price": 8500,
  "address": "Адрес объекта",
  "deadlines": "Сроки выполнения"
}

Подбирай реалистичные цены для экологической лаборатории в СПб:
- Анализ питьевой воды (14 параметров) — 8 500 ₽
- Анализ питьевой воды (28 параметров) — 14 500 ₽
- Анализ сточных вод — 12 000 ₽
- Анализ почвы — 10 000 ₽
- Выездная служба — 3 000 ₽
"""


def create_proposal_llm(
    credentials: str,
    scope: str = "GIGACHAT_API_PERS",
) -> GigaChat:
    """Фабрика LLM для Proposal-агента."""
    return GigaChat(
        credentials=credentials,
        scope=scope,
        model="GigaChat-2-Lite",
        verify_ssl_certs=False,
        timeout=30,
    )


def _parse_proposal_json(text: str) -> ProposalData:
    """Извлекает ProposalData из JSON-ответа LLM."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    raw = json.loads(cleaned)
    items = [ProposalItem(**item) for item in raw.get("items", [])]
    return ProposalData(
        proposal_number=raw.get("proposal_number", f"АЛ-{date.today().year}-001"),
        client_name=raw.get("client_name", ""),
        items=items,
        total_price=raw.get("total_price", sum(i.price for i in items)),
        address=raw.get("address", ""),
        deadlines=raw.get("deadlines", ""),
    )


async def proposal_node(
    state: AgentState,
    *,
    llm: GigaChat | None = None,
) -> dict:
    """LangGraph-нода: формирует КП и генерирует DOCX-файл.

    Возвращает обновлённые поля AgentState.
    """
    intake_data = state.get("intake_data")
    if intake_data is None:
        error_msg = AIMessage(content="Недостаточно данных для формирования КП.")
        return {"messages": [error_msg], "current_agent": "intake"}

    if llm is None:
        raise RuntimeError("LLM не передан в proposal_node")

    intake_json = intake_data.model_dump_json(exclude_none=True)
    messages = [
        SystemMessage(content=PROPOSAL_SYSTEM_PROMPT),
        SystemMessage(content=f"Данные клиента: {intake_json}"),
    ]

    try:
        response = await llm.ainvoke(messages)
        proposal_data = _parse_proposal_json(response.content)
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.error("Не удалось распарсить ответ LLM для КП: {}", exc)
        error_msg = AIMessage(
            content="Не удалось сформировать КП. Попробуем ещё раз."
        )
        return {"messages": [error_msg], "current_agent": "supervisor"}
    except Exception:
        logger.exception("Ошибка при вызове GigaChat в proposal_node")
        error_msg = AIMessage(
            content="Извините, произошла техническая ошибка при формировании КП."
        )
        return {"messages": [error_msg], "current_agent": "supervisor"}

    try:
        docx_path = generate_proposal_docx(proposal_data)
        file_path = str(docx_path)
    except Exception:
        logger.exception("Ошибка генерации DOCX")
        file_path = None

    confirmation = (
        f"Коммерческое предложение {proposal_data.proposal_number} готово!\n"
        f"Итого: {proposal_data.total_price:,.0f} ₽"
    )
    ai_message = AIMessage(content=confirmation)

    return {
        "messages": [ai_message],
        "proposal_data": proposal_data,
        "proposal_file_path": file_path,
        "current_agent": "supervisor",
        "is_complete": True,
    }
