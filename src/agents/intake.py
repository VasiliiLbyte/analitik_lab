"""Intake & Clarification Agent — собирает информацию о заказе через уточняющие вопросы.

Использует GigaChat для анализа диалога и определения недостающих данных.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_gigachat import GigaChat
from loguru import logger

from src.schemas.state import AgentState, IntakeData

INTAKE_SYSTEM_PROMPT = """\
Ты — вежливый ассистент лаборатории «Аналитик.Лаб».
Твоя задача — собрать у клиента всю информацию для подготовки коммерческого предложения.

Тебе нужно узнать:
1. Тип анализа (вода, почва, воздух, отходы и т.д.)
2. Цель анализа (питьевая, техническая, экологический мониторинг и т.д.)
3. Адрес объекта (город, район, улица)
4. Количество точек отбора проб
5. Желаемые сроки выполнения

Правила:
- Задавай по 1–2 вопроса за раз, не перегружай клиента.
- Если клиент в одном сообщении дал несколько ответов — учти их все.
- Будь кратким и дружелюбным.
- Если все данные собраны, ответь СТРОГО фразой: «Спасибо! Все данные собраны, формирую коммерческое предложение.»
- Не выдумывай информацию. Спрашивай, если данных не хватает.
"""

DATA_COMPLETE_MARKER = "все данные собраны"


def _build_intake_data(current: IntakeData | None) -> IntakeData:
    """Возвращает копию текущих данных или пустой объект."""
    if current is not None:
        return current.model_copy()
    return IntakeData()


def create_intake_llm(
    credentials: str,
    scope: str = "GIGACHAT_API_PERS",
) -> GigaChat:
    """Фабрика LLM для Intake-агента."""
    return GigaChat(
        credentials=credentials,
        scope=scope,
        model="GigaChat",
        verify_ssl_certs=False,
        timeout=30,
    )


async def intake_node(
    state: AgentState,
    *,
    llm: GigaChat | None = None,
) -> dict:
    """LangGraph-нода: задаёт уточняющие вопросы и обновляет intake_data.

    Возвращает обновлённые поля AgentState.
    """
    intake_data = _build_intake_data(state.get("intake_data"))
    messages = list(state.get("messages", []))

    conversation = [SystemMessage(content=INTAKE_SYSTEM_PROMPT)]
    if intake_data.missing_fields():
        status = f"Уже известно: {intake_data.model_dump_json(exclude_none=True)}"
        conversation.append(HumanMessage(content=status))
    conversation.extend(messages)

    if llm is None:
        raise RuntimeError("LLM не передан в intake_node")

    try:
        response = await llm.ainvoke(conversation)
    except Exception:
        logger.exception("Ошибка при вызове GigaChat в intake_node")
        error_msg = AIMessage(
            content="Извините, произошла техническая ошибка. Попробуйте ещё раз."
        )
        return {"messages": [error_msg], "current_agent": "intake"}

    ai_message = AIMessage(content=response.content)
    is_done = DATA_COMPLETE_MARKER in response.content.lower()

    return {
        "messages": [ai_message],
        "intake_data": intake_data,
        "current_agent": "supervisor" if is_done else "intake",
        "is_complete": is_done,
    }
