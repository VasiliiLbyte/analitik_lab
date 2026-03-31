"""Supervisor Agent — центральный диспетчер, маршрутизирует запросы к sub-агентам.

Использует GigaChat 2 Pro для анализа intent-а и принятия решений.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage
from langchain_gigachat import GigaChat
from langgraph.types import Command
from loguru import logger

from src.schemas.state import AgentState

SUPERVISOR_SYSTEM_PROMPT = """\
Ты — Supervisor-агент лаборатории «Аналитик.Лаб». Твоя единственная задача —
определить intent пользователя и решить, какого агента запустить.

Возможные intent-ы (ответь ОДНИМ СЛОВОМ):
- INTAKE — нужно уточнить данные у клиента (новый запрос, мало информации)
- PROPOSAL — все данные собраны, можно формировать КП
- GREETING — приветствие или общий вопрос
- UNKNOWN — непонятный запрос

Правила:
- Если intake_data не заполнен или есть пустые обязательные поля → INTAKE
- Если intake_data полностью заполнен → PROPOSAL
- Если пользователь просто здоровается → GREETING
- Отвечай СТРОГО одним словом: INTAKE, PROPOSAL, GREETING или UNKNOWN
"""

GREETING_RESPONSE = (
    "Здравствуйте! Я AI-ассистент лаборатории «Аналитик.Лаб». "
    "Помогу подготовить коммерческое предложение на экологические анализы.\n\n"
    "Расскажите, какой анализ вам нужен?"
)

UNKNOWN_RESPONSE = (
    "Извините, я не совсем понял ваш запрос. "
    "Я могу помочь с оформлением заказа на экологические анализы "
    "(вода, почва, воздух, отходы). Чем могу помочь?"
)


def create_supervisor_llm(
    credentials: str,
    scope: str = "GIGACHAT_API_PERS",
) -> GigaChat:
    """Фабрика LLM для Supervisor-агента."""
    return GigaChat(
        credentials=credentials,
        scope=scope,
        model="GigaChat-2-Pro",
        verify_ssl_certs=False,
        timeout=30,
    )


def _classify_intent(response_text: str) -> str:
    """Извлекает intent из ответа LLM."""
    text = response_text.strip().upper()
    for intent in ("INTAKE", "PROPOSAL", "GREETING", "UNKNOWN"):
        if intent in text:
            return intent
    return "UNKNOWN"


async def supervisor_node(
    state: AgentState,
    *,
    llm: GigaChat | None = None,
) -> Command:
    """LangGraph-нода Supervisor: классифицирует intent и маршрутизирует.

    Возвращает Command(goto=...) для перехода к нужному агенту.
    """
    if llm is None:
        raise RuntimeError("LLM не передан в supervisor_node")

    intake_data = state.get("intake_data")
    if intake_data is not None and intake_data.is_complete:
        logger.info("Supervisor: intake_data заполнен → PROPOSAL")
        return Command(
            goto="proposal",
            update={"current_agent": "proposal"},
        )

    intake_status = "не заполнен"
    if intake_data is not None:
        missing = intake_data.missing_fields()
        if missing:
            intake_status = f"не хватает: {', '.join(missing)}"
        else:
            intake_status = "заполнен полностью"

    messages = list(state.get("messages", []))
    conversation = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        SystemMessage(content=f"Статус intake_data: {intake_status}"),
        *messages[-5:],
    ]

    try:
        response = await llm.ainvoke(conversation)
        intent = _classify_intent(response.content)
    except Exception:
        logger.exception("Ошибка при вызове GigaChat в supervisor_node")
        intent = "INTAKE"

    logger.info("Supervisor intent: {}", intent)

    if intent == "PROPOSAL":
        return Command(
            goto="proposal",
            update={"current_agent": "proposal"},
        )

    if intent == "GREETING":
        return Command(
            goto="__end__",
            update={
                "messages": [AIMessage(content=GREETING_RESPONSE)],
                "current_agent": "supervisor",
            },
        )

    if intent == "UNKNOWN":
        return Command(
            goto="__end__",
            update={
                "messages": [AIMessage(content=UNKNOWN_RESPONSE)],
                "current_agent": "supervisor",
            },
        )

    # INTAKE (default)
    return Command(
        goto="intake",
        update={"current_agent": "intake"},
    )
