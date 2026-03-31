"""Определение LangGraph StateGraph — основной граф агентов Analitik.Lab.

Граф: START → supervisor → (intake | proposal | END)
      intake → supervisor (цикл уточнения)
      proposal → END
"""

from __future__ import annotations

from functools import partial

from langchain_gigachat import GigaChat
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.intake import intake_node
from src.agents.proposal import proposal_node
from src.agents.supervisor import supervisor_node
from src.schemas.state import AgentState


def _bind_llm(node_fn, *, llm: GigaChat):
    """Создаёт partial-версию ноды с привязанным LLM."""
    return partial(node_fn, llm=llm)


def build_graph(
    *,
    supervisor_llm: GigaChat,
    intake_llm: GigaChat,
    proposal_llm: GigaChat,
    checkpointer: MemorySaver | None = None,
) -> CompiledStateGraph:
    """Собирает и компилирует граф агентов.

    Args:
        supervisor_llm: GigaChat 2 Pro для Supervisor
        intake_llm: GigaChat 2 Lite для Intake
        proposal_llm: GigaChat 2 Lite для Proposal
        checkpointer: хранилище состояний (MemorySaver для MVP)

    Returns:
        Скомпилированный граф, готовый к invoke/stream.
    """
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", _bind_llm(supervisor_node, llm=supervisor_llm))
    graph.add_node("intake", _bind_llm(intake_node, llm=intake_llm))
    graph.add_node("proposal", _bind_llm(proposal_node, llm=proposal_llm))

    graph.add_edge(START, "supervisor")
    graph.add_edge("intake", "supervisor")
    graph.add_edge("proposal", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)
