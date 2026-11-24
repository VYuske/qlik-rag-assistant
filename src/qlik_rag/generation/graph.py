"""Grafo de geração RAG com loop de recuperação e avaliação."""

from __future__ import annotations

import logging
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from qlik_rag.config.prompts import (
    GRADE_PROMPT,
    RAG_USER_PROMPT,
    REFORMULATE_PROMPT,
    SYSTEM_PROMPT,
)
from qlik_rag.config.settings import Settings
from qlik_rag.generation.state import RAGState

logger = logging.getLogger(__name__)

RetrieveFn = Callable[[str], list[Document]]
AFFIRMATIVE_ANSWERS = {"sim", "yes", "s", "y"}


def format_context(documents: list[Document]) -> str:
    """Numera os trechos recuperados para que o modelo possa citá-los."""
    if not documents:
        return "(nenhum trecho recuperado)"
    blocks = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "desconhecido")
        page = document.metadata.get("page")
        location = f"{source}, pág. {page}" if page is not None else str(source)
        blocks.append(f"[S{index}] ({location})\n{document.page_content}")
    return "\n\n".join(blocks)


def is_affirmative(answer: str) -> bool:
    """Interpreta a resposta do avaliador como suficiente ou não."""
    normalized = answer.strip().lower().rstrip(".!")
    return normalized in AFFIRMATIVE_ANSWERS


def build_rag_graph(
    settings: Settings,
    retrieve: RetrieveFn,
    llm: BaseChatModel,
) -> object:
    """Monta e compila o grafo: retrieve -> grade -> (reformulate -> retrieve)* -> generate."""
    answer_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="messages"),
                ("human", RAG_USER_PROMPT),
            ]
        )
        | llm
    )
    grade_chain = ChatPromptTemplate.from_template(GRADE_PROMPT) | llm
    reformulate_chain = ChatPromptTemplate.from_template(REFORMULATE_PROMPT) | llm

    def retrieve_node(state: RAGState) -> dict[str, object]:
        question = state["question"]
        documents = retrieve(question)
        writer = get_stream_writer()
        writer({"type": "sources", "documents": documents})
        return {"documents": documents, "attempts": state.get("attempts", 0) + 1}

    def grade_node(state: RAGState) -> dict[str, object]:
        try:
            raw = grade_chain.invoke(
                {
                    "question": state["question"],
                    "context": format_context(state["documents"]),
                }
            )
        except Exception:
            logger.warning("Avaliação falhou; seguindo direto para a resposta.", exc_info=True)
            return {"sufficient": True}
        return {"sufficient": is_affirmative(str(raw.content))}

    def reformulate_node(state: RAGState) -> dict[str, object]:
        try:
            raw = reformulate_chain.invoke({"question": state["question"]})
            new_question = str(raw.content).strip() or state["question"]
        except Exception:
            logger.warning("Reformulação falhou; mantendo a pergunta original.", exc_info=True)
            new_question = state["question"]
        return {"question": new_question}

    def generate_node(state: RAGState) -> dict[str, object]:
        messages = [
            message
            for message in state.get("messages", [])
            if isinstance(message, BaseMessage)
        ]
        response = answer_chain.invoke(
            {
                "context": format_context(state["documents"]),
                "question": state["question"],
                "messages": messages,
            }
        )
        return {"messages": [AIMessage(content=str(response.content))]}

    def route_after_grade(state: RAGState) -> str:
        attempts = state.get("attempts", 0)
        if state.get("sufficient") or attempts > settings.max_retrieval_loops:
            return "generate"
        return "reformulate"

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("reformulate", reformulate_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "reformulate": "reformulate"},
    )
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


def to_messages(history: list[dict[str, str]] | None) -> list[BaseMessage]:
    """Converte o histórico armazenado em mensagens do LangChain."""
    if not history:
        return []
    converted: list[BaseMessage] = []
    for item in history:
        content = item.get("content", "")
        if item.get("role") == "user":
            converted.append(HumanMessage(content=content))
        else:
            converted.append(AIMessage(content=content))
    return converted
