"""Funções puras de formatação usadas pela interface."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

HEADER_KEYS = ("Header 1", "Header 2", "Header 3")


def source_label(document: Document) -> str:
    """Descreve a origem de um trecho em uma linha."""
    source = document.metadata.get("source")
    if not source:
        return "documento desconhecido"
    page = document.metadata.get("page")
    if page is None:
        return str(source)
    return f"{source} — pág. {page}"


def format_sources(documents: list[Document]) -> list[dict[str, Any]]:
    """Normaliza os trechos recuperados para exibição."""
    entries: list[dict[str, Any]] = []
    for document in documents:
        headers = [str(document.metadata[key]) for key in HEADER_KEYS if document.metadata.get(key)]
        entries.append(
            {
                "label": source_label(document),
                "headers": headers,
                "content": document.page_content,
            }
        )
    return entries


def build_history(stored: list[dict[str, str]]) -> list[BaseMessage]:
    """Converte o histórico da sessão em mensagens do LangChain."""
    history: list[BaseMessage] = []
    for item in stored:
        content = item.get("content", "")
        if item.get("role") == "user":
            history.append(HumanMessage(content=content))
        else:
            history.append(AIMessage(content=content))
    return history
