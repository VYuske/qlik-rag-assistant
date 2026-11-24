"""Estado compartilhado entre os nós do grafo de geração."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    """Estado do grafo: histórico, pergunta corrente e trechos recuperados."""

    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    documents: list[Document]
    attempts: int
    sufficient: bool
