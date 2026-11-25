"""Estado compartilhado entre os nós do grafo de geração."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    """Estado do grafo: histórico, pergunta do usuário, busca e trechos recuperados.

    `question` é a pergunta original e nunca é reescrita: é ela que o usuário vê
    e que a resposta precisa endereçar. `search_query` é a consulta corrente de
    busca, que a reformulação atualiza.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    search_query: str
    documents: list[Document]
    attempts: int
    sufficient: bool
