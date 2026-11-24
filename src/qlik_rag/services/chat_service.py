"""Serviço de chat que traduz o grafo RAG em eventos consumíveis pela UI e CLI."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk, BaseMessage

logger = logging.getLogger(__name__)

ANSWER_NODE = "generate"


@dataclass(frozen=True)
class ChatEvent:
    """Evento emitido durante a geração de uma resposta."""

    kind: Literal["sources", "token", "done"]
    text: str = ""
    documents: list[Document] = field(default_factory=list)


class ChatService:
    """Executa o grafo e expõe apenas eventos de alto nível."""

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    def _build_payload(
        self,
        question: str,
        history: list[BaseMessage] | None,
    ) -> dict[str, Any]:
        return {
            "messages": list(history or []),
            "question": question,
            "documents": [],
            "attempts": 0,
            "sufficient": False,
        }

    def stream_answer(
        self,
        question: str,
        history: list[BaseMessage] | None = None,
    ) -> Iterator[ChatEvent]:
        """Emite fontes, tokens e o texto final consolidado."""
        payload = self._build_payload(question, history)
        tokens: list[str] = []
        documents: list[Document] = []

        for mode, chunk in self._graph.stream(
            payload, stream_mode=["messages", "custom"]
        ):
            if mode == "custom" and chunk.get("type") == "sources":
                documents = chunk["documents"]
                yield ChatEvent(kind="sources", documents=documents)
                continue

            if mode == "messages":
                message, metadata = chunk
                if metadata.get("langgraph_node") != ANSWER_NODE:
                    continue
                if not isinstance(message, AIMessageChunk):
                    continue
                text = str(message.content)
                if text:
                    tokens.append(text)
                    yield ChatEvent(kind="token", text=text)

        yield ChatEvent(kind="done", text="".join(tokens), documents=documents)
