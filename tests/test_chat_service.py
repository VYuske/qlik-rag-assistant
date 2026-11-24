"""Testes do ChatService: normalização dos streams em eventos."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk

from qlik_rag.services.chat_service import ChatEvent, ChatService


class FakeGraph:
    """Grafo falso que emite chunks nos formatos do LangGraph."""

    def __init__(self, chunks: list[tuple[str, Any]]) -> None:
        self._chunks = chunks
        self.calls: list[dict[str, Any]] = []

    def stream(self, payload: dict[str, Any], **kwargs: object) -> Iterator[tuple[str, Any]]:
        self.calls.append(payload)
        yield from self._chunks


def message_chunk(text: str) -> tuple[str, tuple[AIMessageChunk, dict[str, Any]]]:
    return ("messages", (AIMessageChunk(content=text), {"langgraph_node": "generate"}))


def sources_event(documents: list[Document]) -> tuple[str, dict[str, Any]]:
    return ("custom", {"type": "sources", "documents": documents})


def test_stream_answer_converts_streams_into_events() -> None:
    documents = [Document(page_content="trecho", metadata={"source": "g.pdf", "page": 1})]
    graph = FakeGraph(
        [
            sources_event(documents),
            message_chunk("Hello "),
            message_chunk("world"),
        ]
    )
    service = ChatService(graph=graph)

    events = list(service.stream_answer("pergunta"))

    assert [event.kind for event in events] == ["sources", "token", "token", "done"]
    assert events[0].documents == documents
    assert "".join(event.text for event in events if event.kind == "token") == "Hello world"
    assert events[-1].text == "Hello world"


def test_stream_answer_ignores_tokens_from_other_nodes() -> None:
    graph = FakeGraph(
        [
            ("messages", (AIMessageChunk(content="NAO"), {"langgraph_node": "grade"})),
            message_chunk("resposta"),
        ]
    )
    service = ChatService(graph=graph)

    events = list(service.stream_answer("pergunta"))

    assert "".join(event.text for event in events if event.kind == "token") == "resposta"


def test_stream_answer_sends_history_and_question() -> None:
    graph = FakeGraph([message_chunk("ok")])
    service = ChatService(graph=graph)

    events = list(service.stream_answer("nova pergunta", history=[]))

    assert events
    payload = graph.calls[0]
    assert payload["question"] == "nova pergunta"
    assert payload["messages"] == []


def test_stream_answer_without_messages_returns_empty_done() -> None:
    service = ChatService(graph=FakeGraph([]))

    events = list(service.stream_answer("pergunta"))

    assert events == [ChatEvent(kind="done", text="", documents=[])]


def test_final_complete_message_is_not_duplicated() -> None:
    """O LangGraph emite os chunks e a mensagem final; só os chunks contam."""
    graph = FakeGraph(
        [
            message_chunk("Hello "),
            message_chunk("world"),
            ("messages", (AIMessage(content="Hello world"), {"langgraph_node": "generate"})),
        ]
    )
    service = ChatService(graph=graph)

    events = list(service.stream_answer("pergunta"))

    assert [event.kind for event in events] == ["token", "token", "done"]
    assert events[-1].text == "Hello world"
