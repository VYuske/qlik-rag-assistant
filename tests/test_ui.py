"""Testes das funções puras usadas pela interface Streamlit."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from qlik_rag.ui.formatting import build_history, format_sources, source_label


def test_source_label_includes_file_and_page() -> None:
    document = Document(page_content="x", metadata={"source": "guide.pdf", "page": 30})

    assert source_label(document) == "guide.pdf — pág. 30"


def test_source_label_handles_missing_metadata() -> None:
    document = Document(page_content="x", metadata={})

    assert source_label(document) == "documento desconhecido"


def test_format_sources_lists_headers_when_present() -> None:
    document = Document(
        page_content="x",
        metadata={"source": "guide.pdf", "page": 30, "Header 1": "Bookmarking", "Header 3": "4.1"},
    )

    entry = format_sources([document])[0]

    assert entry["label"] == "guide.pdf — pág. 30"
    assert "Bookmarking" in entry["headers"]
    assert "4.1" in entry["headers"]


def test_format_sources_omits_empty_headers() -> None:
    document = Document(page_content="x", metadata={"source": "guide.pdf"})

    assert format_sources([document])[0]["headers"] == []


def test_build_history_converts_stored_messages() -> None:
    stored = [
        {"role": "user", "content": "pergunta"},
        {"role": "assistant", "content": "resposta"},
    ]

    history = build_history(stored)

    assert isinstance(history[0], HumanMessage)
    assert isinstance(history[1], AIMessage)
    assert history[0].content == "pergunta"
    assert history[1].content == "resposta"


def test_build_history_handles_empty_session() -> None:
    assert build_history([]) == []
