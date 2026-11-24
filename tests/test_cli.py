"""Testes da CLI com dependências substituídas por doubles."""

from __future__ import annotations

import json

import pytest
from langchain_core.documents import Document

from qlik_rag.cli import main as cli_main
from qlik_rag.services.chat_service import ChatEvent
from qlik_rag.services.ingestion_service import IngestionReport


@pytest.fixture
def patched_cli(monkeypatch):
    class StubVectorStore:
        def __init__(self, settings) -> None:
            self.settings = settings

        def count(self) -> int:
            return 3

    class StubSearch:
        def __init__(self, vector_store) -> None:
            self.vector_store = vector_store

        def search(self, query: str, k: int | None = None) -> list[Document]:
            return [Document(page_content="hit", metadata={"source": "a.pdf", "k": k})]

    class StubIngestion:
        def __init__(self, settings, ingestor, vector_store) -> None:
            self.force_calls = []

        def ensure_indexed(self, force: bool = False) -> IngestionReport:
            self.force_calls.append(force)
            return IngestionReport(indexed_chunks=5, skipped=not force)

    monkeypatch.setattr(cli_main, "VectorStoreService", StubVectorStore)
    monkeypatch.setattr(cli_main, "SearchService", StubSearch)
    monkeypatch.setattr(cli_main, "IngestionService", StubIngestion)
    monkeypatch.setattr(cli_main, "_build_ingestor", lambda settings: object())


def test_ask_streams_tokens_and_sources(patched_cli, monkeypatch, capsys) -> None:
    class StubChatService:
        def stream_answer(self, question, history=None):
            yield ChatEvent(kind="sources", documents=[
                Document(page_content="x", metadata={"source": "a.pdf", "page": 2})
            ])
            yield ChatEvent(kind="token", text="Hello ")
            yield ChatEvent(kind="token", text="world")
            yield ChatEvent(kind="done", text="Hello world")

    monkeypatch.setattr(cli_main, "build_chat_service", lambda settings: StubChatService())

    exit_code = cli_main.main(["ask", "how?"])

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "Hello world" in captured
    assert "a.pdf" in captured


def test_stats_prints_chunk_count(patched_cli, capsys) -> None:
    exit_code = cli_main.main(["stats"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"collection": "qlik_documentation", "chunks": 3}


def test_query_prints_results(patched_cli, capsys) -> None:
    exit_code = cli_main.main(["query", "create bookmark", "-k", "2"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["query"] == "create bookmark"
    assert payload["results"][0]["content"] == "hit"


def test_ingest_forwards_force_flag(patched_cli, capsys) -> None:
    exit_code = cli_main.main(["ingest", "--force"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["skipped"] is False
