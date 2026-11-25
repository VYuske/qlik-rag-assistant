"""Testes da orquestração de ingestão idempotente."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from qlik_rag.config.settings import Settings
from qlik_rag.retrieval.vector_store import VectorStoreService
from qlik_rag.services.ingestion_service import IngestionService


class StubIngestor:
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        self.calls = 0

    def ingest(self) -> list[Document]:
        self.calls += 1
        return self._documents


def build_service(
    settings: Settings, documents: list[Document]
) -> tuple[IngestionService, StubIngestor]:
    ingestor = StubIngestor(documents)
    vector_store = VectorStoreService(settings, embeddings=FakeEmbeddings(size=8))
    return IngestionService(settings, ingestor, vector_store), ingestor


def test_indexes_when_collection_is_empty(settings: Settings) -> None:
    service, ingestor = build_service(settings, [Document(page_content="a", metadata={})])

    report = service.ensure_indexed()

    assert report.skipped is False
    assert report.indexed_chunks == 1
    assert ingestor.calls == 1


def test_skips_when_index_already_has_content(settings: Settings) -> None:
    service, ingestor = build_service(settings, [Document(page_content="a", metadata={})])
    service.ensure_indexed()

    report = service.ensure_indexed()

    assert report.skipped is True
    assert ingestor.calls == 1


def test_force_reindexes_existing_content(settings: Settings) -> None:
    service, ingestor = build_service(settings, [Document(page_content="a", metadata={})])
    service.ensure_indexed()

    report = service.ensure_indexed(force=True)

    assert report.skipped is False
    assert ingestor.calls == 2


def test_empty_ingestion_does_not_crash(settings: Settings) -> None:
    service, _ = build_service(settings, [])

    report = service.ensure_indexed()

    assert report.indexed_chunks == 0
    assert report.skipped is False
