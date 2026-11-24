"""Testes do serviço de vector store usando embeddings falsos e diretório temporário."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from qlik_rag.config.settings import Settings
from qlik_rag.retrieval.vector_store import VectorStoreService


def build_service(settings: Settings) -> VectorStoreService:
    return VectorStoreService(settings, embeddings=FakeEmbeddings(size=8))


def test_starts_empty(settings: Settings) -> None:
    assert build_service(settings).count() == 0


def test_add_documents_increments_count(settings: Settings) -> None:
    service = build_service(settings)

    service.add_documents(
        [
            Document(page_content="create bookmark", metadata={"source": "a.pdf"}),
            Document(page_content="load script", metadata={"source": "b.pdf"}),
        ]
    )

    assert service.count() == 2


def test_reset_empties_the_collection(settings: Settings) -> None:
    service = build_service(settings)
    service.add_documents([Document(page_content="anything", metadata={})])

    service.reset()

    assert service.count() == 0


def test_retriever_respects_k(settings: Settings) -> None:
    service = build_service(settings)
    service.add_documents(
        [Document(page_content=f"chunk {index}", metadata={}) for index in range(5)]
    )

    documents = service.as_retriever(k=2).invoke("chunk 1")

    assert len(documents) == 2
