"""Serviço de acesso ao vector store Chroma com embeddings do Ollama."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from qlik_rag.config.settings import Settings

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Encapsula o ciclo de vida do Chroma: criação, contagem, escrita e reset."""

    def __init__(self, settings: Settings, embeddings: Embeddings | None = None) -> None:
        self._settings = settings
        self._embeddings = embeddings or OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )
        self._store = self._open_store()

    def _open_store(self) -> Chroma:
        return Chroma(
            collection_name=self._settings.collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(self._settings.chroma_persist_dir),
        )

    @property
    def store(self) -> Chroma:
        """Instância Chroma subjacente."""
        return self._store

    def count(self) -> int:
        """Número de chunks indexados na collection."""
        return self._store._collection.count()

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Indexa os documentos e retorna os identificadores gerados."""
        ids = self._store.add_documents(documents)
        logger.info("Indexados %d chunks em '%s'.", len(ids), self._settings.collection_name)
        return ids

    def reset(self) -> None:
        """Remove todos os documentos da collection."""
        self._store.reset_collection()
        logger.info("Collection '%s' reiniciada.", self._settings.collection_name)

    def as_retriever(self, k: int | None = None) -> VectorStoreRetriever:
        """Retorna um retriever por similaridade com `k` resultados."""
        return self._store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k or self._settings.retrieval_k},
        )
