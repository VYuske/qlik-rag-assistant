"""Serviço de busca por similaridade sobre o índice vetorial."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qlik_rag.retrieval.vector_store import VectorStoreService

if TYPE_CHECKING:
    from langchain_core.documents import Document


class SearchService:
    """Executa consultas no índice e devolve os chunks mais similares."""

    def __init__(self, vector_store: VectorStoreService) -> None:
        self._vector_store = vector_store

    def search(self, query: str, k: int | None = None) -> list[Document]:
        """Retorna até `k` documentos relevantes para a consulta."""
        return self._vector_store.as_retriever(k=k).invoke(query)
