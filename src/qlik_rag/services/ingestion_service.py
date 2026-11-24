"""Serviço que garante que o índice vetorial reflita os PDFs de origem."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from qlik_rag.config.settings import Settings
from qlik_rag.ingestion.ingestor import MarkdownIngestor
from qlik_rag.retrieval.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionReport:
    """Resultado de uma execução de ingestão."""

    indexed_chunks: int
    skipped: bool


class IngestionService:
    """Executa a ingestão de forma idempotente, respeitando o estado do índice."""

    def __init__(
        self,
        settings: Settings,
        ingestor: MarkdownIngestor,
        vector_store: VectorStoreService,
    ) -> None:
        self._settings = settings
        self._ingestor = ingestor
        self._vector_store = vector_store

    def ensure_indexed(self, force: bool = False) -> IngestionReport:
        """Indexa os PDFs quando o índice está vazio ou quando `force` é verdadeiro."""
        current_count = self._vector_store.count()
        if current_count > 0 and not force:
            logger.info("Índice já possui %d chunks; ingestão ignorada.", current_count)
            return IngestionReport(indexed_chunks=current_count, skipped=True)

        if force and current_count > 0:
            logger.info("Recriando índice por solicitação de --force.")
            self._vector_store.reset()

        documents = self._ingestor.ingest()
        self._vector_store.add_documents(documents)
        return IngestionReport(indexed_chunks=len(documents), skipped=False)
