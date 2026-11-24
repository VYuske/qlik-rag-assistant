"""Pipeline de ingestão: carrega páginas e produz chunks com metadata isolada."""

from __future__ import annotations

import logging
from typing import Protocol

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentLoader(Protocol):
    """Contrato mínimo de um loader que devolve páginas como `Document`."""

    def load(self) -> list[Document]: ...


class TextSplitter(Protocol):
    """Contrato mínimo de um splitter de documentos."""

    def split_text(self, text: str) -> list[Document]: ...

    def split_documents(self, documents: list[Document]) -> list[Document]: ...


class MarkdownIngestor:
    """Converte páginas de PDF em chunks enriquecidos com metadata de cabeçalho."""

    def __init__(
        self,
        pdf_loader: DocumentLoader,
        header_splitter: TextSplitter,
        chunk_splitter: TextSplitter,
    ) -> None:
        self._pdf_loader = pdf_loader
        self._header_splitter = header_splitter
        self._chunk_splitter = chunk_splitter

    def _split_page(self, page: Document) -> list[Document]:
        sections = self._header_splitter.split_text(page.page_content)
        chunks = self._chunk_splitter.split_documents(sections)
        return [
            Document(
                page_content=chunk.page_content,
                metadata={**page.metadata, **chunk.metadata},
            )
            for chunk in chunks
        ]

    def ingest(self) -> list[Document]:
        """Carrega todas as páginas e retorna a lista de chunks resultante."""
        pages = self._pdf_loader.load()
        documents = [chunk for page in pages for chunk in self._split_page(page)]
        logger.info("Ingestão concluída: %d páginas, %d chunks.", len(pages), len(documents))
        return documents
