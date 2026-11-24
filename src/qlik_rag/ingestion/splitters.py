"""Construção dos splitters de markdown usados na ingestão."""

from __future__ import annotations

from langchain_text_splitters import MarkdownHeaderTextSplitter, MarkdownTextSplitter

from qlik_rag.config.settings import Settings


def build_header_splitter(settings: Settings) -> MarkdownHeaderTextSplitter:
    """Cria o splitter que separa o markdown pelos cabeçalhos configurados."""
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=settings.headers_to_split_on,
        strip_headers=False,
    )


def build_chunk_splitter(settings: Settings) -> MarkdownTextSplitter:
    """Cria o splitter que limita cada chunk ao tamanho configurado."""
    return MarkdownTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
