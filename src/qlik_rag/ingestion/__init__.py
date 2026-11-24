"""Módulo de ingestão: carregamento de PDFs, divisão em chunks e orquestração."""

from qlik_rag.ingestion.ingestor import MarkdownIngestor
from qlik_rag.ingestion.loaders import build_pdf_loader
from qlik_rag.ingestion.splitters import build_chunk_splitter, build_header_splitter

__all__ = [
    "MarkdownIngestor",
    "build_chunk_splitter",
    "build_header_splitter",
    "build_pdf_loader",
]
