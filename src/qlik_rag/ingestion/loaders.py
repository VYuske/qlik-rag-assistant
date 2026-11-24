"""Construção do loader de PDF que converte documentos em markdown."""

from __future__ import annotations

from pathlib import Path

from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader


def build_pdf_loader(paths: list[Path]) -> OpenDataLoaderPDFLoader:
    """Cria um loader de PDF em markdown para os caminhos informados."""
    return OpenDataLoaderPDFLoader(
        file_path=[str(path) for path in paths],
        format="markdown",
        split_pages=True,
    )
