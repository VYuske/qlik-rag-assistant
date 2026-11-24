"""Fixtures compartilhadas da suíte de testes."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from qlik_rag.config.settings import Settings


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    for variable in (
        "OLLAMA_BASE_URL",
        "EMBEDDING_MODEL",
        "LLM_MODEL",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "RETRIEVAL_K",
        "CHROMA_PERSIST_DIR",
        "COLLECTION_NAME",
        "DATA_RAW_DIR",
        "PDF_GLOB",
    ):
        monkeypatch.delenv(variable, raising=False)

    return Settings(
        _env_file=None,
        data_raw_dir=tmp_path / "raw",
        chroma_persist_dir=tmp_path / "chroma",
    )


@pytest.fixture
def page_factory():
    def factory(content: str, **metadata: object) -> Document:
        return Document(page_content=content, metadata=dict(metadata))

    return factory
