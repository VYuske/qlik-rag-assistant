"""Testes das configurações carregadas por ambiente."""

from __future__ import annotations

from pathlib import Path

from qlik_rag.config.settings import Settings


def test_defaults_when_no_env_is_set(tmp_path: Path, monkeypatch) -> None:
    for variable in (
        "OLLAMA_BASE_URL",
        "EMBEDDING_MODEL",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "RETRIEVAL_K",
        "COLLECTION_NAME",
        "DATA_RAW_DIR",
        "CHROMA_PERSIST_DIR",
    ):
        monkeypatch.delenv(variable, raising=False)

    settings = Settings(_env_file=None, chroma_persist_dir=tmp_path / "chroma")

    assert settings.collection_name == "qlik_documentation"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 150
    assert settings.retrieval_k == 5
    assert settings.embedding_model == "qwen3-embedding:0.6b"


def test_env_override_wins_over_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "42")
    monkeypatch.setenv("RETRIEVAL_K", "9")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")

    settings = Settings(_env_file=None, chroma_persist_dir=tmp_path / "chroma")

    assert settings.chunk_size == 42
    assert settings.retrieval_k == 9
    assert settings.ollama_base_url == "http://ollama:11434"


def test_resolved_pdf_paths_returns_every_pdf(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "b.pdf").write_text("b")
    (raw_dir / "a.pdf").write_text("a")
    (raw_dir / "ignore.txt").write_text("x")

    settings = Settings(
        _env_file=None,
        data_raw_dir=raw_dir,
        chroma_persist_dir=tmp_path / "chroma",
    )

    assert [path.name for path in settings.resolved_pdf_paths()] == ["a.pdf", "b.pdf"]


def test_ensure_dirs_creates_both_directories(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        data_raw_dir=tmp_path / "raw",
        chroma_persist_dir=tmp_path / "chroma",
    )

    settings.ensure_dirs()

    assert settings.data_raw_dir.is_dir()
    assert settings.chroma_persist_dir.is_dir()
