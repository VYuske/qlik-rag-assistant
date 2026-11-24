"""Configurações da aplicação, carregadas de variáveis de ambiente e do arquivo .env."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_HEADERS_TO_SPLIT_ON: list[tuple[str, str]] = [
    ("###", "Header 1"),
    ("####", "Header 2"),
    ("#####", "Header 3"),
]


class Settings(BaseSettings):
    """Configuração tipada do assistente de RAG.

    Os caminhos são relativos ao diretório de execução, de modo que o mesmo
    código funciona no repositório e dentro do container.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_raw_dir: Path = Path("data/raw/qlik")
    chroma_persist_dir: Path = Path("data/vectorstore/chroma")
    collection_name: str = "qlik_documentation"
    pdf_glob: str = "*.pdf"

    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "qwen3-embedding:0.6b"
    llm_model: str = "qwen3.5:4b"
    llm_temperature: float = 0.0
    llm_num_ctx: int = 8192

    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_k: int = 5
    max_retrieval_loops: int = 1
    streamlit_port: int = 8501

    headers_to_split_on: list[tuple[str, str]] = DEFAULT_HEADERS_TO_SPLIT_ON

    def resolved_pdf_paths(self) -> list[Path]:
        """Retorna, em ordem estável, todos os PDFs encontrados em `data_raw_dir`."""
        return sorted(self.data_raw_dir.glob(self.pdf_glob))

    def ensure_dirs(self) -> None:
        """Cria os diretórios de dados e de persistência do índice, se necessário."""
        self.data_raw_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
