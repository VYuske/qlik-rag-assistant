# config.py
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class Settings:
    def __init__(self) -> None:
        # --- CAMINHOS DE DIRETÓRIOS ---
        self.ROOT_DIR: Path = PROJECT_ROOT
        self.DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw" / "qlik"
        self.DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed" / "chunks"
        
        self.CHROMA_STORAGE_PATH: str = os.getenv("CHROMA_STORAGE_PATH", str(PROJECT_ROOT / "data" / "vectorstore" / "chroma"))

        # --- CONFIGURAÇÕES DO OLLAMA / MODELOS ---
        self.OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3.5:4b")
        self.EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")

        # --- PARÂMETROS DO RAG / CHUNKING ---
        self.CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
        self.VECTOR_COLLECTION_NAME: str = "qlik_documentation"
        self.MAX_RETRIEVAL_LOOPS: int = 3

    def validate_environment(self) -> None:
        """Valida se o ambiente possui os requisitos mínimos configurados."""
        if not self.DATA_RAW_DIR.exists():
            self.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
