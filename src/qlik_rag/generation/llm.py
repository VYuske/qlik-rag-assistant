"""Construção do modelo de chat local via Ollama."""

from __future__ import annotations

from langchain_ollama import ChatOllama

from qlik_rag.config.settings import Settings


def build_chat_model(settings: Settings) -> ChatOllama:
    """Cria o modelo de chat com temperatura e janela de contexto configuradas."""
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        num_ctx=settings.llm_num_ctx,
    )
