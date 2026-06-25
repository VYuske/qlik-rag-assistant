"""
Qlik RAG Assistant - Código Fonte Principal.
Desenvolvido seguindo princípios de Clean Architecture e processamento local.
"""

__version__ = "0.1.0"
__author__ = "Seu Nome"

# Força a validação inicial do ambiente assim que o pacote src for tocado
from src.config import Settings, Prompts
from src.retriever import ChromaRetrieverFactory