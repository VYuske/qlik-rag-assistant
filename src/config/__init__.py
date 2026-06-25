"""
Módulo de Configurações Globais e Prompts do Sistema.
"""

from src.config.settings import Settings
from src.config.prompts import Prompts
# Define o que será exportado ao usar "from src.config import *"
__all__ = [
    "Settings",
    "Prompts",
]