"""Configuração de logging compartilhada pela aplicação."""

from __future__ import annotations

import logging

DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configura o logging raiz com o nível informado."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=DEFAULT_FORMAT,
    )
