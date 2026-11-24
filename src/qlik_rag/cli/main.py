"""Entrypoint de linha de comando: `stats`, `ingest`, `query` e `ask`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence

from qlik_rag.config.settings import Settings
from qlik_rag.generation.graph import build_rag_graph
from qlik_rag.generation.llm import build_chat_model
from qlik_rag.ingestion.ingestor import MarkdownIngestor
from qlik_rag.ingestion.loaders import build_pdf_loader
from qlik_rag.ingestion.splitters import build_chunk_splitter, build_header_splitter
from qlik_rag.logging_setup import configure_logging
from qlik_rag.retrieval.vector_store import VectorStoreService
from qlik_rag.services.chat_service import ChatService
from qlik_rag.services.ingestion_service import IngestionService
from qlik_rag.services.search_service import SearchService

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qlik-rag",
        description="Assistente de RAG local sobre a documentação do Qlik.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nível de logging.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("stats", help="Mostra a quantidade de chunks indexados.")

    ingest_parser = subparsers.add_parser("ingest", help="Indexa os PDFs de origem.")
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="Recria o índice mesmo que já exista conteúdo.",
    )

    query_parser = subparsers.add_parser("query", help="Busca chunks relevantes.")
    query_parser.add_argument("text", help="Consulta em linguagem natural.")
    query_parser.add_argument(
        "-k",
        type=int,
        default=None,
        help="Quantidade de chunks retornados (padrão: RETRIEVAL_K).",
    )

    ask_parser = subparsers.add_parser("ask", help="Responde uma pergunta com streaming.")
    ask_parser.add_argument("text", help="Pergunta em linguagem natural.")

    return parser


def _build_ingestor(settings: Settings) -> MarkdownIngestor:
    pdf_paths = settings.resolved_pdf_paths()
    if not pdf_paths:
        raise FileNotFoundError(f"Nenhum PDF encontrado em '{settings.data_raw_dir}'.")
    return MarkdownIngestor(
        pdf_loader=build_pdf_loader(pdf_paths),
        header_splitter=build_header_splitter(settings),
        chunk_splitter=build_chunk_splitter(settings),
    )


def _run_stats(settings: Settings) -> int:
    count = VectorStoreService(settings).count()
    print(json.dumps({"collection": settings.collection_name, "chunks": count}))
    return 0


def _run_ingest(settings: Settings, force: bool) -> int:
    settings.ensure_dirs()
    service = IngestionService(
        settings=settings,
        ingestor=_build_ingestor(settings),
        vector_store=VectorStoreService(settings),
    )
    report = service.ensure_indexed(force=force)
    print(
        json.dumps(
            {"indexed_chunks": report.indexed_chunks, "skipped": report.skipped}
        )
    )
    return 0


def _run_query(settings: Settings, text: str, k: int | None) -> int:
    documents = SearchService(VectorStoreService(settings)).search(text, k=k)
    payload = [
        {"content": document.page_content, "metadata": document.metadata}
        for document in documents
    ]
    print(json.dumps({"query": text, "results": payload}, ensure_ascii=False, indent=2))
    return 0


def build_chat_service(settings: Settings) -> ChatService:
    """Monta o serviço de chat ligado ao índice e ao modelo configurados."""
    search = SearchService(VectorStoreService(settings))
    graph = build_rag_graph(
        settings=settings,
        retrieve=lambda question: search.search(question),
        llm=build_chat_model(settings),
    )
    return ChatService(graph=graph)


def _run_ask(settings: Settings, text: str) -> int:
    sources: list = []
    for event in build_chat_service(settings).stream_answer(text):
        if event.kind == "sources":
            sources = event.documents
        elif event.kind == "token":
            print(event.text, end="", flush=True)
    print()
    if sources:
        print("\nFontes:")
        for document in sources:
            source = document.metadata.get("source", "desconhecido")
            page = document.metadata.get("page")
            location = f"{source}, pág. {page}" if page is not None else source
            print(f"- {location}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Ponto de entrada da CLI; retorna o código de saída do processo."""
    args = _build_parser().parse_args(argv)
    configure_logging(args.log_level)
    settings = Settings()

    if args.command == "stats":
        return _run_stats(settings)
    if args.command == "ingest":
        return _run_ingest(settings, force=args.force)
    if args.command == "ask":
        return _run_ask(settings, text=args.text)
    return _run_query(settings, text=args.text, k=args.k)


if __name__ == "__main__":
    sys.exit(main())
