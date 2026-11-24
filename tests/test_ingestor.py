"""Testes do pipeline de ingestão de markdown."""

from __future__ import annotations

from langchain_core.documents import Document

from qlik_rag.config.settings import Settings
from qlik_rag.ingestion.ingestor import MarkdownIngestor
from qlik_rag.ingestion.splitters import build_chunk_splitter, build_header_splitter


class FakeLoader:
    def __init__(self, pages: list[Document]) -> None:
        self._pages = pages

    def load(self) -> list[Document]:
        return self._pages


def build_ingestor(settings: Settings, pages: list[Document]) -> MarkdownIngestor:
    return MarkdownIngestor(
        pdf_loader=FakeLoader(pages),
        header_splitter=build_header_splitter(settings),
        chunk_splitter=build_chunk_splitter(settings),
    )


def test_headers_do_not_leak_between_chunks_of_the_same_page(settings: Settings) -> None:
    page = Document(
        page_content="# Doc\n### First\ncontent one\n### Second\ncontent two\n",
        metadata={"source": "doc.pdf", "page": 1},
    )

    documents = build_ingestor(settings, [page]).ingest()

    by_content = {document.page_content.strip(): document.metadata for document in documents}
    first = next(meta for text, meta in by_content.items() if "content one" in text)
    second = next(meta for text, meta in by_content.items() if "content two" in text)

    assert first["Header 1"] == "First"
    assert second["Header 1"] == "Second"
    assert first is not second


def test_page_metadata_is_preserved_in_every_chunk(settings: Settings) -> None:
    pages = [
        Document(page_content="### A\nalpha", metadata={"source": "one.pdf", "page": 1}),
        Document(page_content="### B\nbeta", metadata={"source": "two.pdf", "page": 7}),
    ]

    documents = build_ingestor(settings, pages).ingest()

    assert {document.metadata["source"] for document in documents} == {"one.pdf", "two.pdf"}
    assert {document.metadata["page"] for document in documents} == {1, 7}


def test_metadata_from_page_is_not_mutated(settings: Settings) -> None:
    page = Document(
        page_content="### Alpha\ntext a\n### Beta\ntext b\n",
        metadata={"source": "doc.pdf"},
    )
    original_metadata = dict(page.metadata)

    build_ingestor(settings, [page]).ingest()

    assert page.metadata == original_metadata
