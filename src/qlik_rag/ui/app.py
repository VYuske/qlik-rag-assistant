"""Interface de chat do Qlik RAG Assistant."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
from langchain_core.documents import Document

from qlik_rag.config.settings import Settings
from qlik_rag.generation.graph import build_rag_graph
from qlik_rag.generation.llm import build_chat_model
from qlik_rag.ingestion.ingestor import MarkdownIngestor
from qlik_rag.ingestion.loaders import build_pdf_loader
from qlik_rag.ingestion.splitters import build_chunk_splitter, build_header_splitter
from qlik_rag.retrieval.vector_store import VectorStoreService
from qlik_rag.services.chat_service import ChatError, ChatService
from qlik_rag.services.ingestion_service import IngestionService
from qlik_rag.services.search_service import SearchService
from qlik_rag.ui.formatting import build_history, format_sources


@st.cache_resource
def get_settings() -> Settings:
    return Settings()


@st.cache_resource
def get_vector_store() -> VectorStoreService:
    return VectorStoreService(get_settings())


@st.cache_resource
def get_chat_service() -> ChatService:
    settings = get_settings()
    vector_store = get_vector_store()
    search = SearchService(vector_store)
    graph = build_rag_graph(
        settings=settings,
        retrieve=lambda question: search.search(question),
        llm=build_chat_model(settings),
    )
    return ChatService(graph=graph)


def build_ingestor(settings: Settings) -> MarkdownIngestor:
    return MarkdownIngestor(
        pdf_loader=build_pdf_loader(settings.resolved_pdf_paths()),
        header_splitter=build_header_splitter(settings),
        chunk_splitter=build_chunk_splitter(settings),
    )


def render_sources(documents: list[Document]) -> None:
    entries = format_sources(documents)
    if not entries:
        return
    with st.expander(f"Fontes ({len(entries)})"):
        for entry in entries:
            st.markdown(f"**{entry['label']}**")
            if entry["headers"]:
                st.caption(" › ".join(entry["headers"]))
            st.text(entry["content"][:500])


def render_sidebar() -> None:
    settings = get_settings()
    vector_store = get_vector_store()
    with st.sidebar:
        st.header("Base de conhecimento")
        chunks = vector_store.count()
        st.metric("Chunks indexados", chunks)
        st.caption(f"Coleção: `{settings.collection_name}`")
        st.caption(f"Modelo: `{settings.llm_model}`")
        st.caption(f"Embeddings: `{settings.embedding_model}`")
        if st.button("Reindexar PDFs", use_container_width=True):
            with st.spinner("Lendo PDFs e gerando embeddings..."):
                IngestionService(
                    settings=settings,
                    ingestor=build_ingestor(settings),
                    vector_store=vector_store,
                ).ensure_indexed(force=True)
            st.cache_resource.clear()
            st.rerun()
        if chunks == 0:
            st.warning("O índice está vazio. Reindexe antes de perguntar.")


def render_history() -> None:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main() -> None:
    st.set_page_config(page_title="Qlik RAG Assistant", page_icon="📘", layout="wide")
    st.title("Qlik RAG Assistant")
    st.caption("Respostas baseadas apenas na documentação indexada, com fontes.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    render_sidebar()
    render_history()

    question = st.chat_input("Pergunte sobre a documentação do Qlik...")
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = build_history(st.session_state["messages"][:-1])
    with st.chat_message("assistant"):
        sources_holder: list = []
        text_placeholder = st.empty()
        collected: list[str] = []
        try:
            for event in get_chat_service().stream_answer(question, history=history):
                if event.kind == "sources":
                    sources_holder = event.documents
                elif event.kind == "token":
                    collected.append(event.text)
                    text_placeholder.markdown("".join(collected) + "▌")
        except ChatError as exc:
            text_placeholder.empty()
            st.error(str(exc))
            st.session_state["messages"].pop()
            return
        answer = "".join(collected) or "Não foi possível gerar uma resposta."
        text_placeholder.markdown(answer)
        render_sources(sources_holder)

    st.session_state["messages"].append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
