#!/usr/bin/env python
# scripts/ingest.py

import sys
from pathlib import Path

# Garante que a pasta raiz está no PATH para evitar erros de importação do pacote 'src'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings, Prompts
from langchain_ollama import OllamaEmbeddings
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
from langchain_text_splitters import MarkdownTextSplitter
from src import ChromaRetrieverFactory

def main():
    settings = Settings()
    print("=" * 60)
    print("🚀 INICIANDO PIPELINE DE INGESTÃO - QLIK RAG ASSISTANT")
    print("=" * 60)
    print(f"📂 Pasta de Origem (PDFs): {settings.DATA_RAW_DIR}")
    print(f"📦 Destino do Banco (Chroma): {settings.CHROMA_STORAGE_PATH}")
    print(f"🧠 Modelo de Embeddings: {settings.EMBEDDING_MODEL}")
    print("-" * 60)

    # Constrói o caminho do PDF dinamicamente usando as configurações do projeto
    pdf_filename = "Script syntax and chart functions.pdf"
    pdf_path = Path(settings.DATA_RAW_DIR) / pdf_filename
    
    # Inicialização elegante e modularizada
    retriever_factory = ChromaRetrieverFactory(
        embeddings=OllamaEmbeddings(model=settings.EMBEDDING_MODEL),
        pdf_loader=OpenDataLoaderPDFLoader(file_path=[str(pdf_path)], format='markdown'),
        text_splitter=MarkdownTextSplitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP),
        persist_directory=settings.CHROMA_STORAGE_PATH,
        collection_name=settings.VECTOR_COLLECTION_NAME
    )

    # Executa a factory (Se for a 1ª vez, lê o PDF, divide e cria o ChromaDB. Se já existir, pula direto para o retriever)
    retriever = retriever_factory.get_retriever(search_k=5)

    print("\n🔍 Testando busca no banco vetorial recém-configurado...")
    query = 'HOW TO BUILD A CROSSTABLE STATEMENT'
    docs = retriever.invoke(query)
    
    print(f"\n📄 Resultados encontrados para '{query}': {len(docs)} chunks.")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"metadata: {doc.metadata}")
        print(doc.page_content)


if __name__ == "__main__":
    main()
