
import sys
from pathlib import Path

# Garante que a pasta raiz está no PATH para evitar erros de importação do pacote 'src'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings, Prompts
from langchain_ollama import OllamaEmbeddings
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
from langchain_text_splitters import MarkdownTextSplitter, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src import ChromaRetrieverFactory
from langchain_chroma import Chroma

settings = Settings()
pdf_filename = "Explore, discover and analyze.pdf"
pdf_filename = "Script syntax and chart functions.pdf"
pdf_path = Path(settings.DATA_RAW_DIR) / pdf_filename


embeddings=OllamaEmbeddings(model=settings.EMBEDDING_MODEL)
pdf_loader=OpenDataLoaderPDFLoader(file_path=[str(pdf_path)], format='markdown', split_pages=False)
text_splitter = MarkdownTextSplitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)

headers_to_split_on = [    
    ("###", "Header 1"),
    ("####", "Header 2"),
    ("#####", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)

pages = pdf_loader.load()

output = []
for i, page in enumerate(pages):
    metadata = page.metadata
    documents = markdown_splitter.split_text(page.page_content)
    documents = text_splitter.split_documents(documents=documents)

    for document in documents:
        metadata.update(document.metadata)        
        
        output.append(Document(page_content=document.page_content, metadata=metadata))


vectorstore = Chroma.from_documents(
    documents=output,
    embedding=embeddings,
    persist_directory=settings.CHROMA_STORAGE_PATH,
    collection_name=settings.VECTOR_COLLECTION_NAME
)
# Executa a factory (Se for a 1ª vez, lê o PDF, divide e cria o ChromaDB. Se já existir, pula direto para o retriever)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

print("\n🔍 Testando busca no banco vetorial recém-configurado...")
query = 'CROSSTABLE STATEMENT'
docs = retriever.invoke(query)

print(f"\n📄 Resultados encontrados para '{query}': {len(docs)} chunks.")
for i, doc in enumerate(docs, 1):
    print(f"\n--- Chunk {i} ---")
    print(f"Fonte: {doc.metadata.get('source')} | Página: {doc.metadata.get('page')}")
    print(doc.page_content)