import os
from pathlib import Path
from langchain_chroma import Chroma

class ChromaRetrieverFactory:
    """Gerencia o ciclo de vida e a conexão do banco de dados vetorial Chroma de forma eficiente."""
    
    def __init__(self, embeddings, pdf_loader, text_splitter, persist_directory: str, collection_name: str):
        self.embeddings = embeddings
        self.pdf_loader = pdf_loader
        self.text_splitter = text_splitter
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._vectorstore = None  # Cache da instância na memória (Singleton temporário)

    def _is_db_initialized(self) -> bool:
        """Verifica se o banco já existe fisicamente no disco e se não está vazio."""
        return self.persist_directory.exists() and any(self.persist_directory.iterdir())

    def _initialize_vectorstore(self):
        """Inicializa o ChromaDB lendo o PDF apenas se necessário."""
        if self._vectorstore is not None:
            return

        if self._is_db_initialized():
            print("🔄 Conectando ao ChromaDB existente no disco...")
            self._vectorstore = Chroma(
                persist_directory=str(self.persist_directory),
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            return

        print("🆕 Banco não encontrado. Iniciando extração e indexação do PDF...")
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Carrega as páginas do PDF
            pages = self.pdf_loader.load()
            print(f'📄 PDF carregado com sucesso: {len(pages)} páginas.')
            
            # 2. Divide em chunks (Correção do parâmetro 'documents')
            chunks = self.text_splitter.split_documents(documents=pages)
            print(f'✂️ Texto dividido em {len(chunks)} chunks.')
            
            # 3. Cria e popula o banco vetorial
            self._vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory),
                collection_name=self.collection_name
            )
            print("✅ Novo banco ChromaDB criado e persistido!")
            
        except Exception as e:
            print(f"❌ Erro ao configurar o ChromaDB: {str(e)}")
            raise

    def get_retriever(self, search_k: int = 5):
        """Retorna o componente retriever pronto para uso pelo LangGraph."""
        self._initialize_vectorstore()
        
        return self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": search_k}
        )
