# src/vectorstore/vectorstore_factory.py
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

class VectorStoreFactory:
    """Gerencia o ciclo de vida e a conexão do banco de dados vetorial Chroma."""
    
    def __init__(self):

    def get_collection(self) -> Collection:
        """
        Retorna a coleção do Chroma. 
        Se ela já existir, apenas instancia. Caso contrário, inicia o fluxo de criação.
        """
        if self._collection is not None:
            return self._collection

        # Lista as coleções existentes no banco físico
        existing_collections = [c.name for c in self.client.list_collections()]

        if self.collection_name in existing_collections:
            print(f"📦 [ChromaDB] Coleção '{self.collection_name}' encontrada. Apenas instanciando...")
            self._collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn
            )
        else:
            print(f"🛠️ [ChromaDB] Coleção '{self.collection_name}' não encontrada.")
            print("🚀 Iniciando fluxo de criação e configuração de nova base vetorial...")
            self._collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"} # Otimizado para similaridade de cosseno técnico
            )
            
        return self._collection

    def reset_database(self):
        """Remove a coleção atual para permitir uma recriação limpa (usado em scripts/reset_db.py)."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self._collection = None
            print(f"🗑️ Coleção '{self.collection_name}' removida com sucesso.")
        except ValueError:
            print(f"⚠️ A coleção '{self.collection_name}' já não existia.")
