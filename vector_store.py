import os
import time
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.helpers import PerformanceTimer
from config import (
    DOCS_DIR, 
    VECTOR_STORE_PATH, 
    LLM_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVER_K,
    RETRIEVER_FETCH_K,
    RETRIEVER_LAMBDA_MULT
)

class VectorStoreService:
    """Serviço para gerenciamento do índice vetorial."""
    
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=LLM_MODEL)
        self.vector_store = None
        
    def carregar_ou_criar_indice(self):
        """Carrega o índice FAISS existente ou cria um novo."""
        with PerformanceTimer("Inicialização do índice vetorial"):
            if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
                print("Carregando índice vetorial existente...")
                self.vector_store = FAISS.load_local(VECTOR_STORE_PATH, self.embeddings)
                return self.vector_store
            
            print("Criando novo índice vetorial...")
            self.vector_store = self._criar_novo_indice()
            return self.vector_store
    
    def _criar_novo_indice(self):
        """Cria um novo índice vetorial a partir dos documentos."""
        # Carrega os documentos
        loader = DirectoryLoader(
            DOCS_DIR,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader
        )
        
        print("Carregando documentos...")
        docs = loader.load()
        print(f"Carregados {len(docs)} documentos")
        
        # Dividir documentos em chunks menores para melhor recuperação
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        print("Dividindo documentos em chunks...")
        chunks = text_splitter.split_documents(docs)
        print(f"Criados {len(chunks)} chunks de texto")
        
        # Criação de embeddings e índice
        print("Criando embeddings e índice vetorial...")
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        # Salvar o índice para uso futuro
        print("Salvando índice vetorial...")
        vector_store.save_local(VECTOR_STORE_PATH)
        
        return vector_store
    
    def get_retriever(self):
        """Retorna o retriever configurado para uso."""
        if not self.vector_store:
            self.carregar_ou_criar_indice()
            
        return self.vector_store.as_retriever(
            search_kwargs={
                "k": RETRIEVER_K,
                "fetch_k": RETRIEVER_FETCH_K,
                "lambda_mult": RETRIEVER_LAMBDA_MULT
            }
        )
    
    def atualizar_indice(self):
        """Força a atualização do índice vetorial."""
        # Remover o índice existente
        if os.path.exists(VECTOR_STORE_PATH):
            import shutil
            shutil.rmtree(VECTOR_STORE_PATH)
        
        # Recriar o índice
        self.vector_store = self._criar_novo_indice()
        return self.vector_store