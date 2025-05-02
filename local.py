from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import BaseModel
from typing import List, Tuple, Any
import os
import time
import torch
import sys

# Verificar disponibilidade da GPU
has_cuda = torch.cuda.is_available()
if has_cuda:
    print(f"GPU disponível: {torch.cuda.get_device_name(0)}")
    print(f"Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Modelos Pydantic
class PerguntaInput(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]]

class RespostaOutput(BaseModel):
    answer: str
    source_documents: Any

# Configurações
PASTA_DOS_MD = r"C:\Users\Gabriel Lopes\Documents\cofrinho\100. Recursos\RAG\Agente Essencialista"
MODELO = "llama3.1:8b-instruct-Q4_K_M"  # Modelo quantizado para desempenho em GPU
CACHE_DIR = os.path.join(os.path.dirname(PASTA_DOS_MD), "cache")
VECTOR_STORE_PATH = os.path.join(CACHE_DIR, "faiss_index")

# Criar diretório de cache se não existir
os.makedirs(CACHE_DIR, exist_ok=True)

def carregar_ou_criar_indice():
    """Carrega o índice FAISS existente ou cria um novo."""
    inicio = time.time()
    
    if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
        print("Carregando índice vetorial existente...")
        embeddings = OllamaEmbeddings(model=MODELO)
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings)
        print(f"Índice carregado em {time.time() - inicio:.2f} segundos")
        return vector_store
    
    print("Criando novo índice vetorial...")
    
    # Carrega os documentos
    loader = DirectoryLoader(
        PASTA_DOS_MD,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader
    )
    
    print("Carregando documentos...")
    docs = loader.load()
    print(f"Carregados {len(docs)} documentos")
    
    # Dividir documentos em chunks menores para melhor recuperação
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    print("Dividindo documentos em chunks...")
    chunks = text_splitter.split_documents(docs)
    print(f"Criados {len(chunks)} chunks de texto")
    
    # Criação de embeddings em lotes para economizar memória
    print("Criando embeddings...")
    embeddings = OllamaEmbeddings(model=MODELO)
    
    # Criar índice vetorial
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Salvar o índice para uso futuro
    print("Salvando índice vetorial...")
    vector_store.save_local(VECTOR_STORE_PATH)
    
    print(f"Índice criado e salvo em {time.time() - inicio:.2f} segundos")
    return vector_store

def formatar_fontes(source_docs):
    """Formata as fontes de documentos para exibição."""
    if not source_docs:
        return "Nenhuma fonte específica."
    
    fontes = []
    for i, doc in enumerate(source_docs):
        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
            fonte = doc.metadata['source']
            # Extrair apenas o nome do arquivo
            nome_arquivo = os.path.basename(fonte)
            fontes.append(f"{i+1}. {nome_arquivo}")
    
    return "\n".join(fontes)

def main():
    # Inicializar o índice vetorial
    vector_store = carregar_ou_criar_indice()
    
    # Template do prompt aprimorado
    template = """Você é Jarvis1, um assistente virtual especializado em ajudar com a organização 
    e otimização da rotina diária usando princípios essencialistas.
    
    Responda apenas com base no contexto fornecido abaixo e no histórico da conversa.
    Se a informação não estiver no contexto, diga que não sabe a resposta com base nos 
    documentos disponíveis.
    
    Seja claro, objetivo e amigável. Responda sempre em português do Brasil.
    
    Contexto do documento:
    {context}
    
    Histórico de conversa:
    {chat_history}
    
    Pergunta: {question}
    
    Resposta:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "chat_history", "question"]
    )
    
    # Configurações otimizadas para o modelo com streaming habilitado
    llm = ChatOllama(
        model=MODELO,
        temperature=0.1,      # Temperatura mais baixa para respostas mais determinísticas
        top_p=0.9,            # Limita a diversidade para respostas mais focadas
        num_ctx=4096,         # Ajuste do contexto para equilibrar desempenho e memória
        callbacks=[StreamingStdOutCallbackHandler()] # Habilita streaming para o terminal
    )
    
    # Cria a cadeia conversacional com o retriever otimizado
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_kwargs={
                "k": 3,                # Número de documentos recuperados
                "fetch_k": 5,          # Busca mais documentos e filtra os melhores
                "lambda_mult": 0.5     # Parâmetro de diversidade na busca
            }
        ),
        return_source_documents=True,
        verbose=False
    )
    
    # Interação com o usuário
    print("\n==== Jarvis1: Assistente Essencialista ====")
    print("Converse com o agente (digite 'sair' para encerrar):")
    chat_history = []
    
    try:
        while True:
            print('\n')
            query = input("Você: ")
            
            if query.lower() in ["sair", "exit", "quit"]:
                break
                
            if not query.strip():
                continue
                
            # Exibir indicador de processamento
            print("Processando contexto...", end="\r")
            
            inicio = time.time()
            
            # Valida entrada com Pydantic e converte com model_dump
            entrada = PerguntaInput(question=query, chat_history=chat_history)
            entrada_dict = entrada.model_dump()
            
            # Preparar para capturar a resposta em streaming
            print("\nJarvis1: ", end="")
            sys.stdout.flush()  # Garante que o texto seja exibido imediatamente
            
            # Usa invoke() da cadeia
            result_raw = qa_chain.invoke(entrada_dict)
            
            # Valida e extrai saída com Pydantic
            resposta = RespostaOutput(**result_raw)
            
            # Atualiza o histórico com a resposta completa
            chat_history.append((query, resposta.answer))
            
            # Limitar tamanho do histórico para economizar memória
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
            
            # Tempo de processamento
            tempo = time.time() - inicio
            
            # Exibe fontes e tempo (depois da resposta gerada via streaming)
            print('\n')
            print(f"\nFontes consultadas:\n{formatar_fontes(resposta.source_documents)}")
            print(f"Tempo de resposta: {tempo:.2f} segundos")
    
    except KeyboardInterrupt:
        print("\nEncerrando o programa...")
    except Exception as e:
        print(f"\nErro: {e}")
    finally:
        print("\nObrigado por usar o Jarvis1!")

if __name__ == "__main__":
    main()