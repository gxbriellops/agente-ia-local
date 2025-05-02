from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from pydantic import BaseModel
from typing import List, Tuple, Any

# Modelos Pydantic
class PerguntaInput(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]]

class RespostaOutput(BaseModel):
    answer: str
    source_documents: Any

# Caminho para a pasta com arquivos Markdown
pasta_dos_md = r"C:\Users\Gabriel Lopes\Documents\cofrinho\100. Recursos\RAG\Agente Essencialista"

# Carrega os documentos
loader = DirectoryLoader(
    pasta_dos_md,
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader
)

docs = loader.load()

# Cria embeddings e índice vetorial
embeddings = OllamaEmbeddings(model="llama3.1:8b-instruct-Q4_K_M")
vector_store = FAISS.from_documents(docs, embeddings)

# Template do prompt
template = """Você é um assistente virtual e seu nome é Jarvis1, especializado em ajudar 
com a organização e otimização da rotina diária usando princípios essencialistas. 
Responda de forma clara, objetiva e amigável. Responda em português do Brasil.

Contexto do documento: {context}
Histórico de conversa: {chat_history}
Pergunta: {question}

Resposta:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "chat_history", "question"]
)

# Inicializar o modelo
llm = ChatOllama(model="llama3.1:8b-instruct-Q4_K_M")

# Cria a cadeia conversacional com o retriever
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    verbose=False
)

# Interação com o usuário
print("Converse com o agente (digite 'sair' para encerrar):")
chat_history = []

while True:
    print('\n')
    query = input("Você: ")
    
    if query.lower() in ["sair", "exit", "quit"]:
        break

    # Valida entrada com Pydantic e converte com model_dump
    entrada = PerguntaInput(question=query, chat_history=chat_history)
    entrada_dict = entrada.model_dump()

    # Usa invoke() da cadeia
    result_raw = qa_chain.invoke(entrada_dict)

    # Valida e extrai saída com Pydantic
    resposta = RespostaOutput(**result_raw)

    # Atualiza o histórico
    chat_history.append((query, resposta.answer))

    # Exibe resposta
    print('\n')
    print(resposta.answer)
