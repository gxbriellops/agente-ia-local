from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import List, Tuple, Optional, Dict, Any
from models.schemas import PerguntaInput, RespostaOutput, AgentAction
import json
from config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_NUM_CTX
)

class LLMService:
    """Serviço para gerenciamento do modelo de linguagem."""
    
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = self._inicializar_llm()
        self.qa_chain = self._criar_qa_chain()
        
    def _inicializar_llm(self):
        """Inicializa o modelo de linguagem com as configurações apropriadas."""
        return ChatOllama(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            num_ctx=LLM_NUM_CTX,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
    
    def _criar_qa_chain(self):
        """Cria a cadeia de conversação com o retriever."""
        prompt_template = """Você é Jarvis1, um assistente virtual especializado em ajudar com a organização 
        e otimização da rotina diária usando princípios essencialistas.
        
        Responda apenas com base no contexto fornecido abaixo, no histórico da conversa e nas informações do calendário.
        Se a informação não estiver no contexto, diga que não sabe a resposta com base nos documentos disponíveis.
        
        Você tem acesso à agenda do usuário e pode realizar ações no calendário como:
        1. Listar eventos futuros
        2. Criar novos eventos
        3. Atualizar eventos existentes
        4. Excluir eventos
        5. Analisar tempo livre
        
        Para realizar estas ações, devolva um objeto JSON com a estrutura adequada.
        
        Seja claro, objetivo e amigável. Responda sempre em português do Brasil.
        
        Contexto do documento:
        {context}
        
        Histórico de conversa:
        {chat_history}
        
        Pergunta: {question}
        
        Resposta:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,
            verbose=False
        )
    
    def processar_pergunta(self, pergunta: str, historico: List[Tuple[str, str]]) -> RespostaOutput:
        """Processa uma pergunta e retorna a resposta com fontes."""
        entrada = PerguntaInput(question=pergunta, chat_history=historico)
        entrada_dict = entrada.model_dump()
        
        result_raw = self.qa_chain.invoke(entrada_dict)
        resposta = RespostaOutput(**result_raw)
        
        return resposta
    
    def extrair_acao(self, texto_resposta: str) -> Optional[AgentAction]:
        """Extrai uma possível ação de calendário da resposta do modelo."""
        # Procura por marcações JSON na resposta
        try:
            # Tentar encontrar JSON entre delimitadores
            import re
            json_match = re.search(r'\{.*?\}', texto_resposta, re.DOTALL)
            if json_match:
                acao_json = json_match.group(0)
                acao_dict = json.loads(acao_json)
                return AgentAction(**acao_dict)
        except Exception as e:
            print(f"Erro ao extrair ação: {e}")
        
        return None