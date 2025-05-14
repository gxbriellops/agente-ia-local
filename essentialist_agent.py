import json
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timedelta
from utils.helpers import formatar_fontes, PerformanceTimer
from models.schemas import AgentAction, CalendarEventCreate
from services.vector_store import VectorStoreService
from services.llm_service import LLMService
from services.calendar_service import GoogleCalendarService

class EssentialistAgent:
    """Agente principal que integra RAG e Google Calendar."""
    
    def __init__(self):
        # Inicializar serviços
        self.vector_store_service = VectorStoreService()
        retriever = self.vector_store_service.carregar_ou_criar_indice().get_retriever()
        self.llm_service = LLMService(retriever)
        self.calendar_service = GoogleCalendarService()
        
        # Estado do agente
        self.chat_history = []
    
    def processar_entrada(self, pergunta: str) -> Dict[str, Any]:
        """Processa a entrada do usuário e retorna uma resposta."""
        with PerformanceTimer("Processamento da resposta"):
            # Obter contexto da agenda para enriquecer a resposta
            info_calendario = self._obter_info_calendario()
            
            # Adicionar informações do calendário à pergunta
            pergunta_enriquecida = f"{pergunta}\n\nInformações do calendário: {info_calendario}"
            
            # Obter resposta do modelo
            resposta = self.llm_service.processar_pergunta(pergunta_enriquecida, self.chat_history)
            
            # Extrair possíveis ações de calendário da resposta
            acao = self.llm_service.extrair_acao(resposta.answer)
            resultado_acao = None
            
            # Executar a ação se existir
            if acao:
                resultado_acao = self._executar_acao(acao)
                
            # Atualizar histórico de conversa
            self.chat_history.append((pergunta, resposta.answer))
            
            # Limitar tamanho do histórico para economizar memória
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]
            
            # Montar resultado
            return {
                "resposta": resposta.answer,
                "fontes": formatar_fontes(resposta.source_documents),
                "acao_realizada": resultado_acao,
                "historico_atualizado": len(self.chat_history)
            }
    
    def _obter_info_calendario(self) -> str:
        """Obtém informações recentes do calendário para contexto."""
        try:
            # Obter eventos dos próximos 3 dias
            eventos = self.calendar_service.listar_eventos(dias=3)
            return self.calendar_service.formatar_eventos(eventos)
        except Exception as e:
            print(f"Erro ao obter informações do calendário: {e}")
            return "Não foi possível obter informações do calendário."
    
    def _executar_acao(self, acao: AgentAction) -> Dict[str, Any]:
        """Executa uma ação no calendário com base na instrução do agente."""
        resultado = {"sucesso": False, "mensagem": "Ação não reconhecida", "dados": None}
        
        try:
            # Ações de listagem
            if acao.action_type == "listar_eventos":
                dias = acao.params.get("dias", 7)
                eventos = self.calendar_service.listar_eventos(dias)
                eventos_formatados = self.calendar_service.formatar_eventos(eventos)
                resultado = {
                    "sucesso": True,
                    "mensagem": f"Encontrados {len(eventos)} eventos para os próximos {dias} dias.",
                    "dados": eventos_formatados
                }
            
            # Busca de eventos
            elif acao.action_type == "buscar_evento":
                query = acao.params.get("query", "")
                eventos = self.calendar_service.buscar_evento(query)
                eventos_formatados = self.calendar_service.formatar_eventos(eventos)
                resultado = {
                    "sucesso": True,
                    "mensagem": f"Encontrados {len(eventos)} eventos para a busca '{query}'.",
                    "dados": eventos_formatados
                }
            
            # Criação de evento
            elif acao.action_type == "criar_evento":
                # Converter strings de data/hora para objetos datetime
                start_str = acao.params.get("start", "")
                end_str = acao.params.get("end", "")
                
                start = datetime.fromisoformat(start_str) if start_str else None
                end = datetime.fromisoformat(end_str) if end_str else None
                
                if not start or not end:
                    return {
                        "sucesso": False,
                        "mensagem": "Datas de início e fim são obrigatórias para criar um evento."
                    }
                
                # Criar objeto de evento
                evento = CalendarEventCreate(
                    summary=acao.params.get("summary", "Evento sem título"),
                    description=acao.params.get("description", ""),
                    location=acao.params.get("location", ""),
                    start=start,
                    end=end,
                    attendees=acao.params.get("attendees", []),
                    reminders=acao.params.get("reminders", None)
                )
                
                # Criar evento no calendário
                evento_criado = self.calendar_service.criar_evento(evento)
                if evento_criado:
                    resultado = {
                        "sucesso": True,
                        "mensagem": f"Evento '{evento.summary}' criado com sucesso.",
                        "dados": evento_criado
                    }
                else:
                    resultado = {
                        "sucesso": False,
                        "mensagem": "Falha ao criar o evento."
                    }
            
            # Atualização de evento
            elif acao.action_type == "atualizar_evento":
                event_id = acao.params.get("event_id", "")
                if not event_id:
                    return {
                        "sucesso": False,
                        "mensagem": "ID do evento é obrigatório para atualização."
                    }
                
                # Converter strings de data/hora para objetos datetime
                start_str = acao.params.get("start", "")
                end_str = acao.params.get("end", "")
                
                start = datetime.fromisoformat(start_str) if start_str else None
                end = datetime.fromisoformat(end_str) if end_str else None
                
                if not start or not end:
                    return {
                        "sucesso": False,
                        "mensagem": "Datas de início e fim são obrigatórias para atualizar um evento."
                    }
                
                # Criar objeto de evento
                evento = CalendarEventCreate(
                    summary=acao.params.get("summary", "Evento sem título"),
                    description=acao.params.get("description", ""),
                    location=acao.params.get("location", ""),
                    start=start,
                    end=end,
                    attendees=acao.params.get("attendees", []),
                    reminders=acao.params.get("reminders", None)
                )
                
                # Atualizar evento no calendário
                evento_atualizado = self.calendar_service.atualizar_evento(event_id, evento)
                if evento_atualizado:
                    resultado = {
                        "sucesso": True,
                        "mensagem": f"Evento '{evento.summary}' atualizado com sucesso.",
                        "dados": evento_atualizado
                    }
                else:
                    resultado = {
                        "sucesso": False,
                        "mensagem": "Falha ao atualizar o evento."
                    }
            
            # Exclusão de evento
            elif acao.action_type == "excluir_evento":
                event_id = acao.params.get("event_id", "")
                if not event_id:
                    return {
                        "sucesso": False,
                        "mensagem": "ID do evento é obrigatório para exclusão."
                    }
                
                # Excluir evento do calendário
                exclusao_sucesso = self.calendar_service.excluir_evento(event_id)
                if exclusao_sucesso:
                    resultado = {
                        "sucesso": True,
                        "mensagem": "Evento excluído com sucesso."
                    }
                else:
                    resultado = {
                        "sucesso": False,
                        "mensagem": "Falha ao excluir o evento."
                    }
            
            # Análise de tempo livre
            elif acao.action_type == "analisar_tempo_livre":
                # Parâmetros para análise
                dias = acao.params.get("dias", 7)
                duracao_minima = acao.params.get("duracao_minima", 30)  # em minutos
                
                # Calcular intervalo de tempo
                agora = datetime.now()
                fim = agora + timedelta(days=dias)
                
                # Obter períodos livres
                periodos_livres = self.calendar_service.analisar_tempo_livre(
                    agora, fim, duracao_minima)
                
                # Formatar resultado
                if periodos_livres:
                    periodos_formatados = []
                    for i, periodo in enumerate(periodos_livres, 1):
                        inicio = periodo['start'].strftime("%d/%m/%Y %H:%M")
                        fim = periodo['end'].strftime("%d/%m/%Y %H:%M")
                        duracao = (periodo['end'] - periodo['start']).total_seconds() / 60
                        periodos_formatados.append(
                            f"Período {i}: {inicio} até {fim} (duração: {duracao:.0f} minutos)")
                    
                    resultado = {
                        "sucesso": True,
                        "mensagem": f"Encontrados {len(periodos_livres)} períodos livres.",
                        "dados": "\n".join(periodos_formatados)
                    }
                else:
                    resultado = {
                        "sucesso": True,
                        "mensagem": "Não foram encontrados períodos livres com a duração mínima especificada.",
                        "dados": []
                    }
        
        except Exception as e:
            resultado = {
                "sucesso": False,
                "mensagem": f"Erro ao executar ação: {str(e)}"
            }
        
        return resultado