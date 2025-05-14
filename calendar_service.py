import os
import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from models.schemas import CalendarEvent, CalendarEventCreate
from typing import List, Dict, Any, Optional
from utils.helpers import get_time_range, formatar_evento_calendario
from config import CALENDAR_SCOPES, TOKEN_FILE, CREDENTIALS_FILE
import json

class GoogleCalendarService:
    """Serviço para interação com a API do Google Calendar."""
    
    def __init__(self):
        self.service = None
        self.autenticar()
    
    def autenticar(self):
        """Autentica o serviço de calendário usando OAuth."""
        creds = None
        
        # Verifica se já existe um token armazenado
        if os.path.exists(TOKEN_FILE):
            creds = Credentials.from_authorized_user_info(json.loads(open(TOKEN_FILE).read()), CALENDAR_SCOPES)
        
        # Se não houver credenciais válidas, solicita autenticação
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    CREDENTIALS_FILE, CALENDAR_SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Salva as credenciais para o próximo uso
            with open(TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('calendar', 'v3', credentials=creds)
        print("Autenticação com Google Calendar concluída com sucesso!")
    
    def listar_eventos(self, dias: int = 7) -> List[Dict[str, Any]]:
        """Lista eventos do calendário para os próximos dias."""
        if not self.service:
            self.autenticar()
        
        time_min, time_max = get_time_range(dias)
        
        try:
            eventos_result = self.service.events().list(
                calendarId='primary',
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            eventos = eventos_result.get('items', [])
            return eventos
            
        except HttpError as error:
            print(f'Erro ao listar eventos: {error}')
            return []
    
    def criar_evento(self, evento: CalendarEventCreate) -> Optional[Dict[str, Any]]:
        """Cria um novo evento no calendário."""
        if not self.service:
            self.autenticar()
        
        # Converter datetime para formato RFC3339
        start_rfc = evento.start.isoformat()
        end_rfc = evento.end.isoformat()
        
        event_body = {
            'summary': evento.summary,
            'location': evento.location or '',
            'description': evento.description or '',
            'start': {
                'dateTime': start_rfc,
                'timeZone': 'America/Sao_Paulo',
            },
            'end': {
                'dateTime': end_rfc,
                'timeZone': 'America/Sao_Paulo',
            }
        }
        
        # Adiciona participantes se fornecidos
        if evento.attendees:
            event_body['attendees'] = evento.attendees
            
        # Adiciona lembretes se fornecidos
        if evento.reminders:
            event_body['reminders'] = evento.reminders
        
        try:
            event = self.service.events().insert(
                calendarId='primary',
                body=event_body
            ).execute()
            
            print(f'Evento criado: {event.get("htmlLink")}')
            return event
            
        except HttpError as error:
            print(f'Erro ao criar evento: {error}')
            return None
    
    def atualizar_evento(self, event_id: str, evento: CalendarEventCreate) -> Optional[Dict[str, Any]]:
        """Atualiza um evento existente no calendário."""
        if not self.service:
            self.autenticar()
        
        # Converter datetime para formato RFC3339
        start_rfc = evento.start.isoformat()
        end_rfc = evento.end.isoformat()
        
        event_body = {
            'summary': evento.summary,
            'location': evento.location or '',
            'description': evento.description or '',
            'start': {
                'dateTime': start_rfc,
                'timeZone': 'America/Sao_Paulo',
            },
            'end': {
                'dateTime': end_rfc,
                'timeZone': 'America/Sao_Paulo',
            }
        }
        
        # Adiciona participantes se fornecidos
        if evento.attendees:
            event_body['attendees'] = evento.attendees
            
        # Adiciona lembretes se fornecidos
        if evento.reminders:
            event_body['reminders'] = evento.reminders
        
        try:
            event = self.service.events().update(
                calendarId='primary',
                eventId=event_id,
                body=event_body
            ).execute()
            
            print(f'Evento atualizado: {event.get("htmlLink")}')
            return event
            
        except HttpError as error:
            print(f'Erro ao atualizar evento: {error}')
            return None
    
    def excluir_evento(self, event_id: str) -> bool:
        """Exclui um evento do calendário."""
        if not self.service:
            self.autenticar()
            
        try:
            self.service.events().delete(
                calendarId='primary',
                eventId=event_id
            ).execute()
            
            print(f'Evento excluído: {event_id}')
            return True
            
        except HttpError as error:
            print(f'Erro ao excluir evento: {error}')
            return False
    
    def buscar_evento(self, query: str) -> List[Dict[str, Any]]:
        """Busca eventos no calendário com base em uma consulta."""
        if not self.service:
            self.autenticar()
            
        time_min, time_max = get_time_range(30)  # Busca nos próximos 30 dias
        
        try:
            eventos_result = self.service.events().list(
                calendarId='primary',
                timeMin=time_min,
                timeMax=time_max,
                q=query,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            eventos = eventos_result.get('items', [])
            return eventos
            
        except HttpError as error:
            print(f'Erro ao buscar eventos: {error}')
            return []
    
    def analisar_tempo_livre(self, inicio: datetime.datetime, fim: datetime.datetime, 
                            duracao_minima: int = 30) -> List[Dict[str, Any]]:
        """
        Analisa os períodos de tempo livre no calendário.
        
        Args:
            inicio: Data e hora de início do período de análise
            fim: Data e hora final do período de análise
            duracao_minima: Duração mínima em minutos para considerar um período livre
            
        Returns:
            Lista de dicionários com períodos livres (start, end)
        """
        if not self.service:
            self.autenticar()
        
        # Converter para RFC3339
        inicio_rfc = inicio.isoformat()
        fim_rfc = fim.isoformat()
        
        try:
            # Obter eventos no período
            eventos_result = self.service.events().list(
                calendarId='primary',
                timeMin=inicio_rfc,
                timeMax=fim_rfc,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            eventos = eventos_result.get('items', [])
            
            # Encontrar períodos livres
            periodos_livres = []
            tempo_atual = inicio
            
            for evento in eventos:
                event_start = datetime.fromisoformat(
                    evento['start'].get('dateTime', evento['start'].get('date')).replace('Z', '+00:00')
                )
                
                # Verificar se há um período livre antes do evento
                if (event_start - tempo_atual).total_seconds() / 60 >= duracao_minima:
                    periodos_livres.append({
                        'start': tempo_atual,
                        'end': event_start
                    })
                
                # Atualizar o tempo atual para depois do evento
                event_end = datetime.fromisoformat(
                    evento['end'].get('dateTime', evento['end'].get('date')).replace('Z', '+00:00')
                )
                tempo_atual = event_end
            
            # Verificar se há um período livre após o último evento até o fim
            if (fim - tempo_atual).total_seconds() / 60 >= duracao_minima:
                periodos_livres.append({
                    'start': tempo_atual,
                    'end': fim
                })
            
            return periodos_livres
            
        except HttpError as error:
            print(f'Erro ao analisar tempo livre: {error}')
            return []
    
    def formatar_eventos(self, eventos: List[Dict[str, Any]]) -> str:
        """Formata uma lista de eventos para exibição ao usuário."""
        if not eventos:
            return "Nenhum evento encontrado para o período."
        
        resultado = []
        for i, evento in enumerate(eventos, 1):
            evento_formatado = formatar_evento_calendario(evento)
            resultado.append(f"Evento {i}:\n{evento_formatado}\n")
        
        return "\n".join(resultado)