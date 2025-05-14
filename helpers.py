import os
import time
from datetime import datetime, timedelta
import pytz
from typing import List, Any

def formatar_fontes(source_docs: List[Any]) -> str:
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

class PerformanceTimer:
    """Utilitário para medir performance de operações."""
    def __init__(self, nome_operacao: str = "Operação"):
        self.nome_operacao = nome_operacao
        self.inicio = None
        
    def __enter__(self):
        self.inicio = time.time()
        return self
        
    def __exit__(self, *args):
        duracao = time.time() - self.inicio
        print(f"{self.nome_operacao} concluída em {duracao:.2f} segundos")

def formatar_evento_calendario(evento: dict) -> str:
    """Formata um evento do Google Calendar para exibição amigável."""
    inicio = evento.get('start', {}).get('dateTime', evento.get('start', {}).get('date', 'N/A'))
    fim = evento.get('end', {}).get('dateTime', evento.get('end', {}).get('date', 'N/A'))
    
    # Converter para datetime se for string
    if isinstance(inicio, str) and 'T' in inicio:
        inicio_dt = datetime.fromisoformat(inicio.replace('Z', '+00:00'))
        inicio_formatado = inicio_dt.strftime("%d/%m/%Y %H:%M")
    else:
        inicio_formatado = inicio
        
    if isinstance(fim, str) and 'T' in fim:
        fim_dt = datetime.fromisoformat(fim.replace('Z', '+00:00'))
        fim_formatado = fim_dt.strftime("%d/%m/%Y %H:%M")
    else:
        fim_formatado = fim
    
    return (
        f"Título: {evento.get('summary', 'Sem título')}\n"
        f"Quando: {inicio_formatado} até {fim_formatado}\n"
        f"Local: {evento.get('location', 'Não especificado')}\n"
        f"Descrição: {evento.get('description', 'Sem descrição')}"
    )

def get_local_timezone():
    """Retorna o fuso horário local."""
    return pytz.timezone('America/Sao_Paulo')  # Ajuste para seu fuso horário

def get_time_range(dias: int = 7):
    """Retorna o intervalo de tempo para consulta de eventos (hoje até x dias)."""
    timezone = get_local_timezone()
    
    now = datetime.now(timezone)
    
    # Início: hoje à meia-noite
    start = datetime.combine(now.date(), datetime.min.time())
    start = timezone.localize(start)
    
    # Fim: daqui a X dias à meia-noite
    end = start + timedelta(days=dias)
    
    # Formatação para RFC3339 com time zone
    return start.isoformat(), end.isoformat()