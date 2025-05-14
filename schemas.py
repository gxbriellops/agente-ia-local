from pydantic import BaseModel, Field
from typing import List, Tuple, Any, Optional, Dict
from datetime import datetime

# Modelos para o RAG
class PerguntaInput(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]]

class RespostaOutput(BaseModel):
    answer: str
    source_documents: Any

# Modelos para o Google Calendar
class CalendarEvent(BaseModel):
    id: str
    summary: str
    description: Optional[str] = None
    location: Optional[str] = None
    start: datetime
    end: datetime
    attendees: Optional[List[Dict[str, str]]] = None
    
    class Config:
        arbitrary_types_allowed = True

class CalendarEventCreate(BaseModel):
    summary: str
    description: Optional[str] = None
    location: Optional[str] = None
    start: datetime
    end: datetime
    attendees: Optional[List[Dict[str, str]]] = None
    reminders: Optional[Dict[str, Any]] = None

class AgentAction(BaseModel):
    action_type: str = Field(..., description="Tipo de ação a ser executada")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parâmetros para a ação")
    context: Optional[str] = None