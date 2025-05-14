import time
import sys
import json
from agents.essentialist_agent import EssentialistAgent
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import os

# Criar a aplicação FastAPI
app = FastAPI(
    title="Jarvis1 - Agente Essencialista",
    description="API para o assistente virtual Jarvis1 com integração ao Google Calendar",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar origens permitidas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos para a API
class PerguntaRequest(BaseModel):
    pergunta: str = Field(..., description="Pergunta ou comando para o agente")

class RespostaResponse(BaseModel):
    resposta: str = Field(..., description="Resposta do agente")
    fontes: Optional[str] = Field(None, description="Fontes consultadas")
    acao_realizada: Optional[dict] = Field(None, description="Detalhes da ação realizada")

# Instanciar o agente (será criado apenas uma vez ao iniciar a aplicação)
agent = None

@app.on_event("startup")
async def startup_event():
    global agent
    print("Inicializando o agente...")
    agent = EssentialistAgent()
    print("Agente inicializado com sucesso!")

@app.get("/")
async def root():
    return {"message": "Jarvis1 - Agente Essencialista API"}

@app.post("/perguntar", response_model=RespostaResponse)
async def perguntar(request: PerguntaRequest):
    global agent
    if not agent:
        raise HTTPException(status_code=500, detail="Agente não inicializado")
    
    try:
        resultado = agent.processar_entrada(request.pergunta)
        return RespostaResponse(
            resposta=resultado["resposta"],
            fontes=resultado.get("fontes"),
            acao_realizada=resultado.get("acao_realizada")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar pergunta: {str(e)}")

@app.get("/calendario/eventos")
async def listar_eventos(dias: int = 7):
    global agent
    if not agent:
        raise HTTPException(status_code=500, detail="Agente não inicializado")
    
    try:
        eventos = agent.calendar_service.listar_eventos(dias)
        return {"eventos": eventos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao listar eventos: {str(e)}")

# Função para executar como CLI
def run_cli():
    global agent
    agent = EssentialistAgent()
    
    print("\n==== Jarvis1: Assistente Essencialista ====")
    print("Converse com o agente (digite 'sair' para encerrar):")
    
    try:
        while True:
            print('\n')
            query = input("Você: ")
            
            if query.lower() in ["sair", "exit", "quit"]:
                break
                
            if not query.strip():
                continue
            
            inicio = time.time()
            
            # Processar a pergunta
            resultado = agent.processar_entrada(query)
            
            # A resposta já é exibida via streaming pelo StreamingStdOutCallbackHandler
            
            # Exibir fontes e tempo
            print('\n')
            print(f"\nFontes consultadas:\n{resultado['fontes']}")
            
            # Exibir resultado da ação, se houver
            if resultado.get('acao_realizada'):
                acao = resultado['acao_realizada']
                print(f"\nAção realizada: {acao['mensagem']}")
                if acao.get('dados') and isinstance(acao['dados'], str):
                    print(f"Resultado:\n{acao['dados']}")
            
            tempo = time.time() - inicio
            print(f"Tempo de resposta: {tempo:.2f} segundos")
    
    except KeyboardInterrupt:
        print("\nEncerrando o programa...")
    except Exception as e:
        print(f"\nErro: {e}")
    finally:
        print("\nObrigado por usar o Jarvis1!")

# Ponto de entrada
if __name__ == "__main__":
    # Verificar argumentos para modo CLI ou API
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        # Executar como API
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Executar como CLI
        run_cli()