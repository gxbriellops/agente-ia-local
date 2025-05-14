import os
import torch

# Diretorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = r"C:\Users\Gabriel Lopes\Documents\cofrinho\100. Recursos\RAG\Agente Essencialista"
CACHE_DIR = os.path.join(os.path.dirname(DOCS_DIR), "cache")
VECTOR_STORE_PATH = os.path.join(CACHE_DIR, "faiss_index")
CREDENTIALS_DIR = os.path.join(BASE_DIR, "credentials")

# Configurações LLM
LLM_MODEL = "llama3.1:8b-instruct-Q4_K_M"
LLM_TEMPERATURE = 0.1
LLM_TOP_P = 0.9
LLM_NUM_CTX = 4096

# Configurações do Vector Store
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 3
RETRIEVER_FETCH_K = 5
RETRIEVER_LAMBDA_MULT = 0.5

# Verificar disponibilidade da GPU
HAS_CUDA = torch.cuda.is_available()
if HAS_CUDA:
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"

# Configurações Google Calendar
CALENDAR_SCOPES = ['https://www.googleapis.com/auth/calendar']
TOKEN_FILE = os.path.join(CREDENTIALS_DIR, 'token.json')
CREDENTIALS_FILE = os.path.join(CREDENTIALS_DIR, 'credentials.json')

# Criar diretórios necessários
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CREDENTIALS_DIR, exist_ok=True)