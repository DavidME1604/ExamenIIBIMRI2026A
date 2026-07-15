"""
Configuración central del sistema RAG.

Carga las variables de entorno desde el archivo .env (usando python-dotenv)
y define las constantes que usan el resto de los módulos del proyecto.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Cargamos el .env que está en la raíz del proyecto (si existe).
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# --- Claves y modelo de Gemini ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# --- Modelo de embeddings ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Base de datos vectorial (ChromaDB) ---
CHROMA_DIR = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "arxiv_papers"

# --- Parámetros de recuperación ---
TOP_K = 5

# Subconjunto de documentos a indexar. Se limita el corpus para que el
# índice resultante sea liviano y fácil de desplegar (por ejemplo, en
# Hugging Face Spaces).
MAX_DOCS = 15000

# --- Rutas de datos ---
DATA_DIR = BASE_DIR / "data"
CORPUS_CSV_PATH = str(DATA_DIR / "corpus_clean.csv")
