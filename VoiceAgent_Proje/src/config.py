import os
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai

# .env'den GOOGLE_API_KEY yükle
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY bulunamadı, .env dosyanı kontrol et.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# Proje ana klasörü (VoiceAgent_Proje)
BASE_DIR = Path(__file__).resolve().parent.parent

# ----------------- DATA / OUTPUT YOLLARI -----------------

# OpenSLR transcript klasörü
DATA_DIR = BASE_DIR / "data"

# Örnek giriş/çıkış dosyaları için (istersen kullanırsın)
SAMPLE_AUDIO_DIR = BASE_DIR / "sample_audios"
SAMPLE_AUDIO_DIR.mkdir(exist_ok=True)

# Teslimat örnekleri ve TTS çıktıları için
OUTPUT_DIR = BASE_DIR / "teslimat_ornekleri"
OUTPUT_DIR.mkdir(exist_ok=True)

# Varsayılan TTS çıktı dosyası
OUTPUT_AUDIO_FILE = OUTPUT_DIR / "response_output.mp3"

# ----------------- VEKTÖR VERİTABANI (KALICI) -----------------

VECTOR_DB_DIR = BASE_DIR / "vector_db"
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_PATH = VECTOR_DB_DIR / "faiss_index.bin"
DOCS_PATH = VECTOR_DB_DIR / "docs.pkl"
EMBEDDINGS_PATH = VECTOR_DB_DIR / "embeddings.npy"

# ----------------- MODEL / RAG AYARLARI -----------------

EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-2.5-flash"

RETRIEVAL_TOP_K = 3
TTS_LANG = "tr"
MAX_WORKERS = 10
