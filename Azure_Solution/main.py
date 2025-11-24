import os
import re
import base64
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import faiss
import pickle

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from openai import AzureOpenAI
from dotenv import load_dotenv
import google.generativeai as genai


# ============================================================
# 1. CONFIG & ENV (.env'den yükleme)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Azure OpenAI (LLM + Whisper + TTS)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # https://xxx.openai.azure.com/
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")          # örn: gpt-4o-mini
AZURE_OPENAI_WHISPER_DEPLOYMENT = os.getenv("AZURE_OPENAI_WHISPER_DEPLOYMENT")    # örn: whisper
AZURE_OPENAI_TTS_DEPLOYMENT = os.getenv("AZURE_OPENAI_TTS_DEPLOYMENT")            # örn: tts
AZURE_OPENAI_TTS_VOICE = os.getenv("AZURE_OPENAI_TTS_VOICE", "alloy")

# Gemini (embedding için)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "models/text-embedding-004")

# Vector DB (FAISS + docs + embeddings)
VECTOR_DIR = Path(os.getenv("VECTOR_DIR", BASE_DIR / "vector_db"))
DOCS_PATH = VECTOR_DIR / "docs.pkl"
EMB_PATH = VECTOR_DIR / "embeddings.npy"
INDEX_PATH = VECTOR_DIR / "faiss_index.bin"

TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "5"))

required_env = {
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_CHAT_DEPLOYMENT": AZURE_OPENAI_CHAT_DEPLOYMENT,
    "AZURE_OPENAI_WHISPER_DEPLOYMENT": AZURE_OPENAI_WHISPER_DEPLOYMENT,
    "AZURE_OPENAI_TTS_DEPLOYMENT": AZURE_OPENAI_TTS_DEPLOYMENT,
    "GEMINI_API_KEY": GEMINI_API_KEY,
}

missing = [k for k, v in required_env.items() if not v]
if missing:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(missing)}"
    )


# ============================================================
# 2. KLIENTLER: Azure OpenAI, Gemini
# ============================================================

# Azure OpenAI client (Chat + Whisper + TTS)
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# Gemini config (embedding)
genai.configure(api_key=GEMINI_API_KEY)


# ============================================================
# 3. VECTOR STORE: FAISS + docs
# ============================================================

if not INDEX_PATH.exists():
    raise RuntimeError(f"FAISS index not found at {INDEX_PATH}")
if not DOCS_PATH.exists():
    raise RuntimeError(f"docs.pkl not found at {DOCS_PATH}")
if not EMB_PATH.exists():
    raise RuntimeError(f"embeddings.npy not found at {EMB_PATH}")

with open(DOCS_PATH, "rb") as f:
    docs: List[str] = pickle.load(f)

embeddings = np.load(EMB_PATH)
faiss_index = faiss.read_index(str(INDEX_PATH))

if faiss_index.ntotal != embeddings.shape[0]:
    raise RuntimeError(
        f"FAISS index size ({faiss_index.ntotal}) != embeddings rows ({embeddings.shape[0]})"
    )


# ========================  EMBEDDING (GEMINI)  ==========================

def embed_query(text: str) -> np.ndarray:
    """
    FAISS index'i Gemini 'models/text-embedding-004' ile oluşturduğun için
    sorgu embedding'ini de aynı modelle üret.
    """
    result = genai.embed_content(
        model=EMBEDDING_MODEL_NAME,
        content=text,
        task_type="retrieval_query",
    )

    emb = result["embedding"]
    if isinstance(emb, dict) and "values" in emb:
        vec = np.array(emb["values"], dtype="float32")[None, :]
    else:
        vec = np.array(emb, dtype="float32")[None, :]

    return vec  # shape: (1, dim)


def retrieve_context(question: str, top_k: int = TOP_K_DEFAULT) -> List[str]:
    if faiss_index.ntotal == 0:
        raise RuntimeError("FAISS index has no vectors.")
    top_k = min(top_k, faiss_index.ntotal)

    query_vec = embed_query(question)
    scores, indices = faiss_index.search(query_vec, top_k)
    idxs = indices[0]

    contexts: List[str] = []
    for idx in idxs:
        if 0 <= idx < len(docs):
            contexts.append(docs[idx])
    return contexts


def is_question_covered_by_kb(question: str, contexts: List[str]) -> bool:
    """
    Sorudaki anlamlı kelimeler bağlam içinde geçiyor mu diye KABACA kontrol eder.
    Tam eşleşme beklemiyoruz, sadece 'tamamen alakasız mı?' diye bakıyoruz.
    """
    q_tokens = re.findall(r"\w+", question.lower())
    # çok kısa kelimeleri (ve, ile, bir vs.) at
    q_tokens = [t for t in q_tokens if len(t) > 3]
    if not q_tokens:
        return False

    joined_context = " ".join(contexts).lower()
    return any(t in joined_context for t in q_tokens)


def build_messages(question: str, contexts: List[str]) -> list:
    """
    LLM'e sadece haber bağlamını kullanarak cevap üretmesini söyler.
    'Bu metin haber metinlerinde yok' cümlesi BURADA asla yok.
    """
    context_str = "\n\n".join(
        f"[Parça {i+1}]\n{c}" for i, c in enumerate(contexts)
    )
    user_content = (
        "Aşağıdaki soruyu verdiğim haber metni parçalarını kullanarak yanıtla.\n"
        "Bağlamı özetleyebilir, mantıksal çıkarımlar yapabilirsin.\n"
        "Sorunun cevabı bağlamda birebir yazmıyorsa bile, bağlamla ilişkili "
        "mantıklı ve tutarlı bir cevap üret.\n"
        "Bağlam yetersiz kalsa bile KESİNLİKLE 'Bu metin haber metinlerinde yok.' "
        "veya 'Bilmiyorum.' gibi bir cümle kurma; bu cevabı sadece backend döndürecek.\n\n"
        f"Soru: {question}\n\n"
        f"Bağlam:\n{context_str}\n"
    )

    return [
        {
            "role": "system",
            "content": (
                "Sen Türkçe konuşan bir haber bilgi tabanı asistanısın. "
                "Verilen bağlamı birincil kaynak olarak kullan, "
                "yanıtlarını net ve mümkünse maddeler halinde ver."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def answer_with_rag(question: str, top_k: int = TOP_K_DEFAULT) -> Tuple[str, List[str]]:
    """
    Hem text-qa hem voice-qa burayı kullanıyor.
    Burada:
      - Önce FAISS'ten bağlam çekiyoruz,
      - Eğer soru ile bağlam arasında HİÇ anlamlı kelime kesişimi yoksa
        direkt 'Bu metin haber metinlerinde yok.' diyoruz,
      - Aksi halde LLM'den normal cevap istiyoruz.
    Böylece:
      - Haberlere uyan sorularda hem text hem voice normal cevap üretecek,
      - Tamamen alakasız sorularda tek tip fallback gelecek.
    """
    contexts = retrieve_context(question, top_k=top_k)

    if not contexts or not is_question_covered_by_kb(question, contexts):
        # gerçekten haber KB ile alakası yoksa
        return "Bu metin haber metinlerinde yok.", contexts

    messages = build_messages(question, contexts)

    completion = openai_client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=messages,
        temperature=0.2,
    )
    answer = completion.choices[0].message.content
    return answer, contexts


# ============================================================
# 4. ASR (Whisper via Azure OpenAI)
# ============================================================

def transcribe_audio_with_whisper(audio_bytes: bytes) -> str:
    """
    Frontend'den gelen audio'yu geçici bir dosyaya yazıp
    Azure OpenAI Whisper ile text'e çevirir.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            result = openai_client.audio.transcriptions.create(
                file=f,
                model=AZURE_OPENAI_WHISPER_DEPLOYMENT,
                response_format="text",
                language="tr",  # Türkçe konuşma
            )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # response_format="text" ile genellikle doğrudan string döner
    if isinstance(result, str):
        return result
    if hasattr(result, "text"):
        return result.text
    return str(result)


# ============================================================
# 5. TTS (Azure OpenAI TTS deployment: tts)
# ============================================================

def synthesize_speech_tr(text: str) -> bytes:
    """
    Azure OpenAI TTS deployment (örn: 'tts') ile
    text'ten ses (wav) üret.
    """
    if not text.strip():
        raise ValueError("Cannot synthesize empty text.")

    response = openai_client.audio.speech.create(
        model=AZURE_OPENAI_TTS_DEPLOYMENT,   # deployment name: 'tts'
        voice=AZURE_OPENAI_TTS_VOICE,        # örn: 'alloy'
        input=text,
        response_format="wav",               # frontend audio/wav bekliyor
    )

    audio_bytes = response.read()
    response.close()
    return audio_bytes


# ============================================================
# 6. FASTAPI MODELLERİ
# ============================================================

class TextQARequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class TextQAResponse(BaseModel):
    answer: str
    contexts: List[str]


class VoiceQAResponse(BaseModel):
    transcript: str
    answer: str
    audio_base64: str


# ============================================================
# 7. FASTAPI APP + FRONTEND SERVE
# ============================================================

app = FastAPI(title="Azure Turkish Voice QA Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", response_class=FileResponse)
def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/text-qa", response_model=TextQAResponse)
def text_qa(req: TextQARequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")

    top_k = req.top_k or TOP_K_DEFAULT
    answer, contexts = answer_with_rag(req.question, top_k=top_k)
    return TextQAResponse(answer=answer, contexts=contexts)


@app.post("/voice-qa", response_model=VoiceQAResponse)
async def voice_qa(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Audio file is required.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    # 1) ASR
    transcript = transcribe_audio_with_whisper(audio_bytes)

    # 2) RAG + LLM
    answer, _contexts = answer_with_rag(transcript, top_k=TOP_K_DEFAULT)

    # 3) TTS
    tts_bytes = synthesize_speech_tr(answer)
    audio_b64 = base64.b64encode(tts_bytes).decode("utf-8")

    return VoiceQAResponse(
        transcript=transcript,
        answer=answer,
        audio_base64=audio_b64,
    )


# ============================================================
# 8. LOCAL ÇALIŞTIRMA (127.0.0.1)
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",   # sadece local makineden erişim
        port=8000,
        reload=True,
    )
