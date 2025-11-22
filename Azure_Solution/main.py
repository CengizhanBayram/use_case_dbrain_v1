import os
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
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import google.generativeai as genai


# ============================================================
# 1. CONFIG & ENV (.env'den yükleme)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Azure OpenAI (LLM + Whisper)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # https://xxx.openai.azure.com/
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")          # örn: gpt-4o-mini
AZURE_OPENAI_WHISPER_DEPLOYMENT = os.getenv("AZURE_OPENAI_WHISPER_DEPLOYMENT")    # whisper deployment adı

# Azure Speech (TTS)
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")  # örn: westeurope
AZURE_SPEECH_VOICE = os.getenv("AZURE_SPEECH_VOICE", "tr-TR-AhmetNeural")

# Gemini (embedding için)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "models/text-embedding-004")

# Vector DB (FAISS + docs + embeddings)
# VECTOR_DIR'i .env içinde tam path olarak da verebilirsin.
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
    "AZURE_SPEECH_KEY": AZURE_SPEECH_KEY,
    "AZURE_SPEECH_REGION": AZURE_SPEECH_REGION,
    "GEMINI_API_KEY": GEMINI_API_KEY,
}

missing = [k for k, v in required_env.items() if not v]
if missing:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(missing)}"
    )


# ============================================================
# 2. KLIENTLER: Azure OpenAI, Azure Speech, Gemini
# ============================================================

# Azure OpenAI client
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# Azure Speech config (TTS)
speech_config = speechsdk.SpeechConfig(
    subscription=AZURE_SPEECH_KEY,
    region=AZURE_SPEECH_REGION,
)
speech_config.speech_synthesis_language = "tr-TR"
speech_config.speech_synthesis_voice_name = AZURE_SPEECH_VOICE

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


# ======================== EMBEDDING ==========================

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
    # library versiyonuna göre embedding dict veya liste olabilir
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


def build_messages(question: str, contexts: List[str]) -> list:
    context_str = "\n\n".join(
        f"[Parça {i+1}]\n{c}" for i, c in enumerate(contexts)
    )
    user_content = (
        "Aşağıdaki soruyu sadece verilen bağlamı kullanarak yanıtla.\n\n"
        f"Soru: {question}\n\n"
        f"Bağlam:\n{context_str}\n\n"
        "Bağlamda yeterli bilgi yoksa 'Bilmiyorum.' de ve uydurma.\n"
    )

    return [
        {
            "role": "system",
            "content": (
                "Sen Türkçe konuşan bir bilgi tabanı asistanısın. "
                "Yalnızca verilen bağlamı kullanarak cevap ver. "
                "Yanıtlarını net, kısa ve gerektiğinde maddeler halinde ver."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def answer_with_rag(question: str, top_k: int = TOP_K_DEFAULT) -> Tuple[str, List[str]]:
    contexts = retrieve_context(question, top_k=top_k)
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
# 5. TTS (Azure Speech)
# ============================================================

def synthesize_speech_tr(text: str) -> bytes:
    if not text.strip():
        raise ValueError("Cannot synthesize empty text.")

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_out.close()

    audio_config = speechsdk.audio.AudioOutputConfig(filename=tmp_out.name)
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )
    result = synthesizer.speak_text_async(text).get()

    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        raise RuntimeError(f"TTS failed: {result.reason}")

    with open(tmp_out.name, "rb") as f:
        audio_bytes = f.read()

    try:
        os.remove(tmp_out.name)
    except OSError:
        pass

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

# CORS (frontend aynı origin'de çalışacağı için zaten sorun yok ama demo için açık)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend klasörü (Azure_Solution/frontend)
FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", response_class=FileResponse)
def serve_index():
    """
    Root isteği geldiğinde ana HTML sayfayı döner.
    http://127.0.0.1:8000 -> frontend/index.html
    """
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/text-qa", response_model=TextQAResponse)
def text_qa(req: TextQARequest):
    """
    Metin tabanlı soru-cevap (RAG + LLM).
    Frontend sendTextMessage burayı çağırıyor.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")

    top_k = req.top_k or TOP_K_DEFAULT
    answer, contexts = answer_with_rag(req.question, top_k=top_k)
    return TextQAResponse(answer=answer, contexts=contexts)


@app.post("/voice-qa", response_model=VoiceQAResponse)
async def voice_qa(file: UploadFile = File(...)):
    """
    Ses tabanlı soru-cevap (ASR + RAG + TTS).
    Frontend toggleRealTimeRecording -> processVoiceMessage burayı çağırıyor.
    Form field adı: 'file'
    """
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
