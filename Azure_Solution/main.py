import os
import base64
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
import pickle

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer
import azure.cognitiveservices.speech as speechsdk


# -----------------------------
# Config & global objects
# -----------------------------

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_OPENAI_WHISPER_DEPLOYMENT = os.getenv("AZURE_OPENAI_WHISPER_DEPLOYMENT")

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
AZURE_SPEECH_VOICE = os.getenv("AZURE_SPEECH_VOICE", "tr-TR-AhmetNeural")

# docs.pkl, embeddings.npy, faiss_index.bin'in olduğu klasör
VECTOR_DIR = Path(os.getenv("VECTOR_DIR", "."))
# DİKKAT: Buradaki model, FAISS index'i oluştururken kullandığın embedding
# modeliyle aynı olmalı (boyutlar uyuşmalı!).
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "5"))

required_env = {
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_CHAT_DEPLOYMENT": AZURE_OPENAI_CHAT_DEPLOYMENT,
    "AZURE_OPENAI_WHISPER_DEPLOYMENT": AZURE_OPENAI_WHISPER_DEPLOYMENT,
    "AZURE_SPEECH_KEY": AZURE_SPEECH_KEY,
    "AZURE_SPEECH_REGION": AZURE_SPEECH_REGION,
}
missing = [name for name, value in required_env.items() if not value]
if missing:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(missing)}"
    )

# Azure OpenAI client
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# Azure Speech config (for Turkish TTS)
speech_config = speechsdk.SpeechConfig(
    subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION
)
speech_config.speech_synthesis_language = "tr-TR"
speech_config.speech_synthesis_voice_name = AZURE_SPEECH_VOICE

# Embedding model (FAISS index'i neyle oluşturduysan onu yaz)
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# -----------------------------
# Load FAISS vector store
# -----------------------------

DOCS_PATH = VECTOR_DIR / "docs.pkl"
EMB_PATH = VECTOR_DIR / "embeddings.npy"
INDEX_PATH = VECTOR_DIR / "faiss_index.bin"

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


# -----------------------------
# RAG helper functions
# -----------------------------

def embed_query(text: str) -> np.ndarray:
    vec = embedder.encode([text], normalize_embeddings=True)
    # FAISS expects float32
    return np.array(vec).astype("float32")


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
        f"[Parça {i+1}]\n{chunk}" for i, chunk in enumerate(contexts)
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
                "Yanıtlarını kısa, net ve doğrudan ver."
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


# -----------------------------
# ASR (Whisper via Azure OpenAI)
# -----------------------------

def transcribe_audio_with_whisper(audio_bytes: bytes) -> str:
    # SDK dosya beklediği için temp'e yazıyoruz
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            result = openai_client.audio.transcriptions.create(
                file=f,
                model=AZURE_OPENAI_WHISPER_DEPLOYMENT,
                response_format="text",
            )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # response_format="text" ise result direkt string olabiliyor
    if hasattr(result, "text"):
        return result.text
    return str(result)


# -----------------------------
# TTS (Azure Speech)
# -----------------------------

def synthesize_speech_tr(text: str) -> bytes:
    if not text:
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


# -----------------------------
# FastAPI models & app
# -----------------------------

class TextQARequest(BaseModel):
    question: str
    top_k: int | None = None


class TextQAResponse(BaseModel):
    question: str
    answer: str
    contexts: List[str]


class VoiceQAResponse(BaseModel):
    question: str
    transcript: str
    answer: str
    audio_base64: str  # WAV bytes as base64 string


app = FastAPI(title="Turkish Voice-based QA Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo için açık bırakıldı
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/text-qa", response_model=TextQAResponse)
def text_qa(req: TextQARequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")

    top_k = req.top_k or TOP_K_DEFAULT
    answer, contexts = answer_with_rag(req.question, top_k=top_k)
    return TextQAResponse(question=req.question, answer=answer, contexts=contexts)


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
    audio_out = synthesize_speech_tr(answer)
    audio_b64 = base64.b64encode(audio_out).decode("utf-8")

    return VoiceQAResponse(
        question=transcript,
        transcript=transcript,
        answer=answer,
        audio_base64=audio_b64,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
