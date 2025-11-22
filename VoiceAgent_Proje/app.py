import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import streamlit as st
from audio_recorder_streamlit import audio_recorder

from src.config import DATA_DIR
from src.data_loader import DataLoader
from src.llm_client import GeminiClient
from src.vector_store import VectorStore
from src.tts_service import TTSService
from src.guardrails import apply_guardrails


# -------------------------------------------------------------------------
# LOG AYARLARI
# -------------------------------------------------------------------------

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "interaction_log.jsonl"


def log_interaction(
    mode: str,
    query: str,
    rewritten_query: Optional[str],
    transcript: Optional[str],
    answer: str,
    retrieved_passages: List[str],
    guardrail_reasons: Dict[str, bool],
    tts_time: Optional[float],
    tts_path: Optional[str],
):
    """Her etkileşimi JSONL formatında log dosyasına ekler."""
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mode": mode,  # "text" veya "voice"
        "query": query,
        "rewritten_query": rewritten_query,
        "transcript": transcript,
        "answer": answer,
        "retrieved_passages": retrieved_passages,
        "guardrail_reasons": guardrail_reasons,
        "tts_time": tts_time,
        "tts_path": tts_path,
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[LOGGING ERROR] {e}")


# -------------------------------------------------------------------------
# SESSION STATE BAŞLATMA
# -------------------------------------------------------------------------

def init_session_state():
    if "gemini_client" not in st.session_state:
        st.session_state.gemini_client = GeminiClient()

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()

    if "tts_service" not in st.session_state:
        st.session_state.tts_service = TTSService()

    if "kb_initialized" not in st.session_state:
        st.session_state.kb_initialized = False

    if "status_message" not in st.session_state:
        st.session_state.status_message = "Henüz başlatılmadı."

    if "chat_history" not in st.session_state:
        # [(user_msg, bot_msg), ...]
        st.session_state.chat_history = []  # type: List[Tuple[str, str]]

    if "last_answer" not in st.session_state:
        st.session_state.last_answer = ""  # type: str

    if "last_transcript" not in st.session_state:
        st.session_state.last_transcript = ""  # type: str

    if "last_audio_path" not in st.session_state:
        st.session_state.last_audio_path = None  # type: Optional[str]

    if "last_tts_time" not in st.session_state:
        st.session_state.last_tts_time = None  # type: Optional[float]

    # Mikrofondan gelen son kaydı takip et (aynı kaydı iki kere işlememek için)
    if "last_mic_audio" not in st.session_state:
        st.session_state.last_mic_audio = None  # type: Optional[bytes]

    # RAG ayarları için varsayılanlar
    st.session_state.setdefault("top_k", 3)
    st.session_state.setdefault("enlarge_factor", 3)
    st.session_state.setdefault("use_rerank", True)
    st.session_state.setdefault("enable_rewrite", True)


# -------------------------------------------------------------------------
# KNOWLEDGE BASE OLUŞTURMA (DISK PERSIST)
# -------------------------------------------------------------------------

def build_knowledge_base() -> str:
    """
    1) Eğer diskte FAISS index + dokümanlar varsa onları yükler.
    2) Yoksa DATA_DIR'den transcriptleri okuyup yeni index oluşturur ve diske kaydeder.

    -> CASE'in 1. görevi: "Transcriptlerden Knowledge Base" BURADA.
    """
    gc: GeminiClient = st.session_state.gemini_client
    vs: VectorStore = st.session_state.vector_store

    try:
        # 1. Önce mevcut indexi diskten yüklemeyi dene (varsa hızlı)
        if hasattr(vs, "load_from_disk") and vs.load_from_disk():
            st.session_state.kb_initialized = True
            return (
                f"✅ Mevcut vektör veritabanı diskten yüklendi. "
                f"Toplam {len(vs.documents)} parça."
            )

        # 2. Diskte yoksa sıfırdan kur
        loader = DataLoader(str(DATA_DIR))
        docs = loader.load()

        if not docs:
            st.session_state.kb_initialized = False
            return "❌ Hiç doküman yüklenemedi. DATA_DIR yolunu ve dataset'i kontrol et."

        embeddings = gc.get_embeddings_batch(docs, batch_size=10)
        if not embeddings:
            st.session_state.kb_initialized = False
            return "❌ Embedding üretilemedi. Gemini ayarlarını ve API anahtarını kontrol et."

        # build_index, save parametresini destekliyorsa kullan, desteklemiyorsa normal çağır
        try:
            vs.build_index(docs, embeddings, save=True)
        except TypeError:
            vs.build_index(docs, embeddings)

        if getattr(vs, "index", None) is None:
            st.session_state.kb_initialized = False
            return "❌ FAISS index oluşturulamadı."

        st.session_state.kb_initialized = True
        return (
            f"✅ Yeni bilgi bankası oluşturuldu ve diske kaydedildi. "
            f"Toplam {len(docs)} parça."
        )

    except Exception as e:
        st.session_state.kb_initialized = False
        return f"❌ Bilgi bankası hazırlanırken hata: {e}"


# -------------------------------------------------------------------------
# QUERY REWRITING
# -------------------------------------------------------------------------

def rewrite_query_if_enabled(raw_query: str) -> str:
    """
    enable_rewrite açıksa, Gemini ile soruyu daha net / arama-dostu hale getirir.
    Değilse, raw_query'i olduğu gibi döner.
    """
    if not st.session_state.enable_rewrite:
        return raw_query

    gc: GeminiClient = st.session_state.gemini_client
    prompt = f"""
Kullanıcının orijinal sorusu:

\"\"\"{raw_query}\"\"\"

GÖREVİN:
- Bu soruyu daha net, kısa ve bilgi aramaya uygun bir Türkçe cümleye dönüştür.
- Anlamı bozma.
- Cevabında sadece yeniden yazılmış soruyu ver, açıklama yazma.
"""

    try:
        resp = gc.model.generate_content(prompt)
        rewritten = (resp.text or "").strip()
        # Çok boşsa veya alakasızsa fallback
        if not rewritten or len(rewritten) < 3:
            return raw_query
        return rewritten
    except Exception as e:
        print(f"[QUERY REWRITE ERROR] {e}")
        return raw_query


# -------------------------------------------------------------------------
# RAG + FALLBACK CEVAP ÜRETİCİ
# -------------------------------------------------------------------------

def generate_rag_answer(raw_query: str):
    """
    Metin soru için:
    - Eğer KB hazırs a→ RAG + Guardrails
    - Eğer KB hazır değilse → direkt LLM cevabı (guardrails opsiyonel)

    CASE'in 2. görevi: "Text Q&A + retrieval" BURADA.
    """
    if not raw_query or len(raw_query.strip()) == 0:
        return ("Lütfen bir soru yazın.", {}, [], raw_query)

    gc: GeminiClient = st.session_state.gemini_client
    vs: VectorStore = st.session_state.vector_store

    # 1) KB HAZIR DEĞİL → SADECE LLM (FALLBACK, HER ZAMAN ÇALIŞSIN)
    if not st.session_state.kb_initialized:
        prompt = f"""
Şu an haber bilgi bankası devreye alınmamış durumda.
Yine de genel bir Türkçe asistan olarak aşağıdaki soruya kısa ve net (2-3 cümle) cevap ver.

[SORU]
{raw_query}
"""
        try:
            response = gc.model.generate_content(prompt)
            raw_answer = (response.text or "").strip()
            if not raw_answer:
                raw_answer = "Şu anda bu soruya yanıt üretemiyorum."

            # Guardrails bu modda opsiyonel, ama yine de deneriz
            try:
                gr_result = apply_guardrails(
                    answer=raw_answer,
                    query=raw_query,
                    context="",
                )
                final_answer = gr_result.answer
                reasons = gr_result.reasons
            except Exception as ge:
                print(f"[GUARDRAIL ERROR - FALLBACK] {ge}")
                final_answer = raw_answer
                reasons = {}

            # RAG yok → retrieved_passages boş, rewritten_query de raw_query olsun
            return final_answer, reasons, [], raw_query

        except Exception as e:
            return (
                f"Şu anda cevap üretilirken bir hata oluştu: {e}",
                {},
                [],
                raw_query,
            )

    # 2) KB HAZIR → RAG PIPELINE

    # Query rewrite (opsiyonel)
    rewritten_query = rewrite_query_if_enabled(raw_query)

    # Sorgu embedding'i
    query_emb = gc.get_query_embedding(rewritten_query)
    if not query_emb:
        return (
            "Sorgu için embedding alınırken bir hata oluştu.",
            {},
            [],
            rewritten_query,
        )

    # Reranker'lı retrieval (tunable)
    top_k = int(st.session_state.top_k)
    enlarge_factor = int(st.session_state.enlarge_factor)
    use_rerank = bool(st.session_state.use_rerank)

    try:
        retrieved_docs = vs.search(
            query_embedding=query_emb,
            k=top_k,
            use_rerank=use_rerank,
            enlarge_factor=enlarge_factor,
        )
    except TypeError:
        # Eğer VectorStore.search bu parametreleri desteklemiyorsa eski stile dön
        retrieved_docs = vs.search(query_embedding=query_emb, k=top_k)

    context = "\n\n".join(retrieved_docs) if retrieved_docs else ""

    prompt = f"""
Sen Türkçe konuşan bir haber asistanısın.

[BAĞLAM]
{context}

[KULLANICI SORUSU]
{raw_query}

GÖREVİN:
- Sadece bağlamdaki bilgilere dayanarak cevap ver.
- Bağlamda yeterli bilgi yoksa, 'Bu soruya mevcut haber metinlerinden net cevap veremiyorum.' de.
- Cevabın 2-3 cümleyi geçmesin, sade Türkçe kullan.
"""

    try:
        response = gc.model.generate_content(prompt)
        raw_answer = (response.text or "").strip()
        if not raw_answer:
            return ("Cevap üretilemedi.", {}, retrieved_docs, rewritten_query)

        # Guardrails
        try:
            gr_result = apply_guardrails(
                answer=raw_answer,
                query=raw_query,
                context=context,
            )
            final_answer = gr_result.answer
            reasons = gr_result.reasons
        except Exception as ge:
            print(f"[GUARDRAIL ERROR - RAG] {ge}")
            final_answer = raw_answer
            reasons = {}

        return final_answer, reasons, retrieved_docs, rewritten_query

    except Exception as e:
        return (
            f"LLM cevabı üretilirken hata oluştu: {str(e)}",
            {},
            retrieved_docs,
            rewritten_query,
        )


# -------------------------------------------------------------------------
# TTS (OTO) – CEVAP İÇİN SES ÜRET VE STATE'E YAZ
# -------------------------------------------------------------------------

def run_tts_for_answer(answer: str) -> None:
    """
    Verilen cevabı TTS ile sese çevirir, süreyi ölçer ve
    last_audio_path + last_tts_time olarak session_state'e yazar.

    CASE'in 3(c): TTS kısmı BURADA.
    """
    # Eski sesi temizle
    st.session_state.last_audio_path = None
    st.session_state.last_tts_time = None

    if not answer or not answer.strip():
        return

    tts: TTSService = st.session_state.tts_service

    t0 = time.time()
    audio_path = tts.text_to_speech(answer)
    t1 = time.time()
    tts_time = t1 - t0

    if audio_path is None:
        st.warning("TTS sırasında bir hata oluştu, ses üretilemedi.")
        return

    st.session_state.last_audio_path = audio_path
    st.session_state.last_tts_time = tts_time


# -------------------------------------------------------------------------
# SESLİ SORU PIPELINE'I (DOSYA YÜKLEME)
# -------------------------------------------------------------------------

def handle_voice_question(audio_file) -> None:
    """
    Dosyadan yüklenen ses için:
    - (a) ASR (Gemini) -> transcript
    - (b) RAG + LLM cevabı
    - (c) TTS ile cevap sesi
    """
    if audio_file is None:
        st.warning("Lütfen önce bir ses dosyası yükleyin.")
        return

    gc: GeminiClient = st.session_state.gemini_client

    # Geçici dosya kaydet
    suffix = Path(audio_file.name).suffix or ".wav"
    temp_path = Path("temp_upload_audio" + suffix)
    with open(temp_path, "wb") as f:
        f.write(audio_file.read())

    t0 = time.time()

    # (a) ASR
    transcript = gc.transcribe_audio(str(temp_path))
    t1 = time.time()
    st.write(f"⏱️ ASR süresi (dosya): {t1 - t0:.2f} sn")

    if transcript.startswith("Hata:") or transcript.startswith("ASR Hatası"):
        st.error(transcript)
        return

    st.session_state.last_transcript = transcript

    # (b) RAG / fallback cevabı
    answer, reasons, retrieved_docs, rewritten_query = generate_rag_answer(transcript)
    t2 = time.time()
    st.write(f"⏱️ RAG/fallback süresi (dosya): {t2 - t1:.2f} sn")
    st.write(f"⏱️ Toplam (ASR + RAG/fallback - dosya): {t2 - t0:.2f} sn")

    # Chat geçmişine ekle
    user_display = f"📂 (Dosya) {transcript}"
    st.session_state.chat_history.append((user_display, answer))
    st.session_state.last_answer = answer

    # (c) TTS (oto + spinner)
    with st.spinner("🔊 Cevap için ses üretiliyor..."):
        run_tts_for_answer(answer)
    tts_time = st.session_state.last_tts_time
    audio_path = st.session_state.last_audio_path

    # Logla
    log_interaction(
        mode="voice",
        query=transcript,
        rewritten_query=rewritten_query,
        transcript=transcript,
        answer=answer,
        retrieved_passages=retrieved_docs,
        guardrail_reasons=reasons,
        tts_time=tts_time,
        tts_path=audio_path,
    )


# -------------------------------------------------------------------------
# SESLİ SORU PIPELINE'I (MİKROFON)
# -------------------------------------------------------------------------

def handle_voice_bytes(audio_bytes: bytes) -> None:
    """
    Mikrofondan gelen raw bytes için:
    - (a) ASR (Gemini) -> transcript
    - (b) RAG + LLM cevabı
    - (c) TTS ile cevap sesi
    """
    if not audio_bytes:
        st.warning("Kayıt alınamadı.")
        return

    gc: GeminiClient = st.session_state.gemini_client

    # Geçici dosya kaydet
    temp_path = Path("temp_mic_audio.wav")
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)

    t0 = time.time()

    # (a) ASR
    transcript = gc.transcribe_audio(str(temp_path))
    t1 = time.time()
    st.write(f"⏱️ ASR süresi (mic): {t1 - t0:.2f} sn")

    if transcript.startswith("Hata:") or transcript.startswith("ASR Hatası"):
        st.error(transcript)
        return

    st.session_state.last_transcript = transcript

    # (b) RAG / fallback cevabı
    answer, reasons, retrieved_docs, rewritten_query = generate_rag_answer(transcript)
    t2 = time.time()
    st.write(f"⏱️ RAG/fallback süresi (mic): {t2 - t1:.2f} sn")
    st.write(f"⏱️ Toplam (ASR + RAG/fallback - mic): {t2 - t0:.2f} sn")

    # Chat geçmişine ekle
    user_display = f"🎙️ (Mikrofon) {transcript}"
    st.session_state.chat_history.append((user_display, answer))
    st.session_state.last_answer = answer

    # (c) TTS (oto + spinner)
    with st.spinner("🔊 Cevap için ses üretiliyor..."):
        run_tts_for_answer(answer)
    tts_time = st.session_state.last_tts_time
    audio_path = st.session_state.last_audio_path

    # Logla
    log_interaction(
        mode="voice",
        query=transcript,
        rewritten_query=rewritten_query,
        transcript=transcript,
        answer=answer,
        retrieved_passages=retrieved_docs,
        guardrail_reasons=reasons,
        tts_time=tts_time,
        tts_path=audio_path,
    )


# -------------------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Türkçe Voice Agent",
        page_icon="🎙️",
        layout="wide",
    )
    init_session_state()

    # İlk açılışta KB'yi otomatik kur (CASE gereği RAG aktif olsun)
    if not st.session_state.kb_initialized:
        with st.spinner("📚 Knowledge base hazırlanıyor (ilk seferde biraz sürebilir)..."):
            msg = build_knowledge_base()
        st.session_state.status_message = msg

    st.title("🎙️ Türkçe Voice Agent (ASR + RAG + TTS)")

    # -------------- SIDEBAR --------------
    with st.sidebar:
        st.header("⚙️ Ayarlar / Bilgi Bankası")

        if st.button("🔄 Bilgi Bankasını Elle Yeniden Başlat"):
            with st.spinner("Bilgi bankası yeniden hazırlanıyor..."):
                msg = build_knowledge_base()
            st.session_state.status_message = msg

        st.info(st.session_state.status_message)

        st.markdown("---")
        st.markdown("**Knowledge Base Durumu:**")
        if st.session_state.kb_initialized:
            st.success("Knowledge base yüklü (RAG aktif).")
        else:
            st.warning(
                "Knowledge base yüklenemedi. "
                "Şu anda asistan genel Türkçe cevap verecek, RAG çalışmayacak."
            )

        st.markdown("---")
        st.header("🧠 RAG Ayarları")

        # Reranker toggle
        use_rerank = st.checkbox(
            "Reranker kullan",
            value=st.session_state.use_rerank,
        )
        st.session_state.use_rerank = use_rerank

        # Top K
        top_k = st.slider(
            "Top K",
            min_value=1,
            max_value=5,
            value=st.session_state.top_k,
        )
        st.session_state.top_k = top_k

        # Enlarge factor
        enlarge_factor = st.slider(
            "Enlarge Factor",
            min_value=1,
            max_value=5,
            value=st.session_state.enlarge_factor,
        )
        st.session_state.enlarge_factor = enlarge_factor

        st.markdown("---")
        st.header("✏️ Query Rewriting")

        enable_rewrite = st.checkbox(
            "Soru yeniden yazma (query rewrite)",
            value=st.session_state.enable_rewrite,
        )
        st.session_state.enable_rewrite = enable_rewrite

    # -------------- ANA GÖVDE --------------
    # ÖNCE KONTROLLER, SONRA CHAT (önce state güncellensin, sonra chat çizilsin)
    col_controls, col_chat = st.columns([1, 2])

    # --- Kontroller (ses + transcript) ---
    with col_controls:
        st.subheader("🎤 Sesli Soru")

        st.markdown("**1) Dosyadan yükle**")
        audio_file = st.file_uploader(
            "Ses dosyası yükle (wav/mp3)",
            type=["wav", "mp3"],
            accept_multiple_files=False,
        )

        if st.button("📂 Yüklenen sesle soruyu çalıştır"):
            handle_voice_question(audio_file)

        st.markdown("---")
        st.markdown("**2) Mikrofondan kaydet**")
        st.caption(
            "Butona bastığında üstte dalga formu belirecek. Dalga formu görünüyorsa o anda ses kaydediyorsun."
        )

        mic_audio = audio_recorder(
            text="🎙️ Kaydı başlat / durdur",
            pause_threshold=3.0,
            sample_rate=16000,
            icon_size="2x",
        )

        if mic_audio is None:
            st.info("Durum: Hazır. Ses kaydı için butona basın.")
        else:
            # Yeni kayıt mı, eski mi kontrol et (aynı kaydı tekrar işleme)
            if st.session_state.last_mic_audio != mic_audio:
                st.session_state.last_mic_audio = mic_audio
                st.success("Durum: Kayıt tamamlandı, ses işleniyor...")
                # Kayıt alındığını göstermek için player
                try:
                    st.audio(mic_audio, format="audio/wav")
                except Exception:
                    pass
                handle_voice_bytes(mic_audio)
            else:
                st.info("Bu mikrofon kaydı zaten işlendi.")

        # Transcript gösterme
        if st.session_state.last_transcript:
            st.markdown("---")
            st.markdown("**Son ASR Transcript:**")
            st.code(st.session_state.last_transcript, language="text")

    # --- Chat Alanı ---
    with col_chat:
        st.subheader("💬 Sohbet (Text Q&A)")

        # Var olan chat geçmişini göster (birden fazla konuşma hep kalacak)
        for user_msg, bot_msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(user_msg)
            with st.chat_message("assistant"):
                st.markdown(bot_msg)

        # Kullanıcıdan yeni metin girişi
        user_input = st.chat_input("Yazılı soru sorabilirsiniz...")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            # Ne olursa olsun TRY/CATCH ile chat bozulmasın
            try:
                answer, reasons, retrieved_docs, rewritten_query = generate_rag_answer(
                    user_input
                )
            except Exception as e:
                print(f"[generate_rag_answer ERROR] {e}")
                answer = f"Beklenmeyen bir hata oluştu: {e}"
                reasons = {}
                retrieved_docs = []
                rewritten_query = user_input

            # Bot cevabını göster
            with st.chat_message("assistant"):
                st.markdown(answer)

                if st.session_state.kb_initialized and retrieved_docs:
                    with st.expander("🔎 Kullanılan pasajlar / Query rewrite"):
                        st.markdown(f"**Rewritten query:** `{rewritten_query}`")
                        st.markdown("**Retrieval pasajları:**")
                        for i, p in enumerate(retrieved_docs, start=1):
                            st.markdown(f"**[{i}]** {p}")
                else:
                    with st.expander("ℹ️ Not"):
                        st.markdown(
                            "Bu cevap bilgi bankası kullanılmadan, sadece LLM ile üretildi."
                        )

            # Geçmişe kaydet
            st.session_state.chat_history.append((user_input, answer))
            st.session_state.last_answer = answer

            # TTS (oto + spinner)
            with st.spinner("🔊 Cevap için ses üretiliyor..."):
                run_tts_for_answer(answer)
            tts_time = st.session_state.last_tts_time
            audio_path = st.session_state.last_audio_path

            # Logla
            log_interaction(
                mode="text",
                query=user_input,
                rewritten_query=rewritten_query,
                transcript=None,
                answer=answer,
                retrieved_passages=retrieved_docs,
                guardrail_reasons=reasons,
                tts_time=tts_time,
                tts_path=audio_path,
            )

        # Son cevabın sesi ve TTS süresi (her etkileşimden sonra güncellenir)
        if st.session_state.last_audio_path:
            st.markdown("---")
            st.subheader("🔊 Son Cevap (TTS)")

            audio_path = Path(st.session_state.last_audio_path)
            mime = (
                "audio/mpeg" if audio_path.suffix.lower() == ".mp3" else "audio/wav"
            )
            try:
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                st.audio(audio_bytes, format=mime)
            except Exception as e:
                st.error(f"Ses dosyası okunamadı: {e}")

            if st.session_state.last_tts_time is not None:
                st.caption(
                    f"TTS süresi: {st.session_state.last_tts_time:.2f} saniye "
                    "(cevabın ses haline getirilme süresi)."
                )


if __name__ == "__main__":
    main()
