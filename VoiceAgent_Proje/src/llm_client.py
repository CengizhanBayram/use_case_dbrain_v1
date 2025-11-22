import asyncio
import google.generativeai as genai
from pathlib import Path
from src.config import GENERATION_MODEL, EMBEDDING_MODEL

class GeminiClient:
    def __init__(self):
        # Tek bir model her şeyi yapacak (Ses + Metin + Akıl Yürütme)
        self.model = genai.GenerativeModel(GENERATION_MODEL)

    def get_embeddings_batch(self, documents, batch_size=10):
        """Dökümanları vektöre çevirir."""
        embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            try:
                result = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=batch,
                    task_type="retrieval_document"
                )
                embeddings.extend(result['embedding'])
            except Exception as e:
                print(f"Embedding hatası: {e}")
        return embeddings

    def get_query_embedding(self, text):
        """Sorgu için vektör alır."""
        try:
            return genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_query"
            )['embedding']
        except Exception as e:
            print(f"Sorgu hatası: {e}")
            return []

    def transcribe_audio(self, audio_path):
        """
        GEMINI ASR (Ses Tanıma):
        Whisper YERİNE Gemini'nin ses duyma özelliğini kullanıyoruz.
        Dosyayı 'Inline Data' olarak gönderdiğimiz için çok hızlıdır.
        """
        path = Path(audio_path)
        if not path.exists():
            return "Hata: Ses dosyası bulunamadı."

        # 1. Ses dosyasını byte (veri) olarak oku
        audio_data = path.read_bytes()

        # 2. Prompt: "Bunu yazıya çevir"
        prompt_parts = [
            "Bu ses dosyasındaki konuşmayı kelimesi kelimesine Türkçe metin olarak yaz. Sadece dediklerini yaz, yorum yapma.",
            {
                "mime_type": "audio/mp3", # Gradio genelde mp3 veya wav verir
                "data": audio_data
            }
        ]

        # 3. Gemini'ye gönder
        try:
            response = self.model.generate_content(prompt_parts)
            return response.text.strip()
        except Exception as e:
            return f"ASR Hatası: {str(e)}"

    async def generate_rag_response_async(self, query, context):
        """RAG cevabı üretir."""
        prompt = f"""
        Sen Türkçe haber asistanısın.
        
        [BAĞLAM]:
        {context}
        
        [SORU]:
        {query}
        
        GÖREV:
        Sadece bağlamdaki bilgiyi kullanarak soruyu KISA ve ÖZ (maksimum 2 cümle) cevapla.
        """
        response = await self.model.generate_content_async(prompt)
        return response.text