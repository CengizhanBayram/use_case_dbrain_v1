# src/tts_service.py

from gtts import gTTS
from pathlib import Path

from src.config import OUTPUT_AUDIO_FILE, TTS_LANG, OUTPUT_DIR


class TTSService:
    """
    Basit Türkçe TTS servisi.
    gTTS kullanarak metni mp3 dosyasına çevirir ve dosya yolunu döner.
    """

    def __init__(self, default_output_path: Path = OUTPUT_AUDIO_FILE):
        # Varsayılan çıktı dosyası (ör: teslimat_ornekleri/response_output.mp3)
        self.default_output_path = Path(default_output_path)

    def text_to_speech(self, text: str, output_path: str | Path | None = None) -> str | None:
        """
        Metni sese çevirir, mp3 olarak kaydeder ve dosya yolunu (str) döner.
        Hata durumunda None döner.
        """
        try:
            # Boş metin kontrolü
            if not text or len(text.strip()) == 0:
                print("TTS: Boş metin verildi, ses üretilmedi.")
                return None

            # Çıkış yolu verilmemişse varsayılanı kullan
            if output_path is None:
                # Klasörden emin ol
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                output_path = self.default_output_path

            save_path = Path(output_path)

            # gTTS ile Türkçe ses üret
            tts = gTTS(text=text, lang=TTS_LANG, slow=False)
            tts.save(str(save_path))

            print(f"✅ TTS: Ses dosyası üretildi -> {save_path}")
            return str(save_path)

        except Exception as e:
            print(f"TTS Hatası: {e}")
            return None
