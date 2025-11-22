import sys
import os
from pathlib import Path

# Proje ana dizinini path'e ekle
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.data_loader import DataLoader
from src.llm_client import GeminiClient
from src.vector_store import VectorStore

def main():
    print("################################################")
    print("   GERÇEK VERİ SETİ - KNOWLEDGE BASE OLUŞTURUCU")
    print("################################################")
    print("Bu script, bilgisayarındaki OpenSLR (veya herhangi bir) veri setini tarar")
    print("ve Transcriptleri Vektör Veritabanına (FAISS) dönüştürür.\n")

    # 1. Kullanıcıdan Veri Setinin Yerini İste
    # Örn: C:\Users\cengh\Downloads\openslr_turkish\data
    dataset_path = input("👉 Lütfen veri setinin (transcriptlerin) olduğu klasör yolunu yapıştırın: ").strip()
    
    # Tırnak işaretlerini temizle (Windows'ta bazen "yol" şeklinde gelir)
    dataset_path = dataset_path.replace('"', '').replace("'", "")
    
    if not os.path.exists(dataset_path):
        print("❌ HATA: Belirtilen klasör bulunamadı!")
        return

    # 2. Servisleri Başlat
    print("\n⚙️  Servisler başlatılıyor...")
    loader = DataLoader(root_directory=dataset_path)
    gemini = GeminiClient()
    store = VectorStore()

    # 3. Dosyaları Oku (Recursive)
    documents = loader.scan_and_load()
    
    if not documents:
        print("❌ İşlem iptal edildi. Okunacak veri yok.")
        return

    # 4. Embedding Oluştur (Paralel/Batch)
    print("\n🧠 Metinler vektörlere çevriliyor (Bu işlem veri boyutuna göre sürebilir)...")
    # Veri çok büyükse (örn: 10.000+ dosya) burası zaman alır.
    vectors = gemini.get_embeddings_batch(documents, batch_size=50) # Batch size'ı artırdık

    # 5. FAISS Index'i Kaydet
    print(f"\n💾 Veritabanı oluşturuluyor (Vektör Sayısı: {len(vectors)})...")
    store.build_index(documents, vectors)
    
    # 6. (Opsiyonel) Indexi Diske Kaydetme Özelliği Eklenebilir
    # Şimdilik app.py her açılışta bellekte tutuyor ama gerçek projede buraya 
    # faiss.write_index(store.index, "my_index.faiss") eklenir.
    # Bizim yapımızda app.py'yi açınca tekrar yüklemesi gerekecek, 
    # ama bu script en azından verinin okunabilir olduğunu test eder.

    print("\n🎉 BAŞARILI! Veri seti tarandı ve Knowledge Base oluşturulabilir durumda.")
    print("Şimdi 'app.py' dosyasını çalıştırıp 'Bilgi Bankasını Kur' dediğinde,")
    print("kodun içindeki DATA_DIR yolunu bu yeni yol ile güncellemeyi unutma!")

if __name__ == "__main__":
    main()