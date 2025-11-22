import os
import glob
import re

class DataLoader:
    def __init__(self, data_folder_path):
        """
        DataLoader sınıfı başlatılırken veri klasörünün yolunu alır.
        """
        self.data_folder_path = data_folder_path

    def clean_text(self, text):
        """
        OpenSLR verisindeki gürültüleri, zaman damgalarını ve etiketleri temizler.
        """
        # Zaman damgaları (Örn: 00:00:15.40)
        text = re.sub(r'\d{2}:\d{2}:\d{2}(\.\d+)?', '', text)
        # XML etiketleri (Örn: <spk1>)
        text = re.sub(r'<[^>]+>', '', text)
        # Gereksiz semboller
        text = re.sub(r'\[.*?\]', '', text)
        # Fazla boşluklar
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def chunk_text(self, text, chunk_size=500, overlap=50):
        """
        Metni parçalara böler.
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space != -1 and last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap if (end - overlap) > start else end
        return chunks

    def load(self):
        """
        Klasördeki dosyaları okur, temizler, parçalar ve listeyi döndürür.
        """
        print(f"Veri yükleniyor: {self.data_folder_path}...")
        processed_docs = []
        
        # Hem txt hem stm dosyalarına bak
        file_paths = glob.glob(os.path.join(self.data_folder_path, "**/*.txt"), recursive=True)
        if not file_paths:
            file_paths = glob.glob(os.path.join(self.data_folder_path, "**/*.stm"), recursive=True)

        print(f"Bulunan dosya sayısı: {len(file_paths)}")

        for file_path in file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
                    cleaned_text = self.clean_text(raw_text)
                    
                    if len(cleaned_text) < 20:
                        continue
                    
                    chunks = self.chunk_text(cleaned_text)
                    processed_docs.extend(chunks)
            except Exception as e:
                print(f"Hata ({file_path}): {e}")

        print(f"İşlem tamamlandı. Toplam {len(processed_docs)} adet parça oluşturuldu.")
        return processed_docs