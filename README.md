
## 📚 Kullanılan Kaynaklar

Bu case kapsamında **aynı problem** için **iki farklı çözüm** geliştirilmiştir:

1. **Çözüm 1 – Streamlit + Gemini tabanlı Voice Agent**
   - Bilgi bankası (knowledge base) için transcriptlerden elde edilen metinler FAISS tabanlı bir vektör veritabanına kaydedilmiştir.
   - Sorgular, bu vektör veritabanı üzerinden **RAG (Retrieval-Augmented Generation)** ile ilgili pasajlar alınarak LLM (Gemini) ile cevaplanmıştır.
   - Türkçe cevaplar, harici bir TTS servisi kullanılarak sese dönüştürülmüştür.
   - Arayüz Streamlit ile geliştirilmiş ve hem **text Q&A** hem de **voice Q&A (ASR + RAG + TTS)** desteği sağlanmıştır.

2. **Çözüm 2 – Azure tabanlı Cloud Çözüm**
   - Aynı veri setinden oluşturulan bilgi bankası bu kez Azure ortamında kullanılmıştır.
   - ASR, LLM ve TTS bileşenleri için Azure servislerinden (Azure OpenAI / Azure Speech vb.) yararlanılmıştır.
   - Sorgular yine offline olarak oluşturulan vektör veritabanı üzerinden alınmış, böylece her iki çözümde de **ortak knowledge base** kullanılmıştır.
   - Bu çözüm, bulut üzerinde çalışan, ölçeklenebilir bir alternatif olarak tasarlanmıştır.

### 🎧 Veri Seti

Her iki çözümde de aşağıdaki veri seti kullanılmıştır:

- **OpenSLR 108 – Turkish Broadcast News Speech**  
  - Kaynak: https://www.openslr.org/108/  
  - Bu veri setindeki:
    - **Audio** dosyaları, **ASR (otomatik konuşma tanıma)** performansını test etmek ve sesli sorular için giriş olarak kullanmak amacıyla,
    - Veri seti ile birlikte gelen veya önceden işlenmiş **transcript**’ler ise, haber içeriklerinden oluşan bir **knowledge base** kurmak ve RAG pipeline’ını beslemek amacıyla kullanılmıştır.

### 📄 Case Dokümanı

Proje boyunca, organizatörler tarafından sağlanan:

<img width="632" height="829" alt="image" src="https://github.com/user-attachments/assets/6a33d17e-3f84-45c7-b96f-654426abed4e" />
<img width="622" height="662" alt="image" src="https://github.com/user-attachments/assets/5c50aaf5-88ca-4fe4-a4d1-354b32c74d6f" />
referans alınmış, sistem mimarisi ve değerlendirme senaryoları bu dokümandaki gereksinimlere göre tasarlanmıştır.

