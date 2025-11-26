
# 🎙️ Türkçe Voice Agent (ASR + RAG + TTS)

Bu proje, **OpenSLR 108 – Turkish Broadcast News Speech** verisindeki transcript’lerden bir **knowledge base** kurup, bu bilgi tabanı üzerinden çalışan **Türkçe bir voice-based question answering agent** geliştirmek için hazırlandı.

Case dokümanındaki gereksinimler birebir karşılanmıştır:

1. **Transcriptlerden Knowledge Base** ✅  
2. **Text Q&A (retrieval + LLM)** ✅  
3. **Voice Agent (ASR + RAG + TTS)** ✅  

---

## 🔍 Genel Mimari

Sistem üç ana pipeline’dan oluşur:

1. **Knowledge Base Oluşturma (Offline / İlk Kurulum)**  
   - `DataLoader` ile OpenSLR transcript’leri okunur, temizlenir ve parçalara bölünür.  
   - `Gemini` embedding modeli ile her parça vektöre dönüştürülür.  
   - `FAISS` ile bir vektör veritabanı oluşturulur ve disk üzerinde saklanır.  

2. **Text Q&A (RAG + LLM)**  
   - Kullanıcı Streamlit arayüzünden **Türkçe bir soru yazar**.  
   - Soru istenirse **query rewriting** ile daha arama-dostu bir cümleye dönüştürülür.  
   - Vektör veritabanından en alakalı pasajlar (`top_k`, `reranker` ayarlı) çekilir.  
   - Gemini generative model, **sadece bu bağlamı kullanarak** kısa ve net bir Türkçe cevap üretir.  
   - Cevap hem ekrana yazılır, hem de TTS ile sese dönüştürülür.

3. **Voice Agent (ASR + RAG + TTS)**  
   - Kullanıcı:
     - ya **ses dosyası yükleyebilir** (wav/mp3),
     - ya da **mikrofondan konuşabilir** (audio waveform ile görsel geri bildirim).
   - Ses, Gemini üzerinden **ASR ile transcript’e** çevrilir.  
   - Bu transcript, RAG pipeline’ına soru olarak verilir.  
   - LLM cevabı üretir, guardrail’lerden geçer.  
   - Cevap **Türkçe TTS (gTTS)** ile audio olarak synthesize edilir ve player’da dinlenebilir.  

Ek olarak:

- **Guardrails**: Cevaplar üzerinde basit güvenlik ve içerik filtreleri uygulanır.  
- **JSONL Loglama**: Her etkileşim `logs/interaction_log.jsonl` dosyasına kaydedilir (soru, transcript, cevap, kullanılan pasajlar, TTS süreleri, guardrail kararları vb.).


## 🧱 Kullanılan Teknolojiler

- **Arayüz**: [Streamlit](https://streamlit.io/)
- **LLM & Embedding**: Google Gemini API
  - `models/text-embedding-004` (embedding)
  - `gemini-2.5-flash` (generation + ASR)
- **Vektör Veritabanı**: FAISS
- **TTS (Text-to-Speech)**: `gTTS` (Google Text-to-Speech, Türkçe ses)
- **Dataset**: [OpenSLR 108 – Turkish Broadcast News Speech](https://www.openslr.org/108/)  
- **Dil**: Python 3.10+

---

## 📁 Proje Yapısı (Özet)

```plaintext
VoiceAgent_Proje/
├─ app.py                  # Streamlit arayüzü ve ana pipeline
├─ .env                    # GOOGLE_API_KEY vb.
├─ requirements.txt
├─ data/                   # OpenSLR transcript dosyaları (.stm / .txt)
├─ logs/
│   └─ interaction_log.jsonl  # JSONL formatında etkileşim logları
└─ src/
    ├─ config.py           # MODEL isimleri, path’ler, TTS ayarları vb.
    ├─ data_loader.py      # OpenSLR transcript’lerini okuma/temizleme/chunk
    ├─ llm_client.py       # GeminiClient: embedding, ASR, RAG cevabı
    ├─ vector_store.py     # FAISS tabanlı vektör veritabanı + persist
    ├─ tts_service.py      # TTSService: metni Türkçe sese çevirme
    └─ guardrails.py       # apply_guardrails: basit güvenlik/filtre kuralları
````

---

## ⚙️ Kurulum

### 1. Ortam Hazırlığı

```bash
# (Opsiyonel) Sanal ortam
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Gereksinimleri yükle
pip install -r requirements.txt
```

`requirements.txt` içinde özetle:

* `streamlit`
* `audio-recorder-streamlit`
* `google-generativeai`
* `faiss-cpu`
* `gTTS`
* vb. bağımlılıklar yer alır.

### 2. Ortam Değişkenleri

Proje köküne bir `.env` dosyası ekleyin:

```env
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY_HERE
```

`src/config.py` içindeki:

* `GENERATION_MODEL`
* `EMBEDDING_MODEL`
* TTS dili (`TTS_LANG = "tr"`)
* FAISS index path’leri

gerektiğinde buradan ayarlanabilir.

### 3. Dataset Kurulumu (OpenSLR 108)

1. [OpenSLR 108](https://www.openslr.org/108/) veri setini indirin.
2. İçindeki transcript dosyalarını (genelde `.stm` veya `.txt`) `data/` klasörüne yerleştirin.

Örnek:

```plaintext
VoiceAgent_Proje/
└─ data/
   ├─ train/
   │   ├─ xxx.stm
   │   └─ ...
   └─ dev/
       ├─ yyy.stm
       └─ ...
```

`DataLoader`, `data/` altında recursive olarak `.txt` yoksa `.stm` uzantılı dosyaları okur.

---

## ▶️ Çalıştırma

Proje kök dizininde:

```bash
streamlit run app.py
```

Tarayıcıda otomatik açılmazsa, terminalde verilen URL’yi (genelde `http://localhost:8501`) açabilirsiniz.

---

## 🧠 Knowledge Base: Ne Yapıyor?

`build_knowledge_base()` fonksiyonu:

1. `data/` klasöründeki transcript dosyalarını `DataLoader` ile okur.
2. `clean_text` ile:

   * Zaman damgalarını (`00:00:12.40`),
   * XML benzeri etiketleri (`<spk1>`, vb.),
   * Köşeli parantez içi notları (`[noise]`, `[laugh]`),
   * Fazla boşlukları, gereksiz karakterleri temizler.
3. `chunk_text` ile ~500 karakterlik, 50 karakter overlap’li parçalara böler.
4. `GeminiClient.get_embeddings_batch` ile her parça için embedding üretir.
5. `VectorStore.build_index` ile FAISS index oluşturur ve:

   * Embedding vektörlerini,
   * Metin parçalarını,
   * Disk üzerinde kalıcı olarak saklar (bir sonraki açılışta tekrar hesaplamaya gerek kalmaz).

---

## 💬 Text Q&A Kullanımı

Sağ tarafta ayarları yaptıktan sonra:

* Ana ekrandaki **“💬 Sohbet (Text Q&A)”** bölümünden bir soru yazabilirsiniz.
* Örnek sorular:

  * “Son günlerde ekonomi haberlerinde hangi başlıklardan bahsediliyor?”
  * “Spor haberlerinde hangi takımlar öne çıkıyor?”

Pipeline:

1. Soru → opsiyonel **query rewriting** ile netleştirilir.
2. Embedding alınır → FAISS üzerinden **top-k** benzer pasajlar çekilir.
3. Pasajlar + soru, Gemini’ye prompt olarak verilir.
4. Cevap:

   * Chat alanında gösterilir,
   * TTS ile sese çevrilir ve player’da dinlenebilir.
5. Kullanılan pasajlar ve rewritten query, “🔎 Kullanılan pasajlar / Query rewrite” expander’ında gösterilir.

---

## 🎤 Sesli Soru Kullanımı

Arayüzün sol tarafında **“🎤 Sesli Soru”** bölümü vardır.

### 1) Dosyadan Yükleme

* `Ses dosyası yükle (wav/mp3)` alanından bir ses dosyası seçin.
* Ardından **“📂 Yüklenen sesle soruyu çalıştır”** butonuna basın.

Arka planda:

1. Ses dosyası geçici olarak kaydedilir.
2. `GeminiClient.transcribe_audio` ile **ASR** yapılır → transcript metin oluşur.
3. Transcript, `generate_rag_answer` ile RAG pipeline’ına verilir.
4. Cevap TTS ile sese çevrilir.
5. `chat_history` içine:

   * Kullanıcı mesajı: `📂 (Dosya) <transcript>`
   * Bot mesajı: `<cevap>`
     olarak eklenir.

### 2) Mikrofondan Kayıt

* **“🎙️ Kaydı başlat / durdur”** butonuna basınca üstte bir **ses dalgası (waveform)** görünür → kayıt alınıyor demektir.
* Tekrar basınca kayıt durur, “Durum: Kayıt tamamlandı, ses işleniyor...” mesajını görürsünüz.
* Kayıt:

  * `handle_voice_bytes` fonksiyonuna gider,
  * ASR → RAG → TTS pipeline’ı çalışır,
  * Chat alanına `🎙️ (Mikrofon) <transcript>` + cevap mesajı eklenir.

### ASR Transcript’i Görüntüleme

* Son yapılan sesli işlem için ASR sonucu, sağdaki panelde **“Son ASR Transcript”** bölümünde text olarak gösterilir.

---

## ✅ Guardrails & Güvenlik

`src/guardrails.py` içinde tanımlanan `apply_guardrails` fonksiyonu şu amaçlarla kullanılır:

* Cevap metni üzerinde basit filtreleme / düzenleme yapmak,
* Gerektiğinde uyarı mesajlarına dönüştürmek,
* Log’lara “hangi guardrail ne karar verdi?” bilgisini eklemek.

`log_interaction` fonksiyonuna her cevaptan sonra `guardrail_reasons` dict’i geçirilir. JSONL log’larda:

* `guardrail_reasons`: `{ "safety_rule_x": true/false, ... }` gibi kayıtlar tutulur.

---

## 📊 Loglama & Analiz

Tüm etkileşimler:

```text
logs/interaction_log.jsonl
```

dosyasına **JSONL** formatında yazılır. Her satır bir etkileşimi temsil eder:

* `timestamp`
* `mode` (`"text"` veya `"voice"`)
* `query` (kullanıcının sorusu / transcript)
* `rewritten_query` (query rewriting sonrası soru)
* `transcript` (sesli modda ASR sonucu)
* `answer` (LLM cevabı)
* `retrieved_passages` (RAG’de kullanılan pasajlar)
* `guardrail_reasons`
* `tts_time` (cevabın sese dönüştürülme süresi)
* `tts_path` (oluşturulan ses dosyasının yolu)

Bu yapı sayesinde:

* Case tesliminde istenen **“3–5 örnek için audio input, transcript, text cevap, TTS output”** bilgileri kolayca log dosyasından çıkarılabilir.
* Daha sonra offline analiz / model geliştirme için bu loglar doğrudan kullanılabilir.

---

## 🧪 Case Gereksinimleri ile Doğrudan Eşleşme

Case PDF’teki maddeler ve projedeki karşılıkları:

1. **Transcriptlerden Knowledge Base**

   * `DataLoader.load()`
   * `GeminiClient.get_embeddings_batch`
   * `VectorStore.build_index` + disk persist
   * `build_knowledge_base()` (otomatik ilk çalıştırmada devreye girer)

2. **Text Q&A (Retrieval + LLM)**

   * Kullanıcı metin sorusunu `st.chat_input` ile girer.
   * `generate_rag_answer()` içinde:

     * query rewriting (opsiyonel),
     * FAISS üzerinden top-k retrieval,
     * Gemini ile bağlamsal cevap üretimi,
     * guardrails ile post-process.

3. **Voice Agent (ASR + RAG + TTS)**

   * **ASR (a)**:

     * `GeminiClient.transcribe_audio` (dosya + mikrofon için)
   * **RAG + LLM (b)**:

     * Transcript → `generate_rag_answer(transcript)`
   * **TTS (c)**:

     * `run_tts_for_answer(answer)` → `TTSService.text_to_speech`

Arayüz tarafında:

* Ses kaydı alınırken waveform ile görsel geri bildirim,
* Cevap süresi (ASR süresi, RAG süresi, toplam) çıktıları,
* TTS süresi,
* Kullanılan pasajlar ve query rewrite detayları,
  demoda anlatım için ek bilgi olarak sunulmaktadır.

---

## 🔮 Geliştirme Fikirleri

* **KV Cache / Response Caching**:
  Sık sorulan sorular için embedding + cevabı cache’leyip latency’i daha da düşürmek.
* **Daha gelişmiş Guardrails**:
  Domain-specific kurallar (örneğin: finans, tıp, hukuk) ile riskli cevapları sınırlamak.
* **Gelişmiş Reranker**:
  Gemini veya başka bir cross-encoder model ile zengin reranking katmanı eklemek.
* **ASR & TTS Seçenekleri**:
  Farklı diller veya farklı TTS ses profilleri (kadın/erkek, hız, tonlama vb.).

---

Bu README, projeyi hem case jürisine hem de başka bir geliştiriciye rahatça anlatabilecek seviyede tasarlandı.
Demo sırasında:

* “Text Q&A”,
* “Voice (dosyadan)”,
* “Voice (mikrofon)”,
  pipeline’larını ayrı ayrı gösterebilir,
  sidebar’daki **Top-k / Rerank / Query Rewrite** seçenekleriyle sistemin davranışını canlı olarak değiştirebilirsin. 🚀

```
```
