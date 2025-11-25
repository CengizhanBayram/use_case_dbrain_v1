<img width="863" height="683" alt="image" src="https://github.com/user-attachments/assets/48a7f042-2b06-40f1-8311-61c0b44f404b" />

<img width="567" height="835" alt="image" src="https://github.com/user-attachments/assets/cb5dc676-5708-4a88-b1f8-6f7739512993" />

Aynen şimdi bu hale geldiğine göre README’yi yazalım. Aşağıyı direkt `README.md` olarak kaydedebilirsin. ✨

---

````markdown
# Azure Türkçe Voice-based Q&A Agent (ASR + RAG + TTS)

Bu proje, **Türkçe haber transkriptlerinden** oluşturulmuş bir bilgi tabanı üzerinde çalışan, uçtan uca **ses tabanlı soru-cevap (voice-based Q&A) ajanı**dır.

Pipeline:

- 🎙 **ASR (Whisper, Azure OpenAI)**  
- 🔎 **RAG (Gemini Embedding + FAISS VectorStore)**  
- 🧠 **LLM (Azure OpenAI Chat)**  
- 🔊 **TTS (Azure OpenAI TTS)**  
- 💻 **Web Arayüzü (HTML + JS, FastAPI üzerinden servis ediliyor)**  

Kullanıcı hem **metinle** hem de **sesli** olarak soru sorabilir, cevapları hem ekranda görür hem de **Türkçe seslendirilmiş** olarak dinleyebilir.

---

## 1. Mimari Genel Bakış

Yüksek seviye akış:

1. **Knowledge Base (offline hazırlık)**  
   - OpenSLR 108 veri setindeki **Türkçe haber transkriptleri** alınır.  
   - Metinler parçalara (chunk) bölünür.  
   - Her parça, **Gemini `models/text-embedding-004`** ile vektörize edilir.  
   - Vektörler **FAISS index** içinde saklanır, ham metinler `docs.pkl` olarak kaydedilir.

2. **Text Q&A (online)**  
   - Kullanıcı Türkçe bir soruyu metin olarak girer.  
   - Soru, yine Gemini embedding modeliyle vektörize edilir.  
   - FAISS üzerinden en alakalı `top_k` haber parçaları çekilir.  
   - Bu parçalar ve soru, Azure OpenAI Chat deployment’ına verilerek **Türkçe cevap** üretilir.  
   - Cevap ve kullanılan bağlam parçaları (contexts) frontend’e döner.

3. **Voice Q&A (online)**  
   - Kullanıcı mikrofonla soru sorar.  
   - Tarayıcı ses kaydını backend’e gönderir (`/voice-qa`).  
   - Azure OpenAI Whisper deployment’ı ile **Türkçe transcript** üretilir.  
   - RAG pipeline (Embedding + FAISS + Chat) transcript üzerinden çalışır.  
   - Azure OpenAI TTS deployment’ı ile Türkçe cevap **seslendirilir**.  
   - Frontend, hem cevabı yazar hem de sesi otomatik çalmaya çalışır.

---

## 2. Proje Yapısı

Önemli dosya/klasörler:

```text
Azure_Solution/
├─ main.py               # FastAPI backend, RAG, ASR, TTS, API endpointleri
├─ .env                  # Ortam değişkenleri (API keyler, deployment adları, vb.)
├─ frontend/
│  └─ index.html         # Modern chat arayüzü (Metin + Ses modu)
└─ vector_db/
   ├─ docs.pkl           # Haber parçalarının listesi (text)
   ├─ embeddings.npy     # Her parça için embedding vektörleri
   └─ faiss_index.bin    # FAISS index
````

> Not: `vector_db` klasörü bu projede **önceden oluşturulmuş** kabul ediliyor. Embedding’ler, Gemini `models/text-embedding-004` ile üretilmiştir.

---

## 3. Kullanılan Teknolojiler

* **Backend**

  * [FastAPI](https://fastapi.tiangolo.com/) – HTTP API ve HTML servis
  * [Uvicorn](https://www.uvicorn.org/) – ASGI server
  * [FAISS](https://github.com/facebookresearch/faiss) – Vektör arama
  * `numpy`, `pickle`, `python-dotenv`

* **LLM / ASR / TTS**

  * **Azure OpenAI**

    * Chat: örn. `gpt-4o-mini` deployment
    * Whisper: `whisper` deployment
    * TTS: `tts` deployment, voice: `alloy` (Azure OpenAI TTS)

* **Embedding**

  * **Google Gemini** – `models/text-embedding-004` (via `google-generativeai`)

* **Frontend**

  * Vanilla HTML + CSS + JavaScript
  * Tek sayfa (SPA benzeri) chat UI, FastAPI root (`/`) üzerinden servis ediliyor.

---

## 4. Kurulum

### 4.1. Gereksinimler

* Python **3.10+**
* Pip
* Bir Azure OpenAI kaynağı:

  * Chat deployment (örnek: `gpt-4o-mini`)
  * Whisper ASR deployment (örnek: `whisper`)
  * TTS deployment (örnek: `tts`)
* Google Gemini API Key (embedding için)

### 4.2. Sanal Ortam (opsiyonel ama tavsiye edilir)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 4.3. Python Paketlerini Kur

```bash
pip install fastapi uvicorn
pip install openai
pip install google-generativeai
pip install python-dotenv
pip install faiss-cpu
pip install numpy
```

(eğer gerekirse)

```bash
pip install pydantic
pip install "fastapi[all]"
```

> Projeye bir `requirements.txt` koymak istersen bu paketleri oraya ekleyebilirsin.

---

## 5. .env Yapılandırması

Projenin kök dizininde ( `main.py` ile aynı klasör) bir `.env` dosyası oluştur:

```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=***
AZURE_OPENAI_ENDPOINT=https://<senin-resource-adın>.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Deployment isimleri (Azure Portal > Deployments)
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_WHISPER_DEPLOYMENT=whisper
AZURE_OPENAI_TTS_DEPLOYMENT=tts

# TTS voice (Azure OpenAI TTS)
AZURE_OPENAI_TTS_VOICE=alloy

# Gemini
GEMINI_API_KEY=***

# Embedding modeli
EMBEDDING_MODEL_NAME=models/text-embedding-004

# Vector DB klasörü
VECTOR_DIR=./vector_db

# RAG için varsayılan top_k
TOP_K_DEFAULT=5
```

> Not: `AZURE_OPENAI_*_DEPLOYMENT` değerleri, kendi Azure OpenAI kaynaklarında oluşturduğun **deployment adları** olmalıdır (model adları değil, deployment name).

---

## 6. Vector DB (FAISS) Hakkında

Bu proje, `vector_db` klasöründeki üç dosyanın **hazır olduğunu** varsayar:

* `docs.pkl`:
  Python listesi (`List[str]`) – her eleman, bir haber metni parçası (chunk).
* `embeddings.npy`:
  `float32` matris (`num_docs x dim`) – her satır bir doküman embedding’i.
* `faiss_index.bin`:
  FAISS index dosyası – `embeddings.npy` ile aynı sırada ve boyutta.

Bu dosyalar, ayrı bir preprocessing script’i ile şu şekilde üretilmiştir:

1. OpenSLR 108 Türkçe haber veri setinden transkriptler okunur.
2. Metinler parçalara bölünür ve `docs` listesine alınır.
3. Her parça için Gemini `models/text-embedding-004` ile embedding üretilir.
4. `embeddings.npy` kaydedilir.
5. FAISS index (`IndexFlatL2` vb.) oluşturulur ve `faiss_index.bin` olarak kaydedilir.

> ÖNEMLİ: Sorgu embedding’ini de **aynı modelle** ürettiğimiz için (`models/text-embedding-004`), FAISS aramaları tutarlı çalışır.

---

## 7. Uygulamayı Çalıştırma

### 7.1. Backend’i başlat

Proje kök dizininde:

```bash
python main.py
```

Log’da şuna benzer bir şey görmelisin:

```text
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

### 7.2. Frontend’e erişim

Tarayıcıdan:

```text
http://127.0.0.1:8000
```

* `main.py` root (`/`) endpoint’inde `frontend/index.html` dosyasını döndüğü için, chat arayüzü otomatik açılır.
* API çağrıları da aynı origin üzerinden (ör. `http://127.0.0.1:8000/text-qa`) gider.

---

## 8. API Endpointleri

### 8.1. `GET /`

* `frontend/index.html` dosyasını döndürür.
* Chat UI’yi açmak için kullanılır.

### 8.2. `GET /health`

Basit healthcheck endpoint’i:

```json
{
  "status": "ok"
}
```

### 8.3. `POST /text-qa`

**Input (JSON)**

```json
{
  "question": "Türkiye ekonomisiyle ilgili haberlerde neler vurgulanıyor?",
  "top_k": 5
}
```

* `question`: Kullanıcının Türkçe sorusu.
* `top_k` (opsiyonel): FAISS’ten kaç bağlam parçası alınacağı (default: `.env`’deki `TOP_K_DEFAULT`).

**Output (JSON)**

```json
{
  "answer": "Cevap metni...",
  "contexts": [
    "Haber parçası 1...",
    "Haber parçası 2..."
  ]
}
```

* `answer`: Azure OpenAI Chat modeli tarafından üretilmiş cevap.
* `contexts`: Bu cevaba bağlı olarak kullanılan haber metni parçaları.

### 8.4. `POST /voice-qa`

**Input (multipart/form-data)**

* `file`: Tarayıcıdan kaydedilmiş `audio.wav` (tek kanal, PCM, vb.)

**Output (JSON)**

```json
{
  "transcript": "Kullanıcının söylediği cümlenin Türkçe transkripti",
  "answer": "Soruya verilen Türkçe cevap",
  "audio_base64": "<base64-encoded-wav>"
}
```

Frontend, `audio_base64` alanını:

* `data:audio/wav;base64,...` formatına çevirerek
* Hem otomatik çalmaya çalışır (`new Audio().play()`),
* Hem de mesaj balonu içinde `<audio controls>` player olarak gösterir.

---

## 9. Demo İçin Notlar

Görüşme/demoda gösterebileceğin akış:

1. **Metin demo:**

   * Arayüzde “📝 Metin” modunu seç.
   * Haber korpusu ile alakalı bir soru yaz:

     * Örn. “Son dönemde enflasyon hakkındaki haberler ne diyor?”
   * Cevabın altında “📚 Kaynak Bağlamlar” kısmında haber paragraflarını göster.

2. **Ses demo:**

   * “🎤 Sesli” moduna geç.
   * Mikrofon butonuna bas, bir soru sor, tekrar basarak kaydı bitir.
   * UI sırasıyla:

     * Whisper transcript’ini yazı olarak gösterir,
     * Answer’ı balonda gösterip,
     * TTS ile üretilmiş sesi otomatik oynatır.

3. **Out-of-domain soru:**

   * Haber korpusu ile alakasız bir soru sor:

     * Örn. “Mars’ta yaşam var mı?”
   * Backend, kelime kesişimi çok zayıfsa:

     * `"Bu metin haber metinlerinde yok."` cevabını üretir,
     * Böylece “hallucination” yerine net bir fallback davranışı gösterirsin.

---

## 10. Kısıtlar ve İyileştirme Fikirleri

* Autoplay (otomatik ses çalma) tarayıcıların güvenlik politikalarına bağlıdır.
  Bu projede:

  * Kullanıcı mikrofon butonuna tıkladığında hafif bir “audio unlock” tekniğiyle izin tetiklenir.
  * Bazı tarayıcılarda yine de manuel play gerekebilir.
* Embedding ve FAISS index oluşturma adımı bu repoda gösterilmemiştir;
  `vector_db` klasörü hazır varsayılmıştır.
* Whisper transcript kalitesi, mikrofona, konuşma hızına ve gürültüye bağlıdır;
  gerekirse ASR tarafında ek temizlik yapılabilir.

---

Her şey bu kadar 🎧
Projeyi çalıştırdıktan sonra tek yapman gereken `http://127.0.0.1:8000`’i açıp metin veya sesli soru sormak.
Case sunumunda bu README’yi de ekleyerek mimari ve tasarım kararlarını net bir şekilde anlatabilirsin.

```
::contentReference[oaicite:0]{index=0}
```
