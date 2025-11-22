# src/vector_store.py

import faiss
import numpy as np
import pickle
from pathlib import Path

from src.config import (
    RETRIEVAL_TOP_K,
    FAISS_INDEX_PATH,
    DOCS_PATH,
    EMBEDDINGS_PATH,
)


class VectorStore:
    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        docs_path: Path = DOCS_PATH,
        embeddings_path: Path = EMBEDDINGS_PATH,
    ):
        self.index = None
        self.documents = []          # metin parçaları
        self.embedding_matrix = None # numpy array (n_docs, dim)

        self.index_path = Path(index_path)
        self.docs_path = Path(docs_path)
        self.embeddings_path = Path(embeddings_path)

    # ---------- INDEX OLUŞTURMA + KAYDETME ----------

    def build_index(self, documents, embeddings, save: bool = True):
        """
        Metinleri ve embeddingleri alır, FAISS index oluşturur.
        İstenirse index ve dokümanları diske kaydeder.
        """
        if not embeddings:
            print("⚠️ Vektör verisi boş, index oluşturulamadı.")
            return

        self.documents = list(documents)

        embedding_matrix = np.array(embeddings).astype("float32")
        self.embedding_matrix = embedding_matrix

        dim = embedding_matrix.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embedding_matrix)

        if save:
            self.save_to_disk()

    def save_to_disk(self):
        """FAISS index + dokümanlar + embeddingleri diske yazar."""
        if self.index is None or self.embedding_matrix is None:
            print("Kaydedilecek index veya embedding bulunamadı.")
            return

        # FAISS index
        faiss.write_index(self.index, str(self.index_path))

        # Dokümanlar
        with open(self.docs_path, "wb") as f:
            pickle.dump(self.documents, f)

        # Embedding matrisi
        np.save(self.embeddings_path, self.embedding_matrix)

        print(f"✅ Vector store diske kaydedildi: {self.index_path}")

    def load_from_disk(self) -> bool:
        """
        Diskteki index/doküman/embedding dosyalarını yükler.
        Başarılıysa True, aksi halde False döner.
        """
        if not (
            self.index_path.exists()
            and self.docs_path.exists()
            and self.embeddings_path.exists()
        ):
            return False

        self.index = faiss.read_index(str(self.index_path))

        with open(self.docs_path, "rb") as f:
            self.documents = pickle.load(f)

        self.embedding_matrix = np.load(self.embeddings_path)

        print(
            f"✅ Vector store diskten yüklendi. Doküman sayısı: {len(self.documents)}"
        )
        return True

    # ---------- ARAMA + RERANKER ----------

    def search(
        self,
        query_embedding,
        k: int = RETRIEVAL_TOP_K,
        use_rerank: bool = True,
        enlarge_factor: int = 3,
    ):
        """
        Sorgu vektörüne en yakın dökümanları getirir.
        - Önce FAISS ile kaba arama (k * enlarge_factor)
        - Sonra istenirse cosine similarity ile yeniden sıralama (rerank)
        """
        if self.index is None or self.embedding_matrix is None:
            print("Index henüz yüklenmemiş veya oluşturulmamış.")
            return []

        if query_embedding is None or len(query_embedding) == 0:
            print("Boş sorgu embedding'i verildi.")
            return []

        q_vec = np.array([query_embedding]).astype("float32")

        # Daha çok aday al → sonra rerank et
        k_search = max(k * enlarge_factor, k)
        k_search = min(k_search, len(self.documents))

        distances, indices = self.index.search(q_vec, k_search)
        candidate_indices = indices[0]

        candidate_docs = [self.documents[i] for i in candidate_indices]

        # Rerank devre dışı ise FAISS sıralamasını kullan
        if not use_rerank:
            return candidate_docs[:k]

        # ---- Basit Cosine Reranker ----
        # FAISS L2’ye göre çekti, biz cosine similarity ile tekrar sıralıyoruz.
        cand_embs = self.embedding_matrix[candidate_indices]  # (k_search, dim)

        # Normalize et (cosine)
        q = np.array(query_embedding, dtype="float32")
        q = q / (np.linalg.norm(q) + 1e-8)

        cand_norm = cand_embs / (
            np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-8
        )

        scores = cand_norm @ q  # (k_search,)

        # Skora göre büyükten küçüğe sırala
        sorted_idx = np.argsort(scores)[::-1][:k]

        reranked_docs = [candidate_docs[i] for i in sorted_idx]
        return reranked_docs
