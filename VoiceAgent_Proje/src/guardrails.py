# src/guardrails.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class GuardrailResult:
    """
    Guardrail çıktısı:
    - answer: Kullanıcıya gösterilecek son cevap
    - blocked: Cevap tamamen bloklandı mı?
    - reasons: Hangi kurallar tetiklendi (dict)
    """
    answer: str
    blocked: bool
    reasons: Dict[str, bool]



#Buraya gerçek küfür/hakaretleri sen ekleyebilirsin.
PROFANITY_KEYWORDS: List[str] = [
    # "örnek_küfür_1",
    # "örnek_küfür_2",
]

SENSITIVE_TOPICS: List[str] = [
    "intihar",
    "kendime zarar",
    "kendi canıma kıy",
    "bomba yap",
    "patlayıcı yap",
    "silah yap",
    "uyuşturucu üret",
    "nefret söylemi",
]


def _contains_keyword(text: str, keywords: List[str]) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in keywords)


def _too_long(text: str, max_chars: int = 600) -> bool:
    return len(text) > max_chars


def apply_guardrails(
    answer: str,
    query: str = "",
    context: str = "",
) -> GuardrailResult:
    """
    LLM cevabına basit güvenlik ve kalite guardrail'leri uygular.

    Kurallar:
    - Küfür / hakaret içeriyorsa -> tamamen blokla
    - Çok hassas konular (self-harm, bomba, vs.) -> tamamen blokla
    - Cevap çok uzunsa -> kısalt
    - Hallucination ipucu varsa -> uyarı ekle
    """
    reasons = {
        "profanity": False,
        "sensitive": False,
        "too_long": False,
        "hallucination_hint": False,
    }

    if not answer:
        return GuardrailResult(answer="", blocked=False, reasons=reasons)

    # 1) Küfür / hakaret
    if _contains_keyword(answer, PROFANITY_KEYWORDS):
        reasons["profanity"] = True

    # 2) Hassas konular
    if _contains_keyword(answer, SENSITIVE_TOPICS):
        reasons["sensitive"] = True

    # 3) Aşırı uzun cevap
    if _too_long(answer, max_chars=600):
        reasons["too_long"] = True

    # 4) Çok basit hallucination sinyali
    hallucination_markers = [
        "bağlamda geçmiyor",
        "tahmin ediyorum",
        "uyduruyorum",
        "kesin olmamakla birlikte",
        "tam emin değilim ama",
    ]
    if _contains_keyword(answer, hallucination_markers):
        reasons["hallucination_hint"] = True

    # ----------------- POLICY -----------------

    # Küfür veya çok hassas konu varsa -> tamamen blokla
    if reasons["profanity"] or reasons["sensitive"]:
        safe_msg = (
            "Bu soru veya cevap güvenlik kurallarına uygun değil, "
            "bu nedenle detay veremiyorum."
        )
        return GuardrailResult(
            answer=safe_msg,
            blocked=True,
            reasons=reasons,
        )

    # Sadece çok uzunsa -> kısalt
    if reasons["too_long"]:
        trimmed = answer[:550].rsplit(" ", 1)[0] + "..."
        safe_msg = trimmed + "\n\n_(Cevap, uzunluk sınırı nedeniyle kısaltıldı.)_"
        return GuardrailResult(
            answer=safe_msg,
            blocked=False,
            reasons=reasons,
        )

    # Hallucination ipucu varsa -> altına not ekle
    if reasons["hallucination_hint"]:
        safe_msg = (
            answer
            + "\n\n> Not: Bu cevap model tarafından tam bağlama dayanmıyor olabilir."
        )
        return GuardrailResult(
            answer=safe_msg,
            blocked=False,
            reasons=reasons,
        )

    # Hiçbir kural tetiklenmediyse -> direkt döndür
    return GuardrailResult(
        answer=answer,
        blocked=False,
        reasons=reasons,
    )
