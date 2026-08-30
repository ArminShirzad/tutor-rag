"""Sensitive-topic routing, checked BEFORE retrieval.

Why before, and not in the prompt:

1. The refusal gate fires pre-generation. If nothing clears `min_score`, the
   LLM never runs -- so a rule living in the system prompt never executes. A
   harassment report retrieves nothing from a leave policy, which is exactly
   the case where the prompt rule would have been needed.

2. Routing a harassment report to a human is a safety control, not a
   preference. Controls do not belong in a prompt, because a prompt is a
   request the model may decline. Same reasoning as not relying on prompting
   alone to stop hallucination.

Detection is two-layer, because neither layer is sufficient alone:

    keywords    fast, free, deterministic, auditable -- and brittle. Catches
                "آزار" but not someone describing the same thing without ever
                using the word.
    semantic    embeds the question and compares it to example phrases per
                category. Catches "مدیرم دائم سر من داد می‌زند" which shares no
                keyword with "آزار" or "شکایت". Costs one embedding call, which
                we are making anyway for retrieval.

Either layer firing is enough. This is deliberately biased toward false
positives: routing an ordinary question to a human is a mild annoyance, while
missing a real harassment report is not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SensitiveCategory:
    name: str
    keywords: list[str]
    examples: list[str]
    message: str


ESCALATE_FA = "این موضوع باید مستقیماً با همکاران منابع انسانی مطرح شود."

CATEGORIES: list[SensitiveCategory] = [
    SensitiveCategory(
        name="harassment_conflict",
        keywords=[
            "آزار", "اذیت", "توهین", "تحقیر", "داد زد", "داد می‌زن", "فحش",
            "تبعیض", "شکایت", "بی‌احترامی", "تهدید", "harassment", "bully",
        ],
        examples=[
            "مدیرم دائم سر من داد می‌زند",
            "همکارم به من توهین کرد",
            "در محل کار مورد آزار قرار گرفتم",
            "با من رفتار تبعیض‌آمیز می‌شود",
            "می‌خواهم از مدیرم شکایت کنم",
        ],
        message=ESCALATE_FA,
    ),
    SensitiveCategory(
        name="termination_negotiation",
        keywords=["اخراج", "استعفا", "تسویه حساب", "افزایش حقوق", "مذاکره حقوق", "resign"],
        examples=[
            "می‌خواهم استعفا بدهم چطور مذاکره کنم",
            "آیا ممکن است اخراج شوم",
            "می‌خواهم برای افزایش حقوقم مذاکره کنم",
        ],
        message=ESCALATE_FA,
    ),
    SensitiveCategory(
        name="other_person_data",
        keywords=["حقوق سارا", "حقوق همکارم", "پروندهٔ", "اطلاعات شخصی"],
        examples=[
            "حقوق همکارم چقدر است",
            "سابقهٔ فلانی در شرکت چقدر است",
            "چرا به او افزایش حقوق دادند",
        ],
        message="اطلاعات شخصی سایر همکاران قابل ارائه نیست. در صورت نیاز با منابع انسانی تماس بگیرید.",
    ),
]


class SensitiveTopicRouter:
    # CALIBRATED against gemini-embedding-001, not guessed:
    #     sensitive questions  0.743 .. 0.775
    #     ordinary questions   0.640 .. 0.696
    # 0.72 is the midpoint. Note the margin is only ~0.05 -- far tighter than
    # the reranker's ~12-point separation, because *every* pair of Persian
    # sentences shares a high baseline similarity in this embedding space.
    #
    # So treat the semantic layer as a safety net, not the primary control: the
    # keyword layer is the deterministic one. Widening the margin means adding
    # more (and more varied) example phrases per category, not nudging this
    # number. Re-measure whenever the embedding model changes -- these values
    # are not portable across models.
    def __init__(self, embedder=None, threshold: float = 0.72):
        self.embedder = embedder
        self.threshold = threshold
        self._vectors: dict[str, np.ndarray] = {}
        if embedder is not None:
            for cat in CATEGORIES:
                self._vectors[cat.name] = embedder.encode(cat.examples, is_query=False)

    def _keyword_hit(self, question: str) -> SensitiveCategory | None:
        low = question.lower()
        for cat in CATEGORIES:
            if any(kw.lower() in low for kw in cat.keywords):
                return cat
        return None

    def _semantic_hit(self, question: str) -> tuple[SensitiveCategory | None, float]:
        if not self._vectors:
            return None, 0.0
        qv = self.embedder.encode_one(question, is_query=True)
        best, best_score = None, 0.0
        for cat in CATEGORIES:
            score = float(np.max(self._vectors[cat.name] @ qv))
            if score > best_score:
                best, best_score = cat, score
        if best_score >= self.threshold:
            return best, best_score
        return None, best_score

    def check(self, question: str) -> tuple[SensitiveCategory | None, str]:
        """Returns (category, reason). category is None when the question is fine."""
        cat = self._keyword_hit(question)
        if cat:
            return cat, "keyword"
        cat, score = self._semantic_hit(question)
        if cat:
            return cat, f"semantic:{score:.2f}"
        return None, ""
