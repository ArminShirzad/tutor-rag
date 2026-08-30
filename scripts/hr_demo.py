"""HR assistant demo: same questions, different people, different answers.

Run:  python scripts/hr_demo.py
"""
import io
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.access import Principal                                    # noqa: E402
from app.config import settings                                     # noqa: E402
from app.engine import RAGEngine                                    # noqa: E402
from app.generation.prompts_hr import (                             # noqa: E402
    REFUSAL_FA,
    SYSTEM_PROMPT_FA,
    build_user_prompt_fa,
)
from app.ingest.build import ensure_index                           # noqa: E402
from app.retrieval.pipeline import Retriever                        # noqa: E402
from app.generation.llm import build_llm                            # noqa: E402
from app.safety import SensitiveTopicRouter                         # noqa: E402

ensure_index(settings)
retriever = Retriever.from_index(settings.index_dir, settings)
engine = RAGEngine(
    retriever=retriever,
    llm=build_llm(settings.generation.provider, settings.generation.model, settings.gemini_api_key),
    config=settings,
    system_prompt=SYSTEM_PROMPT_FA,
    prompt_builder=build_user_prompt_fa,
    refusal=REFUSAL_FA,
    # reuses the retrieval embedder, so the semantic layer costs no extra model
    router=SensitiveTopicRouter(embedder=retriever.embedder),
)

REZA = Principal("e-100", "employee", "رضا (کارمند)")
SAHAR = Principal("e-300", "hr", "سحر (منابع انسانی)")

CASES = [
    (REZA,  "چند روز مرخصی استحقاقی دارم و چقدرش به سال بعد منتقل می‌شود؟", "answers from the handbook"),
    (REZA,  "بازه حقوقی سطح Senior Engineer چقدر است؟",                     "must refuse -- confidential"),
    (SAHAR, "بازه حقوقی سطح Senior Engineer چقدر است؟",                     "same question, HR may see it"),
    (REZA,  "مدیرم دائم سر من داد می‌زند، چه کار کنم؟",                      "must escalate -- keyword"),
    (REZA,  "همکارم مدام رفتار بدی با من دارد و حالم بد می‌شود",              "must escalate -- NO keyword, semantic only"),
    (REZA,  "ساعت کاری شرکت چند تا چند است؟",                               "normal question, must NOT escalate"),
]

for principal, question, expect in CASES:
    ans = engine.answer(question, principal=principal)
    mask = engine.retriever.access_mask(principal)
    print("=" * 70)
    print(f"{principal.describe()}   [sees {int(mask.sum())}/{len(mask)} chunks]")
    print(f"expected: {expect}")
    print(f"سؤال: {question}")
    print(f"پاسخ: {ans.answer}")
    print(f"منابع: {[c.citation for c in ans.citations] or '—'}")
    print(f"answered={ans.answered}  confidence={ans.confidence}")
    print()
