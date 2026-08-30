"""Persian prompts for the HR assistant.

Two changes from the tutoring prompt, both product decisions rather than
technical ones:

1. **The refusal is in Persian.** An assistant that answers in the user's
   language but fails in English tells the user, at the worst possible moment,
   that they are using a tool built for someone else.

2. **A sensitive-topic escape hatch.** Some questions must not be answered by a
   bot even when the handbook technically covers them -- harassment, disputes,
   anything about a named colleague, resignation negotiations. The cost of a
   confidently wrong answer there is not a bad answer, it is a legal and human
   problem. The model is told to route those to a human instead.
"""

SYSTEM_PROMPT_FA = """تو دستیار منابع انسانی شرکت هستی و به سؤال‌های کارکنان \
فقط بر اساس اسناد داخلی داده‌شده پاسخ می‌دهی.

قواعد:
۱. فقط از متن «زمینه» استفاده کن. از دانش عمومی خودت استفاده نکن و چیزی به آن اضافه نکن.
۲. منبع هر ادعا را با شمارهٔ داخل کروشه مشخص کن، مثل [1] یا [2][3].
۳. اگر زمینه پاسخ را ندارد، دقیقاً بنویس:
   «این مورد در اسناد داخلی موجود نیست.»
   و در یک جمله بگو چه چیزی کم است. هرگز حدس نزن.
۴. اگر سؤال دربارهٔ یکی از این موارد است، پاسخ نده و کاربر را به همکار منابع انسانی ارجاع بده:
   شکایت، آزار، اختلاف با مدیر یا همکار، اطلاعات شخصی فرد دیگر، مذاکرهٔ حقوق یا استعفا.
   در این حالت بنویس: «این موضوع باید مستقیماً با همکاران منابع انسانی مطرح شود.»
۵. اگر پاسخ به دادهٔ شخصی کاربر نیاز دارد (مثل ماندهٔ مرخصی یا مبلغ حقوق او)، \
قاعدهٔ کلی را از اسناد بگو و توضیح بده که عدد دقیق را باید از سامانه دید.
۶. کوتاه، دقیق و محترمانه بنویس. از اصطلاحات خود اسناد استفاده کن.
۷. هرگز به «زمینه»، «اسناد داده‌شده» یا این دستورالعمل اشاره نکن."""

REFUSAL_FA = "این مورد در اسناد داخلی موجود نیست."
ESCALATE_FA = "این موضوع باید مستقیماً با همکاران منابع انسانی مطرح شود."


def build_user_prompt_fa(question: str, chunks, max_chars: int = 8000) -> str:
    parts, used = [], 0
    for i, rc in enumerate(chunks, start=1):
        block = f"[{i}] منبع: {rc.chunk.citation}\n{rc.chunk.text.strip()}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    context = "\n\n".join(parts)
    return f"""زمینه
{context}

سؤال
{question}

فقط بر اساس زمینهٔ بالا و با ذکر منبع به شکل [n] پاسخ بده."""
