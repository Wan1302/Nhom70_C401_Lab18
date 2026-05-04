"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.
One-time offline cost; cải thiện mọi query sau đó.

Test: pytest tests/test_m5.py
"""

import os, sys, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tóm tắt chunk 2-3 câu. Embed summary → giảm noise, tăng precision.
    Dùng LLM nếu có OPENAI_API_KEY, fallback extractive nếu không.
    """
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    # Extractive fallback: lấy 2 câu đầu
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return ". ".join(s.rstrip(".") for s in sentences[:2]) + "."


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Bridge vocabulary gap: user hỏi khác cách viết trong doc → HyQA index cả 2.
    """
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            raw = resp.choices[0].message.content.strip().split("\n")
            return [q.strip().lstrip("0123456789.-) ") for q in raw if q.strip()][:n_questions]
        except Exception:
            pass

    # Extractive fallback: sinh câu hỏi từ keyword và con số trong text
    questions: list[str] = []
    numbers = re.findall(r"\d+(?:\s*\w+){1,4}", text)
    if numbers:
        questions.append(f"Số lượng hoặc thời hạn liên quan đến '{numbers[0]}' là bao nhiêu?")
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    if sentences:
        questions.append("Đoạn văn bản này đề cập đến nội dung gì?")
    if len(sentences) > 1:
        questions.append("Thông tin chính được nêu trong đoạn này là gì?")
    return questions[:n_questions]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend 1 câu context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: kỹ thuật này giảm 49% retrieval failure.
    """
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=80,
            )
            context = resp.choices[0].message.content.strip()
            return f"{context}\n\n{text}"
        except Exception:
            pass

    # Extractive fallback: prepend tên tài liệu
    prefix = f"Trích từ tài liệu: {document_title}." if document_title else "Đoạn trích từ tài liệu nội bộ."
    return f"{prefix}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    Extract metadata: topic, entities, category, language.
    Dùng LLM nếu có key; fallback heuristic không cần API.
    """
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            import json as _json
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return _json.loads(resp.choices[0].message.content)
        except Exception:
            pass

    # Heuristic fallback
    lower = text.lower()
    if any(w in lower for w in ["nghỉ phép", "lương", "nhân viên", "tuyển dụng"]):
        category = "hr"
    elif any(w in lower for w in ["mật khẩu", "bảo mật", "vpn", "firewall"]):
        category = "it"
    elif any(w in lower for w in ["doanh thu", "tài chính", "kế toán", "vốn"]):
        category = "finance"
    elif any(w in lower for w in ["dữ liệu cá nhân", "bảo vệ dữ liệu", "nghị định"]):
        category = "policy"
    else:
        category = "general"

    vi_chars = len(re.findall(r"[àáâãèéêìíòóôõùúýăđơưạặắẳầấậảẹẻẽềếệìíịỉòóọỏốồổộớờởùúụủứừựỳ]", text))
    language = "vi" if vi_chars > 5 else "en"

    first_sentence = re.split(r"[.!?]", text)[0].strip()
    entities = re.findall(r"\b[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÙÚÝ][a-zàáâãèéêìíòóôùúýăđơư]+(?:\s+[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÙÚÝ][a-zàáâãèéêìíòóôùúýăđơư]+)*\b", text)

    return {
        "topic": first_sentence[:80],
        "entities": list(dict.fromkeys(entities))[:5],
        "category": category,
        "language": language,
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Subset of ["summary", "hyqa", "contextual", "metadata", "full"].
                 Default: ["contextual", "hyqa", "metadata"]
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    use_summary = "summary" in methods or "full" in methods
    use_hyqa = "hyqa" in methods or "full" in methods
    use_contextual = "contextual" in methods or "full" in methods
    use_metadata = "metadata" in methods or "full" in methods

    enriched: list[EnrichedChunk] = []
    for chunk in chunks:
        text = chunk["text"]
        meta = chunk.get("metadata", {})
        source = meta.get("source", "")

        summary = summarize_chunk(text) if use_summary else ""
        questions = generate_hypothesis_questions(text) if use_hyqa else []
        enriched_text = contextual_prepend(text, source) if use_contextual else text
        auto_meta = extract_metadata(text) if use_metadata else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**meta, **auto_meta},
            method="+".join(methods),
        ))

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")
    print(f"Summary: {summarize_chunk(sample)}\n")
    print(f"HyQA: {generate_hypothesis_questions(sample)}\n")
    print(f"Contextual: {contextual_prepend(sample, 'Sổ tay nhân viên VinUni 2024')}\n")
    print(f"Metadata: {extract_metadata(sample)}")
