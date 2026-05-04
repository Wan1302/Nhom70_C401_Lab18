"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load .md và .pdf files từ data/."""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    md_stems = {os.path.splitext(os.path.basename(f))[0]
                for f in glob.glob(os.path.join(data_dir, "*.md"))}
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        stem = os.path.splitext(os.path.basename(fp))[0]
        if stem in md_stems:
            continue  # .md đã load rồi, bỏ qua PDF trùng tên
        text = ""
        try:
            import pdfplumber
            with pdfplumber.open(fp) as pdf:
                text = "\n\n".join(
                    (page.extract_text() or "").strip() for page in pdf.pages
                )
        except Exception:
            pass
        if not text.strip():
            try:
                import pypdf
                reader = pypdf.PdfReader(fp)
                text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as e:
                print(f"  ⚠️  Không đọc được {os.path.basename(fp)}: {e}")
        if text.strip():
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────

# Cache model ở module level — tránh load lại mỗi lần gọi chunk_semantic
_semantic_model = None


def _get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        # Ngăn transformers cố import TensorFlow — tránh conflict keras/tf_keras
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        from sentence_transformers import SentenceTransformer
        _semantic_model = SentenceTransformer("BAAI/bge-m3")
    return _semantic_model


def chunk_semantic(
    text: str,
    threshold: float = SEMANTIC_THRESHOLD,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    import numpy as np

    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n", text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "chunk_index": 0, "strategy": "semantic"})]

    model = _get_semantic_model()
    embeddings = model.encode(sentences)

    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    chunks: list[Chunk] = []
    current_group = [sentences[0]]

    for i in range(1, len(sentences)):
        if cosine_sim(embeddings[i - 1], embeddings[i]) < threshold:
            chunks.append(Chunk(
                text=" ".join(current_group),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
            ))
            current_group = []
        current_group.append(sentences[i])

    if current_group:
        chunks.append(Chunk(
            text=" ".join(current_group),
            metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
        ))
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Index children vào vector DB; khi retrieve → trả parent cho LLM.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    parents: list[Chunk] = []
    children: list[Chunk] = []
    current_text = ""
    p_index = 0

    def _flush_parent(raw: str, idx: int) -> None:
        pid = f"parent_{idx}"
        parents.append(Chunk(
            text=raw.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
        ))
        for start in range(0, len(raw), child_size):
            child_text = raw[start: start + child_size].strip()
            if child_text:
                children.append(Chunk(
                    text=child_text,
                    metadata={**metadata, "chunk_type": "child"},
                    parent_id=pid,
                ))

    for para in paragraphs:
        if current_text and len(current_text) + len(para) > parent_size:
            _flush_parent(current_text, p_index)
            p_index += 1
            current_text = ""
        current_text += para + "\n\n"

    if current_text.strip():
        _flush_parent(current_text, p_index)

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    sections = re.split(r"(^#{1,3}\s+.+$)", text, flags=re.MULTILINE)

    chunks: list[Chunk] = []
    current_header = ""
    current_content = ""

    for part in sections:
        if re.match(r"^#{1,3}\s+", part):
            if current_content.strip():
                chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={**metadata, "section": current_header.strip(), "strategy": "structure"},
                ))
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    if current_content.strip():
        chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
        chunks.append(Chunk(
            text=chunk_text,
            metadata={**metadata, "section": current_header.strip(), "strategy": "structure"},
        ))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare stats.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    def _stats(chunk_list: list[Chunk]) -> dict:
        lengths = [len(c.text) for c in chunk_list] if chunk_list else [0]
        return {
            "num_chunks": len(chunk_list),
            "avg_length": round(sum(lengths) / len(lengths)),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }

    results: dict = {}

    for name, fn in [
        ("basic", lambda d: chunk_basic(d["text"], metadata=d["metadata"])),
        ("semantic", lambda d: chunk_semantic(d["text"], metadata=d["metadata"])),
        ("structure", lambda d: chunk_structure_aware(d["text"], metadata=d["metadata"])),
    ]:
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(fn(doc))
        results[name] = _stats(all_chunks)

    all_children: list[Chunk] = []
    all_parents: list[Chunk] = []
    for doc in documents:
        p, c = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        all_parents.extend(p)
        all_children.extend(c)
    s = _stats(all_children)
    results["hierarchical"] = {**s, "num_chunks": f"{len(all_parents)}p/{len(all_children)}c"}

    print(f"\n{'Strategy':<15} | {'Chunks':>10} | {'Avg':>6} | {'Min':>6} | {'Max':>6}")
    print("-" * 52)
    for name, st in results.items():
        print(f"{name:<15} | {str(st['num_chunks']):>10} | {st['avg_length']:>6} | {st['min_length']:>6} | {st['max_length']:>6}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
