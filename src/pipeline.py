"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_structure_aware
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60)

    # Step 1: Load & Chunk (M1)
    print("\n[1/3] Chunking documents...")
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        for chunk in chunk_structure_aware(doc["text"], metadata=doc["metadata"]):
            all_chunks.append({"text": chunk.text, "metadata": chunk.metadata})
    print(f"  {len(all_chunks)} chunks from {len(docs)} documents")

    # Step 2: Enrichment (M5)
    print("\n[2/4] Enriching chunks (M5)...")
    enriched = enrich_chunks(all_chunks, methods=["contextual", "hyqa", "metadata"])
    if enriched:
        # Index enriched text (better retrieval), but keep original for LLM context
        all_chunks = [
            {"text": e.enriched_text, "metadata": {**e.auto_metadata, "original_text": e.original_text}}
            for e in enriched
        ]
        print(f"  Enriched {len(enriched)} chunks")
    else:
        print("  ⚠️  M5 not implemented — using raw chunks (fallback)")

    # Step 3: Index (M2)
    print("\n[3/4] Indexing (BM25 + Dense)...")
    search = HybridSearch()
    search.index(all_chunks)

    # Step 4: Reranker (M3)
    print("\n[4/4] Loading reranker...")
    reranker = CrossEncoderReranker()

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    top = reranked if reranked else results[:RERANK_TOP_K]
    # Dùng original_text (trước khi enrich) để tránh noise khi LLM generate
    contexts = [r.metadata.get("original_text", r.text) for r in top]

    answer = _llm_generate(query, contexts)
    return answer, contexts


def _llm_generate(query: str, contexts: list[str]) -> str:
    """Generate answer from LLM. Falls back to contexts[0] if no API key."""
    from config import OPENAI_API_KEY
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Trả lời CHỈ dựa trên context được cung cấp. Nếu không tìm thấy thông tin → trả lời 'Không tìm thấy thông tin trong tài liệu.' Không bịa thêm thông tin."},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"  ⚠️  LLM error: {e}")
    return contexts[0] if contexts else "Không tìm thấy thông tin."


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    print("\n[Eval] Running queries...")
    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...")

    print("\n[Eval] Running RAGAS...")
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
