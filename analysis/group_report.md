# Group Report — Lab 18: Production RAG

**Nhóm:** 70 — C401  
**Ngày:** 2026-05-04

---

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Ho Dac Toan | M1: Chunking + M5: Enrichment | ✓ | 23/23 |
| Ho Tran Dinh Nguyen | M2: Hybrid Search | ✓ | 5/5 |
| Ho Trong Duy Quang | M3: Reranking + M4: Evaluation | ✓ | 9/9 |

**Tổng tests:** 37/37 passed (100%)

---

## Kết quả RAGAS

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.7163 | 0.4792 | -0.2371 |
| Answer Relevancy | 0.2552 | 0.3486 | **+0.0934** ↑ |
| Context Precision | 0.6708 | 0.6708 | 0.0000 |
| Context Recall | 0.5429 | 0.6429 | **+0.1000** ↑ |

> **Evaluator:** gpt-4o-mini + text-embedding-3-small (RAGAS OpenAI)  
> Naive baseline dùng `contexts[0]` làm answer (không LLM); Production dùng gpt-4o-mini generation + bge-reranker reranking.

---

## Kiến trúc Pipeline

```
Documents (PDF → Markdown)
    ↓ M1: Structure-Aware Chunking (theo section markdown)
Chunks
    ↓ M5: Enrichment
        - Contextual Prepend (Anthropic-style, -49% retrieval failure)
        - HyQA (bridge vocabulary gap)
        - Auto Metadata (category: hr/it/finance/policy)
Enriched Chunks (indexed) + Original Chunks (returned to LLM)
    ↓ M2: Hybrid Search
        - BM25 với underthesea word segmentation (tiếng Việt)
        - Dense: bge-m3 (1024 dim) + Qdrant vector DB
        - Reciprocal Rank Fusion (RRF, k=60)
Top-20 Candidates
    ↓ M3: Cross-Encoder Reranking (bge-reranker-v2-m3)
Top-5 Contexts (original text)
    ↓ LLM Generation: gpt-4o-mini (temperature=0.1)
Answer
    ↓ M4: RAGAS Evaluation (4 metrics)
ragas_report.json
```

---

## Key Findings

1. **Answer Relevancy tăng +0.0934:** LLM generation với gpt-4o-mini tạo câu trả lời đúng format câu hỏi hơn so với trả raw context. HyQA enrichment giúp bridge vocabulary gap giữa cách user hỏi và cách văn bản viết.

2. **Context Recall tăng +0.1000:** Structure-aware chunking tạo chunks lớn theo section markdown → mỗi chunk chứa nhiều thông tin liên quan → recall tăng.

3. **Faithfulness giảm -0.2371 — Phân tích:** Có 2 nguyên nhân:
   - **BCTC (8 câu):** PDF bảng số liệu không extract được → LLM trả "Không tìm thấy" → faithfulness = 0
   - **Chunk lớn (2 câu):** LLM trả lời đúng nhưng RAGAS không verify được từng claim trong chunk ~30.000 chars → đánh thấp oan

4. **Context Precision giữ nguyên (0.6708):** Hybrid BM25+Dense+RRF trả về contexts có chất lượng tương đương naive baseline dense-only → precision không giảm dù retrieval corpus lớn hơn nhiều (từ 44 → toàn văn bản).

---

## Failure Pattern Analysis

| Pattern | Số câu | Root cause | Fix |
|---------|--------|-----------|-----|
| BCTC bảng số liệu | 6 | PDF table không extract được | camelot/tabula structured extraction |
| Chunk quá lớn → RAGAS verify kém | 2 | 1 chunk/doc quá lớn | Hierarchical child 512 chars |
| Semantic mismatch | 2 | Lexical gap query ↔ doc | Query expansion / HyDE |

---

## Presentation Notes (5 phút)

1. **RAGAS scores (1 phút):** Answer Relevancy +9.3%, Context Recall +10% — LLM generation + enrichment cải thiện rõ. Faithfulness giảm do BCTC table extraction failure (6/20 câu trả "Không tìm thấy").

2. **Biggest win — M5 Enrichment + M2 Hybrid (1.5 phút):** HyQA generate câu hỏi từ chunk → khi user hỏi cùng ý nhưng khác từ, BM25 vẫn match. underthesea segmentation: "nghỉ phép" = 1 token thay vì 2 → BM25 score chính xác hơn. RRF fusion: BM25 catch keyword match + dense catch semantic → tốt hơn từng method riêng lẻ.

3. **Case study — Failure #4 (1.5 phút):** "Dữ liệu nhạy cảm bao gồm những loại nào?" → LLM trả đúng 9 điểm a-i → RAGAS faithfulness = 0.0 vì chunk quá lớn. Insight: automatic evaluation có blind spot với chunk size lớn → cần human eval bổ sung.

4. **Next optimization (1 phút):** (a) BCTC table extraction với camelot — fix 6/8 BCTC failures; (b) Hierarchical parent-child: index child 512 chars, trả parent cho LLM → faithfulness tăng; (c) Tăng `max_workers` RAGAS → giảm RateLimitError.
