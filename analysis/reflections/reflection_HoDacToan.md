# Individual Reflection — Lab 18

**Tên:** Ho Dac Toan  
**Module phụ trách:** M1: Chunking + M5: Enrichment

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:** M1 (`src/m1_chunking.py`) và M5 (`src/m5_enrichment.py`)
- **Các hàm/class chính đã viết:**
  - M1: `chunk_semantic()` — dùng SentenceTransformer bge-m3 encode từng câu, gom nhóm theo cosine similarity dưới threshold
  - M1: `chunk_hierarchical()` — tạo parent chunks (2048 chars) + child chunks (256 chars), link qua `parent_id`
  - M1: `chunk_structure_aware()` — regex split theo markdown headers `#{1,3}`, giữ nguyên tables và code blocks
  - M1: `compare_strategies()` — A/B test 4 strategies, in bảng stats
  - M1: `load_documents()` — load `.md` (ưu tiên) và `.pdf` (fallback pypdf/pdfplumber), bỏ qua PDF nếu đã có `.md` cùng tên
  - M5: `summarize_chunk()` — OpenAI gpt-4o-mini hoặc extractive 2 câu đầu
  - M5: `generate_hypothesis_questions()` — HyQA với OpenAI hoặc regex keyword extraction
  - M5: `contextual_prepend()` — Anthropic-style context prefix, giảm 49% retrieval failure
  - M5: `extract_metadata()` — JSON extraction qua LLM hoặc heuristic (category: hr/it/finance/policy)
  - M5: `enrich_chunks()` — pipeline tổng hợp, hỗ trợ methods list
- **Số tests pass:** 23/23 (M1: 13/13, M5: 10/10)

## 2. Kiến thức học được

- **Khái niệm mới nhất:** Hierarchical chunking — index children để search nhanh (nhỏ, precision cao), nhưng trả parent cho LLM (đủ context). Tách bạch giữa retrieval granularity và generation context size.
- **Điều bất ngờ nhất:** Enrichment contextual prepend làm tăng retrieval accuracy nhưng khi trả enriched_text làm context cho LLM lại giảm faithfulness — phải index enriched nhưng generate từ original. Trade-off không obvious khi thiết kế.
- **Kết nối với bài giảng:** Semantic chunking (slide "Chunking Strategies") — giải thích tại sao fixed-size chunking cắt giữa ý, và cách cosine similarity giúp giữ coherence. HyQA (slide "Enrichment Techniques") — Anthropic benchmark giảm 49% retrieval failure.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** `sentence_transformers` conflict với `tensorflow/keras` trên môi trường đã cài `tf_keras` — import SentenceTransformer crash với `TFPreTrainedModel` error.
- **Cách giải quyết:** Thêm `os.environ.setdefault("TRANSFORMERS_NO_TF", "1")` ở đầu mỗi module trước khi import bất kỳ thư viện nào. `setdefault` đảm bảo không override nếu đã set.
- **Thời gian debug:** ~45 phút để trace từ ImportError → keras conflict → env var fix.

## 4. Nếu làm lại

- **Sẽ làm khác:** Implement parent-chunk retrieval đúng cách — khi child được retrieve, map về parent_id và trả parent text cho LLM. Hiện tại pipeline chỉ dùng child text → context quá ngắn (256 chars) cho câu hỏi phức tạp.
- **Module muốn thử tiếp:** M2 — muốn thử Late Interaction (ColBERT) thay RRF đơn giản, và thử `underthesea` entity extraction để boost BM25 với named entities tiếng Việt.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 4 |
