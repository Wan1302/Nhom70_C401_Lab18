# Individual Reflection — Lab 18

**Tên:** Hồ Trần Đình Nguyên - 2A202600080  
**Module phụ trách:** M2: Hybrid Search

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:** M2 (`src/m2_search.py`)
- **Các hàm/class chính đã viết:**
  - `segment_vietnamese()` — dùng `underthesea.word_tokenize` tách từ tiếng Việt trước khi đưa vào BM25; không có bước này BM25 coi "nghỉ phép" là 2 token riêng biệt thay vì 1 từ ghép
  - `BM25Search.index()` — segment từng chunk, tokenize thành list, build `BM25Okapi` index
  - `BM25Search.search()` — segment query, gọi `get_scores()`, sort và trả về `SearchResult` list với `method="bm25"`
  - `DenseSearch.index()` — encode toàn bộ chunks bằng `bge-m3` (1024-dim), upload lên Qdrant dùng `create_collection` + `upsert`
  - `DenseSearch.search()` — encode query, gọi `query_points()` lấy top-k hits từ Qdrant
  - `reciprocal_rank_fusion()` — merge BM25 + Dense rankings: `score(d) = Σ 1/(k + rank + 1)`, docs xuất hiện ở cả 2 list được boost tự nhiên
- **Số tests pass:** 5/5

## 2. Kiến thức học được

- **Khái niệm mới nhất:** Reciprocal Rank Fusion — cách đơn giản nhưng hiệu quả để merge 2 hệ thống ranking hoàn toàn khác nhau (BM25 dựa trên TF-IDF, Dense dựa trên embedding cosine similarity) mà không cần normalize scores về cùng scale.
- **Điều bất ngờ nhất:** BM25 với tiếng Việt hoàn toàn vô dụng nếu không có word segmentation — query "nghỉ phép" tách thành `["nghỉ", "phép"]` match sai hoàn toàn với corpus chưa segment. Chỉ thêm 1 dòng `underthesea` mà precision tăng rõ rệt.
- **Kết nối với bài giảng:** Hybrid Search (slide "Retrieval Strategies") — giải thích tại sao sparse (BM25) tốt cho exact keyword match còn dense tốt cho semantic similarity, và RRF là cách kết hợp tận dụng điểm mạnh của cả hai.
- **Kết nối với kết quả thực tế:** Context Recall tăng +10% (0.543 → 0.643) trong production pipeline — Hybrid Search (BM25 + Dense + RRF) retrieve được nhiều chunks liên quan hơn so với naive baseline dense-only, đúng với lý thuyết về complementary strengths của sparse và dense.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** API `qdrant-client` thay đổi giữa các version — `recreate_collection()` và `client.search()` đã deprecated, code scaffold dùng API cũ nên crash khi chạy thực tế.
- **Cách giải quyết:** Đọc changelog qdrant-client, thay `recreate_collection` bằng `delete_collection` + `create_collection`, thay `client.search()` bằng `client.query_points()` và unpack `.points` từ response object.
- **Thời gian debug:** ~20 phút để trace từ `AttributeError` → deprecated API → tìm đúng method mới.

## 4. Nếu làm lại

- **Sẽ làm khác:** Thêm caching cho encoder trong `DenseSearch` ở cấp module thay vì instance — hiện tại mỗi lần tạo `DenseSearch()` mới đều load lại model 1.5GB, lãng phí nếu pipeline tạo nhiều instance.
- **Module muốn thử tiếp:** M3 (Reranking) — muốn xem cross-encoder cải thiện bao nhiêu so với RRF đơn thuần, và thử ColBERT late interaction thay vì bi-encoder cho dense retrieval.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 4 |
