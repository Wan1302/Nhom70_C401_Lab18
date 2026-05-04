# Failure Analysis — Lab 18: Production RAG

**Nhóm:** 70 — C401  
**Ngày:** 2026-05-04

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.7163 | 0.4792 | -0.2371 |
| Answer Relevancy | 0.2552 | 0.3486 | **+0.0934** ↑ |
| Context Precision | 0.6708 | 0.6708 | 0.0000 |
| Context Recall | 0.5429 | 0.6429 | **+0.1000** ↑ |

> **Evaluator:** gpt-4o-mini + text-embedding-3-small (RAGAS OpenAI)  
> **Test set:** 20 câu hỏi (12 về Nghị định 13/2023, 8 về BCTC)

---

## Bottom-5 Failures

### #1 — Doanh thu thuần từ bán hàng
- **Question:** Doanh thu thuần từ bán hàng và cung cấp dịch vụ trong kỳ là bao nhiêu?
- **Expected:** Doanh thu thuần được trình bày trong Báo cáo kết quả hoạt động kinh doanh, phản ánh doanh thu sau khi trừ các khoản giảm trừ doanh thu.
- **Got:** "Không tìm thấy thông tin trong tài liệu."
- **Worst metric:** faithfulness (0.0), avg_score 0.0
- **Error Tree:**
  1. Output đúng? → **Không** — LLM trả "Không tìm thấy"
  2. Context có số liệu doanh thu? → **Không** — chunk BCTC không chứa bảng kết quả kinh doanh
  3. Retrieval đúng? → **Không** — dense search không tìm thấy chunk có con số cụ thể
  4. BCTC.md có bảng số liệu? → **Không** — PDF export markdown mất dữ liệu bảng
- **Root cause:** BCTC PDF chứa bảng số liệu tài chính dạng image/table phức tạp; khi export sang markdown, các ô bảng bị mất → dense embedding không có gì để tìm.
- **Suggested fix:** Dùng `camelot` hoặc `tabula-py` để extract bảng PDF thành CSV, index riêng dưới dạng text có cấu trúc.

### #2 — Lợi nhuận sau thuế
- **Question:** Lợi nhuận sau thuế thu nhập doanh nghiệp trong kỳ báo cáo là bao nhiêu?
- **Expected:** Lợi nhuận sau thuế được thể hiện trong Báo cáo kết quả hoạt động kinh doanh.
- **Got:** "Không tìm thấy thông tin trong tài liệu."
- **Worst metric:** faithfulness (0.0), avg_score 0.0
- **Error Tree:**
  1. Output đúng? → **Không**
  2. Context có thông tin lợi nhuận? → **Không** — tương tự #1, BCTC không có số liệu
  3. BM25 match? → **Kém** — "lợi nhuận sau thuế" xuất hiện nhưng không kèm con số
  4. Fix: Separate pipeline cho BCTC với OCR/structured extraction
- **Root cause:** Giống #1 — BCTC bảng số liệu không được preserve khi export markdown.
- **Suggested fix:** Cần OCR pipeline (PaddleOCR) để extract số liệu từ bảng tài chính trong PDF.

### #3 — Vốn chủ sở hữu
- **Question:** Vốn chủ sở hữu của công ty bao gồm những thành phần nào?
- **Expected:** Vốn chủ sở hữu bao gồm vốn điều lệ, thặng dư vốn cổ phần, lợi nhuận sau thuế chưa phân phối và các quỹ.
- **Got:** "Không tìm thấy thông tin trong tài liệu."
- **Worst metric:** faithfulness (0.0), avg_score 0.0
- **Error Tree:**
  1. Output đúng? → **Không**
  2. Document có section về VCSH? → **Có thể** — nhưng BCTC.md không capture đủ
  3. Chunk có chứa "vốn điều lệ, thặng dư"? → **Không rõ**
  4. Fix: Chunking BCTC theo từng báo cáo thành phần (BCĐKT, KQHĐKD, LCTT)
- **Root cause:** BCTC.md là một chunk lớn; hybrid search không prioritize đúng section Bảng cân đối kế toán.
- **Suggested fix:** Pre-split BCTC theo section (Bảng CĐKT / KQHĐKD / LCTT) trước khi chunk.

### #4 — Dữ liệu cá nhân nhạy cảm
- **Question:** Dữ liệu cá nhân nhạy cảm bao gồm những loại nào?
- **Expected:** Quan điểm chính trị, tôn giáo; sức khỏe; đời sống tình dục; dữ liệu tội phạm; thông tin tài chính; sinh trắc học.
- **Got:** *(LLM đã trả lời chi tiết đúng nội dung Nghị định)* — 9 điểm a-i
- **Worst metric:** faithfulness (0.0), avg_score 0.573
- **Error Tree:**
  1. Output đúng về nội dung? → **Có** — câu trả lời đúng và đầy đủ
  2. RAGAS faithfulness đúng? → **Không** — RAGAS không verify được từng claim chi tiết vì chunk quá lớn
  3. Context chunk chứa Điều 2? → **Có** — nhưng là 1 chunk toàn bộ văn bản dài
  4. Fix: Smaller chunks để RAGAS có thể verify từng claim chính xác hơn
- **Root cause:** Context là toàn bộ văn bản (1 chunk/doc); RAGAS faithfulness cần verify từng câu trong answer với context, nhưng LLM trích dẫn nhiều điểm → RAGAS mark 0.0 vì không thể trace rõ từng claim.
- **Suggested fix:** Dùng hierarchical chunking với child 512 chars để retrieval precision cao hơn, RAGAS verify dễ hơn.

### #5 — Bên kiểm soát dữ liệu — nghĩa vụ
- **Question:** Bên kiểm soát dữ liệu có nghĩa vụ gì khi xử lý dữ liệu cá nhân?
- **Expected:** Thực hiện biện pháp bảo vệ kỹ thuật, thông báo mục đích xử lý, chỉ xử lý đúng mục đích, lưu giữ hồ sơ, phối hợp cơ quan nhà nước.
- **Got:** *(LLM đã trả lời đúng 7 nghĩa vụ theo Nghị định)* — đầy đủ và chính xác
- **Worst metric:** faithfulness (0.0), avg_score 0.601
- **Error Tree:**
  1. Output đúng? → **Có** — câu trả lời đúng nội dung pháp lý
  2. RAGAS mark faithfulness 0? → **Có** — vì LLM dùng nhiều claim từ toàn văn bản dài
  3. Chunk quá lớn? → **Có** — 1 chunk/doc = toàn bộ Nghị định (~30.000 chars)
  4. Fix: Chunking nhỏ hơn → RAGAS verify dễ hơn; hoặc tăng `max_workers` để RAGAS có thêm thời gian xử lý
- **Root cause:** Giống #4 — LLM trả lời đúng nhưng RAGAS faithfulness thấp do chunk size lớn khiến việc verify claim khó.
- **Suggested fix:** Giảm chunk size, hoặc dùng semantic chunking để tạo chunks theo từng nghĩa vụ/điều khoản riêng biệt.

---

## Case Study cho Presentation

**Question chọn:** "Dữ liệu cá nhân nhạy cảm bao gồm những loại nào?" (Failure #4)

**Tại sao chọn:** LLM trả lời đúng nhưng RAGAS vẫn mark thất bại → minh họa limitation của automatic evaluation.

**Error Tree walkthrough:**
1. Output đúng về nội dung? → **Có** — LLM liệt kê đúng 9 loại dữ liệu nhạy cảm theo Nghị định 13
2. RAGAS Faithfulness = 0.0? → **Đúng** — vì chunk context quá lớn (toàn văn bản Nghị định)
3. Claim nào không verify được? → RAGAS không trace được từng điểm a-i về đúng vị trí trong chunk
4. Root fix: Chunking theo từng điều khoản (50-200 chars) → RAGAS verify chính xác hơn

**Insight quan trọng:** Automatic evaluation (RAGAS) có thể underestimate khi chunk size lớn. Cần human evaluation bổ sung cho các trường hợp như vậy.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Hierarchical chunking với child 512 chars + trả parent cho LLM → tăng faithfulness
- Separate BCTC pipeline với table extraction (camelot) → fix 8 BCTC failures
- Tăng `max_workers=4` trong RunConfig → giảm RateLimitError khi RAGAS evaluate
