# Individual Reflection - Lab 18

**Tên:** Hồ Trọng Duy Quang - 2A202600081
**Module phụ trách:** M3 Reranking + M4 RAGAS Evaluation  

---

## 1. Đóng góp kỹ thuật

Trong bài lab này, em phụ trách phần hậu xử lý sau retrieval, gồm Module 3 reranking và Module 4 đánh giá bằng RAGAS. Đây là hai phần quan trọng vì M3 quyết định chất lượng context cuối cùng đưa vào LLM, còn M4 là module bắt buộc để nhóm không bị giới hạn điểm.

Các phần đã thực hiện:

- Implement `CrossEncoderReranker` trong `src/m3_rerank.py`.
- Implement hàm `rerank(query, documents, top_k)` để chấm điểm lại các documents, sắp xếp theo `rerank_score` giảm dần và trả về `RerankResult`.
- Implement `benchmark_reranker()` để đo latency trung bình, nhỏ nhất, lớn nhất.
- Thêm fallback lexical reranker để test và pipeline không bị treo khi model `BAAI/bge-reranker-v2-m3` chưa tải xong hoặc môi trường không có cache.
- Implement `evaluate_ragas()` trong `src/m4_eval.py` với 4 metrics: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`.
- Cấu hình RAGAS dùng OpenAI LLM thật với model `gpt-4o-mini` và embedding `text-embedding-3-small`.
- Implement `failure_analysis()` để tìm các câu hỏi điểm thấp nhất, xác định metric tệ nhất, đưa ra diagnosis và suggested fix.
- Thêm các biến môi trường cần thiết vào `.env` và `.env.example`: `RAGAS_OPENAI_MODEL`, `RAGAS_OPENAI_EMBEDDING_MODEL`, `RAGAS_FORCE_OPENAI`, `RERANK_DOWNLOAD_MODEL`, `RERANK_FORCE_LEXICAL`.

Kết quả kiểm thử:

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_m3.py -v
```

Kết quả: `5 passed`.

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_m4.py -v
```

Kết quả: `4 passed`.

Ngoài ra, em cũng kiểm tra lint cho hai file phụ trách:

```powershell
.\venv\Scripts\python.exe -m ruff check .\src\m3_rerank.py .\src\m4_eval.py
```

Kết quả: `All checks passed!`.

## 2. Đóng góp xử lý dữ liệu PDF sang Markdown

Ngoài hai module chính, em cũng hỗ trợ bước chuẩn bị dữ liệu đầu vào cho pipeline RAG. Hai file PDF trong thư mục `data` ban đầu gần như không có text layer, nên khi dùng `markitdown` trực tiếp thì file Markdown sinh ra bị rỗng hoặc rất ít nội dung.

Các việc đã làm:

- Kiểm tra `data/BCTC.pdf` và phát hiện đây là PDF dạng ảnh/scan, `pypdf.extract_text()` gần như không lấy được ký tự nào.
- Hướng dẫn cài Tesseract OCR và thêm language data tiếng Việt `vie.traineddata`.
- OCR file PDF trước bằng `ocrmypdf`, sau đó mới convert sang Markdown bằng `markitdown`.
- Lưu file Markdown vào thư mục `data` để pipeline có thể đọc dữ liệu tập trung hơn.

Các lệnh chính:

```powershell
.\venv\Scripts\python.exe -m ocrmypdf --language vie+eng --force-ocr ".\data\BCTC.pdf" ".\data\BCTC_ocr.pdf"
.\venv\Scripts\markitdown.exe ".\data\BCTC_ocr.pdf" -o ".\data\BCTC.md"
```

Với file nghị định có chữ ký số, cần thêm tùy chọn:

```powershell
.\venv\Scripts\python.exe -m ocrmypdf --language vie+eng --force-ocr --invalidate-digital-signatures ".\data\Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf" ".\data\Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee_ocr.pdf"
.\venv\Scripts\markitdown.exe ".\data\Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee_ocr.pdf" -o ".\data\Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.md"
```

Phần này giúp dữ liệu đầu vào có dạng text/Markdown, thuận lợi hơn cho chunking, indexing và đánh giá RAG.

## 3. Kiến thức học được

Qua phần M3, em hiểu rõ hơn vai trò của reranking trong Production RAG. Retrieval ban đầu có thể lấy được nhiều context liên quan nhưng còn nhiễu; reranker giúp chọn lại top context sát với câu hỏi hơn trước khi đưa vào LLM.

Qua phần M4, em hiểu RAGAS không chỉ đánh giá câu trả lời đúng hay sai, mà chia chất lượng RAG thành nhiều khía cạnh:

- `faithfulness`: câu trả lời có bám vào context không.
- `answer_relevancy`: câu trả lời có đúng trọng tâm câu hỏi không.
- `context_precision`: context retrieved có ít nhiễu không.
- `context_recall`: context có đủ thông tin cần thiết không.

Em cũng học được rằng evaluation bằng LLM thật cần cấu hình cẩn thận để tránh test bị chậm, tốn API hoặc fail khi mạng không ổn định. Vì vậy em tách giữa unit test fallback local và evaluation thật bằng OpenAI.

## 4. Khó khăn và cách giải quyết

Khó khăn lớn nhất ở M3 là model reranker `BAAI/bge-reranker-v2-m3` tải và load khá lâu. Khi chạy test, nếu mặc định luôn load model thật thì test có thể mất nhiều phút. Em giải quyết bằng cách thêm biến `RERANK_DOWNLOAD_MODEL`: khi cần demo chất lượng thật thì bật `1`, còn khi chạy test nhanh thì để `0` và dùng fallback lexical.

Khó khăn lớn nhất ở M4 là version `ragas 0.1.19` không tương thích hoàn toàn với version LangChain hiện tại, thiếu module `langchain_core.pydantic_v1`. Em xử lý bằng compatibility shim trước khi import RAGAS. Ngoài ra, RAGAS cũng cần event loop khi chạy trên Python mới, nên em bổ sung hàm khởi tạo event loop để tránh lỗi runtime.

Ở phần PDF, khó khăn là PDF scan không có text layer, nên `markitdown` chạy nhưng không ra nội dung. Em xử lý bằng OCR trước, sau đó mới convert sang Markdown. Với file nghị định có digital signature, em dùng thêm `--invalidate-digital-signatures` vì OCR sẽ làm thay đổi nội dung PDF và vô hiệu chữ ký số.

## 5. Nếu làm lại

Nếu có thêm thời gian, em sẽ:

- Cache sẵn model reranker để demo M3 bằng neural reranker thật mà không mất thời gian tải trong buổi lab.
- Thêm latency breakdown cho từng bước: retrieval, reranking, generation, evaluation.
- Lưu kết quả RAGAS chi tiết theo từng câu hỏi vào report để phần failure analysis trực quan hơn.
- Thêm script riêng cho OCR + Markdown conversion để nhóm không cần gõ nhiều lệnh thủ công.

## 6. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) | Ghi chú |
|---|---:|---|
| Hiểu bài giảng | 4 | Hiểu rõ vai trò reranking và RAGAS trong Production RAG |
| Code quality | 4 | Code có fallback, env config và test pass đầy đủ |
| Teamwork | 4 | Hoàn thành module critical để nhóm có thể ghép pipeline |
| Problem solving | 5 | Xử lý được lỗi model chậm, RAGAS compatibility và PDF scan |

## 7. Kết luận

Phần việc của Người B đã hoàn thành các yêu cầu chính trong checklist: M3 pass `5/5` tests, M4 pass `4/4` tests, RAGAS được cấu hình dùng OpenAI thật với `gpt-4o-mini`, và có cơ chế phân tích lỗi để phục vụ báo cáo nhóm. Ngoài ra, phần chuyển PDF scan sang Markdown giúp chuẩn bị dữ liệu đầu vào tốt hơn cho toàn bộ pipeline RAG.
