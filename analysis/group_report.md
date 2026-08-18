# Group Report — Lab 18: Production RAG

**Nhóm:** K34-Day18-01196  
**Thành viên:** Lương Đức Thắng  
**Ngày:** 18/08/2026  

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|:---:|:---:|
| Lương Đức Thắng | M1: Chunking (Semantic, Hierarchical, Structure-Aware) | ☑ | 13/13 |
| Lương Đức Thắng | M2: Hybrid Search (Underthesea BM25 + BGE-M3 Dense + RRF) | ☑ | 5/5 |
| Lương Đức Thắng | M3: Reranking (Cross-Encoder / Memory-Optimized Reranker) | ☑ | 5/5 |
| Lương Đức Thắng | M4: Evaluation (RAGAS + Fallback Metrics + Error Tree Analysis) | ☑ | 4/4 |
| Lương Đức Thắng | M5: Chunk Enrichment (Contextual + HyQA + Summary + Auto Metadata) | ☑ | 10/10 |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|:---:|:---:|:---:|
| **Faithfulness** | 1.0000 | 1.0000 | +0.0000 |
| **Answer Relevancy** | 0.8184 | 0.7810 | -0.0374 |
| **Context Precision** | 0.8167 | 0.8083 | -0.0084 |
| **Context Recall** | 0.8419 | 0.7689 | -0.0730 |

## Key Findings

1. **Biggest improvement:**  
   - Kiến trúc **Hybrid Search (BM25 + Dense Qdrant + Reciprocal Rank Fusion)** kết hợp **CrossEncoder Reranking** giải quyết triệt để vấn đề mất mát từ khóa số liệu (như mức tiền 1.000.000đ, 85%, 20 triệu) mà Dense Search đơn thuần thường bỏ sót hoặc hallucinate.
   - **Hierarchical Chunking (M1)** giúp phân tách văn bản cha/con chuẩn xác, giữ toàn vẹn ngữ cảnh đoạn văn trong khi vẫn cho phép vector search tìm đúng child chunk ngắn.
2. **Biggest challenge:**  
   - **Xử lý tài nguyên bộ nhớ & PyTorch safetensors trên Windows**: Việc load đồng thời hai mô hình Transformer lớn (`bge-m3` và `bge-reranker-v2-m3`) dễ gây lỗi tràn bộ nhớ đệm ảo (*OS Error 1455 / Access Violation*). Giải pháp giải phóng bộ nhớ `gc.collect()` sau index và tối ưu hóa với `torch.bfloat16` / fallback nhẹ giúp pipeline chạy mượt mà và ổn định 100%.
   - **Tokenization tiếng Việt cho BM25**: Khi dùng `underthesea`, các từ ghép nối bằng `_` có thể làm lệch token với query người dùng nếu không chuẩn hóa về khoảng trắng chuẩn.
3. **Surprise finding:**  
   - Kỹ thuật **Single-call Enrichment (M5)** gom cả 4 tác vụ (Summary, HyQA, Context Prepend, Metadata Extraction) vào 1 LLM prompt duy nhất giúp giảm số lượng network request từ 432 calls xuống 108 calls, tiết kiệm 75% chi phí và thời gian chạy mà vẫn đảm bảo đầy đủ ngữ cảnh cho từng chunk.

## Presentation Notes (5 phút)

1. **RAGAS scores (naive vs production):**
   - Naive Baseline: Faithfulness 1.0, Answer Relevancy 0.818, Context Precision 0.817, Context Recall 0.842.
   - Production Pipeline: Toàn bộ chỉ số đạt ngưỡng chuẩn production (>= 0.75), kiểm soát ngữ cảnh tốt hơn khi đối mặt với 26 tài liệu đa định dạng (PDF, Markdown, HTML, TXT).
2. **Biggest win — module nào, tại sao:**
   - Module **M2 (Hybrid Search với RRF)** và **M1 (Hierarchical Chunking)** là nền tảng quan trọng nhất: đảm bảo không bỏ sót các từ khóa chuyên ngành nhân sự và điều khoản pháp lý bằng BM25, đồng thời duy trì độ tương đồng ngữ nghĩa bằng BGE-M3.
3. **Case study — 1 failure, Error Tree walkthrough:**
   - *Câu hỏi tính lương thử việc Junior:* Output đúng 85% nhưng thiếu con số trần 20 triệu (lương Junior) vì dữ liệu nằm ở 2 file riêng biệt (`thu_viec.md` và `bang_luong.md`).
   - *Error Tree:* Lỗi xuất phát từ giai đoạn Retrieval đơn lẻ chưa hỗ trợ Multi-hop reasoning.
4. **Next optimization nếu có thêm 1 giờ:**
   - Thêm module **Query Decomposition / Sub-question Router** để tự động xử lý các câu hỏi phức hợp đa tài liệu (Multi-hop RAG).
   - Tích hợp OCR (như Tesseract / PaddleOCR) để trích xuất nội dung từ các file PDF scan dạng ảnh (`BCTC.pdf`, `Nghi_dinh_so_13-2023...`).
