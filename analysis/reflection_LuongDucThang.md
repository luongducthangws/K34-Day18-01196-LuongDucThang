# Individual Reflection — Lab 18: Production RAG Pipeline

**Tên:** Lương Đức Thắng  
**MSSV/ID:** 01196  
**Lớp:** K34  
**Module phụ trách:** M1 (Advanced Chunking), M2 (Hybrid Search), M3 (Reranking), M4 (Evaluation & Failure Analysis), M5 (Chunk Enrichment)

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:** Hoàn thiện toàn bộ 5 Modules từ M1 đến M5 cùng End-to-End Production Pipeline (`src/pipeline.py`) và Naive Baseline (`naive_baseline.py`).
- **Các hàm/class chính đã viết:**
  - `M1 (m1_chunking.py)`: `chunk_semantic` (SentenceTransformer similarity break), `chunk_hierarchical` (Parent 2048 / Child 256 linked via parent_id), `chunk_structure_aware` (Markdown header section parsing).
  - `M2 (m2_search.py)`: `segment_vietnamese` (Underthesea word segmentation), `BM25Search` (BM25Okapi indexing & retrieval), `DenseSearch` (BGE-M3 + Qdrant vector collection), `reciprocal_rank_fusion` (RRF algorithm with constant $k=60$).
  - `M3 (m3_rerank.py)`: `CrossEncoderReranker` (Cross-encoder reranking with memory optimization, torch.bfloat16 and safe fallback).
  - `M4 (m4_eval.py)`: `evaluate_ragas` (RAGAS 4-metric evaluation + robust fallback metric computation), `failure_analysis` (Diagnostic Tree analyzer), `save_report` (dual-path report generator).
  - `M5 (m5_enrichment.py)`: `summarize_chunk`, `generate_hypothesis_questions` (HyQA), `contextual_prepend` (Anthropic-style context), `extract_metadata`, and `_enrich_single_call` (Single-call cost-optimized 4-in-1 enrichment).
- **Số tests pass:** 37 / 37 unit tests (100% pass across all modules).

---

## 2. Mapping bài giảng (Lecture → Code Implementation)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích kỹ thuật |
|----------------|:------:|------------|-----------------------------------|
| **Semantic Chunking** | M1 | `chunk_semantic()` | Dùng cosine distance giữa các câu liên tiếp qua `all-MiniLM-L6-v2`. Threshold 0.5 giúp giữ trọn các đoạn có cùng ngữ nghĩa thay vì cắt ngang dòng. |
| **Hierarchical Parent-Child Chunking** | M1 | `chunk_hierarchical()` | Tạo parent chunks (2048 chars) chứa ngữ cảnh rộng và child chunks (256 chars) cho vector indexing. Khi truy vấn, Qdrant tìm trúng child chunk nhưng trả về parent chunk để đưa vào LLM context, giải quyết triệt để context loss. |
| **Structure-Aware Chunking** | M1 | `chunk_structure_aware()` | Parse theo cây tiêu đề Markdown (`#`, `##`, `###`). Đính kèm đường dẫn Breadcrumb (ví dụ `Section: Chính sách > Điều khoản`) vào từng chunk giúp LLM hiểu nguồn gốc điều khoản. |
| **BM25 + Dense Fusion (RRF)** | M2 | `reciprocal_rank_fusion()` | RRF ($1/(k + \text{rank} + 1)$) giải quyết bài toán không đồng nhất thang điểm (BM25 score uncalibrated vs Cosine similarity [-1, 1]), giúp match chính xác cả số liệu (1.000.000đ) lẫn ngữ nghĩa. |
| **Vietnamese Word Segmentation** | M2 | `segment_vietnamese()` | Sử dụng `underthesea` để phân đoạn từ ghép tiếng Việt, chuẩn hóa khoảng trắng để từ khóa tìm kiếm khớp chính xác với corpus. |
| **Cross-Encoder Reranking** | M3 | `CrossEncoderReranker.rerank()` | Bi-Encoder (Dense) chỉ tính embedding độc lập. Cross-Encoder tính toán self-attention trực tiếp giữa query và document, nâng cao độ chính xác Top-K trước khi truyền vào LLM context window. |
| **RAGAS 4 Metrics Triad** | M4 | `evaluate_ragas()` | Đo lường hệ thống đa chiều: Faithfulness (độ trung thực), Answer Relevancy (sự liên quan), Context Precision (tỷ lệ chunk đúng trong Top-K), Context Recall (độ phủ thông tin ground truth). |
| **Diagnostic Failure Tree** | M4 | `failure_analysis()` | Tự động phân loại lỗi: Context Recall thấp → tăng Top-K/bổ sung BM25; Context Precision thấp → thêm Reranker; Faithfulness thấp → cải thiện system prompt chống hallucination. |
| **Contextual Prepend** | M5 | `contextual_prepend()` | Theo kỹ thuật của Anthropic, gắn tiền tố mô tả vị trí văn bản vào đầu chunk, giúp giảm 49% lỗi truy xuất khi chunk bị tách rời. |
| **HyQA (Hypothetical Questions)** | M5 | `generate_hypothesis_questions()` | Sinh câu hỏi tiềm năng cho chunk để index song song, thu hẹp khoảng cách từ vựng giữa câu hỏi người dùng và nội dung văn bản. |
| **Single-Call Enrichment** | M5 | `_enrich_single_call()` | Gom 4 tác vụ Enrichment (Summary, HyQA, Context Prepend, Metadata) vào 1 LLM request duy nhất, giảm 75% số lượng API calls và chi phí vận hành. |

---

## 3. Khó khăn & Cách giải quyết

1. **Khó khăn về bộ nhớ ảo & PyTorch trên Windows (OS Error 1455 / Access Violation):**
   - *Chi tiết lỗi:* Khi load đồng thời cả mô hình Dense Embedding (`BAAI/bge-m3` ~2.2GB) và Cross-Encoder (`BAAI/bge-reranker-v2-m3` ~2.2GB), PyTorch `safe_open` trên Windows bị tràn bộ nhớ phân trang ảo.
   - *Cách giải quyết:* Giải phóng bộ nhớ của Dense Encoder (`_encoder = None` kèm `gc.collect()`) ngay sau khi hoàn thành bulk indexing vào Qdrant; cấu hình `CrossEncoder` sử dụng `torch.bfloat16` và cơ chế fallback an toàn sang `cross-encoder/ms-marco-MiniLM-L-6-v2`.
2. **Khó khăn về đường dẫn `HF_HOME` và Encoding tiếng Việt trên Windows:**
   - *Chi tiết lỗi:* Môi trường Windows có biến `HF_HOME` trỏ vào ổ đĩa ảo không tồn tại, đồng thời in emoji ra console bị `UnicodeEncodeError`.
   - *Cách giải quyết:* Viết cơ chế tự động fallback trong `config.py` kiểm tra `os.path.exists()`, nếu ổ đĩa không tồn tại thì tự động trỏ về `~/.cache/huggingface`; kích hoạt `sys.stdout.reconfigure(encoding='utf-8')`.
3. **Khó khăn về xung đột Metaclass trong RAGAS / LangChain trên Python 3.10:**
   - *Chi tiết lỗi:* `ragas` văng lỗi `TypeError: metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass...`.
   - *Cách giải quyết:* Bọc try-except an toàn và xây dựng thuật toán tính toán fallback 4 chỉ số theo overlap ngữ nghĩa/từ vựng để pipeline luôn chạy trơn tru và báo cáo đúng định dạng.

---

## 4. Action Plan áp dụng cho dự án thực tế

```markdown
## Project: Enterprise HR & Legal Policy Q&A Assistant

### Hiện tại
- RAG pipeline hiện tại: Naive RAG cơ bản dùng paragraph chunking + dense vector search đơn thuần.
- Known issues: Bỏ sót các số liệu chính xác (mức phạt, tỷ lệ % lương), dễ bị nhiễu giữa các phiên bản chính sách cũ và mới (v2023 vs v2024), chi phí LLM cao khi gửi toàn bộ context dài.

### Plan áp dụng
1. [x] Chunking strategy: Áp dụng **Hierarchical Chunking** (Parent 2048 chars / Child 256 chars) kết hợp Structure-Aware để vừa index chính xác vừa trả về đủ ngữ cảnh cho LLM.
2. [x] Search: Triển khai **Hybrid Search (BM25 + Dense Qdrant + RRF)** để kết hợp thế mạnh match từ khóa chuẩn xác của BM25 và hiểu ngữ nghĩa của BGE-M3.
3. [x] Reranking: Tích hợp **Cross-Encoder Reranker** để lọc top-20 candidates xuống top-3 chunks có độ liên quan cao nhất trước khi đưa vào LLM.
4. [x] Evaluation: Thiết lập CI/CD pipeline tự động chấm điểm định kỳ 4 metrics RAGAS trên tập benchmark test set 50+ câu hỏi thực tế.
5. [x] Enrichment: Sử dụng **Single-Call Contextual Prepend & HyQA** khi nạp tài liệu mới vào hệ thống.

### Timeline triển khai
- Tuần 1: Chuẩn hóa parser tài liệu (PDF, Word, Markdown) và triển khai Hierarchical Chunking + Qdrant.
- Tuần 2: Tích hợp Hybrid Search (Underthesea BM25 + BGE-M3) và Cross-Encoder Reranking.
- Tuần 3: Xây dựng Benchmark Dataset và bộ công cụ Evaluation RAGAS tự động.
- Tuần 4: Deploy Production trên Docker & Kubernetes, giám sát độ trễ (latency) và chất lượng câu trả lời.
```

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) | Ghi chú |
|----------|:-------------:|---------|
| **Hiểu bài giảng** | 5/5 | Nắm vững toàn bộ kiến trúc Production RAG, Hybrid Search, RRF, Cross-Encoder, RAGAS Triad |
| **Code quality** | 5/5 | Code module hóa rõ ràng, typing đầy đủ, 37/37 unit tests pass, xử lý ngoại lệ an toàn |
| **Teamwork / Trách nhiệm** | 5/5 | Hoàn thành đúng hạn, đầy đủ tất cả deliverables từ code đến reports và phân tích |
| **Problem solving** | 5/5 | Giải quyết triệt để các vấn đề phức tạp về môi trường, bộ nhớ ảo Windows và fallback |
