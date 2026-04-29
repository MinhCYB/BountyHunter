# BountyHunter

## Giới thiệu
Giải pháp dự báo doanh thu thương mại điện tử sử dụng kiến trúc Modular Pipeline, tối ưu cho bài toán chuỗi thời gian.

## Cài đặt & Kích hoạt

Copy và chạy các lệnh sau trong Terminal/Command Prompt:

```bash
    git clone https://github.com/MinhCYB/BountyHunter.git
    cd BountyHunter
    python -m venv venv
    # Windows: venv\Scripts\activate
    # Mac/Linux: source venv/bin/activate
    pip install -r requirements.txt
```

## Cấu trúc 

```text
BountyHunter/
├── data/                   # Phân hệ quản lý vòng đời dữ liệu (Data Lifecycle)
│   ├── raw/                # Vùng chứa dữ liệu nguyên bản (Raw Data) từ hệ thống nguồn.
│   ├── processed/          # Dữ liệu đã qua tiền xử lý, làm sạch và chuẩn hóa định dạng.
│   └── features/           # Kho lưu trữ đặc trưng (Feature Store) phục vụ huấn luyện.
│
├── models/                 # Không gian lưu trữ mô hình đã huấn luyện và tập siêu tham số tối ưu.
├── reports/                # Báo cáo kiểm toán tự động về cấu trúc và chất lượng dữ liệu.
├── submissions/            # Kết quả dự báo cuối cùng và dữ liệu chẩn đoán (Diagnostics).
├── logs/                   # Nhật ký hệ thống (System Logs) lưu vết quá trình thực thi.
│
├── src/                    # Phân hệ mã nguồn cốt lõi (Core Source Code)
│   ├── config.py           # Quản lý siêu tham số, đường dẫn và cấu hình tham chiếu toàn cục.
│   ├── data_audit.py       # Kịch bản kiểm toán, nhận diện điểm bất thường và sai lệch logic.
│   ├── data_profiler.py    # Kịch bản hồ sơ hóa, trích xuất siêu dữ liệu và tỷ lệ khuyết thiếu.
│   ├── data_prep.py        # Tiền xử lý, chuẩn hóa kiểu dữ liệu và tổng hợp theo cụm.
│   ├── feature_eng.py      # Trích xuất đặc trưng chuỗi thời gian (Calendar, Lag, Seasonality).
│   ├── validation.py       # Phân chia dữ liệu (Train/Test Split), nội suy (Imputation) và Cross-Validation.
│   ├── model_selection.py  # Khung tối ưu hóa siêu tham số (Grid Search) và đánh giá kiến trúc.
│   ├── train.py            # Huấn luyện mô hình độc lập cho từng mục tiêu dự báo.
│   └── inference.py        # Nội suy tương lai, áp dụng luật kinh doanh và xuất kết quả.
│
├── main.py                 # Bộ điều phối trung tâm (Orchestrator) tự động hóa luồng thực thi.
└── requirements.txt        # Danh sách định tuyến thư viện phụ thuộc của môi trường dự án.           
```