# BountyHunter — Dự báo Revenue & Margin

> **Datathon Submission** · Team **BountyHunter**

Pipeline machine learning dự báo **Revenue** và **Gross Margin** theo ngày.

---

## Cấu trúc thư mục

```
BOUNTYHUNTER/
├── data/
│   ├── raw/                    # Dữ liệu thô đầu vào (CSV)
│   └── processed/              # Dữ liệu trung gian (Parquet)
├── logs/                       # Log tự động sinh ra sau mỗi lần chạy
├── models/                     # Model đã huấn luyện (pkl, json)
├── notebooks/                  # Notebook EDA & trực quan hóa
├── outputs/
│   ├── qa_report.csv           # Báo cáo kiểm tra chất lượng dữ liệu
│   ├── baseline_metrics.json   # Metrics của baseline model
│   ├── submission.csv          # File dự báo nộp cuộc thi
│   ├── shap_report_revenue.html
│   └── shap_report_margin.html
├── src/
│   ├── data_audit.py           # Kiểm tra chất lượng dữ liệu thô
│   ├── data_prep.py            # Làm sạch & chuẩn bị dữ liệu
│   ├── feature_eng.py          # Feature engineering
│   ├── train.py                # Huấn luyện mô hình
│   └── inference.py            # Dự báo & xuất submission
├── config.yaml                 # Cấu hình toàn bộ pipeline
├── main.py                     # Entry point
└── requirements.txt
```

---

## Hướng dẫn chạy

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 2. Đặt dữ liệu thô vào thư mục `data/raw/`

```
data/raw/
├── sales.csv
├── orders.csv
├── order_items.csv
├── products.csv
├── customers.csv
├── inventory.csv
├── promotions.csv
├── web_traffic.csv
└── sample_submission.csv
```

### 3. Chạy toàn bộ pipeline

```bash
python main.py --config config.yaml --step all
```

### 4. Hoặc chạy từng bước riêng lẻ

```bash
python main.py --step audit      # Kiểm tra chất lượng dữ liệu
python main.py --step prep       # Chuẩn bị dữ liệu
python main.py --step feature    # Feature engineering
python main.py --step train      # Huấn luyện mô hình
python main.py --step inference  # Dự báo & xuất submission.csv
```

> Kết quả dự báo được lưu tại `outputs/submission.csv`.