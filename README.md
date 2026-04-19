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
datathon_2026_teamX/
│
├── data/                   
│   ├── raw/              Dữ liệu thô  
│   ├── processed/        Dữ liệu đã qua xử lý
│   └── features/         Chứa các feature đã tạo
│
├── notebooks/            Nháp, vẽ biểu đồ, eda cá nhân
│   └── ....ipynb 
│
├── src/                      
│   ├── config.py         Chứa các biến môi trường  
│   ├── data_prep.py      Gộp dữ liệu, xử lý ngoại lệ, lỗi 
│   ├── feature_eng.py    Tạo đặc trưng
│   ├── validation.py     Cắt data (nà ná na na train_test_split)
│   ├── train.py          
│   └── inference.py      Hàm dự đoán  
│
├── models/               Chứa model
│
├── submissions/          Chứa bài nộp
│
├── reports/              Chứa báo cáo        
│   └── figures/          Chứa biểu đồ ...
│
├── requirements.txt        
├── README.md               
└── main.py                 
```