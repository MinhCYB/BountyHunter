"""
================================================================================
MODULE: DATA PROFILER & DATA DICTIONARY GENERATOR
================================================================================
Mục đích: Quét các bảng dữ liệu THÔ (Raw CSV) từ config.SOURCE_TABLES, 
trích xuất siêu dữ liệu, phân bố giá trị và nhận diện các cột bất thường.
Đầu ra: Báo cáo Markdown tối giản phục vụ phân tích hệ thống và lập hồ sơ dữ liệu.
"""

import pandas as pd
from pathlib import Path
from src.config import config

logger = config.get_logger(__name__)

def run_prof():
    logger.info("Bắt đầu quét danh sách SOURCE_TABLES để lập báo cáo chất lượng dữ liệu...")
    
    # Định dạng file đầu ra chuyên nghiệp
    output_file = config.REPORTS / "data_profiling_report.md"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# BÁO CÁO CẤU TRÚC VÀ CHẤT LƯỢNG DỮ LIỆU THÔ (DATA PROFILING REPORT)\n\n")
        f.write("Bản báo cáo này cung cấp thông số chi tiết về cấu trúc, tỷ lệ khuyết thiếu "
                "và đặc điểm phân bố của dữ liệu nguồn trước khi thực hiện tiền xử lý.\n\n")
        
        source_tables = config.SOURCE_TABLES
        if not source_tables:
            logger.warning("Danh sách config.SOURCE_TABLES trống.")
            return
            
        for table_name in source_tables:
            file_path = config.RAW / f"{table_name}.csv"
            
            if not file_path.exists():
                logger.warning(f"Không tìm thấy file: {file_path}")
                f.write(f"## Bảng: `{table_name}.csv`\n")
                f.write(f"> CẢNH BÁO: Tệp dữ liệu không tồn tại trong hệ thống.\n\n---\n\n")
                continue
                
            logger.info(f"-> Phân tích bảng: {table_name}")
            
            df = pd.read_csv(file_path, low_memory=False)
            total_rows = len(df)
            
            f.write(f"## Bảng: `{table_name}.csv`\n")
            f.write(f"- Tổng số dòng: {total_rows:,}\n")
            f.write(f"- Tổng số cột: {len(df.columns)}\n\n")
            
            # Bảng Metadata (Đã loại bỏ cột PK)
            f.write("| Tên Cột | Kiểu Dữ Liệu | Null (%) | Unique | Phân bố / Giá trị (Distribution) | Ghi chú (Flags) |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
            
            for col in df.columns:
                dtype = str(df[col].dtype)
                null_pct = (df[col].isna().sum() / total_rows) * 100 if total_rows > 0 else 0
                n_unique = df[col].nunique(dropna=True)
                
                # CHỈ GIỮ LẠI CÁC CẢNH BÁO VỀ CHẤT LƯỢNG DỮ LIỆU
                flags = []
                if n_unique == 1:
                    flags.append("CONSTANT")
                if null_pct >= 90.0:
                    flags.append("HIGH NULL")
                
                flag_str = ", ".join(flags) if flags else "-"
                
                # Phân tích phân bố giá trị
                distribution_text = ""
                if n_unique == 0:
                    distribution_text = "All NaN"
                elif pd.api.types.is_numeric_dtype(df[col]):
                    distribution_text = f"Range: {df[col].min():g} -> {df[col].max():g}"
                else:
                    valid_data = df[col].dropna()
                    if n_unique <= 15:
                        distribution_text = f"Values: {', '.join(map(str, valid_data.unique()))}"
                    else:
                        top_5 = valid_data.value_counts().head(5).index.tolist()
                        distribution_text = f"Top 5: {', '.join(map(str, top_5))}..."
                
                distribution_text = distribution_text.replace("|", "\\|").replace("\n", " ")
                
                f.write(f"| `{col}` | `{dtype}` | {null_pct:.1f}% | {n_unique:,} | {distribution_text} | {flag_str} |\n")
            
            f.write("\n---\n\n")
            
        # --- PHỤ LỤC GHI CHÚ ---
        f.write("## Phụ lục: Giải thích các Ghi chú (Flags)\n")
        f.write("- **CONSTANT:** Cột chỉ chứa duy nhất một giá trị (biến thiên bằng 0). Không mang lại giá trị phân tích, cần cân nhắc loại bỏ.\n")
        f.write("- **HIGH NULL:** Cột có tỷ lệ dữ liệu trống từ 90% trở lên. Cần kiểm tra lại quy trình thu thập hoặc nghiệp vụ liên quan.\n")

    logger.info(f"Hoàn tất! Báo cáo đã được lưu tại: {output_file}")
