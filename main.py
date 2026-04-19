import argparse
from src import data_prep, feature_eng, train, inference

def main():
    parser = argparse.ArgumentParser(description="Pipeline dự báo doanh thu - Datathon 2026 - BountyHunter")
    
    parser.add_argument("--prep", action="store_true", help="Làm sạch dữ liệu")
    parser.add_argument("--feat", action="store_true", help="Tạo đặc trưng")
    parser.add_argument("--train", action="store_true", help="Huấn luyện mô hình")
    parser.add_argument("--infer", action="store_true", help="Dự đoán và xuất submissions")
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ pipeline")
    
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    if args.all or args.prep:
        data_prep.run_prep() 

    if args.all or args.feat:
        feature_eng.run_features()

    if args.all or args.train:
        train.run_train()

    if args.all or args.infer:
        inference.run_inference()
    
    print("Done!")

if __name__ == "__main__":
    main()