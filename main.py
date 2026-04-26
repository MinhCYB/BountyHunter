import argparse
from src.config import config
from src import data_audit, data_profiler, data_prep, feature_eng, train, model_selection, inference 

logger = config.get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Pipeline dự báo doanh thu - Datathon 2026 - BountyHunter")
    
    parser.add_argument("--audit", action="store_true", help="Quét lỗi và xuất báo cáo QA") 
    parser.add_argument("--prof", action="store_true", help="Hồ sơ hoá dữ liệu") 
    parser.add_argument("--prep", action="store_true", help="Làm sạch dữ liệu")
    parser.add_argument("--feat", action="store_true", help="Tạo đặc trưng")
    parser.add_argument("--train", action="store_true", help="Huấn luyện mô hình")
    parser.add_argument("--infer", action="store_true", help="Dự đoán và xuất submissions")
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ pipeline prep -> feat -> train -> infer")
    
    model_selection.add_arguments(parser)

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return
    
    model_selection.dispatch(args)
    
    if args.audit:
        data_audit.run_audit()

    if args.prof:
        data_profiler.run_prof()

    if args.all or args.prep:
        data_prep.run_prep() 

    if args.all or args.feat:
        feature_eng.run_feature_engineering()

    if args.all or args.train:
        train.run_train()

    if args.all or args.infer:
        inference.run_inference()
    
    logger.info("Done!!!")

if __name__ == "__main__":
    main()