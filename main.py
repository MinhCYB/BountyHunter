"""
main.py — Orchestrator cho pipeline dự báo Revenue & Margin.

Entry point duy nhất để chạy toàn bộ hoặc từng step của pipeline.
Mọi tham số được đọc từ config.yaml — không hardcode bất kỳ giá trị nào.

Cách chạy:
    python main.py --config config.yaml --step all
    python main.py --step prep
    python main.py --step train
    python main.py --step inference
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Logger module-level — sẽ được cấu hình lại trong setup_logging()
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping step → lệnh script tương ứng
# ---------------------------------------------------------------------------
STEP_SCRIPT_MAP: dict[str, str] = {
    "audit":     "src/data_audit.py",
    "prep":      "src/data_prep.py",
    "feature":   "src/feature_eng.py",
    "train":     "src/train.py",
    "inference": "src/inference.py",
}

# Thứ tự chạy khi --step all
ALL_STEPS: list[str] = ["audit", "prep", "feature", "train", "inference"]


# ---------------------------------------------------------------------------
# Các hàm chính
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """
    Đọc và trả về nội dung config.yaml dưới dạng dict.

    Parameters
    ----------
    config_path : Path
        Đường dẫn tới file config.yaml.

    Returns
    -------
    dict
        Toàn bộ cấu hình pipeline.

    Raises
    ------
    FileNotFoundError
        Nếu file config không tồn tại tại đường dẫn đã cho.
    yaml.YAMLError
        Nếu nội dung file không parse được thành YAML hợp lệ.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file config: {config_path.resolve()}"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config: dict = yaml.safe_load(f)
    return config


def setup_logging(config: dict) -> None:
    """
    Cấu hình logging toàn bộ pipeline từ config.

    Ghi log đồng thời ra file (logs/pipeline_{timestamp}.log)
    và ra console (StreamHandler).
    Tên file log được tạo tự động theo thời gian khởi chạy.

    Parameters
    ----------
    config : dict
        Dict cấu hình đã load từ config.yaml.

    Returns
    -------
    None
    """
    log_dir = Path(config["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"

    log_level_str: str = config["logging"]["level"]
    log_format: str = config["logging"]["format"]
    log_level: int = getattr(logging, log_level_str.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger.info("Logging đã khởi tạo — ghi vào: %s", log_file)
    logger.info("Log level: %s", log_level_str)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments cho pipeline orchestrator.

    Hỗ trợ hai tham số:
      --config : đường dẫn tới config.yaml (mặc định: config.yaml)
      --step   : bước cần chạy — một trong các giá trị hợp lệ hoặc 'all'

    Returns
    -------
    argparse.Namespace
        Namespace chứa các giá trị đã parse: args.config (Path), args.step (str).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Orchestrator pipeline dự báo Revenue & Margin.\n"
            "Chạy toàn bộ pipeline hoặc từng step riêng lẻ."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        metavar="PATH",
        help="Đường dẫn tới file config.yaml (mặc định: config.yaml)",
    )
    valid_steps = list(STEP_SCRIPT_MAP.keys()) + ["all"]
    parser.add_argument(
        "--step",
        choices=valid_steps,
        default="all",
        help=(
            "Bước pipeline cần chạy. Các giá trị hợp lệ:\n"
            "  audit     — Kiểm tra dữ liệu thô\n"
            "  prep      — Chuẩn bị dữ liệu, tạo base_table.parquet\n"
            "  feature   — Feature engineering, tạo feature_table.parquet\n"
            "  train     — Huấn luyện model\n"
            "  inference — Sinh submission.csv và shap_report.html\n"
            "  all       — Chạy tuần tự tất cả các bước (mặc định)"
        ),
    )
    return parser.parse_args()


def run_step(step_name: str, config_path: Path) -> int:
    """
    Chạy một step pipeline bằng subprocess và trả về exit code.

    Lệnh được thực thi: python <script_path> --config <config_path>
    Log thời gian bắt đầu, kết thúc, duration và exit code của step.
    Stdout và stderr của subprocess được chuyển tiếp trực tiếp ra console
    (không bắt lại) để giữ tính real-time của log con.

    Parameters
    ----------
    step_name : str
        Tên step cần chạy. Phải là key trong STEP_SCRIPT_MAP.
    config_path : Path
        Đường dẫn tới file config.yaml, truyền qua cho subprocess.

    Returns
    -------
    int
        Exit code của subprocess (0 = thành công, khác 0 = thất bại).
    """
    script_path = STEP_SCRIPT_MAP[step_name]
    cmd: list[str] = [sys.executable, script_path, "--config", str(config_path)]

    logger.info("=" * 60)
    logger.info("BẮT ĐẦU STEP: [%s]", step_name.upper())
    logger.info("Lệnh: %s", " ".join(cmd))

    start_time = datetime.now()

    result = subprocess.run(
        cmd,
        # Không dùng capture_output — để stdout/stderr của subprocess
        # hiển thị trực tiếp ra terminal và được logging handler bắt
    )

    end_time = datetime.now()
    duration_sec = (end_time - start_time).total_seconds()
    exit_code: int = result.returncode

    if exit_code == 0:
        logger.info(
            "HOÀN THÀNH STEP: [%s] — Duration: %.2fs — Exit code: %d",
            step_name.upper(),
            duration_sec,
            exit_code,
        )
    else:
        logger.error(
            "THẤT BẠI STEP: [%s] — Duration: %.2fs — Exit code: %d",
            step_name.upper(),
            duration_sec,
            exit_code,
        )

    return exit_code


def run_pipeline(steps: list[str], config_path: Path) -> None:
    """
    Chạy tuần tự danh sách các step pipeline.

    Dừng ngay lập tức nếu bất kỳ step nào thất bại (exit code != 0).
    Sau khi tất cả step hoàn thành, log tổng thời gian pipeline.

    Parameters
    ----------
    steps : list[str]
        Danh sách tên step cần chạy theo thứ tự.
        Mỗi phần tử phải là key hợp lệ trong STEP_SCRIPT_MAP.
    config_path : Path
        Đường dẫn tới file config.yaml.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        Gọi sys.exit(exit_code) nếu bất kỳ step nào trả về exit code != 0.
    """
    logger.info("Pipeline khởi động với %d step(s): %s", len(steps), steps)
    pipeline_start = datetime.now()

    for step_name in steps:
        exit_code = run_step(step_name, config_path)

        if exit_code != 0:
            logger.error(
                "Pipeline dừng tại step [%s] — exit code: %d. "
                "Các step chưa chạy: %s",
                step_name,
                exit_code,
                steps[steps.index(step_name) + 1 :],
            )
            sys.exit(exit_code)

    pipeline_end = datetime.now()
    total_duration_sec = (pipeline_end - pipeline_start).total_seconds()

    logger.info("=" * 60)
    logger.info(
        "PIPELINE HOÀN THÀNH — Tổng thời gian: %.2fs (%.1f phút)",
        total_duration_sec,
        total_duration_sec / 60,
    )
    logger.info("=" * 60)


def _ensure_directories(config: dict) -> None:
    """
    Tạo các thư mục cần thiết của pipeline nếu chưa tồn tại.

    Tạo: data/processed/, models/, outputs/, logs/.
    Tất cả đường dẫn được đọc từ config.

    Parameters
    ----------
    config : dict
        Dict cấu hình đã load từ config.yaml.

    Returns
    -------
    None
    """
    dirs_to_create: list[Path] = [
        Path(config["paths"]["processed_dir"]),
        Path(config["paths"]["model_dir"]),
        Path(config["paths"]["output_dir"]),
        Path(config["paths"]["log_dir"]),
    ]

    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug("Thư mục đã sẵn sàng: %s", dir_path.resolve())

    logger.info(
        "Đã kiểm tra và tạo đủ %d thư mục pipeline.", len(dirs_to_create)
    )


def main() -> None:
    """
    Entry point chính của Orchestrator.

    Thứ tự thực hiện:
      1. Parse CLI arguments.
      2. Load config.yaml.
      3. Cấu hình logging.
      4. Tạo các thư mục cần thiết.
      5. Xác định danh sách step cần chạy.
      6. Gọi run_pipeline().

    Returns
    -------
    None
    """
    args = parse_args()
    config_path: Path = args.config
    step_arg: str = args.step

    # --- Load config trước để setup logging ---
    config = load_config(config_path)

    # --- Cấu hình logging từ config ---
    setup_logging(config)

    logger.info("Config đã load từ: %s", config_path.resolve())
    logger.info("Step được yêu cầu: [%s]", step_arg.upper())

    # --- Tạo thư mục output cần thiết ---
    _ensure_directories(config)

    # --- Xác định danh sách step cần chạy ---
    steps_to_run: list[str] = ALL_STEPS if step_arg == "all" else [step_arg]

    # --- Chạy pipeline ---
    run_pipeline(steps_to_run, config_path)


if __name__ == "__main__":
    main()