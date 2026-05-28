from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from research_agent.config import DIRECTORY_KEYS, load_config


def main() -> None:
    config = load_config()

    for key in DIRECTORY_KEYS:
        path = Path(config[key])
        status = "OK" if path.exists() else "MISSING"
        print(f"{key}: {path} [{status}]")

    data_dir = Path(config["data_dir"])
    test_file = data_dir / ".write_test"

    try:
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        print(f"writable: {data_dir} [OK]")
    except OSError as exc:
        print(f"writable: {data_dir} [FAILED: {exc}]")


if __name__ == "__main__":
    main()
