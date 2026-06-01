from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

DIRECTORY_KEYS = (
    "data_dir",
    "papers_dir",
    "texts_dir",
    "chroma_dir",
    "metadata_dir",
    "reports_dir",
    "downloads_dir",
    "zotero_dir",
    "logs_dir",
)


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load project configuration and ensure runtime data directories exist."""
    load_project_dotenv()
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")

    for key in DIRECTORY_KEYS:
        directory = config.get(key)
        if not directory:
            raise KeyError(f"Missing required directory config: {key}")
        Path(str(directory)).expanduser().mkdir(parents=True, exist_ok=True)

    return config


def load_project_dotenv() -> bool:
    """Load .env from the project root without using dotenv's search behavior."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return False
    return load_dotenv(dotenv_path=env_path, override=True)
