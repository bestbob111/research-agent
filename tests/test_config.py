from pathlib import Path
import os

from research_agent import config as config_module
from research_agent.config import DIRECTORY_KEYS, load_config, load_project_dotenv


def test_load_config_reads_default_values():
    config = load_config()

    assert config["chat_model"] == "qwen3:14b"
    assert config["papers_dir"] == "/mnt/bigdata/research-agent/papers"


def test_load_config_creates_data_directories():
    config = load_config()

    for key in DIRECTORY_KEYS:
        assert Path(config[key]).exists()


def test_load_project_dotenv_loads_project_root_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "old-key")
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=new-key\n", encoding="utf-8")

    loaded = load_project_dotenv()

    assert loaded is True
    assert os.environ["DEEPSEEK_API_KEY"] == "new-key"
