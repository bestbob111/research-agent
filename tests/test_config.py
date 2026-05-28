from pathlib import Path

from research_agent.config import DIRECTORY_KEYS, load_config


def test_load_config_reads_default_values():
    config = load_config()

    assert config["chat_model"] == "qwen3:14b"
    assert config["papers_dir"] == "/mnt/bigdata/research-agent/papers"


def test_load_config_creates_data_directories():
    config = load_config()

    for key in DIRECTORY_KEYS:
        assert Path(config[key]).exists()
