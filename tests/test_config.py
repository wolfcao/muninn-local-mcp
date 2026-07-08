import os
import pytest
from muninn_local.config import Config, get_config


class TestConfigDefaults:
    def test_default_data_dir(self):
        config = Config()
        assert config.data_dir == os.path.expanduser("~/.config/opencode/muninn")

    def test_default_ollama_url(self):
        config = Config()
        assert config.ollama_url == "http://localhost:11434"

    def test_default_embed_model(self):
        config = Config()
        assert config.embed_model == "mxbai-embed-large"

    def test_chroma_dir_is_under_data_dir(self):
        config = Config()
        assert config.chroma_dir == os.path.join(
            config.data_dir, "chroma"
        )


class TestConfigEnvOverride:
    def test_data_dir_from_env(self, monkeypatch):
        monkeypatch.setenv("MUNINN_DATA_DIR", "/tmp/muninn-test")
        config = get_config()
        assert config.data_dir == "/tmp/muninn-test"

    def test_ollama_url_from_env(self, monkeypatch):
        monkeypatch.setenv("MUNINN_OLLAMA_URL", "http://ollama:9999")
        config = get_config()
        assert config.ollama_url == "http://ollama:9999"

    def test_embed_model_from_env(self, monkeypatch):
        monkeypatch.setenv("MUNINN_EMBED_MODEL", "all-minilm")
        config = get_config()
        assert config.embed_model == "all-minilm"

    def test_tilde_expansion_in_data_dir(self, monkeypatch):
        monkeypatch.setenv("MUNINN_DATA_DIR", "~/my-muninn-data")
        config = get_config()
        assert config.data_dir.startswith("/")
        assert config.data_dir.endswith("/my-muninn-data")
        assert "~" not in config.data_dir