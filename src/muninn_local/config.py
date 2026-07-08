import os
from dataclasses import dataclass, field


@dataclass
class Config:
    data_dir: str = field(
        default_factory=lambda: os.path.expanduser("~/.config/opencode/muninn")
    )
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "mxbai-embed-large"

    @property
    def chroma_dir(self) -> str:
        return os.path.join(self.data_dir, "chroma")


def get_config() -> Config:
    return Config(
        data_dir=os.path.expanduser(
            os.environ.get("MUNINN_DATA_DIR", "~/.config/opencode/muninn")
        ),
        ollama_url=os.environ.get("MUNINN_OLLAMA_URL", "http://localhost:11434"),
        embed_model=os.environ.get("MUNINN_EMBED_MODEL", "mxbai-embed-large"),
    )