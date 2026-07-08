import hashlib
import os
from pathlib import Path


def find_git_root(cwd: str | Path) -> Path | None:
    current = Path(cwd).resolve()
    for parent in [current] + list(current.parents):
        git_path = parent / ".git"
        if git_path.exists():
            return parent
    return None


def get_project_id(cwd: str | Path | None = None) -> str:
    env_id = os.environ.get("MUNINN_PROJECT_ID")
    if env_id:
        return env_id
    search_cwd = cwd if cwd is not None else os.getcwd()
    git_root = find_git_root(search_cwd)
    if git_root is not None:
        return hashlib.sha256(str(git_root).encode()).hexdigest()[:12]
    if search_cwd is not None:
        return hashlib.sha256(str(Path(search_cwd).resolve()).encode()).hexdigest()[:12]
    return "default"


def get_project_name(cwd: str | Path | None = None) -> str:
    search_cwd = cwd if cwd is not None else os.getcwd()
    git_root = find_git_root(search_cwd)
    if git_root is not None:
        return git_root.name
    return Path(search_cwd).resolve().name
