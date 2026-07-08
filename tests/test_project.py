import hashlib
import os
import tempfile
from pathlib import Path

import pytest

from muninn_local.project import find_git_root, get_project_id, get_project_name


def _resolved_tmp():
    return Path(tempfile.mkdtemp()).resolve()


class TestFindGitRoot:
    def test_finds_git_root_from_cwd(self):
        tmp = _resolved_tmp()
        git_root = tmp / "repo"
        (git_root / ".git").mkdir(parents=True)
        subdir = git_root / "sub" / "deep"
        subdir.mkdir(parents=True)

        result = find_git_root(subdir)
        assert result == git_root

    def test_finds_git_root_from_git_root_itself(self):
        tmp = _resolved_tmp()
        git_root = tmp / "repo"
        (git_root / ".git").mkdir(parents=True)

        result = find_git_root(git_root)
        assert result == git_root

    def test_returns_none_when_no_git(self):
        tmp = _resolved_tmp()
        nodir = tmp / "no_git"
        nodir.mkdir(parents=True)

        result = find_git_root(nodir)
        assert result is None

    def test_accepts_string_path(self):
        tmp = _resolved_tmp()
        git_root = tmp / "repo"
        (git_root / ".git").mkdir(parents=True)

        result = find_git_root(str(git_root))
        assert result == git_root

    def test_finds_git_file_for_worktrees(self):
        tmp = _resolved_tmp()
        git_root = tmp / "repo"
        git_root.mkdir(parents=True)
        git_file = git_root / ".git"
        git_file.write_text("gitdir: /some/other/path/.git")

        result = find_git_root(git_root)
        assert result == git_root


class TestGetProjectId:
    def test_uses_env_variable_if_set(self, monkeypatch):
        monkeypatch.setenv("MUNINN_PROJECT_ID", "my-custom-project")
        result = get_project_id()
        assert result == "my-custom-project"

    def test_uses_git_root_hash_when_git_exists(self):
        tmp = _resolved_tmp()
        git_root = tmp / "repo"
        (git_root / ".git").mkdir(parents=True)
        expected_hash = hashlib.sha256(str(git_root).encode()).hexdigest()[:12]

        result = get_project_id(str(git_root))
        assert result == expected_hash

    def test_uses_cwd_hash_when_no_git(self):
        tmp = _resolved_tmp()
        nodir = tmp / "no_git"
        nodir.mkdir(parents=True)
        expected_hash = hashlib.sha256(str(nodir).encode()).hexdigest()[:12]

        result = get_project_id(str(nodir))
        assert result == expected_hash

    def test_returns_default_when_cwd_is_none_and_no_env(self, monkeypatch):
        monkeypatch.delenv("MUNINN_PROJECT_ID", raising=False)
        result = get_project_id()
        assert isinstance(result, str)
        assert len(result) == 12


class TestGetProjectName:
    def test_returns_git_root_basename(self):
        tmp = _resolved_tmp()
        git_root = tmp / "myproject"
        (git_root / ".git").mkdir(parents=True)

        result = get_project_name(str(git_root))
        assert result == "myproject"

    def test_returns_cwd_basename_when_no_git(self):
        tmp = _resolved_tmp()
        nodir = tmp / "fallback-project"
        nodir.mkdir(parents=True)

        result = get_project_name(str(nodir))
        assert result == "fallback-project"

    def test_returns_cwd_basename_as_default(self):
        result = get_project_name()
        assert isinstance(result, str)
        assert len(result) > 0
