"""Repository survey, on fixtures shaped like the real repos it must handle."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from claimscope.repo import (
    RepoError,
    _downloads_data,
    extract_arguments,
    read_file,
    survey,
)

CIFAR_MAIN = '''\
"""Train CIFAR10 with PyTorch."""
import argparse
import torchvision

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--resume", "-r", action="store_true")
args = parser.parse_args()

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

for epoch in range(200):
    train(epoch)
'''

SELF_CONTAINED = """\
import numpy as np

def main():
    data = np.random.rand(100, 10)
    print(data.mean())

main()
"""


def _git_repo(path: Path, files: dict[str, str]) -> Path:
    """Create a real git repository so clone() has something to work with."""
    path.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    for command in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"],
    ):
        subprocess.run(command, cwd=path, check=True, capture_output=True)
    return path


class TestExtractArguments:
    def test_finds_argparse_flags(self) -> None:
        assert extract_arguments(CIFAR_MAIN) == ["--lr", "--resume", "-r"]

    def test_returns_nothing_when_there_are_no_flags(self) -> None:
        assert extract_arguments(SELF_CONTAINED) == []

    def test_ignores_flags_in_comments_and_strings(self) -> None:
        # A regex would match both of these; the AST does not.
        source = '# --fake-flag\nmessage = "use --another-fake"\n'

        assert extract_arguments(source) == []

    def test_survives_a_syntax_error(self) -> None:
        # Repos contain Python 2 files and templates; they must not crash the survey.
        assert extract_arguments("print 'python 2'") == []

    def test_finds_click_options(self) -> None:
        source = '@click.option("--epochs", default=10)\ndef main(epochs): pass\n'

        assert extract_arguments(source) == ["--epochs"]


class TestDownloadDetection:
    """The sandbox has no network, so this has to be right."""

    @pytest.mark.parametrize(
        "source",
        [
            "datasets.CIFAR10(root='./data', download=True)",
            "urllib.request.urlretrieve(url, path)",
            "requests.get(url)",
            "model = AutoModel.from_pretrained('bert-base')",
            "load_dataset('glue')",
        ],
    )
    def test_detects_runtime_downloads(self, source: str) -> None:
        assert _downloads_data(source)

    def test_does_not_flag_offline_code(self) -> None:
        assert not _downloads_data(SELF_CONTAINED)

    def test_does_not_flag_download_false(self) -> None:
        assert not _downloads_data("datasets.CIFAR10(root='./data', download=False)")


class TestImportInference:
    """Most research repos ship no requirements file, so imports are the signal."""

    def test_finds_third_party_imports(self, tmp_path: Path) -> None:
        source = _git_repo(tmp_path / "src", {"train.py": CIFAR_MAIN})

        result = survey(str(source), tmp_path / "clone")

        assert "torchvision" in result.imported_packages

    def test_ignores_the_standard_library(self, tmp_path: Path) -> None:
        source = _git_repo(
            tmp_path / "src", {"train.py": "import os, json, argparse\nimport torch\n"}
        )

        result = survey(str(source), tmp_path / "clone")

        assert result.imported_packages == ["torch"]

    def test_ignores_the_repos_own_modules(self, tmp_path: Path) -> None:
        # "models" is a directory in the repo, not a package to install.
        source = _git_repo(
            tmp_path / "src",
            {
                "train.py": "import torch\nfrom models import ResNet\nimport utils\n",
                "models/__init__.py": "",
                "utils.py": "",
            },
        )

        result = survey(str(source), tmp_path / "clone")

        assert result.imported_packages == ["torch"]

    def test_maps_import_names_to_package_names(self, tmp_path: Path) -> None:
        source = _git_repo(
            tmp_path / "src", {"train.py": "import cv2\nimport sklearn\nfrom PIL import Image\n"}
        )

        result = survey(str(source), tmp_path / "clone")

        assert "opencv-python-headless" in result.imported_packages
        assert "scikit-learn" in result.imported_packages
        assert "pillow" in result.imported_packages

    def test_ignores_relative_imports(self, tmp_path: Path) -> None:
        source = _git_repo(
            tmp_path / "src", {"train.py": "from .layers import Block\nimport torch\n"}
        )

        result = survey(str(source), tmp_path / "clone")

        assert result.imported_packages == ["torch"]


class TestSurvey:
    def test_surveys_a_flat_repo(self, tmp_path: Path) -> None:
        source = _git_repo(tmp_path / "src", {"main.py": CIFAR_MAIN, "README.md": "# CIFAR"})

        result = survey(str(source), tmp_path / "clone")

        assert result.commit != "unknown"
        entry = result.primary_entrypoint()
        assert entry is not None
        assert entry.path == "main.py"
        assert entry.arguments == ["--lr", "--resume", "-r"]
        assert entry.downloads_data

    def test_finds_config_files(self, tmp_path: Path) -> None:
        source = _git_repo(
            tmp_path / "src",
            {
                "train.py": SELF_CONTAINED,
                "config/small.py": "batch_size = 8",
                "config/base.yaml": "lr: 0.1",
            },
        )

        result = survey(str(source), tmp_path / "clone")

        assert "config/small.py" in result.config_files
        assert "config/base.yaml" in result.config_files

    def test_prefers_train_py_over_other_scripts(self, tmp_path: Path) -> None:
        source = _git_repo(
            tmp_path / "src", {"train.py": SELF_CONTAINED, "main.py": SELF_CONTAINED}
        )

        result = survey(str(source), tmp_path / "clone")

        entry = result.primary_entrypoint()
        assert entry is not None
        assert entry.path == "train.py"

    def test_reads_declared_dependencies(self, tmp_path: Path) -> None:
        source = _git_repo(
            tmp_path / "src",
            {"train.py": SELF_CONTAINED, "requirements.txt": "torch==2.0\nnumpy\n"},
        )

        result = survey(str(source), tmp_path / "clone")

        assert result.dependencies == ["torch==2.0", "numpy"]

    def test_reads_the_readme(self, tmp_path: Path) -> None:
        source = _git_repo(
            tmp_path / "src",
            {"train.py": SELF_CONTAINED, "README.md": "# How to run\npython train.py"},
        )

        result = survey(str(source), tmp_path / "clone")

        assert "How to run" in result.readme_excerpt

    def test_skips_vendored_directories(self, tmp_path: Path) -> None:
        source = _git_repo(
            tmp_path / "src",
            {"train.py": SELF_CONTAINED, "node_modules/lib/index.py": "x = 1"},
        )

        result = survey(str(source), tmp_path / "clone")

        assert not any("node_modules" in path for path in result.python_files)

    def test_reuses_an_existing_clone(self, tmp_path: Path) -> None:
        source = _git_repo(tmp_path / "src", {"train.py": SELF_CONTAINED})
        clone_dir = tmp_path / "clone"

        first = survey(str(source), clone_dir)
        second = survey(str(source), clone_dir)

        assert first.commit == second.commit

    def test_a_bad_url_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RepoError, match="could not clone"):
            survey(str(tmp_path / "does-not-exist"), tmp_path / "clone")


class TestReadFile:
    def test_reads_a_file_from_the_clone(self, tmp_path: Path) -> None:
        source = _git_repo(tmp_path / "src", {"train.py": SELF_CONTAINED})
        clone_dir = tmp_path / "clone"
        survey(str(source), clone_dir)

        assert "numpy" in read_file(clone_dir, "train.py")

    def test_refuses_to_escape_the_repository(self, tmp_path: Path) -> None:
        source = _git_repo(tmp_path / "src", {"train.py": SELF_CONTAINED})
        clone_dir = tmp_path / "clone"
        survey(str(source), clone_dir)

        with pytest.raises(RepoError, match="escapes the repository"):
            read_file(clone_dir, "../../../etc/passwd")

    def test_missing_files_raise(self, tmp_path: Path) -> None:
        source = _git_repo(tmp_path / "src", {"train.py": SELF_CONTAINED})
        clone_dir = tmp_path / "clone"
        survey(str(source), clone_dir)

        with pytest.raises(RepoError, match="no such file"):
            read_file(clone_dir, "absent.py")
