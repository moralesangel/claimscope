"""Clone and survey a paper's official repository (PLAN.md section 6, phase 5).

Repositories differ too much for a fixed adapter: one exposes a single ``--lr``
flag with the batch size and epoch count hardcoded, another keeps Python config
files in ``config/``. So this module does not try to *understand* a repo. It
gathers evidence -- entrypoints, config files, declared dependencies, the
arguments each entrypoint accepts -- and the codegen prompt asks the model to
write an adapter against that evidence.
"""

from __future__ import annotations

import ast
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

CLONE_TIMEOUT_S = 300

# Files that usually hold the training entrypoint, most likely first.
_ENTRYPOINT_NAMES = ("train.py", "main.py", "run.py", "train_net.py", "experiment.py")

_CONFIG_SUFFIXES = (".yaml", ".yml", ".json", ".toml", ".ini", ".cfg")
_CONFIG_DIRS = ("config", "configs", "conf", "experiments")

_DEPENDENCY_FILES = ("requirements.txt", "pyproject.toml", "setup.py", "environment.yml")

# Taken from the interpreter rather than hand-listed, so it cannot drift.
_STDLIB_MODULES = frozenset(sys.stdlib_module_names)

# Import names whose PyPI package is spelled differently.
_IMPORT_TO_PACKAGE = {
    "cv2": "opencv-python-headless",
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "skimage": "scikit-image",
}

# Directories never worth surveying.
_SKIP_DIRS = {".git", "__pycache__", ".github", "node_modules", ".venv", "venv", "assets", "docs"}

MAX_LISTED_FILES = 60
MAX_EXCERPT_CHARS = 4_000


class RepoError(RuntimeError):
    """The repository could not be cloned or inspected."""


@dataclass
class EntrypointInfo:
    """A script that looks like it trains something."""

    path: str
    arguments: list[str] = field(default_factory=list)
    """Command line flags it accepts, e.g. ["--lr", "--resume"]."""

    downloads_data: bool = False
    """Whether it fetches data at runtime, which the sandbox forbids."""


@dataclass
class RepoSurvey:
    """Everything codegen needs to know about a repository."""

    url: str
    local_path: str
    commit: str
    python_files: list[str] = field(default_factory=list)
    entrypoints: list[EntrypointInfo] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    """Lines from requirements.txt or similar, if the repo declares any."""

    imported_packages: list[str] = field(default_factory=list)
    """Third-party packages the code imports. Often the only signal available."""

    readme_excerpt: str = ""

    def primary_entrypoint(self) -> EntrypointInfo | None:
        return self.entrypoints[0] if self.entrypoints else None


def clone(url: str, destination: Path, timeout_s: int = CLONE_TIMEOUT_S) -> str:
    """Shallow-clone a repository and return the commit it landed on.

    Shallow because we only ever read the current state, and history can be
    hundreds of megabytes.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        logger.info("reusing existing clone at %s", destination)
    else:
        logger.info("cloning %s", url)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(destination)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if result.returncode != 0:
            raise RepoError(f"could not clone {url}: {result.stderr.strip()[:300]}")

    return _current_commit(destination)


def _current_commit(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _iter_files(repo_dir: Path) -> list[Path]:
    """Every file worth looking at, skipping vendored and generated directories."""
    files: list[Path] = []
    for path in sorted(repo_dir.rglob("*")):
        if not path.is_file():
            continue
        if _SKIP_DIRS & set(path.relative_to(repo_dir).parts):
            continue
        files.append(path)
    return files


def extract_arguments(source: str) -> list[str]:
    """Command line flags a script declares, via argparse or click.

    Parsed from the AST rather than by regex, so a flag in a comment or a string
    is not mistaken for a real one.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    flags: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in {"add_argument", "option", "argument"}:
            continue
        for arg in node.args:
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("-")
            ):
                flags.append(arg.value)
    return sorted(set(flags))


def _downloads_data(source: str) -> bool:
    """Whether a script fetches data at runtime, which the sandbox blocks."""
    patterns = (
        r"download\s*=\s*True",
        r"urlretrieve|urlopen|requests\.get|wget|curl",
        r"hf_hub_download|from_pretrained|snapshot_download",
        # load_dataset is as often imported directly as called on the module.
        r"\bload_dataset\s*\(",
    )
    return any(re.search(pattern, source) for pattern in patterns)


def _find_entrypoints(repo_dir: Path, files: list[Path]) -> list[EntrypointInfo]:
    """Scripts that look like training entrypoints, best guess first."""
    candidates: list[tuple[int, Path]] = []
    for path in files:
        if path.suffix != ".py":
            continue
        relative = path.relative_to(repo_dir)
        depth = len(relative.parts) - 1
        if path.name in _ENTRYPOINT_NAMES:
            # Prefer the conventional names, and shallower paths.
            candidates.append((_ENTRYPOINT_NAMES.index(path.name) + depth * 10, path))
        elif depth == 0 and "train" in path.stem.lower():
            candidates.append((50 + depth, path))

    entrypoints: list[EntrypointInfo] = []
    for _rank, path in sorted(candidates, key=lambda item: item[0]):
        source = path.read_text(encoding="utf-8", errors="replace")
        entrypoints.append(
            EntrypointInfo(
                path=path.relative_to(repo_dir).as_posix(),
                arguments=extract_arguments(source),
                downloads_data=_downloads_data(source),
            )
        )
    return entrypoints


def _find_configs(repo_dir: Path, files: list[Path]) -> list[str]:
    """Config files, by extension or by living in a config directory."""
    configs: list[str] = []
    for path in files:
        relative = path.relative_to(repo_dir)
        in_config_dir = any(part.lower() in _CONFIG_DIRS for part in relative.parts[:-1])
        if path.suffix.lower() in _CONFIG_SUFFIXES or (in_config_dir and path.suffix == ".py"):
            configs.append(relative.as_posix())
    return configs


def _read_dependencies(repo_dir: Path) -> list[str]:
    """Declared dependencies, as raw lines the model can read."""
    for name in _DEPENDENCY_FILES:
        path = repo_dir / name
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            return [line.strip() for line in text.splitlines() if line.strip()][:40]
    return []


def _is_local_module(name: str, repo_dir: Path) -> bool:
    """Whether an import refers to the repository's own code."""
    return (repo_dir / name).is_dir() or (repo_dir / f"{name}.py").exists()


def infer_imports(repo_dir: Path, files: list[Path]) -> list[str]:
    """Third-party packages the repository actually imports.

    Many research repos ship no requirements file, so the declared dependencies
    are empty while the code needs torch. Without this, the sandbox image would
    lack the one package that matters and every run would fail on the import.
    """
    found: set[str] = set()
    for path in files:
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])

    external = {
        _IMPORT_TO_PACKAGE.get(name, name)
        for name in found
        if name not in _STDLIB_MODULES and not _is_local_module(name, repo_dir)
    }
    return sorted(external)


def _read_readme(repo_dir: Path) -> str:
    for name in ("README.md", "README.rst", "README.txt", "readme.md"):
        path = repo_dir / name
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")[:MAX_EXCERPT_CHARS]
    return ""


def survey(url: str, repo_dir: Path) -> RepoSurvey:
    """Clone the repository if needed and gather what codegen must know."""
    commit = clone(url, repo_dir)
    files = _iter_files(repo_dir)

    python_files = [
        path.relative_to(repo_dir).as_posix() for path in files if path.suffix == ".py"
    ][:MAX_LISTED_FILES]

    result = RepoSurvey(
        url=url,
        local_path=str(repo_dir),
        commit=commit,
        python_files=python_files,
        entrypoints=_find_entrypoints(repo_dir, files),
        config_files=_find_configs(repo_dir, files)[:MAX_LISTED_FILES],
        dependencies=_read_dependencies(repo_dir),
        imported_packages=infer_imports(repo_dir, files),
        readme_excerpt=_read_readme(repo_dir),
    )
    logger.info(
        "surveyed %s at %s: %d python files, %d entrypoints, %d configs",
        url,
        commit[:8],
        len(result.python_files),
        len(result.entrypoints),
        len(result.config_files),
    )
    return result


def read_file(repo_dir: Path, relative_path: str, max_chars: int = MAX_EXCERPT_CHARS) -> str:
    """Read one file from the clone, refusing paths that escape it."""
    target = (repo_dir / relative_path).resolve()
    root = repo_dir.resolve()
    if not target.is_relative_to(root):
        raise RepoError(f"path escapes the repository: {relative_path}")
    if not target.is_file():
        raise RepoError(f"no such file in the repository: {relative_path}")
    return target.read_text(encoding="utf-8", errors="replace")[:max_chars]
