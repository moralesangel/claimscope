"""Checks that a generated script matches the experiment its plan describes.

The report names the plan's dataset. When the script quietly trains on
something else, the report describes an experiment that never ran and a reader
has no way to tell -- worse than a crash, because a crash is visible.

A real run substituted ``sklearn.datasets.load_digits`` for CIFAR-10 and
``make_classification`` for Reuters, reported both under the original names, and
produced a "not consistent at reduced scale" verdict from synthetic noise. The
codegen prompt forbids this, but a prompt is a request; these checks make the
substitution visible in the report even when the model ignores it.

The checks are deliberately advisory. Dataset naming is fuzzy, generated code is
varied, and a false positive that failed the claim would cost more than the
warning it replaced. They flag, they do not block.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claimscope.schemas import ReductionPlan

# Loader -> the datasets it actually provides. A script calling one of these
# while the plan names something else is training on different data.
_LOADERS = {
    "load_digits": ("digits", "sklearn digits", "8x8 digits"),
    "load_iris": ("iris",),
    "load_wine": ("wine",),
    "load_breast_cancer": ("breast cancer", "wdbc"),
    "load_diabetes": ("diabetes",),
    "load_linnerud": ("linnerud",),
    "fetch_20newsgroups": ("20 newsgroups", "newsgroups"),
    "fetch_olivetti_faces": ("olivetti",),
    "fetch_lfw_people": ("lfw", "labeled faces"),
    "fetch_covtype": ("covertype", "covtype"),
    "fetch_california_housing": ("california housing",),
}

# Generators. These produce no real signal at all, so a plan naming any real
# dataset is not being followed.
_SYNTHETIC = (
    "make_classification",
    "make_regression",
    "make_blobs",
    "make_moons",
    "make_circles",
    "make_gaussian_quantiles",
    "make_hastie_10_2",
    "make_multilabel_classification",
)

# Real datasets a plan might name. Matched against the plan text to decide
# whether a substitution has happened; a plan naming none of them is assumed to
# be describing a synthetic task, and generators are then legitimate.
_REAL_DATASETS = (
    "mnist",
    "fashion-mnist",
    "fashion mnist",
    "cifar-10",
    "cifar10",
    "cifar-100",
    "cifar100",
    "imagenet",
    "reuters",
    "imdb",
    "svhn",
    "celeba",
    "penn treebank",
    "wikitext",
    "timit",
    "librispeech",
    "squad",
    "glue",
    "coco",
    "pascal voc",
    "tiny imagenet",
    "stl-10",
    "stl10",
)


def _mentioned_datasets(plan_text: str) -> list[str]:
    """Real datasets the plan names, lowercased."""
    lowered = plan_text.lower()
    return [name for name in _REAL_DATASETS if name in lowered]


def _calls(code: str, name: str) -> bool:
    """Whether the code calls this function, ignoring comments.

    Comments matter: a model that substitutes a dataset tends to narrate the
    decision, and matching its explanation instead of its code would flag
    scripts that do the right thing.
    """
    without_comments = re.sub(r"#[^\n]*", "", code)
    return re.search(rf"\b{re.escape(name)}\s*\(", without_comments) is not None


# How a plan describes each stand-in in prose. The planner is asked to name its
# substitute in reduced_setup, and it writes "sklearn's 8x8 digits dataset" at
# least as often as it writes load_digits, so matching only the function name
# would flag plans that documented themselves correctly.
_PROSE = {
    "load_digits": ("8x8 digit", "8x8 image", "sklearn digit", "scikit-learn digit", "1797"),
    "load_breast_cancer": ("breast cancer",),
    "load_wine": ("wine dataset",),
    "load_iris": ("iris dataset",),
    "load_diabetes": ("diabetes dataset",),
}

_SYNTHETIC_PROSE = ("synthetic", "generated data", "generate synthetic", "artificial data")


def _declares(plan_text: str, name: str) -> bool:
    """Whether the plan itself declares the stand-in this call provides.

    A plan may legitimately say "sklearn's 8x8 digits standing in for MNIST":
    the sandbox has no network, so a documented stand-in is the only way to test
    an MNIST claim at all, and invariant 4 asks for exactly that -- the change
    written down with its justification. What these checks are for is the
    *undocumented* swap, where the plan promises one dataset and the script
    quietly uses another.

    Both spellings count, the function name and the prose, because a warning on
    a plan that did document itself is the fastest way to teach a reader that
    these warnings can be ignored.
    """
    lowered = plan_text.lower()
    if re.search(rf"\b{re.escape(name)}\b", lowered) is not None:
        return True

    # A loader is only excused by prose describing that loader's own data; a
    # generator is excused by the plan saying the task is synthetic at all.
    phrases = _SYNTHETIC_PROSE if name in _SYNTHETIC else _PROSE.get(name, ())
    return any(phrase in lowered for phrase in phrases)


def dataset_substitutions(code: str, plan_text: str, claim_id: str) -> list[str]:
    """Warnings for a script training on data its plan does not name.

    Empty when the plan names no real dataset: the experiment is then synthetic
    by design and a generator is the correct choice. Also empty when the plan
    declares the stand-in it is using, which is a documented reduction rather
    than a misreported one.
    """
    named = _mentioned_datasets(plan_text)
    if not named:
        return []

    expected = ", ".join(named)
    warnings: list[str] = []

    for loader, provides in _LOADERS.items():
        if not _calls(code, loader) or _declares(plan_text, loader):
            continue
        # The loader is fine when it provides one of the datasets the plan names.
        if any(any(p in name for p in provides) for name in named):
            continue
        warnings.append(
            f"{claim_id}: the plan names {expected} but the script trains on "
            f"{loader}() -- the reported dataset is not the one used"
        )

    for generator in _SYNTHETIC:
        if _calls(code, generator) and not _declares(plan_text, generator):
            warnings.append(
                f"{claim_id}: the plan names {expected} but the script generates "
                f"synthetic data with {generator}() -- the result says nothing about {expected}"
            )

    return warnings


def corrupted_labels(code: str, plan_text: str, claim_id: str) -> list[str]:
    """Warnings for label noise injected into a real dataset.

    Real data carries its own irreducible error. Scrambling its labels on top
    buries the effect under noise: a quarter of MNIST's labels randomised took
    the error rate from roughly 5% to 61%, and both arms then scored the same
    because both were fitting nonsense.
    """
    if not _mentioned_datasets(plan_text):
        return []

    without_comments = re.sub(r"#[^\n]*", "", code)
    # Assigning randint into a slice of the labels is what label flipping looks
    # like in practice, whatever the surrounding variable names.
    flips = re.search(
        r"\by[_a-z]*(?:train|tr)?\s*\[[^\]]+\]\s*=\s*(?:np\.)?random\.(?:randint|choice)",
        without_comments,
    )
    if flips is None:
        return []

    return [
        f"{claim_id}: the script randomises a fraction of the training labels of a real "
        f"dataset -- this buries the effect being measured, and both arms score alike"
    ]


def integrity_warnings(code: str, plan_text: str, claim_id: str) -> list[str]:
    """Every advisory check, in one call."""
    return dataset_substitutions(code, plan_text, claim_id) + corrupted_labels(
        code, plan_text, claim_id
    )


def plan_text(plan: ReductionPlan) -> str:
    """The plan's prose, for checking what the script was supposed to use.

    Both codegen and debug check their output against this, so it lives here
    rather than in either node.
    """
    return " ".join([plan.original_setup, plan.reduced_setup, *plan.changes, *plan.preserved])
