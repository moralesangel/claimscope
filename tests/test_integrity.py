"""Catching a script that does not run the experiment its plan describes.

A real run reported a CIFAR-10 result computed from sklearn's 8x8 digits and a
Reuters result computed from make_classification, and reached a verdict on both.
Nothing in the pipeline noticed. These checks make that visible in the report.
"""

from __future__ import annotations

import pytest

from claimscope.integrity import (
    corrupted_labels,
    dataset_substitutions,
    integrity_warnings,
)

MNIST_PLAN = "Subsample MNIST to 5000 training images and train a small MLP."
SYNTHETIC_PLAN = "Generate a synthetic binary classification task with overlapping clusters."


class TestDatasetSubstitution:
    def test_a_stand_in_loader_is_flagged(self) -> None:
        code = "from sklearn.datasets import load_digits\nX, y = load_digits(return_X_y=True)\n"

        warnings = dataset_substitutions(code, MNIST_PLAN, "c1")

        assert len(warnings) == 1
        assert "load_digits" in warnings[0]
        assert "mnist" in warnings[0]

    def test_synthetic_data_under_a_real_plan_is_flagged(self) -> None:
        code = "from sklearn.datasets import make_classification\nX, y = make_classification()\n"

        warnings = dataset_substitutions(code, MNIST_PLAN, "c1")

        assert len(warnings) == 1
        assert "make_classification" in warnings[0]

    def test_a_synthetic_plan_may_generate_data(self) -> None:
        """Nothing is being impersonated when the plan asks for synthetic data."""
        code = "from sklearn.datasets import make_classification\nX, y = make_classification()\n"

        assert dataset_substitutions(code, SYNTHETIC_PLAN, "c1") == []

    def test_the_named_dataset_is_not_flagged(self) -> None:
        code = "from sklearn.datasets import load_digits\nX, y = load_digits(return_X_y=True)\n"
        plan = "Subsample the sklearn digits dataset to 500 images."

        assert dataset_substitutions(code, plan, "c1") == []

    def test_a_comment_mentioning_a_loader_is_not_a_call(self) -> None:
        """Otherwise a script that considered and rejected a stand-in is flagged."""
        code = "# We could use load_digits() here but the plan says MNIST\nX = load_mnist()\n"

        assert dataset_substitutions(code, MNIST_PLAN, "c1") == []

    def test_every_substitution_is_reported(self) -> None:
        code = (
            "from sklearn.datasets import load_digits, make_blobs\n"
            "X, y = load_digits(return_X_y=True)\n"
            "Xb, yb = make_blobs()\n"
        )

        assert len(dataset_substitutions(code, MNIST_PLAN, "c1")) == 2


class TestCorruptedLabels:
    @pytest.mark.parametrize(
        "line",
        [
            "y_train[noise_idx] = np.random.randint(0, 10, size=n)",
            "y_train[flip_mask] = np.random.choice(10, size=k)",
            "y[idx] = random.randint(0, 9)",
        ],
    )
    def test_label_flipping_on_real_data_is_flagged(self, line: str) -> None:
        code = f"X, y_train = load_mnist()\n{line}\n"

        warnings = corrupted_labels(code, MNIST_PLAN, "c1")

        assert len(warnings) == 1
        assert "randomises" in warnings[0]

    def test_label_flipping_on_a_synthetic_task_is_allowed(self) -> None:
        """There is no real signal to bury, and the prompt recommends it there."""
        code = "y_train[idx] = np.random.randint(0, 2, size=n)\n"

        assert corrupted_labels(code, SYNTHETIC_PLAN, "c1") == []

    def test_ordinary_label_use_is_not_flagged(self) -> None:
        code = "y_train = y[:1000]\npreds = model.predict(X_test)\nacc = (preds == y_test).mean()\n"

        assert corrupted_labels(code, MNIST_PLAN, "c1") == []

    def test_shuffling_is_not_corruption(self) -> None:
        """A permutation preserves every label; only reassignment destroys them."""
        code = "perm = np.random.permutation(len(y))\ny_train = y[perm]\n"

        assert corrupted_labels(code, MNIST_PLAN, "c1") == []


class TestCombined:
    def test_a_clean_script_produces_no_warnings(self) -> None:
        code = (
            "import numpy as np\n"
            "X, y = load_mnist_from_disk()\n"
            "X_train, y_train = X[:5000], y[:5000]\n"
        )

        assert integrity_warnings(code, MNIST_PLAN, "c1") == []

    def test_both_kinds_are_reported_together(self) -> None:
        code = (
            "from sklearn.datasets import load_digits\n"
            "X, y_train = load_digits(return_X_y=True)\n"
            "y_train[idx] = np.random.randint(0, 10, size=n)\n"
        )

        warnings = integrity_warnings(code, MNIST_PLAN, "c1")

        assert len(warnings) == 2

    def test_the_claim_id_is_named(self) -> None:
        """The report lists warnings for every claim, so each must say which."""
        code = "from sklearn.datasets import make_blobs\nX, y = make_blobs()\n"

        assert all(
            w.startswith("cifar_claim:")
            for w in integrity_warnings(code, MNIST_PLAN, "cifar_claim")
        )


class TestDeclaredSubstitution:
    """A stand-in the plan names is a documented reduction, not a misreport.

    The sandbox has no network, so a documented stand-in is the only way to test
    an MNIST claim at all. Warning about it would fire on every correct plan and
    train the reader to ignore the warnings that matter.
    """

    def test_a_loader_the_plan_names_is_allowed(self) -> None:
        code = "from sklearn.datasets import load_digits\nX, y = load_digits()\n"
        plan = (
            "Use sklearn load_digits (1797 8x8 images) standing in for MNIST, "
            "which the sandbox cannot download."
        )

        assert dataset_substitutions(code, plan, "c1") == []

    def test_a_generator_the_plan_names_is_allowed(self) -> None:
        code = "X, y = make_classification(n_samples=2000)\n"
        plan = "Reuters is unavailable offline; use make_classification as a stand-in corpus."

        assert dataset_substitutions(code, plan, "c1") == []

    def test_an_undeclared_stand_in_is_still_flagged(self) -> None:
        """The plan must name the substitute, not merely admit a problem."""
        code = "from sklearn.datasets import load_digits\nX, y = load_digits()\n"
        plan = "Subsample MNIST to 5000 training images."

        assert len(dataset_substitutions(code, plan, "c1")) == 1


class TestProseDeclaration:
    """The planner writes prose, not function names.

    It says "sklearn's 8x8 digits standing in for MNIST" at least as often as it
    says load_digits. Matching only the function name flagged plans that had
    documented themselves correctly -- and a warning on a correct plan teaches
    the reader to ignore all of them.
    """

    CODE = "from sklearn.datasets import load_digits\nX, y = load_digits()\n"

    def test_a_prose_declaration_is_enough(self) -> None:
        plan = (
            "Train a small MLP on sklearn's 8x8 digits dataset (1797 images) "
            "standing in for MNIST, which the sandbox cannot download."
        )

        assert dataset_substitutions(self.CODE, plan, "c1") == []

    def test_the_sample_count_also_declares_it(self) -> None:
        plan = "Use the bundled 1797-image digit set in place of MNIST."

        assert dataset_substitutions(self.CODE, plan, "c1") == []

    def test_prose_about_a_different_dataset_does_not_excuse_it(self) -> None:
        """Naming breast cancer does not license quietly using digits."""
        plan = "Subsample MNIST to 5000 images; the breast cancer set is unsuitable here."

        assert len(dataset_substitutions(self.CODE, plan, "c1")) == 1

    def test_calling_the_task_synthetic_does_not_excuse_a_real_loader(self) -> None:
        """Synthetic prose excuses a generator, never a stand-in dataset."""
        plan = "Subsample MNIST; add synthetic perturbations to the inputs."

        assert len(dataset_substitutions(self.CODE, plan, "c1")) == 1

    def test_synthetic_prose_excuses_a_generator(self) -> None:
        code = "X, y = make_classification(n_samples=2000)\n"
        plan = "TIMIT is unavailable offline; use synthetic sequence data standing in for it."

        assert dataset_substitutions(code, plan, "c1") == []
