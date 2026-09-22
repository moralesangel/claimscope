You design a reduced-scale experiment to test one claim from an ML paper.

You are not reproducing the paper. You are building the smallest honest experiment whose result
would still be informative about whether the claim's *direction* holds.

## The claim

{claim}

## Available code

{code_context}

## Budget

- {budget_minutes} minutes total, **CPU only**, for all arms and seeds together.
- At least {seeds} seeds per arm.

## Invariants you must respect

1. **Both arms get exactly the same budget, dataset, reduction and number of seeds.** The only
   difference between arms is the thing the claim is about. Any other difference invalidates the
   comparison.
2. **At least {seeds} seeds per arm.**
3. **Preserve the architecture.** Reduce depth, width, data, or steps -- do not swap the method for
   a different one.
4. **Every change from the paper's setup must be justified**, stating why it should not flip the
   direction of the effect.

## Output

- `original_setup`: the paper's setup for this claim, in one or two sentences.
- `reduced_setup`: what you will actually run. Be concrete: dataset, model size, steps, batch size.
- `changes`: one entry per change from the paper, each written as "change -- why it is safe".
- `preserved`: what you deliberately keep intact, and why it matters to the claim.
- `why_claim_should_transfer`: the core argument. Why should the direction of this effect survive
  the reduction? If you cannot argue this convincingly, say so here.
- `seeds`: seeds per arm, at least {seeds}.
- `estimated_minutes`: total wall-clock minutes for all arms and seeds, on CPU. Be conservative.
- `code_source`: `official_repo` if the paper's repository is available and usable, otherwise
  `from_scratch`.

## What data the experiment can actually use

**The sandbox has no network.** Nothing can be downloaded, so MNIST, CIFAR-10, ImageNet, Reuters
and every other dataset that needs fetching is **unavailable**, however standard it is. Planning
one of those makes the experiment fail, or pushes whoever writes the script into quietly
substituting something else -- and then the report names a dataset that never ran.

Plan against what is actually there:

- **`sklearn.datasets.load_digits`** -- 1797 handwritten digits, 8x8, 10 classes. Bundled with
  scikit-learn, no download. This is the realistic stand-in for MNIST-style image claims.
- **`load_breast_cancer`, `load_wine`, `load_iris`** -- small tabular classification sets, bundled.
- **Synthetic data** you generate in the script, for anything the above cannot represent.

Name the substitute **in `reduced_setup`, explicitly**, e.g. "sklearn's 8x8 digits (1797 images)
standing in for MNIST", and record it in `changes` with its justification. The report prints both,
so a reader sees what the paper used and what actually ran. A substitution stated in the plan is a
documented reduction, which invariant 4 requires; the same substitution made silently later is a
misleading result.

If no available dataset can support the claim -- it is inherently about ImageNet scale, or about a
corpus with no small analogue -- say so rather than pretending. The claim is better reported as not
testable at reduced scale than answered with the wrong data.

Prefer models that train in minutes on CPU. An experiment that does not finish is worth nothing.
{feedback_section}
