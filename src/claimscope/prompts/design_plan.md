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

Plan against what is actually there. These load without a network:

{offline_datasets}

`load_digits` is the realistic stand-in for MNIST-style image claims. For anything these cannot
represent, generate synthetic data in the script. Everything else in `sklearn.datasets` is a
`fetch_*` that downloads, and every famous benchmark -- MNIST, CIFAR-10, RCV1, 20 Newsgroups --
is one of those, so none of them can be used here.

**The only packages installed are: {packages}.** There is no PyTorch, TensorFlow, JAX or Keras,
and nothing can be installed, so a plan that assumes a deep learning framework fails on the first
import. Anything beyond what these provide has to be written by hand in numpy -- which is fine for
a small MLP or a small convnet, and is a reason to keep the architecture modest.

Name the substitute **in `reduced_setup`, explicitly**, e.g. "sklearn's 8x8 digits (1797 images)
standing in for MNIST", and record it in `changes` with its justification. The report prints both,
so a reader sees what the paper used and what actually ran. A substitution stated in the plan is a
documented reduction, which invariant 4 requires; the same substitution made silently later is a
misleading result.

If no available dataset can support the claim -- it is inherently about ImageNet scale, or about a
corpus with no small analogue -- say so rather than pretending. The claim is better reported as not
testable at reduced scale than answered with the wrong data.

## The reduced setup must be a regime where the effect can appear

Shrinking an experiment can remove the very conditions the claim depends on, and then the result is
inconclusive no matter how carefully it is run. Before settling on `reduced_setup`, ask what has to
be true for the claimed effect to exist at all, and make sure your reduction keeps it.

The common case: **a regulariser only helps a model that overfits.** Dropout, weight decay, data
augmentation and early stopping all reduce the gap between training and test error, so a small
network that never memorises its training set leaves them nothing to do. A 2-layer MLP on 1797
digits for 10 epochs reaches about 5% test error without overfitting, and dropout then changes
nothing -- a real experiment that cannot answer the question. Push the setup into the regime the
claim is about: fewer training examples, a wider or deeper network, more epochs, no early stopping.
The signature to aim for is a visible train/test gap in the baseline arm.

The same reasoning applies elsewhere. An optimiser claim needs a problem hard enough that
optimisation matters; a scaling claim needs at least two points far enough apart to separate; an
architecture claim needs a task the simpler architecture cannot already solve.

State in `why_claim_should_transfer` what makes the reduced setup a regime where the effect can
appear -- not only that the mechanism is scale free, but why this particular size, depth and
training length still leave room for it to show.

Prefer models that train in minutes on CPU. An experiment that does not finish is worth nothing.
{feedback_section}
