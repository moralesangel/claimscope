You write a single self-contained Python script that runs one reduced-scale experiment.

## The claim being tested

{claim}

## The approved reduction plan

{plan}

## Required interface

Write **one file, `run.py`**, that:

1. Accepts exactly these arguments: `--arm <name>` and `--seed <int>`.
2. Accepts `--steps <int>`, used to cut the run short for timing calibration. When it is absent,
   run the full experiment from the plan.
3. Seeds every source of randomness from `--seed` so a given seed is reproducible.
4. Runs **one arm only**, selected by `--arm`. The valid arm names are exactly: {arms}
5. Writes its result to `result.json` in the working directory, with this shape:

```json
{{"arm": "<arm>", "seed": <seed>, "metric": <float>, "metric_name": "<name>"}}
```

6. Exits with status 0 on success and non-zero on failure.

## The task must be hard enough to measure the effect

This is where these experiments usually fail. If both arms score the same on every seed, the study
measured nothing and the whole run is wasted. Design the task so the metric has room to move
**before** worrying about anything else.

**The task needs irreducible error.** A problem a small network can solve perfectly leaves no gap
for regularisation, optimisation or architecture to affect. Check your design against this: could a
reasonable model reach 100% on the test set? If so, the experiment cannot detect anything.

**Well-separated Gaussian clusters in high dimensions are the classic trap.** Random class centres
in hundreds of dimensions are always linearly separable -- the noise concentrates on a shell and the
classes never actually overlap. Scaling the centres down does not fix it. Avoid this shape entirely.

Tasks that do work, all verified to produce a 30-40% train/test accuracy gap in a small MLP:

- **Label noise.** Flip 20-40% of the training labels to random classes. This guarantees
  irreducible error at any dimension, and is the most reliable option for anything about
  overfitting or regularisation.
- **A noisy rule.** Generate `y = f(x) + noise` where the noise is large enough to blur the
  boundary, e.g. `y = (x @ w + 3.0 * randn() > 0)`.
- **Very few training samples.** 50-200 examples against a few hundred parameters forces
  memorisation. Keep the test set large (1000+) so the test metric is not itself noisy.
- **Overlapping clusters in low dimensions** (2 to 20 features, not 784), with noise comparable to
  the distance between centres.

If a real dataset ships inside an installed package, prefer it and subsample it small. Synthetic
data is the fallback, not the first choice.

## Other constraints

- **No network access.** The sandbox has none, so nothing can be downloaded. Never call
  `download=True`, `urlretrieve`, `from_pretrained` or similar.
- **Only these packages are available:** {packages}. Anything else must be implemented by hand.
- **Both arms must share every code path** except the one thing the claim is about. Write the
  difference as a single conditional on `--arm`, so nothing else can diverge. Both arms must see
  identical data, identical initialisation for a given seed, and identical training steps.
- **The metric must vary across seeds.** If it can only take a handful of values -- an error count
  on a tiny test set, say -- the arms will tie. Prefer a continuous metric (a loss, an accuracy on a
  large test set) over a count.
- **Keep it fast.** The whole study is {total_runs} runs within {budget_minutes} minutes on CPU, so
  one run must finish in roughly {per_run_seconds:.0f} seconds. Prefer small models and few epochs.
- The metric must be a single float where the claim's expected direction is meaningful.
- **Print the metric to stdout** as well as writing it, so a failed run is diagnosable from the log.

## Output

Return the complete contents of `run.py` in the `code` field, and a one-line `summary` of what the
script does. Do not wrap the code in markdown fences. The script must run as written.
{feedback_section}
