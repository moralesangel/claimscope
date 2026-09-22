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

**A real dataset already has irreducible error, so do not add any.** Corrupting real data is the
mistake this section most often causes. Injecting label noise into MNIST pushed the error rate from
about 5% to 61% and buried the dropout effect entirely: both arms were fitting scrambled labels, so
neither could win. On a real dataset, get the headroom from **size** instead -- a small training
subset (a few hundred to a few thousand examples) against a network big enough to memorise it
overfits honestly, which is exactly the regime the claim is about.

The techniques below are for a task you are **generating from scratch**, where there is no real
signal to protect. They are verified to produce a 30-40% train/test accuracy gap in a small MLP:

- **A noisy rule.** Generate `y = f(x) + noise` where the noise is large enough to blur the
  boundary, e.g. `y = (x @ w + 3.0 * randn() > 0)`.
- **Label noise.** Flip 20-40% of the labels of a *synthetic* task to random classes. Never on real
  data -- see above.
- **Very few training samples.** 50-200 examples against a few hundred parameters forces
  memorisation. Keep the test set large (1000+) so the test metric is not itself noisy.
- **Overlapping clusters in low dimensions** (2 to 20 features, not 784), with noise comparable to
  the distance between centres.

Whichever you use, sanity-check the scale: if the metric comes out near chance (for 10 classes, an
error near 0.9, or anything above roughly 0.5), the task is too hard and the comparison is dead.
Both arms failing is not a measurement.

## Use the dataset the plan names, or fail

The report states the plan's dataset by name. If the script quietly trains on something else, the
report describes an experiment that never happened, and a reader has no way to tell. This is worse
than a crash: a crash is visible.

**Substituting a different dataset is not allowed**, however reasonable it looks. A run on
`load_digits` is not a run on CIFAR-10, and `make_classification` is not Reuters. Do not reach for a
stand-in because the named dataset needs the network, is large, or is not installed. There is no
phrasing of a comment that makes a substitution acceptable.

If a real dataset ships inside an installed package, subsample it small -- that is the good case.
If the plan's dataset is genuinely unavailable, **let the script fail**: raise, or let the
ImportError propagate. The debug node sees the error and can fix it, and a failed claim is reported
honestly as a failure. A confident number computed from the wrong data is not.

Likewise **never wrap the dataset load in a try/except that falls back to random data**, and never
pad, reshape or relabel one dataset to impersonate another.

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
