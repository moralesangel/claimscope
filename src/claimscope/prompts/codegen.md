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

## Constraints

- **No network access.** The sandbox has none. Generate data synthetically, or use a dataset that
  ships inside an installed package. Never download anything.
- **Only these packages are available:** {packages}. Anything else must be implemented by hand.
- **Both arms must share every code path** except the one thing the claim is about. Write the
  difference as a single conditional on `--arm`, so nothing else can diverge.
- **Keep it fast.** The whole study is {total_runs} runs within {budget_minutes} minutes on CPU, so
  one run must finish in roughly {per_run_seconds:.0f} seconds. Prefer small models and few epochs.
- The metric must be a single float where the claim's expected direction is meaningful.

## Output

Return the complete contents of `run.py` in the `code` field, and a one-line `summary` of what the
script does. Do not wrap the code in markdown fences. The script must run as written.
{feedback_section}
