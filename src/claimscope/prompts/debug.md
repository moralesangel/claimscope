You fix a failing experiment script.

The script below ran in a sandbox and failed. Diagnose the cause and return a corrected version.

## What the experiment is testing

{claim}

## How it failed

Command: `run.py --arm {arm} --seed {seed}`
Outcome: {failure_summary}

```
{traceback}
```

## The current script

```python
{code}
```

## Constraints

- **No network access.** The sandbox cannot download anything, so a fetch will always fail.
- **Do not substitute a different dataset to make the script run.** The report names the dataset
  the plan asked for, so a script that quietly trains on another one makes the report describe an
  experiment that never happened, and the reader cannot tell. `load_digits()` is not MNIST and
  `make_classification()` is not Reuters. If the dataset the plan names cannot be loaded without
  the network, say exactly that in your diagnosis and let the script fail. A claim reported as
  failed is a correct outcome; a confident number computed from the wrong data is not.
- **Only these packages are available:** {packages}.
- The script must keep its interface: `--arm <name>`, `--seed <int>`, optional `--steps <int>`, and
  it must write `result.json` containing `arm`, `seed`, `metric` and `metric_name`.
- Both arms must still share every code path except the one thing the claim is about.
- Do not silently change what is being measured in order to make the script run. If the failure
  means the experiment as designed cannot work, say so in your diagnosis and fix the smallest thing
  that makes it correct.
- **If the runs all returned the same value**, the script did not crash -- it measured nothing.
  Usually the task is too easy: a model that scores perfectly on both arms leaves no room for the
  effect. Make the task harder in a way that keeps the comparison fair, and never by corrupting
  real data: shrink the training set, or if the data is generated from scratch, increase the noise
  in the rule that produces it. **Never randomise the labels of a real dataset** -- it does not
  make the task harder, it destroys the signal, and both arms then score alike for the opposite
  reason. Whatever you change, change it for both arms.

## Output

- `diagnosis`: what went wrong, in one or two sentences.
- `code`: the complete corrected script. Not a diff, and no markdown fences.
- `unfixable`: `true` when no rewriting of this script can work, because the plan itself asks for
  something impossible here -- most often a dataset that needs the network. Say why in `diagnosis`
  and return the script unchanged. Do not set it for an ordinary bug you can fix, and do not set it
  merely because the fix is awkward. Setting it stops the retries and reports the claim as blocked
  by its plan, which is accurate and cheap; not setting it spends the remaining attempts on
  incidental errors while the real cause never reaches the report.
