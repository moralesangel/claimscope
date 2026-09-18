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

- **No network access.** If the script tries to download anything, replace that with synthetic data
  or a dataset bundled in an installed package.
- **Only these packages are available:** {packages}.
- The script must keep its interface: `--arm <name>`, `--seed <int>`, optional `--steps <int>`, and
  it must write `result.json` containing `arm`, `seed`, `metric` and `metric_name`.
- Both arms must still share every code path except the one thing the claim is about.
- Do not silently change what is being measured in order to make the script run. If the failure
  means the experiment as designed cannot work, say so in your diagnosis and fix the smallest thing
  that makes it correct.

## Output

- `diagnosis`: what went wrong, in one or two sentences.
- `code`: the complete corrected script. Not a diff, and no markdown fences.
