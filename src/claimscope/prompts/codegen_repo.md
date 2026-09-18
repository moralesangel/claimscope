You adapt a paper's official code to run one reduced-scale experiment.

The repository has been cloned into the sandbox workspace under `repo/`. Your job is to write a
`run.py` that sits **next to** it and drives it, without rewriting the authors' method.

## The claim being tested

{claim}

## The approved reduction plan

{plan}

## What the repository looks like

Cloned from {repo_url} at commit `{repo_commit}`.

**Entrypoints found**

{entrypoints}

**Config files**

{config_files}

**Declared dependencies**

{dependencies}

**Packages the code imports** (detected from the source, and installed in the sandbox)

{imported_packages}

**Python files**

{python_files}

**README excerpt**

{readme_excerpt}

## Required interface

Write **one file, `run.py`**, placed beside `repo/`, that:

1. Accepts `--arm <name>`, `--seed <int>`, and an optional `--steps <int>` that cuts the run short
   for timing calibration. Valid arm names are exactly: {arms}
2. Seeds every source of randomness from `--seed`.
3. Runs **one arm only**, selected by `--arm`.
4. Writes `result.json` in its own working directory (not inside `repo/`), containing:

```json
{{"arm": "<arm>", "seed": <seed>, "metric": <float>, "metric_name": "<name>"}}
```

5. Exits 0 on success, non-zero on failure.

## How to drive the repository

Prefer, in this order:

1. **Import the repo's modules** and call them, if it is structured as a library. Most controllable.
2. **Run its entrypoint as a subprocess** with flags, if it exposes the knobs you need, then parse
   its output or the checkpoint it writes.
3. **Write a config file** the entrypoint reads, if that is how it is configured.

Only patch the repository's own files if there is no other way, and if you do, say so clearly in
your summary.

## Constraints that will break the run if ignored

- **There is no network.** The survey above says whether the entrypoint downloads data; if it does,
  you must prevent that. Point the repo at a local directory, generate synthetic data in the
  expected format, or use whatever the repo bundles. Never let it call out.
- **These packages are installed:** {packages}. They cover what the repository imports, but not
  necessarily every optional extra the paper used. If you need something outside this list, avoid
  that code path or implement the piece yourself.
- **CPU only.** Anything that requires CUDA must be switched to CPU.
- **Reduce the scale as the plan says.** The repo's defaults are the paper's full-scale settings and
  will not finish. Override epochs, dataset size, model width and depth to match the reduced setup.
- **Both arms must differ only in the thing the claim is about.** Every other setting -- data,
  seeds, steps, batch size -- must be identical. Write the difference as a single conditional on
  `--arm`.
- One run must finish in roughly {per_run_seconds:.0f} seconds on CPU.

## Output

- `code`: the complete contents of `run.py`, with no markdown fences.
- `summary`: one line saying how you drove the repository, and noting any file you patched.
{feedback_section}
