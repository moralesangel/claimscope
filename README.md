# ClaimScope

[![CI](https://github.com/moralesangel/claimscope/actions/workflows/ci.yml/badge.svg)](https://github.com/moralesangel/claimscope/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

ClaimScope is an agent that takes a machine learning paper from arXiv and answers one narrow but
useful question: **does this claim still hold when the experiment is shrunk?**

It does not reproduce papers. It extracts the paper's claims, discards the ones there is no sense in
testing at reduced scale, designs a small version of the experiment, runs it in an isolated sandbox
and issues a verdict with a confidence interval.

**A negative result at reduced scale does not refute the paper.** An effect that disappears when the
model, the data or the compute shrinks may still be real at full scale, and that is exactly what
this method cannot tell apart. Every report says so explicitly.

## How it works

```
  ingest ──► extract_claims ──► triage ──► design_plan ──► review (interrupt)
    │              │              │             ▲             │
    │              │              │             └── rejected ─┤
    │              │              │                           │ approved
  arXiv +        structured   nothing testable                ▼
  PyMuPDF          claims           │                      codegen
                                    │                         │
                                    │        ┌── failure ─► execute (sandbox)
                                    │        │                │
                                    │      debug ◄────────────┤
                                    │     (max N)             │ ok
                                    │                         ▼
                                    └──────────────────►  analyze ──► report
                                                          bootstrap    report.md
                                                           + Welch
```

1. **ingest** downloads the PDF from arXiv, extracts the text and finds the official repository if
   there is one.
2. **extract_claims** turns the text into structured claims, each with its type, its arms, its
   metric and the direction the paper predicts.
3. **triage** decides which are worth testing. Absolute claims never are; neither are the ones that
   depend on scale.
4. **design_plan** designs the reduced experiment, documenting every change from the paper and why
   it should not invert the effect.
5. **review** stops and shows you the plan. You approve it, or reject it with feedback that goes
   back to the designer. You can close the process here and pick it up later.
6. **codegen** writes the experiment: adapting the official repository where there is one, or
   writing it from scratch where there is not.
7. **execute** measures with a short run first and extrapolates; if it will not fit the budget, the
   claim goes back to design rather than starting something that cannot finish.
8. **debug** fixes code that fails, up to a capped number of attempts.
9. **analyze** compares the arms and **report** writes the report.

## Installation

Needs Python 3.11+, [uv](https://docs.astral.sh/uv/), and Docker to run experiments.

```bash
uv sync --all-extras
cp .env.example .env
```

ClaimScope works with Anthropic, Google Gemini, or a local model through Ollama. In `.env`, set
`CLAIMSCOPE_PROVIDER=anthropic|google|ollama` and fill in the matching key (ollama needs none).

**On Gemini's free tier:** it is 20 requests a day **per model**, not per account. When one model is
exhausted ClaimScope moves to the next in the chain on its own, so in practice you get several times
that. `doctor` tells you which one is answering right now.

**On the local model:** it is useful for checking the pipeline runs without spending quota, not for
judging the agent's quality. Tested with Qwen3 4B on a laptop with no GPU: minutes per call, it
leaves comparison arms unfilled, and its triage lets through claims that are not testable. For
results worth trusting, use a hosted model.

**On Windows, export `PYTHONIOENCODING=utf-8` before running.** The console is cp1252, and a run
dies with `UnicodeEncodeError` the moment the model writes an arrow or a dash into a plan summary —
and it dies *before* the report is saved, so the whole batch is lost.

## Usage

```bash
# Check everything is ready before spending quota.
uv run python -m claimscope.cli doctor

# Analyse a paper.
uv run python -m claimscope.cli analyze 1706.03762

# Resume a run that stopped waiting for your approval.
uv run python -m claimscope.cli threads
uv run python -m claimscope.cli resume 1706.03762-ce8eced0

# Show the resolved configuration, with keys masked.
uv run python -m claimscope.cli config
```

Everything lands in `runs/<paper_id>/`: the PDF, the text, `claims.json`, `triage.json`,
`plans.json`, each experiment's workspace, and `report.md`. The cache is reused across runs.

On machines with Smart App Control, the virtualenv's `.exe` files are blocked, which is why the
commands are invoked as `python -m`.

## Un resultado real

[`docs/example-report.md`](docs/example-report.md) is the report from a real run on the dropout
paper (arXiv 1207.0580), with `gemini-3.5-flash` on a laptop with no GPU. In summary:

Of 8 claims extracted, triage discarded 7 with concrete reasons — TIMIT for being a proprietary
dataset, ImageNet and the Boltzmann machines for compute cost. The remaining one was reduced to an
MLP over `sklearn.datasets.load_digits` and run with 5 seeds per arm:

| Arm | Test error | Std. dev. |
|---|---|---|
| Without dropout | 18.7% | 0.014 |
| With 50% dropout + L2 | **11.9%** | 0.010 |

**Verdict: `consistent_at_reduced_scale`.** Effect +0.068, 95% CI [+0.055, +0.082], Welch
p=4.4e-05. The interval sits entirely in the direction the paper predicts.

What that means: the *direction* of dropout's effect survives the shrinking. Not that the paper's
160→130 errors were reproduced.

### Twelve papers

A batch over twelve well-known papers (VGG, Adam, BatchNorm, DenseNet, MAML, Cutout, mixup, Lottery
Ticket, NTK, RoBERTa, ViT and dropout), with `claude-sonnet-5` and the subprocess sandbox. 120
claims extracted in total:

| Verdict | Count |
|---|---|
| Not testable at reduced scale | 96 |
| Inconclusive | 13 |
| Consistent at reduced scale | 7 |
| Not consistent at reduced scale | 4 |

**That 80% comes back "not testable" is the result, not a shortfall.** Most claims in an ML paper
rest on ImageNet, on pretraining, or on architectures that do not fit a CPU budget. A system that
returned a verdict for all of them would be inventing them.

The cleanest effect came from BatchNorm: `bn_enables_sigmoid_training`, +0.84 with 95% CI
[+0.823, +0.854]. A sigmoid network that simply does not train without batch normalisation.

All twelve reports are in [`docs/reports/`](docs/reports/), each with the claims extracted, the
reason for every triage rejection, the reduction plan, and the per-seed data.

## How to read a verdict

Both arms run with the same budget, the same dataset and the same seeds. The effect is the
difference of means, signed in the direction the paper predicts, and the confidence interval comes
from a bootstrap over the seeds. The Welch test is reported as support, not as the criterion.

| Verdict | What it means |
|---|---|
| `consistent_at_reduced_scale` | The interval sits entirely in the predicted direction. |
| `not_consistent_at_reduced_scale` | The interval sits entirely in the opposite direction. |
| `inconclusive` | The interval crosses zero, or the experiment could not be completed. |
| `not_testable` | Triage discarded it, with the reason recorded. |

With three seeds the statistical power is low: `inconclusive` almost always means "not enough
evidence", not "no effect".

## Probarlo sin instalar nada

[`notebooks/claimscope_colab.ipynb`](notebooks/claimscope_colab.ipynb) runs the whole pipeline on
Google Colab: it downloads a paper, extracts its claims, designs the experiment, **actually runs
it**, and issues a verdict. All you need is a Gemini key, which has a free tier.

Colab has no Docker, so the notebook uses the weaker sandbox (see below). It starts by checking the
network block works before spending any model calls.

## Security

Model-generated code **never runs outside a sandbox**. There are two, and the difference matters:

**`docker` (the default).** A container with no network, a non-root user, a read-only root, all
capabilities dropped, `no-new-privileges`, and caps on CPU, memory, processes and time. The only
thing mounted is that claim's workspace. Dependencies are installed when the image is built, the
one step with network access. **It is the only backend that really contains what it runs.**

**`subprocess`** (`CLAIMSCOPE_SANDBOX_BACKEND=subprocess`). For environments without Docker, such as
Colab. It runs in a separate process with network calls blocked, memory and CPU capped, and the
environment scrubbed of credentials. **It blocks accidental downloads but does not contain hostile
code**: whatever runs in that process can undo the restrictions from inside. It has to be asked for
explicitly, and every report produced this way says so.

## Evaluating the agent

`eval/` measures the agent's own quality against 13 hand-annotated papers in `eval/annotations/`
(90 claims).

```bash
uv run python -m eval.run_eval --output runs/eval.json
uv run python -m eval.run_eval --paper 1512.03385
```

It measures extraction precision and recall, classification accuracy, verdict agreement and cost.
The corpus includes deliberately hard cases: RoBERTa, where **no** claim is testable because they
all depend on pretraining scale; Vision Transformer, whose central finding only exists at large
scale; and Lottery Ticket and MAML, whose later replications were troubled.

The tests in `tests/test_eval_metrics.py` keep the corpus from drifting out of balance: all four
claim types present, all three verdicts represented, and at least one paper that is entirely
untestable. Without claims it is expected to contradict, the harness would give a perfect score to
an agent that always says yes.

## Tracing

Optional and off by default. `CLAIMSCOPE_TRACING_ENABLED=true` with
`CLAIMSCOPE_TRACING_BACKEND=langsmith` (included) or `langfuse` (`uv sync --extra tracing`,
self-hostable). If the backend is misconfigured, ClaimScope warns and carries on without tracing: it
never takes a run down.

## Limitations

Worth being explicit about what this cannot do.

- **It is not a reproduction.** A positive verdict says the direction of the effect survived the
  shrinking, not that the paper's numbers were reproduced.
- **A negative refutes nothing.** This is the central limitation, and it is irreducible.
- **Low statistical power.** Three seeds per arm detect large effects and little else.
- **Scale-dependent claims are out of scope** by construction: language model pretraining, emergent
  capabilities, scaling laws.
- **The reduced experiment is written by a model** from the paper's description, so it may differ
  from the authors' implementation in ways that matter.
- **It tests claims, not papers.** A paper with five claims may have two that hold at reduced scale
  and three that are not testable.
- **Some claim shapes the analysis cannot settle**, and it says so rather than trying: those
  comparing three or more arms (`adam ≈ sgd_nesterov > adagrad`) and equivalence claims
  (`LRN does not improve`). The comparison is two-arm and one-sided, so forcing them would give a
  verdict on a different question from the one the claim asks. In this corpus that is 22 and 15
  claims respectively. Testing an equivalence properly needs a test against a pre-stated margin,
  which is not implemented.

## Against related work

**PaperBench** (OpenAI, 2025) measures whether an agent can reproduce a whole ICML paper from
scratch, against rubrics written by the authors themselves. That is a far more ambitious and far
more expensive question: every attempt means replicating the entire body of work. It evaluates
fidelity to the reproduction.

**CORE-Bench** (Siegel et al., 2024) measures computational reproducibility: given the authors' code
and data, can the agent run the repository and obtain the published results? The focus is the
environment and the dependencies, not whether the scientific finding is sound.

**ClaimScope asks something else.** Not "can you reproduce this?" but "does this particular claim
survive shrinking the experiment?". The differences that matter:

- The unit is the **claim**, not the paper. Triaging them is part of the work, and deciding
  something is not testable is a legitimate answer rather than a failure.
- The budget is **deliberately small** — CPU minutes, not GPU days. That rules out whole classes of
  claim, and the agent has to recognise it instead of trying anyway.
- The output is **statistical and calibrated**: an effect with a confidence interval, and an honest
  `inconclusive` when there is not enough evidence.
- It does not need the authors' code. It uses it where it exists, and writes the experiment from the
  paper's description where it does not.

Neither replaces the other. PaperBench and CORE-Bench measure whether a paper's machinery can be
made to run again; ClaimScope asks whether one specific claim holds up when put through a cheaper
experiment.

## Project status

The pipeline is complete and **validated end to end with a real model**: see
[`docs/example-report.md`](docs/example-report.md). `tests/test_end_to_end.py` guards that path on
every run of the suite.

One thing remains unverified, and it is worth knowing before trusting a result:

- **The Docker sandbox has never run on the development machine**, which does not have it installed.
  Its containment properties are tested against a fake client, and
  `tests/test_sandbox_integration.py` holds the real checks, which run in CI. **Every report in
  `runs/` was produced with the subprocess sandbox**, which blocks the network and caps resources
  but does not contain hostile code. Each report says so in its header. Treat them as a
  demonstration of the pipeline, not as measurements to lean on.

With Docker available: `uv run python -m pytest -m docker`, then a real `analyze`.

The implementation plan is in [PLAN.md](PLAN.md) and the working rules in [CLAUDE.md](CLAUDE.md).

## Development

```bash
uv run python -m ruff check .
uv run python -m mypy src eval
uv run python -m pytest
uv run python -m pytest -m docker   # needs Docker; skips without it
```
