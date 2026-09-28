# ClaimScope: implementation plan

A document for Claude Code. Read it in full before writing code. Work phase by phase, and do not move to the next one until the current one's acceptance criteria are met.

## 1. Goal

An agent built with LangGraph that, given an ML paper (arXiv), extracts its claims, decides which are testable within a limited compute budget, designs a reduced version of the experiment, runs it in a sandbox and issues a statistically grounded verdict.

The project does NOT aim to reproduce papers at full scale. It aims to answer: "does this relative claim hold at reduced scale?".

### Non-goals

- Reproducing absolute numbers from the paper's tables.
- Supporting papers whose effect depends on scale (LLM pretraining, emergent capabilities, large scaling laws).
- A web interface in the early phases. The interface is a CLI.

## 2. Domain concepts

### Claim types

| Type | Example | Testable at reduced scale? |
|---|---|---|
| `absolute` | "We reach 84.3% top-1 on ImageNet" | No. Recorded and discarded. |
| `comparative` | "Method A beats baseline B" | Yes. The main target. |
| `ablation` | "Removing component X makes the result worse" | Yes. |
| `scaling_trend` | "The improvement grows with model size" | Partly, with 3 or 4 small points. |

### Verdicts

- `consistent_at_reduced_scale`
- `not_consistent_at_reduced_scale`
- `inconclusive`
- `not_testable` (assigned at triage)

Report-writing rule: a negative result at reduced scale does NOT refute the paper. The report always says so.

### Reduction rules (invariants)

1. Both arms of a comparison get exactly the same budget, dataset, reduction and number of seeds.
2. At least 3 seeds per arm.
3. The architecture is preserved; depth, width, data or steps are what shrink.
4. Every change from the paper is documented in the reduction plan with its justification.

## 3. Stack

- Python 3.11+, managed with `uv`.
- `langgraph` and `langchain-core`. Model through `langchain-anthropic`, with the model name configurable (default `claude-sonnet-5`).
- `langgraph-checkpoint-sqlite` for persistence and resuming after interrupts.
- `pydantic` v2 for every schema and structured LLM output.
- `arxiv` (the package) and `pymupdf` for PDF ingestion.
- `docker` (the Python SDK) for the sandbox.
- `scipy` and `numpy` for statistics.
- `typer` and `rich` for the CLI.
- Tracing with Langfuse (self-hosted or cloud) or LangSmith, behind an environment variable. It must work with tracing off.
- `pytest`, `ruff`, `mypy`.

**Important:** LangGraph's API changes often (`interrupt`, `Command`, checkpointers). Before implementing the graph, consult the current documentation for the installed version rather than assuming the API from memory.

## 4. Repository layout

```
claimscope/
  pyproject.toml
  README.md
  CLAUDE.md                 # persistent rules for Claude Code (see section 11)
  .env.example
  src/claimscope/
    config.py               # settings with pydantic-settings
    state.py                # the graph state schema
    schemas.py              # Claim, ReductionPlan, RunResult, Verdict...
    graph.py                # graph construction
    nodes/
      ingest.py
      extract_claims.py
      triage.py
      design_plan.py
      review.py             # human approval interrupt
      codegen.py
      execute.py
      debug.py
      analyze.py
      report.py
    sandbox/
      docker_runner.py
      images/Dockerfile.cpu
    stats.py
    prompts/                # prompts as versioned .md files
    cli.py
  eval/
    annotations/            # hand-annotated papers (JSON)
    run_eval.py
    metrics.py
  tests/
  runs/                     # per-run outputs (git-ignored)
```

## 5. Graph state

```python
class Claim(BaseModel):
    id: str
    text: str                     # paraphrased, referencing a section/table
    source_location: str          # e.g. "Table 2", "Sec. 4.1"
    claim_type: Literal["absolute", "comparative", "ablation", "scaling_trend"]
    arms: list[str]               # e.g. ["method_A", "baseline_B"]
    metric: str
    expected_direction: str       # "A > B", "decreases without X"...
    testable: bool | None = None
    triage_reason: str | None = None

class ReductionPlan(BaseModel):
    claim_id: str
    original_setup: str
    reduced_setup: str
    changes: list[str]            # each change, with its justification
    preserved: list[str]
    why_claim_should_transfer: str
    seeds: int = 3
    estimated_minutes: float
    code_source: Literal["official_repo", "from_scratch"]

class RunResult(BaseModel):
    arm: str
    seed: int
    metric_value: float
    runtime_s: float
    log_path: str

class ClaimVerdict(BaseModel):
    claim_id: str
    verdict: Literal[...]         # see section 2
    effect_estimate: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    notes: str

class GraphState(TypedDict):
    paper_id: str
    paper_text: str
    repo_url: str | None
    claims: list[Claim]
    selected_claim_ids: list[str]
    plans: dict[str, ReductionPlan]
    approved_plan_ids: list[str]
    workspace_dir: str
    run_results: dict[str, list[RunResult]]
    debug_attempts: dict[str, int]
    budget_minutes_total: float
    budget_minutes_used: float
    verdicts: list[ClaimVerdict]
    report_path: str | None
    errors: list[str]
```

## 6. Nodes and flow

```
ingest -> extract_claims -> triage -> design_plan -> review (interrupt)
   review --approved--> codegen -> execute
   review --rejected with feedback--> design_plan
   execute --error--> debug -> execute   (max N attempts per claim)
   execute --N exhausted--> analyze (claim marked inconclusive, with the reason)
   execute --ok--> analyze -> report -> END
```

### Per-node contracts

- **ingest**: downloads the arXiv PDF by ID, extracts the text with PyMuPDF, finds the link to the official repo (GitHub) if there is one. Caches in `runs/<paper_id>/`.
- **extract_claims**: structured output `list[Claim]`. Paraphrase, never long blocks copied from the paper. Always a `source_location`.
- **triage**: sets `testable` and `triage_reason`. `absolute` is never testable. Rejects claims whose effect depends on scale or whose estimated cost exceeds the budget. Selects at most K claims (configurable, default 2).
- **design_plan**: produces a `ReductionPlan` respecting the invariants in section 2. Where there is an official repo, prefers `official_repo`.
- **review**: a LangGraph `interrupt` that shows the plan in the CLI. The user approves, rejects, or gives feedback as text. The feedback goes back to `design_plan`.
- **codegen**: with `official_repo`, clones and adapts configs and sizes; with `from_scratch`, generates a minimal script. Either way it produces a single entry point, `run.py --arm <arm> --seed <seed>`, which writes a JSON with the metric.
- **execute**: a short dry run first (say 20-50 steps) to measure real time and extrapolate. If the extrapolation exceeds the remaining budget, it returns to `design_plan` with that figure. If it fits, it runs every arm and seed.
- **debug**: receives the traceback and the code, proposes a patch. Attempt limit configurable (default 3).
- **analyze**: see section 8.
- **report**: Markdown at `runs/<paper_id>/report.md` with claims, triage, plan, results, verdicts and limitations.

## 7. Sandbox

- A Docker container, non-root user, with CPU, memory and time limits.
- Dependencies installed in a separate phase; the experiment itself runs with no network.
- Only the claim's working directory is mounted.
- A `Runner` abstraction with a `DockerCPURunner` implementation. Leave the interface ready for a future `GPURunner` (local or remote GPU), without implementing it yet.
- The development hardware has no CUDA: phases 1 to 5 must work on CPU with experiments of a few minutes.

## 8. Statistics and the verdict rule

- Per arm: mean and standard deviation over seeds.
- Effect = difference of means in the expected direction.
- Confidence interval by bootstrap (over seeds), with a Welch test as support.
- The rule:
  - CI entirely in the expected direction -> `consistent_at_reduced_scale`
  - CI entirely in the opposite direction -> `not_consistent_at_reduced_scale`
  - CI crossing zero -> `inconclusive`
- With 3 seeds the power is low: the report must say so explicitly. If budget remains, the agent may propose more seeds rather than more steps.

## 9. Evaluating the agent itself

Annotation format (`eval/annotations/<paper_id>.json`):

```json
{
  "paper_id": "xxxx.xxxxx",
  "claims": [
    {"text": "...", "claim_type": "comparative", "testable": true,
     "expected_verdict": "consistent_at_reduced_scale"}
  ]
}
```

Metrics:
- Extraction: claim precision and recall against the annotation (matched with the LLM as judge, plus manual review of a sample).
- Classification: accuracy of `claim_type` and `testable`.
- Execution: share of claims whose code runs without errors, and mean debug attempts.
- Verdict: agreement with `expected_verdict`.
- Cost: tokens and compute minutes per claim.

Target: 10-15 annotated papers, including some with known failed replications (for instance ML Reproducibility Challenge reports).

## 10. Phases

Every phase ends with tests passing, a commit, and a short update to `CLAUDE.md`.

**Phase 0. Skeleton**
- `uv init`, folder structure, `pyproject.toml`, ruff, mypy, pytest, `.env.example`, an empty CLI.
- Acceptance: `uv run claimscope --help` works; local CI (`ruff`, `mypy`, `pytest`) green.

**Phase 1. Ingestion and claim extraction**
- `ingest` and `extract_claims` nodes, a minimal linear graph.
- Acceptance: produces a valid `claims.json` on 2 test papers; tests with the LLM mocked.

**Phase 2. Triage, plan and interrupt**
- `triage`, `design_plan`, `review` with a SQLite checkpointer.
- Acceptance: the run can be interrupted, the process closed, and resumed from the CLI with `claimscope resume <thread_id>`.

**Phase 3. Sandbox and execution**
- Docker runner, `codegen` in `from_scratch` mode, `execute` with a dry run, the `debug` loop.
- Acceptance: a toy claim (say "dropout narrows the gap between train and test loss in a small MLP on MNIST") runs end to end on CPU in under 15 minutes.

**Phase 4. Analysis and report**
- `stats.py`, `analyze`, `report`.
- Acceptance: a complete Markdown report for the toy claim; unit tests of the verdict rule with synthetic data.

**Phase 5. Official repo mode**
- `codegen` in `official_repo` mode: clone, locate configs, reduce scale.
- Acceptance: a real paper with official code, chosen by the user, reaches a verdict.

**Phase 6. Evaluation harness**
- `eval/run_eval.py` and the metrics from section 9.
- Acceptance: a metrics table over at least 5 annotated papers.

**Phase 7. Polish**
- Tracing (Langfuse or LangSmith), a README with the graph diagram, the evaluation results, honest limitations, and positioning against related work (PaperBench, CORE-Bench).

## 11. Rules for Claude Code (copy into CLAUDE.md)

- Code, names and comments in English. User-facing documentation in English too.
- Do not move on from a phase without meeting its acceptance criteria.
- For design decisions this plan does not cover, ask before implementing.
- Consult the current LangGraph documentation before using `interrupt`, `Command` or checkpointers.
- Never hardcode API keys; use `.env`.
- Never run LLM-generated code outside the Docker sandbox.
- Prompts live in `src/claimscope/prompts/` as files, not embedded in the code.
- Every LLM output validated with Pydantic; if validation fails, retry once and log the error.
- Tests with the LLM mocked by default; tests that call the real API are marked and run only on demand.
- Small commits with descriptive messages, at least one per phase.
