# CLAUDE.md

Persistent rules for working on ClaimScope. The full plan is in [PLAN.md](PLAN.md); this file
carries the rules from its section 11 plus the project's current state.

## Rules

- Code, names and comments in English. User-facing documentation in English too.
- Do not move on from a phase without meeting its acceptance criteria (PLAN.md, section 10).
- For design decisions the plan does not cover, ask before implementing.
- Consult the current LangGraph documentation before using `interrupt`, `Command` or
  checkpointers. The API changes often; do not assume it from memory.
- Never hardcode API keys; use `.env` (see `.env.example`).
- Never run LLM-generated code outside the Docker sandbox.
- Prompts live in `src/claimscope/prompts/` as versioned `.md` files, not embedded in the code.
- Every LLM output validated with Pydantic; if validation fails, retry once and log the error.
- Tests with the LLM mocked by default; tests that call the real API are marked
  (`@pytest.mark.llm`) and run only on demand.
- Small commits with descriptive messages, at least one per phase.

## Domain invariants

When designing or reviewing a `ReductionPlan` (PLAN.md, section 2):

1. Both arms of a comparison get exactly the same budget, dataset, reduction and number of seeds.
2. At least 3 seeds per arm.
3. The architecture is preserved; depth, width, data or steps are what shrink.
4. Every change from the paper is documented with its justification.

A negative result at reduced scale does NOT refute the paper. The report must always say so.

## Commands

```bash
uv sync --all-extras                    # install dependencies
uv run python -m claimscope.cli --help  # CLI
uv run python -m ruff check .           # lint
uv run python -m ruff format .          # formatting
uv run python -m mypy src               # types
uv run python -m pytest                 # tests (LLM mocked)
uv run python -m pytest -m llm          # tests against the real API, on demand
```

## Environment

- Python 3.11+ (mypy targets 3.12: the numpy stubs use that version's syntax).
- Development without CUDA: experiments must run on CPU in a few minutes.
- **LangGraph 1.x**, not the 0.2.x series the plan suggests. Consult its documentation before
  touching `interrupt`, `Command` or checkpointers.
- Commands are invoked as `uv run python -m <module>` rather than by executable name: on some
  Windows machines the virtualenv's `.exe` shims are blocked by code integrity policies.
- `vendor/xxhash_pure/` exists because the native `xxhash` wheel can be blocked on Windows with
  Smart App Control, and LangGraph imports it at module level. It implements XXH3-128 in pure
  Python, validated against the 12,483 official vectors, so checkpoints stay portable. On a machine
  without that restriction, removing the `[tool.uv.sources]` entry is enough.

Anything specific to one machine goes in `CLAUDE.local.md`, which git ignores.

## Status

- **Phase 0 (skeleton): complete.** Package structure, `pyproject.toml`, ruff, mypy, pytest,
  `.env.example`, and a CLI with `--help`.
- **Phase 1 (ingestion and extraction): complete.** `ingest` and `extract_claims` nodes, a linear
  graph, Pydantic schemas, an LLM client with validation and one retry, and
  `claimscope analyze <arxiv_id>`.
- **Phase 2 (triage, plan and interrupt): complete.** `triage`, `design_plan` and `review` nodes
  with `interrupt`, a SQLite checkpointer, and the `resume` and `threads` commands.
- **Phase 3 (sandbox and execution): implemented, ACCEPTANCE PENDING.** `Runner`,
  `DockerCPURunner`, `codegen` from_scratch, `execute` with a dry run, and the `debug` loop.

  **Docker is not installed on this machine**, so the acceptance criterion (a toy claim end to end
  on CPU in under 15 min) **has not been verified** locally. The code is complete and tested
  against a `FakeRunner`; `tests/test_sandbox_integration.py` holds the real containment checks,
  which skip on their own without a daemon and run in CI. With Docker installed:
  `uv run python -m pytest -m docker`, then a real run.

  Docker Desktop needs WSL2, which in turn needs administrator rights and a reboot. Neither can be
  done from this session.
- **Phase 4 (analysis and report): complete.** `stats.py` (bootstrap over seeds, Welch as support,
  the verdict rule), the `analyze` and `report` nodes. Acceptance met: a complete Markdown report
  for the toy claim, and verdict-rule tests with synthetic data.
- **Phase 5 (official repo mode): implemented, ACCEPTANCE PENDING.** `repo.py` clones and inspects
  the paper's repository, and `codegen` generates an adapter that runs it at reduced scale.
  Acceptance ("a real paper with official code reaches a verdict") **needs Docker**, as phase 3
  does. The inspection itself is verified against real repos (pytorch-cifar, nanoGPT).
- **Phase 6 (evaluation harness): complete in what can be measured.** `eval/` with annotations,
  claim matching, metrics and `run_eval.py`. Table produced over 2 annotated papers.

  **Known limitation:** the execution and verdict metrics cannot be measured without Docker. The
  harness returns `measured=False` and the table shows `—` rather than `0.00`: a zero would assert
  that the agent failed, when the truth is that it was not measured.
- **Evaluation corpus extended to 13 papers, 90 claims.** Meets the plan's 10-15, with tests that
  watch its balance.
- **Phase 7 (polish): complete.** Optional tracing (LangSmith and Langfuse), a rewritten README
  with the graph diagram, limitations, and positioning against PaperBench and CORE-Bench.
- **Subprocess sandbox + Colab notebook: added.** `CLAIMSCOPE_SANDBOX_BACKEND=subprocess` allows
  running where there is no Docker. **The pipeline has now run end to end**: a real experiment
  runs, produces metrics and reaches a verdict (`tests/test_end_to_end.py`).
- **VALIDATED END TO END WITH A REAL LLM.** A full run over arXiv 1207.0580 with
  `gemini-3.5-flash` and the subprocess sandbox: 8 claims extracted, 7 discarded by triage with
  correct reasons, experiment reduced to `load_digits`, 10 runs (2 arms × 5 seeds), verdict
  `consistent_at_reduced_scale` with effect +0.068, CI [+0.055, +0.082], Welch p=4.4e-05. The
  report is kept at `docs/example-report.md`.

  The measured effect is real, but **that run's report says "MNIST" while the script used
  `load_digits`**. It is noted in the header of `docs/example-report.md`. See below.
- **Experiment integrity: fixed after finding silently false results.** A later run produced a
  CIFAR-10 verdict computed over sklearn's 8x8 digits, a Reuters one computed with
  `make_classification`, and all three scripts randomised 25% of the training labels. MNIST error
  came out at 61% instead of ~5%: both arms were fitting destroyed labels, so neither could win.

  **The cause was three prompts contradicting each other.** `design_plan` asked for MNIST and
  CIFAR-10 to be preferred, which the sandbox cannot download; `codegen` forbade substitution; and
  `debug` ordered "replace that with synthetic data" on any download failure. The planner promised
  impossible data and the `debug` node -- which sees exactly that failure -- quietly swapped it.

  The three prompts now agree: the planner knows which data exists without a network and has to
  name the substitute in `reduced_setup` (invariant 4), and `debug` no longer recommends
  substituting or corrupting labels. `src/claimscope/integrity.py` inspects the generated script
  and records in `errors` -- which the report prints -- any undocumented substitution or label
  corruption. It advises rather than blocks: a false positive that killed a claim would cost more
  than the warning.

  **When touching one of those three prompts, review the other two.** Each is reasonable on its
  own; the failure only appears when they compose, and it produces a confident wrong answer rather
  than an error. And after any change, **read the generated `run.py`**, not the report: the report
  prints the plan, not what the script did.
- **A correct experiment can still be unable to answer.** With honest data, all four claims came
  back `inconclusive`: a 2-layer MLP on 1797 easy digits reaches 5% error without overfitting, and
  dropout has nothing to regularise. The planner's prompt now asks for the regime where the effect
  can appear to be preserved (for a regulariser, a visible train/test gap in the baseline arm).
- Outstanding, and blocked only by the environment: the Docker sandbox has never run locally (no
  WSL2), and the evaluation harness has not been run over all 13 papers.

### Subprocess backend notes

- **It is not containment, and must not be presented as such.** It blocks the network, caps
  resources and scrubs the environment, but code running in that process can undo all of it from
  inside. It is opt-in, warns on prepare, and sets `unconfined_execution` so the report tells the
  reader.
- **The network block is injected through `sitecustomize.py`**, which Python imports at startup. It
  lives in a directory beside the workspace, not inside it, so `run.py` cannot import it by
  accident.
- `socket.connect`, `create_connection`, `getaddrinfo` and `urllib` are patched, plus the
  `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` variables as a second barrier.
- The `resource` limits are POSIX-only: on Windows they do not apply and the matching test skips.
  On Colab (Linux) they work.
- `prepare()` installs into the current environment, not into an image. That is a real side effect,
  and another reason this backend is opt-in.
- The notebook `notebooks/claimscope_colab.ipynb` verifies the network block **before** spending
  any model calls.

### Phase 7 notes

- **Tracing must never take a run down.** A misconfigured backend, a missing package or an
  unreachable host: warn and carry on without tracing. `tests/test_tracing.py` checks this
  explicitly.
- **Flush on exit.** Tracing clients batch in the background, and a short CLI run finishes before
  anything is sent.
- `langsmith` works here because the `xxhash` shim unblocks it; `langfuse` is an optional extra
  (`uv sync --extra tracing`).
- **The evaluation corpus had no claim it was expected to contradict.** That would have given a
  perfect score to a sycophantic agent. MAML contributes two, and a test prevents the regression.

### Phase 6 notes

- **Matching is lexical by default, not LLM-based.** The plan suggests the LLM as judge, but that
  is non-deterministic and costs a call per pair. `eval/matching.py` uses Jaccard over content
  words, weighting the metric and the arm names, which is what separates claims from the same
  paper. Greedy, one-to-one matching.
- **Classification is scored only over matched claims.** Penalising the type of a claim the agent
  never extracted would count the same failure twice.
- **The evaluation auto-approves plans.** A human approving each plan would measure the human, not
  the agent. That means code would run unreviewed, so without a sandbox the harness **stops before
  executing** rather than running it unprotected.
- `eval/` needs an `__init__.py` and is included in mypy (`files = ["src", "eval"]`).
- Ruff's B008 is disabled in the CLIs: `typer.Option` in defaults is its documented API.

### Phase 5 notes

- **There is no common configuration format.** `pytorch-cifar` exposes only `--lr` with the rest
  hardcoded; nanoGPT uses Python files in `config/`. That is why `repo.py` does not try to
  *understand* the repo: it gathers evidence (entrypoints, flags, configs, imports) and the prompt
  asks the model to write the adapter against that evidence.
- **Imports are inferred from the AST, not from `requirements.txt`.** Many research repos do not
  declare dependencies: `pytorch-cifar` has no requirements and needs torch. Without
  `infer_imports()` the sandbox image would have no torch and **every run would fail on the first
  import**. The stdlib (via `sys.stdlib_module_names`) and the repo's own modules are filtered out.
- **`execute` calls `runner.prepare()`** before anything else. This was a gap in phase 3: the image
  was never built. Dependencies are installed there, the one step with network access.
- **A repo that cannot be cloned does not cost the claim**: it falls back to `from_scratch` and the
  reason is recorded in `errors`.
- The repo's commit is stored in `repo_commits` and appears in the report. Without it, the result
  is not reproducible.

### Phase 4 notes

- **The effect's sign depends on the metric.** `metric_direction()` detects metrics where lower is
  better (loss, error, perplexity, RMSE, FID...) by whole words, not substrings: otherwise
  "lossless" would be classified as loss. Getting this wrong inverts the verdict silently.
- **The interval is signed before the rule is applied**, so positive always means "in the direction
  the paper predicts". That makes section 8's rule a single comparison against zero.
- **An interval touching zero is `inconclusive`**, not consistent. A bound exactly at zero is not
  evidence of direction.
- **Every path through the graph reaches `analyze`**, including "triage selected nothing".
  Otherwise a paper with no testable claims would produce no report, and the silence would read as
  success.
- **`build_graph` applies the serialiser to any checkpointer**, not only the SQLite one. Testing
  with `InMemorySaver` brought back the unregistered-type warnings, which will be errors in a
  future version.
- Arms must be matched: `analyze` refuses to compare 3 seeds against 2 and reports it as
  `inconclusive` with the reason.

### Phase 3 notes

- **The sandbox's types go in `_ALLOWED_MODULES` too.** `ExecutionFailure` and `ExecutionResult`
  travel in the graph state; `ExecutionRequest` and `ImageSpec` do not. The guard in
  `tests/test_session.py` caught this omission when `GeneratedCode` and `CodePatch` were added.
- **`execute` stops at the first failure.** A broken script fails the same way for every seed;
  carrying on only spends budget before `debug` can fix it.
- **The dry run measures rather than guesses.** It runs `--steps 20`, extrapolates with
  `full_run_steps`, and if it does not fit the remaining budget marks the claim in
  `over_budget_claim_ids` instead of starting a study that will not finish.
- **The code validator strips markdown fences** (`strip_markdown_fences`) because the model adds
  them even though the prompt forbids it, and it respects backticks that are inside the code.
- Careful testing fences from PowerShell: the backtick is its escape character and corrupts the
  input. Use a `.py` file, not `-c` with a here-string.

### Phase 2 notes

- **On resume, the node re-runs from the top.** Everything before `interrupt()` runs twice, so that
  part must have no side effects and the decision is only applied afterwards. Documented in the
  docstring of `nodes/review.py`.
- **Schemas must be registered in `session._ALLOWED_MODULES`.** LangGraph warns when deserialising
  unregistered types and will block them in a future version, which would leave interrupted runs
  unable to resume. `tests/test_session.py` fails if you add a schema and forget to register it;
  verified with `LANGGRAPH_STRICT_MSGPACK=true`.
- **`max_retries=0` on both providers.** Their SDKs retry on their own (~40 s on a daily quota
  error that will never resolve) and hide the attempts from our logging. `_is_transient()` decides
  better: it tells congestion apart from exhausted quota.
- **Gemini's free tier gives 20 requests/day PER MODEL**, not per account
  (`GenerateRequestsPerDayPerProjectPerModel`). This blocked the project for days before it was
  found: with `gemini-3.6-flash` exhausted, `gemini-flash-latest` answered perfectly, and there are
  more than 20 models available on the same key.

  `Settings.model_chain()` and `DEFAULT_FALLBACKS` implement the automatic switch: if a model runs
  out of quota **or stays congested through the whole backoff**, it moves to the next. A per-minute
  429 does not switch models, because waiting does resolve that one.
- **`claimscope doctor`** checks the key, a model that answers, and the sandbox before spending
  quota. It is the first thing to run when something fails.
- The "`absolute` is never testable" rule is enforced in code, not trusted to the model.

### Local model (Ollama)

`CLAIMSCOPE_PROVIDER=ollama` uses a local model: no key, no quota. Installed and verified with
**Qwen3 4B Q4_K_M** (2.5 GB).

**The network blocks Ollama's CDN.** `registry.ollama.ai` answers, but `r2.cloudflarestorage.com`
times out, so `ollama pull` always fails. Hugging Face does work: download the `.gguf` from there
and `ollama create` with a Modelfile. The Modelfile has to set `num_ctx`.

**`num_ctx` is critical.** Ollama defaults to 2048 tokens, and the extraction prompt carries up to
60k characters (~15k tokens): the paper would be truncated silently. It is set to 24,000 both in
the Modelfile and in `Settings.ollama_context_tokens`.

**What was measured (Intel Ultra 5 225H, 14 cores, no usable GPU):**

| Task | Time | Quality |
|---|---|---|
| Extraction over a short excerpt | 97 s | Valid JSON, but `arms=[]` on all of them |
| Triage of 3 claims | 47 s | Misclassified: approved the TIMIT claim (a licensed dataset) |
| Extraction over the real paper (20k chars) | **>15 min, aborted** | — |

**Conclusion: useful for testing the wiring, not for production.** Three quality problems Gemini
does not have: it does not fill `arms` (which breaks `codegen`), it confuses `ablation` with
`absolute`, and its triage lets untestable claims through. And on a full paper the time per call
makes it impractical: one run is 4+ calls.

Use it to check the pipeline connects without spending quota; use Gemini or Anthropic to judge the
agent's quality.

### LLM providers

The plan fixes on Anthropic. Since that account had no balance, `llm.py` abstracts the provider
behind the `StructuredLLM` Protocol and it is chosen with `CLAIMSCOPE_PROVIDER`
(`anthropic` | `google`), without touching code. An empty `CLAIMSCOPE_MODEL_NAME` takes the
provider's default model.

- **Phase 1 was validated with `gemini-3.6-flash`**, not with Claude. Going back to Anthropic means
  revalidating: models differ in how many claims they extract and in their fidelity.
- **The `gemini-2.5-*` models are retired** for new accounts; the current series is 3.x.
- **`gemini-3.1-pro-preview` returns 429 with `limit: 0`** on the free tier: that is not
  congestion, it is unavailability. Only `flash` works with this key.
- Gemini keys can carry an `AQ.` prefix as well as the classic `AIza`; both go in the
  `x-goog-api-key` header.
- `_is_transient()` tells retryable failures (503, 429 from congestion) apart from permanent ones
  (`limit: 0`, exhausted balance, auth). Retrying a `limit: 0` achieves nothing.

### Phase 1 notes

- **`arxiv` 4.0.1 removed `Result.download_pdf`.** It only exposes `pdf_url`; the download is done
  with `requests` in `nodes/ingest.py`.
- **URLs break across lines in PDFs.** PyMuPDF extracts
  `https://github.com/\ntensorflow/tensor2tensor`, so `find_repo_url` rejoins the breaks that
  follow a `/` or a `-` before searching. Only those: joining after a sentence's full stop swallows
  the prose that follows. Both cases have regression tests with real text from arXiv 1706.03762.
- **`ChatAnthropic` types its `__init__` as `(*args, **kwargs)`**, so mypy does not validate its
  kwargs. The aliases are used (`model_name`, `api_key`, `timeout`, `stop`), verified at runtime.
- **mypy targets Python 3.12**, not 3.11, because the stubs bundled with numpy use 3.12 syntax. The
  package's `requires-python` is still 3.11.
- Tests share doubles in `tests/stubs.py`, with `pythonpath = ["tests"]` in the pyproject.
- **Tests are isolated from the environment.** An `autouse` fixture in `conftest.py` clears every
  variable `Settings` reads and `chdir`s to a temporary directory. Without it, `Settings` loads the
  developer's real `.env` and the suite only passes on their machine. When adding a new field to
  `Settings`, add its variable to `_SETTINGS_ENV_VARS`.
