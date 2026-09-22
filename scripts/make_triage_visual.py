"""Build the triage funnel page from a finished run.

The claim ids and triage reasons change on every run, because extraction is a
model call. Hand-writing the page meant it described whichever run happened to
be open at the time, so it is generated from the run's own triage.json instead.

    uv run python scripts/make_triage_visual.py runs/1207.0580 out.html
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

TEMPLATE_HEAD = """<title>What the Agent Refuses</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
  :root {
    --ground: #fbfbfc;
    --surface: #ffffff;
    --ink: #16181d;
    --ink-soft: #4a5058;
    --ink-faint: #767d87;
    --rule: #e3e6ea;
    --rule-soft: #eef0f3;
    --kept: #2a78d6;
    --refused-bg: #f2f4f6;
  }

  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --ground: #16171a;
      --surface: #1e2024;
      --ink: #f0f1f3;
      --ink-soft: #b3b8c0;
      --ink-faint: #868d97;
      --rule: #32353b;
      --rule-soft: #26282d;
      --kept: #3987e5;
      --refused-bg: #232529;
    }
  }

  :root[data-theme="dark"] {
    --ground: #16171a;
    --surface: #1e2024;
    --ink: #f0f1f3;
    --ink-soft: #b3b8c0;
    --ink-faint: #868d97;
    --rule: #32353b;
    --rule-soft: #26282d;
    --kept: #3987e5;
    --refused-bg: #232529;
  }

  body {
    background: var(--ground);
    color: var(--ink);
    font-family: "IBM Plex Sans", system-ui, -apple-system, sans-serif;
    line-height: 1.5;
  }

  .page {
    max-width: 880px;
    margin-inline: auto;
    padding-inline: 20px;
    padding-block: 48px 56px;
  }

  .eyebrow {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink-faint);
    margin: 0 0 14px;
  }

  h1 {
    font-size: clamp(27px, 4.4vw, 40px);
    font-weight: 600;
    letter-spacing: -0.021em;
    line-height: 1.16;
    text-wrap: balance;
    margin: 0 0 14px;
  }

  .standfirst {
    font-size: 16px;
    color: var(--ink-soft);
    max-width: 62ch;
    margin: 0 0 38px;
  }

  .tally { display: flex; gap: 3px; margin-bottom: 8px; }
  .tally-seg { height: 10px; border-radius: 3px; }
  .tally-seg.kept { background: var(--kept); }
  .tally-seg.refused { background: var(--refused-bg); }

  .tally-key {
    display: flex;
    gap: 22px;
    font-size: 13px;
    color: var(--ink-soft);
    margin: 0 0 34px;
    flex-wrap: wrap;
  }

  .tally-key span { display: flex; align-items: center; gap: 7px; }

  .dot { width: 9px; height: 9px; border-radius: 2px; flex: none; }
  .dot.kept { background: var(--kept); }
  .dot.refused { background: var(--refused-bg); border: 1px solid var(--rule); }

  h2 { font-size: 13px; font-weight: 600; letter-spacing: 0.02em; margin: 0 0 14px; }

  .group { margin-bottom: 34px; }

  .claims {
    display: flex;
    flex-direction: column;
    gap: 1px;
    background: var(--rule-soft);
    border: 1px solid var(--rule);
    border-radius: 6px;
    overflow: hidden;
  }

  .claim {
    background: var(--surface);
    padding: 14px 18px;
    display: grid;
    grid-template-columns: 3px 1fr;
    gap: 0 14px;
    align-items: start;
  }

  .stripe { width: 3px; border-radius: 2px; align-self: stretch; min-height: 34px; }
  .stripe.kept { background: var(--kept); }
  .stripe.refused { background: var(--rule); }

  .claim-body { min-width: 0; }

  .claim-id {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 12px;
    color: var(--ink);
    font-weight: 500;
    margin: 0 0 3px;
    overflow-wrap: anywhere;
  }

  .claim-reason { font-size: 13px; color: var(--ink-soft); margin: 0; }

  .closing {
    border-left: 2px solid var(--rule);
    padding-left: 16px;
    font-size: 14px;
    color: var(--ink-soft);
    max-width: 64ch;
    margin: 0 0 26px;
  }

  .closing strong { color: var(--ink); font-weight: 600; }

  .provenance {
    border-top: 1px solid var(--rule);
    padding-top: 18px;
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 11.5px;
    color: var(--ink-faint);
    display: flex;
    flex-wrap: wrap;
    gap: 6px 20px;
  }
</style>
"""


def _claim_block(claim: dict[str, object], kept: bool) -> str:
    kind = "kept" if kept else "refused"
    claim_id = html.escape(str(claim.get("id", "")))
    reason = html.escape(str(claim.get("triage_reason") or ""))
    return f"""      <div class="claim">
        <div class="stripe {kind}"></div>
        <div class="claim-body">
          <p class="claim-id">{claim_id}</p>
          <p class="claim-reason">{reason}</p>
        </div>
      </div>"""


def build(run_dir: Path) -> str:
    """Render the page from a run's triage.json."""
    triage = json.loads((run_dir / "triage.json").read_text(encoding="utf-8"))
    kept = [c for c in triage if c.get("testable")]
    refused = [c for c in triage if not c.get("testable")]
    total = len(triage)

    # Width encodes the split, so the bar cannot disagree with the counts.
    tally = (
        f'    <div class="tally-seg kept" style="flex: {len(kept)}"></div>\n'
        f'    <div class="tally-seg refused" style="flex: {len(refused)}"></div>'
        if kept and refused
        else f'    <div class="tally-seg {"kept" if kept else "refused"}" style="flex: 1"></div>'
    )

    groups = []
    if refused:
        groups.append(
            '  <div class="group">\n    <h2>Refused &mdash; and why</h2>\n'
            '    <div class="claims">\n'
            + "\n".join(_claim_block(c, kept=False) for c in refused)
            + "\n    </div>\n  </div>"
        )
    if kept:
        groups.append(
            '  <div class="group">\n'
            "    <h2>Kept &mdash; small enough to shrink and still mean something</h2>\n"
            '    <div class="claims">\n'
            + "\n".join(_claim_block(c, kept=True) for c in kept)
            + "\n    </div>\n  </div>"
        )

    paper_id = run_dir.name
    # The headline states the split, so it cannot contradict the counts beneath
    # it: "most claims cannot be tested" is wrong on a run that kept three of five.
    if len(refused) > len(kept):
        headline = "Most claims in this paper cannot be tested cheaply. The agent says which."
    elif refused:
        headline = f"{len(refused)} of these {total} claims are not worth testing cheaply."
    else:
        headline = "Every claim in this paper survived triage."
    plural = "claim" if len(refused) == 1 else "claims"
    return f"""{TEMPLATE_HEAD}
<div class="page">
  <p class="eyebrow">arXiv {html.escape(paper_id)}</p>
  <h1>{headline}</h1>
  <p class="standfirst">
    ClaimScope read the paper and extracted {total} empirical claims. Deciding which
    {len(refused)} {plural} to leave alone is most of the work &mdash; and refusing one is a
    correct answer, not a failure.
  </p>

  <div class="tally">
{tally}
  </div>
  <p class="tally-key">
    <span><i class="dot kept"></i> {len(kept)} testable at reduced scale</span>
    <span><i class="dot refused"></i> {len(refused)} refused, with reasons</span>
  </p>

{chr(10).join(groups)}

  <p class="closing">
    <strong>The refusals are the interesting part.</strong> An agent that tried to shrink
    ImageNet onto a laptop would produce a confident number that means nothing. Knowing which
    questions a cheap experiment cannot answer is what makes the answers it does give worth
    reading.
  </p>

  <p class="provenance">
    <span>Reasons quoted verbatim from the run</span>
    <span>Budget: CPU only</span>
  </p>
</div>
"""


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    run_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.write_text(build(run_dir), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
