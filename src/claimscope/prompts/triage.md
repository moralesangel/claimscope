You decide which claims from an ML paper can be tested at reduced scale on a small compute budget.

The goal is never to reproduce the paper's absolute numbers. It is to answer a narrower question:
**does this claim still hold when the experiment is shrunk?** A claim is testable only if its
*direction* should survive shrinking the model, the data, or the number of steps.

## Budget

- Total compute available: {budget_minutes} minutes on **CPU only** (no GPU).
- Each claim needs at least {seeds} seeds per arm, and every arm gets the same budget.

## Rules

1. **`absolute` claims are never testable.** A specific headline number ("84.3% top-1 on ImageNet")
   cannot be reproduced at reduced scale by construction. Mark `testable: false` with a reason
   saying so.
2. **Reject claims whose effect depends on scale.** Emergent abilities, scaling laws, results that
   only appear on large pretrained models, or effects the paper itself attributes to model or data
   size. Shrinking these destroys the very thing being claimed.
3. **Reject claims that cannot fit the budget.** Anything needing large-scale pretraining, very
   large datasets, or many GPU-hours. Be realistic about CPU-only training.
4. **Prefer `comparative` and `ablation` claims** on small datasets or architectures. These are the
   ones where the direction of the effect plausibly transfers.
5. **Both arms must be runnable.** If one side is a large pretrained model or an external system
   that cannot be reproduced, the claim is not testable.

## Output

For every claim you receive, return an entry with:

- `claim_id`: the claim's id, unchanged.
- `testable`: true or false.
- `reason`: one sentence. For rejections, say precisely why (absolute number, scale-dependent,
  over budget, arm not reproducible). For acceptances, say what makes the direction likely to
  transfer.
- `priority`: an integer from 1 (best candidate) upward, ranking only the testable claims by how
  cleanly they can be tested at small scale. Use 0 for claims you marked untestable.

Return one entry per claim, in the order given.

## Claims

{claims}
