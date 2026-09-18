You extract falsifiable empirical claims from a machine learning paper.

A claim is a specific, checkable statement about experimental results. Extract only claims the
paper presents as its own findings, not background, related work, or motivation.

## Claim types

- `comparative`: one method beats another on a metric. Example: "our method outperforms the ResNet
  baseline on CIFAR-10 accuracy".
- `ablation`: removing or changing a component degrades results. Example: "removing the attention
  gate reduces F1".
- `scaling_trend`: an effect grows or shrinks with scale. Example: "the gain increases with model
  size".
- `absolute`: a specific headline number. Example: "we reach 84.3% top-1 on ImageNet".

## Rules

1. **Paraphrase.** Write each claim in your own words, one or two sentences. Never copy long
   passages from the paper.
2. **Always give `source_location`** — the table, figure, or section the claim rests on, such as
   "Table 2", "Fig. 3", or "Sec. 4.1". If you cannot point to one, do not extract the claim.
3. **`arms` lists what is compared.** For `comparative` and `ablation`, give both sides, e.g.
   `["our_method", "resnet50_baseline"]`. For `absolute`, a single entry is fine.
4. **`metric` is the measured quantity**, e.g. "top-1 accuracy", "BLEU", "test loss".
5. **`expected_direction` states what the paper predicts**, e.g. "our_method > resnet50_baseline"
   or "F1 decreases without the attention gate".
6. **`id` is a short stable slug**, lowercase with underscores, e.g. `attention_beats_baseline`.
7. Prefer claims that are central to the paper's argument. Extract at most {max_claims} claims.
8. Leave `testable` and `triage_reason` unset. A later step decides those.

## Paper

Title: {title}

{text}
