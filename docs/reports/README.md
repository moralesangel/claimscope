# Reports

Papers analysed with `claude-sonnet-5` and the subprocess sandbox. Each report lists every claim
extracted from the paper, which ones were judged testable at reduced scale, the reduction plan
that was designed, and the verdict with its confidence interval.

**These results were produced without container isolation.** Each report says so in its header;
treat them as a demonstration of the pipeline, not as measurements to lean on.

| Paper | arXiv | Consistent | Not consistent | Inconclusive | Not testable |
|---|---|---|---|---|---|
| [Improving neural networks by preventing co-adapta...](1207.0580.md) | `1207.0580` | 0 | 0 | 2 | 8 |
| [Very Deep Convolutional Networks for Large-Scale ...](1409.1556.md) | `1409.1556` | 1 | 0 | 1 | 8 |
| [Adam: A Method for Stochastic Optimization](1412.6980.md) | `1412.6980` | 0 | 0 | 2 | 8 |
| [Batch Normalization: Accelerating Deep Network Tr...](1502.03167.md) | `1502.03167` | 2 | 0 | 0 | 8 |
| [Densely Connected Convolutional Networks](1608.06993.md) | `1608.06993` | 0 | 1 | 1 | 8 |
| [Model-Agnostic Meta-Learning for Fast Adaptation ...](1703.03400.md) | `1703.03400` | 0 | 0 | 2 | 8 |
| [Improved Regularization of Convolutional Neural N...](1708.04552.md) | `1708.04552` | 0 | 1 | 1 | 8 |
| [mixup: Beyond Empirical Risk Minimization](1710.09412.md) | `1710.09412` | 0 | 0 | 2 | 8 |
| [The Lottery Ticket Hypothesis: Finding Sparse, Tr...](1803.03635.md) | `1803.03635` | 1 | 1 | 0 | 8 |
| [A Convergence Theory for Deep Learning via Over-P...](1811.03962.md) | `1811.03962` | 2 | 0 | 0 | 8 |
| [RoBERTa: A Robustly Optimized BERT Pretraining Ap...](1907.11692.md) | `1907.11692` | 1 | 0 | 1 | 8 |
| [An Image is Worth 16x16 Words: Transformers for I...](2010.11929.md) | `2010.11929` | 0 | 1 | 1 | 8 |
| **Total** | | **7** | **4** | **13** | **96** |

That most come back "not testable" is the expected result rather than a shortfall: nearly every
claim in an ML paper rests on ImageNet, on pretraining, or on architectures that do not fit a CPU
budget. A system that returned a verdict for all of them would be inventing them.
