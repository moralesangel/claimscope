# Informes

Papers analizados con `claude-sonnet-5` y el sandbox de subproceso. Cada informe lista todas
las afirmaciones extraídas del paper, cuáles se consideraron comprobables a escala reducida,
el plan de reducción que se diseñó y el veredicto con su intervalo de confianza.

**Estos resultados se produjeron sin aislamiento por contenedor.** Cada informe lo dice en su
cabecera; trátalos como una demostración del pipeline, no como medidas en las que apoyarse.

| Paper | arXiv | Consistente | No consistente | Inconcluso | No comprobable |
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

Que la mayoría salga "no comprobable" es el resultado esperado, no un fallo: casi todas las
afirmaciones de un paper de ML dependen de ImageNet, de preentrenamiento o de arquitecturas que
no caben en un presupuesto de CPU. Un sistema que diera un veredicto para todas se los estaría
inventando.
