<!--
A real ClaimScope run, kept as an example of what the tool produces.

arXiv 1207.0580 (Dropout), analysed with gemini-3.5-flash and executed in the
subprocess sandbox on a laptop CPU. Every number below came out of an actual
run; nothing here is illustrative.
-->

# ClaimScope report: Improving neural networks by preventing co-adaptation of feature detectors

arXiv `1207.0580` · generated 2026-09-21 23:20 UTC

This report tests whether individual claims from the paper still hold **at reduced scale**. It is not a reproduction of the paper.

## Verdicts

| Claim | Type | Verdict | Effect | 95% CI |
|---|---|---|---|---|
| `mnist_dropout_improves_backprop` | comparative | Consistent at reduced scale | +0.068 | [+0.0551, +0.0822] |
| `mnist_input_dropout_improves_hidden_dropout` | ablation | Not testable at reduced scale | — | — |
| `mnist_dbn_dropout_finetuning` | comparative | Not testable at reduced scale | — | — |
| `mnist_dbm_dropout_finetuning` | comparative | Not testable at reduced scale | — | — |
| `timit_dropout_recognition_rate` | comparative | Not testable at reduced scale | — | — |
| `cifar10_dropout_improves_cnn` | comparative | Not testable at reduced scale | — | — |
| `imagenet_dropout_improves_cnn` | comparative | Not testable at reduced scale | — | — |
| `reuters_dropout_improves_nn` | comparative | Not testable at reduced scale | — | — |

## Tested claims

### mnist_dropout_improves_backprop

> A standard feedforward neural network trained on MNIST with 50% dropout and separate L2 weight constraints achieves fewer test errors than one trained without dropout.

Source: Sec. 2 · Metric: test errors · Expected: dropout_50_with_l2_constraints < standard_feedforward_nn

**Verdict: Consistent at reduced scale**

standard_feedforward_nn: 0.1867 (sd 0.0142, n=5); dropout_50_with_l2_constraints: 0.1187 (sd 0.0098, n=5). Effect in the predicted direction: +0.068 [95% CI +0.05505, +0.08219], Welch p=4.39e-05.

#### Reduction plan

_This is what was **designed**. The experiment was then written by a model from this description, so read the script in the workspace to see what actually ran -- the two can differ, for instance where the sandbox has no network and the data had to be generated instead of downloaded._

**Original setup.** A deep feedforward neural network with 2 to 3 hidden layers of 1024 to 2048 units trained on the MNIST dataset for hundreds of epochs with 50% dropout and max-norm weight constraints.

**Reduced setup.** A 2-layer MLP with 256 units per hidden layer trained on MNIST for 10 epochs using a batch size of 128 and SGD with momentum. Arm 1: No dropout or weight constraints. Arm 2: 50% dropout on hidden layers and a max-norm constraint of 3.0 on weight vectors.

**Why the claim should transfer.** The regularizing effect of dropout is fundamental and manifests rapidly during training. Even with a smaller MLP and only 10 epochs of training on MNIST, the network without dropout will begin to overfit, leading to a higher test error rate compared to the network trained with dropout and max-norm constraints.

**Seeds per arm:** 5 · **Estimated cost:** 10 min · **Code source:** from_scratch

**Changes from the paper**

- Reduced hidden layer size from 1024/2048 to 256 to significantly speed up training on CPU while retaining sufficient parameters to overfit.
- Reduced training length to 10 epochs to fit the CPU budget while still allowing the generalization gap to become apparent.
- Used standard SGD with momentum 0.9 instead of custom learning rate schedules for simplicity.

**Deliberately preserved**

- The MNIST dataset is kept intact to maintain the complexity of the digit recognition task.
- The 50% dropout rate on hidden layers is preserved as it is the core mechanism of the claim.
- The max-norm constraint on weight vectors is preserved because dropout requires this constraint to prevent weight values from compensating excessively.

#### Runs

| Arm | Seed | Metric | Runtime (s) |
|---|---|---|---|
| dropout_50_with_l2_constraints | 0 | 0.117194 | 3.5 |
| dropout_50_with_l2_constraints | 1 | 0.118736 | 3.5 |
| dropout_50_with_l2_constraints | 2 | 0.104086 | 3.7 |
| dropout_50_with_l2_constraints | 3 | 0.122591 | 3.8 |
| dropout_50_with_l2_constraints | 4 | 0.131072 | 3.5 |
| standard_feedforward_nn | 0 | 0.208173 | 4.0 |
| standard_feedforward_nn | 1 | 0.184271 | 2.8 |
| standard_feedforward_nn | 2 | 0.170393 | 2.7 |
| standard_feedforward_nn | 3 | 0.19121 | 2.8 |
| standard_feedforward_nn | 4 | 0.179645 | 2.8 |

## Claims not tested

- **`mnist_input_dropout_improves_hidden_dropout`** (ablation) — Adding 20% input dropout to an MNIST network already using 50% hidden unit dropout further reduces the number of test errors. _Adding input dropout to an MNIST network is computationally trivial and can be verified quickly on CPU._
- **`mnist_dbn_dropout_finetuning`** (comparative) — Fine-tuning a pre-trained Deep Belief Network on MNIST with 50% dropout reduces test errors compared to standard backpropagation fine-tuning. _Pre-training and fine-tuning Deep Belief Networks is computationally expensive and complex to implement on a CPU-only budget._
- **`mnist_dbm_dropout_finetuning`** (comparative) — Fine-tuning a pre-trained Deep Boltzmann Machine on MNIST using 50% dropout lowers the average test errors compared to standard fine-tuning. _Pre-training and fine-tuning Deep Boltzmann Machines is too computationally intensive and complex for a limited CPU budget._
- **`timit_dropout_recognition_rate`** (comparative) — Using 50% dropout on hidden units and 20% on inputs during TIMIT acoustic model fine-tuning yields a lower phone recognition error rate compared to standard backpropagation. _The TIMIT dataset is proprietary and its speech preprocessing pipeline is too complex to run within a tight CPU budget._
- **`cifar10_dropout_improves_cnn`** (comparative) — Adding dropout to the final hidden layer of a deep convolutional neural network lowers the test error rate on CIFAR-10. _A scaled-down CNN on CIFAR-10 can easily be trained on CPU to verify if dropout reduces overfitting and lowers test error._
- **`imagenet_dropout_improves_cnn`** (comparative) — Incorporating 50% dropout in the globally connected layers of a deep convolutional neural network reduces the classification error rate on ImageNet. _Training a CNN on ImageNet is far too computationally expensive for a 120-minute CPU-only budget._
- **`reuters_dropout_improves_nn`** (comparative) — Training a two-layer feedforward neural network with 50% dropout on the Reuters text classification dataset reduces test error compared to standard backpropagation. _Training a simple two-layer MLP on the Reuters text dataset is extremely fast on CPU and highly suitable for testing dropout._

## How these experiments were run

> **These results were produced without container isolation.** The experiments ran in a restricted subprocess, with network access blocked and resources capped, but not inside a sandbox that can contain what it runs. Treat the results as a demonstration of the pipeline rather than as a trustworthy measurement, and re-run under Docker before relying on them.

## Limitations

Read these results narrowly.

- **A negative result here does not refute the paper.** These experiments run at a fraction of the
  original scale. An effect that disappears when the model, the data, or the training budget shrinks
  may still be real at full scale; that is precisely what reduced-scale testing cannot tell us.
- **A positive result is not a reproduction.** It says the direction of the claim survived
  shrinking, not that the paper's numbers were reproduced.
- **Seed counts are small.** With a handful of seeds per arm, statistical power is low. An
  inconclusive verdict usually means there was not enough evidence either way, not that the effect
  is absent.
- **Absolute claims are never testable this way** and are recorded without being run.
- The reduced experiments were written from the paper's description, so they may differ from the
  authors' implementation in ways that matter.
