# Paper analysis

## Scope

The requested architecture combines course semantics, learner context,
Mahalanobis similarity/clustering, Actor–Critic networks, and PPO-style
optimization for top-K course recommendation.

The 20-page file `s41598-026-40952-2.pdf` is present and was text-extracted on
June 18, 2026. The paper specifies a 128-token BERT input, 768-dimensional CLS
embedding, last-four-layer fine-tuning, AdamW at 2e-5, cosine annealing,
Actor/Critic hidden sizes 256 and 128, PPO clipping 0.2, discount 0.99, and
100 training epochs with early stopping patience 10.

## Implemented interpretation

The default state is:

```text
10 learner features + 768 semantic dimensions = 778 dimensions
```

Mahalanobis similarity and cluster affinity are fused into course logits rather
than appended to the state. This resolves the prompt's noted dimensional
inconsistency.

The actor uses hidden sizes 256 and 128; the critic uses the same hidden sizes
and emits a scalar value. Reward follows the requested
`0.4 completion + 0.4 quiz + 0.2 engagement` formula when those fields exist.

## Reproduction limits

- The paper does not enumerate the ten learner features.
- The RL transition process, unseen-action reward, negative sampling, and
  exact covariance regularization are under-specified.
- The paper-reported ~38M parameter count is inconsistent with the total size
  of `bert-base-uncased`; it may refer only to trainable parameters.
- Random splitting can leak future behavior, so temporal splitting is default.
- Exact processed research data is unavailable.

No paper-reported number is presented as a reproduced result by this project.
