# Architecture

```text
MOOCCube JSON/relations
  → schema adapter + encoding repair + anonymization
  → temporal train/validation/test split
  → 10-D learner features
  → 768-D semantic course vectors
  → 778-D course-conditioned state
  → supervised next-course catalog policy pretraining
  → Gymnasium rollout + Actor–Critic PPO/GAE
  → semantic + Mahalanobis + cluster logit fusion
  → action masking for enrolled courses
  → top-K ranking
  → English translation of top-K only
```

The Streamlit application reads cached artifacts and does not retrain models on
rerun.

Two action spaces are supported:

- `full_catalog`: policy/ranking over all 706 courses.
- `candidate_set`: semantic, popularity, and exploration candidates followed by
  policy ranking and action masking.

The default full experiment uses `full_catalog`, as this catalog is small.
