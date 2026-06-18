# Assumptions and disclosures

## Dataset

MOOCCube is the primary dataset. `entities/course.json`,
`entities/user.json`, `relations/course-concept.json`, and
`relations/course-video.json` are adapted to a unified schema.

MOOCCube text in this copy contains UTF-8-as-Latin-1 corruption. The adapter
repairs that encoding before stripping HTML and normalizing whitespace.

Learner IDs are SHA-256 hashes salted from YAML configuration.

## Missing feedback

Enrollment records do not contain completion rate, quiz score, or engagement
time. Fast mode creates explicit neutral values of `0.5`, marks every such row
`is_derived=true`, and computes reward as:

```text
0.4 × 0.5 + 0.4 × 0.5 + 0.2 × 0.5 = 0.5
```

These values are pipeline scaffolding, not empirical learner outcomes.

## Ten learner features

The following list is an implementation assumption:

1. average completion rate
2. average quiz score
3. normalized engagement time
4. average rating
5. normalized click count
6. normalized video-view count
7. normalized access frequency
8. normalized session duration
9. difficulty preference
10. recent activity score

Features are fit from the training split only.

## Embeddings

Fast mode uses deterministic signed feature hashing with 768 dimensions to
remain offline and CPU-compatible. Full mode loads the real checkpoint from
`models/bert-base-multilingual-cased`, fine-tunes only its last four encoder
layers, saves the result under the selected artifact directory, then extracts
L2-normalized 768-dimensional CLS vectors. Multilingual BERT is used instead
of the paper's English-only `bert-base-uncased` because MOOCCube text is Chinese.

## Translation

Translation runs only on final top-K titles/descriptions. Fast mode uses a small
offline glossary. Full mode loads the downloaded MarianMT checkpoint from
`models/opus-mt-zh-en`.

## Evaluation

With neutral derived rewards, ranking metrics are engineering smoke-test
results, not evidence of recommendation quality or paper reproduction.

The full experiment instead uses video-activity-derived rewards. Because these
remain proxies, all full metrics are labeled experimental rather than
paper-reproduced.

Reward z-score normalization is disabled in the final configuration after an
ablation showed lower Recall@10. Advantage normalization remains enabled.
