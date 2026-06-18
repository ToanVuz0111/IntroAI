# Reproduction report

## Completed implementation

The project now includes real local multilingual BERT, last-four-layer
fine-tuning, 768-dimensional CLS embeddings, ten train-only learner features,
a 778-dimensional state, PCA/Ledoit-Wolf Mahalanobis similarity, clustering,
a full-catalog policy network, Gymnasium environment, action masking,
supervised next-course policy pretraining, PPO clipping, GAE, Actor–Critic
checkpoints, baselines, ablations, reward sensitivity, multi-seed evaluation,
candidate-set mode, translation of final top-K only, and a nine-section
Streamlit application.

Paper-aligned extensions additionally include train-only mean imputation,
stratified random splitting, course-ID hashing, configurable outlier removal,
difficulty normalization when labels exist, combined semantic/behavioral
Mahalanobis representations, Euclidean similarity sensitivity, PPO cosine
scheduling and early stopping, five-seed mean ± standard-deviation reporting,
efficiency profiling, and EDA/statistical artifacts.

## Full temporal experiment

The verified run used 706 courses, 2,000 activity users, and 11,455 derived
course-level interactions. BERT was fine-tuned for three epochs from 6,914
course triplets:

```text
BERT triplet loss: 0.09703 → 0.04374 → 0.03435
Total BERT parameters: 177,853,440
Trainable BERT parameters: 28,942,080
```

The catalog policy used 6,914 supervised transitions, 30 supervised pretraining
epochs, and 100 PPO iterations. Evaluation covered 490 temporal test users.

```text
K    Precision    Recall     F1       MRR      NDCG
5    0.02776      0.12585    0.04495  0.06456  0.07760
10   0.02306      0.20748    0.04100  0.07690  0.10474
20   0.01602      0.27680    0.02996  0.08229  0.12306
```

These are experimental results, not paper-reported results.

## Baselines at K=10

```text
Model                         Precision  Recall   MRR
Popularity                    0.01980    0.17534  0.07210
BERT cosine                   0.02388    0.20827  0.08849
Mahalanobis                   0.02020    0.18170  0.08431
Actor–Critic + PPO + fusion   0.02306    0.20748  0.07690
Random                        0.00184    0.01548  0.00500
```

The full policy exceeds popularity but remains slightly below the pure BERT
baseline on some ranking metrics. It has a different coverage/diversity tradeoff.

## Ablation at K=10

```text
Full                          Recall 0.20748
Without PPO                   Recall 0.17571
Without BERT                  Recall 0.15306
Without Mahalanobis           Recall 0.20748
Without cluster affinity      Recall 0.20544
Candidate set                 Recall 0.20646
```

BERT and PPO make material contributions. Mahalanobis and clustering provide
small ranking changes on this public-data reconstruction.

## Reward and seed sensitivity

The paper-aligned `(0.4, 0.4, 0.2)` reward weights performed best among the five
tested combinations, with Recall@10 `0.20748`.

Three-seed Recall@10 values were:

```text
seed 42: 0.20748
seed 52: 0.16973
seed 62: 0.15340
```

This variance is reported rather than hidden.

Raw bounded rewards outperformed z-score reward normalization. Advantage
normalization remains enabled. This also avoids treating Mahalanobis similarity
as reward normalization, an ambiguity in the paper.

## Random versus temporal split

The paper-style random smoke split reached Precision@10 `0.0460` and
Recall@10 `0.2043` on 50 users. It is reported separately because random
splitting can leak future preference information. The main result uses temporal
evaluation.

## Remaining irreducible limitations

- The exact processed dataset used by the authors is not public.
- Public MOOCCube does not contain the paper-described observed quiz labels.
- Quiz, completion, and engagement are explicitly marked video-log proxies.
- The paper uses English-only `bert-base-uncased`; this implementation uses
  multilingual BERT because the available course text is Chinese.
- Logged data cannot provide counterfactual rewards for unseen actions, so PPO
  uses an explicit simulator for those actions.
- Consequently, the paper's reported Precision@10 `0.842` is not reproduced
  and must not be compared as if both experiments used identical data.
