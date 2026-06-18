# IntroAI — MOOCCube Course Recommendation

This repository implements a CPU-friendly course recommender inspired by the
paper *A hybrid actor–critic and BERT framework for intelligent course
recommendation in IoT-aware e-learning systems*.

The primary dataset is the real MOOCCube dump under `dataset/MOOCCube`.
The synthetic CSV files in `dataset/` are retained for reference but are not
the default source.

## What is implemented

- Streaming MOOCCube adapter with mojibake/HTML cleanup and user-ID hashing.
- Temporal and paper-style random splits.
- Ten learner features and a 768-dimensional semantic representation.
- A verified 778-dimensional Actor–Critic state.
- Actor–Critic training with normalized advantages.
- Semantic, Mahalanobis, and cluster-affinity score fusion.
- Precision, Recall, F1, MRR, and NDCG evaluation.
- Streamlit demo and CSV export.
- English translation only after the final top-K has been selected.

Fast demo uses deterministic 768-D hash embeddings so it works offline on CPU.
Full modes use the real local multilingual BERT checkpoint and MOOCCube video
activity. The hash backend proves the pipeline, but its metrics must not be
described as BERT-reproduced results.

## Downloaded models

- BERT: `D:\code\python\IntroAI\models\bert-base-multilingual-cased`
- Chinese-to-English translation:
  `D:\code\python\IntroAI\models\opus-mt-zh-en`

Both are loaded with `local_files_only=True`. After the initial download the
pipeline does not need Hugging Face network access.

## Windows quick start

```powershell
cd D:\code\python\IntroAI
..\.venv\Scripts\python.exe scripts\inspect_data.py
..\.venv\Scripts\python.exe scripts\prepare_data.py --config configs\fast_demo.yaml
..\.venv\Scripts\python.exe scripts\extract_embeddings.py --config configs\fast_demo.yaml
..\.venv\Scripts\python.exe scripts\train.py --config configs\fast_demo.yaml
..\.venv\Scripts\python.exe scripts\evaluate.py --config configs\fast_demo.yaml
..\.venv\Scripts\python.exe scripts\recommend.py --config configs\fast_demo.yaml --top-k 10
..\.venv\Scripts\python.exe -m streamlit run app.py
```

Run tests:

```powershell
..\.venv\Scripts\python.exe -m pytest -q
```

## Full experiment

```powershell
..\.venv\Scripts\python.exe scripts\download_models.py
..\.venv\Scripts\python.exe scripts\run_pipeline.py --config configs\full_smoke.yaml
```

The command above is the verified real-BERT smoke run. It includes MOOCCube
video-activity ETL, last-four-layer BERT fine-tuning, embedding extraction,
Actor–Critic/PPO training, negative sampling, and multi-K evaluation.

For the larger configuration:

```powershell
..\.venv\Scripts\python.exe scripts\prepare_data.py --config configs\full_experiment.yaml
..\.venv\Scripts\python.exe scripts\finetune_bert.py --config configs\full_experiment.yaml
..\.venv\Scripts\python.exe scripts\extract_embeddings.py --config configs\full_experiment.yaml
..\.venv\Scripts\python.exe scripts\train.py --config configs\full_experiment.yaml
..\.venv\Scripts\python.exe scripts\evaluate.py --config configs\full_experiment.yaml
```

Set `data.max_users: null` in `configs/full_experiment.yaml` to consume every
available MOOCCube activity user. This can take substantial time and storage.

Run individual research experiments:

```powershell
..\.venv\Scripts\python.exe scripts\train_ppo.py --config configs\full_experiment.yaml
..\.venv\Scripts\python.exe scripts\run_baselines.py --config configs\full_experiment.yaml
..\.venv\Scripts\python.exe scripts\run_ablation.py --config configs\full_experiment.yaml
..\.venv\Scripts\python.exe scripts\run_ablation.py --config configs\full_experiment.yaml --extended
..\.venv\Scripts\python.exe scripts\run_ablation.py --config configs\full_experiment.yaml --five-seeds
..\.venv\Scripts\python.exe scripts\run_eda.py --config configs\full_experiment.yaml
..\.venv\Scripts\python.exe scripts\profile_efficiency.py --config configs\full_experiment.yaml
```

Separate split configurations are available:

```text
configs/paper_random.yaml
configs/temporal_evaluation.yaml
```

Do not merge their metrics into one unlabeled series.

## Paper-aligned additions

The preprocessing pipeline now performs train-only mean imputation and Min-Max
fitting, configurable IQR outlier filtering, normalized difficulty labels when
observed, stable hashing of both user and course identifiers, and stratified
paper-random splitting by difficulty plus interaction-frequency bins.

The similarity module fits PCA-reduced BERT semantics jointly with ten
behavioral dimensions. Euclidean, cosine, and Mahalanobis experiments are
reported separately. PPO uses a cosine learning-rate scheduler and validation
early stopping. Five-seed ablation exports raw results and mean ± standard
deviation. EDA and efficiency scripts produce correlations, outlier summaries,
latency, memory, parameter counts, and convergence artifacts.

## Translation boundary

Source titles and descriptions remain Chinese throughout preprocessing,
embedding, training, and ranking. The local MarianMT translator is instantiated
only after the top-K indices are known. Exported recommendation CSVs contain
English fields plus the original title for auditability.

## Important limitations

- The exact paper PDF named in the assignment is not present in this workspace,
  so paper analysis is based on the supplied prompt and is explicitly labeled.
- Public MOOCCube enrollment records do not provide the paper's completion,
  quiz, and engagement feedback. Neutral reward proxies are marked
  `is_derived=true`; they are implementation scaffolding, not observed labels.
- Fast-demo hash embeddings are not BERT.
- Offline glossary translation is intentionally small. Use the configured
  MarianMT backend for fluent English when model download is available.
