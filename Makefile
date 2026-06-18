PYTHON ?= python
CONFIG ?= configs/fast_demo.yaml

data:
	$(PYTHON) scripts/prepare_data.py --config $(CONFIG)

embeddings:
	$(PYTHON) scripts/extract_embeddings.py --config $(CONFIG)

train:
	$(PYTHON) scripts/train.py --config $(CONFIG)

evaluate:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG)

test:
	$(PYTHON) -m pytest -q

app:
	$(PYTHON) -m streamlit run app.py

demo: data embeddings train evaluate
	$(PYTHON) -m streamlit run app.py

full-smoke:
	$(PYTHON) scripts/run_pipeline.py --config configs/full_smoke.yaml

full:
	$(PYTHON) scripts/run_pipeline.py --config configs/full_experiment.yaml

eda:
	$(PYTHON) scripts/run_eda.py --config $(CONFIG)

profile:
	$(PYTHON) scripts/profile_efficiency.py --config $(CONFIG)

five-seed:
	$(PYTHON) scripts/run_ablation.py --config $(CONFIG) --five-seeds
