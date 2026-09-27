PYTHON ?= python
MPLCONFIGDIR ?= /tmp/cafe-matplotlib
export MPLCONFIGDIR
export PYTHONDONTWRITEBYTECODE = 1

.PHONY: help tables test smoke verify-data

help:
	@printf '%s\n' \
	  'make tables       Build every table and figure of the paper into build/' \
	  'make test         Run the tests' \
	  'make smoke        Run Resource Allocation on Oklahoma (CPU, ~30 s)' \
	  'make verify-data  Check the included datasets'

tables:
	$(PYTHON) -m tables

test:
	$(PYTHON) -m pytest -q tests

smoke:
	@test -f data/SNAP/oklahoma/processed/data.pt
	$(PYTHON) -u experiments/run_baselines.py --data_name oklahoma --data_path $(CURDIR)/data/SNAP \
		--methods ra --runs 1 --epochs 1 --batch_size 16384 \
		--use_equality_gain True --use_centrality_gain True --centrality_types pagerank,closeness \
		--results_dir /tmp/cafe-smoke

verify-data:
	sha256sum --check data/processed.sha256
