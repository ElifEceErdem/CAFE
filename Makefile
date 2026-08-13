PYTHON ?= python
SMOKE_OUTPUT ?= /tmp/cafe-oklahoma-smoke
MPLCONFIGDIR ?= /tmp/cafe-matplotlib

.PHONY: help smoke verify-certified verify-data check-release

help:
	@printf '%s\n' \
	  'make smoke             Run the Oklahoma RA CPU smoke test' \
	  'make verify-certified  Verify the paper, configs, and certified results' \
	  'make verify-data       Verify the three processed public datasets' \
	  'make check-release     Run non-training release checks'

smoke:
	@test -f data/SNAP/oklahoma/processed/data.pt
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -u experiments/run_baselines.py \
		--data_name oklahoma \
		--data_path data/SNAP \
		--methods ra \
		--runs 1 \
		--epochs 1 \
		--batch_size 16384 \
		--use_equality_gain True \
		--use_centrality_gain True \
		--centrality_types pagerank,closeness \
		--results_dir $(SMOKE_OUTPUT)

verify-certified:
	sha256sum --check configs/certified-artifacts.sha256

verify-data:
	sha256sum --check data/processed.sha256

check-release: verify-certified verify-data
	@test -f LICENSE
	@test -f README.md
	@test -f environment-smoke.yml
	@test -f paper/CAFE_ToRS_EceErdem.pdf
	@test -f data/SNAP/ego-facebook/processed/data.pt
	@test -f data/SNAP/oklahoma/processed/data.pt
	@test -f data/SNAP/unc/processed/data.pt
	@printf '%s\n' 'Release files and three-dataset public subset are present.'
