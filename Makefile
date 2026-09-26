PYTHON ?= python
SMOKE_OUTPUT ?= /tmp/cafe-oklahoma-smoke
MPLCONFIGDIR ?= /tmp/cafe-matplotlib

.PHONY: help smoke reproduce verify-paper verify-evidence test verify-certified verify-data check-release

help:
	@printf '%s\n' \
	  'make reproduce         Build every paper table and figure into build/' \
	  'make verify-paper      Build and compare every number with the manuscript' \
	  'make verify-evidence   Check the evidence snapshot checksums' \
	  'make test              Run the test suite' \
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

reproduce:
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m reproduce build

verify-paper:
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONDONTWRITEBYTECODE=1 CAFE_EVIDENCE_ONLY=1 $(PYTHON) -m reproduce check

verify-evidence:
	$(PYTHON) -m reproduce.tools.freeze_evidence --check

test:
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -q tests

verify-certified:
	sha256sum --check configs/certified-artifacts.sha256

verify-data:
	sha256sum --check data/processed.sha256

check-release: verify-certified verify-data verify-evidence verify-paper
	@test -f LICENSE
	@test -f README.md
	@test -f environment-smoke.yml
	@test -f paper/CAFE_ToRS_EceErdem.pdf
	@test -f data/SNAP/ego-facebook/processed/data.pt
	@test -f data/SNAP/oklahoma/processed/data.pt
	@test -f data/SNAP/unc/processed/data.pt
	@printf '%s\n' 'Release files and three-dataset public subset are present.'
