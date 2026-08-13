# Release configurations

`best_configs.csv` is a byte-identical copy of the configuration file used by
the certified `results_legacy_eg_check` campaign.

- SHA-256: `44ea8142fa6e953dc2c79fbe3d5b6ae5e4eb77ad08a3a72636db31942e9c4ade`
- Evaluation cutoff recorded by the campaign adapter: K=10
- Penalty reduction recorded by the certified CAFE runs: `mean`
- Equality-Gain result marker: `legacy_indegree_reconstructed`

This file is frozen provenance. Do not regenerate it from another results tree
or replace it with `results_repro/tables/best_configs.csv`; that table-oriented
file has different contents.

`certified-artifacts.sha256` records the immutable public-release files. Run
`make verify-certified` from the repository root to check them.
