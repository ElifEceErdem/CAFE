# Scripts

The scientific entry points remain under `experiments/` and `analysis/` in this
release. Moving them would change historical imports and job paths, so the safe
cleanup preserves those locations.

The root `Makefile` supplies the reviewer-facing commands:

- `make smoke`
- `make verify-certified`
- `make check-release`

Future wrappers may be added here without relocating the certified legacy
entry points.
