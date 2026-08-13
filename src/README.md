# Source-layout note

A conventional `src/` migration is intentionally deferred. The certified
legacy experiments import the top-level `backbone`, `baselines`, `core`,
`fairness_methods`, and `metrics` packages directly. Relocating those modules
would alter executable paths and increase reproduction risk.

This directory documents that decision rather than presenting a second copy of
the scientific code.
