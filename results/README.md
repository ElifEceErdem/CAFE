# Results

Generated experiment results normally live below this directory and are
ignored by Git to avoid accidentally committing large reruns.

The only certified release outputs are currently stored, without modification,
in the sibling directory:

```text
results_legacy_eg_check/
```

Those files are hash-locked by `configs/certified-artifacts.sha256`. Other
`results*` directories are retained as historical or diagnostic campaigns but
are not designated as certified paper evidence in this release.

Do not delete cached outputs during cleanup. New runs should use a new campaign
directory and record the configuration, data split, seeds, environment, and
Git revision alongside the result.
