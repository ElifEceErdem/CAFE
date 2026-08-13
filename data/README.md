# Data

The GitHub reproducibility subset contains three graphs:

| Public name | Code name | Included content |
|---|---|---|
| Ego-Facebook | `ego-facebook` | raw edge list and processed PyG data |
| Oklahoma97 | `oklahoma` | raw archive/files, sensitive attribute, processed data |
| UNC28 | `unc` | raw archive/files, sensitive attribute, processed data |

The Oklahoma smoke test requires:

```text
data/SNAP/oklahoma/processed/data.pt
```

Verify all three processed files with:

```bash
make verify-data
```

Oklahoma97 and UNC28 can be restored using:

```bash
python data/SNAP/download_oklahoma_unc.py
```

The full paper also uses Pokec-n and Brightkite. They are intentionally not
distributed through GitHub because of their size. Existing local copies remain
ignored and are not deleted by repository cleanup.

Dataset files are governed by the terms of their original providers. The MIT
license at the repository root applies to the code, not automatically to the
datasets. Oklahoma, UNC, and Pokec include sensitive attributes; use and
redistribute them in accordance with their source terms and applicable policy.
