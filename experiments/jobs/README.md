# Table-indexed job suite

One SLURM job per manuscript table, at the **published** configurations
(`results_repro/tables/best_configs.csv`, mirrored in `_common.sh` and
`experiments/tables/_framework.py` — change all three together).

```bash
cd experiments/jobs
bash submit_all.sh                  # everything, aggregation chained after
sbatch t04_cafe.job                 # one job, all five datasets
sbatch --array=1 t07_ablation.job   # oklahoma only
sbatch --array=0,4 t11_exposure.job # the two datasets Table 6(b) is missing
```

Array index → dataset: `0=ego-facebook 1=oklahoma 2=unc 3=pokec_n 4=brightkite`.
`t10_closeness.job` is `0-2` only (Ego-FB, Oklahoma, UNC).

## Job → table map

| Job | Tables / figures | GPU-hours (rough) |
|---|---|---|
| `t04_cafe.job` | 4, 7, 8 — 7 arms per dataset | ~55 (Pokec-n dominates) |
| `t04_baselines.job` | 4, 8 — 10–13 baselines per dataset | ~120 |
| `t07_ablation.job` | 7 — CI-In, CI-Pre, MB-Post | ~25 |
| `t05_stratified.job` | 5, 6(a), 18, Fig. 4 | ~15 |
| `t11_exposure.job` | 11, 6(b) | ~55 |
| `t16_significance.job` | 16, 17 | ~55 |
| `t10_closeness.job` | 10 | ~3 |
| `t12_runtime.job` | 12 | ~10, `--exclusive` |
| `cpu_analyses.job` | 2, 9 | CPU only |
| `build_tables.job` | verify + build all | CPU only |

Estimates come from Table 12's per-run training times × arms × 10 seeds; treat
them as an ordering, not a budget.

## Things that will bite you

**Runner flag sets are not interchangeable.** `run_cafe.py` and
`run_significance.py` accept the reduced-validation flags for the large graphs
(`${EXTRA}`); `run_cafe_p.py`, `run_coreness_inverse.py`,
`run_mobility_boost.py` and `run_baselines.py` do not — they get `${SPLIT}`.
`run_significance.py` takes `--num_seeds`, everything else takes `--runs`.
`run_baselines.py`'s `--results_dir` is the tree **root**, not a per-dataset
directory, and its method names are lower-case registry keys.

**Brightkite always uses `data/SNAP/brightkite/temporal_split.pt`.** A random
split there silently destroys the proxy-temporal ordering the whole Brightkite
narrative rests on.

**`t12_runtime.job` requests `--exclusive` on purpose.** The currently stored
Pokec-n timings (8797 ± 351 s, visibly bimodal) do not match the published row
(8540 ± 16 s); that is what a contended GPU looks like, and it invalidates the
1.00× / 1.02–1.07× overhead claims of §5.5.

**`t11_exposure.job` also produces Table 6(b).** The exposure block used to be
gated on the dataset having a sensitive attribute, which is why the draft marks
the target-side audit unavailable on Ego-FB and Brightkite. The gate is gone —
the audit pools groups and never needed an attribute. Only ΔSP does.

**The cluster is heterogeneous — check where your jobs land.**

| Feature | Nodes | GPU | Memory |
|---|---|---|---|
| `TitanRTX` | node206, node219, node220, node221 | TITAN RTX | 24 GB |
| `GTX1080Ti` | node201, node203, node205 | GTX 1080 Ti | 11 GB |
| `TitanX` | node202 | GTX TITAN X | 12 GB |
| `TitanXpascal` | node204 | TITAN X (Pascal) | 12 GB |

Training *results* are device-independent (up to the cuDNN nondeterminism
already absorbed by the reported ±std), so Tables 4/7/8/16 do not care. **Table
12 does**: `t12_runtime.job` pins `--constraint=TitanRTX` because Appendix A.2
names that device and a mixed-hardware runtime table is meaningless.

To see what a running job actually got:

```bash
srun --overlap --jobid=<id> bash -lc 'source ~/.bashrc; conda activate hl_gnn;
  export LD_LIBRARY_PATH=/cm/local/apps/cuda-driver/libs/current/lib64:$LD_LIBRARY_PATH;
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"'
```

`nvidia-smi` is not on the compute nodes' PATH; ask torch instead. Note also
that `scontrol show job` reports the GPU under `JOB_GRES` / `TresPerNode`, not
under `AllocTRES` — an empty `AllocTRES` gres field does **not** mean the job
is CPU-only.

**Attribute baselines on Pokec-n need memory.** FairAdj and LBA build N×N
identity features (66,569² floats ≈ 17.7 GB), which OOMs on anything but a
TITAN RTX:
`sbatch --constraint=TitanRTX --array=3 t04_baselines.job`.

## After the jobs finish

`build_tables.job` runs `--stage verify` before `--stage build`. Read the
verify report: it fails any table whose result JSONs lack
`eg_graph_semantics = simple_undirected_set_union`, i.e. whose Degree- and
PageRank-EG columns came from the metric superseded on 2026-07-23. Mixing
marked and unmarked sources inside one EG column is not comparable.
