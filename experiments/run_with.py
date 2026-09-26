"""Run an experiment runner under the paper's evaluation protocol options.

    python experiments/run_with.py [--exact-closeness] [--split-seed S] \
        [--record-dir DIR] <runner>.py [runner arguments ...]

``--exact-closeness``  Evaluate Closeness EG with the exact generalized
                       closeness of ``metrics/closeness_exact_v1.py`` (used for
                       Brightkite and Pokec-n in Tables 4, 7, 11 and 15).
                       Degree and PageRank EG are unaffected.
``--split-seed S``     Draw the train/valid/test split with seed S instead of the
                       fixed split (the split-resampled protocol of Table 15:
                       seed s trains on split s).  Brightkite keeps its temporal
                       split regardless.
``--record-dir DIR``   Where to write provenance: the exact-closeness call log
                       (``exact_metric_calls.jsonl``) and the split fingerprint
                       (``split_evidence.json``).

The runner itself is unchanged; this wrapper reproduces what the September
validation campaigns (``reevaluate_closeness_v1.py``,
``prepare_table15_aligned.py``) did around it.
"""
import argparse
import contextlib
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def split_fingerprint(split):
    hashes = {}
    for name in ("train", "valid", "test"):
        for key in ("edge", "edge_neg"):
            tensor = split[name].get(key)
            if tensor is not None:
                array = tensor.detach().cpu().contiguous().numpy()
                digest = hashlib.sha256(str((str(array.dtype), array.shape)).encode())
                digest.update(array.tobytes())
                hashes[f"{name}/{key}"] = digest.hexdigest()
    return hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exact-closeness", action="store_true")
    ap.add_argument("--split-seed", type=int)
    ap.add_argument("--record-dir", type=Path)
    ap.add_argument("runner")
    ap.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "backbone"), str(ROOT / "fairness_methods"), str(ROOT)]
    record = args.record_dir
    if record:
        record.mkdir(parents=True, exist_ok=True)
    script = Path(args.runner)
    if not script.is_absolute():
        script = ROOT / "experiments" / script
    module_name = script.stem

    if args.split_seed is not None:
        import importlib
        from core.acceptance_probe import preserve_training_rng
        runner = importlib.import_module(module_name)
        original = runner.load_dataset

        def load(run_args):
            run_args.data_split_seed = args.split_seed
            result = original(run_args)
            if record:
                with preserve_training_rng():
                    evidence = dict(dataset=run_args.data_name, training_seed=run_args.seed,
                                    split_seed=args.split_seed, splits=split_fingerprint(result[1]))
                (record / "split_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
            return result
        runner.load_dataset = load

    context = contextlib.nullcontext()
    calls = []
    if args.exact_closeness:
        from metrics.closeness_exact_v1 import VERSION, activated

        def observer(entry):
            calls.append(entry)
            if record:
                with open(record / "exact_metric_calls.jsonl", "a") as fh:
                    fh.write(json.dumps(entry, allow_nan=False) + "\n")
        context = activated(observer)

    sys.argv = [str(script), *args.runner_args]
    os.chdir(ROOT / "experiments")
    with context:
        try:
            if args.split_seed is not None:
                sys.modules[module_name].main()
            else:
                runpy.run_path(str(script), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
    if args.exact_closeness:
        if not calls:
            raise SystemExit("exact closeness was requested but the runner evaluated no closeness")
        if record:
            (record / "exact_process_complete.json").write_text(
                json.dumps(dict(metric_version=VERSION, calls=len(calls)), indent=2) + "\n")


if __name__ == "__main__":
    main()
