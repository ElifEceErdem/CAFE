"""Non-publication GPU integration tests; no predictions or checkpoints.

Two Oklahoma seed blocks exercise all five real method paths with two epochs.
Separate loader-only checks exercise every dataset and Brightkite's fixed split.
The success marker is written only after every check passes.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import prepare_table15_aligned as paired

PREPARED = paired.ROOT / 'results/validation/20260905_163304_table15_aligned_prepared'


def prepare(out):
    manifest = json.loads((PREPARED / 'authority.json').read_text())
    paired.verify(manifest)
    manifest.update(protocol='NON_PUBLICATION_table15_two_epoch_smoke', smoke_only=True,
                    epochs=2, purpose='Integration gate only; never use these metrics in paper tables')
    for path in [Path(__file__).resolve(), paired.ROOT / 'tests/test_table15_aligned.py',
                 paired.ROOT / 'experiments/jobs/t15_aligned_smoke.job', PREPARED / 'authority.json']:
        manifest['frozen_sha256'][str(path.relative_to(paired.ROOT))] = paired.sha(path)
    out.mkdir(parents=True, exist_ok=False)
    paired.write_new(out / 'authority.json', manifest)
    print(f'Prepared non-publication smoke test: {out}')


def loader_checks(out):
    # Each loader call is CPU-only and reads existing processed data; it does
    # not construct/swap graph files or change the canonical temporal split.
    sys.path[:0] = [str(paired.ROOT / 'backbone'), str(paired.ROOT)]
    from types import SimpleNamespace
    from core import set_seed
    from core.data_loader import load_dataset
    import torch
    torch.set_num_threads(4)
    evidence = {}
    for ds in paired.DATASETS:
        hashes = []
        for seed in [0, 1]:
            set_seed(seed)
            args = SimpleNamespace(data_name=ds, data_path=str(paired.ROOT / 'data/SNAP'),
                                   encoder='HLGNN', device='cpu', val_ratio=0.05, test_ratio=0.1,
                                   data_split_seed=seed, split_file=None)
            dataset, split, data, degrees = load_dataset(args)
            digest = hashlib.sha256()
            for name in ['train', 'valid', 'test']:
                digest.update(name.encode())
                digest.update(split[name]['edge'].cpu().numpy().tobytes())
            hashes.append(digest.hexdigest())
            del dataset, split, data, degrees
        if (hashes[0] == hashes[1]) != (ds == 'brightkite'):
            raise ValueError(f'Unexpected split resampling behavior: {ds}')
        evidence[ds] = dict(seed0_sha256=hashes[0], seed1_sha256=hashes[1],
                            fixed=(ds == 'brightkite'))
        print(f'LOADER PASS: {ds}; fixed={ds == "brightkite"}', flush=True)
    paired.write_new(out / 'loader_checks.json', evidence)


def run(out):
    manifest = json.loads((out / 'authority.json').read_text())
    paired.verify(manifest)
    loader_checks(out)
    blocks = []
    for seed in [0, 1]:
        task = dict(dataset='oklahoma', seed=seed, split_seed=seed)
        block = out / f'seed_{seed}'
        block.mkdir(exist_ok=False)
        observations = []
        for arm in paired.ARMS:
            paired.verify(manifest)
            folder = block / arm.lower()
            folder.mkdir()
            argv = paired.child_command(manifest, task, arm, folder)
            argv[argv.index('--epochs') + 1] = '2'
            if arm != 'RA':
                argv[argv.index('--eval_steps') + 1] = '1'
                argv[argv.index('--patience') + 1] = '2'
            paired.write_new(folder / 'smoke_launch.json', dict(argv=argv, smoke_only=True))
            print(f'SMOKE START: Oklahoma seed={seed} {arm}; two epochs, not paper evidence', flush=True)
            rc = paired.execute(argv)
            if rc:
                raise SystemExit(rc)
            observation = paired.collect(manifest, task, arm, folder)
            paired.write_new(folder / 'observation.json', observation)
            observations.append(observation)
        if not all(r['split_evidence'] == observations[0]['split_evidence'] for r in observations):
            raise ValueError(f'Method pairing failed in seed {seed}')
        paired.write_new(block / 'smoke_observations.json', dict(task=task, observations=observations,
                                                               pairing_verified=True, smoke_only=True))
        blocks.append(observations)
        print(f'PAIRING PASS: all five method paths, Oklahoma seed={seed}', flush=True)
    if blocks[0][0]['split_evidence']['splits'] == blocks[1][0]['split_evidence']['splits']:
        raise ValueError('Oklahoma split did not change between seed blocks')
    paired.verify(manifest)
    paired.write_new(out / 'smoke_pass.json', dict(status='PASS', smoke_only=True,
                     tested_methods=paired.ARMS, seeds=[0, 1], paired_blocks=2,
                     all_dataset_loader_checks=True, predictions_saved=False,
                     production_authority_sha256=paired.sha(PREPARED / 'authority.json')))
    print('SMOKE PASS: safe to consider the separate full campaign; do not publish smoke values', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    out = args.output_root.resolve()
    if not out.is_relative_to(paired.ROOT / 'results/validation'):
        raise ValueError('Use an isolated validation directory')
    prepare(out) if args.prepare else run(out)
