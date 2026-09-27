"""Generate Table 2 (Section 6.1): dataset statistics.

Nodes, undirected edges, average degree, average clustering coefficient and
the sensitive attribute each network carries.

Computed through the SAME loader the experiments use (``core.load_dataset``),
so |V| is the graph after isolated-node removal -- the number Appendix A.1's
evaluation-population counts are consistent with.  Reading the raw SNAP files
instead would give Oklahoma 3,111 and UNC 4,018 and contradict the appendix.

Statistics are over the full graph (train + validation + test edges), not the
training split: Table 2 describes the dataset, and every structural quantity a
*method* sees is computed on G_train elsewhere.

Average clustering is exact, via a sparse triangle count
(C(v) = 2 T(v) / (d(v)(d(v)-1)), nodes of degree < 2 contributing 0), which is
the standard definition and avoids a networkx traversal on the large graphs.

Usage:
    python dataset_statistics.py --data_path ../data/SNAP \
        --out ../results/summaries/dataset_table.tex
"""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

CURRENT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(CURRENT, ".."))
for path in [REPO, os.path.join(REPO, "backbone")]:
    if path not in sys.path:
        sys.path.insert(0, path)

from core import load_dataset  # noqa: E402
from core.data_loader import load_sensitive_attr  # noqa: E402

DATASET_ORDER = ["ego-facebook", "oklahoma", "unc", "pokec_n", "brightkite"]
DATASET_LABELS = {
    "ego-facebook": "Ego-FB", "oklahoma": "Oklahoma", "unc": "UNC",
    "pokec_n": "Pokec-n", "brightkite": "Brightkite",
}
ATTR_LABELS = {"oklahoma": "Gen.", "unc": "Gen.", "pokec_n": "Reg."}


def undirected_adjacency(edges, num_nodes):
    """Binary, symmetric, zero-diagonal adjacency from an [E, 2] edge array."""
    e = np.asarray(edges)
    if e.ndim == 2 and e.shape[0] == 2 and e.shape[1] != 2:
        e = e.T  # tolerate a [2, E] layout
    r, c = e[:, 0], e[:, 1]
    keep = r != c
    r, c = r[keep], c[keep]
    # int32, not a narrow dtype: entries of A@A count common neighbours and
    # exceed 127 on the hub-heavy graphs, which would silently overflow.
    data = np.ones(len(r), dtype=np.int32)
    a = sp.coo_matrix((data, (r, c)), shape=(num_nodes, num_nodes)).tocsr()
    a = a + a.T
    a.data[:] = 1
    a.setdiag(0)
    a.eliminate_zeros()
    return a


def clustering(a):
    """Return (average local clustering, global transitivity).

    Local:  mean_v 2 T(v) / (d(v)(d(v)-1)), nodes of degree < 2 counted as 0
            -- the networkx ``average_clustering`` convention.
    Global: 3 * triangles / connected triples -- networkx ``transitivity``.

    The published Table 2 column matches the GLOBAL figure (Ego-FB 0.52),
    not the local one (0.61); both are returned so the caption can say which.
    """
    deg = np.asarray(a.sum(axis=1)).ravel().astype(np.float64)
    # triangles(v) = (A^3)_vv / 2, via A@A restricted to A's sparsity pattern
    tri = np.asarray(a.multiply(a @ a).sum(axis=1)).ravel() / 2.0
    denom = deg * (deg - 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        local = np.where(denom > 0, 2.0 * tri / denom, 0.0)
    triples = denom.sum() / 2.0
    global_cc = float(tri.sum() / triples) if triples > 0 else 0.0
    return float(local.mean()), global_cc


def stats_for(ds, data_path, split_seed=42):
    args = SimpleNamespace(
        data_name=ds, data_path=data_path, val_ratio=0.05, test_ratio=0.1,
        encoder="HLGNN", device=0, split_file=None, data_split_seed=split_seed)
    dataset, split_edge, data, _ = load_dataset(args)
    num_nodes = (dataset._data.num_nodes if hasattr(dataset, "_data")
                 else dataset.data.num_nodes)

    parts = []
    for key in ("train", "valid", "test"):
        if key in split_edge and "edge" in split_edge[key]:
            e = split_edge[key]["edge"]
            parts.append(e.cpu().numpy() if hasattr(e, "cpu") else np.asarray(e))
    edges = np.concatenate(parts, axis=0)

    a = undirected_adjacency(edges, num_nodes)
    num_edges = int(a.nnz // 2)
    deg = np.asarray(a.sum(axis=1)).ravel()

    try:
        load_sensitive_attr(ds, num_nodes, root=data_path)
        attr = ATTR_LABELS.get(ds, "yes")
    except (ValueError, FileNotFoundError):
        attr = "---"

    local_cc, global_cc = clustering(a)
    return dict(dataset=ds, num_nodes=num_nodes, num_edges=num_edges,
                avg_degree=float(deg.mean()),
                avg_clustering=local_cc, transitivity=global_cc, attr=attr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_path", default="../data/SNAP")
    ap.add_argument("--datasets", default=",".join(DATASET_ORDER))
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--out", default="../results/summaries/dataset_table.tex")
    ap.add_argument("--clustering", choices=["global", "local"], default="global",
                    help="which clustering coefficient the table prints; "
                         "'global' (transitivity) matches the published Table 2")
    args = ap.parse_args()

    rows = []
    for ds in [d.strip() for d in args.datasets.split(",")]:
        print(f"  {ds} ...", flush=True)
        rows.append(stats_for(ds, args.data_path, args.split_seed))
        r = rows[-1]
        print(f"    |V|={r['num_nodes']:,}  |E|={r['num_edges']:,}  "
              f"avg deg={r['avg_degree']:.2f}  "
              f"transitivity={r['transitivity']:.2f}  "
              f"avg local cc={r['avg_clustering']:.2f}")

    lines = [
        r"\begin{table}[t]",
        r"\caption{Dataset statistics. $|V|$: number of nodes; $|E|$: number of "
        r"undirected edges; ``Attr.'': sensitive attribute (Gen.\ = gender, "
        r"Reg.\ = region). Counts are for the loaded graph, after isolated-node "
        r"removal.}",
        r"\label{tab:datasets}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrc}",
        r"\toprule",
        r"Dataset & $|V|$ & $|E|$ & Avg. deg. & Avg. cluster. & Attr. \\",
        r"\midrule",
    ]
    csv_rows = ["dataset,num_nodes,num_edges,avg_degree,transitivity,"
                "avg_local_clustering,attribute"]
    key = "transitivity" if args.clustering == "global" else "avg_clustering"
    for r in rows:
        lines.append(
            f"{DATASET_LABELS.get(r['dataset'], r['dataset'])} & "
            f"{r['num_nodes']:,} & {r['num_edges']:,} & "
            f"{r['avg_degree']:.2f} & {r[key]:.2f} & {r['attr']} \\\\")
        csv_rows.append(
            f"{r['dataset']},{r['num_nodes']},{r['num_edges']},"
            f"{r['avg_degree']:.4f},{r['transitivity']:.4f},"
            f"{r['avg_clustering']:.4f},{r['attr']}")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w") as fh:
        fh.write("\n".join(csv_rows) + "\n")
    print(f"  wrote {args.out}")
    print(f"  wrote {csv_path}")


if __name__ == "__main__":
    main()
