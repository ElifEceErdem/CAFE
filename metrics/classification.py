"""
Classification metrics for link prediction, adapted from PyGDebias.

Implements:
  - AUCROC, F1, Accuracy (standard binary classification metrics)
  - Rawlsian Degree Bias (variance of per-node loss across degree groups)

Reference: PyGDebias/pygdebias/debiasing/RawlsGCN.py (Evaluator class)
  - _get_accuracy  (L358-362)
  - _get_bias      (L378-392)

PyGDebias uses:
  - sklearn.metrics.roc_auc_score   (metrics/metrics.py:10, debiasing/GNN.py)
  - sklearn.metrics.f1_score        (debiasing/GNN.py, average='micro' for multiclass)
  - Manual accuracy                 (RawlsGCN.py:358-362)
  - Bias = Variance(Mean_Loss_per_Degree)  (RawlsGCN.py:378-392)

Adaptation notes (Node Classification -> Link Prediction):
  - Classification metrics (AUC/F1/Acc): Binary classification on test edges
    (positive edges = label 1, negative edges = label 0).
  - Rawlsian Bias: Per-node mean loss aggregated from incident test edges,
    then grouped by degree. Returns variance of group means.
    This preserves the node-as-unit-of-fairness semantics from RawlsGCN.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from collections import defaultdict
import statistics


def compute_classification_metrics(pos_scores, neg_scores):
    """
    Compute AUCROC, F1, and Accuracy for link prediction.

    Following PyGDebias convention:
      - AUC:  sklearn.metrics.roc_auc_score(y_true, y_prob)
      - F1:   sklearn.metrics.f1_score(y_true, y_pred, average='binary')
      - Acc:  sklearn.metrics.accuracy_score(y_true, y_pred)

    Args:
        pos_scores: Tensor of predicted scores for positive (true) edges.
        neg_scores: Tensor of predicted scores for negative edges.

    Returns:
        dict with keys 'AUCROC', 'F1', 'Accuracy', each mapped to (0.0, value).
    """
    # Flatten and move to numpy
    pos_np = pos_scores.detach().cpu().numpy().flatten()
    neg_np = neg_scores.detach().cpu().numpy().flatten()

    scores = np.concatenate([pos_np, neg_np])
    labels = np.concatenate([np.ones(len(pos_np)), np.zeros(len(neg_np))])

    # Guard against NaN scores (e.g. from numerical instability in VAE training)
    if np.any(np.isnan(scores)):
        nan_count = np.isnan(scores).sum()
        print(f"[WARNING] {nan_count}/{len(scores)} NaN scores detected in classification metrics. "
              f"Replacing NaN with 0.0. This likely indicates numerical instability during training.")
        scores = np.nan_to_num(scores, nan=0.0)

    # Apply sigmoid if scores appear to be logits (values outside [0, 1])
    if scores.min() < -0.01 or scores.max() > 1.01:
        scores = _sigmoid(scores)

    # Binary predictions with threshold = 0.5
    preds = (scores >= 0.5).astype(np.int32)

    # Compute metrics
    auc = roc_auc_score(labels, scores)
    f1 = f1_score(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)

    return {
        'AUCROC': (0.0, float(auc)),
        'F1': (0.0, float(f1)),
        'Accuracy': (0.0, float(acc)),
    }


def compute_rawlsian_degree_bias(recs, pos_test_edge, degrees, num_nodes, k=10):
    """
    Compute Rawlsian Degree Bias using Per-Node Precision@K.

    Principled Adaptation for Link Prediction:
    1.  Compute Precision@K for each node u in the test set.
        (Precision@K = Fraction of top-K recommendations that are true positive edges).
    2.  Group nodes by degree d.
    3.  Compute mean Precision@K for each degree group.
    4.  Rawlsian Bias = Variance of these group means.

    This preserves the node-as-unit-of-fairness semantics from RawlsGCN (RawlsGCN.py)
    while using a metric (Precision@K) appropriate for link prediction.

    Args:
        recs: Dict {user_id: [top_k_items]} or equivalent recommendations.
        pos_test_edge: (N_test, 2) tensor of positive test edges.
        degrees: (num_nodes,) tensor of node degrees.
        num_nodes: Total number of nodes.
        k: K value for Precision@K (default 10).

    Returns:
        float: Rawlsian degree bias (variance of mean Precision@K across degree groups).
    """
    # 1. Compute Per-Node Precision@K
    # Build ground truth sets for fast lookup
    test_adj = defaultdict(set)
    pos_edge_np = pos_test_edge.cpu().numpy()
    for u, v in pos_edge_np:
        test_adj[int(u)].add(int(v))
        test_adj[int(v)].add(int(u))  # Undirected

    node_precision = {}
    users_with_test_edges = []
    
    test_nodes = set(test_adj.keys())

    for u in test_nodes:
        if u not in recs:
            node_precision[u] = 0.0
            continue
            
        my_recs = recs[u][:k]
        if not my_recs:
            node_precision[u] = 0.0
            continue
        hits = 0
        for v in my_recs:
            if v in test_adj[u]:
                hits += 1
        
        node_precision[u] = hits / len(my_recs)
    
    # 2. Group by Degree
    deg_np = degrees.cpu().numpy().flatten()
    precision_by_deg = defaultdict(list)
    
    for u, precision in node_precision.items():
        if u >= len(deg_np): continue
        d = int(deg_np[u])
        precision_by_deg[d].append(precision)
        
    # 3. Compute Group Means
    group_means = []
    for d, precisions in precision_by_deg.items():
        group_means.append(statistics.mean(precisions))
        
    # 4. Compute Variance
    if len(group_means) > 1:
        bias = statistics.variance(group_means)
    else:
        bias = 0.0
        
    return bias


def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )
