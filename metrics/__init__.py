"""
Metrics package for fairness experiments.

Includes:
- Precision and Distribution Fairness metrics
- PA-aware fairness metrics
- NDKL fairness metrics
"""

from .precision import compute_metrics_from_recs
from .fairness import distribution_fairness_from_recs
from .pa_aware import (
    compute_pa_fairness_from_overlap,
    generate_pa_recommendations,
    compute_overlap_at_k
)
from .ndkl import (
    compute_ndkl_metrics,
    get_evaluation_user_set,
    build_edge_scores_for_ndkl
)

__all__ = [
    'compute_metrics_from_recs',
    'distribution_fairness_from_recs',
    'compute_pa_fairness_from_overlap',
    'generate_pa_recommendations',
    'compute_overlap_at_k',
    'compute_ndkl_metrics',
    'get_evaluation_user_set',
    'build_edge_scores_for_ndkl',
]