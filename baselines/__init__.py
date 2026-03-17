"""
Baseline models for link prediction comparison
"""

from .baseline_models import (
    HeuristicBaseline,
    RandomWalkEmbedding,
    PureGNN,
    RandomBaseline,
    TailSTEAKBaseline,
    LightGCNBaseline,
    SSNet
)

from .fairness_aware_baseline_models import (
    FairWalkEmbedding,
    CrossWalkEmbedding,
    UGEBaseline,
    FairAdj
)

from .baseline_wrapper import BaselineWrapper

__all__ = [
    'HeuristicBaseline',
    'RandomWalkEmbedding',
    'PureGNN',
    'RandomBaseline',
    'TailSTEAKBaseline',
    'FairWalkEmbedding',
    'CrossWalkEmbedding',
    'UGEBaseline',
    'BaselineWrapper',
    'LightGCNBaseline',
    'SSNet',
    'FairAdj'
]