"""
Core module for fairness experiments.

Provides data loading, training, evaluation, and statistics utilities.
"""

from .utils import set_seed, str2bool
from .data_loader import load_dataset
from .training import train_model
from .training_inproc import train_inprocessing_model
from .statistics import (
    compute_statistics,
    compute_k_wise_statistics,
    print_summary_table,
    print_enhanced_results_table,
    print_k_wise_summary,
    save_results,
    save_multi_lambda_results,
    create_experiment_info
)

__all__ = [
    # Utils
    'set_seed',
    'str2bool',
    
    # Data
    'load_dataset',
    
    # Training
    'train_model',
    'train_inprocessing_model',
    
    # Statistics
    'compute_statistics',
    'compute_k_wise_statistics',
    'print_summary_table',
    'print_enhanced_results_table',
    'print_k_wise_summary',
    'save_results',
    'save_multi_lambda_results',
    'create_experiment_info',
]