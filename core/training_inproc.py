"""
Training logic specifically for in-processing fairness methods.

Handles fairness-aware loss computation and evaluation with advanced metrics.
"""

import copy
import torch
from tqdm import tqdm

from backbone.snap_evaluator import SNAPEvaluator
from .evaluation import compute_baseline_metrics


def train_inprocessing_model(model, data, split_edge, args, degrees, patience=50,
                             baseline_results=None, coreness=None):
    """
    Train model with in-processing fairness and comprehensive evaluation.

    Args:
        model: FairnessAwareModel instance
        data: Graph data object
        split_edge: Edge splits
        args: Arguments object
        degrees: Node degrees tensor
        patience: Early stopping patience
        baseline_results: Optional baseline results for comparison
        coreness: Optional coreness tensor

    Returns:
        model: Trained model
        best_epoch: Epoch of best model
        best_results: Full evaluation results at best epoch
    """
    evaluator = SNAPEvaluator()
    best_val_score, best_epoch, patience_counter = -1.0, 0, 0
    best_state = None

    # Determine validation metric
    metric_map = {
        'mrr': 'MRR',
        'precision': 'Precision@100',
        'fairness': 'Distribution_Fairness@100'
    }
    metric_name = metric_map.get(
        getattr(args, 'eval_metric', 'all'),
        'Precision@10'
    )

    print(f"\nTraining for up to {args.epochs} epochs (patience={patience})")
    print(f"Model selection based on: {metric_name}")
    print(f"Fairness method: {model.fairness_method}")
    if model.fairness_loss is not None:
        print(f"Lambda: {model.fairness_loss.lambda_fair}")

    # Progress bar
    pbar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")

    for epoch in pbar:
        # Training step
        if hasattr(model, 'train_with_fairness') and model.fairness_loss is not None:
            loss, base_loss, fair_penalty = model.train_with_fairness(
                data, split_edge,
                batch_size=args.batch_size,
                neg_sampler_name=args.neg_sampler,
                num_neg=args.num_neg,
                degrees=degrees,
                coreness=coreness
            )

            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'base': f'{base_loss:.4f}',
                'fair': f'{fair_penalty:.4f}',
                'best_epoch': best_epoch,
                'patience': f'{patience_counter}/{patience}'
            })
        else:
            # Standard training (baseline)
            loss = model.train(
                data, split_edge,
                batch_size=args.batch_size,
                neg_sampler_name=args.neg_sampler,
                num_neg=args.num_neg
            )

            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'best_epoch': best_epoch,
                'patience': f'{patience_counter}/{patience}'
            })

        # Evaluation
        if epoch % args.eval_steps == 0:
            results = model.test(
                data, split_edge,
                batch_size=args.batch_size,
                evaluator=evaluator,
                eval_metric='precision10',
                fairness_mode=args.fairness_mode,
                eval_chunk_size=getattr(args, 'eval_chunk_size', 1024),
                eval_batch_cands=getattr(args, 'eval_batch_cands', 4096)
            )

            val_score = results[metric_name][0]

            # Check for improvement
            if val_score > best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                best_state = {
                    'encoder': copy.deepcopy(model.encoder.state_dict()),
                    'predictor': copy.deepcopy(model.predictor.state_dict()),
                    'emb': copy.deepcopy(model.emb.state_dict()) if model.emb is not None else None
                }
                patience_counter = 0

                if model.fairness_loss is not None:
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'base': f'{base_loss:.4f}',
                        'fair': f'{fair_penalty:.4f}',
                        metric_name: f'{val_score:.4f}',
                        'status': ' NEW BEST'
                    })
                else:
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        metric_name: f'{val_score:.4f}',
                        'status': ' NEW BEST'
                    })
            else:
                patience_counter += 1

                if model.fairness_loss is not None:
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'base': f'{base_loss:.4f}',
                        'fair': f'{fair_penalty:.4f}',
                        metric_name: f'{val_score:.4f}',
                        'best': f'{best_val_score:.4f}',
                        'patience': f'{patience_counter}/{patience}'
                    })
                else:
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        metric_name: f'{val_score:.4f}',
                        'best': f'{best_val_score:.4f}',
                        'patience': f'{patience_counter}/{patience}'
                    })

            # Early stopping
            if patience_counter >= patience:
                pbar.close()
                print(f"\n Early stopping at epoch {epoch} (best: {best_epoch})")
                break

    pbar.close()

    # Load best model
    if best_state is not None:
        model.encoder.load_state_dict(best_state['encoder'])
        model.predictor.load_state_dict(best_state['predictor'])
        if best_state['emb'] is not None and model.emb is not None:
            model.emb.load_state_dict(best_state['emb'])
        print(f"\n Loaded best model from epoch {best_epoch}")
        print(f" Best validation {metric_name}: {best_val_score:.4f}")

    # Comprehensive evaluation at best epoch
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EVALUATION AT BEST EPOCH")
    print(f"{'='*80}")

    final_results = compute_baseline_metrics(model, data, split_edge, args)

    return model, best_epoch, final_results
