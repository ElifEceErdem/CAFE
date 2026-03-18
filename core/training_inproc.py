"""
Training logic specifically for in-processing fairness methods.

Handles fairness-aware loss computation and evaluation with advanced metrics.
"""

import copy
import torch
from tqdm import tqdm
import sys
import os

from backbone.snap_evaluator import SNAPEvaluator
from .evaluation import compute_baseline_metrics, compute_fairness_metrics_with_advanced
from fairness_methods.inprocessing import dropout_adj_coreness

def train_inprocessing_model(model, data, split_edge, args, degrees, patience=50, 
                             baseline_results=None, coreness=None,
                             use_curriculum=False, warmup_epochs=50, max_curriculum_epochs=100):
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
    
    Returns:
        model: Trained model
        best_epoch: Epoch of best model
        best_results: Full evaluation results at best epoch
    """
    evaluator = SNAPEvaluator()
    best_val_score, best_epoch, patience_counter = -1.0, 0, 0
    best_state, best_all_results = None, None
    
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
    
    # Store target lambda for curriculum
    target_lambda = 0.0
    if model.fairness_loss is not None:
        target_lambda = model.fairness_loss.lambda_fair

    for epoch in pbar:
        # --- Curriculum Learning Update ---
        current_lambda = target_lambda
        if use_curriculum and model.fairness_loss is not None:
            if epoch < warmup_epochs:
                current_lambda = 0.0
            elif epoch < max_curriculum_epochs:
                # Linear ramp-up
                progress = (epoch - warmup_epochs) / (max_curriculum_epochs - warmup_epochs)
                current_lambda = target_lambda * progress
            else:
                current_lambda = target_lambda
            
            # Update mode lambda
            model.fairness_loss.lambda_fair = current_lambda
        
        # Check for Coreness-Aware DropEdge
        # FIX: Check model.fairness_method, not args.fairness_method.
        # This ensures Baseline (method='none') doesn't trigger this.
        if model.fairness_method == 'core_dropedge' and coreness is not None:
             # Apply dropout to edge_index for this epoch
             # Note: data.edge_index is used by get_pos_neg_edges (for training pairs)
             # AND data.adj_t might be used by GNN (for message passing)
             # If GNN uses adj_t (SparseTensor), we need to update it too or rebuild it.
             # HLGNN uses adj_t.
             
             # Drop edges from the FULL edge_index (used for message passing)
             # But wait, data.adj_t is SparseTensor. 
             # Re-building SparseTensor every epoch is expensive?
             # Let's drop from edge_index key and assume loader rebuilt adj_t or we rebuild it.
             # Actually, simpler: drop from data.edge_index, then re-create adj_t if allowed.
             # BUT 'data' is shared. We should clone or modify a temporary view.
             
             # Optimization: Do this only if we really support dynamic graph structure in HLGNN.
             # If too complex, maybe skip for now or use a simple mask on adj_t values?
             # Let's assume we pass the DROPPED adj to the model forward pass.
             
             # For now, let's implement a simplified version: 
             # If core_dropedge, we just use the custom function to drop from edge_index
             # And we need to convert that to adj_t for HLGNN.
             
             train_edge_index = data.edge_index
             dropped_edge_index = dropout_adj_coreness(train_edge_index, coreness, p=float(args.lambda_fair))
             
             # Rebuild SparseTensor from dropped edges
             # Lazy import
             from torch_sparse import SparseTensor
             from backbone.utils import gcn_normalization  # Ensure this is available from SNAP/utils.py
             
             dropped_adj_t = SparseTensor(row=dropped_edge_index[1], col=dropped_edge_index[0], 
                                          sparse_sizes=(data.num_nodes, data.num_nodes)).to(data.edge_index.device)
             
             # Normalize!
             dropped_adj_t = gcn_normalization(dropped_adj_t)
             
             # We need to temporarily patch data.adj_t for the model call
             # Or pass it explicitly. train_with_fairness calls create_input_feat which uses data.adj_t?
             # Model.encoder forwarding uses data.adj_t.
             # Let's monkey-patch data object strictly for this epoch/step.
             original_adj_t = data.adj_t
             data.adj_t = dropped_adj_t
        else:
             original_adj_t = None

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
            
            # Restore original adj_t if we modified it
            if original_adj_t is not None:
                data.adj_t = original_adj_t
            
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'base': f'{base_loss:.4f}',
                'fair': f'{fair_penalty:.4f}',
                'best_epoch': best_epoch,
                'patience': f'{patience_counter}/{patience}',
                'lam': f'{current_lambda:.4f}'
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
                best_all_results = results
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
                        'status': '✓ NEW BEST'
                    })
                else:
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        metric_name: f'{val_score:.4f}',
                        'status': '✓ NEW BEST'
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
                print(f"\n✓ Early stopping at epoch {epoch} (best: {best_epoch})")
                break
    
    pbar.close()
    
    # Load best model
    if best_state is not None:
        model.encoder.load_state_dict(best_state['encoder'])
        model.predictor.load_state_dict(best_state['predictor'])
        if best_state['emb'] is not None and model.emb is not None:
            model.emb.load_state_dict(best_state['emb'])
        print(f"\n✓ Loaded best model from epoch {best_epoch}")
        print(f"✓ Best validation {metric_name}: {best_val_score:.4f}")
    
    # Comprehensive evaluation at best epoch
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EVALUATION AT BEST EPOCH")
    print(f"{'='*80}")
    
    # For in-processing fairness, we should NOT apply post-processing re-ranking
    # The fairness was already applied during training via the loss function
    # Just compute baseline metrics which includes all advanced metrics
    # (Equality Gain, Stratified metrics, NDKL, PA-aware, etc.)
    
    # Always call compute_baseline_metrics() for in-processing methods
    # This computes all metrics without any post-processing re-ranking
    final_results = compute_baseline_metrics(model, data, split_edge, args)
    
    return model, best_epoch, final_results