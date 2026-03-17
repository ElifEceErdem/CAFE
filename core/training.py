# fairness_experiments_scaffold/core/training.py
import copy
import sys
import torch
from tqdm import tqdm
from snap_evaluator import SNAPEvaluator

def train_model(model, data, split_edge, args, patience=50):
    """Full training logic for BaseModel architecture with progress tracking."""
    evaluator = SNAPEvaluator()
    best_val_score, best_epoch, patience_counter = -1.0, 0, 0
    best_state, best_all_results = None, None

    metric_name = 'Precision@10'  # Default
    
    print(f"\nTraining for up to {args.epochs} epochs (patience={patience})...")
    print(f"Model selection based on: {args.eval_metric if hasattr(args, 'eval_metric') else 'Precision@10'}")
    
    # Create progress bar for epochs
    pbar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")
    
    for epoch in pbar:
        loss = model.train(data, split_edge, batch_size=args.batch_size,
                          neg_sampler_name=args.neg_sampler, num_neg=args.num_neg)
        
        # Update progress bar with current loss
        pbar.set_postfix({
            'loss': f'{loss:.4f}',
            'best_epoch': best_epoch,
            'patience': f'{patience_counter}/{patience}'
        })
        
        # Always run validation on the last epoch or evaluation steps
        if (epoch % args.eval_steps == 0) or (epoch == args.epochs):
            results = model.test(data, split_edge, batch_size=args.batch_size,
                               evaluator=evaluator, eval_metric='precision10', 
                               fairness_mode=args.fairness_mode)
            
            # Metric Selection for validation
            metric_map = {
                'mrr': 'MRR', 
                'precision': 'Precision@100', 
                'fairness': 'Distribution_Fairness@100'
            }
            metric_name = metric_map.get(
                getattr(args, 'eval_metric', 'all'), 
                'Precision@10'
            )
            val_score = results[metric_name][0]

            # Check for improvement
            if val_score > best_val_score:
                best_val_score, best_epoch, best_all_results = val_score, epoch, results
                best_state = {
                    'encoder': copy.deepcopy(model.encoder.state_dict()),
                    'predictor': copy.deepcopy(model.predictor.state_dict()),
                    'emb': copy.deepcopy(model.emb.state_dict()) if model.emb is not None else None
                }
                patience_counter = 0
                
                # Update progress bar with improvement
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    metric_name: f'{val_score:.4f}',
                    'status': '✓ NEW BEST',
                    'best_epoch': best_epoch
                })
            else:
                patience_counter += 1
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    metric_name: f'{val_score:.4f}',
                    'best': f'{best_val_score:.4f}',
                    'patience': f'{patience_counter}/{patience}'
                })
            
            # Early stopping check
            if patience_counter >= patience:
                pbar.close()
                sys.stdout.write('\n')  # force newline after tqdm \r
                sys.stdout.flush()
                print(f"✓ Early stopping at epoch {epoch} (best: {best_epoch})", flush=True)
                break
    
    pbar.close()
    sys.stdout.write('\n')  # force newline after tqdm \r
    sys.stdout.flush()
    
    # Load best model
    if best_state:
        model.encoder.load_state_dict(best_state['encoder'])
        model.predictor.load_state_dict(best_state['predictor'])
        if best_state['emb'] and model.emb: 
            model.emb.load_state_dict(best_state['emb'])
        print(f"✓ Loaded best model from epoch {best_epoch}", flush=True)
        print(f"✓ Best validation {metric_name}: {best_val_score:.4f}", flush=True)
    
    # If no validation happened (very short training), return None or reasonable defaults
    if best_all_results is None:
         print("\nWARNING: No validation step completed. Returning last state.")
         # Force one eval to generate results for return
         best_all_results = model.test(data, split_edge, batch_size=args.batch_size,
                               evaluator=evaluator, eval_metric='precision10', 
                               fairness_mode=args.fairness_mode)
         metric_name = 'Precision@10'
         best_val_score = best_all_results[metric_name][0]
         best_epoch = args.epochs

    return model, best_epoch, best_all_results, metric_name, best_val_score, []