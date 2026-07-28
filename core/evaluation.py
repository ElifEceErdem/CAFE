"""
Enhanced evaluation.py with new metrics:
- Equality Gain
- Stratified Precision and Equality Gain (20/80 split)
- Degree-based analysis (Precision vs Degree, Equality Gain vs Degree)
"""

import torch
from metrics.precision import compute_metrics_from_recs
from metrics.pa_aware import compute_pa_fairness_from_overlap
from metrics.ndkl import compute_ndkl_metrics, get_evaluation_user_set, categorize_nodes_by_degree, compute_target_distribution
from metrics.equality_gain import compute_equality_gain_metrics
from metrics.stratified import compute_stratified_metrics
from metrics.degree_analysis import analyze_degree_effects
from metrics.demographics import compute_demographic_metrics, compute_demographic_parity_ratio_change
from core.data_loader import build_train_adjacency
from metrics.centrality_equality_gain import compute_all_centrality_equality_gains
from metrics.classification import compute_classification_metrics, compute_rawlsian_degree_bias
from metrics.networked_inequality import (
    compute_within_group_unfairness_metric,
    validate_taylor_approximation
)
from fairness_methods.reranking_coreness import CorenessReRanker
from metrics.ndkl import categorize_nodes_by_degree, compute_target_distribution


import numpy as np
import scipy.sparse as sp
from metrics.ndkl import categorize_nodes_by_degree, compute_target_distribution
import sys
from tqdm import tqdm


def canon_edges_2col(edges):
    """
    Canonicalize an edge tensor/array to [E, 2] (one edge per row).

    split_edge['*']['edge'] is [E, 2] throughout this codebase
    (backbone/snap_dataset.py), but some tensors circulate as [2, E]
    (e.g. data.edge_index). Reading rows [0]/[1] of an [E, 2] tensor
    silently yields the first two EDGES instead of the src/dst columns,
    which is how the pre-fix versions of compute_baseline_metrics and
    build_recommendations ended up with a ~4-entry training mask.
    """
    e = edges.cpu() if hasattr(edges, 'cpu') else torch.as_tensor(edges)
    if e.dim() != 2 or (e.size(0) != 2 and e.size(1) != 2):
        raise ValueError(f"Edge tensor must be [E,2] or [2,E], got {tuple(e.shape)}")
    if e.size(0) == 2 and e.size(1) != 2:
        e = e.t()
    return e


def compute_hits_mrr_from_scores(pos_scores, neg_scores):
    """Hits@K and MRR logic from clean script."""
    M = neg_scores.numel() // pos_scores.numel()
    neg = neg_scores.view(-1, M)
    ranks = 1 + (neg > pos_scores.view(-1, 1)).sum(dim=1)
    results = {'MRR': (0.0, (1.0 / ranks.float()).mean().item())}
    for K in [10, 20, 50, 100]:
        results[f'Hits@{K}'] = (0.0, (ranks <= K).float().mean().item())
    return results


def compute_baseline_metrics(model, data, split_edge, args, sens_attr=None):
    """
    Baseline recommendation generation with ALL metrics including new ones.
    """
    print("\n" + "="*80)
    print("COMPUTING BASELINE METRICS")
    print("="*80)
    
    # Get configuration
    use_pa_aware = getattr(args, 'use_pa_aware', False)
    use_ndkl = getattr(args, 'use_ndkl', False)
    use_equality_gain = getattr(args, 'use_equality_gain', False)
    use_stratified = getattr(args, 'use_stratified', False)
    use_degree_plots = getattr(args, 'use_degree_plots', False)
    use_centrality_gain = getattr(args, 'use_centrality_gain', False)  # NEW
    use_within_group_unfairness = getattr(args, 'use_within_group_unfairness', False)

    ndkl_mode = getattr(args, 'ndkl_mode', 'per_user')
    eval_user_mode = getattr(args, 'eval_user_mode', 'test_users')
    centrality_types = getattr(args, 'centrality_types', ['degree', 'katz', 'pagerank', 'eigenvector', 'closeness'])  # NEW
    
    if use_pa_aware:
        print("PA-aware metrics: ENABLED")
    if use_ndkl:
        print(f"NDKL metrics: ENABLED (mode={ndkl_mode})")
    if use_equality_gain:
        print("Equality Gain metrics: ENABLED")
    if use_stratified:
        print("Stratified metrics: ENABLED")
    if use_degree_plots:
        print("Degree analysis plots: ENABLED")
    if use_pa_aware or use_ndkl:
        print(f"Evaluation user set: {eval_user_mode}")
    if use_centrality_gain:
        print(f"Centrality Gain metrics: ENABLED (types={centrality_types})")  # NEW
    if use_within_group_unfairness:
        print("Within-Group Unfairness (Networked Inequality): ENABLED")
    
    # Call eval() on encoder and predictor separately
    model.encoder.eval()
    model.predictor.eval()
    
    with torch.no_grad():
        # Get embeddings
        input_feat = model.create_input_feat(data)
        h = model.encoder(input_feat, data.adj_t, data.edge_weight)
        mean_h = torch.mean(h, dim=0, keepdim=True)
        h_with_mean = torch.cat([h, mean_h], dim=0)
        h_clean = h  # Without mean node
        
        # Get edges
        pos_test_edge = split_edge['test']['edge'].to(model.device)
        neg_test_edge = split_edge['test']['edge_neg'].to(model.device)
        train_edges = split_edge['train']['edge'].to(model.device)
        
        # Get degrees - Use raw adjacency sum (vectorized)
        num_nodes = model.num_nodes
        train_edges_cpu = canon_edges_2col(train_edges)
        src, dst = train_edges_cpu[:, 0], train_edges_cpu[:, 1]
        # Filter valid edges
        valid = (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_nodes)
        src, dst = src[valid], dst[valid]
        deg = torch.zeros(num_nodes, dtype=torch.float32)
        deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        
        print(f"Degree statistics: min={deg.min():.1f}, max={deg.max():.1f}, mean={deg.mean():.1f}")
        
        # Build training adjacency
        train_adj = build_train_adjacency(train_edges, num_nodes)
        
        # Determine evaluation user set
        if use_pa_aware or use_ndkl:
            eval_users = get_evaluation_user_set(
                split_edge, num_nodes, mode=eval_user_mode
            )
        else:
            eval_users = list(range(num_nodes))
        
        print(f"\nGenerating baseline recommendations for {len(eval_users)} users...")
        
        # Build recommendations using trained predictor (not raw dot product)
        score_fn = make_predictor_score_fn(model.predictor, predictor_type='two_arg')
        recs_baseline = build_recommendations(
            h_clean, train_edges, eval_users, k=100, device=model.device,
            score_fn=score_fn
        )
        
        # Compute Precision and Distribution Fairness
        results = compute_metrics_from_recs(
            recs_baseline, pos_test_edge, num_nodes, args.fairness_mode
        )
        
        # Compute Hits@K and MRR
        pos_test_pred = model.batch_predict(h_with_mean, pos_test_edge, args.batch_size)
        neg_test_pred = model.batch_predict(h_with_mean, neg_test_edge.view(-1, 2), args.batch_size)
        hits_mrr = compute_hits_mrr_from_scores(pos_test_pred, neg_test_pred)
        results.update(hits_mrr)

        # === Classification Metrics (AUC, F1, Accuracy) + Rawlsian Bias ===
        cls_metrics = compute_classification_metrics(pos_test_pred, neg_test_pred)
        results.update(cls_metrics)
        rawls_bias = compute_rawlsian_degree_bias(
            recs_baseline, pos_test_edge, deg, num_nodes, k=10
        )
        results['RawlsBias'] = (0.0, rawls_bias)

        # === NEW: Demographic Metrics (SP, EO, IF, GDIF) ===
        # Ensure sensitive attribute available
        sens = sens_attr
        if sens is None and hasattr(data, 'sens'):
            sens = data.sens
            
        if sens is not None:
            # Need features for Ranking_IF
            # Extract features from data if available, else None
            features = getattr(data, 'x', None)
            
            # Need adjacency for IF
            adj = None # Will be built from train_edges inside function if None
            
            demo_metrics = compute_demographic_metrics(
                recs_by_u=recs_baseline,
                train_edges=train_edges,
                test_edges=pos_test_edge,
                num_nodes=num_nodes,
                sens=sens,
                h_emb=h_clean,
                features=features,
                adj=adj,
                k=10
            )
            results.update(demo_metrics)
            
            # DPR Change
            dpr_change = compute_demographic_parity_ratio_change(
                train_edges, recs_baseline, num_nodes, sens, k=10
            )
            results['Delta_DPR'] = dpr_change

        else:
            print("Warning: Skipping demographic metrics (no sensitive attribute)")

        # === Networked Inequality: Within-Group Unfairness (ICML 2024) ===
        if use_within_group_unfairness:
            social_groups = getattr(data, 'y', None)
            wgu_sens = sens  # reuse sensitive attribute from above
            if social_groups is not None and wgu_sens is not None:
                wgu_results = compute_within_group_unfairness_metric(
                    model, data, split_edge, social_groups, wgu_sens,
                    batch_size=args.batch_size
                )
                results.update({
                    'WithinGroupUnfairness': wgu_results['WithinGroupUnfairness'],
                })
                results['_wgu_per_group'] = wgu_results.get(
                    'WithinGroupUnfairness_per_group', {}
                )

                # Optional: Taylor approximation validation (GCN only)
                conv_type = getattr(args, 'conv_type', 'sym')
                use_taylor = getattr(args, 'use_taylor_validation', False)
                if use_taylor:
                    taylor_results = validate_taylor_approximation(
                        model, data, split_edge, social_groups, conv_type
                    )
                    if taylor_results is not None:
                        results['Taylor_PCC'] = taylor_results['Taylor_PCC']
                        results['Taylor_NRMSE'] = taylor_results['Taylor_NRMSE']
            else:
                missing = []
                if social_groups is None:
                    missing.append("data.y (social groups)")
                if wgu_sens is None:
                    missing.append("sens_attr (sensitive attribute)")
                print(f"Warning: Skipping Within-Group Unfairness "
                      f"(missing: {', '.join(missing)})")

        # === Equality Gain Metrics ===
        if getattr(args, 'use_equality_gain', False):
            eq_metrics = compute_equality_gain_metrics(
                recs_baseline, train_edges, pos_test_edge, num_nodes,
                k_values=[10, 20, 50, 100]
            )
            results.update(eq_metrics)

        # === NEW: Centrality-based Equality Gain Metrics ===
        if use_centrality_gain:
            cent_metrics = compute_all_centrality_equality_gains(
                recs_baseline, train_edges, pos_test_edge, num_nodes,
                k_values=[10, 20, 50, 100],
                centrality_types=centrality_types,
                use_sampling_closeness=getattr(args, 'use_sampling_closeness', True)
            )
            results.update(cent_metrics)

        # === NEW: Stratified Metrics ===
        if getattr(args, 'use_stratified', False):
            strat_metrics = compute_stratified_metrics(
                recs_baseline, train_edges, pos_test_edge, deg, num_nodes,
                k=10,
                low_percentile=20.0,
                high_percentile=80.0
            )
            results.update(strat_metrics)

        # === NEW: Degree Analysis Plots ===
        if getattr(args, 'use_degree_plots', False):
            save_dir = getattr(args, 'plot_save_dir', 'plots')
            degree_analysis = analyze_degree_effects(
                recs_baseline, train_edges, pos_test_edge, deg, num_nodes,
                k=10,
                num_bins=10,
                save_dir=save_dir
            )
            results['_degree_analysis'] = degree_analysis
        
        # Store data for fairness metrics
        results['_recs_baseline'] = recs_baseline
        results['_deg'] = deg
        results['_train_adj'] = train_adj
        results['_eval_users'] = eval_users
        results['_num_nodes'] = num_nodes
        results['_h_clean'] = h_clean
        results['_train_edges'] = train_edges
        results['_test_edges'] = pos_test_edge
        results['_centrality_types'] = centrality_types 
        
        # (Equality Gain, Stratified, and Degree Analysis already computed above)

        # Compute PA-aware metrics (baseline has 0 change by definition)
        if use_pa_aware:
            print("\n" + "="*60)
            print("COMPUTING PA-AWARE METRICS (BASELINE)")
            print("="*60)
            
            pa_results = compute_pa_fairness_from_overlap(
                recs_baseline=recs_baseline,
                recs_postprocessed=recs_baseline,  # Same for baseline
                degrees=deg,
                train_adj=train_adj,
                test_nodes=eval_users,
                num_nodes=num_nodes,
                k_values=[10, 20, 50, 100],
                normalization='auc'
            )
            
            results['PA_Bias_Score'] = (0.0, 0.0)  # Baseline = 0 by definition
            results['PA_Fairness_Score'] = (0.0, 0.0)
            results['PA_Overlap_Mean'] = (0.0, pa_results['mean_overlap_postproc'])
        
        # Compute NDKL metrics
        if use_ndkl:
            print("\n" + "="*60)
            print("COMPUTING NDKL METRICS (BASELINE)")
            print("="*60)
            
            # Categorize nodes and compute target distribution
            node_categories, thresholds = categorize_nodes_by_degree(deg, method='cumulative')
            
            print(f"Node categorization:")
            print(f"  Thresholds: {thresholds}")
            for group_id in range(len(thresholds) + 1):
                group_nodes = (node_categories == group_id).sum().item()
                print(f"  Group {group_id}: {group_nodes} nodes")
            
            target_dist, group_counts = compute_target_distribution(train_edges, node_categories)
            
            print(f"Target distribution: {target_dist}")
            print(f"Group counts: {group_counts}")
            
            # Store for reuse in fairness evaluation
            results['_node_categories'] = node_categories
            results['_target_dist'] = target_dist
            
            # Compute NDKL
            ndkl_results = compute_ndkl_metrics(
                recs_by_u=recs_baseline,
                degrees=deg,
                train_edges=train_edges,
                h_emb=h_clean,
                predictor=None,  # Use dot product for consistency
                device=model.device,
                k_values=[10, 20, 50, 100],
                ndkl_mode=ndkl_mode,
                node_categories=node_categories,
                target_dist=target_dist
            )
            
            # Add NDKL scores to results
            for k in [10, 20, 50, 100]:
                if k in ndkl_results['ndkl_scores']:
                    results[f'NDKL@{k}'] = (0.0, ndkl_results['ndkl_scores'][k])
        
        print("\n Baseline metrics computed and stored")
        return results


def make_predictor_score_fn(predictor, predictor_type='two_arg', batch_cands=512):
    """
    Create a batched scoring function from a trained predictor.
    
    This ensures P@K evaluation uses the same scoring function the model
    was trained with, instead of raw dot product.
    
    Args:
        predictor: The predictor module (e.g., MLPPredictor, nn.Sequential)
        predictor_type:
            'two_arg' for predictors called as predictor(x_i, x_j)
                      (e.g., HLGNN's MLPPredictor: element-wise product -> MLP)
            'concat'  for predictors called as predictor(cat(x_i, x_j))
                      (e.g., PureGNN's nn.Sequential)
        batch_cands: Number of candidates to process at once (controls memory)
    
    Returns:
        score_fn(h_sources [B,D], h_all [N,D]) -> scores [B,N]
    """
    @torch.no_grad()
    def score_fn(h_sources, h_all):
        B, D = h_sources.size()
        N = h_all.size(0)
        out = []
        for c0 in range(0, N, batch_cands):
            c1 = min(c0 + batch_cands, N)
            cand = h_all[c0:c1]  # [C, D]
            C = cand.size(0)
            # Broadcast: [B, C, D]
            u_rep = h_sources[:, None, :].expand(B, C, D).reshape(-1, D)  # [B*C, D]
            v_rep = cand[None, :, :].expand(B, C, D).reshape(-1, D)      # [B*C, D]
            
            if predictor_type == 'two_arg':
                s = predictor(u_rep, v_rep).squeeze(-1)  # [B*C]
            elif predictor_type == 'concat':
                h_cat = torch.cat([u_rep, v_rep], dim=1)  # [B*C, 2D]
                s = predictor(h_cat).squeeze(-1)           # [B*C]
            else:
                raise ValueError(f"Unknown predictor_type: {predictor_type}")
            
            out.append(torch.sigmoid(s.view(B, -1)))  # [B, C]
        return torch.cat(out, dim=1)  # [B, N]
    
    return score_fn


def build_recommendations(h_emb, train_edges, eval_users, k, device, score_fn=None):
    """
    Core recommendation loop without fairness intervention.

    Args:
        h_emb: Node embeddings [N, D]
        train_edges: Training edges to exclude
        eval_users: List of user node IDs to generate recommendations for
        k: Number of recommendations per user
        device: torch device
        score_fn: Optional callable(h_sources [B,D], h_all [N,D]) -> [B,N] scores.
                  Created by make_predictor_score_fn(). If None, uses dot product.
    """
    n = h_emb.size(0)

    # Build sparse mask matrix for training edges (CSR for fast row slicing)
    edges_cpu = canon_edges_2col(train_edges).numpy()
    # Symmetric: both directions
    rows = np.concatenate([edges_cpu[:, 0], edges_cpu[:, 1]])
    cols = np.concatenate([edges_cpu[:, 1], edges_cpu[:, 0]])
    valid = (rows < n) & (cols < n)
    rows, cols = rows[valid], cols[valid]
    train_mask_csr = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, n)
    )

    recs_by_u = {}
    chunk_size = 64
    scoring_method = "trained predictor" if score_fn is not None else "dot product"
    num_chunks = (len(eval_users) + chunk_size - 1) // chunk_size

    for i in tqdm(range(0, len(eval_users), chunk_size), total=num_chunks,
                  desc=f"Building recs ({scoring_method})", unit="chunk", file=sys.stdout):
        chunk_end = min(i + chunk_size, len(eval_users))
        chunk = eval_users[i:chunk_end]

        # Compute scores
        chunk_tensor = torch.tensor(chunk, dtype=torch.long, device=device)
        h_chunk = h_emb[chunk_tensor]
        if score_fn is not None:
            scores = score_fn(h_chunk, h_emb).cpu()
        else:
            scores = torch.sigmoid(h_chunk @ h_emb.t()).cpu()
        del h_chunk  # Free memory

        # Vectorized masking: build mask for entire chunk at once
        chunk_arr = np.array(chunk)
        # Self-loop mask
        scores[range(len(chunk)), chunk_arr] = float('-inf')
        # Training edge mask: extract rows from sparse matrix
        mask_block = train_mask_csr[chunk_arr]  # [chunk_size, n] sparse
        mask_rows, mask_cols = mask_block.nonzero()
        scores[mask_rows, mask_cols] = float('-inf')

        # Batch top-K
        topk_scores, topk_indices = torch.topk(scores, min(k, n), dim=1)

        for idx, u in enumerate(chunk):
            valid_mask = torch.isfinite(topk_scores[idx])
            recs_by_u[u] = topk_indices[idx][valid_mask][:k].tolist()

        del scores  # Free memory
    
    return recs_by_u

def compute_fairness_metrics_with_weak_tie(model, data, split_edge, args, baseline_results):
    """
    Evaluate the weak-tie diversification baseline
    (Sanz-Cruzado & Castells, RecSys 2018) on top of the trained model.

    Post-processing only: re-ranks the baseline candidate pools stored in
    baseline_results['_recs_baseline'] (top-100 lists ordered by the
    trained predictor) with the greedy weak-tie re-ranker, then computes
    the same metric battery as the other methods. The Louvain partition
    is computed once per run and cached in baseline_results.

    Reads from args: weak_tie_variant ('cegc'|'mc'), weak_tie_lambda,
    weak_tie_k, community_seed.
    """
    from fairness_methods.weak_tie_reranking import (
        WeakTieReranker, detect_communities,
        modularity_complement, community_edge_gini_complement,
        weak_tie_fraction,
    )

    variant = getattr(args, 'weak_tie_variant', 'cegc')
    lam = float(getattr(args, 'weak_tie_lambda', 0.5))
    wt_k = int(getattr(args, 'weak_tie_k', 10))

    print("\n" + "=" * 80)
    print("COMPUTING WEAK-TIE DIVERSIFICATION METRICS "
          f"(variant={variant}, lambda={lam}, k={wt_k})")
    print("=" * 80)

    use_equality_gain = getattr(args, 'use_equality_gain', False)
    use_centrality_gain = getattr(args, 'use_centrality_gain', False)
    use_stratified = getattr(args, 'use_stratified', False)
    centrality_types = baseline_results.get(
        '_centrality_types',
        getattr(args, 'centrality_types', ['degree', 'pagerank', 'closeness']))

    train_edges = baseline_results['_train_edges']
    pos_test_edge = baseline_results['_test_edges']
    num_nodes = baseline_results['_num_nodes']
    deg = baseline_results['_deg']
    recs_baseline = baseline_results['_recs_baseline']
    h_clean = baseline_results['_h_clean']

    # Louvain communities of the training graph (computed once, cached)
    membership = baseline_results.get('_wt_membership')
    if membership is None:
        membership = detect_communities(
            train_edges, num_nodes, seed=getattr(args, 'community_seed', 42))
        baseline_results['_wt_membership'] = membership

    reranker = WeakTieReranker(variant=variant, lam=lam, k=wt_k)
    recs_postprocessed = reranker.rerank(
        recs_baseline, train_edges, num_nodes, membership=membership)

    # --- standard battery (mirrors the other post-processing methods) ---
    results = compute_metrics_from_recs(
        recs_postprocessed, pos_test_edge, num_nodes, args.fairness_mode)

    with torch.no_grad():
        mean_h = torch.mean(h_clean, dim=0, keepdim=True)
        h_with_mean = torch.cat([h_clean, mean_h], dim=0)
        neg_test_edge = split_edge['test']['edge_neg'].to(model.device)
        pos_test_pred = model.batch_predict(h_with_mean, pos_test_edge, args.batch_size)
        neg_test_pred = model.batch_predict(h_with_mean, neg_test_edge.view(-1, 2), args.batch_size)
    results.update(compute_hits_mrr_from_scores(pos_test_pred, neg_test_pred))
    results.update(compute_classification_metrics(pos_test_pred, neg_test_pred))
    results['RawlsBias'] = (0.0, compute_rawlsian_degree_bias(
        recs_postprocessed, pos_test_edge, deg, num_nodes, k=10))

    if use_equality_gain:
        results.update(compute_equality_gain_metrics(
            recs_postprocessed, train_edges, pos_test_edge, num_nodes,
            k_values=[10, 20, 50, 100]))

    if use_centrality_gain:
        results.update(compute_all_centrality_equality_gains(
            recs_postprocessed, train_edges, pos_test_edge, num_nodes,
            k_values=[10, 20, 50, 100], centrality_types=centrality_types,
            use_sampling_closeness=getattr(args, 'use_sampling_closeness', True)))

    if use_stratified:
        results.update(compute_stratified_metrics(
            recs_postprocessed, train_edges, pos_test_edge, deg, num_nodes,
            k=10, low_percentile=20.0, high_percentile=80.0,
            stratification_mode=getattr(args, 'stratification_mode', 'source')))

    sens = getattr(data, 'sens', None)
    if sens is not None:
        demo_metrics = compute_demographic_metrics(
            recs_by_u=recs_postprocessed, train_edges=train_edges,
            test_edges=pos_test_edge, num_nodes=num_nodes, sens=sens,
            h_emb=h_clean, features=getattr(data, 'x', None), adj=None, k=10)
        results.update(demo_metrics)
        results['Delta_DPR'] = compute_demographic_parity_ratio_change(
            train_edges, recs_postprocessed, num_nodes, sens, k=10)

    # --- structural-diversity diagnostics (paper Sec. 4 metrics) ---
    for label, recs in (('base', recs_baseline), ('reranked', recs_postprocessed)):
        results[f'WT_MC_{label}@{wt_k}'] = (0.0, modularity_complement(
            train_edges, recs, num_nodes, membership, k=wt_k))
        results[f'WT_CEGC_{label}@{wt_k}'] = (0.0, community_edge_gini_complement(
            train_edges, recs, num_nodes, membership, k=wt_k))
        results[f'WT_WeakFrac_{label}@{wt_k}'] = (0.0, weak_tie_fraction(
            recs, membership, k=wt_k))

    results['_recs_reranked'] = recs_postprocessed
    results['_deg'] = deg
    results['_train_edges'] = train_edges
    results['_test_edges'] = pos_test_edge
    results['_num_nodes'] = num_nodes
    return results


def compute_fairness_metrics_with_coreness(model, data, split_edge, args, baseline_results):
    """
    Compute fairness metrics using Coreness-based re-ranking.
    """
    print("\n" + "="*80)
    print(f"COMPUTING FAIRNESS METRICS (CORENESS RE-RANKING)")
    print(f"Lambda Mobility: {args.lambda_mobility}, Lambda Bridge: {args.lambda_bridge}")
    print(f"Alpha: {args.coreness_alpha}")
    print("="*80)
    
    # Get configuration
    use_pa_aware = getattr(args, 'use_pa_aware', False)
    use_ndkl = getattr(args, 'use_ndkl', False)
    use_equality_gain = getattr(args, 'use_equality_gain', False)
    use_stratified = getattr(args, 'use_stratified', False)
    use_degree_plots = getattr(args, 'use_degree_plots', False)
    use_centrality_gain = getattr(args, 'use_centrality_gain', False)
    ndkl_mode = getattr(args, 'ndkl_mode', 'per_user')
    centrality_types = getattr(args, 'centrality_types', ['degree', 'katz', 'pagerank', 'eigenvector', 'closeness'])
    
    model.encoder.eval()
    model.predictor.eval()
    
    with torch.no_grad():
        # Get embeddings
        input_feat = model.create_input_feat(data)
        h = model.encoder(input_feat, data.adj_t, data.edge_weight)
        mean_h = torch.mean(h, dim=0, keepdim=True)
        h_with_mean = torch.cat([h, mean_h], dim=0)
        h_clean = h
        
        # Get edges and stored data
        pos_test_edge = split_edge['test']['edge'].to(model.device)
        neg_test_edge = split_edge['test']['edge_neg'].to(model.device)
        train_edges = split_edge['train']['edge'].to(model.device)
        
        # Retrieve stored data from baseline
        deg = baseline_results['_deg']
        train_adj = baseline_results['_train_adj']
        eval_users = baseline_results['_eval_users']
        num_nodes = baseline_results['_num_nodes']
        recs_baseline = baseline_results.get('_recs_baseline')
        centrality_types = baseline_results.get('_centrality_types', centrality_types)
        
        print(f"\nGenerating coreness-based recommendations for {len(eval_users)} users...")
        

        reranker = CorenessReRanker(
            lambda_mobility=args.lambda_mobility,
            lambda_bridge=args.lambda_bridge,
            alpha=args.coreness_alpha,
            boosting_mode=getattr(args, 'boosting_mode', 'continuous'),
            reranking_approach=getattr(args, 'reranking_approach', 'boost')
        )
        
        # Generate coreness re-ranked recommendations using trained predictor
        score_fn = make_predictor_score_fn(model.predictor, predictor_type='two_arg')
        recs_postprocessed = reranker.rerank_topk(
            h_clean, train_edges, eval_users, k=100, degrees=deg,
            score_fn=score_fn
        )
        
        # Compute standard metrics
        results = compute_metrics_from_recs(
            recs_postprocessed, pos_test_edge, num_nodes, args.fairness_mode
        )
        
        # Compute Hits@K and MRR
        pos_test_pred = model.batch_predict(h_with_mean, pos_test_edge, args.batch_size)
        neg_test_pred = model.batch_predict(h_with_mean, neg_test_edge.view(-1, 2), args.batch_size)
        hits_mrr = compute_hits_mrr_from_scores(pos_test_pred, neg_test_pred)
        results.update(hits_mrr)

        # === Classification Metrics (AUC, F1, Accuracy) + Rawlsian Bias ===
        cls_metrics = compute_classification_metrics(pos_test_pred, neg_test_pred)
        results.update(cls_metrics)
        rawls_bias = compute_rawlsian_degree_bias(
            recs_postprocessed, pos_test_edge, deg, num_nodes, k=10
        )
        results['RawlsBias'] = (0.0, rawls_bias)
        
        # === Equality Gain Metrics ===
        if use_equality_gain:
            eq_metrics = compute_equality_gain_metrics(
                recs_postprocessed, train_edges, pos_test_edge, num_nodes,
                k_values=[10, 20, 50, 100]
            )
            results.update(eq_metrics)
        
        # === NEW: Centrality-based Equality Gain Metrics ===
        if use_centrality_gain:
            cent_metrics = compute_all_centrality_equality_gains(
                recs_postprocessed, train_edges, pos_test_edge, num_nodes,
                k_values=[10, 20, 50, 100],
                centrality_types=centrality_types,
                use_sampling_closeness=getattr(args, 'use_sampling_closeness', True)
            )
            results.update(cent_metrics)
        
        # === NEW: Stratified Metrics ===
        if use_stratified:
            strat_metrics = compute_stratified_metrics(
                recs_postprocessed, train_edges, pos_test_edge, deg, num_nodes,
                k=10,
                low_percentile=20.0,
                high_percentile=80.0,
                stratification_mode=args.stratification_mode
            )
            results.update(strat_metrics)
        
        # === NEW: Degree Analysis Plots ===
        if use_degree_plots:
            save_dir = getattr(args, 'plot_save_dir', 'plots')
            degree_analysis = analyze_degree_effects(
                recs_postprocessed, train_edges, pos_test_edge, deg, num_nodes,
                k=10,
                num_bins=10,
                save_dir=save_dir
            )
            results['_degree_analysis'] = degree_analysis
        
        # Compute PA-aware metrics if enabled
        if use_pa_aware and recs_baseline is not None:
            print("\n" + "="*60)
            print(f"COMPUTING PA-AWARE METRICS")
            print("="*60)
            
            pa_results = compute_pa_fairness_from_overlap(
                recs_baseline=recs_baseline,
                recs_postprocessed=recs_postprocessed,
                degrees=deg,
                train_adj=train_adj,
                test_nodes=eval_users,
                num_nodes=num_nodes,
                k_values=[10, 20, 50, 100],
                normalization='auc'
            )
            
            results['PA_Bias_Score'] = (0.0, pa_results['pa_bias_reduction'])
            results['PA_Fairness_Score'] = (0.0, pa_results['pa_fairness_score'])
            results['PA_Overlap_Mean'] = (0.0, pa_results['mean_overlap_postproc'])
        
        # Compute NDKL metrics if enabled
        if use_ndkl:
            print("\n" + "="*60)
            print(f"COMPUTING NDKL METRICS")
            print("="*60)
            
            node_categories = baseline_results.get('_node_categories')
            target_dist = baseline_results.get('_target_dist')
            
            if node_categories is None or target_dist is None:
                node_categories, _ = categorize_nodes_by_degree(deg, method='cumulative')
                target_dist, _ = compute_target_distribution(train_edges, node_categories)
            
            ndkl_results = compute_ndkl_metrics(
                recs_by_u=recs_postprocessed,
                degrees=deg,
                train_edges=train_edges,
                h_emb=h_clean,
                predictor=None,
                device=model.device,
                k_values=[10, 20, 50, 100],
                ndkl_mode=ndkl_mode,
                node_categories=node_categories,
                target_dist=target_dist
            )
            
            for k in [10, 20, 50, 100]:
                if k in ndkl_results['ndkl_scores']:
                    results[f'NDKL@{k}'] = (0.0, ndkl_results['ndkl_scores'][k])
        
        results['_recs_reranked'] = recs_postprocessed
        results['_deg'] = deg
        results['_train_adj'] = train_adj
        results['_eval_users'] = eval_users
        results['_num_nodes'] = num_nodes
        results['_h_clean'] = h_clean
        results['_train_edges'] = train_edges
        results['_test_edges'] = pos_test_edge
        
        return results
