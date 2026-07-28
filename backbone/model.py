from torch.utils.data import DataLoader
from layer import *
from loss import *
import numpy as np
import scipy.sparse as sp
from utils import *


class BaseModel(object):
    """
        Parameters
        ----------
        lr : double
            Learning rate
        dropout : double
            dropout probability for gnn and mlp layers
        gnn_num_layers : int
            number of gnn layers
        mlp_num_layers : int
            number of gnn layers
        *_hidden_channels : int
            dimension of hidden
        num_nodes : int
            number of graph nodes
        num_node_feats : int
            dimension of raw node features
        gnn_encoder_name : str
            gnn encoder name
        predictor_name: str
            link predictor name
        loss_func: str
            loss function name
        optimizer_name: str
            optimization method name
        device: str
            device name: gpu or cpu
        use_node_feats: bool
            whether to use raw node features as input
        train_node_emb: bool
            whether to train node embeddings based on node id
        pretrain_emb: str
            whether to load pretrained node embeddings
    """

    def __init__(self, lr, dropout, grad_clip_norm, gnn_num_layers, mlp_num_layers, emb_hidden_channels,
                 gnn_hidden_channels, mlp_hidden_channels, num_nodes, num_node_feats, gnn_encoder_name,
                 predictor_name, loss_func, optimizer_name, device, use_node_feats, train_node_emb,
                 pretrain_emb, alpha, init):
        self.loss_func_name = loss_func
        self.num_nodes = num_nodes
        self.num_node_feats = num_node_feats
        self.use_node_feats = use_node_feats
        self.train_node_emb = train_node_emb
        self.clip_norm = grad_clip_norm
        self.device = device

        # Input Layer
        self.input_channels, self.emb = create_input_layer(num_nodes=num_nodes,
                                                           num_node_feats=num_node_feats,
                                                           hidden_channels=emb_hidden_channels,
                                                           use_node_feats=use_node_feats,
                                                           train_node_emb=train_node_emb,
                                                           pretrain_emb=pretrain_emb)
        if self.emb is not None:
            self.emb = self.emb.to(device)

        # GNN Layer
        self.encoder = create_gnn_layer(input_channels=self.input_channels,
                                        hidden_channels=gnn_hidden_channels,
                                        num_layers=gnn_num_layers,
                                        dropout=dropout,
                                        encoder_name=gnn_encoder_name,
                                        alpha=alpha, init=init).to(device)

        # Predict Layer
        self.predictor = create_predictor_layer(hidden_channels=mlp_hidden_channels,
                                                num_layers=mlp_num_layers,
                                                dropout=dropout,
                                                predictor_name=predictor_name).to(device)

        # Parameters and Optimizer
        self.para_list = list(self.encoder.parameters()) + list(self.predictor.parameters())
        if self.emb is not None:
            self.para_list += list(self.emb.parameters())

        if optimizer_name == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.para_list, lr=lr)
        elif optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(self.para_list, lr=lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
        else:
            self.optimizer = torch.optim.Adam(self.para_list, lr=lr)

    def param_init(self):
        self.encoder.reset_parameters()
        self.predictor.reset_parameters()
        if self.emb is not None:
            torch.nn.init.xavier_uniform_(self.emb.weight)

    def create_input_feat(self, data):
        if self.use_node_feats:
            input_feat = data.x.to(self.device)
            if self.train_node_emb:
                input_feat = torch.cat([self.emb.weight, input_feat], dim=-1)
        else:
            input_feat = self.emb.weight
        return input_feat

    def calculate_loss(self, pos_out, neg_out, num_neg, margin=None):
        if self.loss_func_name == 'CE':
            loss = ce_loss(pos_out, neg_out)
        elif self.loss_func_name == 'InfoNCE':
            loss = info_nce_loss(pos_out, neg_out, num_neg)
        elif self.loss_func_name == 'LogRank':
            loss = log_rank_loss(pos_out, neg_out, num_neg)
        elif self.loss_func_name == 'HingeAUC':
            loss = hinge_auc_loss(pos_out, neg_out, num_neg)
        elif self.loss_func_name == 'AdaAUC' and margin is not None:
            loss = adaptive_auc_loss(pos_out, neg_out, num_neg, margin)
        elif self.loss_func_name == 'WeightedAUC' and margin is not None:
            loss = weighted_auc_loss(pos_out, neg_out, num_neg, margin)
        elif self.loss_func_name == 'AdaHingeAUC' and margin is not None:
            loss = adaptive_hinge_auc_loss(pos_out, neg_out, num_neg, margin)
        elif self.loss_func_name == 'WeightedHingeAUC' and margin is not None:
            loss = weighted_hinge_auc_loss(pos_out, neg_out, num_neg, margin)
        else:
            loss = auc_loss(pos_out, neg_out, num_neg)
        return loss

    def train(self, data, split_edge, batch_size, neg_sampler_name, num_neg):
        self.encoder.train()
        self.predictor.train()

        pos_train_edge, neg_train_edge = get_pos_neg_edges('train', split_edge,
                                                           edge_index=data.edge_index,
                                                           num_nodes=self.num_nodes,
                                                           neg_sampler_name=neg_sampler_name,
                                                           num_neg=num_neg)

        pos_train_edge, neg_train_edge = pos_train_edge.to(self.device), neg_train_edge.to(self.device)

        if 'weight' in split_edge['train']:
            edge_weight_margin = split_edge['train']['weight'].to(self.device)
        else:
            edge_weight_margin = None

        total_loss = total_examples = 0

        for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
            self.optimizer.zero_grad(set_to_none=True)

            input_feat = self.create_input_feat(data)
            h = self.encoder(input_feat, data.adj_t, data.edge_weight)
            pos_edge = pos_train_edge[perm].t()
            neg_edge = torch.reshape(neg_train_edge[perm], (-1, 2)).t()

            pos_out = self.predictor(h[pos_edge[0]], h[pos_edge[1]])
            neg_out = self.predictor(h[neg_edge[0]], h[neg_edge[1]])

            weight_margin = edge_weight_margin[perm] if edge_weight_margin is not None else None

            loss = self.calculate_loss(pos_out, neg_out, num_neg, margin=weight_margin)
            loss.backward()

            if self.clip_norm >= 0:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_norm)
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), self.clip_norm)

            self.optimizer.step()

            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

        return total_loss / total_examples

    @torch.no_grad()
    def batch_predict(self, h, edges, batch_size):
        preds = []
        for perm in DataLoader(range(edges.size(0)), batch_size):
            edge = edges[perm].t()
            preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred

    @torch.no_grad()
    def test(self, data, split_edge, batch_size, evaluator, eval_metric, fairness_mode='indegree',
             eval_chunk_size=1024, eval_batch_cands=4096, skip_test_eval=False, max_eval_nodes=0):
        """
        UPDATED: Fairness matches PULL/Bloom implementation
        - Only evaluates on TEST NODES (not all nodes)
        - Supports indegree/outdegree/total modes
        - Removed Ranking Fairness Gap
        """
        self.encoder.eval()
        self.predictor.eval()

        # ----- Embeddings once -----
        input_feat = self.create_input_feat(data)
        h = self.encoder(input_feat, data.adj_t, data.edge_weight)
        mean_h = torch.mean(h, dim=0, keepdim=True)  # unseen node embedding
        h = torch.cat([h, mean_h], dim=0)

        # ----- Validation edges -----
        pos_valid_edge, neg_valid_edge = get_pos_neg_edges('valid', split_edge)
        pos_valid_edge, neg_valid_edge = pos_valid_edge.to(self.device), neg_valid_edge.to(self.device)
        validation_only = eval_metric == 'precision10' and skip_test_eval

        # Configuration selection must not merely ignore a computed test
        # value: it must avoid reading/scoring test edges altogether.
        if not validation_only:
            pos_test_edge, neg_test_edge = get_pos_neg_edges(
                'test', split_edge
            )
            pos_test_edge, neg_test_edge = (
                pos_test_edge.to(self.device),
                neg_test_edge.to(self.device),
            )

            # ----- Edge predictions -----
            pos_valid_pred = self.batch_predict(
                h, pos_valid_edge, batch_size
            )
            neg_valid_pred = self.batch_predict(
                h, neg_valid_edge, batch_size
            )
            pos_test_pred = self.batch_predict(
                h, pos_test_edge, batch_size
            )
            neg_test_pred = self.batch_predict(
                h, neg_test_edge, batch_size
            )

        # ----- Ranks from pos/neg scores -----
        def make_ranks(pos_scores, neg_scores):
            assert pos_scores.numel() > 0, "No positive scores."
            assert neg_scores.numel() % pos_scores.numel() == 0, "Negatives not aligned per positive."
            M = neg_scores.numel() // pos_scores.numel()
            neg_scores = neg_scores.view(-1, M)
            ranks = 1 + (neg_scores > pos_scores.view(-1, 1)).sum(dim=1)
            return ranks, M

        if not validation_only:
            ranks_v, Mv = make_ranks(pos_valid_pred, neg_valid_pred)
            ranks_t, Mt = make_ranks(pos_test_pred, neg_test_pred)

        # ----- Metric helpers -----
        def hits_at_k(ranks, k):      
            return (ranks <= k).float().mean().item()
        
        def mrr(ranks):               
            return (1.0 / ranks.float()).mean().item()
        
        
        def _score_with_predictor(u_emb, all_emb, batch_cands=eval_batch_cands):
            """Score all (source, candidate) pairs using the trained predictor.
            
            This ensures P@K evaluation uses the same scoring function
            the model was trained with, instead of raw dot product.
            
            Args:
                u_emb: Source embeddings [B, D]
                all_emb: All node embeddings [N, D]
                batch_cands: Candidates per batch (controls memory)
            
            Returns:
                Scores [B, N] after sigmoid
            """
            B, D = u_emb.size(0), u_emb.size(1)
            N = all_emb.size(0)
            # Cap pairs per predictor call: u_rep/v_rep and the MLP
            # intermediates each materialize [B*C, D] floats, so B*C must be
            # bounded by memory, not by candidate count alone (B=1024 x
            # C=4096 needs ~12 GB and OOMs on 12 GB GPUs).
            max_pairs = 1 << 19
            batch_cands = max(1, min(batch_cands, max_pairs // max(B, 1)))
            out = []
            for c0 in range(0, N, batch_cands):
                c1 = min(c0 + batch_cands, N)
                cand = all_emb[c0:c1]                        # [C, D]
                C = cand.size(0)
                # Broadcast to pairs (B*C, D) without huge memory spikes
                u_rep = u_emb[:, None, :].expand(B, C, D).reshape(-1, D)
                v_rep = cand[None, :, :].expand(B, C, D).reshape(-1, D)
                s = self.predictor(u_rep, v_rep).squeeze(-1)  # [B*C]
                out.append(torch.sigmoid(s.view(B, -1)))      # [B, C]
            return torch.cat(out, dim=1)                       # [B, N]


        def traditional_precision_at_k(h_emb, train_edges, test_edges, k, chunk_size=eval_chunk_size,
                                       max_nodes=0):
            """
            Optimized Traditional Recommendation Precision@K.
            
            Optimizations vs original:
            1. Only evaluates nodes appearing in test_edges (others always score 0)
            2. Builds train adjacency as sparse CSR (vectorized, no Python loop)
            3. Masks train edges via sparse CSR row slicing (no per-node loop)
            4. Batch topk + vectorized hit counting (no per-row loop)
            
            Args:
                h_emb: Node embeddings [N, D]
                train_edges: Training edges to exclude [E_train, 2]
                test_edges: Test edges (ground truth) [E_test, 2]
                k: Top-K recommendations per node
                chunk_size: Process nodes in chunks for memory efficiency
            
            Returns:
                Precision@K score (micro-averaged across test nodes)
            """

            device = h_emb.device
            n = h_emb.size(0)
            
            # --- 1. Build train adjacency as sparse CSR (vectorized) ---
            te_cpu = train_edges.cpu().numpy()
            row = np.concatenate([te_cpu[:, 0], te_cpu[:, 1]])
            col = np.concatenate([te_cpu[:, 1], te_cpu[:, 0]])
            valid = (row < n) & (col < n)
            row, col = row[valid], col[valid]
            train_mask_csr = sp.csr_matrix(
                (np.ones(len(row), dtype=np.float32), (row, col)),
                shape=(n, n)
            )
            
            # --- 2. Only evaluate nodes appearing in test edges ---
            test_edges_cpu = test_edges.cpu()
            eval_nodes = torch.unique(
                torch.cat([test_edges_cpu[:, 0], test_edges_cpu[:, 1]])
            )
            eval_nodes = eval_nodes[eval_nodes < n].numpy()
            eval_nodes_sorted = np.sort(eval_nodes)

            # Fixed-seed subsample so every eval during a training run scores
            # the same node subset (selection signal stays comparable).
            if max_nodes and len(eval_nodes_sorted) > max_nodes:
                rng = np.random.RandomState(12345)
                eval_nodes_sorted = np.sort(
                    rng.choice(eval_nodes_sorted, size=max_nodes, replace=False))
            
            # Build test-edge lookup as sparse CSR for vectorized hit counting
            te_np = test_edges_cpu.numpy()
            te_row = np.concatenate([te_np[:, 0], te_np[:, 1]])
            te_col = np.concatenate([te_np[:, 1], te_np[:, 0]])
            te_valid = (te_row < n) & (te_col < n)
            te_row, te_col = te_row[te_valid], te_col[te_valid]
            test_edge_csr = sp.csr_matrix(
                (np.ones(len(te_row), dtype=np.float32), (te_row, te_col)),
                shape=(n, n)
            )
            
            total_precision = 0.0
            total_users = 0
            num_eval = len(eval_nodes_sorted)
            num_chunks = (num_eval + chunk_size - 1) // chunk_size
            
            import time as _time
            _t0 = _time.time()
            print(f"  [P@{k}] Evaluating {num_eval} nodes in {num_chunks} chunks...", flush=True)
            
            for start_idx in range(0, num_eval, chunk_size):
                end_idx = min(start_idx + chunk_size, num_eval)
                chunk_node_ids = eval_nodes_sorted[start_idx:end_idx]
                B = len(chunk_node_ids)
                chunk_i = start_idx // chunk_size
                
                chunk_tensor = torch.tensor(chunk_node_ids, dtype=torch.long, device=device)
                h_chunk = h_emb[chunk_tensor]
                scores = _score_with_predictor(h_chunk, h_emb)
                scores_cpu = scores.cpu()
                
                # --- 3. Vectorized masking: self-loops + train edges ---
                scores_cpu[torch.arange(B), torch.tensor(chunk_node_ids, dtype=torch.long)] = float('-inf')
                adj_block = train_mask_csr[chunk_node_ids]
                nb_r, nb_c = adj_block.nonzero()
                scores_cpu[nb_r, nb_c] = float('-inf')
                
                # --- 4. Batch topk + vectorized hit counting ---
                topk_scores, topk_indices = torch.topk(scores_cpu, min(k, n), dim=1)  # [B, k]
                topk_np = topk_indices.numpy()
                
                for i in range(B):
                    u = chunk_node_ids[i]
                    valid_k = min(k, int(torch.isfinite(topk_scores[i]).sum().item()))
                    if valid_k == 0:
                        continue
                    # Vectorized hit counting via sparse CSR lookup
                    cands = topk_np[i, :valid_k]
                    hits = int(test_edge_csr[u, cands].sum())
                    
                    total_precision += hits / valid_k
                    total_users += 1
                
                if (chunk_i + 1) % 5 == 0 or chunk_i == num_chunks - 1:
                    elapsed = _time.time() - _t0
                    pct = (chunk_i + 1) / num_chunks * 100
                    print(f"  [P@{k}] Chunk {chunk_i+1}/{num_chunks} ({pct:.0f}%) - {elapsed:.0f}s elapsed", flush=True)
            
            elapsed = _time.time() - _t0
            avg_precision = total_precision / total_users if total_users > 0 else 0.0
            print(f"  [P@{k}] Done: {avg_precision:.6f} ({total_users} nodes, {elapsed:.0f}s)", flush=True)
            return avg_precision

        def gini_coefficient_1d(x):
            """Calculate Gini coefficient for a 1D tensor."""
            if x.numel() == 0:
                return 0.0
            x = x.float().clamp_min(0)
            s = x.sum()
            if s.item() == 0:
                return 0.0
            x_sorted, _ = torch.sort(x.view(-1))
            n = x_sorted.numel()
            idx = torch.arange(1, n + 1, dtype=x_sorted.dtype, device=x_sorted.device)
            gini = 1.0 + (1.0 / n) - 2.0 * (torch.sum((n + 1 - idx) * x_sorted) / (n * s))
            return float(gini)

        def recommend_topk_for_test_nodes(h_emb, train_edges, test_edges, k, chunk_size=eval_chunk_size):
            """
            Optimized: Generate top-k recommendations ONLY for TEST NODES.
            
            Uses sparse CSR masking and batch topk instead of Python loops.
            
            Args:
                h_emb: Node embeddings [N, D]
                train_edges: Training edges to exclude
                test_edges: Test edges (used to identify test nodes)
                k: Top-K recommendations per node
                chunk_size: Process nodes in chunks
            
            Returns:
                recs_by_u: dict mapping node -> list of recommended nodes
                unique_edges: set of recommended edges (for fairness calculation)
            """

            device = h_emb.device
            n = h_emb.size(0)
            
            # Build training adjacency as sparse CSR (vectorized)
            te_cpu = train_edges.cpu().numpy()
            row = np.concatenate([te_cpu[:, 0], te_cpu[:, 1]])
            col = np.concatenate([te_cpu[:, 1], te_cpu[:, 0]])
            valid = (row < n) & (col < n)
            row, col = row[valid], col[valid]
            train_mask_csr = sp.csr_matrix(
                (np.ones(len(row), dtype=np.float32), (row, col)),
                shape=(n, n)
            )
            
            # Get TEST NODES (nodes that appear in test edges)
            test_edges_cpu = test_edges.cpu()
            test_nodes = torch.unique(
                torch.cat([test_edges_cpu[:, 0], test_edges_cpu[:, 1]])
            )
            test_nodes = sorted(test_nodes[test_nodes < n].numpy().tolist())
            
            if len(test_nodes) == 0:
                return {}, set()
            
            recs_by_u = {}
            unique_edges = set()
            
            # Process test nodes in chunks
            for start_idx in range(0, len(test_nodes), chunk_size):
                end_idx = min(start_idx + chunk_size, len(test_nodes))
                chunk_nodes = test_nodes[start_idx:end_idx]
                chunk_node_ids = np.array(chunk_nodes)
                B = len(chunk_nodes)
                
                chunk_nodes_tensor = torch.tensor(chunk_nodes, dtype=torch.long, device=device)
                h_chunk = h_emb[chunk_nodes_tensor]
                
                scores = _score_with_predictor(h_chunk, h_emb)
                scores_cpu = scores.cpu()
                
                # Vectorized masking: self-loops + train edges
                scores_cpu[torch.arange(B), torch.tensor(chunk_nodes, dtype=torch.long)] = float('-inf')
                adj_block = train_mask_csr[chunk_node_ids]
                nb_r, nb_c = adj_block.nonzero()
                scores_cpu[nb_r, nb_c] = float('-inf')
                
                # Batch topk
                topk_scores, topk_indices = torch.topk(scores_cpu, min(k, n), dim=1)  # [B, k]
                
                for i in range(B):
                    u = chunk_nodes[i]
                    valid_mask = torch.isfinite(topk_scores[i])
                    chosen = topk_indices[i][valid_mask][:k].tolist()
                    recs_by_u[u] = chosen
                    
                    for v in chosen:
                        if u == v:
                            continue
                        a, b = (u, v) if u < v else (v, u)
                        unique_edges.add(frozenset((a, b)))
            
            return recs_by_u, unique_edges

        def distribution_fairness_at_k(h_emb, train_edges, test_edges, k, num_nodes, 
                                    fairness_mode='indegree', chunk_size=1024):
            """
            Evaluates fairness ONLY on TEST NODES (not all nodes).
            
            Args:
                fairness_mode: 'indegree' (exposure), 'outdegree' (activity), or 'total' (participation)
            
            Returns:
                fairness_score: 1 - Gini (higher = more fair)
            """
            # Get recommendations for TEST NODES ONLY
            recs_by_u, unique_edges = recommend_topk_for_test_nodes(
                h_emb, train_edges, test_edges, k, chunk_size
            )
            
            if not recs_by_u:
                return 1.0  # Perfect fairness if no recommendations
            
            # Get test nodes
            test_edges_cpu = test_edges.cpu()
            test_nodes_set = set(test_edges_cpu[:, 0].tolist()) | set(test_edges_cpu[:, 1].tolist())
            test_nodes = [u for u in test_nodes_set if u < num_nodes]
            
            # Find max node ID in recommendations
            nodes_in_test_recs = set()
            for e in unique_edges:
                a, b = tuple(e)
                nodes_in_test_recs.add(a)
                nodes_in_test_recs.add(b)
            
            if not nodes_in_test_recs:
                return 1.0
            
            max_node_id = max(nodes_in_test_recs)
            
            # Calculate degree based on fairness mode
            if fairness_mode == 'indegree':
                # Exposure fairness: count how many times each node is recommended TO
                degree = torch.zeros(max_node_id + 1, dtype=torch.float)
                for u in test_nodes:
                    if u not in recs_by_u:
                        continue
                    for v in recs_by_u[u][:k]:
                        if v <= max_node_id:
                            degree[v] += 1
                
                # Only consider nodes that received recommendations
                nonzero_deg = degree[degree > 0]
                
            elif fairness_mode == 'outdegree':
                # Activity fairness: count how many recommendations each node makes
                degree = torch.zeros(max_node_id + 1, dtype=torch.float)
                for u in test_nodes:
                    if u not in recs_by_u:
                        continue
                    if u <= max_node_id:
                        degree[u] += len(recs_by_u[u][:k])
                
                # Only consider test nodes (sources)
                nonzero_deg = degree[degree > 0]
                
            elif fairness_mode == 'total':
                # Participation fairness: count total degree (both in and out)
                degree = torch.zeros(max_node_id + 1, dtype=torch.float)
                for e in unique_edges:
                    a, b = tuple(e)
                    if a <= max_node_id:
                        degree[a] += 1
                    if b <= max_node_id:
                        degree[b] += 1
                
                # Only consider nodes in the test recommendation network
                nonzero_deg = degree[degree > 0]
            
            else:
                raise ValueError(f"Unknown fairness_mode: {fairness_mode}")
            
            # Calculate Gini and fairness score
            if nonzero_deg.numel() == 0:
                return 1.0  # Perfect fairness if no valid degrees
            
            gini = gini_coefficient_1d(nonzero_deg)
            fairness = 1.0 - gini
            
            print(f"Fairness ({fairness_mode}): {nonzero_deg.numel()} nodes, "
                f"mean degree: {nonzero_deg.mean():.2f}, Gini: {gini:.4f}, "
                f"Fairness: {fairness:.4f}")
            
            return fairness

        # ----- Compute metrics based on eval_metric -----
        results = {}

        # Fast mode: only compute Precision@10 (for training validation)
        if eval_metric == 'precision10':
            h_clean = h[:-1]
            train_edges = split_edge['train']['edge'].to(self.device)
            print("\n  === Validation P@10 ===", flush=True)
            val_prec = traditional_precision_at_k(h_clean, train_edges, pos_valid_edge, 10,
                                                  max_nodes=max_eval_nodes)
            if validation_only:
                test_prec = float('nan')
            else:
                print("  === Test P@10 ===", flush=True)
                test_prec = traditional_precision_at_k(
                    h_clean, train_edges, pos_test_edge, 10
                )
            results['Precision@10'] = (val_prec, test_prec)
            return results

        Ks = [10, 20, 50, 100]

        want_hits      = eval_metric in ('hits', 'all')
        want_precision = eval_metric in ('precision', 'all')
        want_fairness  = eval_metric in ('fairness', 'all')
        want_mrr       = eval_metric in ('mrr', 'all')

        if want_hits:
            for K in Ks:
                results[f'Hits@{K}'] = (hits_at_k(ranks_v, K), hits_at_k(ranks_t, K))

        if want_precision:
            h_clean = h[:-1]  # Remove mean embedding
            train_edges = split_edge['train']['edge'].to(self.device)
            
            for K in Ks:
                #print(f"Computing Precision@{K} for validation...")
                val_prec = traditional_precision_at_k(
                    h_clean, train_edges, pos_valid_edge, K
                )
                #print(f"Computing Precision@{K} for test...")
                test_prec = traditional_precision_at_k(
                    h_clean, train_edges, pos_test_edge, K
                )
                results[f'Precision@{K}'] = (val_prec, test_prec)

        if want_fairness:
            h_clean = h[:-1]  # Remove mean embedding
            train_edges = split_edge['train']['edge'].to(self.device)
            num_nodes = self.num_nodes
            
            for K in Ks:
                #print(f"Computing Distribution Fairness@{K} ({fairness_mode}) for validation...")
                val_dist_fair = distribution_fairness_at_k(
                    h_clean, train_edges, pos_valid_edge, K, num_nodes, 
                    fairness_mode=fairness_mode
                )
                #print(f"Computing Distribution Fairness@{K} ({fairness_mode}) for test...")
                test_dist_fair = distribution_fairness_at_k(
                    h_clean, train_edges, pos_test_edge, K, num_nodes,
                    fairness_mode=fairness_mode
                )
                results[f'Distribution_Fairness@{K}'] = (val_dist_fair, test_dist_fair)

        if want_mrr:
            results['MRR'] = (mrr(ranks_v), mrr(ranks_t))

        # Fallback
        if not results:
            for K in Ks:
                results[f'Hits@{K}'] = (hits_at_k(ranks_v, K), hits_at_k(ranks_t, K))

        return results

def create_input_layer(num_nodes, num_node_feats, hidden_channels, use_node_feats=True,
                       train_node_emb=False, pretrain_emb=None):
    emb = None
    if use_node_feats:
        input_dim = num_node_feats
        if train_node_emb:
            emb = torch.nn.Embedding(num_nodes, hidden_channels)
            input_dim += hidden_channels
        elif pretrain_emb is not None and pretrain_emb != '':
            weight = torch.load(pretrain_emb)
            emb = torch.nn.Embedding.from_pretrained(weight)
            input_dim += emb.weight.size(1)
    else:
        if pretrain_emb is not None and pretrain_emb != '':
            weight = torch.load(pretrain_emb)
            emb = torch.nn.Embedding.from_pretrained(weight)
            input_dim = emb.weight.size(1)
        else:
            emb = torch.nn.Embedding(num_nodes, hidden_channels)
            input_dim = hidden_channels
    return input_dim, emb


def create_gnn_layer(input_channels, hidden_channels, num_layers, dropout, encoder_name, alpha, init):
    if encoder_name.upper() == 'HLGNN':
        return HLGNN(input_channels, hidden_channels, hidden_channels, num_layers, dropout, alpha, init)
    elif encoder_name.upper() == 'GCN':
        return GCN(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
    elif encoder_name.upper() == 'WSAGE':
        return WSAGE(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
    elif encoder_name.upper() == 'TRANSFORMER':
        return Transformer(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
    else:
        return SAGE(input_channels, hidden_channels, hidden_channels, num_layers, dropout)


def create_predictor_layer(hidden_channels, num_layers, dropout=0, predictor_name='MLP'):
    predictor_name = predictor_name.upper()
    if predictor_name == 'DOT':
        return DotPredictor()
    elif predictor_name == 'BIL':
        return BilinearPredictor(hidden_channels)
    elif predictor_name == 'MLP':
        return MLPPredictor(hidden_channels, hidden_channels, 1, num_layers, dropout)
    elif predictor_name == 'MLPCAT':
        return MLPCatPredictor(hidden_channels, hidden_channels, 1, num_layers, dropout)
