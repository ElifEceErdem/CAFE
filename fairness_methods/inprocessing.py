import torch

try:
    from backbone.model import BaseModel
    from backbone.utils import get_pos_neg_edges, get_pos_neg_edges_fair
    from backbone.loss import auc_loss, ce_loss, info_nce_loss, hinge_auc_loss
    from torch.utils.data import DataLoader
except ImportError:
    # Fallback if running from a different context
    BaseModel = object

try:
    from torch_sparse import SparseTensor
except ImportError:
    SparseTensor = None

# ==================== FAIRNESS-AWARE LOSS FUNCTIONS ====================

class FairnessAwareLoss:
    """
    Base class for fairness-aware loss functions.
    Combines task loss (link prediction) with fairness penalty.

    penalty_reduction controls how the fairness penalty is reduced over the
    batch, RELATIVE to the base loss:

      'mean' (default, historical behaviour)
          penalty = mean over positives, while every base loss in
          backbone/loss.py reduces with .sum(). The two terms therefore differ
          in scale by |E+_batch| (16384 by default), so lambda_fair is divided
          by the batch size in effect and the penalty is numerically inert
          (measured ratio lambda_fair*penalty / base_loss ~ 1e-5..1e-4).

      'sum'
          penalty is multiplied by |E+_batch| so that BOTH terms are sums over
          the same positives. lambda_fair then means what Eq. (8) of the paper
          says it means: a per-example trade-off weight between the task loss
          and the fairness penalty.

    IMPORTANT: 'sum' is only meaningful when the base loss is itself
    sum-reduced. In backbone/loss.py the AUC family (auc, hinge_auc, and the
    weighted/adaptive variants) reduces with .sum(), but ce_loss, info_nce_loss
    and log_rank_loss reduce with .mean(). Pairing 'sum' with a mean-reduced
    base loss would inflate the penalty by |E+_batch| relative to a base loss
    that was never scaled that way, so it is rejected at construction.

    The default is 'mean', which leaves the historical loss formula untouched.
    """

    #: base losses reachable from compute_base_loss() that reduce with .mean()
    _MEAN_REDUCED_BASE_LOSSES = frozenset({'CE', 'InfoNCE', 'LogRank'})

    def __init__(self, base_loss_name='AUC', lambda_fair=0.1,
                 penalty_reduction='mean'):
        if penalty_reduction not in ('mean', 'sum'):
            raise ValueError(
                f"penalty_reduction must be 'mean' or 'sum', got {penalty_reduction!r}")
        if (penalty_reduction == 'sum'
                and base_loss_name in self._MEAN_REDUCED_BASE_LOSSES):
            raise ValueError(
                f"penalty_reduction='sum' is incompatible with base loss "
                f"'{base_loss_name}', which reduces with .mean() "
                f"(backbone/loss.py). 'sum' scales the penalty by the batch "
                f"size to match a SUM-reduced base loss; use it only with the "
                f"AUC family (AUC, HingeAUC, ...), or leave "
                f"penalty_reduction='mean'.")
        self.base_loss_name = base_loss_name
        self.lambda_fair = lambda_fair
        self.penalty_reduction = penalty_reduction
        self._warned_num_neg = False

    def compute_base_loss(self, pos_out, neg_out, num_neg):
        """Compute the base link prediction loss"""
        if self.base_loss_name == 'CE':
            return ce_loss(pos_out, neg_out)
        elif self.base_loss_name == 'InfoNCE':
            return info_nce_loss(pos_out, neg_out, num_neg)
        elif self.base_loss_name == 'HingeAUC':
            return hinge_auc_loss(pos_out, neg_out, num_neg)
        else:  # AUC
            return auc_loss(pos_out, neg_out, num_neg)

    def compute_fairness_penalty(self, pos_edge, neg_edge, pos_out, neg_out, **kwargs):
        """To be implemented by subclasses"""
        raise NotImplementedError

    def __call__(self, pos_out, neg_out, num_neg, pos_edge, neg_edge, **kwargs):
        """
        Compute total loss = base_loss + lambda * fairness_penalty

        Args:
            pos_out: Predictions for positive edges [batch_size]
            neg_out: Predictions for negative edges [batch_size * num_neg]
            num_neg: Number of negative samples per positive
            pos_edge: Positive edges [batch_size, 2]
            neg_edge: Negative edges [batch_size, num_neg, 2]
            **kwargs: Additional arguments (e.g., degrees)
        """
        base_loss = self.compute_base_loss(pos_out, neg_out, num_neg)
        fairness_penalty = self.compute_fairness_penalty(
            pos_edge, neg_edge, pos_out, neg_out, **kwargs
        )

        if self.penalty_reduction == 'sum':
            # Match the base loss's .sum() reduction over the same positives,
            # so lambda_fair is a per-example weight rather than a per-batch one.
            #
            # NOTE: the scaling is PER POSITIVE. The AUC family sums over
            # B * num_neg (positive, negative) comparisons, so with num_neg > 1
            # the base loss grows with num_neg while this term does not, and the
            # effective trade-off implied by a given lambda_fair shifts. The
            # corrected reduction is defined for num_neg = 1; anything else
            # needs an explicit decision on whether the fairness term is
            # weighted per positive or per comparison.
            if num_neg not in (1, None):
                raise ValueError(
                    f"penalty_reduction='sum' requires num_neg=1, got {num_neg}. "
                    f"The penalty is scaled per POSITIVE, while the AUC base loss "
                    f"sums over num_neg comparisons per positive, so the intended "
                    f"trade-off would silently depend on num_neg. Using num_neg>1 "
                    f"needs an explicit decision on whether the fairness term is "
                    f"weighted per positive or per positive-negative comparison.")
            fairness_penalty = fairness_penalty * pos_out.size(0)

        total_loss = base_loss + self.lambda_fair * fairness_penalty

        return total_loss, base_loss.item(), fairness_penalty.item()


class PAFairnessLoss(FairnessAwareLoss):
    """
    Preferential Attachment (PA) based fairness loss.

    Penalizes predictions on high-degree node pairs to reduce rich-get-richer effect.

    Intuition:
    - High scores on high-degree pairs -> reinforce existing hubs
    - Penalty: score * PA_score(u,v) where PA_score = sqrt(deg_u * deg_v)
    """

    def __init__(self, base_loss_name='AUC', lambda_fair=0.1,
                 pa_aggregation='product', normalize=True):
        super().__init__(base_loss_name, lambda_fair)
        self.pa_aggregation = pa_aggregation
        self.normalize = normalize
        self.max_pa_score = None

    def compute_pa_score(self, degrees, edges):
        """
        Compute PA score for edges.

        Args:
            degrees: Node degrees [num_nodes]
            edges: Edge indices [num_edges, 2]

        Returns:
            PA scores [num_edges]
        """
        deg_u = degrees[edges[:, 0]]
        deg_v = degrees[edges[:, 1]]

        if self.pa_aggregation == 'product':
            pa_scores = torch.sqrt(deg_u.float() * deg_v.float())
        elif self.pa_aggregation == 'sum':
            pa_scores = (deg_u + deg_v).float()
        elif self.pa_aggregation == 'max':
            pa_scores = torch.maximum(deg_u, deg_v).float()
        elif self.pa_aggregation == 'log_product':
            pa_scores = torch.log1p(deg_u.float() * deg_v.float())
        else:
            raise ValueError(f"Unknown pa_aggregation: {self.pa_aggregation}")

        # Normalize to [0, 1]
        if self.normalize and self.max_pa_score is not None:
            pa_scores = pa_scores / (self.max_pa_score + 1e-8)

        return pa_scores

    def compute_fairness_penalty(self, pos_edge, neg_edge, pos_out, neg_out, degrees=None, **kwargs):
        """
        Compute PA-based fairness penalty.

        Penalty = mean(|predictions| * PA_scores)

        This encourages the model to give lower scores to high-degree pairs.
        """
        if degrees is None:
            raise ValueError("degrees required for PA fairness loss")

        # Initialize max PA score if not set
        if self.max_pa_score is None:
            max_deg = degrees.max().float()
            if self.pa_aggregation == 'product':
                self.max_pa_score = torch.sqrt(max_deg * max_deg)
            elif self.pa_aggregation == 'sum':
                self.max_pa_score = max_deg + max_deg
            elif self.pa_aggregation == 'max':
                self.max_pa_score = max_deg
            elif self.pa_aggregation == 'log_product':
                self.max_pa_score = torch.log1p(max_deg * max_deg)

        # Compute PA scores for positive edges
        pa_pos = self.compute_pa_score(degrees, pos_edge)

        # Penalty: penalize high scores on high-PA pairs
        pos_probs = torch.sigmoid(pos_out)
        penalty_pos = (pos_probs * pa_pos).mean()

        # We want to reduce scores on high-PA pairs
        total_penalty = penalty_pos

        return total_penalty


class ExposureFairnessLoss(FairnessAwareLoss):
    """
    Exposure-based fairness loss.

    Penalizes repeated high scores to frequently-predicted nodes.
    Maintains running exposure counts during training.

    Intuition:
    - Track how often each node is predicted with high score
    - Penalize models that keep predicting same popular nodes
    """

    def __init__(self, base_loss_name='AUC', lambda_fair=0.1,
                 num_nodes=None, exposure_decay=1.0):
        super().__init__(base_loss_name, lambda_fair)
        self.num_nodes = num_nodes
        self.exposure_decay = exposure_decay

        # Exposure tracking (momentum-based)
        self.node_exposure = None
        self.max_exposure = 1.0

    def reset_exposure(self, device):
        """Reset exposure counters (call at start of each epoch)"""
        if self.num_nodes is None:
            raise ValueError("num_nodes must be set")
        self.node_exposure = torch.zeros(self.num_nodes, device=device)
        self.max_exposure = 1.0

    def update_exposure(self, edges, scores):
        """
        Update exposure counts based on predictions.

        Args:
            edges: Edge indices [num_edges, 2]
            scores: Prediction scores [num_edges]
        """
        if self.node_exposure is None:
            self.reset_exposure(edges.device)

        # Decay previous exposure (momentum)
        self.node_exposure *= self.exposure_decay

        # Add new exposure (weighted by prediction score)
        probs = torch.sigmoid(scores).detach()

        # Accumulate exposure for destination nodes
        dst = edges[:, 1]

        # Ensure we only update valid nodes (safety check matching original logic)
        mask = dst < self.num_nodes

        if mask.all():
            self.node_exposure.index_add_(0, dst, probs.view(-1))
        else:
            self.node_exposure.index_add_(0, dst[mask], probs.view(-1)[mask])

        # Update max for normalization
        self.max_exposure = max(self.max_exposure, self.node_exposure.max().item())

    def compute_fairness_penalty(self, pos_edge, neg_edge, pos_out, neg_out, **kwargs):
        """
        Compute exposure-based fairness penalty.

        Penalty = mean(predictions * normalized_exposure)

        This discourages high scores to nodes with high exposure.
        """
        if self.node_exposure is None:
            self.reset_exposure(pos_edge.device)

        # Get normalized exposure for edge destinations
        exposure_norm = self.node_exposure / (self.max_exposure + 1e-8)

        # Positive edges
        pos_dst = pos_edge[:, 1]
        exp_pos = exposure_norm[pos_dst]

        # Negative edges
        neg_edge_flat = neg_edge.view(-1, 2)
        neg_dst = neg_edge_flat[:, 1]
        exp_neg = exposure_norm[neg_dst]

        # Penalty: high scores to high-exposure nodes
        pos_probs = torch.sigmoid(pos_out)
        neg_probs = torch.sigmoid(neg_out)

        penalty_pos = (pos_probs * exp_pos).mean()
        penalty_neg = (neg_probs * exp_neg).mean()

        total_penalty = penalty_pos + 0.1 * penalty_neg

        # Update exposure for next batch
        self.update_exposure(pos_edge, pos_out)
        self.update_exposure(neg_edge_flat, neg_out)

        return total_penalty


class HybridFairnessLoss(FairnessAwareLoss):
    """
    Hybrid fairness loss combining PA and Exposure penalties.
    """

    def __init__(self, base_loss_name='AUC', lambda_fair=0.1, pa_aggregation='product', exposure_decay=1.0, normalize=True, num_nodes=None,
                 pa_weight=0.5, exposure_weight=0.5, **kwargs):
        super().__init__(base_loss_name, lambda_fair)
        self.pa_weight = pa_weight
        self.exposure_weight = exposure_weight
        self.num_nodes = num_nodes

        # PA part: DOES NOT need num_nodes
        self.pa_loss = PAFairnessLoss(
            base_loss_name=base_loss_name,
            lambda_fair=lambda_fair,
            pa_aggregation=pa_aggregation,
            normalize=normalize
        )

        # Exposure part: DOES need num_nodes
        self.exposure_loss = ExposureFairnessLoss(
            base_loss_name=base_loss_name,
            lambda_fair=lambda_fair,
            num_nodes=num_nodes,
            exposure_decay=exposure_decay
        )

    def reset_exposure(self, device):
        self.exposure_loss.reset_exposure(device)

    def compute_fairness_penalty(self, pos_edge, neg_edge, pos_out, neg_out, **kwargs):
        """Combine PA and Exposure penalties"""
        pa_penalty = self.pa_loss.compute_fairness_penalty(
            pos_edge, neg_edge, pos_out, neg_out, **kwargs
        )
        exposure_penalty = self.exposure_loss.compute_fairness_penalty(
            pos_edge, neg_edge, pos_out, neg_out, **kwargs
        )

        total_penalty = (self.pa_weight * pa_penalty +
                         self.exposure_weight * exposure_penalty)

        return total_penalty


# ==================== FAIRNESS-AWARE MODEL ====================

class FairnessAwareModel(BaseModel):
    """
    Extension of BaseModel with fairness-aware training.

    Supports:
    - inproc_pa: PAFairnessLoss
    - inproc_exposure: ExposureFairnessLoss
    - inproc_hybrid: HybridFairnessLoss
    - coreness_inverse_inproc: CorenessInverseLoss (suppressive)
    - mobility_boost_inproc: MobilityBoostLoss (promotive)
    - none: No fairness loss (baseline)
    """

    def __init__(self, fairness_method='none', lambda_fair=0.1,
                 pa_aggregation='product', num_nodes=None,
                 exposure_decay=1.0, **kwargs):

        # Store fairness-specific parameters BEFORE calling super().__init__()
        self.fairness_method = fairness_method
        self.exposure_decay = exposure_decay
        self.pa_aggregation = pa_aggregation

        # Extract coreness_alpha before passing kwargs to BaseModel
        self.coreness_alpha = kwargs.pop('coreness_alpha', 1.0)
        # 'mean' (default) reproduces the historical behaviour exactly; 'sum'
        # matches the base loss's reduction. See FairnessAwareLoss.
        self.penalty_reduction = kwargs.pop('penalty_reduction', 'mean')

        # Now call parent __init__ with ONLY the kwargs it expects
        super().__init__(num_nodes=num_nodes, **kwargs)

        # Initialize fairness loss AFTER parent init
        if fairness_method == 'inproc_pa':
            self.fairness_loss = PAFairnessLoss(
                base_loss_name=kwargs['loss_func'],
                lambda_fair=lambda_fair,
                pa_aggregation=pa_aggregation
            )
        elif fairness_method == 'inproc_exposure':
            self.fairness_loss = ExposureFairnessLoss(
                base_loss_name=kwargs['loss_func'],
                lambda_fair=lambda_fair,
                num_nodes=num_nodes,
                exposure_decay=exposure_decay
            )
        elif fairness_method == 'inproc_hybrid':
            self.fairness_loss = HybridFairnessLoss(
                base_loss_name=kwargs['loss_func'],
                lambda_fair=lambda_fair,
                pa_aggregation=pa_aggregation,
                num_nodes=num_nodes,
                exposure_decay=exposure_decay
            )
        elif fairness_method == 'coreness_inverse_inproc':
            self.fairness_loss = CorenessInverseLoss(
                base_loss_name=kwargs['loss_func'],
                lambda_fair=lambda_fair,
                alpha=self.coreness_alpha,
                penalty_reduction=self.penalty_reduction
            )
        elif fairness_method == 'mobility_boost_inproc':
            self.fairness_loss = MobilityBoostLoss(
                base_loss_name=kwargs['loss_func'],
                lambda_fair=lambda_fair,
                alpha=self.coreness_alpha,
                penalty_reduction=self.penalty_reduction
            )
        else:
            self.fairness_loss = None

    def train_with_fairness(self, data, split_edge, batch_size, neg_sampler_name, num_neg, degrees=None, coreness=None):
        """
        Training loop with fairness-aware loss.
        """
        self.encoder.train()
        self.predictor.train()

        # Reset exposure if using exposure-based method
        if self.fairness_method in ['inproc_exposure', 'inproc_hybrid']:
            self.fairness_loss.reset_exposure(self.device)

        # Use fair sampling if available
        try:
             pos_train_edge, neg_train_edge = get_pos_neg_edges_fair(
                'train', split_edge,
                edge_index=data.edge_index,
                num_nodes=self.num_nodes,
                neg_sampler_name=neg_sampler_name,
                num_neg=num_neg,
                coreness=coreness
            )
        except NameError:
             pos_train_edge, neg_train_edge = get_pos_neg_edges(
                'train', split_edge,
                edge_index=data.edge_index,
                num_nodes=self.num_nodes,
                neg_sampler_name=neg_sampler_name,
                num_neg=num_neg
            )

        pos_train_edge = pos_train_edge.to(self.device)
        neg_train_edge = neg_train_edge.to(self.device)

        if degrees is not None:
            degrees = degrees.to(self.device)

        if coreness is not None:
            coreness = coreness.to(self.device)

        total_loss = total_base_loss = total_fair_penalty = 0
        total_examples = 0

        for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
            self.optimizer.zero_grad()

            input_feat = self.create_input_feat(data)
            h = self.encoder(input_feat, data.adj_t, data.edge_weight)

            # Get edges for this batch
            pos_edge = pos_train_edge[perm]
            neg_edge = neg_train_edge[perm]

            # Predictions
            pos_out = self.predictor(h[pos_edge[:, 0]], h[pos_edge[:, 1]])
            neg_edge_flat = neg_edge.view(-1, 2)
            neg_out = self.predictor(h[neg_edge_flat[:, 0]], h[neg_edge_flat[:, 1]])

            # Compute loss
            if self.fairness_loss is not None:
                loss, base_loss, fair_penalty = self.fairness_loss(
                    pos_out, neg_out, num_neg,
                    pos_edge, neg_edge,
                    degrees=degrees,
                    coreness=coreness
                )

                total_base_loss += base_loss * pos_out.size(0)
                total_fair_penalty += fair_penalty * pos_out.size(0)
            else:
                base_loss = self.calculate_loss(pos_out, neg_out, num_neg)
                loss = base_loss

                total_base_loss += base_loss.item() * pos_out.size(0)

            # Backward pass
            loss.backward()

            if self.clip_norm >= 0:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_norm)
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), self.clip_norm)

            self.optimizer.step()

            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

        avg_loss = total_loss / total_examples
        avg_base = total_base_loss / total_examples
        avg_fair = total_fair_penalty / total_examples if self.fairness_loss else 0

        return avg_loss, avg_base, avg_fair


def apply_coreness_inverse_aggregation(adj_t, coreness, alpha=1.0):
    """
    Modifies the adjacency matrix weights to penalize edges connected to high-coreness nodes.

    A'_uv = A_uv / (Core(u) * Core(v))^alpha

    Args:
        adj_t: SparseTensor [num_nodes, num_nodes]
        coreness: Tensor [num_nodes]
        alpha: Penalty strength

    Returns:
        Modified SparseTensor
    """
    if SparseTensor is None:
        raise ImportError("torch_sparse not installed")

    # Get rows, cols, and values
    row, col, val = adj_t.coo()

    if val is None:
        val = torch.ones(row.size(0), device=row.device)

    # Get coreness for each edge
    core_u = coreness[row].float()
    core_v = coreness[col].float()

    # Avoid division by zero (coreness >= 1 typically)
    core_u = torch.clamp(core_u, min=1.0)
    core_v = torch.clamp(core_v, min=1.0)

    denominator = (core_u * core_v).pow(alpha)

    # Apply penalty
    new_val = val / denominator

    # Create new SparseTensor
    new_adj_t = SparseTensor(row=row, col=col, value=new_val,
                             sparse_sizes=adj_t.sparse_sizes())

    return new_adj_t


def apply_mobility_boost_aggregation(adj_t, coreness, alpha=1.0):
    """
    Modifies the adjacency matrix weights to boost edges connected to low-coreness (periphery) nodes.

    A'_uv = A_uv * (1 + alpha * mobility(u) * mobility(v))
    where mobility(v) = 1 / log(percentile(Core(v)) * 100 + e)

    This is the pre-processing counterpart of the post-processing mobility boost.

    Args:
        adj_t: SparseTensor [num_nodes, num_nodes]
        coreness: Tensor [num_nodes]
        alpha: Boost strength

    Returns:
        Modified SparseTensor
    """
    if SparseTensor is None:
        raise ImportError("torch_sparse not installed")

    # Get rows, cols, and values
    row, col, val = adj_t.coo()

    if val is None:
        val = torch.ones(row.size(0), device=row.device)

    # Compute percentile-based coreness (matching post-processing approach)
    coreness_float = coreness.float()
    num_nodes = coreness.size(0)

    # Rank-based percentile: rank / N
    sorted_vals, sorted_idx = torch.sort(coreness_float)
    ranks = torch.zeros_like(coreness_float)
    ranks[sorted_idx] = torch.arange(1, num_nodes + 1, dtype=torch.float32, device=coreness.device)
    percentile = ranks / num_nodes

    # Mobility weight: 1 / log(percentile * 100 + e)
    # High for periphery (low percentile), low for core (high percentile)
    scaling_factor = 100.0
    e = 2.71828
    mobility = 1.0 / torch.log(percentile * scaling_factor + e).pow(alpha)

    # Get mobility for each edge endpoint
    mob_u = mobility[row]
    mob_v = mobility[col]

    # Normalize the product so max boost is bounded
    mob_product = mob_u * mob_v
    max_mob = mob_product.max()
    if max_mob > 0:
        mob_product_norm = mob_product / max_mob  # [0, 1]
    else:
        mob_product_norm = mob_product

    # Apply boost: A' = A * (1 + alpha * normalized_mobility)
    multiplier = 1.0 + alpha * mob_product_norm
    new_val = val * multiplier

    # Create new SparseTensor
    new_adj_t = SparseTensor(row=row, col=col, value=new_val,
                             sparse_sizes=adj_t.sparse_sizes())

    return new_adj_t


class CorenessInverseLoss(FairnessAwareLoss):
    """
    Coreness-Inverse in-processing fairness loss (Suppressive approach).

    Penalizes high prediction scores on edges between high-coreness nodes,
    encouraging the model to allocate score capacity to periphery nodes.

    Penalty = mean(σ(pos_scores) * coreness_weight(u,v))
    where coreness_weight = (Core(u) * Core(v))^alpha / max_product^alpha
    """

    def __init__(self, base_loss_name='AUC', lambda_fair=0.1, alpha=1.0,
                 penalty_reduction='mean'):
        super().__init__(base_loss_name, lambda_fair, penalty_reduction)
        self.alpha = alpha
        self.max_coreness_product = None

    def compute_fairness_penalty(self, pos_edge, neg_edge, pos_out, neg_out, coreness=None, **kwargs):
        """
        Compute coreness-inverse penalty.

        Penalty = mean(σ(score) * normalized_coreness_product)
        """
        if coreness is None:
            raise ValueError("coreness required for CorenessInverseLoss")

        # Get coreness for source and destination of positive edges
        core_u = coreness[pos_edge[:, 0]].float()
        core_v = coreness[pos_edge[:, 1]].float()

        # Clamp to avoid zero (coreness >= 1 typically)
        core_u = torch.clamp(core_u, min=1.0)
        core_v = torch.clamp(core_v, min=1.0)

        # Compute coreness product weight
        core_product = (core_u * core_v).pow(self.alpha)

        # Initialize max for normalization
        if self.max_coreness_product is None:
            max_core = coreness.float().max()
            self.max_coreness_product = (max_core * max_core).pow(self.alpha)

        # Normalize to [0, 1]
        core_weight = core_product / (self.max_coreness_product + 1e-8)

        # Penalty: high sigmoid scores on high-coreness pairs
        pos_probs = torch.sigmoid(pos_out)
        penalty = (pos_probs * core_weight).mean()

        return penalty


class MobilityBoostLoss(FairnessAwareLoss):
    """
    Mobility Boost in-processing fairness loss (Promotive approach).

    Penalizes the model for giving LOW scores to edges involving periphery nodes,
    encouraging higher predictions for periphery connections.

    Penalty = mean((1 - σ(pos_scores)) * mobility_weight(u,v))
    where mobility_weight = 1/log(percentile*100 + e)^alpha (normalized)

    This mirrors the post-processing mobility boost formula.
    """

    def __init__(self, base_loss_name='AUC', lambda_fair=0.1, alpha=1.0,
                 penalty_reduction='mean'):
        super().__init__(base_loss_name, lambda_fair, penalty_reduction)
        self.alpha = alpha
        self._mobility_cache = None  # Cache mobility weights per coreness tensor
        self._cache_id = None

    def _compute_mobility_weights(self, coreness):
        """Compute percentile-based mobility weights (cached)."""
        cache_id = id(coreness)
        if self._cache_id == cache_id and self._mobility_cache is not None:
            return self._mobility_cache

        coreness_float = coreness.float()
        num_nodes = coreness.size(0)

        # Rank-based percentile
        sorted_vals, sorted_idx = torch.sort(coreness_float)
        ranks = torch.zeros_like(coreness_float)
        ranks[sorted_idx] = torch.arange(1, num_nodes + 1, dtype=torch.float32, device=coreness.device)
        percentile = ranks / num_nodes

        # Mobility: 1 / log(percentile * 100 + e)^alpha
        scaling_factor = 100.0
        e = 2.71828
        mobility = 1.0 / torch.log(percentile * scaling_factor + e).pow(self.alpha)

        # Normalize to [0, 1]
        max_mob = mobility.max()
        if max_mob > 0:
            mobility = mobility / max_mob

        self._mobility_cache = mobility
        self._cache_id = cache_id
        return mobility

    def compute_fairness_penalty(self, pos_edge, neg_edge, pos_out, neg_out, coreness=None, **kwargs):
        """
        Compute mobility boost penalty.

        Penalty = mean((1 - σ(score)) * mobility_weight)
        High penalty when periphery edges get low scores.
        """
        if coreness is None:
            raise ValueError("coreness required for MobilityBoostLoss")

        mobility = self._compute_mobility_weights(coreness)

        # Get mobility weights for edge endpoints (use product for consistency with pre-proc)
        mob_u = mobility[pos_edge[:, 0]]
        mob_v = mobility[pos_edge[:, 1]]
        edge_mobility = mob_u * mob_v

        # Penalty: low sigmoid scores on high-mobility (periphery) pairs
        pos_probs = torch.sigmoid(pos_out)
        penalty = ((1.0 - pos_probs) * edge_mobility).mean()

        return penalty
