"""Shared graph construction for Equality Gain metrics.

The paper defines both the observed and post-recommendation graphs as
simple, undirected set unions.  Dataset splits store each undirected edge
once, while recommendation lists are directional and can contain both
``u -> v`` and ``v -> u``.  All EG implementations must therefore pass
through the helpers in this module before computing graph centralities.
"""

from typing import Dict, Iterable, List, Optional

import torch


def edge_tensor_2col(edges: torch.Tensor) -> torch.Tensor:
    """Return a detached CPU ``[E, 2]`` long tensor.

    ``split_edge`` tensors use ``[E, 2]`` in this repository.  Supporting
    ``[2, E]`` as well makes the boundary explicit and prevents accidental
    row/column interpretation bugs.
    """
    edge_tensor = torch.as_tensor(edges).detach().cpu()
    if edge_tensor.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long)
    if edge_tensor.ndim != 2:
        raise ValueError(
            f"Edge tensor must have two dimensions, got {edge_tensor.ndim}"
        )
    if edge_tensor.shape[1] == 2:
        pass
    elif edge_tensor.shape[0] == 2:
        edge_tensor = edge_tensor.t()
    else:
        raise ValueError(
            "Edge tensor must have shape [E, 2] or [2, E], "
            f"got {tuple(edge_tensor.shape)}"
        )
    return edge_tensor.to(dtype=torch.long).contiguous()


def canonicalize_undirected_edges(
    edges: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Validate and canonicalize an edge list as a simple undirected set.

    Endpoints are ordered as ``(min(u, v), max(u, v))``.  Self-loops and
    duplicate/reverse rows are removed.  Invalid node identifiers raise an
    error rather than being silently discarded.
    """
    if num_nodes < 0:
        raise ValueError(f"num_nodes must be non-negative, got {num_nodes}")

    edge_tensor = edge_tensor_2col(edges)
    if edge_tensor.numel() == 0:
        return edge_tensor

    invalid = (
        (edge_tensor[:, 0] < 0)
        | (edge_tensor[:, 0] >= num_nodes)
        | (edge_tensor[:, 1] < 0)
        | (edge_tensor[:, 1] >= num_nodes)
    )
    if invalid.any():
        first_bad = edge_tensor[invalid][0].tolist()
        raise ValueError(
            f"Edge endpoint outside [0, {num_nodes}): {tuple(first_bad)}"
        )

    edge_tensor = edge_tensor[edge_tensor[:, 0] != edge_tensor[:, 1]]
    if edge_tensor.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long)

    lower = torch.minimum(edge_tensor[:, 0], edge_tensor[:, 1])
    upper = torch.maximum(edge_tensor[:, 0], edge_tensor[:, 1])
    keys = lower * num_nodes + upper
    unique_keys = torch.unique(keys, sorted=True)
    return torch.stack(
        (torch.div(unique_keys, num_nodes, rounding_mode="floor"),
         unique_keys.remainder(num_nodes)),
        dim=1,
    )


def merge_undirected_edges(
    edge_sources: Iterable[torch.Tensor],
    num_nodes: int,
) -> torch.Tensor:
    """Return the simple undirected set union of multiple edge sources."""
    normalized = [edge_tensor_2col(edges) for edges in edge_sources]
    nonempty = [edges for edges in normalized if edges.numel() > 0]
    if not nonempty:
        return torch.empty((0, 2), dtype=torch.long)
    return canonicalize_undirected_edges(torch.cat(nonempty, dim=0), num_nodes)


def recommendations_to_edges(
    recs_by_u: Dict[int, List[int]],
    num_nodes: int,
    k: Optional[int] = None,
) -> torch.Tensor:
    """Convert directional recommendation events to a validated edge list.

    Canonicalization is intentionally left to the caller so event counts can
    still be inspected before the recommendations materialize as graph edges.
    """
    sources, targets = [], []
    for raw_u, values in recs_by_u.items():
        u = int(raw_u)
        selected = values if k is None else values[:k]
        for raw_v in selected:
            sources.append(u)
            targets.append(int(raw_v))

    if not sources:
        return torch.empty((0, 2), dtype=torch.long)

    edges = torch.tensor([sources, targets], dtype=torch.long).t()
    invalid = (
        (edges[:, 0] < 0)
        | (edges[:, 0] >= num_nodes)
        | (edges[:, 1] < 0)
        | (edges[:, 1] >= num_nodes)
    )
    if invalid.any():
        first_bad = edges[invalid][0].tolist()
        raise ValueError(
            f"Recommendation endpoint outside [0, {num_nodes}): "
            f"{tuple(first_bad)}"
        )
    return edges


def build_control_edge_set(
    train_edges: torch.Tensor,
    test_edges: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Build ``E_train ∪ E_test`` as a simple undirected edge set."""
    return merge_undirected_edges((train_edges, test_edges), num_nodes)


def build_predicted_edge_set(
    train_edges: torch.Tensor,
    recs_by_u: Dict[int, List[int]],
    num_nodes: int,
    k: int,
) -> torch.Tensor:
    """Build ``E_train ∪ Rec_K`` as a simple undirected edge set."""
    rec_edges = recommendations_to_edges(recs_by_u, num_nodes, k)
    return merge_undirected_edges((train_edges, rec_edges), num_nodes)


def degree_from_undirected_edges(
    edges: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Compute degree on a canonical simple undirected edge set."""
    canonical = canonicalize_undirected_edges(edges, num_nodes)
    return degree_from_simple_undirected_edges(canonical, num_nodes)


def degree_from_simple_undirected_edges(
    edges: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Compute degree when ``edges`` already is a simple undirected set.

    Graph constructors in this module return canonical edge sets. Keeping
    this fast path separate avoids sorting the full graph a second time while
    retaining the defensive public helper above for arbitrary input.
    """
    canonical = edge_tensor_2col(edges)
    if canonical.numel() == 0:
        return torch.zeros(num_nodes, dtype=torch.float32)
    return torch.bincount(
        canonical.reshape(-1), minlength=num_nodes
    ).to(dtype=torch.float32)


def novel_edges_against_simple_undirected_base(
    base_edges: torch.Tensor,
    candidate_edges: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Return canonical candidate edges that are absent from ``base_edges``.

    ``base_edges`` must be the sorted output of
    :func:`canonicalize_undirected_edges`. Candidates may have arbitrary
    orientation, duplicates, train overlaps, or self-loops. The search is
    performed only over candidate keys, avoiding a new sort of the full base
    graph for every source stratum.
    """
    base = edge_tensor_2col(base_edges)
    candidates = canonicalize_undirected_edges(candidate_edges, num_nodes)
    if candidates.numel() == 0 or base.numel() == 0:
        return candidates

    base_keys = base[:, 0] * num_nodes + base[:, 1]
    candidate_keys = candidates[:, 0] * num_nodes + candidates[:, 1]
    positions = torch.searchsorted(base_keys, candidate_keys)
    clamped = positions.clamp(max=len(base_keys) - 1)
    present = (positions < len(base_keys)) & (
        base_keys[clamped] == candidate_keys
    )
    return candidates[~present]
