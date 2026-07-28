"""
Weak-Tie Structural-Diversity Re-ranking
(Sanz-Cruzado & Castells, RecSys 2018)

From-scratch re-implementation of the greedy structural-diversity
re-ranker of:

    Javier Sanz-Cruzado and Pablo Castells. 2018. Enhancing Structural
    Diversity in Social Networks by Recommending Weak Ties. In RecSys'18.
    https://doi.org/10.1145/3240323.3240371

There is no official public implementation of this re-ranker (the
authors' later RELISON framework is a separate Java library), so this
module implements Algorithm 1 of the paper together with the two
community-based diversity objectives (MC, CEGC) and the corresponding
evaluation metrics, adapted from directed (Twitter) to undirected
graphs. It is used in the CAFE study as the attribute-free structural-
diversity re-ranking baseline ("Weak-tie diversification").

Mapping to the paper
--------------------
* Extended network (paper Sec. 3):  G' = <U, E ∪ Ê> where Ê is the set
  of top-k recommended edges of ALL users. We use set semantics (an
  undirected edge recommended by both endpoints, e.g. (u,v) and (v,u),
  counts once), exactly as E' = E ∪ Ê implies.
* Weak tie (paper Sec. 4.1, following De Meo et al.): an edge is weak
  iff its endpoints lie in different communities of a partition C of
  the training graph. Communities come from Louvain, mirroring the
  Cross-Link-Bias baseline in this codebase (networkit PLM), but
  seeded and single-threaded for reproducibility.
* MC — modularity complement (Sec. 4.1): MC = (1 - mod(G'|C)) / 2 with
  mod normalized by its maximum. Undirected specialization:
      mod = (2·intra - S/(2m)) / (2m - S/(2m)),  S = sum_c D_c^2,
  where m = |E'|, intra = #intra-community edges of G', and D_c is the
  total G'-degree of community c.
* CEGC — community edge Gini complement (Sec. 4.1): with X the vector
  of per-community-pair cross-edge counts n_ij (ALL unordered pairs,
  including zero counts) plus a single aggregate intra count n_0,
  sorted ascending, N = |X| = m(m-1)/2 + 1:
      CEGC = 1 - (1/(N-1)) * sum_i (2i - N - 1) x_i / |E'|.
  (The complement of the Gini index of X; sum(X) = |E'|.)
* Greedy re-ranking (Algorithm 1 + Sec. 6.1): per user u, for each
  position i = 1..k, the best below-cutoff candidate j0 by the dual
  objective
      phi(j) = (1-lam) * norm(f(S_u[j])) + lam * norm(mu(G'_{S(u:i/j)@k}))
  is swapped into position i iff phi(j0) > phi(i) (strict). The swap
  exchanges positions i and j0, so displaced items remain eligible at
  later positions. The graph state G'_{S@k} is GLOBAL: it reflects the
  current top-k of every user and is updated whenever a swap commits.
  Users are processed in ascending node-id order (the paper does not
  specify an order).
* Practical simplifications, exactly as published (Sec. 6.1):
    - variant='mc':   the diversity component is simplified to the
      binary indicator "is the candidate edge weak" (already in [0,1];
      it is used raw, not rank-normalized).
    - variant='cegc': only weak candidates are eligible to be swapped
      in, and the diversity component is the true CEGC of the extended
      graph after the hypothetical swap.
* Normalization: rank-sim (Lee 1997), norm = 1 - (rank-1)/(n-1) with
  rank 1 = best. Interpretation choices (the paper does not fully pin
  them down): the f-component uses the candidate's fixed rank in the
  user's retrieved pool (its position in the initial f-sorted list,
  which is invariant under swaps); the mu-component is ranked within
  the per-step comparison set {current occupant} ∪ {eligible
  candidates}, with ties sharing average ranks.

Complexity
----------
CEGC after a hypothetical swap is evaluated EXACTLY in O(1) per
candidate group via an incremental Gini structure based on the
pairwise-difference identity
    sum_i (2i - N - 1) x_(i)  =  sum_{a<b} |x_a - x_b|,
whose change under a unit increment/decrement of one entry only needs
the count of entries <= a value (a Fenwick tree over the value domain).
Candidates sharing (community pair, already-present flag) have equal
CEGC deltas and are evaluated once per position step.
"""

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Edge canonicalization (self-contained to avoid circular imports)
# ---------------------------------------------------------------------------

def _canon_edges(edges):
    """Return edges as an int64 numpy array of shape [E, 2]."""
    e = edges.cpu().numpy() if hasattr(edges, 'cpu') else np.asarray(edges)
    if e.ndim != 2 or (e.shape[0] != 2 and e.shape[1] != 2):
        raise ValueError(f"Edge tensor must be [E,2] or [2,E], got {e.shape}")
    if e.shape[0] == 2 and e.shape[1] != 2:
        e = e.T
    return e.astype(np.int64, copy=False)


def _dedup_undirected(edges_2col):
    """Canonical (min,max) undirected edge set without self-loops, [E,2]."""
    src = np.minimum(edges_2col[:, 0], edges_2col[:, 1])
    dst = np.maximum(edges_2col[:, 0], edges_2col[:, 1])
    keep = src != dst
    pairs = np.stack([src[keep], dst[keep]], axis=1)
    return np.unique(pairs, axis=0)


# ---------------------------------------------------------------------------
# Communities (Louvain via networkit PLM, as used by Cross-Link-Bias)
# ---------------------------------------------------------------------------

def detect_communities(train_edges, num_nodes, seed=42):
    """
    Louvain communities of the (undirected) training graph.

    Same algorithm as the Cross-Link-Bias baseline
    (baselines/fairness_aware_baseline_models.py: networkit PLM with
    default parameters), but seeded and run single-threaded so the
    partition is reproducible across runs. Returns a [num_nodes] int
    array with compact community ids 0..m-1 (isolated nodes get their
    own singleton communities, as PLM assigns every node a subset).
    """
    import networkit as nk

    try:
        nk.engineering.setSeed(seed, False)
        nk.engineering.setNumberOfThreads(1)
    except Exception as e:  # pragma: no cover - engineering API differences
        print(f"[WeakTie] Warning: could not fix networkit seed/threads ({e}); "
              f"the Louvain partition may vary across runs.")

    edges = _dedup_undirected(_canon_edges(train_edges))
    g = nk.Graph(int(num_nodes), weighted=False, directed=False)
    for u, v in edges.tolist():
        g.addEdge(u, v)

    plm = nk.community.PLM(g)
    plm.run()
    part = plm.getPartition()
    raw = np.array([part.subsetOf(i) for i in range(int(num_nodes))],
                   dtype=np.int64)
    _, membership = np.unique(raw, return_inverse=True)
    n_comm = int(membership.max()) + 1 if membership.size else 0
    n_weak = int((membership[edges[:, 0]] != membership[edges[:, 1]]).sum())
    print(f"[WeakTie] Louvain (PLM, seed={seed}): {n_comm} communities; "
          f"{n_weak}/{len(edges)} training edges are weak (cross-community)")
    return membership


# ---------------------------------------------------------------------------
# Incremental Gini over the community-pair count vector X
# ---------------------------------------------------------------------------

class _Fenwick:
    """Fenwick (BIT) over integer values 0..size-1; prefix counts."""

    def __init__(self, size):
        self.size = size
        self.tree = np.zeros(size + 1, dtype=np.int64)

    def add(self, idx, delta):
        i = idx + 1
        while i <= self.size:
            self.tree[i] += delta
            i += i & (-i)

    def cnt_le(self, idx):
        """Number of stored entries with value <= idx."""
        if idx < 0:
            return 0
        i = min(idx, self.size - 1) + 1
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return int(s)


class CommunityPairGini:
    """
    Exact, incrementally-updatable CEGC of the extended graph.

    State: for each unordered community pair p (i<j) a cross-edge count
    n_p, plus one aggregate intra-community count n_0 ("pair id" -1).
    The Gini numerator over the N = P+1 entries is maintained through
    the identity  sum_i (2i-N-1) x_(i) = sum_{a<b} |x_a - x_b|, whose
    change under a unit +/-1 of one entry is
        inc v -> v+1:  delta = 2 * cnt_le(v) - N - 1
        dec v -> v-1:  delta = N - 1 - 2 * cnt_le(v-1)
    with cnt_le taken BEFORE the move (O(log V) via a Fenwick tree).

    CEGC = 1 - num / ((N-1) * total), total = sum(X) = |E'|.
    """

    def __init__(self, num_communities, pair_counts, intra_count, max_value):
        """
        Args:
            num_communities: m (>= 1)
            pair_counts: dict {pair_id -> count>0} for cross pairs,
                pair_id = i * m + j with i < j (communities i, j)
            intra_count: n_0 (aggregate intra-community edge count)
            max_value: upper bound on any entry value ever reached
        """
        m = int(num_communities)
        self.m = m
        self.P = m * (m - 1) // 2  # number of cross-pair entries
        self.N = self.P + 1        # + the aggregate intra entry
        self.value = dict(pair_counts)   # pair_id -> value (>0 only)
        self.value[-1] = int(intra_count)

        self.fen = _Fenwick(int(max_value) + 2)
        nonzero = [v for v in self.value.values() if v > 0]
        # zero-valued entries: cross pairs absent from `pair_counts`,
        # plus the intra entry if n_0 == 0
        n_zero = self.N - len(nonzero)
        if n_zero:
            self.fen.add(0, n_zero)
        for v in nonzero:
            self.fen.add(v, 1)

        self.total = int(sum(nonzero))
        # Gini numerator from scratch: zeros contribute nothing; the
        # sorted nonzero values occupy global ranks n_zero+1 .. N.
        xs = np.sort(np.asarray(nonzero, dtype=np.int64))
        ranks = np.arange(n_zero + 1, n_zero + 1 + len(xs), dtype=np.int64)
        self.num = int(np.sum((2 * ranks - self.N - 1) * xs))

    def _get(self, pair_id):
        return self.value.get(pair_id, 0)

    def inc(self, pair_id):
        v = self._get(pair_id)
        self.num += 2 * self.fen.cnt_le(v) - self.N - 1
        self.fen.add(v, -1)
        self.fen.add(v + 1, 1)
        self.value[pair_id] = v + 1
        self.total += 1

    def dec(self, pair_id):
        v = self._get(pair_id)
        if v <= 0:
            raise ValueError(f"dec on empty pair {pair_id}")
        self.num += self.N - 1 - 2 * self.fen.cnt_le(v - 1)
        self.fen.add(v, -1)
        self.fen.add(v - 1, 1)
        self.value[pair_id] = v - 1
        self.total -= 1

    def cegc(self):
        if self.total <= 0 or self.N <= 1:
            return 0.0
        return 1.0 - self.num / ((self.N - 1) * self.total)


# ---------------------------------------------------------------------------
# Evaluation metrics on the extended graph (paper Sec. 4)
# ---------------------------------------------------------------------------

def _extended_edge_set(train_edges, recs_by_u, k):
    """Distinct undirected edges of G' = E ∪ Ê (top-k recs of all users)."""
    train = _dedup_undirected(_canon_edges(train_edges))
    src, dst = [], []
    for u, vs in recs_by_u.items():
        for v in vs[:k]:
            src.append(u)
            dst.append(v)
    if src:
        recs = _dedup_undirected(
            np.stack([np.asarray(src), np.asarray(dst)], axis=1))
        allp = np.concatenate([train, recs], axis=0)
    else:
        allp = train
    return np.unique(allp, axis=0)


def modularity_complement(train_edges, recs_by_u, num_nodes, membership, k=10):
    """MC(G'|C) = (1 - mod(G'|C))/2 on the extended graph (undirected)."""
    edges = _extended_edge_set(train_edges, recs_by_u, k)
    m_edges = len(edges)
    if m_edges == 0:
        return 0.0
    deg = np.zeros(num_nodes, dtype=np.int64)
    np.add.at(deg, edges[:, 0], 1)
    np.add.at(deg, edges[:, 1], 1)
    comm = np.asarray(membership)
    intra = int((comm[edges[:, 0]] == comm[edges[:, 1]]).sum())
    d_c = np.zeros(int(comm.max()) + 1, dtype=np.float64)
    np.add.at(d_c, comm, deg.astype(np.float64))
    s = float(np.sum(d_c ** 2)) / (2.0 * m_edges)
    den = 2.0 * m_edges - s
    if den <= 0:
        return 0.0
    mod = (2.0 * intra - s) / den
    return (1.0 - mod) / 2.0


def community_edge_gini_complement(train_edges, recs_by_u, num_nodes,
                                   membership, k=10):
    """CEGC(G'|C) on the extended graph (undirected, unordered pairs)."""
    edges = _extended_edge_set(train_edges, recs_by_u, k)
    total = len(edges)
    if total == 0:
        return 0.0
    comm = np.asarray(membership, dtype=np.int64)
    m = int(comm.max()) + 1
    ci, cj = comm[edges[:, 0]], comm[edges[:, 1]]
    lo, hi = np.minimum(ci, cj), np.maximum(ci, cj)
    cross = lo != hi
    n0 = int((~cross).sum())
    pair_ids, counts = np.unique(lo[cross] * m + hi[cross], return_counts=True)
    n_pairs = m * (m - 1) // 2
    N = n_pairs + 1
    if N <= 1:
        return 0.0
    n_zero = n_pairs - len(pair_ids) + (1 if n0 == 0 else 0)
    xs = np.sort(np.concatenate([counts, [n0]])) if n0 > 0 else np.sort(counts)
    xs = xs[xs > 0]
    ranks = np.arange(n_zero + 1, n_zero + 1 + len(xs), dtype=np.int64)
    gini_num = float(np.sum((2 * ranks - N - 1) * xs))
    return 1.0 - gini_num / ((N - 1) * total)


def weak_tie_fraction(recs_by_u, membership, k=10):
    """Fraction of recommended top-k edges that are weak (cross-community)."""
    comm = np.asarray(membership)
    weak = tot = 0
    for u, vs in recs_by_u.items():
        for v in vs[:k]:
            tot += 1
            weak += int(comm[u] != comm[v])
    return weak / tot if tot else 0.0


# ---------------------------------------------------------------------------
# The greedy re-ranker (paper Algorithm 1 + Sec. 6.1)
# ---------------------------------------------------------------------------

def _rank_sim(values_best_first_desc):
    """
    Rank-sim normalization (Lee 1997): 1 - (rank-1)/(n-1), rank 1 =
    largest value; ties share average ranks. Returns array in [0,1].
    """
    v = np.asarray(values_best_first_desc, dtype=np.float64)
    n = len(v)
    if n <= 1:
        return np.ones(n)
    ranks = stats.rankdata(-v, method='average')
    return 1.0 - (ranks - 1.0) / (n - 1.0)


class WeakTieReranker:
    """
    Global greedy weak-tie diversification re-ranker (Algorithm 1).

    Operates purely on candidate LISTS: `recs_by_u` must map each user
    to its retrieved pool ordered by DESCENDING base-recommender score
    (any base model; position in the list is the f-rank used by the
    rank-sim normalization). Returns re-ordered lists of the same
    length whose top-k prefix is the diversified recommendation.

    Args:
        variant: 'cegc' (paper's best; only weak candidates may be
                 swapped in, diversity component = exact CEGC of the
                 extended graph after the swap) or 'mc' (diversity
                 component = binary weak-tie indicator).
        lam: lambda in [0,1] — 0 keeps the base ranking, 1 ignores it.
        k: diversification cutoff (top-k of every user forms G').
        seed: Louvain seed (only used if membership is not supplied).
    """

    def __init__(self, variant='cegc', lam=0.5, k=10, seed=42):
        if variant not in ('mc', 'cegc'):
            raise ValueError(f"variant must be 'mc' or 'cegc', got {variant!r}")
        if not (0.0 <= lam <= 1.0):
            raise ValueError(f"lam must be in [0,1], got {lam}")
        self.variant = variant
        self.lam = lam
        self.k = int(k)
        self.seed = seed

    # -- state initialization ------------------------------------------------

    def _init_state(self, train_edges, recs_by_u, membership):
        """Build rec-edge multiplicities and the CEGC state of G'_{S@k}."""
        comm = membership
        m = int(comm.max()) + 1

        train = _dedup_undirected(_canon_edges(train_edges))
        rec_mult = {}
        for u, vs in recs_by_u.items():
            for v in vs[:self.k]:
                key = (u, v) if u < v else (v, u)
                rec_mult[key] = rec_mult.get(key, 0) + 1

        # distinct edges of G' = train ∪ recs (train never overlaps recs:
        # training neighbours are masked out of the candidate pools)
        pair_counts = {}
        intra = 0
        max_rec_edges = len(rec_mult)

        def account(a, b):
            nonlocal intra
            ca, cb = comm[a], comm[b]
            if ca == cb:
                intra += 1
            else:
                lo, hi = (ca, cb) if ca < cb else (cb, ca)
                pid = int(lo) * m + int(hi)
                pair_counts[pid] = pair_counts.get(pid, 0) + 1

        for a, b in train.tolist():
            account(a, b)
        for (a, b) in rec_mult:
            account(a, b)

        max_value = len(train) + max_rec_edges + 2
        gini = CommunityPairGini(m, pair_counts, intra, max_value)
        return rec_mult, gini

    @staticmethod
    def _pair_id(ca, cb, m):
        """Community-pair id: -1 for intra, i*m+j (i<j) for cross."""
        if ca == cb:
            return -1
        lo, hi = (ca, cb) if ca < cb else (cb, ca)
        return int(lo) * m + int(hi)

    # -- main entry ----------------------------------------------------------

    def rerank(self, recs_by_u, train_edges, num_nodes, membership=None,
               verbose=True):
        """
        Re-rank all users' candidate pools.

        Args:
            recs_by_u: {user -> list of candidate node ids, descending
                        base score}. Pools longer than k are required
                        for any diversification to happen (paper: pool
                        of 100, k = 10).
            train_edges: training edges ([E,2] or [2,E]); together with
                        the top-k lists they form the extended graph.
            num_nodes: number of nodes.
            membership: optional precomputed community array; computed
                        from train_edges (seeded Louvain) if None.

        Returns:
            {user -> re-ordered candidate list (same length)}.
        """
        lam = self.lam
        k = self.k

        if membership is None:
            membership = detect_communities(train_edges, num_nodes,
                                            seed=self.seed)
        comm = np.asarray(membership, dtype=np.int64)
        m = int(comm.max()) + 1

        users = sorted(recs_by_u.keys())
        out = {u: list(recs_by_u[u]) for u in users}

        if lam == 0.0:
            # phi(j) = norm_f(j) can never strictly beat phi(i) on a
            # descending-f list: identity re-ranking by construction.
            if verbose:
                print(f"[WeakTie] lam=0 -> identity re-ranking")
            return out

        use_gini = self.variant == 'cegc'
        if use_gini:
            rec_mult, gini = self._init_state(train_edges, out, membership)
        else:
            rec_mult, gini = None, None

        n_swaps = 0
        for u in users:
            lst = out[u]
            n_cand = len(lst)
            if n_cand <= k:
                continue

            # fixed f-rank of each pool entry = its initial position
            norm_f_by_node = {}
            denom = n_cand - 1
            for pos, v in enumerate(lst):
                norm_f_by_node[v] = 1.0 - pos / denom

            order = list(lst)  # current permutation of the pool

            for i in range(min(k, n_cand)):
                occ = order[i]
                below = order[k:]

                if self.variant == 'mc':
                    # phi = (1-lam)*norm_f + lam*1[weak]; the binary
                    # indicator is used raw (paper Sec. 6.1).
                    phi_occ = ((1 - lam) * norm_f_by_node[occ]
                               + lam * float(comm[occ] != comm[u]))
                    best_phi, best_j = None, None
                    for j, v in enumerate(below):
                        phi = ((1 - lam) * norm_f_by_node[v]
                               + lam * float(comm[v] != comm[u]))
                        if best_phi is None or phi > best_phi:
                            best_phi, best_j = phi, k + j
                    if best_j is not None and best_phi > phi_occ:
                        order[i], order[best_j] = order[best_j], order[i]
                        n_swaps += 1
                    continue

                # --- CEGC variant ---
                eligible = [(j, v) for j, v in enumerate(below)
                            if comm[v] != comm[u]]
                if not eligible:
                    continue

                occ_key = (u, occ) if u < occ else (occ, u)
                occ_pid = self._pair_id(comm[u], comm[occ], m)
                # removing u's rec of occ only removes the edge from G'
                # if no other user currently recommends it
                eff_dec = rec_mult[occ_key] == 1

                # candidates group by (pair id, edge-already-present):
                # equal CEGC after the swap within a group
                cegc_cache = {}

                def cegc_after(pid, present):
                    key = (pid, present)
                    if key in cegc_cache:
                        return cegc_cache[key]
                    # apply effective operations (pid -1 = intra entry)
                    if eff_dec:
                        gini.dec(occ_pid)
                    if not present:
                        gini.inc(pid)
                    val = gini.cegc()
                    # undo
                    if not present:
                        gini.dec(pid)
                    if eff_dec:
                        gini.inc(occ_pid)
                    cegc_cache[key] = val
                    return val

                mu_occ = gini.cegc()  # no-swap: current graph
                mu_cands = np.empty(len(eligible))
                for idx, (j, v) in enumerate(eligible):
                    key = (u, v) if u < v else (v, u)
                    present = rec_mult.get(key, 0) > 0
                    pid = self._pair_id(comm[u], comm[v], m)
                    mu_cands[idx] = cegc_after(pid, present)

                # rank-sim normalize mu over the comparison set
                mu_all = np.concatenate([[mu_occ], mu_cands])
                mu_norm = _rank_sim(mu_all)

                phi_occ = (1 - lam) * norm_f_by_node[occ] + lam * mu_norm[0]
                phi_cands = np.array(
                    [(1 - lam) * norm_f_by_node[v] for _, v in eligible]
                ) + lam * mu_norm[1:]

                best_idx = int(np.argmax(phi_cands))
                if phi_cands[best_idx] > phi_occ:
                    j, v = eligible[best_idx]
                    # commit the swap: update list and global graph state
                    order[i], order[k + j] = order[k + j], order[i]
                    n_swaps += 1

                    if eff_dec:
                        gini.dec(occ_pid)
                    if rec_mult[occ_key] == 1:
                        del rec_mult[occ_key]
                    else:
                        rec_mult[occ_key] -= 1

                    new_key = (u, v) if u < v else (v, u)
                    if rec_mult.get(new_key, 0) == 0:
                        gini.inc(self._pair_id(comm[u], comm[v], m))
                    rec_mult[new_key] = rec_mult.get(new_key, 0) + 1

            out[u] = order

        if verbose:
            final_cegc = gini.cegc() if use_gini else float('nan')
            print(f"[WeakTie] variant={self.variant} lam={lam} k={k}: "
                  f"{n_swaps} swaps"
                  + (f", CEGC(G'@k)={final_cegc:.4f}" if use_gini else ""))
        return out
