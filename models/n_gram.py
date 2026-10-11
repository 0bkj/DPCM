import math
from typing import Optional, Union

import numpy as np
import torch

TensorLike = Union[np.ndarray, torch.Tensor]


def _resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_long_tensor(x: TensorLike) -> torch.Tensor:
    """Convert dtype only; intentionally do not copy the full training set to GPU."""
    if torch.is_tensor(x):
        return x.to(dtype=torch.long)
    return torch.as_tensor(x, dtype=torch.long)


def _to_device_long(x: TensorLike, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.long, non_blocking=True)
    return torch.as_tensor(x, dtype=torch.long, device=device)


class _VectorizedGroupedBackoff:
    """Sparse vectorized backoff n-gram engine.

    Training is chunked to avoid torch.unique() temporarily sorting all windows at
    once. Compact sparse count tables are placed on ``device`` for GPU scoring.
    The only Python loops are over n-gram order and coarse chunks, never over
    individual samples, channels, or latent positions.
    """

    def __init__(
        self,
        num_groups: int,
        order: int,
        alpha: float,
        device: Optional[Union[str, torch.device]] = None,
        fit_chunk_size: int = 8192,
    ):
        if order < 1:
            raise ValueError("order 必须 >= 1")
        if alpha <= 0:
            raise ValueError("alpha 必须 > 0")
        if num_groups < 1:
            raise ValueError("num_groups 必须 >= 1")
        if fit_chunk_size < 1:
            raise ValueError("fit_chunk_size 必须 >= 1")
        self.num_groups = int(num_groups)
        self.order = int(order)
        self.alpha = float(alpha)
        self.device = _resolve_device(device)
        self.fit_chunk_size = int(fit_chunk_size)
        self.vocab: Optional[torch.Tensor] = None
        self.vocab_size = 0
        self.base = 0
        self.total_tokens: Optional[torch.Tensor] = None
        self.ngram_keys = {}
        self.ngram_values = {}
        self.context_keys = {}
        self.context_values = {}
        self._fitted = False

    def _validate_overflow(self) -> None:
        max_key_space = self.num_groups * pow(self.base, self.order)
        if max_key_space > torch.iinfo(torch.int64).max:
            raise OverflowError(
                "vocab_size/order 组合过大，int64 n-gram 键可能溢出；"
                "请降低 order 或先压缩 code 字典。"
            )

    def _map_tokens(self, sequences: torch.Tensor, vocab: torch.Tensor) -> torch.Tensor:
        flat = sequences.reshape(-1)
        positions = torch.searchsorted(vocab, flat)
        safe_positions = positions.clamp_max(self.vocab_size - 1)
        known = (positions < self.vocab_size) & (vocab[safe_positions] == flat)
        mapped = torch.where(known, positions, torch.full_like(positions, self.vocab_size))
        return mapped.view_as(sequences)

    def _keys(self, windows: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        n = windows.shape[-1]
        powers = torch.tensor(
            [pow(self.base, i) for i in range(n - 1, -1, -1)],
            dtype=torch.long,
            device=windows.device,
        )
        return (windows * powers).sum(dim=-1) + groups * pow(self.base, n)

    @staticmethod
    def _lookup(sorted_keys: torch.Tensor, values: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if sorted_keys.numel() == 0:
            return torch.zeros(query.shape, dtype=values.dtype, device=query.device)
        flat_query = query.reshape(-1)
        idx = torch.searchsorted(sorted_keys, flat_query)
        safe_idx = idx.clamp_max(sorted_keys.numel() - 1)
        hits = (idx < sorted_keys.numel()) & (sorted_keys[safe_idx] == flat_query)
        found = torch.where(hits, values[safe_idx], torch.zeros_like(values[safe_idx]))
        return found.view_as(query)

    @staticmethod
    def _merge_counts(
        old_keys: Optional[torch.Tensor],
        old_values: Optional[torch.Tensor],
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
    ):
        if old_keys is None or old_keys.numel() == 0:
            return new_keys, new_values.to(torch.float32)
        keys = torch.cat((old_keys, new_keys), dim=0)
        values = torch.cat((old_values, new_values.to(torch.float32)), dim=0)
        uniq, inverse = torch.unique(keys, sorted=True, return_inverse=True)
        merged = torch.zeros(uniq.shape, dtype=torch.float32, device=keys.device)
        merged.scatter_add_(0, inverse, values)
        return uniq, merged

    @torch.no_grad()
    def fit(self, sequences: TensorLike, count_device: Optional[Union[str, torch.device]] = None):
        x = _as_long_tensor(sequences)
        if x.ndim != 3 or x.shape[1] != self.num_groups:
            raise ValueError(
                f"sequences 必须为 [num_sequences, {self.num_groups}, length]，"
                f"实际为 {tuple(x.shape)}"
            )
        if x.shape[0] == 0 or x.shape[-1] == 0:
            raise ValueError("训练序列不能为空")

        # Joint mode can build counts on CPU because its training matrix can be huge;
        # only compact sparse tables are later copied to CUDA for scoring.
        work_device = torch.device(count_device) if count_device is not None else self.device
        num_sequences, groups_count, length = x.shape
        chunk_size = min(self.fit_chunk_size, num_sequences)

        # Pass 1: discover vocabulary without placing all train_codes on CUDA.
        vocab = None
        for start in range(0, num_sequences, chunk_size):
            part = x[start:start + chunk_size].to(work_device, non_blocking=True)
            part_vocab = torch.unique(part.reshape(-1), sorted=True)
            vocab = part_vocab if vocab is None else torch.unique(
                torch.cat((vocab, part_vocab)), sorted=True
            )
        self.vocab_size = int(vocab.numel())
        self.base = self.vocab_size + 1
        self._validate_overflow()

        ngram_keys = {n: None for n in range(1, min(self.order, length) + 1)}
        ngram_values = {n: None for n in range(1, min(self.order, length) + 1)}
        context_keys = {n: None for n in range(2, min(self.order, length) + 1)}
        context_values = {n: None for n in range(2, min(self.order, length) + 1)}

        # Pass 2: count one manageable chunk at a time.
        max_order = min(self.order, length)
        for start in range(0, num_sequences, chunk_size):
            part = x[start:start + chunk_size].to(work_device, non_blocking=True)
            ids = self._map_tokens(part, vocab)
            s = ids.shape[0]
            group_seed = torch.arange(
                groups_count, dtype=torch.long, device=work_device
            ).view(1, groups_count, 1)
            for n in range(1, max_order + 1):
                windows = ids.unfold(-1, n, 1)
                group_grid = group_seed.expand(s, groups_count, windows.shape[-2])
                keys = self._keys(windows, group_grid).reshape(-1)
                uniq, cnt = torch.unique(keys, sorted=True, return_counts=True)
                ngram_keys[n], ngram_values[n] = self._merge_counts(
                    ngram_keys[n], ngram_values[n], uniq, cnt
                )
                if n >= 2:
                    ctx = self._keys(windows[..., :-1], group_grid).reshape(-1)
                    uniq_ctx, cnt_ctx = torch.unique(ctx, sorted=True, return_counts=True)
                    context_keys[n], context_values[n] = self._merge_counts(
                        context_keys[n], context_values[n], uniq_ctx, cnt_ctx
                    )
            del part, ids

        # Move only vocabulary and sparse tables to inference device.
        self.vocab = vocab.to(self.device, non_blocking=True)
        self.total_tokens = torch.full(
            (groups_count,), num_sequences * length, dtype=torch.float32, device=self.device
        )
        self.ngram_keys = {n: k.to(self.device, non_blocking=True) for n, k in ngram_keys.items()}
        self.ngram_values = {n: v.to(self.device, non_blocking=True) for n, v in ngram_values.items()}
        self.context_keys = {n: k.to(self.device, non_blocking=True) for n, k in context_keys.items()}
        self.context_values = {n: v.to(self.device, non_blocking=True) for n, v in context_values.items()}
        self._fitted = True
        return self

    @torch.no_grad()
    def score_tensor(self, sequences: TensorLike) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("请先调用 fit()")
        x = _to_device_long(sequences, self.device)
        if x.ndim != 3 or x.shape[1] != self.num_groups:
            raise ValueError(
                f"sequences 必须为 [num_sequences, {self.num_groups}, length]，"
                f"实际为 {tuple(x.shape)}"
            )
        if x.shape[-1] == 0:
            return torch.empty(x.shape, dtype=torch.float32, device=self.device)

        ids = self._map_tokens(x, self.vocab)
        s, g, length = ids.shape
        groups = torch.arange(g, device=self.device, dtype=torch.long).view(1, g, 1)
        group_grid = groups.expand(s, g, length)
        uni_keys = self._keys(ids.unsqueeze(-1), group_grid)
        uni_counts = self._lookup(self.ngram_keys[1], self.ngram_values[1], uni_keys)
        denominator = self.total_tokens.view(1, g, 1) + self.alpha * self.vocab_size
        probs = (uni_counts + self.alpha) / denominator
        selected = torch.zeros_like(probs, dtype=torch.bool)

        max_order = min(self.order, length, max(self.ngram_keys))
        for n in range(max_order, 1, -1):
            windows = ids.unfold(-1, n, 1)
            local_groups = groups.expand(s, g, windows.shape[-2])
            ngram_keys = self._keys(windows, local_groups)
            ctx_keys = self._keys(windows[..., :-1], local_groups)
            ngram_counts = self._lookup(self.ngram_keys[n], self.ngram_values[n], ngram_keys)
            context_counts = self._lookup(self.context_keys[n], self.context_values[n], ctx_keys)
            candidate = (ngram_counts + self.alpha) / (
                context_counts + self.alpha * self.vocab_size
            )
            region = slice(n - 1, None)
            use = (~selected[..., region]) & (context_counts > 0)
            probs[..., region] = torch.where(use, candidate, probs[..., region])
            selected[..., region] |= use
        return -torch.log(probs.clamp_min(1e-12))


class BackoffNGramScorer:
    def __init__(
        self,
        order=3,
        alpha=0.1,
        device: Optional[Union[str, torch.device]] = None,
        fit_chunk_size: int = 8192,
    ):
        self.order = order
        self.alpha = alpha
        self.device = _resolve_device(device)
        self._engine = _VectorizedGroupedBackoff(1, order, alpha, self.device, fit_chunk_size)

    def fit(self, sequences):
        x = _as_long_tensor(sequences)
        if x.ndim != 2:
            raise ValueError("sequences 必须是等长二维数组/张量 [num_sequences, length]")
        self._engine.fit(x.unsqueeze(1))
        return self

    def score(self, sequence):
        x = _as_long_tensor(sequence)
        if x.ndim != 1:
            raise ValueError("sequence 必须是一维数组/张量")
        out = self._engine.score_tensor(x.view(1, 1, -1))[0, 0]
        return out.detach().cpu().numpy().astype(np.float32, copy=False)


class TimeNGramDetector:
    def __init__(
        self,
        channels,
        order=3,
        alpha=0.1,
        device: Optional[Union[str, torch.device]] = None,
        fit_chunk_size: int = 8192,
        score_chunk_size: int = 256,
    ):
        self.channels = int(channels)
        self.order = order
        self.alpha = alpha
        self.device = _resolve_device(device)
        self.score_chunk_size = int(score_chunk_size)
        self._engine = _VectorizedGroupedBackoff(
            self.channels, order, alpha, self.device, fit_chunk_size
        )

    def fit(self, train_codes):
        x = _as_long_tensor(train_codes)
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(f"train_codes 必须为 [num_samples, {self.channels}, latent_time]")
        self._engine.fit(x)
        return self

    def score(self, test_codes, aggregate="mean", topk=3):
        x = _as_long_tensor(test_codes)
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(f"test_codes 必须为 [batch, {self.channels}, latent_time]")
        outputs = []
        for start in range(0, x.shape[0], self.score_chunk_size):
            all_scores = self._engine.score_tensor(x[start:start + self.score_chunk_size])
            outputs.append(_aggregate(all_scores, aggregate, topk).cpu())
        return torch.cat(outputs, dim=0).numpy().astype(np.float32, copy=False)


class VarNGramDetector:
    """沿 channel 维建模联合模式；默认采用 CPU 分块计数 + GPU 查表评分。

    原始调用 ``VarNGramDetector(order=2, alpha=0.1).fit(train_codes)``
    无需变化。若 CPU 内存非常紧张，可进一步降低 ``fit_chunk_size``。
    """

    def __init__(
        self,
        order=2,
        alpha=0.1,
        device: Optional[Union[str, torch.device]] = None,
        fit_chunk_size: int = 8192,
        score_chunk_size: int = 32768,
        count_on_cpu: bool = True,
    ):
        self.order = order
        self.alpha = alpha
        self.device = _resolve_device(device)
        self.fit_chunk_size = int(fit_chunk_size)
        self.score_chunk_size = int(score_chunk_size)
        self.count_on_cpu = bool(count_on_cpu)
        self._engine = _VectorizedGroupedBackoff(1, order, alpha, self.device, fit_chunk_size)

    def fit(self, train_codes):
        x = _as_long_tensor(train_codes)
        if x.ndim != 3:
            raise ValueError("train_codes 必须为 [num_samples, channels, latent_time]")
        n, c, t = x.shape
        sequences = x.permute(0, 2, 1).reshape(n * t, 1, c)
        work_device = "cpu" if self.count_on_cpu else self.device
        self._engine.fit(sequences, count_device=work_device)
        return self

    def score(self, test_codes, aggregate="mean", topk=3):
        x = _as_long_tensor(test_codes)
        if x.ndim != 3:
            raise ValueError("test_codes 必须为 [batch, channels, latent_time]")
        b, c, t = x.shape
        sequences = x.permute(0, 2, 1).reshape(b * t, 1, c)
        flat_scores = torch.empty((b * t, c), dtype=torch.float32, device="cpu")
        for start in range(0, b * t, self.score_chunk_size):
            end = min(start + self.score_chunk_size, b * t)
            chunk_scores = self._engine.score_tensor(sequences[start:end])[:, 0, :]
            flat_scores[start:end].copy_(chunk_scores.detach().cpu())
        all_scores = flat_scores.reshape(b, t, c).permute(0, 2, 1)
        return _aggregate(all_scores, aggregate, topk).numpy().astype(np.float32, copy=False)


class ChowLiuVCPMDetector:
    """Tree-structured variable-collaboration scorer for discrete VQ states.

    The detector is deliberately a post-hoc statistical component: it only
    consumes normal training-set code indices and never changes VQ-VAE weights.
    A maximum-weight spanning tree is learned with (normalized) mutual
    information, then each selected edge owns its own smoothed conditional
    2-gram table.  The test score is the normalized negative log likelihood
    of the resulting Chow--Liu tree factorization.
    """

    def __init__(
        self,
        n_states: Optional[int] = None,
        state_modulo: Optional[int] = None,
        gamma: float = 0.1,
        tree_weight: str = "nmi",
        epsilon: float = 1e-6,
    ):
        if n_states is not None and n_states < 1:
            raise ValueError("n_states must be >= 1")
        if state_modulo is not None and state_modulo < 1:
            raise ValueError("state_modulo must be >= 1")
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        if tree_weight not in {"mi", "nmi"}:
            raise ValueError("tree_weight must be 'mi' or 'nmi'")
        self.n_states = n_states
        self.state_modulo = state_modulo
        self.gamma = float(gamma)
        self.tree_weight = tree_weight
        self.epsilon = float(epsilon)
        self.edges = []
        self.edge_weights = []
        self.mutual_information = None
        self.normalized_mutual_information = None
        self._fitted = False

    @staticmethod
    def _maximum_spanning_tree(weights: np.ndarray):
        """Kruskal's algorithm without adding a scipy/networkx dependency."""
        channels = weights.shape[0]
        parent = list(range(channels))

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        edges = sorted(
            ((float(weights[i, j]), i, j) for i in range(channels) for j in range(i + 1, channels)),
            reverse=True,
        )
        tree = []
        for weight, i, j in edges:
            root_i, root_j = find(i), find(j)
            if root_i == root_j:
                continue
            parent[root_i] = root_j
            tree.append((i, j, weight))
            if len(tree) == channels - 1:
                break
        return tree

    def _validate_codes(self, codes: TensorLike) -> np.ndarray:
        x = _as_long_tensor(codes)
        if x.ndim != 3:
            raise ValueError("codes must be [num_samples, channels, latent_time]")
        x = x.detach().cpu().numpy().astype(np.int64, copy=False)
        if x.size == 0:
            raise ValueError("codes must not be empty")
        if x.min() < 0:
            raise ValueError("prototype states must be non-negative")
        # DPCM stores channel-local prototypes in disjoint global blocks
        # (channel * K + local_state).  Chow--Liu operates on Q_i in
        # {1,...,K}, so restore the local state before estimating dependence.
        if self.state_modulo is not None:
            x = np.mod(x, self.state_modulo)
        if self.n_states is not None and x.max() >= self.n_states:
            raise ValueError("code index exceeds n_states")
        return x

    @torch.no_grad()
    def fit(self, train_codes: TensorLike):
        x = self._validate_codes(train_codes)
        samples, channels, latent_time = x.shape
        self.channels = channels
        self.n_states = int(self.n_states or (int(x.max()) + 1))
        states = x.transpose(0, 2, 1).reshape(samples * latent_time, channels)
        total = states.shape[0]
        k = self.n_states

        counts = np.empty((channels, k), dtype=np.float64)
        marginals = np.empty((channels, k), dtype=np.float64)
        entropy = np.empty(channels, dtype=np.float64)
        denominator = total + self.gamma * k
        for channel in range(channels):
            counts[channel] = np.bincount(states[:, channel], minlength=k)
            marginals[channel] = (counts[channel] + self.gamma) / denominator
            entropy[channel] = -np.sum(marginals[channel] * np.log(marginals[channel]))

        mi = np.zeros((channels, channels), dtype=np.float64)
        nmi = np.zeros((channels, channels), dtype=np.float64)
        joint_probabilities = {}
        for i in range(channels):
            for j in range(i + 1, channels):
                pair_counts = np.bincount(
                    states[:, i] * k + states[:, j], minlength=k * k
                ).astype(np.float64)
                joint = (pair_counts.reshape(k, k) + self.gamma) / (total + self.gamma * k * k)
                value = float(np.sum(joint * np.log(joint / (marginals[i, :, None] * marginals[j, None, :]))))
                mi[i, j] = mi[j, i] = value
                nmi_value = value / np.sqrt(max(entropy[i] * entropy[j], self.epsilon))
                nmi[i, j] = nmi[j, i] = nmi_value
                joint_probabilities[(i, j)] = joint

        weights = nmi if self.tree_weight == "nmi" else mi
        tree = self._maximum_spanning_tree(weights)
        self.edges = [(i, j) for i, j, _ in tree]
        self.edge_weights = [weight for _, _, weight in tree]
        self.marginals = marginals
        self.joint_probabilities = {(i, j): joint_probabilities[(i, j)] for i, j in self.edges}
        self.mutual_information = mi
        self.normalized_mutual_information = nmi

        # Orient the undirected tree once.  This does not reintroduce an
        # arbitrary channel chain: every conditional remains tied to a true
        # Chow--Liu edge.  A high-degree root yields a compact, deterministic
        # factorization; all roots describe the same unsmoothed tree density.
        adjacency = [[] for _ in range(channels)]
        for i, j in self.edges:
            adjacency[i].append(j)
            adjacency[j].append(i)
        self.root = max(range(channels), key=lambda node: (len(adjacency[node]), -node))
        visited = {self.root}
        frontier = [self.root]
        self.directed_edges = []
        while frontier:
            parent = frontier.pop(0)
            for child in sorted(adjacency[parent]):
                if child in visited:
                    continue
                visited.add(child)
                frontier.append(child)
                self.directed_edges.append((parent, child))

        self.conditionals = {}
        for parent, child in self.directed_edges:
            key = (min(parent, child), max(parent, child))
            joint = self.joint_probabilities[key]
            # Stored pair tables are ordered by their ascending channel ids.
            if parent > child:
                joint = joint.T
            self.conditionals[(parent, child)] = joint / joint.sum(axis=1, keepdims=True)
        self._fitted = True
        return self

    def _tree_nll(self, x: np.ndarray) -> np.ndarray:
        """Negative log p(root) - sum log p(child | parent), per latent time."""
        batch, _, latent_time = x.shape
        root_state = x[:, self.root, :]
        nll = -np.log(np.maximum(self.marginals[self.root][root_state], self.epsilon))
        for parent, child in self.directed_edges:
            conditional = self.conditionals[(parent, child)]
            nll -= np.log(np.maximum(
                conditional[x[:, parent, :], x[:, child, :]], self.epsilon
            ))
        # Match the original variable n-gram's mean-per-variable convention.
        return nll / max(1, self.channels)

    @torch.no_grad()
    def score(self, test_codes: TensorLike, aggregate="mean", topk=3):
        del aggregate, topk  # Tree edges are always averaged as defined by VCPM.
        if not self._fitted:
            raise RuntimeError("fit() must be called before score()")
        x = self._validate_codes(test_codes)
        if x.shape[1] != self.channels:
            raise ValueError("test channel count differs from fitted detector")
        return self._tree_nll(x).astype(np.float32, copy=False)

    def summary(self):
        """Small serializable record for experiment artifacts and paper tables."""
        return {
            "channels": self.channels,
            "n_states": self.n_states,
            "state_modulo": self.state_modulo,
            "gamma": self.gamma,
            "tree_weight": self.tree_weight,
            "edges": [list(edge) for edge in self.edges],
            "edge_weights": [float(weight) for weight in self.edge_weights],
            "root": self.root,
            "directed_edges": [list(edge) for edge in self.directed_edges],
            "scoring": "mean negative log Chow-Liu tree likelihood",
        }


def _aggregate(all_scores: torch.Tensor, aggregate: str, topk: int) -> torch.Tensor:
    if aggregate == "none":
        return all_scores
    if aggregate == "mean":
        return all_scores.mean(dim=1)
    if aggregate == "max":
        return all_scores.max(dim=1).values
    if aggregate == "topk":
        k = min(int(topk), all_scores.shape[1])
        if k < 1:
            raise ValueError("topk 必须 >= 1")
        return torch.topk(all_scores, k=k, dim=1).values.mean(dim=1)
    raise ValueError(f"未知 aggregate: {aggregate}")
