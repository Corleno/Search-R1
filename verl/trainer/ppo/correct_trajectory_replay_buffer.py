# Copyright 2024 Bytedance Ltd. and/or its affiliates
"""FIFO replay buffer for EM-correct trajectories used by online SFT (RFT)."""

from __future__ import annotations

import random
from typing import Iterable, List, Optional, Sequence

import torch
from tensordict import TensorDict

from verl import DataProto

# Tensor keys needed for SFT actor updates (plus optional masks).
DEFAULT_STORE_KEYS = (
    'input_ids',
    'attention_mask',
    'position_ids',
    'responses',
    'response_mask',
    'loss_mask',
)


def _pad_1d_or_2d(tensor: torch.Tensor, target_len: int, pad_value: float = 0) -> torch.Tensor:
    """Pad the last dimension of a 1D/2D tensor to target_len."""
    cur = tensor.size(-1)
    if cur == target_len:
        return tensor
    if cur > target_len:
        return tensor[..., :target_len]
    pad_shape = list(tensor.shape)
    pad_shape[-1] = target_len - cur
    pad = tensor.new_full(pad_shape, pad_value)
    return torch.cat([tensor, pad], dim=-1)


def pad_dataproto_seq_dims(data: DataProto, seq_len: int, response_len: int) -> DataProto:
    """Return a copy padded to fixed seq/response lengths for cross-step concat."""
    tensors = {}
    for key, tensor in data.batch.items():
        t = tensor.cpu() if tensor.device.type != 'cpu' else tensor
        if key in ('input_ids', 'attention_mask', 'position_ids'):
            pad_value = 0
            tensors[key] = _pad_1d_or_2d(t, seq_len, pad_value=pad_value)
        elif key in (
            'responses',
            'response_mask',
            'loss_mask',
            'old_log_probs',
            'advantages',
            'token_level_scores',
            'token_level_rewards',
        ):
            tensors[key] = _pad_1d_or_2d(t, response_len, pad_value=0)
        else:
            # Drop non-aligned aux tensors so cross-step concat stays safe.
            continue
    return DataProto(batch=TensorDict(tensors, batch_size=data.batch.batch_size),
                     non_tensor_batch={},
                     meta_info=dict(data.meta_info) if data.meta_info else {})


class CorrectTrajectoryReplayBuffer:
    """In-memory FIFO buffer of correct trajectory tensors (driver process only)."""

    def __init__(
        self,
        capacity: int = 4096,
        store_keys: Optional[Sequence[str]] = None,
    ):
        if capacity <= 0:
            raise ValueError(f'capacity must be positive, got {capacity}')
        self.capacity = int(capacity)
        self.store_keys = tuple(store_keys) if store_keys is not None else DEFAULT_STORE_KEYS
        self._items: List[dict] = []

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()

    def _row_to_item(self, data: DataProto, idx: int) -> dict:
        item = {}
        for key in self.store_keys:
            if key not in data.batch.keys():
                continue
            row = data.batch[key][idx].detach().cpu().clone()
            item[key] = row
        if 'responses' not in item:
            raise KeyError('CorrectTrajectoryReplayBuffer requires responses in the batch')
        return item

    def push(self, data: DataProto) -> int:
        """Append all rows from data; evict oldest on overflow. Returns number pushed."""
        if data is None or len(data) == 0:
            return 0
        n = len(data)
        for i in range(n):
            self._items.append(self._row_to_item(data, i))
        overflow = len(self._items) - self.capacity
        if overflow > 0:
            del self._items[:overflow]
        return n

    def sample(self, n: int, generator: Optional[random.Random] = None) -> Optional[DataProto]:
        """Sample up to n trajectories uniformly without replacement."""
        if n <= 0 or len(self._items) == 0:
            return None
        n = min(int(n), len(self._items))
        rng = generator if generator is not None else random
        indices = rng.sample(range(len(self._items)), n)
        return self._items_to_dataproto(indices)

    def _items_to_dataproto(self, indices: Iterable[int]) -> DataProto:
        idx_list = list(indices)
        if not idx_list:
            raise ValueError('indices must be non-empty')

        # Pad variable-length rows to the max within this sample.
        seq_keys = ('input_ids', 'attention_mask', 'position_ids')
        resp_keys = ('responses', 'response_mask', 'loss_mask', 'old_log_probs', 'advantages')
        present_keys = [k for k in self.store_keys if k in self._items[idx_list[0]]]

        max_seq = 0
        max_resp = 0
        for i in idx_list:
            item = self._items[i]
            for k in seq_keys:
                if k in item:
                    max_seq = max(max_seq, item[k].shape[-1])
            for k in resp_keys:
                if k in item:
                    max_resp = max(max_resp, item[k].shape[-1])

        stacked = {}
        for key in present_keys:
            rows = []
            for i in idx_list:
                t = self._items[i][key]
                if key in seq_keys:
                    rows.append(_pad_1d_or_2d(t, max_seq, pad_value=0))
                elif key in resp_keys:
                    rows.append(_pad_1d_or_2d(t, max_resp, pad_value=0))
                else:
                    rows.append(t)
            stacked[key] = torch.stack(rows, dim=0)

        batch = TensorDict(stacked, batch_size=[len(idx_list)])
        return DataProto(batch=batch, non_tensor_batch={}, meta_info={})
