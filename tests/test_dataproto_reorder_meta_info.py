import numpy as np
import torch
from tensordict import TensorDict

from verl.protocol import DataProto


def _make_batch(batch_size: int = 4) -> DataProto:
    batch = TensorDict(
        {
            'input_ids': torch.arange(batch_size * 3, dtype=torch.long).reshape(batch_size, 3),
            'attention_mask': torch.ones(batch_size, 3, dtype=torch.long),
        },
        batch_size=(batch_size,),
    )
    non_tensor_batch = {
        'uid': np.array([f'sample_{i}' for i in range(batch_size)], dtype=object),
    }
    meta_info = {
        'turns_stats': [1, 2, 3, 4],
        'valid_action_stats': [0, 1, 2, 3],
        'valid_search_stats': [0, 0, 1, 2],
        'temperature': 1.0,
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)


def test_reorder_keeps_per_sample_meta_info_aligned():
    data = _make_batch()
    reorder_idx = torch.tensor([2, 0, 3, 1])

    data.reorder(reorder_idx)

    assert data.non_tensor_batch['uid'].tolist() == ['sample_2', 'sample_0', 'sample_3', 'sample_1']
    assert data.meta_info['turns_stats'] == [3, 1, 4, 2]
    assert data.meta_info['valid_action_stats'] == [2, 0, 3, 1]
    assert data.meta_info['valid_search_stats'] == [1, 0, 2, 0]
    assert data.meta_info['temperature'] == 1.0


def test_select_by_mask_keeps_per_sample_meta_info_aligned():
    data = _make_batch()
    mask = torch.tensor([False, True, True, False])

    selected = data.select_by_mask(mask)

    assert selected.non_tensor_batch['uid'].tolist() == ['sample_1', 'sample_2']
    assert selected.meta_info['turns_stats'] == [2, 3]
    assert selected.meta_info['valid_action_stats'] == [1, 2]
    assert selected.meta_info['valid_search_stats'] == [0, 1]
    assert selected.meta_info['temperature'] == 1.0
