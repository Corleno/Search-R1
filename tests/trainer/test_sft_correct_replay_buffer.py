import random

import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.correct_trajectory_replay_buffer import (
    CorrectTrajectoryReplayBuffer,
    pad_dataproto_seq_dims,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _is_sft_mode


def _make_batch(bsz: int, seq_len: int = 4, resp_len: int = 2, score: float = 1.0):
    scores = torch.zeros(bsz, resp_len)
    scores[:, -1] = score
    return DataProto.from_dict(
        tensors={
            "responses": torch.arange(bsz * resp_len).view(bsz, resp_len),
            "input_ids": torch.arange(bsz * seq_len).view(bsz, seq_len),
            "attention_mask": torch.ones(bsz, seq_len),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(bsz, -1),
            "loss_mask": torch.ones(bsz, resp_len),
            "token_level_scores": scores,
        }
    )


def test_replay_buffer_fifo_eviction():
    buf = CorrectTrajectoryReplayBuffer(capacity=3)
    buf.push(_make_batch(2))
    assert len(buf) == 2
    buf.push(_make_batch(2))
    assert len(buf) == 3  # capacity capped
    # Oldest two from first push should be gone; last push contributed 2, plus 1 survivor.
    sampled = buf.sample(3, generator=random.Random(0))
    assert sampled is not None
    assert len(sampled) == 3


def test_replay_buffer_sample_before_push_semantics():
    buf = CorrectTrajectoryReplayBuffer(capacity=8)
    first = _make_batch(2, score=1.0)
    # Encode distinctive response values.
    first.batch["responses"] = torch.tensor([[10, 11], [12, 13]])
    buf.push(first)

    sampled = buf.sample(2, generator=random.Random(1))
    assert sampled is not None
    assert len(sampled) == 2
    assert torch.equal(sampled.batch["responses"].sort(dim=0).values,
                       first.batch["responses"].sort(dim=0).values)

    second = _make_batch(2, score=1.0)
    second.batch["responses"] = torch.tensor([[90, 91], [92, 93]])
    buf.push(second)
    assert len(buf) == 4


def test_pad_dataproto_seq_dims_allows_concat():
    a = _make_batch(1, seq_len=4, resp_len=2)
    b = _make_batch(1, seq_len=6, resp_len=3)
    a2, b2 = pad_dataproto_seq_dims(a, 6, 3), pad_dataproto_seq_dims(b, 6, 3)
    mixed = DataProto.concat([a2, b2])
    assert mixed.batch["input_ids"].shape == (2, 6)
    assert mixed.batch["responses"].shape == (2, 3)


def test_is_sft_mode_helper():
    cfg = OmegaConf.create({
        "actor_rollout_ref": {"actor": {"policy_loss": {"loss_mode": "sft"}}}
    })
    assert _is_sft_mode(cfg) is True
    cfg.actor_rollout_ref.actor.policy_loss.loss_mode = "sdpo"
    assert _is_sft_mode(cfg) is False


def test_correct_trajectory_mask_threshold():
    # Build a minimal trainer-like object without running __init__.
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create({
        "actor_rollout_ref": {
            "actor": {
                "policy_loss": {"loss_mode": "sft"},
                "sft": {"success_reward_threshold": 0.5},
            }
        }
    })
    batch = _make_batch(3)
    batch.batch["token_level_scores"] = torch.tensor([
        [0.0, 1.0],   # sum 1.0 -> correct
        [0.0, 0.0],   # sum 0.0 -> incorrect
        [0.0, 0.4],   # sum 0.4 -> incorrect
    ])
    mask = trainer._correct_trajectory_mask(batch)
    assert mask.tolist() == [True, False, False]
