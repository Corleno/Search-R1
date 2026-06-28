from unittest.mock import MagicMock, patch

import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import reduce_metrics
from verl.workers.actor.dp_actor import DataParallelPPOActor


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sdpo_actor_config(**overrides):
    cfg = OmegaConf.create({
        "policy_loss": {"loss_mode": "sdpo"},
        "self_distillation": {
            "ppo_clip": False,
            "full_logit_distillation": False,
            "alpha": 1.0,
        },
        "ppo_mini_batch_size": 2,
        "ppo_micro_batch_size": 2,
        "use_dynamic_bsz": False,
        "state_masking": True,
        "clip_ratio": 0.2,
        "entropy_coeff": 0.001,
        "calculate_entropy": False,
        "use_kl_loss": False,
        "use_remove_padding": False,
        "ulysses_sequence_parallel_size": 1,
    })
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def _sdpo_batch(bsz: int = 2, resp_len: int = 2):
    device = _device()
    seq_len = 4
    loss_mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]], device=device)[:bsz, :resp_len]
    teacher_log_probs = torch.tensor([[-1.0, -1.1], [-1.2, -1.3]], device=device)[:bsz, :resp_len]
    return DataProto.from_dict(
        tensors={
            "responses": torch.zeros(bsz, resp_len, dtype=torch.long, device=device),
            "input_ids": torch.zeros(bsz, seq_len, dtype=torch.long, device=device),
            "attention_mask": torch.ones(bsz, seq_len, device=device),
            "position_ids": torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1),
            "old_log_probs": torch.full((bsz, resp_len), -1.0, device=device),
            "advantages": torch.zeros(bsz, resp_len, device=device),
            "loss_mask": loss_mask,
            "teacher_input_ids": torch.zeros(bsz, seq_len, dtype=torch.long, device=device),
            "teacher_attention_mask": torch.ones(bsz, seq_len, device=device),
            "teacher_position_ids": torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1),
            "self_distillation_mask": torch.ones(bsz, device=device),
            "teacher_log_probs": teacher_log_probs,
        },
        meta_info={"temperature": 1.0},
    )


@patch.object(DataParallelPPOActor, "_optimizer_step", return_value=torch.tensor(1.0))
@patch.object(DataParallelPPOActor, "_forward_micro_batch")
def test_sdpo_update_logs_actor_entropy_loss(mock_forward, _mock_optimizer_step):
    device = _device()
    student_log_prob = torch.tensor([[-0.5, -0.6], [-0.7, -0.8]], device=device, requires_grad=True)
    entropy = torch.tensor([[1.0, 2.0], [1.5, 2.5]], device=device)
    mock_forward.return_value = (entropy, student_log_prob, None, None)

    actor = DataParallelPPOActor(
        config=_sdpo_actor_config(),
        actor_module=MagicMock(),
        actor_optimizer=MagicMock(),
    )
    actor.actor_module.train = MagicMock()

    metrics = actor.update_policy(_sdpo_batch())
    reduced = reduce_metrics(metrics)
    assert "actor/entropy_loss" in reduced
    assert reduced["actor/entropy_loss"] > 0.0
    # loss_mask [[1,0],[1,1]]: masked mean of entropy over 3 active agent tokens
    expected = (1.0 + 1.5 + 2.5) / 3.0
    assert abs(reduced["actor/entropy_loss"] - expected) < 1e-5


@patch.object(DataParallelPPOActor, "_optimizer_step", return_value=torch.tensor(1.0))
@patch.object(DataParallelPPOActor, "_forward_micro_batch")
def test_sdpo_calculate_entropy_without_entropy_coeff(mock_forward, _mock_optimizer_step):
    device = _device()
    student_log_prob = torch.tensor([[-0.5, -0.6], [-0.7, -0.8]], device=device, requires_grad=True)
    entropy = torch.tensor([[1.0, 2.0], [1.5, 2.5]], device=device)
    mock_forward.return_value = (entropy, student_log_prob, None, None)

    actor = DataParallelPPOActor(
        config=_sdpo_actor_config(entropy_coeff=0.0, calculate_entropy=True),
        actor_module=MagicMock(),
        actor_optimizer=MagicMock(),
    )
    actor.actor_module.train = MagicMock()

    metrics = actor.update_policy(_sdpo_batch())
    reduced = reduce_metrics(metrics)
    assert "actor/entropy_loss" in reduced
    mock_forward.assert_called()
    assert mock_forward.call_args.kwargs["calculate_entropy"] is True
