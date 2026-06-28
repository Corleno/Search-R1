import torch
from omegaconf import OmegaConf

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _distill_inactive_samples_with_external_ref,
)


def _sdpo_config(
    loss_mode: str = 'sdpo',
    teacher_regularization: str = 'ref',
    actor_path: str = 'Qwen/Qwen2.5-3B',
    ref_path: str = 'Qwen/Qwen2.5-7B-Instruct',
):
    return OmegaConf.create({
        'actor_rollout_ref': {
            'model': {'path': actor_path},
            'ref': {'model': {'path': ref_path}},
            'actor': {
                'policy_loss': {'loss_mode': loss_mode},
                'self_distillation': {
                    'teacher_regularization': teacher_regularization,
                    'filter_reprompt_before_update': True,
                },
            },
        },
    })


def test_external_ref_different_paths_returns_true():
    assert _distill_inactive_samples_with_external_ref(
        _sdpo_config(ref_path='Qwen/Qwen2.5-7B-Instruct')
    )


def test_same_path_ref_returns_false():
    assert not _distill_inactive_samples_with_external_ref(
        _sdpo_config(ref_path='Qwen/Qwen2.5-3B')
    )


def test_teacher_regularization_actor_returns_false():
    assert not _distill_inactive_samples_with_external_ref(
        _sdpo_config(teacher_regularization='actor')
    )


def test_teacher_regularization_ema_returns_false():
    assert not _distill_inactive_samples_with_external_ref(
        _sdpo_config(teacher_regularization='ema')
    )


def test_sdpo_grpo_mode_returns_false():
    assert not _distill_inactive_samples_with_external_ref(
        _sdpo_config(loss_mode='sdpo_grpo')
    )


def test_filter_keeps_inactive_samples_when_external_ref():
    config = _sdpo_config(ref_path='Qwen/Qwen2.5-7B-Instruct')
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = config
    trainer.actor_rollout_wg = type('WG', (), {'world_size': 8})()

    mask = torch.tensor([1.0, 0.0, 0.0, 1.0])
    batch = DataProto.from_dict(tensors={
        'self_distillation_mask': mask,
        'attention_mask': torch.ones(4, 8),
    })

    metrics = {}
    result_batch, should_update = trainer._filter_and_rebalance_sdpo_actor_batch(batch, metrics)

    assert should_update
    assert len(result_batch) == 4
    assert metrics['self_distillation/external_ref_full_batch_distill'] == 1.0
    assert 'self_distillation/reprompt_samples_before_actor_filter' not in metrics


def test_filter_drops_inactive_samples_when_same_path_ref():
    config = _sdpo_config(ref_path='Qwen/Qwen2.5-3B')
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = config
    trainer.actor_rollout_wg = type('WG', (), {'world_size': 1})()

    mask = torch.tensor([1.0, 0.0, 0.0, 1.0])
    batch = DataProto.from_dict(tensors={
        'self_distillation_mask': mask,
        'attention_mask': torch.ones(4, 8),
    })

    metrics = {}
    result_batch, should_update = trainer._filter_and_rebalance_sdpo_actor_batch(batch, metrics)

    assert should_update
    assert len(result_batch) == 2
    assert metrics['self_distillation/reprompt_samples_before_actor_filter'] == 2
