import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.core_algos import (
    apply_sdpo_ppo_clip,
    compute_self_distillation_loss,
)


def _base_config(ppo_clip: bool = True):
    return OmegaConf.create({"ppo_clip": ppo_clip, "full_logit_distillation": False, "alpha": 1.0})


def test_default_is_no_clip_when_ppo_clip_omitted():
    student = torch.tensor([[-1.0, -0.5]], requires_grad=True)
    teacher = torch.tensor([[-1.5, -0.8]])
    mask = torch.ones_like(student)
    config = OmegaConf.create({"full_logit_distillation": False, "alpha": 1.0})

    loss, metrics = compute_self_distillation_loss(
        student_log_probs=student,
        teacher_log_probs=teacher,
        response_mask=mask,
        self_distillation_config=config,
        old_log_probs=None,
        clip_ratio=None,
    )

    expected = ((student - teacher).detach() * student).mean()
    assert torch.allclose(loss, expected)
    assert "actor/sdpo_clipfrac" not in metrics


def test_no_clip_matches_unclipped_distillation_loss():
    student = torch.tensor([[-1.0, -0.5]], requires_grad=True)
    teacher = torch.tensor([[-1.5, -0.8]])
    old = torch.tensor([[-1.2, -0.6]])
    mask = torch.ones_like(student)

    loss, metrics = compute_self_distillation_loss(
        student_log_probs=student,
        teacher_log_probs=teacher,
        response_mask=mask,
        self_distillation_config=_base_config(ppo_clip=False),
        old_log_probs=old,
        clip_ratio=None,
    )

    expected = ((student - teacher).detach() * student).mean()
    assert torch.allclose(loss, expected)
    assert "actor/sdpo_clipfrac" not in metrics


def test_ppo_clip_increases_loss_when_ratio_below_lower_bound_and_g_positive():
    """When g > 0 and r < 1-eps, max picks clip(r)*g (pessimistic upper bound for minimization)."""
    clip_ratio = 0.2
    student = torch.tensor([[0.5]])
    teacher = torch.tensor([[-1.0]])
    old = torch.tensor([[1.0]])
    mask = torch.ones_like(student)

    g = (student - teacher).detach() * student
    ratio = torch.exp(student - old)
    unclipped = (ratio * g).mean()

    loss, metrics = compute_self_distillation_loss(
        student_log_probs=student,
        teacher_log_probs=teacher,
        response_mask=mask,
        self_distillation_config=_base_config(ppo_clip=True),
        old_log_probs=old,
        clip_ratio=clip_ratio,
    )

    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    expected = torch.max(ratio * g, clipped_ratio * g).mean()
    assert g.item() > 0.0
    assert ratio.item() < 1.0 - clip_ratio
    assert torch.allclose(loss, expected)
    assert loss > unclipped
    assert metrics["actor/sdpo_clipfrac"] > 0.0


def test_ppo_clip_increases_loss_when_ratio_above_upper_bound_and_g_negative():
    """When g < 0 and r > 1+eps, max picks clip(r)*g (pessimistic upper bound for minimization)."""
    clip_ratio = 0.2
    student = torch.tensor([[-0.5]])
    teacher = torch.tensor([[-1.0]])
    old = torch.tensor([[-2.0]])
    mask = torch.ones_like(student)

    g = (student - teacher).detach() * student
    ratio = torch.exp(student - old)
    unclipped = (ratio * g).mean()

    loss, metrics = compute_self_distillation_loss(
        student_log_probs=student,
        teacher_log_probs=teacher,
        response_mask=mask,
        self_distillation_config=_base_config(ppo_clip=True),
        old_log_probs=old,
        clip_ratio=clip_ratio,
    )

    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    expected = torch.max(ratio * g, clipped_ratio * g).mean()
    assert g.item() < 0.0
    assert ratio.item() > 1.0 + clip_ratio
    assert torch.allclose(loss, expected)
    assert loss > unclipped
    assert metrics["actor/sdpo_clipfrac"] > 0.0


def test_apply_sdpo_ppo_clip_clipfrac_zero_when_ratio_in_range():
    student = torch.tensor([[-1.0, -1.1]])
    old = torch.tensor([[-1.05, -1.08]])
    g = torch.tensor([[0.5, 0.3]])
    mask = torch.ones_like(student)

    _, metrics = apply_sdpo_ppo_clip(
        per_token_loss=g,
        student_log_probs=student,
        old_log_probs=old,
        clip_ratio=0.2,
        loss_mask=mask,
    )
    assert metrics["actor/sdpo_clipfrac"] == 0.0


def test_ppo_clip_backward_pass_runs():
    student = torch.tensor([[-0.5, -0.8]], requires_grad=True)
    teacher = torch.tensor([[-1.5, -1.2]])
    old = torch.tensor([[-2.0, -1.0]])
    mask = torch.ones_like(student)

    loss, _ = compute_self_distillation_loss(
        student_log_probs=student,
        teacher_log_probs=teacher,
        response_mask=mask,
        self_distillation_config=_base_config(ppo_clip=True),
        old_log_probs=old,
        clip_ratio=0.2,
    )
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_ppo_clip_requires_old_log_probs():
    student = torch.tensor([[-0.5]], requires_grad=True)
    teacher = torch.tensor([[-1.0]])
    mask = torch.ones_like(student)

    try:
        compute_self_distillation_loss(
            student_log_probs=student,
            teacher_log_probs=teacher,
            response_mask=mask,
            self_distillation_config=_base_config(ppo_clip=True),
            old_log_probs=None,
            clip_ratio=0.2,
        )
        raised = False
    except ValueError as exc:
        raised = True
        assert "old_log_probs" in str(exc)

    assert raised
