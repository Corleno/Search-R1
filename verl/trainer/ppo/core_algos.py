# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Any, Optional

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config): # seems never used?
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_token_reward_direct_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
):
    """Use token-level rewards directly as advantages (OPD / on-policy distillation).

    Supports (bs, response_length) or (bs, response_length, k) for top-k weighted rewards.

    Returns ``(advantages, returns)`` with ``returns == advantages`` so the trainer can
    populate ``batch['returns']`` like GAE/GRPO; there is no separate value target for OPD.
    """
    with torch.no_grad():
        if token_level_rewards.dim() == 3:
            response_mask = response_mask.unsqueeze(-1)
        advantages = token_level_rewards * response_mask
        returns = advantages.clone()
    return advantages, returns


def compute_token_reward_direct_plus_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index,
    grpo_outcome_weight: float = 1.0,
    epsilon: float = 1e-6,
):
    """Combine dense token_reward_direct with GRPO-style outcome baseline."""
    direct_adv, _ = compute_token_reward_direct_advantage(token_level_rewards, response_mask)
    grpo_adv, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        eos_mask=response_mask,
        index=index,
        epsilon=epsilon,
    )
    combined = direct_adv + grpo_outcome_weight * grpo_adv
    # Same contract as compute_token_reward_direct_advantage: returns track the policy target tensor.
    combined_returns = combined.clone()
    return combined, combined_returns, {'token_level_advantage_direct': direct_adv}


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length) or (bs, response_length, k) for top-k OPD
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length) or (bs, response_length, k)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length) or (bs, response_length, k)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    if advantages.dim() == 3:
        eos_mask = eos_mask.unsqueeze(-1).expand_as(advantages)
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str = "token-mean") -> torch.Tensor:
    """Aggregate per-token loss matrix into a scalar."""
    if loss_agg_mode == "token-mean":
        return verl_F.masked_mean(loss_mat, loss_mask)
    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")


def compute_self_distillation_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    self_distillation_config: Any,
    old_log_probs: Optional[torch.Tensor] = None,
    student_all_log_probs: Optional[torch.Tensor] = None,
    teacher_all_log_probs: Optional[torch.Tensor] = None,
    student_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    self_distillation_mask: Optional[torch.Tensor] = None,
    loss_agg_mode: str = "token-mean",
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Self-distillation loss (SDPO): match student log-probs to teacher on the same response tokens."""
    metrics: dict[str, Any] = {}

    loss_mask = response_mask
    if self_distillation_mask is not None:
        loss_mask = loss_mask * self_distillation_mask.unsqueeze(1)

    if self_distillation_config.get("full_logit_distillation", False):
        use_topk = self_distillation_config.get("distillation_topk") is not None
        if use_topk:
            if student_topk_log_probs is None or teacher_topk_log_probs is None:
                raise ValueError("top-k distillation requires student_topk_log_probs and teacher_topk_log_probs.")

            def add_tail(log_probs: torch.Tensor) -> torch.Tensor:
                log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
                log_s = torch.clamp(log_s, max=-1e-7)
                tail_log = torch.log(-torch.expm1(log_s))
                return torch.cat([log_probs, tail_log], dim=-1)

            def renorm_topk_log_probs(logp: torch.Tensor) -> torch.Tensor:
                logZ = torch.logsumexp(logp, dim=-1, keepdim=True)
                return logp - logZ

            student_distill_log_probs = student_topk_log_probs
            teacher_distill_log_probs = teacher_topk_log_probs
            if self_distillation_config.get("distillation_add_tail", True):
                student_distill_log_probs = add_tail(student_distill_log_probs)
                teacher_distill_log_probs = add_tail(teacher_distill_log_probs)
            else:
                student_distill_log_probs = renorm_topk_log_probs(student_distill_log_probs)
                teacher_distill_log_probs = renorm_topk_log_probs(teacher_distill_log_probs)
        else:
            if student_all_log_probs is None or teacher_all_log_probs is None:
                raise ValueError("full_logit_distillation requires student_all_log_probs and teacher_all_log_probs.")
            student_distill_log_probs = student_all_log_probs
            teacher_distill_log_probs = teacher_all_log_probs

        alpha = self_distillation_config.get("alpha", 0.5)
        if alpha == 0.0:
            kl_loss = F.kl_div(
                student_distill_log_probs, teacher_distill_log_probs, reduction="none", log_target=True
            )
        elif alpha == 1.0:
            kl_loss = F.kl_div(
                teacher_distill_log_probs, student_distill_log_probs, reduction="none", log_target=True
            )
        else:
            alpha_t = torch.tensor(
                alpha, dtype=student_distill_log_probs.dtype, device=student_distill_log_probs.device
            )
            mixture_log_probs = torch.logsumexp(
                torch.stack([
                    student_distill_log_probs + torch.log(1 - alpha_t),
                    teacher_distill_log_probs + torch.log(alpha_t),
                ]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, teacher_distill_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_distill_log_probs, reduction="none", log_target=True)
            kl_loss = torch.lerp(kl_student, kl_teacher, alpha_t)

        per_token_loss = kl_loss.sum(-1)
    else:
        assert self_distillation_config.get("alpha", 1.0) == 1.0, (
            "Only reverse KL is supported for non-full-logit distillation"
        )
        log_ratio = student_log_probs - teacher_log_probs
        per_token_loss = log_ratio.detach() * student_log_probs

    is_clip = self_distillation_config.get("is_clip")
    if is_clip is not None:
        if old_log_probs is None:
            raise ValueError("old_log_probs is required for distillation IS ratio.")
        negative_approx_kl = (student_log_probs - old_log_probs).detach()
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl).clamp(max=is_clip)
        per_token_loss = per_token_loss * ratio

    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights


    # if not all zeros in loss_mask, caculate the loss using agg_loss, otherwise loss is 0
    if not torch.all(loss_mask == 0):
        loss = agg_loss(
            loss_mat=per_token_loss,
            loss_mask=loss_mask,
            loss_agg_mode=loss_agg_mode,
        )
    else:
        loss = torch.tensor(0.0, device=per_token_loss.device)
    return loss, metrics
