# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple, Union, Optional

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy: bool = False,
        top_k: int = 0,
        student_top_k_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
            entropy: (bs, response_len) or None
            log_probs: (bs, response_len)
            topk_ids: (bs, response_len, k) or None
            topk_log_probs: (bs, response_len, k) or None
        """
        response_length = micro_batch['responses'].size(-1)
        entropy = None
        topk_ids = None
        topk_log_probs = None
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                need_topk = top_k > 0
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                if need_topk and not calculate_entropy:
                    entropy_rmpad = None

                if need_topk:
                    log_probs_all = torch.log_softmax(logits_rmpad, dim=-1)
                    log_probs = log_probs_all.gather(
                        dim=-1, index=input_ids_rmpad_rolled.unsqueeze(-1)).squeeze(-1)
                else:
                    log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                if need_topk:
                    if student_top_k_ids is not None:
                        if student_top_k_ids.ndim == 3:
                            full_student_top_k_ids = torch.zeros(
                                (batch_size, seqlen, top_k),
                                dtype=student_top_k_ids.dtype,
                                device=student_top_k_ids.device)
                            full_student_top_k_ids[:, -response_length - 1:-1, :] = student_top_k_ids
                            flat_ids = full_student_top_k_ids.view(-1, top_k)
                        else:
                            flat_ids = student_top_k_ids
                        topk_ids_rmpad = flat_ids[indices]
                        topk_ids = topk_ids_rmpad
                    else:
                        _, topk_ids = torch.topk(logits_rmpad, k=top_k, dim=-1)
                    assert topk_ids is not None
                    topk_log_probs = torch.log_softmax(logits_rmpad, dim=-1).gather(dim=-1, index=topk_ids)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if entropy_rmpad is not None:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if need_topk:
                        topk_ids = gather_outpus_and_unpad(topk_ids, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                        topk_log_probs = gather_outpus_and_unpad(
                            topk_log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                # pad back to (bsz, seqlen)
                if entropy_rmpad is not None:
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                             indices=indices,
                                             batch=batch_size,
                                             seqlen=seqlen)
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                if need_topk:
                    full_topk_ids = pad_input(
                        hidden_states=topk_ids, indices=indices, batch=batch_size, seqlen=seqlen)
                    full_topk_log_probs = pad_input(
                        hidden_states=topk_log_probs, indices=indices, batch=batch_size, seqlen=seqlen)
                    topk_ids = full_topk_ids[:, -response_length - 1:-1, :]
                    topk_log_probs = full_topk_log_probs[:, -response_length - 1:-1, :]

            else:  # not using rmpad and no ulysses sp
                entropy = None
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab)
                need_topk = top_k > 0
                if need_topk:
                    log_probs_all = torch.log_softmax(logits, dim=-1)
                    log_probs = log_probs_all.gather(
                        dim=-1, index=micro_batch['responses'].unsqueeze(-1)).squeeze(-1)
                else:
                    log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                if calculate_entropy or not need_topk:
                    entropy = verl_F.entropy_from_logits(logits)
                if need_topk:
                    if student_top_k_ids is not None:
                        topk_ids = student_top_k_ids
                    else:
                        _, topk_ids = torch.topk(logits, k=top_k, dim=-1)
                    topk_log_probs = torch.log_softmax(logits, dim=-1).gather(dim=-1, index=topk_ids)

            return entropy, log_probs, topk_ids, topk_log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(
        self, data: DataProto
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
                                   Optional[torch.Tensor]]]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            By default (``top_k`` is 0 and ``calculate_entropy`` is false in ``data.meta_info``): a
            ``torch.Tensor`` of shape ``[batch_size, response_length]`` — log-probability of each
            response token under the actor.

            If ``top_k > 0`` or ``calculate_entropy`` is true: a tuple
            ``(log_probs, entropys, topk_ids, topk_log_probs)``. ``log_probs`` has the same shape as
            above. ``entropys`` is per-token entropy when entropy is computed, else ``None``.
            ``topk_ids`` and ``topk_log_probs`` are ``None`` when ``top_k`` is 0; when ``top_k > 0``,
            they have shape ``[batch_size, response_length, top_k]`` (token ids and log-probs for the
            top-``k`` logits at each response position).
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        top_k = int(data.meta_info.get('top_k', 0) or 0)
        calculate_entropy = bool(data.meta_info.get('calculate_entropy', False))

        log_probs_lst = []
        entropy_lst = []
        topk_ids_lst = []
        topk_log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                entropy, log_probs, topk_ids, topk_log_probs = self._forward_micro_batch(
                    micro_batch,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy or top_k > 0,
                    top_k=top_k,
                    student_top_k_ids=None,
                )
            log_probs_lst.append(log_probs)
            if entropy is not None:
                entropy_lst.append(entropy)
            if topk_ids is not None:
                topk_ids_lst.append(topk_ids)
            if topk_log_probs is not None:
                topk_log_probs_lst.append(topk_log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = torch.concat(entropy_lst, dim=0) if entropy_lst else None
        topk_ids_tensor = torch.concat(topk_ids_lst, dim=0) if topk_ids_lst else None
        topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0) if topk_log_probs_lst else None

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long, device=log_probs.device)
            log_probs = log_probs[revert_indices]
            if entropys is not None:
                entropys = entropys[revert_indices]
            if topk_ids_tensor is not None:
                topk_ids_tensor = topk_ids_tensor[revert_indices]
            if topk_log_probs_tensor is not None:
                topk_log_probs_tensor = topk_log_probs_tensor[revert_indices]

        if top_k > 0 or calculate_entropy:
            return log_probs, entropys, topk_ids_tensor, topk_log_probs_tensor
        return log_probs

    def compute_log_probs_for_ids(self, data: DataProto) -> torch.Tensor:
        if data.meta_info.get('use_dynamic_bsz', False):
            raise NotImplementedError('compute_log_probs_for_ids with use_dynamic_bsz is not supported')
        self.actor_module.eval()
        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']
        top_k = data.batch['target_ids'].shape[-1]
        batch = data.select(batch_keys=['responses', 'input_ids', 'attention_mask', 'position_ids', 'target_ids']).batch
        out_lst = []
        for micro_batch in batch.split(micro_batch_size):
            with torch.no_grad():
                _, _, _, topk_lp = self._forward_micro_batch(
                    micro_batch,
                    temperature=temperature,
                    calculate_entropy=False,
                    top_k=top_k,
                    student_top_k_ids=micro_batch['target_ids'],
                )
            out_lst.append(topk_lp)
        return torch.concat(out_lst, dim=0)

    def compute_distillation_reward(self, data: DataProto) -> DataProto:
        if data.meta_info.get('use_dynamic_bsz', False):
            raise NotImplementedError('compute_distillation_reward with use_dynamic_bsz is not supported')
        self.actor_module.eval()
        top_k = data.meta_info.get('log_prob_top_k', 0)
        strategy = data.meta_info.get('top_k_strategy', 'only_stu')
        reward_weight_mode = data.meta_info.get('reward_weight_mode', 'student_p')
        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']
        device = torch.cuda.current_device()
        S_on_T = None
        if strategy in ('only_tch', 'intersection', 'union', 'union-intersection'):
            mb_data = data.select(
                batch_keys=['responses', 'input_ids', 'attention_mask', 'position_ids', 'teacher_top_k_ids']).batch
            S_on_T_lst = []
            for micro_batch in mb_data.split(micro_batch_size):
                micro_batch = micro_batch.to(device)
                with torch.no_grad():
                    _, _, _, topk_lp = self._forward_micro_batch(
                        micro_batch,
                        temperature=temperature,
                        calculate_entropy=False,
                        top_k=top_k,
                        student_top_k_ids=micro_batch['teacher_top_k_ids'],
                    )
                S_on_T_lst.append(topk_lp)
            S_on_T = torch.concat(S_on_T_lst, dim=0)
        S_ids = data.batch['student_top_k_ids'].to(device)
        S_logp = data.batch['student_top_k_log_probs'].to(device)
        T_on_S = data.batch['teacher_on_student_log_probs'].to(device)
        T_ids = data.batch.get('teacher_top_k_ids')
        if T_ids is not None:
            T_ids = T_ids.to(device)
        T_logp = data.batch.get('teacher_top_k_log_probs')
        if T_logp is not None:
            T_logp = T_logp.to(device)
        overlap_mask = data.batch.get('overlap_mask')
        if overlap_mask is not None:
            overlap_mask = overlap_mask.to(device)

        def compute_reward_weights(S_logp, T_logp, valid_mask, weight_mode, normalize=True):
            if weight_mode == 'student_p':
                log_probs = S_logp
            elif weight_mode == 'teacher_p':
                log_probs = T_logp
            elif weight_mode == 'none':
                log_probs = torch.zeros_like(S_logp)
            else:
                raise ValueError(f'Unknown reward_weight_mode: {weight_mode}')
            log_probs = torch.where(valid_mask, log_probs, torch.full_like(log_probs, -float('inf')))
            if normalize:
                norm_log_weights = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
                weights = torch.exp(norm_log_weights)
            else:
                weights = torch.exp(log_probs)
            return torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

        res_tensors = {}
        if strategy == 'only_stu':
            kl_val = S_logp - T_on_S
            valid_mask = torch.ones_like(S_logp, dtype=torch.bool)
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
        elif strategy == 'only_tch':
            kl_val = S_on_T - T_logp
            valid_mask = torch.ones_like(S_on_T, dtype=torch.bool)
            norm_weights = compute_reward_weights(S_on_T, T_logp, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            res_tensors['union_top_k_ids'] = T_ids
        elif strategy == 'intersection':
            valid_mask = overlap_mask.bool()
            kl_val = S_logp - T_on_S
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
        elif strategy == 'union':
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)
            T_in_S = data.batch['teacher_in_student_mask'].bool().to(device)
            valid_mask = torch.cat([torch.ones_like(S_ids, dtype=torch.bool), ~T_in_S], dim=-1)
            kl_val = S_logp_union - T_logp_union
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            res_tensors['union_top_k_ids'] = union_ids
            res_tensors['union_top_k_log_probs'] = S_logp_union
            res_tensors['student_log_probs_on_teacher_ids'] = S_on_T
        elif strategy == 'union-intersection':
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)
            S_in_T = overlap_mask.bool().to(device)
            T_in_S = data.batch['teacher_in_student_mask'].bool().to(device)
            valid_mask = torch.cat([~S_in_T, ~T_in_S], dim=-1)
            kl_val = S_logp_union - T_logp_union
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(
                S_logp_union, T_logp_union, valid_mask, reward_weight_mode, normalize=False)
            rm_scores = -kl_val * norm_weights
            res_tensors['union_top_k_ids'] = union_ids
            res_tensors['union_top_k_log_probs'] = S_logp_union
            res_tensors['student_log_probs_on_teacher_ids'] = S_on_T
        else:
            raise ValueError(f'Unknown top_k_strategy: {strategy}')
        res_tensors['rm_scores'] = rm_scores
        return DataProto.from_dict(tensors=res_tensors)

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if 'response_mask' in data.batch.keys():
            select_keys.append('response_mask')
        if self.config.state_masking:
            select_keys.append('loss_mask')
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        if 'student_top_k_ids' in data.batch.keys():
            select_keys.append('student_top_k_ids')
        if 'student_top_k_log_probs' in data.batch.keys():
            select_keys.append('student_top_k_log_probs')
        if 'union_top_k_ids' in data.batch.keys():
            select_keys.append('union_top_k_ids')
            if 'student_top_k_ids' in select_keys:
                select_keys.remove('student_top_k_ids')
        if 'union_top_k_log_probs' in data.batch.keys():
            select_keys.append('union_top_k_log_probs')
            if 'student_top_k_log_probs' in select_keys:
                select_keys.remove('student_top_k_log_probs')
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1)
                attention_mask = data['attention_mask']
                if 'response_mask' in data:
                    response_mask = data['response_mask']
                else:
                    response_mask = attention_mask[:, -response_length:]
                if self.config.state_masking:
                    response_mask = data['loss_mask']
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']

                clip_ratio = self.config.clip_ratio
                entropy_coeff = self.config.entropy_coeff

                calculate_entropy = entropy_coeff != 0
                if advantages.dim() == 3:
                    top_k = advantages.shape[-1]
                    student_top_k_ids = None
                    if 'union_top_k_ids' in data:
                        student_top_k_ids = data['union_top_k_ids']
                    elif 'student_top_k_ids' in data:
                        student_top_k_ids = data['student_top_k_ids']
                    entropy, _, _, topk_log_probs = self._forward_micro_batch(
                        data, temperature=temperature, calculate_entropy=calculate_entropy,
                        top_k=top_k, student_top_k_ids=student_top_k_ids)
                    log_prob_for_loss = topk_log_probs
                    if 'union_top_k_log_probs' in data:
                        old_log_prob = data['union_top_k_log_probs']
                    elif 'student_top_k_log_probs' in data:
                        old_log_prob = data['student_top_k_log_probs']
                else:
                    entropy, log_prob, _, _ = self._forward_micro_batch(
                        data, temperature=temperature, calculate_entropy=calculate_entropy, top_k=0)
                    log_prob_for_loss = log_prob

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob_for_loss,
                    advantages=advantages,
                    eos_mask=response_mask,
                    cliprange=clip_ratio)

                if entropy is not None:
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)
                else:
                    entropy_loss = torch.tensor(0.0, device=log_prob_for_loss.device)

                policy_loss = pg_loss - entropy_loss * entropy_coeff

                log_metrics = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                }
                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    if advantages.dim() == 3:
                        ref_log_prob = ref_log_prob.unsqueeze(-1).expand_as(log_prob_for_loss)
                    kld = core_algos.kl_penalty(logprob=log_prob_for_loss,
                                                ref_logprob=ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    kl_mask = response_mask.unsqueeze(-1) if advantages.dim() == 3 else response_mask
                    kl_loss = masked_mean(kld, kl_mask)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    log_metrics['actor/kl_loss'] = kl_loss.detach().item()
                    log_metrics['actor/kl_coef'] = self.config.kl_loss_coef

                loss = policy_loss / self.gradient_accumulation
                loss.backward()

                append_to_dict(metrics, log_metrics)

            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
