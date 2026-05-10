# Copyright 2024 Bytedance Ltd. and/or its affiliates
# OPD teacher reward worker adapted from thunlp/OPD (verl fork).
# Loads AutoModelForCausalLM as teacher and returns token-level distillation tensors.
"""FSDP reward worker for on-policy distillation (OPD): causal LM teacher, not token-classification RM."""

import logging
import os
import warnings

import torch
import torch.distributed
import torch.nn.functional as F
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig
from tensordict import TensorDict

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, CPUOffload
from transformers import AutoConfig, AutoModelForCausalLM

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


class OPDTeacherRewardWorker(Worker):
    """Causal LM teacher for OPD: outputs teacher log probs and optional top-k diagnostics."""

    def __init__(self, config: DictConfig):
        super().__init__()
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')
        self.config = config

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        if self.ulysses_sequence_parallel_size > 1:
            raise NotImplementedError(
                'OPDTeacherRewardWorker currently requires ulysses_sequence_parallel_size==1')
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.use_remove_padding = self.config.model.get('use_remove_padding', False)
        if self.use_remove_padding:
            raise NotImplementedError(
                'OPD teacher with use_remove_padding=True is not supported in this port; set '
                'reward_model.model.use_remove_padding=false for OPD.')

        self.config.micro_batch_size //= torch.distributed.get_world_size()

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True

    def _build_model(self, config):
        local_path = copy_local_path_from_hdfs(config.model.path)
        if self._do_switch_chat_template:
            input_tokenizer_local_path = copy_local_path_from_hdfs(config.model.input_tokenizer)
            self.input_tokenizer = hf_tokenizer(
                input_tokenizer_local_path, trust_remote_code=config.model.get('trust_remote_code', False))
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get('trust_remote_code', False))
        else:
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get('trust_remote_code', False))

        trust_remote_code = config.model.get('trust_remote_code', False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_dtype_str = config.model.get('dtype', 'bf16')
        model_dtype = PrecisionType.to_dtype(model_dtype_str)

        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings)
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            reward_module = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=model_dtype,
                attn_implementation='flash_attention_2',
                trust_remote_code=trust_remote_code,
            )
            reward_module.to(model_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)
        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            sync_module_states=True,
            cpu_offload=CPUOffload(offload_params=self.config.model.fsdp_config.param_offload),
            forward_prefetch=False)
        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.utils.import_utils import import_external_libs
        import_external_libs(self.config.model.get('external_lib', None))
        self.reward_module = self._build_model(config=self.config)
        torch.cuda.empty_cache()

    def _compute_entropy_safe(self, logits, chunk_size=4096):
        original_shape = logits.shape
        vocab_size = original_shape[-1]
        logits_flat = logits.reshape(-1, vocab_size)
        entropy_list = []
        for i in range(0, logits_flat.size(0), chunk_size):
            chunk = logits_flat[i:i + chunk_size]
            log_probs = F.log_softmax(chunk, dim=-1)
            probs = torch.exp(log_probs)
            entropy = -torch.sum(probs * log_probs, dim=-1)
            entropy_list.append(entropy)
        entropy_flat = torch.cat(entropy_list, dim=0)
        return entropy_flat.reshape(original_shape[:-1])

    def _compute_teacher_top_k_log_probs(self, logits, student_ids, top_k, strategy='only_stu', chunk_size=1024):
        n_samples = logits.size(0)
        results = []
        valid_counts_list = []
        overlap_counts_list = []
        teacher_top_k_ids_list = []
        teacher_top_k_log_probs_list = []
        teacher_in_student_list = []

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            logits_chunk = logits[start:end]
            student_ids_chunk = student_ids[start:end]

            t_logits, t_ids = torch.topk(logits_chunk, k=top_k, dim=-1)
            t_logsumexp = torch.logsumexp(logits_chunk, dim=-1, keepdim=True)
            t_log_probs_top_k = t_logits - t_logsumexp

            teacher_top_k_ids_list.append(t_ids)
            teacher_top_k_log_probs_list.append(t_log_probs_top_k)

            s_ids_exp = student_ids_chunk.unsqueeze(-1)
            t_ids_exp = t_ids.unsqueeze(-2)
            matches = (s_ids_exp == t_ids_exp)
            is_in_teacher = matches.any(dim=-1)
            overlap_counts_list.append(is_in_teacher.float())
            is_in_student = matches.any(dim=-2)
            teacher_in_student_list.append(is_in_student.float())

            if strategy in ('only_stu', 'union', 'union-intersection'):
                chunk_log_probs = torch.gather(logits_chunk, dim=-1, index=student_ids_chunk) - t_logsumexp
                results.append(chunk_log_probs)
                chunk_valid_counts = torch.full(
                    (logits_chunk.size(0),), student_ids_chunk.size(-1), device=logits.device, dtype=torch.long)
                valid_counts_list.append(chunk_valid_counts)
            else:
                t_vals_exp = t_log_probs_top_k.unsqueeze(-2)
                vals_masked = t_vals_exp.masked_fill(~matches, float('-inf'))
                chunk_res, _ = vals_masked.max(dim=-1)
                results.append(chunk_res)
                valid_counts_list.append(is_in_teacher.float().sum(dim=-1).long())

        return (
            torch.cat(results, dim=0),
            torch.cat(valid_counts_list, dim=0),
            torch.cat(overlap_counts_list, dim=0),
            torch.cat(teacher_top_k_ids_list, dim=0),
            torch.cat(teacher_top_k_log_probs_list, dim=0),
            torch.cat(teacher_in_student_list, dim=0),
        )

    def _forward_micro_batch(self, micro_batch, student_top_k_ids=None, compute_entropy=False,
                             top_k=0, strategy='only_stu', teacher_temperature=1.0):
        response_length = micro_batch['responses'].size(-1)
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            teacher_on_student_log_probs = None
            teacher_top_k_ids = None
            teacher_top_k_log_probs = None
            teacher_entropy = None
            teacher_valid_counts = None
            teacher_overlap_mask = None
            teacher_in_student_mask = None

            output = self.reward_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            rm_output_logits = output[0] if isinstance(output, tuple) else output.logits
            rm_logits_resp = rm_output_logits[:, -response_length - 1:-1, :]
            rm_logits_resp = rm_logits_resp.div_(teacher_temperature)

            if compute_entropy:
                teacher_entropy = self._compute_entropy_safe(rm_logits_resp)

            rm_log_probs = verl_F.logprobs_from_logits(rm_logits_resp, micro_batch['responses'])

            if student_top_k_ids is not None:
                if top_k > 0:
                    original_shape = student_top_k_ids.shape
                    flat_logits = rm_logits_resp.reshape(-1, rm_logits_resp.size(-1))
                    flat_ids = student_top_k_ids.reshape(-1, student_top_k_ids.size(-1))
                    flat_on_student_log_probs, flat_counts, flat_overlap_mask, flat_teacher_top_k_ids, \
                        flat_teacher_top_k_log_probs, flat_teacher_in_student = self._compute_teacher_top_k_log_probs(
                            logits=flat_logits,
                            student_ids=flat_ids,
                            top_k=top_k,
                            strategy=strategy,
                        )
                    teacher_on_student_log_probs = flat_on_student_log_probs.view(original_shape)
                    teacher_valid_counts = flat_counts.view(original_shape[:-1])
                    teacher_overlap_mask = flat_overlap_mask.view(original_shape)
                    teacher_top_k_ids = flat_teacher_top_k_ids.view(
                        original_shape[0], original_shape[1], top_k)
                    teacher_top_k_log_probs = flat_teacher_top_k_log_probs.view(
                        original_shape[0], original_shape[1], top_k)
                    teacher_in_student_mask = flat_teacher_in_student.view(
                        original_shape[0], original_shape[1], top_k)
                else:
                    teacher_on_student_log_probs = torch.gather(
                        rm_logits_resp, dim=-1, index=student_top_k_ids)
                    teacher_logsumexp = torch.logsumexp(rm_logits_resp, dim=-1, keepdim=True)
                    teacher_on_student_log_probs = teacher_on_student_log_probs - teacher_logsumexp

        return (rm_log_probs, teacher_on_student_log_probs, teacher_top_k_ids, teacher_top_k_log_probs,
                teacher_entropy, teacher_valid_counts, teacher_overlap_mask, teacher_in_student_mask)

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch['attention_mask'].shape[-1]
        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer
        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            chat = list(data.non_tensor_batch['raw_prompt'][i])
            response_ids = data.batch['responses'][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch['attention_mask'][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response = src_tokenizer.decode(valid_response_ids)
            response = response.replace(src_tokenizer.eos_token, '')
            chat.append({'role': 'assistant', 'content': response})
            prompt_with_chat_template = target_tokenizer.apply_chat_template(
                chat, add_generation_prompt=False, tokenize=False)
            max_length = self.config.get('max_length', src_max_length)
            if max_length is None:
                max_length = src_max_length
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=target_tokenizer,
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,
                truncation=self.config.get('truncation', 'right'))
            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)
        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)
        rm_inputs = {'input_ids': rm_input_ids, 'attention_mask': rm_attention_mask, 'position_ids': rm_position_ids}
        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        data = data.to('cuda')
        student_logp = data.batch['old_log_probs']
        student_top_k_ids = data.batch.get('student_top_k_ids', None)
        student_top_k_log_probs = data.batch.get('student_top_k_log_probs', None)

        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)
        else:
            rm_inputs = {
                'input_ids': data.batch['input_ids'],
                'attention_mask': data.batch['attention_mask'],
                'position_ids': data.batch['position_ids'],
                'responses': data.batch['responses'],
            }
            rm_data = DataProto.from_dict(rm_inputs)

        rm_data.batch = rm_data.batch.cuda()
        if student_top_k_ids is not None:
            rm_data.batch['student_top_k_ids'] = student_top_k_ids

        top_k = data.meta_info.get('log_prob_top_k', self.config.get('log_prob_top_k', 0))
        top_k_strategy = data.meta_info.get('top_k_strategy', self.config.get('top_k_strategy', 'only_stu'))
        teacher_temperature = data.meta_info.get(
            'teacher_temperature', self.config.get('teacher_temperature', 1.0))

        compute_entropy = True

        with self.ulysses_sharding_manager:
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)

            if self.config.use_dynamic_bsz:
                raise NotImplementedError('OPD teacher RM does not support use_dynamic_bsz; disable on reward_model.')
            micro_batches = rm_data.batch.split(self.config.micro_batch_size)

            output_logp = []
            output_on_student_logp = []
            output_teacher_top_k_ids = []
            output_teacher_top_k_logp = []
            output_entropy = []
            output_valid_counts = []
            output_overlap_counts = []
            output_teacher_in_student = []

            for micro_batch in micro_batches:
                mb_top_k_ids = None
                if isinstance(micro_batch, TensorDict):
                    mb_top_k_ids = micro_batch.get('student_top_k_ids', None)
                elif hasattr(micro_batch, 'get'):
                    mb_top_k_ids = micro_batch.get('student_top_k_ids', None)

                teacher_logp_batch, teacher_on_student_logp_batch, teacher_top_k_ids_batch, \
                    teacher_top_k_logp_teacher_batch, teacher_entropy_batch, teacher_valid_counts_batch, \
                    teacher_overlap_mask_batch, teacher_in_student_mask_batch = self._forward_micro_batch(
                        micro_batch,
                        student_top_k_ids=mb_top_k_ids,
                        compute_entropy=compute_entropy,
                        top_k=top_k,
                        strategy=top_k_strategy,
                        teacher_temperature=teacher_temperature,
                    )
                output_logp.append(teacher_logp_batch)
                if teacher_on_student_logp_batch is not None:
                    output_on_student_logp.append(teacher_on_student_logp_batch)
                if teacher_top_k_ids_batch is not None:
                    output_teacher_top_k_ids.append(teacher_top_k_ids_batch)
                if teacher_top_k_logp_teacher_batch is not None:
                    output_teacher_top_k_logp.append(teacher_top_k_logp_teacher_batch)
                if teacher_entropy_batch is not None:
                    output_entropy.append(teacher_entropy_batch)
                if teacher_valid_counts_batch is not None:
                    output_valid_counts.append(teacher_valid_counts_batch)
                if teacher_overlap_mask_batch is not None:
                    output_overlap_counts.append(teacher_overlap_mask_batch)
                if teacher_in_student_mask_batch is not None:
                    output_teacher_in_student.append(teacher_in_student_mask_batch)

            teacher_logp = torch.cat(output_logp, dim=0)
            teacher_on_student_logp = torch.cat(output_on_student_logp, dim=0) if output_on_student_logp else None
            teacher_top_k_ids = torch.cat(output_teacher_top_k_ids, dim=0) if output_teacher_top_k_ids else None
            teacher_top_k_logp = torch.cat(output_teacher_top_k_logp, dim=0) if output_teacher_top_k_logp else None
            teacher_entropy = torch.cat(output_entropy, dim=0) if output_entropy else None
            teacher_valid_counts = torch.cat(output_valid_counts, dim=0) if output_valid_counts else None
            teacher_overlap_mask = torch.cat(output_overlap_counts, dim=0) if output_overlap_counts else None
            teacher_in_student_mask = torch.cat(output_teacher_in_student, dim=0) if output_teacher_in_student else None

            if top_k > 0:
                rm_scores = None
                overlap_mask = teacher_overlap_mask
            else:
                reverse_kl = student_logp - teacher_logp
                rm_scores = -reverse_kl
                teacher_valid_counts = None
                overlap_mask = None

            tensors = {}
            if rm_scores is not None:
                tensors['rm_scores'] = rm_scores
            if teacher_on_student_logp is not None:
                tensors['teacher_on_student_log_probs'] = teacher_on_student_logp
            if teacher_top_k_ids is not None:
                tensors['teacher_top_k_ids'] = teacher_top_k_ids
            if teacher_top_k_logp is not None:
                tensors['teacher_top_k_log_probs'] = teacher_top_k_logp
            if teacher_entropy is not None:
                tensors['teacher_entropy'] = teacher_entropy
            if teacher_valid_counts is not None:
                tensors['teacher_valid_counts'] = teacher_valid_counts
            if overlap_mask is not None:
                tensors['overlap_mask'] = overlap_mask
            if teacher_in_student_mask is not None:
                tensors['teacher_in_student_mask'] = teacher_in_student_mask

            output = DataProto.from_dict(tensors=tensors)
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to('cpu')
        torch.cuda.empty_cache()
        return output
