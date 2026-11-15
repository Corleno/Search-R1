# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Modern vLLM (>=0.7) SPMD rollout using stock vllm.LLM (external_launcher).
# Kept separate from the legacy third_party-wrapped vLLMRollout for 0.6.3.
"""SPMD vLLM rollout compatible with Search-R1 DataProto + SDPO trainer."""
from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.distributed
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from vllm import LLM, SamplingParams

from verl import DataProto
from verl.utils.qwen36 import get_max_position_embeddings
from verl.utils.torch_functional import get_eos_mask
from verl.workers.rollout.base import BaseRollout


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    return prompt_token_ids[non_pad_index:].tolist()


class vLLMRollout(BaseRollout):
    """vLLM rollout for vLLM >= 0.7 using stock LLM + weight sync via sharding manager."""

    def __init__(
        self,
        actor_module: nn.Module,
        config: DictConfig,
        tokenizer,
        model_hf_config,
        model_path: str,
        trust_remote_code: bool = False,
        device_mesh: Optional[DeviceMesh] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.device_mesh = device_mesh
        self.pad_token_id = tokenizer.pad_token_id

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )

        max_position_embeddings = get_max_position_embeddings(model_hf_config)
        assert max_position_embeddings >= config.prompt_length + config.response_length, (
            "model context length should be greater than total sequence length"
        )

        max_model_len = int(config.get("max_model_len") or (config.prompt_length + config.response_length))
        max_num_batched_tokens = int(config.get("max_num_batched_tokens", max(8192, max_model_len)))
        load_format = config.load_format
        if isinstance(load_format, str) and load_format.startswith("dummy"):
            # Still load real HF weights once; sharding manager will overwrite from FSDP.
            load_format = "auto"

        engine_kwargs = {}
        # Text-only MoE path for Qwen3.6 multimodal checkpoints when supported.
        if config.get("language_model_only", True):
            # vLLM 0.29 accepts this for Qwen3.5/3.6 MoE VL checkpoints.
            engine_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=bool(config.get("free_cache_engine", True)),
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.get("enforce_eager", True),
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=True,
            max_num_batched_tokens=max_num_batched_tokens,
            # Qwen3.5/3.6 hybrid (GDN/mamba) defaults to mamba_cache_mode=align with
            # prefix caching, which requires chunked prefill in vLLM >= 0.29.
            enable_chunked_prefill=config.get("enable_chunked_prefill", True),
            enable_prefix_caching=config.get("enable_prefix_caching", True),
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **engine_kwargs,
        )

        # Sleep after init so FSDP can own GPU memory until the first sync.
        if config.get("free_cache_engine", True):
            self.inference_engine.sleep(level=1)

        sampling_kwargs = dict(
            n=1,
            logprobs=1,
            max_tokens=config.response_length,
            detokenize=False,
        )
        for k in config.keys():
            if k == "seed":
                continue
            if hasattr(SamplingParams(), str(k)):
                sampling_kwargs[k] = config.get(k)
        print(f"vLLM SPMD sampling kwargs: {sampling_kwargs}")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_sampling_params_args[key] = getattr(self.sampling_params, key)
                    setattr(self.sampling_params, key, value)
        yield
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)

        idx_list = [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)]

        do_sample = prompts.meta_info.get("do_sample", True)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,
            }

        vllm_inputs = [{"prompt_token_ids": tokens} for tokens in idx_list]
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

        # Convert RequestOutput list -> response token tensor.
        response_tokens: List[List[int]] = []
        log_probs_list: List[List[float]] = []
        for out in outputs:
            seq = out.outputs[0]
            response_tokens.append(list(seq.token_ids))
            if seq.logprobs is not None and len(seq.logprobs) > 0:
                step_lps = []
                for t_id, lp_map in zip(seq.token_ids, seq.logprobs):
                    if lp_map is None:
                        step_lps.append(0.0)
                    else:
                        # lp_map: Dict[token_id, Logprob]
                        entry = lp_map.get(t_id)
                        step_lps.append(float(entry.logprob) if entry is not None else 0.0)
                log_probs_list.append(step_lps)
            else:
                log_probs_list.append([0.0] * len(seq.token_ids))

        max_len = self.config.response_length
        response = torch.full((batch_size, max_len), self.pad_token_id, dtype=idx.dtype, device=idx.device)
        log_probs = torch.zeros((batch_size, max_len), dtype=torch.float32, device=idx.device)
        for i, (toks, lps) in enumerate(zip(response_tokens, log_probs_list)):
            n = min(len(toks), max_len)
            if n > 0:
                response[i, :n] = torch.tensor(toks[:n], dtype=idx.dtype, device=idx.device)
                log_probs[i, :n] = torch.tensor(lps[:n], dtype=torch.float32, device=idx.device)

        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n

        seq = torch.cat([idx, response], dim=-1)
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        # Logprobs are recomputed by the actor; keep tensor construction for parity.
        _ = log_probs
        return DataProto(batch=batch)
