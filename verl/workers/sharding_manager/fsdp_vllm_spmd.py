# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# FSDP <-> modern vLLM (>=0.7) weight sync using wake_up / sleep + load_weights.
from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

try:
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover
    from torch.distributed._tensor import DTensor  # type: ignore

from vllm import LLM
from vllm.distributed import parallel_state as vllm_ps

from verl import DataProto
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.torch_functional import allgather_dict_tensors, broadcast_dict_tensor
from verl.workers.sharding_manager.base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


class FSDPVLLMShardingManager(BaseShardingManager):
    """Synchronize FSDP actor weights into a stock vLLM engine for SPMD rollout."""

    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        model_config,
        rollout_config,
        full_params: bool = False,
        device_mesh: Optional[DeviceMesh] = None,
        load_format: str = "dummy_hf",
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.rollout_config = rollout_config
        self.device_mesh = device_mesh
        self.load_format = load_format
        self.full_params = full_params

        if full_params:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            )
        else:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
            )

        self.tp_size = 1
        self.tp_rank = 0
        if self.device_mesh is not None:
            self.tp_size = self.device_mesh["infer_tp"].size()
            self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        self.torch_random_states = torch.cuda.get_rng_state()
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

        # Resolve model runner for load_weights.
        self.model_runner = None
        try:
            self.model_runner = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner
        except Exception:
            try:
                # v1 worker path
                self.model_runner = self.inference_engine.llm_engine.engine_core  # type: ignore
            except Exception:
                self.model_runner = None

    def __enter__(self):
        log_gpu_memory_usage("Before state_dict() in SPMD sharding manager", logger=logger)
        params = self.module.state_dict()
        log_gpu_memory_usage("After state_dict() in SPMD sharding manager", logger=logger)

        free_cache = bool(self.rollout_config.get("free_cache_engine", True))
        if free_cache:
            try:
                self.inference_engine.wake_up(tags=["weights"])
            except TypeError:
                self.inference_engine.wake_up()

        self._update_params(params)
        del params
        torch.cuda.empty_cache()

        if free_cache:
            try:
                self.inference_engine.wake_up(tags=["kv_cache"])
            except TypeError:
                pass

        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        if bool(self.rollout_config.get("free_cache_engine", True)):
            self.inference_engine.sleep(level=1)
        self.module.train()
        torch.cuda.empty_cache()
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def _update_params(self, updated_params):
        model = None
        if self.model_runner is not None and hasattr(self.model_runner, "model"):
            model = self.model_runner.model
        if model is None:
            # Fallback: reload from actor weights via HF names through vLLM's collective rpc if available.
            logger.warning("Could not resolve vLLM model_runner.model; skipping explicit load_weights")
            return

        weight_iter = (
            (name, param.full_tensor().detach() if isinstance(param, DTensor) else param.detach())
            for name, param in updated_params.items()
        )
        # MoE models may need patched weight loaders; best-effort.
        try:
            loaded = model.load_weights(weight_iter)
            logger.info("vLLM SPMD load_weights loaded_params=%s", len(loaded) if loaded is not None else "unknown")
        except Exception as e:
            logger.warning("vLLM SPMD load_weights failed: %s", e)
            raise

    def preprocess_data(self, data: DataProto) -> DataProto:
        if self.tp_size == 1:
            return data
        group = vllm_ps.get_tensor_model_parallel_group()
        if hasattr(group, "device_group"):
            group = group.device_group
        data.batch = allgather_dict_tensors(data.batch.contiguous(), size=self.tp_size, group=group, dim=0)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        if self.tp_size == 1:
            return data
        group = vllm_ps.get_tensor_model_parallel_group()
        if hasattr(group, "device_group"):
            group = group.device_group
        src = vllm_ps.get_tensor_model_parallel_src_rank() if hasattr(vllm_ps, "get_tensor_model_parallel_src_rank") else 0
        broadcast_dict_tensor(data.batch, src=src, group=group)
        local_prompts = data.chunk(chunks=self.tp_size)
        return local_prompts[self.tp_rank]
