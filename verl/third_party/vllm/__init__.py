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

from importlib.metadata import PackageNotFoundError, version

from packaging.version import InvalidVersion, Version


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


package_name = "vllm"
package_version = get_version(package_name)

_LEGACY = {"0.3.1", "0.4.2", "0.5.4", "0.6.3"}


def _parsed_version(ver: str):
    try:
        return Version(ver.split("+")[0])
    except (InvalidVersion, AttributeError):
        return None


if package_version in _LEGACY:
    vllm_version = package_version
    if package_version == "0.3.1":
        from .vllm_v_0_3_1 import parallel_state
        from .vllm_v_0_3_1.llm import LLM, LLMEngine
    elif package_version == "0.4.2":
        from .vllm_v_0_4_2 import parallel_state
        from .vllm_v_0_4_2.llm import LLM, LLMEngine
    elif package_version == "0.5.4":
        from .vllm_v_0_5_4 import parallel_state
        from .vllm_v_0_5_4.llm import LLM, LLMEngine
    else:  # 0.6.3
        from .vllm_v_0_6_3 import parallel_state
        from .vllm_v_0_6_3.llm import LLM, LLMEngine
elif package_version is not None and (_parsed_version(package_version) or Version("0")) >= Version("0.7.0"):
    # Modern stock vLLM used by SPMD rollout (Search-R1 Qwen3.6 path).
    vllm_version = package_version
    from vllm import LLM
    from vllm.distributed import parallel_state

    LLMEngine = None  # unused by SPMD path
else:
    raise ValueError(
        f"vllm version {package_version} not supported. "
        "Supported: 0.3.1, 0.4.2, 0.5.4, 0.6.3, or >=0.7.0 (SPMD)."
    )
