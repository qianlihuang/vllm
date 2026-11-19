# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash latent experts with HYV4's routing, shared experts and weight format."""

import torch
from torch import nn

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.common.ops.sequence_parallel import sp_all_gather, sp_reduce_scatter
from vllm.models.hy_v4.nvidia.moe import HYV4FeedForward, HYV4MoEFused
from vllm.models.hy_v4_flash.configs import HYV4FlashConfig


class HYV4FlashFeedForward(HYV4FeedForward):
    def __init__(
        self,
        config: HYV4FlashConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
        *,
        reduce_results: bool = True,
        apply_post_norm: bool = True,
    ) -> None:
        super().__init__(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=prefix,
        )
        self.apply_post_norm = apply_post_norm
        self.post_fc2_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.sandwich_norm
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = super().forward(x)
        if self.apply_post_norm and self.post_fc2_norm is not None:
            return self.post_fc2_norm(result)
        return result


class HYV4FlashMoE(HYV4MoEFused):
    def __init__(
        self,
        config: HYV4FlashConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
        *,
        reduce_results: bool = True,
        apply_post_norm: bool = True,
        separate_shared_experts: bool = False,
    ) -> None:
        latent = config.routed_expert_hidden_size
        down: nn.Module | None = None
        up: nn.Module | None = None
        if latent is not None:
            down = ReplicatedLinear(
                config.hidden_size,
                latent,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.routed_expert_down_proj",
            )
            up = ReplicatedLinear(
                latent,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.routed_expert_up_proj",
            )
            # Bias-free unquantized up-proj commutes with TP partial sum.
            up.reduce_commutative = isinstance(up.quant_method, UnquantizedLinearMethod)
        super().__init__(
            config,
            quant_config=quant_config,
            prefix=prefix,
            reduce_results=reduce_results,
            routed_hidden_size=latent,
            routed_input_transform=down,
            routed_output_transform=up,
            separate_shared_experts=separate_shared_experts,
        )
        self.apply_post_norm = apply_post_norm
        self.routed_expert_down_proj = down
        self.routed_expert_up_proj = up
        self.combine_output_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.sandwich_norm
            else None
        )

    def forward_sp_passthrough(
        self, hidden_states: torch.Tensor, full_num_tokens: int
    ) -> torch.Tensor:
        """Run passthrough EP routed experts on local SP shard [T_local, D]
        and TP-sharded shared experts on all-gathered [T, D] -> reduce-scatter."""
        routed = super().forward(hidden_states)
        if self.shared_experts is not None:
            shared_in = sp_all_gather(hidden_states)[:full_num_tokens]
            shared = self.shared_experts(shared_in)
            shared = sp_reduce_scatter(shared)
            result = routed + shared
        else:
            result = routed
        return (
            self.combine_output_norm(result)
            if self.combine_output_norm is not None
            else result
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        result = super().forward(hidden_states)
        if self.separate_shared_experts and self.shared_experts is not None:
            shared = self.shared_experts(hidden_states)
            if self.tp_size > 1:
                shared = tensor_model_parallel_all_reduce(shared)
            result = result + shared
        if self.apply_post_norm and self.combine_output_norm is not None:
            return self.combine_output_norm(result)
        return result
