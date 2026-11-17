# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_reduce_scatter,
    sp_shard,
)


@dataclass(frozen=True)
class ResidualBoundary:
    """Move between residual rows and full-row TP computation.

    Tensors have shape `[tokens, hidden]` with the full hidden width. Shards
    use TP rank order and pad token rows. A complete output can alias its
    input, and a local shard can be a view.

    Args:
        output_is_partial: Whether compute leaves its TP output sum pending.

    """

    output_is_partial: bool

    def prepare_input(
        self,
        x: torch.Tensor,
        *,
        num_tokens: int,
        sequence_parallel: bool,
    ) -> torch.Tensor:
        """Gather local residual readouts and trim collective padding."""
        if sequence_parallel:
            return sp_all_gather(x)[:num_tokens]
        return x

    def complete_output(
        self, x: torch.Tensor, *, sequence_parallel: bool
    ) -> torch.Tensor:
        """Complete a producer's sum and return the residual's token rows."""
        if self.output_is_partial:
            if sequence_parallel:
                return sp_reduce_scatter(x)
            return tensor_model_parallel_all_reduce(x)
        if sequence_parallel:
            return sp_shard(x)
        return x
