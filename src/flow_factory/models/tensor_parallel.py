# Copyright 2026 Jayce-Ping
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


"""Compatibility imports for model parallelism. New code uses ``models.parallelism``."""

from .parallelism.collectives import _REGION_ENTRY_NODES as _REGION_ENTRY_NODES
from .parallelism.collectives import _AllReduceGradient as _AllReduceGradient
from .parallelism.collectives import _GatherTokens as _GatherTokens
from .parallelism.collectives import _passed_region_entry as _passed_region_entry
from .parallelism.collectives import _RegionAllReduceGradient as _RegionAllReduceGradient
from .parallelism.collectives import _settled as _settled
from .parallelism.guards import _comparable_bits as _comparable_bits
from .parallelism.guards import _guard_sequence_divisibility as _guard_sequence_divisibility
from .parallelism.guards import _install_norm_pair as _install_norm_pair
from .parallelism.guards import _install_region_input_checks as _install_region_input_checks
from .parallelism.guards import _leaf_styles as _leaf_styles
from .parallelism.guards import _NormPair as _NormPair
from .parallelism.guards import _Output as _Output
from .parallelism.guards import _pair_across_head_norms as _pair_across_head_norms
from .parallelism.guards import _require_region_entries as _require_region_entries
from .parallelism.guards import _same_key as _same_key
from .parallelism.guards import _scatter_entries as _scatter_entries
from .parallelism.guards import _settle_collective_outputs as _settle_collective_outputs
from .parallelism.guards import _shared_input_projections as _shared_input_projections
from .parallelism.guards import _tensor_leaves as _tensor_leaves
from .parallelism.guards import install_replicated_input_check as install_replicated_input_check
from .parallelism.packed import PackedColwiseParallel as PackedColwiseParallel
from .parallelism.packed import PackedRowwiseParallel as PackedRowwiseParallel
from .parallelism.peft import _lora_adapter_names as _lora_adapter_names
from .parallelism.peft import derive_lora_plan as derive_lora_plan
from .parallelism.peft import peel_peft_wrapper as peel_peft_wrapper
from .parallelism.plan import _ACTIVATION_ONLY_STYLES as _ACTIVATION_ONLY_STYLES
from .parallelism.plan import ComposedParallelStyle as ComposedParallelStyle
from .parallelism.plan import ConditionalEntry as ConditionalEntry
from .parallelism.plan import _adapter_name as _adapter_name
from .parallelism.plan import _expand_globs as _expand_globs
from .parallelism.plan import _modules_owning_parameters as _modules_owning_parameters
from .parallelism.plan import _reject_unmatched as _reject_unmatched
from .parallelism.plan import _resolve_patterns as _resolve_patterns
from .parallelism.plan import build_total_plan as build_total_plan
from .parallelism.plan import resolve_parallel_plan as resolve_parallel_plan
from .parallelism.plan import sequence_parallel_enabled as sequence_parallel_enabled
from .parallelism.runtime import _LOCALIZED_HEADS_MARKER as _LOCALIZED_HEADS_MARKER
from .parallelism.runtime import _owns_a_column_shard as _owns_a_column_shard
from .parallelism.runtime import apply_tensor_parallel as apply_tensor_parallel
from .parallelism.runtime import localize_attention_head_counts as localize_attention_head_counts
from .parallelism.runtime import logger as logger
from .parallelism.sequence import ColumnParallelRegion as ColumnParallelRegion
from .parallelism.sequence import SequenceAlignedOutput as SequenceAlignedOutput
from .parallelism.sequence import SequenceBroadcastOutput as SequenceBroadcastOutput
from .parallelism.sequence import SequenceBroadcastParameter as SequenceBroadcastParameter
from .parallelism.sequence import SequenceParallelRegion as SequenceParallelRegion
from .parallelism.sequence import SequenceShardedArgument as SequenceShardedArgument
from .parallelism.sequence import _argument_positions as _argument_positions
from .parallelism.sequence import _RegionEntry as _RegionEntry
from .parallelism.sequence import _validated_names as _validated_names
from .parallelism.tensor import ReplicateParallel as ReplicateParallel
from .parallelism.tensor import ShardedAcrossHeadRMSNorm as ShardedAcrossHeadRMSNorm
from .parallelism.tensor import ShardedInputReplicatedWeight as ShardedInputReplicatedWeight
from .parallelism.tensor import SharedInputColwiseParallel as SharedInputColwiseParallel
from .parallelism.tensor import _across_head_normalize as _across_head_normalize
from .parallelism.tensor import _across_head_square_sum as _across_head_square_sum
from .parallelism.tensor import _distribute_own_parameters as _distribute_own_parameters
from .parallelism.tensor import _LocalViewStyle as _LocalViewStyle
from .parallelism.tensor import _norm_accumulation_input as _norm_accumulation_input
from .parallelism.tensor import _OwnParameterStyle as _OwnParameterStyle
from .parallelism.tensor import _replicated_view as _replicated_view
from .parallelism.tensor import _ToLocalWithPartialGradient as _ToLocalWithPartialGradient

__all__ = [
    "ColumnParallelRegion",
    "PackedColwiseParallel",
    "PackedRowwiseParallel",
    "ComposedParallelStyle",
    "ReplicateParallel",
    "SequenceParallelRegion",
    "SharedInputColwiseParallel",
    "SequenceBroadcastOutput",
    "SequenceBroadcastParameter",
    "ShardedAcrossHeadRMSNorm",
    "peel_peft_wrapper",
    "ShardedInputReplicatedWeight",
    "apply_tensor_parallel",
    "ConditionalEntry",
    "build_total_plan",
    "resolve_parallel_plan",
    "localize_attention_head_counts",
]
