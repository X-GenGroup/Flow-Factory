# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""SenseNova output boundaries for token-local and spatial generation heads."""

from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import ParallelStyle, PrepareModuleOutput

from ..parallelism.collectives import gather_tokens


class PointwiseHeadSetup(ParallelStyle):
    """Keep the ordinary MLP head local; spatial/deep heads retain their full inputs."""

    def _apply(self, module, device_mesh):
        model = module.model
        pointwise = not model.use_pixel_head and not model.use_deep_fm_head
        module._ff_sensenova_pointwise_sp_mesh = device_mesh if pointwise else None
        model.language_model.model.norm_mot_gen._ff_sensenova_pointwise_sp = pointwise
        return module


class DecoderOutput(ParallelStyle):
    """Gather decoder tokens for head paths outside the pointwise SP region."""

    def _apply(self, module, device_mesh):
        if not getattr(module, "_ff_sensenova_pointwise_sp", False):
            PrepareModuleOutput(
                output_layouts=Shard(1),
                desired_output_layouts=Replicate(),
                use_local_output=True,
            )._apply(module, device_mesh)
        return module


def gather_pointwise_output(denoiser, output, token_count):
    """Restore global token order after the MLP projection, before native velocity arithmetic."""
    tp_mesh = getattr(denoiser, "_ff_sensenova_pointwise_sp_mesh", None)
    if tp_mesh is None:
        return output
    output = (
        DTensor.from_local(output, tp_mesh, [Shard(1)], run_check=False)
        .redistribute(placements=[Replicate()])
        .to_local()
    )
    parallel = getattr(denoiser, "_parallel_config", None)
    if parallel is not None:
        mesh = parallel.context_parallel_config._flattened_mesh
        # The outer CP output boundary assigns disjoint loss rows to CP ranks.
        # Sum those contributions before returning to the local head output.
        output = gather_tokens(output, mesh, 1)
    return output[:, :token_count]
