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

"""Reference-image Guard lifecycle with the Flux dual-stream image SP region."""

import json
from contextlib import contextmanager

import flux_kontext_coupled_fixture as kontext

from diffusers import FluxTransformer2DModel
from flow_factory.utils.dist import context_parallel_geometry, tensor_parallel_geometry

wrap_classes = kontext.wrap_classes
reward_media_field = kontext.reward_media_field
score_media = kontext.score_media
reference_media_score = kontext.reference_media_score
observe_objectives = kontext.observe_objectives
assert_objectives = kontext.assert_objectives
audit_kind = kontext.audit_kind
audit_steps = kontext.audit_steps
build_fixture = kontext.build_fixture
dataset_record = kontext.dataset_record
audit_samples = kontext.audit_samples


def make_config(root, result_dir, options):
    config = kontext.make_config(root, result_dir, options)
    config.parallelism_args.sequence_parallel = True
    return config


@contextmanager
def observe_training_terms(trainer, terms):
    """Observe actual TP/CP peers and SP activation lengths through training/evaluation."""
    rank = trainer.accelerator.process_index
    tp = tensor_parallel_geometry(trainer.accelerator)
    cp = context_parallel_geometry(trainer.accelerator)
    assert tp == (2, rank % 2) and cp == (2, rank // 2 % 2)
    assert trainer.config.parallelism_args.sequence_parallel is True
    modules = [
        module
        for module in trainer.model_bundle.modules()
        if isinstance(module, FluxTransformer2DModel)
    ]
    assert len(modules) == 1
    shapes = set()
    native = modules[0]
    handles = [
        native.transformer_blocks[0].norm1.register_forward_pre_hook(
            lambda _m, args, kwargs: shapes.add(("dual_image", args[0].shape[1])),
            with_kwargs=True,
        ),
        native.single_transformer_blocks[0].norm.register_forward_pre_hook(
            lambda _m, args, kwargs: shapes.add(("single_joint", args[0].shape[1])),
            with_kwargs=True,
        ),
    ]
    try:
        with kontext.observe_training_terms(trainer, terms):
            yield
    finally:
        for handle in handles:
            handle.remove()
    assert shapes == {("dual_image", 6), ("single_joint", 268)}, shapes
    print(
        json.dumps(
            {
                "flux_sp_runtime_rank": rank,
                "tp_geometry": tp,
                "cp_geometry": cp,
                "sequence_parallel": True,
                "stream_shapes": sorted(shapes),
            }
        )
    )
