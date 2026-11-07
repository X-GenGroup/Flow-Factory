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

"""One-shot preparation of every materialized variant under one backend root."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from ...acceleration.attention_backend import AttentionBackendAccelerator
from ...hparams.parallelism_args import ParallelismArguments
from ...loading.coordinator import ModelLoadCoordinator
from ...models.context_parallel import (
    PreparedContextParallel,
    context_parallel_mesh,
)
from ...models.model_bundle import ModelBundle, RoutedComponentProxy
from ...models.tensor_parallel import (
    apply_tensor_parallel,
    install_replicated_input_check,
    peel_peft_wrapper,
    resolve_parallel_plan,
    sequence_parallel_enabled,
)
from ...utils.dist import parallel_topology
from ...utils.logger_utils import setup_logger
from ..multirole.backend import tensor_parallel_size

logger = setup_logger(__name__)


class ModelPreparation:
    """Own destructive transforms and rebinding for one trainer's model root.

    The adapter materializes variants/LoRA and establishes precision first. The
    trainer then requests parallel transforms, constructs the optimizer over
    their final parameters, prepares one backend root, validates role ownership,
    and installs routes. Snapshots/resume and ordinary shared acceleration stay
    after this sequence. Declared CP consumes its attention owner before hooks.

    A failed mutation leaves its phase in progress: retrying could transform a
    partially prepared module twice, so the caller must construct a fresh model.
    """

    def __init__(
        self, load_coordinator: ModelLoadCoordinator, parallelism: ParallelismArguments
    ) -> None:
        self.load_coordinator = load_coordinator
        self.adapter = load_coordinator.adapter
        self.accelerator = load_coordinator.accelerator
        self.parallelism = parallelism
        self._phase = "new"
        self._canonical_names: tuple[str, ...] = ()
        self._model_bundle: nn.Module | None = None
        self._inner_bundle: ModelBundle | None = None
        self._context_parallel: PreparedContextParallel | None = None

    def _require_phase(self, expected: str, operation: str) -> None:
        if self._phase != expected:
            raise RuntimeError(
                f"cannot {operation}: model preparation phase is {self._phase!r}, "
                f"expected {expected!r}; do not prepare or transform a model twice"
            )

    def parallelize(self, attention_backend: AttentionBackendAccelerator | None = None) -> None:
        """Transform all materialized variant routes before optimizer construction."""
        self._require_phase("new", "apply parallel transforms")
        registry = getattr(self.adapter, "component_variant_registry", None)
        if registry is None or not registry.is_frozen or not registry.variant_names:
            raise RuntimeError(
                "parallel transforms require materialized, frozen variant declarations"
            )
        self._phase = "parallelizing"
        self.load_coordinator.synchronize_initial_state(registry.bundle_members().values())
        self._apply_tensor_parallel()
        self._apply_context_parallel(attention_backend)
        self._phase = "parallelized"

    def _apply_context_parallel(
        self, attention_backend: AttentionBackendAccelerator | None
    ) -> None:
        """Select attention, then install Diffusers hooks once before FSDP and optimizer setup."""
        degree = self.parallelism.effective_context_parallel_size
        if degree == 1:
            return
        if parallel_topology(self.accelerator).cp_size != degree:
            raise ValueError("configured context parallelism differs from the preparation mesh")
        from diffusers.models._modeling_parallel import ContextParallelConfig

        capability = self.adapter.context_parallel_capability
        if capability is None:
            raise ValueError("context parallel preparation requires an adapter capability")
        if type(
            attention_backend
        ) is not AttentionBackendAccelerator or attention_backend.params != {"backend": "native"}:
            raise ValueError(
                "context parallel preparation requires the native attention backend owner"
            )
        registry = self.adapter.component_variant_registry
        routes = registry.bundle_members()
        transformer_routes = {
            registry.resolve_route(variant, "transformer") for variant in registry.variant_names
        }
        if set(routes) != transformer_routes:
            raise ValueError(
                "context parallel preparation requires only declared transformer variant routes; "
                f"expected {sorted(transformer_routes)}, received {tuple(routes)}"
            )
        modules = {}
        seen: set[int] = set()
        for route, wrapped in routes.items():
            module = peel_peft_wrapper(wrapped)
            if id(module) not in seen:
                seen.add(id(module))
                modules[route] = module
        tp_size = tensor_parallel_size(self.accelerator)
        if tp_size > 1 and not capability.tensor_parallel:
            raise ValueError("adapter context-parallel capability does not support TP composition")
        for module in modules.values():
            self.adapter.validate_context_parallel_module(module, degree, tp_size)
            if getattr(module, "_parallel_config", None) is not None:
                raise ValueError(
                    "context parallel hooks must be installed exactly once by ModelPreparation"
                )
        mesh = context_parallel_mesh(self.accelerator)
        attention_backend.setup_components(modules, log=self.accelerator.is_main_process)
        boundaries = []
        compose_sp = tp_size > 1 and sequence_parallel_enabled(
            self.adapter, self.parallelism.sequence_parallel
        )
        for module in modules.values():
            module.enable_parallelism(
                config=ContextParallelConfig(ring_degree=1, ulysses_degree=degree, mesh=mesh)
            )
            if compose_sp:
                boundaries.extend(self.adapter.compose_context_parallel_sequence_boundaries(module))
        self._context_parallel = PreparedContextParallel(
            capability, degree, tuple(modules.values()), tuple(boundaries)
        )

    def prepare(self, optimizer: torch.optim.Optimizer, *eval_dataloaders: Any) -> tuple[Any, ...]:
        """Prepare one bundle and rebind registry identities through the existing backend."""
        self._require_phase("parallelized", "prepare the backend root")
        registry = self.adapter.component_variant_registry
        expected = {
            id(parameter)
            for variant in registry.variant_names
            for parameter in registry.parameters(variant)
        }
        received = [
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        ]
        if len(received) != len(set(received)) or set(received) != expected:
            raise ValueError(
                "optimizer must own each materialized variant parameter exactly once before prepare; "
                f"missing={len(expected - set(received))}, "
                f"unexpected={len(set(received) - expected)}, "
                f"duplicates={len(received) - len(set(received))}"
            )
        self._canonical_names = tuple(self.adapter.target_module_map)
        model_bundle = ModelBundle(registry.bundle_members())
        if self._context_parallel is not None:
            model_bundle.context_parallel_contract = self._context_parallel
        self._phase = "preparing"
        prepared = self.load_coordinator.prepare(model_bundle, optimizer, *eval_dataloaders)
        self._model_bundle = prepared[0]
        self._inner_bundle = self.accelerator.unwrap_model(self._model_bundle)
        registry.rebind_parameters(self._inner_bundle.members)
        if getattr(self.accelerator, "is_fsdp2", False):
            registry.bind_parameter_write_preparation(self._prepare_parameter_write)
        self._phase = "prepared"
        return tuple(prepared)

    def _prepare_parameter_write(self) -> None:
        """Discard FSDP2 gathered views before snapshot installation/restoration.

        The FSDP2 root normally retains unsharded values between forwards. A write
        to its authoritative DTensor shards must invalidate those views, including
        when a policy graph is pending backward across a reference query.
        """
        from torch.distributed.fsdp import FSDPModule

        self.require_ready()
        groups = [
            (name, module, module._get_fsdp_state()._fsdp_param_group)
            for name, module in self._inner_bundle.named_modules()
            if isinstance(module, FSDPModule)
        ]
        # Snapshot scopes surround forwards; a mutation from inside a running
        # FSDP forward/backward is a different, unsupported ownership boundary.
        for name, _, group in groups:
            if group is not None and group._training_state.name != "IDLE":
                raise RuntimeError(
                    f"parameter write requires idle FSDP2 groups; module={name!r}, "
                    f"state={group._training_state}"
                )
        for name, module, group in groups:
            # A prefetched gather can still contain the old values while the
            # module is sharded. reshard() alone does not consume it; the next
            # forward would install that stale result. Inspect only this FSDP2
            # backend field, then settle through public APIs.
            if group is not None and group._all_gather_result is not None:
                module.unshard()
            module.reshard()
            if group is not None and not group.is_sharded:
                raise RuntimeError(f"FSDP2 parameter views remain unsharded for module={name!r}")

    def install_routes(self) -> None:
        """Install canonical proxies after the trainer validates prepared role ownership."""
        self._require_phase("prepared", "install prepared component routes")
        self._phase = "binding"
        registry = self.adapter.component_variant_registry
        for name in self._canonical_names:
            self.adapter.set_component(
                name,
                RoutedComponentProxy(
                    self._model_bundle, name, registry, self._inner_bundle.members
                ),
            )
        self._phase = "ready"

    def require_ready(self) -> None:
        """Require final parameter identities and routes before snapshots or resume."""
        self._require_phase("ready", "initialize model runtime state")

    def _shard_component(
        self,
        component: torch.nn.Module,
        tp_mesh: Any,
        tp_size: int,
        plan: Mapping[str, Any],
        label: str,
    ) -> None:
        """Apply the resolved plan to one bundled module and install its optional input check."""
        applied = apply_tensor_parallel(component, tp_mesh, plan)
        logger.info(
            "Tensor-parallel plan applied to %s over %d ranks (%d entries)",
            label,
            tp_size,
            len(applied),
        )
        if self.parallelism.verify_replicated_inputs:
            install_replicated_input_check(component, tp_mesh, label)

    def _apply_tensor_parallel(self) -> None:
        """Shard the trainable denoiser across the tensor-parallel mesh.

        Runs here and nowhere else. By this point the weights are loaded, LoRA is
        injected, component variants are materialized and their dtypes aligned,
        so every parameter that will exist already does. It still precedes
        optimizer construction, the ``ModelBundle``, and ``prepare``:

        * the optimizer must be built over the final (DTensor) parameters, or
          Accelerate's ``data_ptr`` remap fails with "a parameter in the
          optimizer couldn't be switched to its sharded version";
        * ``fully_shard`` must see DTensor parameters to compose the 2-D spec
          rather than sharding them a second time;
        * ``rebind_parameters`` runs afterwards and compares global shapes,
          which tensor parallelism leaves unchanged.

        A run without a ``parallelism`` block, or with ``tp_size`` of 1, does
        nothing here.
        """
        tp_size = tensor_parallel_size(self.accelerator)
        if tp_size <= 1:
            return
        device_mesh = self.accelerator.torch_device_mesh
        if device_mesh is None:
            raise RuntimeError(
                f"tensor parallelism (tp_size={tp_size}) was configured but the "
                "accelerator built no device mesh to shard across."
            )
        parallelism = self.parallelism
        plan = resolve_parallel_plan(self.adapter, sequence_parallel=parallelism.sequence_parallel)
        # Every *route* of a trainable component, not just the active one. A
        # multi-role algorithm materializes one variant per role, and the two
        # kinds behave differently: LoRA variants are extra PEFT adapters on the
        # same module, so there is one object to shard and `derive_lora_plan`
        # reaches every adapter's matrices; full variants are
        # `copy.deepcopy(base_module)` under their own route, so sharding only
        # the active one leaves the replicas holding plain tensors. `prepare`
        # then gets a root mixing meshes and the first `clip_grad_norm_` raises
        # "DTensor does not support cross-mesh operation" -- measured, after the
        # weights are already loaded.
        #
        # De-duplicated by identity because a shared route is the *designed*
        # LoRA case: `ComponentVariantRegistry.declare` permits two variants to
        # share one route precisely when both are named LoRA adapters, so
        # `resolve_route` returns the same module for both and this loop reaches
        # it twice. Applying the plan twice to one module is a hard error --
        # torch's `distribute_module` refuses a second call -- so this is not a
        # tidiness measure. `localize_attention_head_counts` is separately
        # idempotent, which covers any *other* path that re-walks the bundle.
        registry = self.adapter.component_variant_registry
        routes = registry.bundle_members()
        seen: set[int] = set()
        sharded_architectures: set[type] = set()
        for component_name in self.adapter.trainable_component_names:
            for variant_name in registry.variant_names:
                # `resolve_route` is the authority on where a variant's copy of a
                # component lives; matching route names by prefix would guess.
                module = routes.get(registry.resolve_route(variant_name, component_name))
                if module is None:
                    continue
                # The plan names paths in the architecture's own tree, so a PEFT
                # wrapper has to be peeled first: its children sit under
                # `base_model.model.` and every pattern would match nothing.
                component = peel_peft_wrapper(module)
                if id(component) in seen:
                    continue
                seen.add(id(component))
                sharded_architectures.add(type(component))
                self._shard_component(
                    component, device_mesh["tp"], tp_size, plan, f"{variant_name}/{component_name}"
                )
        if not seen:
            raise RuntimeError(
                "tensor parallelism found no trainable component to shard; expected a route "
                f"for {sorted(self.adapter.trainable_component_names)} among {sorted(routes)}"
            )
        # A frozen member still goes through `fully_shard`; left plain, FSDP2
        # shards it over `dp_shard` alone and every rank holds `tp_size` times
        # its share -- Wan2.2's `transformer_2` when only `transformer` trains.
        # The plan names paths in the trained architecture, so only members of
        # that architecture can take it.
        for route, module in routes.items():
            component = peel_peft_wrapper(module)
            if id(component) in seen:
                continue
            seen.add(id(component))
            if type(component) not in sharded_architectures:
                logger.warning(
                    "Bundled member %r is a %s, not an architecture the tensor-parallel plan "
                    "was written for, so it stays unsharded and FSDP2 shards it over the "
                    "data-parallel dimension only.",
                    route,
                    type(component).__name__,
                )
                continue
            self._shard_component(component, device_mesh["tp"], tp_size, plan, f"frozen/{route}")

        # Sharding replaces every planned parameter object, so the ownership
        # records taken during variant materialization now point at tensors the
        # modules no longer hold. The optimizer is built from those records next,
        # and Accelerate maps optimizer parameters onto their sharded versions by
        # storage address during `prepare` -- stale identities surface there as
        # "a parameter in the optimizer couldn't be switched to its sharded
        # version", far from the cause. Rebinding is the same seam `prepare` uses
        # afterwards, and tensor parallelism leaves global shapes unchanged, so
        # its shape check still holds.
        registry = self.adapter.component_variant_registry
        registry.rebind_parameters(registry.bundle_members())
