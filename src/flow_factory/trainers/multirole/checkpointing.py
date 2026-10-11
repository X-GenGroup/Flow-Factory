"""Checkpoint metadata and custom state for multi-role trainers."""

import json
import os
from collections.abc import Mapping
from typing import Any

import torch

from ..execution import TrainingProgress

MULTIROLE_METADATA_FILENAME = "flow_factory_multirole_metadata.json"
MULTIROLE_METADATA_VERSION = 1
MULTIROLE_RUNTIME_CHILD_NAME = "multirole"
MULTIROLE_STATE_KEYS = {
    "version",
    "metadata",
    "coordinator",
    "trainer_step",
    "variant_snapshots",
}


class _MultiRoleCheckpointState:
    """Delegate multi-role checkpoint state and metadata gates to one trainer."""

    def __init__(self, trainer: Any) -> None:
        self._trainer = trainer

    def state_dict(self) -> dict[str, Any]:
        """Return multi-role counters and defensive compatibility metadata."""
        return self._trainer._multirole_state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore multi-role counters after Accelerate restores prepared state."""
        self._trainer._load_multirole_state_dict(state)

    def validate_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate counters and snapshots without mutating live trainer state."""
        self._trainer._validate_multirole_state_dict(state)

    def validate_runtime_progress(
        self,
        progress: TrainingProgress,
        state: Mapping[str, Any],
    ) -> None:
        """Require the runtime's optimizer counter to equal the primary role."""
        self.validate_state_dict(state)
        trainer_step = state["trainer_step"]
        if progress.optimizer_step != trainer_step:
            raise ValueError(
                "registered multi-role counter mismatch with trainer runtime progress: "
                f"expected optimizer_step={progress.optimizer_step}, received "
                f"trainer_step={trainer_step}"
            )

    def prepare_save(self, output_dir: str) -> None:
        """Validate a closed boundary and write metadata before Accelerate saves."""
        try:
            custom_state = self.state_dict()
        except RuntimeError as error:
            raise RuntimeError(
                "cannot checkpoint invalid multi-role training state before "
                f"model/optimizer save; validation reported: {error}"
            ) from error
        accelerator = self._trainer.accelerator
        save_on_each_node = accelerator.project_configuration.save_on_each_node
        should_write_metadata = accelerator.is_main_process or (
            save_on_each_node and accelerator.is_local_main_process
        )
        if should_write_metadata:
            os.makedirs(output_dir, exist_ok=True)
            metadata_path = os.path.join(output_dir, MULTIROLE_METADATA_FILENAME)
            with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                json.dump(custom_state["metadata"], metadata_file, indent=2, sort_keys=True)
                metadata_file.write("\n")

    def validate_load(self, input_dir: str) -> None:
        """Validate metadata before Accelerate can mutate prepared state."""
        metadata_path = os.path.join(input_dir, MULTIROLE_METADATA_FILENAME)
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                "multi-role metadata compatibility gate expected file "
                f"{metadata_path!r}, received missing file"
            )
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        self._trainer._validate_multirole_metadata(metadata)


class MultiRoleCheckpointingMixin:
    """Own the compatibility contract for multi-role prepared-state checkpoints."""

    def _multirole_metadata(self) -> dict[str, Any]:
        """Return deterministic metadata used to validate resume compatibility."""
        role_metadata = self.adapter.component_variant_registry.training_state_dict()
        update_plan = self._role_update_plan()
        return {
            "version": MULTIROLE_METADATA_VERSION,
            "roles": role_metadata["variants"],
            "optimizer_group_roles": [
                group.get("role_name") for group in self.optimizer.param_groups
            ],
            "update_plan": [
                {
                    "role_name": phase.role_name,
                    "repeats": phase.repeats,
                }
                for phase in update_plan.phases
            ],
        }

    def _validate_multirole_metadata(self, state: Mapping[str, Any]) -> None:
        """Validate all metadata that must match before prepared-state mutation."""
        if not isinstance(state, Mapping):
            raise TypeError(
                "expected multi-role metadata as a mapping, "
                f"received {type(state).__name__}: {state!r}"
            )
        expected_keys = {"version", "roles", "optimizer_group_roles", "update_plan"}
        received_keys = set(state)
        if received_keys != expected_keys:
            raise ValueError(
                "multi-role metadata keys mismatch: expected "
                f"{tuple(sorted(expected_keys))!r}, "
                f"received {tuple(sorted(received_keys))!r}"
            )
        received_version = state.get("version")
        if (
            not isinstance(received_version, int)
            or isinstance(received_version, bool)
            or received_version != MULTIROLE_METADATA_VERSION
        ):
            raise ValueError(
                "multi-role metadata version mismatch: expected "
                f"{MULTIROLE_METADATA_VERSION}, received {received_version!r}"
            )

        self.adapter.component_variant_registry.load_training_state_dict(
            {
                "version": received_version,
                "variants": state.get("roles"),
            }
        )
        expected = self._multirole_metadata()
        for field_name in ("optimizer_group_roles", "update_plan"):
            expected_value = expected[field_name]
            received_value = state.get(field_name)
            if received_value != expected_value:
                raise ValueError(
                    f"multi-role metadata {field_name} mismatch: expected "
                    f"{expected_value!r}, received {received_value!r}"
                )

    def _primary_role(self) -> str:
        """Return the role whose optimizer step defines this run's public step."""
        return self._required_trainable_roles()[0]

    def _multirole_state_dict(self) -> dict[str, Any]:
        """Return registered custom state without duplicating prepared state."""
        coordinator_state = self.role_optimization.state_dict()
        primary_role = self._primary_role()
        primary_step = coordinator_state["role_steps"][primary_role]
        if self.step != primary_step:
            raise RuntimeError(
                "multi-role checkpoint counter mismatch: expected trainer step to equal "
                f"{primary_role!r} role step {primary_step}, received trainer_step={self.step}"
            )
        return {
            "version": 1,
            "metadata": self._multirole_metadata(),
            "coordinator": coordinator_state,
            "trainer_step": self.step,
            "variant_snapshots": self._multirole_snapshot_state_dict(),
        }

    def _multirole_snapshot_state_dict(self) -> dict[str, Any]:
        """Return snapshot references for the immediately synchronous serializer.

        ``TrainerRuntimeState`` performs the one required CPU clone while extracting
        tensors into safetensors. Cloning every full-model snapshot here first would
        transiently double accelerator memory before that extraction begins.
        """
        registry = self.adapter.component_variant_registry
        snapshots = getattr(registry, "_snapshots", None)
        if not isinstance(snapshots, Mapping):
            raise TypeError(
                "component variant registry must expose snapshot declarations as a "
                f"mapping, received {type(snapshots).__name__}"
            )
        return {
            "version": 1,
            "snapshots": {
                snapshot_name: {
                    "variant_name": snapshot["variant_name"],
                    "parameters": dict(snapshot["parameters"]),
                }
                for snapshot_name, snapshot in snapshots.items()
            },
            "update_counts": {
                snapshot_name: snapshot["update_count"]
                for snapshot_name, snapshot in snapshots.items()
            },
        }

    def _load_multirole_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore custom multi-role counters after complete validation."""
        self._validate_multirole_state_dict(state)
        trainer_step = state["trainer_step"]
        runtime_state = getattr(self, "runtime_state", None)
        if runtime_state is not None and getattr(runtime_state, "load_received", False):
            if self.step != trainer_step:
                raise ValueError(
                    "registered multi-role counter mismatch with trainer runtime progress: "
                    f"expected optimizer_step={self.step}, received trainer_step={trainer_step}"
                )
        else:
            self.step = trainer_step
        self.role_optimization.load_state_dict(state["coordinator"])
        self.adapter.component_variant_registry.load_snapshot_state_dict(state["variant_snapshots"])
        self.adapter.component_variant_registry.activate(
            self.adapter.component_variant_registry.base_variant
        )

    def _validate_multirole_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate custom multi-role state without changing counters or snapshots."""
        if not isinstance(state, Mapping):
            raise TypeError(
                "expected registered multi-role state as a mapping, "
                f"received {type(state).__name__}: {state!r}"
            )
        received_keys = set(state)
        if received_keys != MULTIROLE_STATE_KEYS:
            raise ValueError(
                "registered multi-role state keys mismatch: expected "
                f"{tuple(sorted(MULTIROLE_STATE_KEYS))!r}, "
                f"received {tuple(sorted(received_keys))!r}"
            )
        state_version = state["version"]
        if (
            not isinstance(state_version, int)
            or isinstance(state_version, bool)
            or state_version != 1
        ):
            raise ValueError(
                "registered multi-role state version mismatch: expected 1, "
                f"received {state_version!r}"
            )
        self._validate_multirole_metadata(state["metadata"])
        trainer_step = state["trainer_step"]
        if not isinstance(trainer_step, int) or isinstance(trainer_step, bool) or trainer_step < 0:
            raise ValueError(
                "expected non-negative int trainer_step in registered multi-role state, "
                f"received {trainer_step!r}"
            )
        coordinator_state = state["coordinator"]
        if not isinstance(coordinator_state, Mapping):
            raise TypeError(
                "expected coordinator in registered multi-role state as a mapping, "
                f"received {type(coordinator_state).__name__}: {coordinator_state!r}"
            )
        role_steps = coordinator_state.get("role_steps")
        primary_role = self._primary_role()
        if not isinstance(role_steps, Mapping) or role_steps.get(primary_role) != trainer_step:
            received_primary_step = (
                role_steps.get(primary_role) if isinstance(role_steps, Mapping) else role_steps
            )
            raise ValueError(
                "registered multi-role counter mismatch: expected trainer_step "
                f"{trainer_step} to equal {primary_role!r} role step, "
                f"received {primary_role!r} step {received_primary_step!r}"
            )
        self._validate_multirole_coordinator_state(coordinator_state)
        self._validate_multirole_snapshot_state(state["variant_snapshots"])

    def _validate_multirole_coordinator_state(self, state: Mapping[str, Any]) -> None:
        """Validate the coordinator payload without applying received role steps."""
        expected = self.role_optimization.state_dict()
        expected_keys = set(expected)
        received_keys = set(state)
        if received_keys != expected_keys:
            raise ValueError(
                "multi-role coordinator state keys mismatch: expected "
                f"{tuple(sorted(expected_keys))!r}, received "
                f"{tuple(sorted(received_keys))!r}"
            )
        if state["version"] != expected["version"]:
            raise ValueError(
                "multi-role coordinator state version mismatch: expected "
                f"{expected['version']!r}, received {state['version']!r}"
            )
        if state["active_phase"] is not None:
            raise ValueError(
                "multi-role coordinator state expected active_phase=None, received "
                f"{state['active_phase']!r}"
            )
        if state["optimizer_group_roles"] != expected["optimizer_group_roles"]:
            raise ValueError(
                "multi-role coordinator optimizer_group_roles mismatch: expected "
                f"{expected['optimizer_group_roles']!r}, received "
                f"{state['optimizer_group_roles']!r}"
            )
        role_steps = state["role_steps"]
        expected_role_names = tuple(expected["role_steps"])
        if not isinstance(role_steps, Mapping) or tuple(role_steps) != expected_role_names:
            received_role_names = (
                tuple(role_steps) if isinstance(role_steps, Mapping) else role_steps
            )
            raise ValueError(
                "multi-role coordinator role_steps mismatch: expected roles "
                f"{expected_role_names!r}, received {received_role_names!r}"
            )
        for role_name, role_step in role_steps.items():
            if not isinstance(role_step, int) or isinstance(role_step, bool) or role_step < 0:
                raise ValueError(
                    "multi-role coordinator expected non-negative int step for "
                    f"{role_name!r}, received {role_step!r}"
                )

    def _validate_multirole_snapshot_state(self, state: Mapping[str, Any]) -> None:
        """Validate variant snapshot tensors without copying into live snapshots."""
        if not isinstance(state, Mapping):
            raise TypeError(
                "expected parameter EMA state as a mapping, "
                f"received {type(state).__name__}: {state!r}"
            )
        registry = self.adapter.component_variant_registry
        # Validation must not clone full-model snapshots merely to inspect their
        # schema. The registry owns this immutable declaration mapping; received
        # values are copied only later by its public load_snapshot_state_dict().
        expected_snapshots = getattr(registry, "_snapshots", None)
        if not isinstance(expected_snapshots, Mapping):
            raise TypeError(
                "component variant registry must expose snapshot declarations as a "
                f"mapping, received {type(expected_snapshots).__name__}"
            )
        expected_keys = {"version", "snapshots", "update_counts"}
        if set(state) != expected_keys or state.get("version") != 1:
            raise ValueError(
                "multi-role snapshot state keys/version mismatch: expected keys "
                f"{tuple(sorted(expected_keys))!r} and version 1, "
                f"received keys={tuple(sorted(state))!r}, version={state.get('version')!r}"
            )
        snapshots = state["snapshots"]
        update_counts = state["update_counts"]
        if not isinstance(snapshots, Mapping) or tuple(snapshots) != tuple(expected_snapshots):
            received_names = tuple(snapshots) if isinstance(snapshots, Mapping) else snapshots
            raise ValueError(
                "multi-role snapshot names mismatch: expected "
                f"{tuple(expected_snapshots)!r}, received {received_names!r}"
            )
        if not isinstance(update_counts, Mapping) or tuple(update_counts) != tuple(
            expected_snapshots
        ):
            received_names = (
                tuple(update_counts) if isinstance(update_counts, Mapping) else update_counts
            )
            raise ValueError(
                "multi-role snapshot update-count names mismatch: expected "
                f"{tuple(expected_snapshots)!r}, received {received_names!r}"
            )
        for snapshot_name, expected_snapshot in expected_snapshots.items():
            received_snapshot = snapshots[snapshot_name]
            if not isinstance(received_snapshot, Mapping):
                raise TypeError(
                    f"multi-role snapshot {snapshot_name!r} must be a mapping, "
                    f"received {type(received_snapshot).__name__}: {received_snapshot!r}"
                )
            if set(received_snapshot) != {"variant_name", "parameters"}:
                raise ValueError(
                    f"multi-role snapshot {snapshot_name!r} keys mismatch: expected "
                    "('parameters', 'variant_name'), received "
                    f"{tuple(sorted(received_snapshot))!r}"
                )
            if received_snapshot["variant_name"] != expected_snapshot["variant_name"]:
                raise ValueError(
                    f"multi-role snapshot {snapshot_name!r} variant mismatch: expected "
                    f"{expected_snapshot['variant_name']!r}, received "
                    f"{received_snapshot['variant_name']!r}"
                )
            received_parameters = received_snapshot["parameters"]
            expected_parameters = expected_snapshot["parameters"]
            if not isinstance(received_parameters, Mapping) or tuple(received_parameters) != tuple(
                expected_parameters
            ):
                received_names = (
                    tuple(received_parameters)
                    if isinstance(received_parameters, Mapping)
                    else received_parameters
                )
                raise ValueError(
                    f"multi-role snapshot {snapshot_name!r} parameter names mismatch: "
                    f"expected {tuple(expected_parameters)!r}, received {received_names!r}"
                )
            for parameter_name, expected_tensor in expected_parameters.items():
                received_tensor = received_parameters[parameter_name]
                if type(received_tensor) is not torch.Tensor:
                    raise TypeError(
                        f"multi-role snapshot {snapshot_name!r}/{parameter_name!r} must "
                        f"be a plain torch.Tensor, received {type(received_tensor).__name__}"
                    )
                if (
                    received_tensor.shape != expected_tensor.shape
                    or received_tensor.dtype != expected_tensor.dtype
                ):
                    raise ValueError(
                        f"multi-role snapshot {snapshot_name!r}/{parameter_name!r} tensor "
                        f"metadata mismatch: expected shape={tuple(expected_tensor.shape)}, "
                        f"dtype={expected_tensor.dtype}, received "
                        f"shape={tuple(received_tensor.shape)}, dtype={received_tensor.dtype}"
                    )
            update_count = update_counts[snapshot_name]
            if (
                not isinstance(update_count, int)
                or isinstance(update_count, bool)
                or update_count < 0
            ):
                raise ValueError(
                    f"multi-role snapshot {snapshot_name!r} update count must be a "
                    f"non-negative int, received {update_count!r}"
                )

    def _register_multirole_checkpointing(self) -> None:
        """Register Accelerate metadata gates and custom state for multi-role runs."""
        if len(self._required_trainable_roles()) <= 1:
            return
        if getattr(self, "_multirole_checkpoint_registered", False):
            raise RuntimeError(
                "cannot register multi-role checkpointing twice for trainer "
                f"{type(self).__name__}"
            )

        def save_metadata_hook(
            models: list[torch.nn.Module],
            weights: list[dict[str, torch.Tensor]],
            output_dir: str,
        ) -> None:
            del models, weights
            checkpoint_state.prepare_save(output_dir)

        def load_metadata_hook(models: list[torch.nn.Module], input_dir: str) -> None:
            del models
            checkpoint_state.validate_load(input_dir)

        checkpoint_state = _MultiRoleCheckpointState(self)
        self._multirole_checkpoint_state = checkpoint_state
        self.adapter._multirole_checkpoint_state = checkpoint_state
        self.accelerator.register_save_state_pre_hook(save_metadata_hook)
        self.accelerator.register_load_state_pre_hook(load_metadata_hook)
        # Real trainers serialize this object through TrainerRuntimeState as strict
        # JSON+safetensors. Hosts without that runtime (lightweight test hosts and
        # external users) register it directly as Accelerate custom checkpoint state.
        if getattr(self, "runtime_state", None) is None:
            self.accelerator.register_for_checkpointing(checkpoint_state)
        self._multirole_checkpoint_registered = True


__all__ = [
    "MULTIROLE_METADATA_FILENAME",
    "MULTIROLE_RUNTIME_CHILD_NAME",
    "MultiRoleCheckpointingMixin",
]
