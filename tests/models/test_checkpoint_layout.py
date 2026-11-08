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

"""Prove a checkpoint reads back exactly what it wrote, role by role.

Save and load used to reconstruct the same directory rule independently, in four
places. These tests are round trips rather than file-existence checks because the
failure that costs a training run is not a missing file -- it is a fake score
restored onto the generator, which trains happily and converges to nothing.
"""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, List, Tuple

import pytest
import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file

from flow_factory.models import abc as adapter_module
from flow_factory.models.abc import (
    CHECKPOINT_MANIFEST_NAME,
    BaseAdapter,
)
from flow_factory.models.model_bundle import ModelBundle, RoutedComponentProxy
from flow_factory.models.variants import DEFAULT_BASE_VARIANT as BASE_VARIANT


class TinyModule(torch.nn.Module):
    """Hold one target projection that a checkpoint can round-trip."""

    def __init__(self) -> None:
        super().__init__()
        self.target = torch.nn.Linear(2, 2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the tiny projection."""
        return self.target(inputs)

    def save_pretrained(
        self,
        save_directory: str,
        max_shard_size: str = "10GB",
        safe_serialization: bool = True,
    ) -> None:
        """Write a minimal Diffusers-compatible full checkpoint."""
        del max_shard_size, safe_serialization
        output = Path(save_directory)
        output.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), output / "diffusion_pytorch_model.safetensors")

    @classmethod
    def from_pretrained(cls, load_directory: str) -> "TinyModule":
        """Read back what :meth:`save_pretrained` wrote.

        Raises:
            OSError: If the directory holds no weights, which is the signal the
                adapter uses to fall back to a manual state-dict load.
        """
        weights = Path(load_directory) / "diffusion_pytorch_model.safetensors"
        if not weights.is_file():
            raise OSError(f"expected weights at {weights}, received a directory without them")
        module = cls()
        module.load_state_dict(load_file(weights))
        return module


class TinyAdapter(BaseAdapter):
    """Expose the checkpoint surface over tiny components."""

    def __init__(
        self,
        accelerator: Accelerator,
        finetune_type: str,
        component_names: Tuple[str, ...] = ("transformer",),
    ) -> None:
        self.accelerator = accelerator
        self.model_args = SimpleNamespace(
            finetune_type=finetune_type,
            target_components=list(component_names),
            target_modules=["target"],
            lora_alpha=1,
            trainable_parameters_dtype=torch.float32,
        )
        self.target_module_map = {name: ["target"] for name in component_names}
        self._component_names = component_names
        self._components: dict = {}
        for name in component_names:
            component: torch.nn.Module = TinyModule()
            component.requires_grad_(False)
            if finetune_type == "lora":
                component = get_peft_model(
                    component,
                    LoraConfig(
                        r=1,
                        lora_alpha=1,
                        init_lora_weights="gaussian",
                        target_modules=["target"],
                    ),
                )
            else:
                for parameter_name, parameter in component.named_parameters():
                    parameter.requires_grad = "target" in parameter_name
            self._components[name] = component
        self.ema_wrapper = None
        self._ref_ema = None

    @property
    def trainable_component_names(self) -> List[str]:
        """Return every component that owns target modules."""
        return list(self._component_names)

    def has_component(self, name: str) -> bool:
        """Report whether a component exists."""
        return name in self._components

    def get_component(self, name: str) -> torch.nn.Module:
        """Return a component."""
        return self._components[name]

    def set_component(self, name: str, module: torch.nn.Module) -> None:
        """Replace a component."""
        self._components[name] = module

    @contextmanager
    def use_ref_parameters(self) -> Iterator[None]:
        """Provide a no-op reference context for registry activation."""
        yield

    def load_pipeline(self) -> object:
        """Satisfy the adapter abstract contract."""
        raise NotImplementedError

    def decode_latents(self, *args: object, **kwargs: object) -> object:
        """Satisfy the adapter abstract contract."""
        raise NotImplementedError

    def inference(self, *args: object, **kwargs: object) -> object:
        """Satisfy the adapter abstract contract."""
        raise NotImplementedError

    def forward(self, *args: object, **kwargs: object) -> object:
        """Satisfy the adapter abstract contract."""
        raise NotImplementedError


def _adapter(
    finetune_type: str,
    roles: Tuple[str, ...] = (BASE_VARIANT,),
    component_names: Tuple[str, ...] = ("transformer",),
) -> TinyAdapter:
    adapter = TinyAdapter(Accelerator(cpu=True), finetune_type, component_names)
    adapter.declare_component_variants(roles)
    _route_through_registry(adapter, component_names)
    return adapter


def _route_through_registry(adapter: TinyAdapter, component_names: Tuple[str, ...]) -> None:
    """Make component access follow the active variant.

    Without this every role would read the base module and a round trip would pass
    while proving nothing.
    """
    registry = adapter.component_variant_registry
    bundle = ModelBundle(registry.bundle_members())
    for name in component_names:
        adapter.set_component(name, RoutedComponentProxy(bundle, name, registry, bundle.members))


def _fill(adapter: TinyAdapter, role: str, value: float) -> None:
    for parameter in adapter.component_variant_registry.parameters(role):
        parameter.data.fill_(value)


def _values(adapter: TinyAdapter, role: str) -> List[torch.Tensor]:
    return [
        parameter.detach().clone()
        for parameter in adapter.component_variant_registry.parameters(role)
    ]


@pytest.mark.parametrize("finetune_type", ["lora", "full"])
def test_a_single_role_checkpoint_keeps_the_flat_layout_and_reads_back(
    tmp_path: Path,
    finetune_type: str,
) -> None:
    source = _adapter(finetune_type)
    _fill(source, BASE_VARIANT, 4.0)
    expected = _values(source, BASE_VARIANT)

    source.save_checkpoint(str(tmp_path), save_ema=False, include_training_roles=True)

    # The shape released checkpoints already use: weights at the root, no nesting.
    assert (tmp_path / CHECKPOINT_MANIFEST_NAME).is_file()
    assert not (tmp_path / "roles").exists()
    assert not (tmp_path / "transformer").exists()

    target = _adapter(finetune_type)
    _fill(target, BASE_VARIANT, 9.0)
    target.load_checkpoint(str(tmp_path))

    for actual, want in zip(_values(target, BASE_VARIANT), expected):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)


@pytest.mark.parametrize("finetune_type", ["lora", "full"])
def test_every_role_returns_to_its_own_variant_not_to_whichever_is_active(
    tmp_path: Path,
    finetune_type: str,
) -> None:
    """The failure this guards against is silent: fake weights landing on base."""
    source = _adapter(finetune_type, roles=(BASE_VARIANT, "fake"))
    _fill(source, BASE_VARIANT, 1.0)
    _fill(source, "fake", 2.0)
    expected_base = _values(source, BASE_VARIANT)
    expected_fake = _values(source, "fake")

    source.save_checkpoint(str(tmp_path), save_ema=False, include_training_roles=True)

    assert (tmp_path / "roles" / "fake").is_dir()
    manifest = json.loads((tmp_path / CHECKPOINT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert {(e["role"], e["path"]) for e in manifest["entries"]} == {
        (BASE_VARIANT, "."),
        ("fake", "roles/fake"),
    }

    target = _adapter(finetune_type, roles=(BASE_VARIANT, "fake"))
    _fill(target, BASE_VARIANT, 9.0)
    _fill(target, "fake", 9.0)
    target.load_checkpoint(str(tmp_path))

    for actual, want in zip(_values(target, BASE_VARIANT), expected_base):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)
    for actual, want in zip(_values(target, "fake"), expected_fake):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)
    # The two roles must stay distinguishable, or the assertions above would pass
    # even if one role had overwritten the other.
    assert not torch.equal(_values(target, BASE_VARIANT)[0], _values(target, "fake")[0])


@pytest.mark.parametrize("finetune_type", ["lora", "full"])
def test_three_roles_each_return_to_their_own_variant(
    tmp_path: Path,
    finetune_type: str,
) -> None:
    """TDM-R1's shape: a generator, the fake score, and a surrogate between them.

    Three roles is where an off-by-one in the layout stops being invisible: two roles
    can be swapped and still look plausible, three cannot.
    """
    roles = (BASE_VARIANT, "fake", "surrogate")
    source = _adapter(finetune_type, roles=roles)
    values = {role: float(index + 1) for index, role in enumerate(roles)}
    for role, value in values.items():
        _fill(source, role, value)
    expected = {role: _values(source, role) for role in roles}

    source.save_checkpoint(str(tmp_path), save_ema=False, include_training_roles=True)

    assert (tmp_path / "roles" / "fake").is_dir()
    assert (tmp_path / "roles" / "surrogate").is_dir()

    target = _adapter(finetune_type, roles=roles)
    for role in roles:
        _fill(target, role, 9.0)
    target.load_checkpoint(str(tmp_path))

    for role in roles:
        for actual, want in zip(_values(target, role), expected[role]):
            torch.testing.assert_close(actual, want, rtol=0, atol=0)
    # No two roles may have collapsed onto the same weights.
    restored = [_values(target, role)[0] for role in roles]
    assert not torch.equal(restored[0], restored[1])
    assert not torch.equal(restored[1], restored[2])
    assert not torch.equal(restored[0], restored[2])


@pytest.mark.parametrize("finetune_type", ["lora", "full"])
def test_several_components_nest_by_component_and_still_round_trip(
    tmp_path: Path,
    finetune_type: str,
) -> None:
    components = ("transformer", "transformer_2")
    source = _adapter(finetune_type, roles=(BASE_VARIANT, "fake"), component_names=components)
    _fill(source, BASE_VARIANT, 3.0)
    _fill(source, "fake", 5.0)
    expected_base = _values(source, BASE_VARIANT)
    expected_fake = _values(source, "fake")

    source.save_checkpoint(str(tmp_path), save_ema=False, include_training_roles=True)

    for component in components:
        assert (tmp_path / component).is_dir()
        assert (tmp_path / component / "roles" / "fake").is_dir()

    target = _adapter(finetune_type, roles=(BASE_VARIANT, "fake"), component_names=components)
    _fill(target, BASE_VARIANT, 9.0)
    _fill(target, "fake", 9.0)
    target.load_checkpoint(str(tmp_path))

    for actual, want in zip(_values(target, BASE_VARIANT), expected_base):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)
    for actual, want in zip(_values(target, "fake"), expected_fake):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)


@pytest.mark.parametrize("finetune_type", ["lora", "full"])
def test_a_checkpoint_written_before_manifests_still_loads(
    tmp_path: Path,
    finetune_type: str,
) -> None:
    source = _adapter(finetune_type)
    _fill(source, BASE_VARIANT, 6.0)
    expected = _values(source, BASE_VARIANT)
    source.save_checkpoint(str(tmp_path), save_ema=False)
    (tmp_path / CHECKPOINT_MANIFEST_NAME).unlink()

    target = _adapter(finetune_type)
    _fill(target, BASE_VARIANT, 9.0)
    target.load_checkpoint(str(tmp_path))

    for actual, want in zip(_values(target, BASE_VARIANT), expected):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)


def test_loading_an_export_into_a_multi_role_run_names_the_roles_it_cannot_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initializing from an export is legitimate; leaving it unsaid is not."""
    export = _adapter("lora")
    _fill(export, BASE_VARIANT, 7.0)
    export.save_checkpoint(str(tmp_path), save_ema=False)

    warnings: List[str] = []
    monkeypatch.setattr(
        adapter_module.logger, "warning", lambda message, *a, **k: warnings.append(str(message))
    )

    target = _adapter("lora", roles=(BASE_VARIANT, "fake"))
    _fill(target, "fake", 9.0)
    untouched_fake = _values(target, "fake")
    target.load_checkpoint(str(tmp_path))

    assert any("roles ['fake'] keep their initial weights" in message for message in warnings)
    for actual, want in zip(_values(target, "fake"), untouched_fake):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)


@pytest.mark.parametrize("finetune_type", ["lora", "full"])
def test_the_manifest_states_the_format_so_detection_never_has_to_guess(
    tmp_path: Path,
    finetune_type: str,
) -> None:
    adapter = _adapter(finetune_type)
    adapter.save_checkpoint(str(tmp_path), save_ema=False)

    assert adapter._detect_checkpoint_type(str(tmp_path)) == finetune_type


def test_a_manifest_from_a_future_layout_is_rejected_rather_than_misread(
    tmp_path: Path,
) -> None:
    adapter = _adapter("lora")
    adapter.save_checkpoint(str(tmp_path), save_ema=False)
    manifest_path = tmp_path / CHECKPOINT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="format_version 1.*received 99"):
        adapter.load_checkpoint(str(tmp_path))


def test_an_export_ships_the_base_role_only(tmp_path: Path) -> None:
    adapter = _adapter("lora", roles=(BASE_VARIANT, "fake"))
    adapter.save_checkpoint(str(tmp_path), save_ema=False)

    assert not (tmp_path / "roles").exists()
    manifest = json.loads((tmp_path / CHECKPOINT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert [entry["role"] for entry in manifest["entries"]] == [BASE_VARIANT]


def test_naming_a_role_that_owns_nothing_is_refused_with_the_declared_names(
    tmp_path: Path,
) -> None:
    adapter = _adapter("lora", roles=(BASE_VARIANT, "fake"))

    with pytest.raises(ValueError, match="variant 'surrogate'.*declared variants.*fake"):
        adapter.save_checkpoint(str(tmp_path), save_ema=False, variant="surrogate")


def test_a_resume_that_runs_before_the_roles_exist_is_finished_once_they_do(
    tmp_path: Path,
) -> None:
    """The adapter loads weights while it is built; the trainer names roles later.

    Placing a second role during that early load would route it to the one live
    adapter and overwrite the first, so only the primary artifact is placed then and
    the rest wait for the variants to exist.
    """
    source = _adapter("lora", roles=(BASE_VARIANT, "fake"))
    _fill(source, BASE_VARIANT, 1.0)
    _fill(source, "fake", 2.0)
    expected_base = _values(source, BASE_VARIANT)
    expected_fake = _values(source, "fake")
    source.save_checkpoint(str(tmp_path), save_ema=False, include_training_roles=True)

    target = TinyAdapter(Accelerator(cpu=True), "lora")
    target.load_checkpoint(str(tmp_path))
    target.declare_component_variants((BASE_VARIANT, "fake"))
    _route_through_registry(target, ("transformer",))
    _fill(target, "fake", 9.0)

    # The early load must not have let the fake weights land on the generator.
    for actual, want in zip(_values(target, BASE_VARIANT), expected_base):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)

    target.restore_training_roles(str(tmp_path))

    for actual, want in zip(_values(target, "fake"), expected_fake):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)
    for actual, want in zip(_values(target, BASE_VARIANT), expected_base):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)


@pytest.mark.parametrize("finetune_type", ["lora", "full"])
def test_a_late_variant_gets_the_dtype_the_base_was_already_given(finetune_type: str) -> None:
    """Variants are materialized after `_mix_precision` ran, from PEFT fp32 defaults.

    Left alone, a DMD2 fake score trains in fp32 beside a bf16 generator: twice the
    memory, a different numerical path, and every RMSNorm it feeds drops off the
    fused kernel because the weight dtype stops matching the activations.
    """
    adapter = TinyAdapter(Accelerator(cpu=True), finetune_type)
    adapter.model_args.trainable_parameters_dtype = torch.bfloat16
    for parameter in adapter.get_component("transformer").parameters():
        parameter.data = parameter.data.to(torch.bfloat16)

    adapter.declare_component_variants((BASE_VARIANT, "fake"))

    fake = list(adapter.component_variant_registry.parameters("fake"))
    assert fake
    assert all(parameter.dtype == torch.bfloat16 for parameter in fake)


@pytest.mark.parametrize("finetune_type", ["lora", "full"])
def test_checkpoint_mutated_variant_dtypes_are_realigned_before_prepare(
    finetune_type: str,
) -> None:
    adapter = TinyAdapter(Accelerator(cpu=True), finetune_type)
    adapter.model_args.trainable_parameters_dtype = torch.bfloat16
    adapter.declare_component_variants((BASE_VARIANT, "fake"))
    registry = adapter.component_variant_registry
    for variant_name in registry.variant_names:
        for parameter in registry.parameters(variant_name):
            parameter.data = parameter.data.to(torch.float32)

    adapter.align_component_variant_dtypes()

    assert all(
        parameter.dtype == torch.bfloat16
        for variant_name in registry.variant_names
        for parameter in registry.parameters(variant_name)
    )


def test_a_dtype_that_names_nothing_is_refused_with_what_it_received() -> None:
    adapter = TinyAdapter(Accelerator(cpu=True), "lora")
    adapter.model_args.trainable_parameters_dtype = "not-a-dtype"

    with pytest.raises(TypeError, match="trainable_parameters_dtype.*received str: 'not-a-dtype'"):
        adapter.declare_component_variants((BASE_VARIANT, "fake"))


@pytest.mark.parametrize("finetune_type", ["lora", "full"])
def test_saving_over_another_model_leaves_nothing_of_it_behind(
    tmp_path: Path,
    finetune_type: str,
) -> None:
    """Re-running an experiment under the same run_name writes into a used directory.

    A shard index is the dangerous leftover: it is trusted ahead of a single-file save,
    so one written by a larger model sends the loader after shards that no longer exist.
    """
    stale_index = tmp_path / "diffusion_pytorch_model.safetensors.index.json"
    stale_index.write_text(
        json.dumps({"weight_map": {"a": "diffusion_pytorch_model-00001-of-00002.safetensors"}})
    )
    (tmp_path / "diffusion_pytorch_model-00007-of-00009.safetensors").write_bytes(b"")

    adapter = _adapter(finetune_type)
    _fill(adapter, BASE_VARIANT, 2.0)
    expected = _values(adapter, BASE_VARIANT)
    adapter.save_checkpoint(str(tmp_path), save_ema=False, include_training_roles=True)

    assert not (tmp_path / "diffusion_pytorch_model-00007-of-00009.safetensors").exists()

    target = _adapter(finetune_type)
    _fill(target, BASE_VARIANT, 9.0)
    target.load_checkpoint(str(tmp_path))

    for actual, want in zip(_values(target, BASE_VARIANT), expected):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)


def test_clearing_one_entry_does_not_touch_the_roles_nested_inside_it(tmp_path: Path) -> None:
    adapter = _adapter("lora", roles=(BASE_VARIANT, "fake"))
    _fill(adapter, BASE_VARIANT, 1.0)
    _fill(adapter, "fake", 2.0)
    expected_fake = _values(adapter, "fake")

    adapter.save_checkpoint(str(tmp_path), save_ema=False, include_training_roles=True)
    adapter.save_checkpoint(str(tmp_path), save_ema=False, include_training_roles=True)

    target = _adapter("lora", roles=(BASE_VARIANT, "fake"))
    _fill(target, "fake", 9.0)
    target.load_checkpoint(str(tmp_path))

    for actual, want in zip(_values(target, "fake"), expected_fake):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)


def test_leaving_the_reference_context_hands_back_variant_trainability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PEFT keeps only the active adapter trainable, so this has to be reasserted.

    Measured on TDM-R1: both non-base roles went from 382 trainable parameters to 0
    across the reference query, and since autograd reads requires_grad when backward
    executes, the roles lost every gradient and the optimizer stepped on nothing --
    with no error anywhere.

    Asserted through the call rather than the resulting flags because this tiny harness
    does not reproduce PEFT's freezing; a state assertion would pass with or without the
    fix and guard nothing.
    """
    adapter = _adapter("lora", roles=(BASE_VARIANT, "fake"))
    registry = adapter.component_variant_registry
    calls: List[str] = []
    monkeypatch.setattr(registry, "restore_trainable_parameters", lambda: calls.append("restored"))

    # The adapter under test overrides this hook to a no-op, so exercise the real one.
    with BaseAdapter.use_ref_parameters(adapter):
        pass

    assert calls == ["restored"]


def test_restoring_trainability_is_a_no_op_without_variants() -> None:
    bare = TinyAdapter(Accelerator(cpu=True), "lora")

    with BaseAdapter.use_ref_parameters(bare):
        pass


def test_restoring_roles_before_declaring_them_is_refused(tmp_path: Path) -> None:
    adapter = _adapter("lora")
    adapter.save_checkpoint(str(tmp_path), save_ema=False)
    bare = TinyAdapter(Accelerator(cpu=True), "lora")

    with pytest.raises(RuntimeError, match="declared component variants, received none"):
        bare.restore_training_roles(str(tmp_path))


def test_each_role_directory_is_directly_loadable_by_peft(tmp_path: Path) -> None:
    """A user pointing PEFT at a role folder should not need to know our layout."""
    adapter = _adapter("lora", roles=(BASE_VARIANT, "fake"))
    adapter.save_checkpoint(str(tmp_path), save_ema=False, include_training_roles=True)

    for directory in (tmp_path, tmp_path / "roles" / "fake"):
        assert (directory / "adapter_config.json").is_file()
        assert (directory / "adapter_model.safetensors").is_file()
        assert load_file(directory / "adapter_model.safetensors")


@pytest.mark.parametrize("finetune_type", ["lora", "full"])
@pytest.mark.parametrize("mixed_precision", ["no", "bf16"])
@pytest.mark.parametrize("fsdp2", [False, True])
def test_late_fsdp2_variants_preserve_the_original_parameter_dtype(
    finetune_type: str, mixed_precision: str, fsdp2: bool, monkeypatch
) -> None:
    """Align new and restored roles before TP can capture their DTensor dtype."""
    from flow_factory.models.runtime import PseudoPipelineRuntime

    adapter = TinyAdapter(Accelerator(cpu=True), finetune_type)
    adapter.component_runtime = PseudoPipelineRuntime(
        SimpleNamespace(), {"transformer": adapter.get_component("transformer")}
    )
    adapter.model_args.trainable_parameters_dtype = torch.bfloat16
    adapter.accelerator = SimpleNamespace(mixed_precision=mixed_precision, is_main_process=False)
    monkeypatch.setattr(adapter, "_is_fsdp2", lambda: fsdp2)
    expected = torch.float32 if fsdp2 and mixed_precision != "no" else torch.bfloat16
    adapter._mix_precision()
    late_component = TinyModule()
    adapter._apply_component_precision_policy("transformer", late_component)
    assert all(parameter.dtype == expected for parameter in late_component.parameters())
    adapter.declare_component_variants((BASE_VARIANT, "fake"))
    registry = adapter.component_variant_registry
    assert all(parameter.dtype == expected for parameter in registry.parameters("fake"))
    for variant in registry.variant_names:
        for parameter in registry.parameters(variant):
            parameter.data = parameter.data.to(torch.bfloat16)
    adapter.align_component_variant_dtypes()
    assert all(
        parameter.dtype == expected
        for variant in registry.variant_names
        for parameter in registry.parameters(variant)
    )
