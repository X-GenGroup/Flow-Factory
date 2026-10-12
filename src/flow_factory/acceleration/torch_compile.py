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

# src/flow_factory/acceleration/torch_compile.py
"""Apply ``torch.compile`` persistently to the rollout/training transformer(s)."""

import functools
from typing import TYPE_CHECKING, Any, Dict

import torch

from ..utils.logger_utils import setup_logger
from .abc import BaseAccelerator

if TYPE_CHECKING:
    from ..models.abc import BaseAdapter

logger = setup_logger(__name__)


class CompileAccelerator(BaseAccelerator):
    """Apply ``torch.compile`` to every transformer the adapter exposes.

    Stage-``both`` but ``safety='lossy'``. The compiled module backs both rollout
    ``inference()`` and the training ``forward()`` (they share ``adapter.transformer``),
    so it is applied *symmetrically* — but, unlike an exact or symmetric-approximate
    transform, ``torch.compile`` is NOT numerically identical across the two stages:
    Inductor compiles a separate graph for grad vs no-grad mode, and even with the
    grad-forced rollout fix below an intermittent ~1e-5 on-policy ratio residual remains
    on a minority of samples (see :meth:`_wrap_forward_grad_consistent`). That is why it
    is marked ``lossy`` rather than ``lossless`` (which here means *bit-exact across
    stages*, not merely "applied to the shared module").

    It stays well within ``clip_range`` — numerically on-policy — and is the main
    compute speedup on real hardware, so it remains a ``stage='both'`` accelerator
    allowed on coupled algorithms. The validator does not reject it, but it WARNS on a
    coupled trainer that the on-policy PPO ratio will be ~1, not bit-exact. Use eager or
    the ``attention_backend`` accelerator if a strictly bit-exact ratio is required.

    Both modes compile **in place** (``nn.Module.compile`` /
    ``compile_repeated_blocks``), which preserves parameter identity and leaves
    ``state_dict`` keys unchanged (no ``_orig_mod.`` prefix). Consequences:

    * **Checkpointing** stays correct — save/load operate on the same keys.
    * **EMA / reference / named-parameter swaps** stay correct — they mutate
      ``param.data`` in place (``copy_``), which the compiled graph reads.
    * **LoRA reference forwards** (``use_ref_parameters`` -> PEFT
      ``disable_adapter``) toggle control flow, so Dynamo recompiles the
      adapter-disabled path once; this is correct but adds a one-time recompile.

    Applied after ``post_init`` (see ``BaseTrainer._apply_shared_acceleration``),
    so it wraps the final, fully-loaded weights.

    Args:
        mode: ``"auto"`` (default) selects ``"regional"`` per transformer when
            the unwrapped base model declares non-empty ``_repeated_blocks``;
            otherwise it selects ``"full"``. ``"regional"`` forces diffusers'
            ``compile_repeated_blocks`` and fails when the declaration is absent.
            ``"full"`` compiles the whole module in place.
        compile_kwargs: Extra kwargs forwarded to the underlying compile call
            (e.g. ``{"mode": "max-autotune", "dynamic": true}``).
    """

    # `lossy` not because it is asymmetric (it is stage='both', applied to the shared
    # module) but because it is NOT bit-exact across rollout vs training — see the class
    # docstring and `_wrap_forward_grad_consistent`. The validator warns (does not
    # reject) when this runs on a coupled trainer.
    safety = "lossy"
    stage = "both"

    def setup(self, adapter: "BaseAdapter") -> None:
        mode = self.params.get("mode", "auto")
        compile_kwargs: Dict[str, Any] = self.params.get("compile_kwargs", {})

        if mode not in ("auto", "regional", "full"):
            raise ValueError(
                f"CompileAccelerator: unknown mode={mode!r}; "
                "expected 'auto', 'regional', or 'full'."
            )

        transformer_names = adapter.transformer_names
        if not transformer_names:
            raise ValueError(
                "CompileAccelerator: adapter exposes no transformer components to compile."
            )

        for name in transformer_names:
            module = adapter.get_component(name)
            # The routed call chain is proxy(...) -> bundle(name, ...) ->
            # members[name](...) -> PeftModel -> LoraModel -> the BASE transformer's
            # __call__. So compilation and the grad-consistency wrap must target the
            # unwrapped *base* transformer:
            #   * `adapter._unwrap` peels the RoutedComponentProxy + DDP/FSDP/DeepSpeed
            #     wrapper, but NOT PEFT — it returns the PeftModel.
            #   * compiling the PeftModel is wrong: Dynamo specializes the LoRA wrapper
            #     on grad mode in a way the outer grad-force cannot unify (velocity
            #     drifts ~2.0 between rollout/train), AND the PeftModel has no
            #     `_repeated_blocks` so regional compile is unavailable under LoRA.
            #   * the base transformer keeps its LoRA submodules inside the compiled
            #     graph (they still train), exposes `_repeated_blocks`, and the
            #     grad-force wrap keeps rollout/training on the same grad-mode compiled
            #     path (ratio ≈ 1 within `clip_range`, but not bit-exact).
            inner: Any = self._peel_peft(adapter._unwrap(module))
            effective_mode = self._resolve_compile_mode(inner, mode, name)
            if effective_mode == "regional":
                inner.compile_repeated_blocks(**compile_kwargs)
            else:
                # nn.Module.compile compiles the module's forward in place, so the
                # routed bundle call hits the compiled path on subsequent forwards.
                inner.compile(**compile_kwargs)
            # Force every forward of the compiled transformer to run under
            # torch.enable_grad() (overriding the @torch.no_grad() on
            # adapter.inference()). This keeps rollout (Stage 3) and the training
            # forward (Stage 6) on the same grad-mode compiled path — Inductor emits
            # numerically different kernels for the grad vs no-grad graph. The outer
            # rollout remains no_grad, and collectors detach direct model outputs.
            self._wrap_forward_grad_consistent(adapter, inner)
            if adapter.accelerator.is_main_process:
                logger.info(
                    "CompileAccelerator: compiled '%s' (mode=%s, effective=%s).",
                    name,
                    mode,
                    effective_mode,
                )

    @staticmethod
    def _resolve_compile_mode(module, mode: str, component_name: str) -> str:
        """Resolve the configured compile mode for one unwrapped transformer."""
        has_repeated_blocks = bool(getattr(module, "_repeated_blocks", None))
        if mode == "auto":
            return "regional" if has_repeated_blocks else "full"
        if mode == "regional" and not has_repeated_blocks:
            raise ValueError(
                f"CompileAccelerator: component '{component_name}' "
                f"({type(module).__name__}) does not declare `_repeated_blocks`, so regional "
                "compilation is unavailable. Use `mode: auto` or `mode: full` for whole-module "
                "compilation."
            )
        return mode

    @staticmethod
    def _peel_peft(module):
        """Return the base transformer under a PEFT/LoRA wrapper (else ``module``).

        ``adapter._unwrap`` peels the routing proxy + DDP/FSDP/DeepSpeed wrapper but
        leaves a ``PeftModel`` in place. Compiling the ``PeftModel`` is incorrect:
        Dynamo specializes the LoRA wrapper on grad mode in a way the grad-force wrap
        cannot unify, and the wrapper hides the base model's ``_repeated_blocks``. The
        base transformer keeps its LoRA submodules (they still train) while being the
        actual compiled call target, so we compile/wrap it instead.
        """
        get_base = getattr(module, "get_base_model", None)
        if callable(get_base):
            return get_base()
        return module

    @staticmethod
    def _wrap_forward_grad_consistent(adapter: "BaseAdapter", module) -> None:
        """Force the compiled transformer to run grad-consistently across stages.

        Wraps the module's call entry point(s) so every forward runs under
        ``torch.enable_grad()`` — overriding the ``@torch.no_grad()`` on
        ``adapter.inference()``. This pins rollout (Stage 3) and the training forward
        (Stage 6) to the same grad-mode compiled path: Inductor compiles a separate,
        numerically non-identical graph for grad vs no-grad mode (Dynamo guards on
        ``grad_mode``), so a no-grad rollout would diverge from the grad training
        forward and undermine coupled on-policy consistency.

        Crucially the output is **NOT detached here**: detaching inside the wrapper
        lets Inductor see the result is unused-for-grad and pick a different
        (inference-optimized) kernel, re-introducing the divergence. The surrounding
        rollout stays under ``torch.no_grad()``, so scheduler and CFG operations do not
        extend the graph; collectors detach any direct model output before storage.
        Every transformer call still uses the same grad-mode compiled path (near-exact
        velocity → on-policy ratio ≈ 1 within ``clip_range``).

        Known residual (NOT strictly bit-exact): forcing grad removes the *dominant*
        divergence (the grad-vs-no-grad graph split), but it does not make the rollout
        and training forwards a literally identical Inductor kernel invocation. They are
        different call sites: the stored-then-reloaded latents may differ in
        stride/contiguity, and the surrounding autograd graphs differ (rollout detaches
        stored outputs and discards the graph; training keeps it and backwards). For some
        inputs Inductor's shape/layout-specialized, autotuned kernels then take a
        different reduction order, and bf16's non-associative accumulation turns that
        into an *intermittent* ~1e-5 on-policy ratio drift on a minority of
        samples/ranks (value-dependent — some runs/shardings show exactly 0). This is
        well within ``clip_range`` (1e-4), i.e. numerically on-policy, but not
        bit-exact. Eager and the attention-backend accelerator stay exactly 0
        (an attention kernel's forward is grad-mode-independent). Measured on 16×H20
        (2-node ZeRO-2/FSDP2): SD3.5 full compilation, Qwen-Image regional/full
        compilation, and CFG/no-CFG.

        Two entry points must be covered because the two compile modes dispatch
        differently:
          * ``mode='regional'`` (``compile_repeated_blocks``): the top ``forward``
            stays eager and calls the compiled blocks — wrapping ``forward`` suffices.
          * ``mode='full'`` (``nn.Module.compile``): ``module(...)`` routes through
            ``module._compiled_call_impl``, bypassing a reassigned ``forward`` — so
            that attribute must be wrapped too.

        Idempotent per attribute (``_ff_grad_consistent`` marker).
        """

        def _wrap(fn):
            if fn is None or getattr(fn, "_ff_grad_consistent", False):
                return fn

            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                with torch.enable_grad():
                    return fn(*args, **kwargs)

            setattr(wrapped, "_ff_grad_consistent", True)
            return wrapped

        # Regional + eager fallback: top-level forward.
        module.forward = _wrap(module.forward)
        # Full mode: nn.Module.compile() dispatches through _compiled_call_impl.
        if getattr(module, "_compiled_call_impl", None) is not None:
            module._compiled_call_impl = _wrap(module._compiled_call_impl)
