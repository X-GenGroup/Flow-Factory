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

"""Record native gradients, clipping and Adam state without changing the update."""

from contextlib import contextmanager

import torch
from test_context_parallel_state import _cpu_tree
from test_context_parallel_training import _compare
from test_model_preparation import full_value


@contextmanager
def observe_optimizer_steps(trainer, records, *, cp_size, is_saving, common_state=None):
    """Observe the existing optimizer path, restoring every wrapper on exit.

    ``records`` owns ``updates`` and ``backward_losses`` lists. ``is_saving``
    identifies the exact-state writer's internal optimizer callback. The optional
    common-state input is the existing explicit diagnostic intervention; ordinary
    executions leave it unset and retain their uninterrupted parameter trajectory.
    """
    parameters = dict(trainer.model_bundle.named_parameters())
    names = {id(parameter): name for name, parameter in parameters.items()}
    role_names = {
        role: {names[id(parameter)] for parameter in spec.parameters}
        for role, spec in trainer.optimization_roles.items()
    }
    backward = trainer.accelerator.backward
    clip = trainer.accelerator.clip_grad_norm_
    optimizer_step = trainer.optimizer.step
    pending = {}

    def observed_backward(loss, **kwargs):
        records["backward_losses"].append(loss.detach().cpu() / cp_size)
        return backward(loss, **kwargs)

    def observed_clip(*args, **kwargs):
        role = trainer.role_optimization._active_role_name or trainer._primary_role()
        gradients = {
            name: full_value(p.grad).cpu().clone()
            for name, p in parameters.items()
            if p.grad is not None
        }
        assert set(gradients) == role_names[role]
        pending.update(
            role=role,
            gradients=gradients,
            before=_cpu_tree(parameters),
            loss=torch.stack(records["backward_losses"]),
        )
        records["backward_losses"].clear()
        return clip(*args, **kwargs)

    def observed_step(*args, **kwargs):
        if is_saving():
            return optimizer_step(*args, **kwargs)
        optimizer = trainer.optimizer.optimizer
        role = pending["role"]
        spec = trainer.optimization_roles[role]
        assert len(spec.optimizer_group_ids) == 1
        group = optimizer.param_groups[spec.optimizer_group_ids[0]]
        settings = {key: group[key] for key in ("lr", "betas", "eps", "weight_decay")}
        settings["max_grad_norm"] = spec.config.max_grad_norm
        clipped = {
            name: full_value(parameters[name].grad).cpu().clone() for name in pending["gradients"]
        }
        old_states = {
            name: _cpu_tree(optimizer.state.get(p, {}))
            for name, p in parameters.items()
            if name not in role_names[role]
        }
        returned = optimizer_step(*args, **kwargs)
        after = _cpu_tree(parameters)
        for name, p in parameters.items():
            if name not in role_names[role]:
                _compare(after[name], pending["before"][name], exact=True)
                _compare(_cpu_tree(optimizer.state.get(p, {})), old_states[name], exact=True)
        records["updates"].append(
            {
                "role": role,
                "before": pending["before"],
                "parameters": after,
                "gradients": pending["gradients"],
                "clipped_gradients": clipped,
                "optimizer": settings,
                "loss": pending["loss"],
                "deltas": {
                    name: after[name] - pending["before"][name] for name in sorted(role_names[role])
                },
                "moments": {
                    name: _cpu_tree(optimizer.state[parameters[name]])
                    for name in sorted(role_names[role])
                },
            }
        )
        if common_state is not None:
            from torch.distributed.tensor import DTensor, distribute_tensor

            index = len(records["updates"]) - 1
            reference_update = common_state["updates"][index]
            assert reference_update["role"] == role
            expected_before = (
                common_state["initial_parameters"]
                if index == 0
                else common_state["updates"][index - 1]["parameters"]
            )
            _compare(pending["before"], expected_before, exact=True)
            # Diagnostic intervention: discard this leg's updated weights,
            # so the next forward starts at the reference trajectory's state.
            # It proves local gradients, not uninterrupted optimization.
            trainer.adapter.component_variant_registry.prepare_parameter_write()
            with torch.no_grad():
                for name, parameter in parameters.items():
                    value = reference_update["parameters"][name].to(parameter.device)
                    if isinstance(parameter, DTensor):
                        value = distribute_tensor(
                            value,
                            parameter.device_mesh,
                            parameter.placements,
                            src_data_rank=None,
                        )
                    parameter.copy_(value)
            _compare(_cpu_tree(parameters), reference_update["parameters"], exact=True)
        return returned

    trainer.accelerator.backward = observed_backward
    trainer.accelerator.clip_grad_norm_ = observed_clip
    trainer.optimizer.step = observed_step
    try:
        yield
    finally:
        trainer.accelerator.backward = backward
        trainer.accelerator.clip_grad_norm_ = clip
        trainer.optimizer.step = optimizer_step
