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

"""Independent objective arithmetic for the retained single-latent TinyWan case.

Bounded native Wan test reference: B=1, no active masks, noise-ward flow, GAS=2.
The model VJP itself is outside these scalar FP32/BF16 error budgets.
"""

import torch


def gamma(n, dtype=torch.float32):
    u = torch.finfo(dtype).eps / 2
    assert n * u < 1
    return n * u / (1 - n * u)


def state(event, key):
    value = event["objective"][key]
    assert set(value["components"]) == {"latent"} and value["active_masks"] is None
    tensor = value["components"]["latent"]
    assert tensor.shape[0] == 1 and torch.isfinite(tensor).all()
    return tensor.double()


def active_call(event):
    name = event.get(
        "active_root",
        "members.fake__transformer" if event["role"] == "fake" else "members.transformer",
    )
    stage = "loss" if event["role"] == "fake" else "generator_clean_replay"
    rows = [r for r in event["modules"] if r["name"] == name and r["stage"] == stage]
    assert len(rows) == 1
    return rows[0]


def replay_sigma(event):
    times = event["replay_step"]["times"]
    sigma = (
        times["sigma"]["latent"]
        if times["sigma"] is not None
        else times["timestep"]["latent"].float() / 1000.0
    )
    assert torch.isfinite(sigma).all() and ((sigma >= 0) & (sigma <= 1)).all()
    return sigma.double().reshape(-1, 1, 1, 1, 1)


def direction(event):
    gen, real, fake = [state(event, k) for k in ("gen", "real", "fake")]
    n = (gen - real).abs().flatten(1).mean(1).clamp_min(1e-6).reshape(-1, 1, 1, 1, 1)
    return (fake - real) / n, n


def objective_reference(event, cp, gas=2):
    """FP64 upstream and a data-dependent scalar-operation roundoff envelope.

    A serial nonnegative sum bounds any normalizer reduction tree. Sixteen
    additional operations cover derivative constants, mean divisor, loss
    scaling, GAS division and conversions. This is not a fitted tolerance.
    """
    u = torch.finfo(torch.float32).eps / 2
    tiny = torch.finfo(torch.float32).tiny
    if event["role"] == "fake":
        pred, target = state(event, "predicted_velocity"), state(event, "target_velocity")
        residual = pred - target
        residual_error = u * (pred.abs() + target.abs()) + tiny
        scale = 2 * cp / (gas * residual.numel())
        ideal = residual * scale
        error = (
            residual_error * scale + gamma(16) * (residual.abs() + residual_error) * scale + tiny
        )
        normalizer = None
    else:
        gen, real, fake = [state(event, k) for k in ("gen", "real", "fake")]
        ideal_direction, normalizer = direction(event)
        sub_error = u * (gen.abs() + real.abs()) + tiny
        mean_error = (
            sub_error.mean() + gamma(gen.numel() + 2) * ((gen - real).abs() + sub_error).mean()
        )
        normalizer_error = mean_error + u * 1e-6 + tiny
        lower = (normalizer - normalizer_error).clamp_min(float(torch.tensor(1e-6)))
        numerator = fake - real
        numerator_error = u * (fake.abs() + real.abs()) + tiny
        direction_error = numerator_error / lower + numerator.abs() * normalizer_error / (
            normalizer * lower
        )
        direction_error += gamma(2) * (numerator.abs() + numerator_error) / lower + tiny
        # target=fl(gen-direction).detach(); residual=fl(gen-target).
        target_error = (
            direction_error + u * (gen.abs() + ideal_direction.abs() + direction_error) + tiny
        )
        residual_error = (
            target_error + u * (gen.abs() + (gen - ideal_direction).abs() + target_error) + tiny
        )
        scale = cp / (gas * gen.numel())
        ideal = ideal_direction * scale
        error = (
            residual_error * scale
            + gamma(16) * (ideal_direction.abs() + residual_error) * scale
            + tiny
        )
    observed = event["objective"]["upstream"]["latent"]
    storage_u = torch.finfo(observed.dtype).eps / 2
    error += storage_u * (ideal.abs() + error) + torch.finfo(observed.dtype).tiny
    observed_error = (observed.double() - ideal).abs()
    result = {
        "latent_ideal": ideal,
        "latent_budget": error,
        "latent_observed": observed.double(),
        "latent_pass": bool((observed_error <= error).all()),
        "latent_max_ratio": float((observed_error / error.clamp_min(tiny)).max()),
    }
    if event["role"] == "generator":
        sigma = replay_sigma(event)
        root_ideal = -sigma * ideal
        root_error = sigma * error + gamma(2) * sigma * (ideal.abs() + error) + tiny
    else:
        root_ideal, root_error = ideal, error
    call = active_call(event)
    if "dy" in call:
        actual = call["dy"].double()
        ru = torch.finfo(call["dy"].dtype).eps / 2
        root_error += ru * (root_ideal.abs() + root_error) + torch.finfo(call["dy"].dtype).tiny
        result.update(
            root_observed=actual,
            root_pass=bool(((actual - root_ideal).abs() <= root_error).all()),
            root_max_ratio=float(((actual - root_ideal).abs() / root_error.clamp_min(tiny)).max()),
        )
    result.update(root_ideal=root_ideal, root_budget=root_error, normalizer=normalizer)
    return result


def direction_difference(left, right):
    """Forward-only componentwise envelope; no parameter derivatives are used."""
    da, na = direction(left)
    dr, nr = direction(right)
    ua = state(left, "fake") - state(left, "real")
    ur = state(right, "fake") - state(right, "real")
    du, dn = (ua - ur).abs(), (na - nr).abs()
    bound = torch.minimum(du / na + ur.abs() * dn / (na * nr), du / nr + ua.abs() * dn / (na * nr))
    absolute_path = (
        da.abs()
        + dr.abs()
        + du / na
        + du / nr
        + ur.abs() * dn / (na * nr)
        + ua.abs() * dn / (na * nr)
    )
    bound += gamma(64 + da.numel(), torch.float64) * absolute_path + torch.finfo(torch.float64).tiny
    normalizer_bound = (
        (state(left, "gen") - state(right, "gen") - state(left, "real") + state(right, "real"))
        .abs()
        .mean()
    )
    normalizer_bound = normalizer_bound + gamma(32 + da.numel(), torch.float64) * (
        na + nr + normalizer_bound
    )
    return {
        "actual": da - dr,
        "bound": bound,
        "normalizer_difference": dn,
        "normalizer_bound": normalizer_bound,
        "pass": bool(((da - dr).abs() <= bound).all() and (dn <= normalizer_bound).all()),
    }


def objective_negative_controls(event, cp):
    ref = objective_reference(event, cp)
    assert ref["latent_pass"] and ref.get("root_pass", True)
    actual = ref["latent_observed"]
    controls = {}
    for label, value in (
        ("zero", torch.zeros_like(actual)),
        ("half", actual / 2),
        ("double", actual * 2),
        ("sign", -actual),
    ):
        controls[label] = not bool(
            ((value - ref["latent_ideal"]).abs() <= ref["latent_budget"]).all()
        )
    index = int(ref["latent_ideal"].abs().flatten().argmax())
    wrong = actual.clone().flatten()
    wrong[index] += 4 * (
        ref["latent_budget"].flatten()[index]
        + (actual - ref["latent_ideal"]).abs().flatten()[index]
    )
    controls["localized"] = not bool(
        ((wrong.reshape_as(actual) - ref["latent_ideal"]).abs() <= ref["latent_budget"]).all()
    )
    assert all(controls.values()), controls
    return controls
