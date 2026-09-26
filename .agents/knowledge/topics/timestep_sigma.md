# Timestep & Sigma Convention

**Read when**: Touching `TimeSampler`, `adapter.forward(t=...)`, `scheduler.step(timestep=...)`, `flow_match_sigma()`, or `timestep_range` config fields.

---

Throughout the codebase, two related but distinct scales are used for time:

| Name | Variable | Scale | Meaning |
|------|----------|-------|---------|
| **Timestep** | `t`, `timestep` | `[0, 1000]` | Scheduler-scale time. All public interfaces (`TimeSampler` outputs, `adapter.forward(t=...)`, `scheduler.step(timestep=...)`) use this scale. |
| **Sigma** | `σ`, `sigma` | `[0, 1]` | Flow-matching noise level. Used for latent interpolation `x_t = (1-σ) x_0 + σ ε` and loss weighting. Obtained via `flow_match_sigma(t) = t / 1000`. |

## Rules

1. `TimeSampler` always returns `t` in `[0, 1000]`. Trainers pass it directly to `adapter.forward(t=...)` without scaling.
2. When interpolating latents or computing noise-level-dependent weights, convert explicitly: `sigma = flow_match_sigma(t)`.
3. Each model adapter internally converts `t` to whatever its underlying transformer expects (e.g., Flux divides by 1000, SD3.5 passes as-is). This conversion is encapsulated inside the adapter's `forward()` method.
4. `timestep_range=(frac_lo, frac_hi)` is a fraction along the denoising axis from 1000 (noisy) toward 0 (clean), mapped via `t = 1000 * (1 - frac)`. So `(0, 0.99)` corresponds to `t ∈ [10, 1000]`.
5. Validate a stored `(timestep, sigma)` pair with `validate_flow_match_coordinates()`. It compares in sigma space and permits one native ULP after independent materialization.
6. Treat stored replay coordinates as authoritative at discrete boundaries. Call `adapter.build_training_component_times()` only for newly sampled continuous coordinates or legacy replay data that did not store sigma.
7. Preserve every stored component coordinate's native dtype and device when widening scalar replay values to a batch. Do not cast secondary timesteps or sigmas to the primary timestep representation.
8. A coordinate-validation refactor must A/B the complete generated timestep/sigma schedules against its pre-change baseline at the raw-bit level. Compare each schedule independently; do not require an analytic cross-component remap to reproduce another scheduler's independently rounded grid.

## Gotchas

1. **Don't divide `t` by 1000 before passing to `adapter.forward()`** — the adapter handles internal conversion.
2. **`timestep_range` uses denoising-axis fractions, not raw timesteps** — `(0, 0.5)` means the noisier half `t ∈ [500, 1000]`, not the cleaner half.
3. **`flow_match_sigma()` is the only sanctioned conversion** — do not use `t / 1000` directly; use the function for traceability.
4. **Do not validate redundant coordinates with a fixed scaled tolerance** — use `validate_flow_match_coordinates()`, which delegates native-spacing math to the shared precision utilities. For a discrete multi-component endpoint, reuse each scheduler's authoritative stored coordinate instead of reconstructing it through inverse/forward shifts.
5. **A primary open-interval point may round onto another component's endpoint** — nonlinear component-time transforms can compress or amplify float32 spacing. TDM must redraw that primary point until every mapped component has a representable open-interior timestep and sigma; do not snap or independently nudge component coordinates.
6. **Bit-exact schedule parity does not mean preserving a buggy replay reconstruction** — rollout schedule generation and same-input continuous mapping must remain bit-identical, while replay must consume each stored component endpoint instead of matching the old value reconstructed from the primary component.

## Fix records

### Independently rounded flow-matching coordinates
- **Date**: 2026-08-27
- **Symptom**: MiniMax H3 rejected valid TDM forward noising because `990.4219970703125` and `0.9904220700263977 * 1000` differed by roughly `6e-5`; independently shifted audio endpoints could also miss exact trajectory topology checks.
- **Root Cause**: H3 compared redundant float32 coordinates after multiplying sigma by 1000 with a fixed `1e-5` tolerance. TDM then reconstructed independently rounded audio endpoints through the nonlinear video-shift inverse/audio-shift forward path, whose error varies with schedule position and step count.
- **Fix**: Added shared native-ULP primitives and a canonical flow-match coordinate validator; reused stored per-component replay endpoints in TDM and DMD2; removed H3-specific grid snapping; validated H3/LTX2 schedule producers; and made TDM redraw only continuous samples that round onto a component endpoint.
- **Verification**: On CPU with PyTorch 2.11.0, diffusers 0.40.0.dev0, and NumPy 2.4.4, independent Python processes loaded main and the fix worktree and compared 164 selected tensor snapshots by dtype, shape, and raw bytes across H3 video/audio schedules and mapping, LTX2, FlowMatchEuler, UniPC, and explicit ODE/SDE replay; the tested generated schedules and observable outputs were bit-identical. Permanent tests freeze independent main-formula oracles for the touched coordinate conversions. Replay endpoint consumption intentionally changed from analytic reconstruction to the bit-exact stored trajectory values and is tested against that authority instead of the buggy baseline.
- **Lesson**: One-ULP tolerance is appropriate only for redundant representations of the same coordinate. Discrete multi-component endpoints must retain producer authority, while continuous mappings need bounded rejection when no open-interior representation is produced.
- **Related Constraint**: N/A

### Conditional TDM score-query sampling
- **Date**: 2026-09-25
- **Symptom**: TDM could not express source-coordinate uniform or conditional logit-normal queries, while a reverse cap of 0.98 made the production H3 shift-12 six-step schedule empty at its first boundary.
- **Root Cause**: Interval topology, probability distribution, and rollout shift provenance were coupled through trainer branches and sample `extra_kwargs`, with defaults that silently changed the released objective.
- **Fix**: An immutable query policy separates `trajectory`/`reverse` topology from `actual_uniform`/`conditional_logit_normal`/`source_uniform` distributions. Compatibility defaults retain trajectory-uniform sampling; recipes opt into new policies and use an open `max_sigma=1.0`. A shared microbatch query context maps the reverse cap once. Source shifts live in typed rank-local provenance registered before reward submission and never enter samples or distributed payloads. Exact resume hashes only active policy fields.
- **Lesson**: Query topology, coordinate distribution, and runtime provenance have distinct owners. Preserve the released numerical path as a named policy, represent bounds explicitly, and keep optimization-only metadata outside shared samples.
- **Related Constraint**: #7, #18a.

## Cross-refs

- UP: [`constraints.md` #7](../constraints.md#7-coupled-vs-decoupled-paradigm), [Architecture Timestep and Sigma Convention](../architecture.md#timestep--sigma-convention)
- PEER: [Train/Inference Consistency](train_inference_consistency.md), [Adapter Conventions](adapter_conventions.md)
