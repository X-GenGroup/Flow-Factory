# Acceleration Scope

Follow the repository root `AGENTS.md`. Use `/ff-new-accelerator` for a plugin and `/ff-develop`
for shared acceleration infrastructure.

- Lossless `stage="both"` plugins that reuse existing adapter APIs may use R2.
- Rollout-scoped, lossy, cached, compiled, checkpointed, offloaded, or backend-stateful behavior
  requires at least R3 and cleanup plus numerical-parity evidence.
- Changes to `abc.py`, `validator.py`, prepared component routing, or the shared execution lifecycle
  are R4 candidates.
- Keep stage, safety, config slot, registry key, ordering, and affected trainer paradigms aligned.
