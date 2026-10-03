# Model Scope

Follow the repository root `AGENTS.md`. Use `/ff-new-model` for a new adapter and `/ff-develop`
for shared model infrastructure.

- Start with [adapter conventions](../../../.agents/knowledge/topics/adapter_conventions.md) and
  [parity testing](../../../.agents/knowledge/topics/parity_testing.md).
- Load component-runtime, structured-trajectory, dataset, or distributed detail only when claimed.
- Family-local classic online adapters may use R1. Custom runtime, structured state, offline codec,
  packed batches, caches, or backend capabilities raise the profile.
- Changes to `abc.py`, `runtime/`, `trajectory_bridge/`, or shared loading are R4 candidates.
- Preserve train-inference parity and fail unsupported capabilities before heavyweight loading.
