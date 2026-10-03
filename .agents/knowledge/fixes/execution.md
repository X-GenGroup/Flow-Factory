# Execution Fix Patterns

**Read when**: Debugging or changing execution behavior.

### Custom Transformers models must implement their declared checkpointing seam
- **Date**: 2026-08-30
- **Symptom**: Every Bagel trainer failed during initialization when full gradient checkpointing
  called `Qwen2ForCausalLM.gradient_checkpointing_enable()` and Transformers reported that the
  architecture was incompatible.
- **Root Cause**: Bagel's custom Qwen2-NaViT model inherited
  `supports_gradient_checkpointing = True` but did not expose a `gradient_checkpointing` state or
  invoke the installed checkpoint function in its active decoder loop.
- **Fix**: The custom Qwen2 model now owns the standard checkpointing flag and routes each pure
  decoder layer through Transformers' installed non-reentrant checkpoint function while training.
  Cache-updating and TaylorSeer paths remain direct to avoid replaying mutations. A backward
  regression proves the layer is recomputed rather than merely accepting the API call.
- **Lesson**: A custom `PreTrainedModel` must pair its capability declaration with both the state
  seam expected by Transformers and an execution-time checkpoint boundary in the forward path.
- **Related Constraint**: #7

## Cross-refs

- UP: [Fix Pattern Router](README.md), [Hard Constraints](../constraints.md)
- WORKFLOW: [`ff-debug`](../../skills/ff-debug/SKILL.md)
