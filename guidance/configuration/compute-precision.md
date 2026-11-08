# Operation compute precision

Parameter storage and operation compute precision have separate owners. Native loader rules such
as Diffusers' `_keep_in_fp32_modules` protect weights; they do not force an attention operation
to execute in FP32 inside BF16 autocast. Flow-Factory therefore lets an adapter declare a named,
versioned `ComputePrecisionPolicy` containing module-path and operation rules.

The default `model.compute_precision_policy: null` preserves native computation. The first
qualified policy is Wan T2V's explicit cross-attention policy, also declared for the
bounded [Wan2.1 I2V CP layout](../../docs/reference/parallelism/models/wan-i2v.md):

```yaml
model:
  model_type: wan2_t2v
  compute_precision_policy: cross_attention_fp32_v1
```

It selects `blocks.*.attn2` in each transformer and computes native PyTorch SDPA with FP32 query,
key and value, with autocast disabled for that operation. Floating attention masks use FP32;
boolean masks retain their semantics. The output returns to the incoming query dtype. Projection
weights, optimizer master weights, self-attention, normalization and output projections retain
their existing precision. Autograd differentiates through the casts and FP32 operation normally.
FP32 here specifies operands, output and autocast. The native backend still owns its kernel's
internal multiplication and reduction arithmetic; see the bounded numerical evidence in the
[CP guide](../../docs/reference/parallelism/context-parallel.md#numerical-evidence).

An adapter declares the supported rule set on its class, for example:

```python
compute_precision_policies = (
    ComputePrecisionPolicy(
        name="cross_attention_fp32_v1",
        rules=(ComputePrecisionRule(module_pattern="blocks.*.attn2"),),
    ),
)
```

The types live in `flow_factory.models.compute_precision`. New models need focused native
forward/backward and complete-training evidence for their declared paths before advertising a
policy. Rules describe transformer-relative module patterns, not parameter-name matches.

The same persistent module forward is used in training, inference, checkpoint recomputation and
parameter snapshot contexts. Policies install after loading and LoRA injection, before variant
cloning and parallel preparation. They preserve the original inspectable signature used by TP
region entries. Bound original methods remain on their modules so full-weight clones bind to
their own parameters. Parameter identities, module paths and state-dict keys do not change.
Runtime identity records the selected rules and resolved module paths; exact resume rejects
policy drift. Model-only weights must be used with the same explicit compute policy when
reproducing outputs.

Only native SDPA rules are currently implemented. Other adapters default to no declared policy;
unsupported names, unmatched or overlapping module patterns, and incompatible accelerators fail
with concrete errors. Selected Diffusers attention modules explicitly use the native backend.
An alternate processor that no longer calls native SDPA fails at the selected forward boundary.
Compile, feature caching and non-native attention are not qualified with this policy.

The bounded [Wan CP and TP+CP capabilities](../../docs/reference/parallelism/context-parallel.md) require this policy for full and
LoRA tuning. Native BF16 cross-attention failed the unchanged update bound on retained LoRA
fixtures. Precision changes are explicit and have a cost: an L20Y operator-only measurement
at B=1, H=20, K=512, D=128 and Q=8192/16384 measured approximately 4.5–6.2 times the SDPA
forward/backward latency and 2.2 times its additional peak allocation, including casts. These
are attention-operator measurements, not full-model speed or memory multipliers.

## Design references

- [Diffusers layerwise casting](https://huggingface.co/docs/diffusers/api/models/overview)
  demonstrates model-owned exclusions for precision-sensitive layers; storage and compute
  dtypes remain distinct.
- [Wan's native model](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py)
  explicitly keeps selected modulation arithmetic in FP32. It does not establish FP32
  cross-attention as the upstream default.
- [PyTorch function modes](https://docs.pytorch.org/docs/stable/notes/extending.html#extending-all-torch-api-with-modes)
  provide the scoped interception used here, without replacing global functions, tensor types,
  native attention processors or the model's attention mathematics.
