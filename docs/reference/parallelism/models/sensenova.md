# SenseNova-U1 context parallelism

The `sensenova` adapter supports native dense U1.0 and U1.5 T2I and ordered
multi-reference I2I with Ulysses CP, existing TP, and generation-only SP. U1.0's
MLP flow head and U1.5's spatial convolution head retain their native operations.
The declared layout is `sensenova_prefix_spatial_flow`, version 1. The shared
[CP prerequisites](../context-parallel.md) apply, including the qualified PyTorch,
Accelerate and Diffusers release lines, FSDP2, BF16 runtime training and the native
attention backend. Query heads must be divisible by TP × CP and KV heads by TP.
Dense full attention without attention dropout is required; other decoder layouts
fail before CP installation; shared preparation may already have applied TP.

Prefix understanding and vision calls enter the canonical `transformer` route
through typed prefix requests. This keeps FSDP's parameter materialization around
all trainable component calls. Each detached prefix cache retains native text,
ordered image tokens and temporal/spatial rotary coordinates. CP reads the cache
without writing generation tokens into it. Reference preparation and the existing
native full-prefix attention remain unchanged.

Generated tokens are padded to CP × SP before CP splits the first decoder input.
The existing generation-only SP boundaries then operate inside that CP slice.
Native rotary indexes follow the same slices. Each attention invocation shards an
immutable prefix copy and masks all prefix/target padding. Repeated native GQA
heads and independent dummy queries make the rectangular prefix-plus-image
attention compatible with native Ulysses backward. Dummy queries have no loss;
no additional gradient scaling is introduced.

The ordinary U1.0 pointwise MLP head remains on SP tokens through its final projection. Plain
replicated SP parameter styles sum its token-partial weight and LoRA-factor gradients. Output
restoration gathers TP first, then uses the differentiable CP gather before native velocity
arithmetic. The outer native output boundary retains the `local_slice` gradient contract.

The pixel head receives complete decoder output for its native spatial forward. Its
differentiable CP gather sums input-gradient contributions from each CP rank because neighboring
output pixels can depend on the same input token. Extending SP through this head would require
spatial/neighbor ownership, not just a token slice. Prefix construction and timestep/noise
embedding production remain outside generation SP.

The optional deep-head variant is unqualified: the vendored `FlowMatchingHead` constructor's
keyword arguments do not match the live `SimpleMLPAdaLN` signature. Its global-row reshape and
timestep arguments also have no SP boundary. This is separate from the supported ordinary MLP
and pixel heads and is not a spatial-convolution constraint.

Model checkpointing replays only read-only generation blocks. Detached prefix
writers do not replay. When FSDP2 backend checkpointing is selected, preparation
disables all model checkpoint flags so backend wrappers remain the sole owner.
The existing one-generated-sample-per-forward contract remains. Parameter-sharded
peers must use the same raw-versus-cached prefix mode, T2I/I2I and CFG branch
schedule, and batch count. When rebuilding a prefix, every peer must supply a raw
prompt; an empty string remains valid, while `None` or an empty prompt list is
absent. The entry guard rejects mismatches before either prefix construction or
denoising work; later guards reject different conditioning/CFG call schedules. Different numbers of ordered I2I
references can still share that schedule because each branch enters one vision
and prefix call.

The selected evidence covers both heads, full and LoRA TP2/CP2 with SP off/on,
independent unsharded derivatives and three Adam steps, immutable caches,
checkpoint recomputation and an explicit lost-spatial-derivative negative.
Native DP2 × TP2 × CP2 Guard lifecycle cases use real ordered reference images,
three CFG caches, same-policy replay, evaluation/EMA, exact checkpoint loading
and fresh-process continuation under the
[ordered contract](../context-parallel.md#ordered-continuation-evidence): exact loaded
state/work identity, the first same-state update under the original bounds, and each
later actual objective/derivative, optimizer, EMA and state check. Later cross-run
numerical statistics are diagnostic. The native probes preserve
32 query heads, 8 KV heads and head width 128. CPU Gloo checks are separate from
the required GPU evidence.

The native FP32 probe compares each CFG output after dividing both tensors by
`max(1, reference.abs().max())`, using the existing `3e-4` relative and `2e-6`
absolute constants. This gives near-zero outputs a two-ppm reference-full-scale
floor; it is a new output comparison contract, not the original raw pointwise
rule. The original raw maximum-error cap remains enforced. Receipts retain each
CFG's scale, raw pointwise verdict and relative L2 error. The historical raw
pointwise failures remain failures. Parameter-gradient, CP-factor, parameter-delta
and three-step Adam checks keep their existing unscaled criteria. This numerical
comparison changes no production computation and is not a formal whole-model
floating-point error bound.

Single-sample forwards preserve the scheduler's original tensor endpoints in the
public output container. The returned mean and velocity remain ancestors of the
returned log probability, so independent Guard gradient observers see the full
objective derivative. Output values and parameter gradients remain unchanged;
multi-sample forwards retain the existing concatenation path.

At CP1, this model CP capability adds no identity fields. The
[shared scheduler compatibility note](bagel.md) applies independently to the full runtime
identity. CP greater than one was not a previously admitted SenseNova mode; its
model-owned capability is now serialized in runtime identity. Future incompatible
prefix, transport or output ownership
semantics must change that layout/version. There is no checkpoint migration or
claim of exact continuation across a different CP topology.

## Pointwise-head SP selection

`sensenova-sp-pointwise-head-reference` selects the new prepared U1.0 output boundary.
The CP1 CPU control covers ordinary/pixel heads, odd T2I/IT2I prefixes, all parameter gradients
and an optimizer-step comparison. The TP2CP2 CPU matrix checks eager/checkpoint outputs,
all parameter gradients, immutable cache and actual decoder/head token rows; it does not run
three optimizer updates. Three Adam updates/moments belong to the selected GPU native reference.
The pixel-head lost-remote-derivative negative control remains necessary. Final selected results
must bind the changed source; earlier decoder-only-SP receipts are not new head-SP passes.
