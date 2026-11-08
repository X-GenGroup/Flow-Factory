# Bagel tensor, sequence and context parallelism

Bagel owns a `bagel_packed_prefix_mot` layout on its native `Qwen2ForCausalLM`
component. Context parallelism uses Diffusers Ulysses and the native attention
backend. Its optional dependency qualification is exact: Diffusers
`>=0.42.1` (the release includes the previously required Ulysses fix), PyTorch 2.10 and
Accelerate 1.15, with the real FlashAttention kernel used by native prefix
prefill. The declaration does not change other adapters' dependency contracts.

The existing TP plan keeps attention replicated and shards both MoT feed-forward
branches. Explicit `sequence_parallel: true` adds attention TP and shards the
packed generation residual. It requires TP2 or TP4 dividing both native query
and KV heads. Prefix construction remains whole over tokens; both attention
branches share the same local GQA head ownership. Ulysses additionally requires
the remaining query heads to be divisible by CP. For native 28 query heads and
4 KV heads, TP2/SP+CP2 is valid; TP4/SP+CP2 is rejected because its seven local
query heads cannot be divided across CP2. SP stays off by default.

## Parameter coverage of the two plans

Coverage counts unique canonical base-parameter **elements**, including frozen or
currently inactive parameters, in the native `Qwen2ForCausalLM` component. The
denominator includes both MoT branches, vocabulary embeddings, the untied LM head
and every norm; it is not the whole Bagel adapter. It excludes the separate ViT,
VAE, connectors, timestep/latent projections, derived LoRA factors, optimizer
state and activations. TP coverage is also distinct from FSDP sharding.

The [released BAGEL-7B-MoT language configuration](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT/blob/main/llm_config.json)
has width 3,584, MLP width 18,944, 28 layers, 28 query/4 KV heads of width 128,
and vocabulary size 152,064. The loader selects `Qwen2MoTDecoderLayer`, enables
Q/K norms and leaves embeddings untied. This gives **14,141,252,608** parameters
in 788 tensors. The two resolved plans place them as follows:

| Explicit SP | TP-sharded weight elements | TP-sharded bias elements | Replicated weight elements | Replicated bias elements | Sharded share |
|---|---:|---:|---:|---:|---:|
| Off (default) | 11,406,409,728 | 0 | 2,734,584,832 | 258,048 | 80.6605% |
| On | 13,050,576,896 | 258,048 | 1,090,417,664 | 0 | 92.2891% |

SP off shards 168 MLP tensors and replicates 620 tensors. SP on additionally
shards 392 attention tensors, leaving 228 replicated tensors: the embedding and
LM-head matrices each contain 544,997,376 elements, and 226 norm weights contain
422,912 elements in total. Sequence/QK norm gradient ownership does not turn
those replicated weights into TP shards. Native MLP and attention-output
projections have no biases; all 258,048 biases belong to Q/K/V projections.

For reproducibility, with `H=3584`, `I=18944`, `L=28`, `K=4*128`, `D=128` and
`V=152064`, both MLP branches contain `6*L*H*I` elements. Both attention branches
contain `L*(4*H*H + 4*H*K + 2*H + 4*K)` weight/bias elements. Norms contain
`L*(4*H + 4*D) + 2*H`, and the two vocabulary matrices contain `2*V*H`.
The focused CPU test constructs this native model on `meta`, expands the actual
SP-off/on plans, and checks every parameter's placement and these independent
counts without allocating model storage or running kernels:

```bash
PYTHONPATH=src python -m pytest -q tests/models/test_bagel_parallel_contracts.py \
  -k native_meta_parameter_coverage
```

This is a parameter-layout audit, not a new native numerical or capacity result;
the existing full/LoRA and CP/SP evidence below keeps its original scope.

Each packed sample receives its own masked internal padding. Queries and keys
keep the native prefix-then-target order, rotary positions, text/latent branch
routing and per-sample attention isolation. Native Diffusers splitting,
Ulysses autograd and output gathering own the CP communication. Dummy query
rows satisfy Ulysses' backward geometry for rectangular prefix attention and
are removed before output projections. Attention/MLP regions gather before routing uneven
text/latent subsets. Final role norms instead receive indexes filtered and rebased to each
TP-local residual slice. SP gathers after both final norms, followed by CP unsharding and
canonical packed-sample restoration. No extra
backward scaling is introduced; the existing trainer's full-output CP loss
owner remains authoritative.
Noncanonical packed prefix or query index maps are rejected before parallel generation;
absent or empty prefix maps remain valid when every prefix length is zero.

Native text, VAE and ViT updates keep their detached cache semantics. Ordered
multi-reference input and the three image/text CFG contexts retain their
separate lengths and positions. The existing FSDP restriction on differing
reference-image counts within a pack or across ranks remains in force. Ragged
native packing does not imply admission of those parameter-sharded workloads.

The selected numerical evidence separates three claims. Independent FP64 controls
prove packed CP transport, GQA/masking derivatives, virtual TP contractions and
warm LoRA factor ownership. A canonical full-parameter arithmetic reference then
reproduces the native TP2 contraction and CP2 attention geometry while rebuilding
its own native text/VAE/ViT prefix caches. Generation input and post-attention
RMSNorm calls use the physical residual row groups: CP-local rows, additionally
split into contiguous TP chunks before role selection when SP is enabled. Final
root norms now use those same local groups; the output gather follows the norms. Native norm functions,
weights and epsilon stay unchanged; prefix and Q/K norm paths retain their own
native grouping. Focused CPU controls check these maps, outputs and input/weight
VJPs with nonunit weights after three Adam updates, including empty roles and
prefix delegation. For full and LoRA, SP off/on, each of
three steps compares outputs and every gradient element at the current actual
weights. The unchanged shared gradient criterion applies its `atol=1e-3`,
`rtol=0.02` budget to every element and its `2/128` relative L2 cap once to the
complete trainable-gradient dictionary. Per-parameter and memory-window relative
statistics are diagnostic only; low-energy tensors or regions are not promised
an individual `2/128` relative bound. Partitioning a tensor or changing diagnostic
window sizes must not change this acceptance contract.
The actual model and optimizer advance without resets; the reference has no
optimizer. Separate FP64 own-gradient Adam equations check every actual parameter,
first/second moment and update over all three warm steps. Bounded deterministic
windows retain the observed source process count and limit CPU memory; the direct
model control proves no clipping is applied. Frozen isolation, checkpoint replay
and the omitted-CP-factor negative remain required.

The original monolithic Flash autonomous reference is retained as diagnostic
failure evidence. Its initial all-element output comparison and later gradient
comparison failed the original budgets. Same-state arithmetic evidence does not
turn those historical failures into passes or promise BF16 equality between
arbitrary serial and parallel contraction orders. The earlier arithmetic shadow
that normalized all generation role rows together also remains a diagnostic
failure: native CUDA RMSNorm on two versus four identical selected rows produced
a different FP32 reduction result at a BF16 rounding midpoint. Matching the
physical norm row groups resolves that fixed-operand discrepancy without changing
the native model or comparison budgets. This does not promise bitwise equality
for arbitrary norm launch geometry or an autonomous serial optimizer trajectory.

Earlier SP-on full and LoRA arithmetic-shadow attempts also remain recorded
failures under their then-enforced per-parameter/window relative caps. Raw audits
found every element and the original shared whole-gradient signal within its
unchanged limits. Applying that global cap independently to arbitrary memory
windows, or to each tensor, had strengthened the shared contract unintentionally.
The current gate restores the shared owner directly; it retains those regional
statistics for inspection and does not relabel the earlier attempts as passes.
Independent FP64 linear/attention VJPs, mask/head and omitted-CP-factor negatives
remain separate safeguards for mathematical ownership. Adam windows continue to
check every actual element and all warm moment recurrences; their scope is unchanged.

The conditioned Guard lifecycle uses two ordered images, native preprocessing,
preserved one-sample rollout/replay packs, EMA and evaluation. Fresh processes
restore checkpoint parameters, optimizer, runtime, work identity and RNG exactly.
The first resumed model-gradient/update step from identical parameters retains
the original BF16 bounds. Every actual continuation step keeps independent
objective and public-output derivative checks, FP64 clipping/Adam and EMA checks,
and exact work, optimizer settings and final RNG. Later autonomous cross-run
gradient/update statistics are diagnostic; matching future trajectories is not
promised. See the shared [ordered continuation contract](../context-parallel.md#ordered-continuation-evidence).
Native model controls separately cover B2 packing. Bagel declares
`requires_preserved_replay_batch_composition`: preprocessing and native BF16
projection results can depend on which samples share a pack. Numerical
serial/parallel comparisons therefore preserve the same ordered B2 operands;
whole-model packed-versus-singleton equality is not a supported contract.
The unsharded ragged smoke control keeps its original pack across all three CFG
calls and checks finite outputs, nonzero backward and unchanged detached prefix
caches. The separate fixed-operand FP64 attention oracle checks per-sample
attention isolation. FSDP count rejection has its own negative contract checks.

At CP1, Bagel's model CP capability adds no CP-specific identity fields. Prepared TP
placements still follow the shared [exact-state identity](../topology.md#data-feedback-and-checkpoint-boundaries).
CP was previously undeclared for Bagel, so there is no supported older CP checkpoint to migrate.
New CP checkpoints bind
the model-owned layout and capability version. The shared released-dependency gate is enforced
by preflight; a source revision is not a separate serialized checkpoint field. A future change
with incompatible execution semantics must bump the capability version. Exact
continuation requires the same declared parallel execution semantics. Explicit
SP changes TP attention parameter placement and must not be used to resume a legacy MLP-only
TP checkpoint under a different sequence-parallel setting.

The shared scheduler identity now sorts Diffusers' unordered default-argument
names without modifying the scheduler or its realized settings. This prevents
identical programmatically constructed schedulers from receiving different
digests in fresh processes. Older checkpoints containing a differently ordered
defaults list remain incompatible; no automatic migration or relaxed resume
check is provided. Configurations without that metadata retain their identity.

## Final role-norm SP selection

`bagel-sp-final-norm-reference` selects the expanded prepared output boundary. Native CPU tests
observe both role-norm inputs and require their total rows to match the local residual slice.
The independent canonical-parameter arithmetic reference retains native norm arithmetic on the
physical role groups, including empty roles and whole prefix delegation. Earlier references
that gathered before the final norms retain their historical scope. No new VAE/connector SP,
prefix-cache sharding or pretrained performance claim follows from this change.
