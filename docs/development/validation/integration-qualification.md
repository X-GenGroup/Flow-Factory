# P02-P08 merged-baseline qualification

Checked on **2026-11-05** at
[`fbb038e3570cb8f0e19cf74d910d6ea1d722ffd2`](https://github.com/X-GenGroup/Flow-Factory/commit/fbb038e3570cb8f0e19cf74d910d6ea1d722ffd2),
after PRs #281, #282, #283, #285, #286, #287 and #288 merged. This record qualifies that source;
it does not publish a release or transfer evidence to a later commit.

## Integrated source

The cumulative plan compares `27a2ab15d32b3c4a1dc999ccc86593ba253c10fa` with the tested commit.
Its SHA-256 is `2b343efea23d8bf8c852605b83967956e0ef454ad4cb36ce82aa21fe02a3f13c`.
All **53 selected cases pass**, with no uncovered obligations, failed assertions or skips:

| Evidence | Executed assertions |
|---|---|
| Primary CPU contracts | 1,027 tests, plus repository consistency |
| CPU/Gloo geometry, TP/SP, checkpoint and feedback contracts | 39 tests |
| Optional model capabilities and Bagel/FLUX.3 structure parity | 22 tests |
| CUDA, DDP, FSDP2, ZeRO-2 and pretrained Wan | 21 tests |

The 1,109 test count excludes repeated runs used to capture actual per-rank backend identity.
The final report checks the exact source, test/configuration hashes, dependency identities,
assertion artifacts and observed backend plugins. The cumulative code-quality check also passes.

The primary environment is Python 3.10.22, PyTorch 2.10.0+cu129, Accelerate 1.15.0,
Diffusers 0.41.0, Transformers 5.20.2, PEFT 0.21.2 and DeepSpeed 0.19.8. An immutable environment
archive was deployed to four nodes and all installed-package fingerprints matched. Each node
provides eight NVIDIA L20Y GPUs with approximately 80 GiB per GPU; driver version is 570.172.08.

The three optional CPU/Gloo cases use Python 3.13 and Diffusers 0.42.0.dev0 at
`48cdbaef5ee3d718aaee392b44f2099ae511c4a5`, with FlashAttention 2.8.3.post2 and matching
PyTorch/Accelerate/PEFT/Transformers versions. That auxiliary environment has unrelated inherited
package conflicts and is not the primary training environment. See the existing
[model-specific qualifications](../../reference/parallelism/tensor-parallel.md) for scope beyond these fixtures.

The Wan canaries use the transformer from `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, revision
`0fad780a534b6463e45facd96134c9f345acfa5b`, staged locally with file hashes. They cover the denoiser,
LoRA gradients/updates and FSDP2 TP/SP; they are not full-pipeline or output-quality evaluations.

## Physical cross-node follow-up

An additional **15 tests pass on two physical nodes**, using the same environment and source.
The original numerical workers and pytest parent assertions are retained. A temporary launcher
places the workers over SSH and uses TCP rendezvous; checkpoints and rank reports use shared storage.

| Boundary | Evidence |
|---|---|
| Two-rank NCCL DP, TP and CP scopes | Group membership, logical sampling, RNG and feedback reductions |
| Wan FSDP2 DP2 × TP2, with TP crossing nodes | Full and LoRA, TP and checkpointed SP; four updates with elementwise gradients and populated optimizer moments |
| Two-role preparation with TP crossing nodes | Full and shared LoRA; alternating role updates and inactive-state isolation |
| Fixed-topology FSDP2 and FSDP2+TP resume | Resumed next update and optimizer moments exactly match continuous execution; EMA/reference/snapshot state is exercised |
| Bounded Wan DP2 × CP2 probe, with CP crossing nodes | Three updates and the explicit unscaled-loss negative control |
| DP crossing nodes | Checkpointed Wan SP/LoRA, FSDP2+TP resume and the bounded CP probe |

For four ranks, interleaved placement assigns ranks `{0, 2}` to node A and `{1, 3}` to node B,
so TP or CP peers communicate across nodes. Contiguous placement assigns `{0, 1}` and `{2, 3}`,
so logical DP peers communicate across nodes. Physical host observations were recorded per rank.

Two launcher failures were diagnosed before acceptance: shared-filesystem `FileStore` destruction
spun after the NCCL assertions completed, and a fixed TCP port was already occupied. The launcher
uses TCPStore with an OS-assigned port and bounded worker lifetimes. Failed attempts remain in the
investigation record; they are not counted as passing tests.

## Reproduction and limits

Use the [validation catalog](validation-catalog.md) commands and the
[automatic validation workflow](automatic-validation.md) to generate the cumulative plan and
verify the report. The qualification PR carries the bounded result bundle, including concise
assertion logs and hashes of the full original logs. The full experiment archive has SHA-256
`d56419283093d6f7e6f29b2e3540d83f1f02731b6f6fda33077721ef983ffb8a` and is 126,281 bytes.
It preserves the plan/results, original logs, rank JSON, JUnit reports and cross-node launcher.

This closes the merged-baseline handoff to P09. Production CP, TP+CP, EP, CP exact resume,
topology-changing exact resume, cross-node pretrained-weight training, the unresolved
Qwen-Image-2.1 real-layout discrepancy and H3 loading-capacity limitations remain outside this
qualification. No throughput, scaling-efficiency or universally hidden reward-latency claim is made.

The proposed v0.3 target remains P04 integration commit `17e8be7c`. Evidence collected on P08
does not qualify that earlier target; release preparation must validate each exact target separately.
