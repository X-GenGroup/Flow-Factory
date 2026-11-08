# AWM context parallelism

AWM declares the shared full-output CP training contract. Its objective stays in `AWMTrainer`:
weighted matching produces the old/current score ratio; existing clipped advantages and optional
velocity-space reference/EMA penalties are combined before the shared CP-normalized backward.
The adapter owns component masks, active degrees of freedom and time/noise transformations.

The selected native witness combines Qwen Image full tuning, FSDP2/BF16, CP2 × DP2, active CFG,
`ghuber` weighting, off-policy EMA sampling, shared group noise, and both reference and EMA KL.
Two independent training times reuse each old-policy noised state for the live/reference/EMA
queries. Multiple ordered reward tiles and acquisitions exercise pending feedback during optimizer
work, EMA evaluation, drained checkpointing, fresh-process restore and subsequent updates.
Qwen's exact unused final-text parameter set retains its absent gradients and optimizer state.

Independent CPU checks cover all five weightings and their derivatives, ratio clipping for both
advantage signs, per-component sigmas and masks, CP/GAS scaling, detached old/snapshot state,
noise/time/conditioning identity and exception restoration. A passive native observer applies the
same analytical checks to the real output tensors with their observed precision. The shared
optimizer observer verifies canonical parameter gradients, clipping, Adam moments and updates.

Use the existing overlap constraints: tile-local statistics (`global_std: false`), one inner epoch,
no sample shuffling and whole accumulation windows. The witness is rank-local/ordered; it does not
admit Qwen TP+CP or editing, nor claim every algorithm/model/topology combination was run.
The concrete model and algorithm capability contracts remain authoritative.

Run the focused tests with the training environment:

```bash
python -m pytest -q tests/distributed/test_awm_objective_reference.py
python -m pytest -q -s tests/distributed/test_awm_context_parallel.py::test_cuda_qwen_awm_ordered_full
```

The second command needs four local GPUs. Catalog cases `awm-objective-contracts` and
`qwen-awm-overlap-native` bind the commands, transitive fixture inputs and runtime capability
`qwen-image-awm-cp-v1`. A support claim requires their exact-source result bundle; this guide alone
is not a successful GPU-run record. See [automatic validation](../../../development/validation/automatic-validation.md).
