# Trainer Scope

Follow the repository root `AGENTS.md`. Use `/ff-new-algorithm` for an algorithm and `/ff-develop`
for shared execution infrastructure.

- An objective-only trainer using an existing execution contract and hooks may use R1.
- Strict inheritance, custom sampling scope, and existing snapshots require at least R2.
- Advanced lifecycle hooks, runtime children, role topology, collectives, overlap, or resume identity
  require R3.
- Changes to `abc.py`, `execution.py`, shared role optimization, feedback, sampler, loading, or
  checkpoint infrastructure are R4 candidates.
- Keep trainer and argument registries and immutable execution contracts aligned.
