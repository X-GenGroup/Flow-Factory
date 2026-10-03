# Reward Scope

Follow the repository root `AGENTS.md`. Use `/ff-new-reward` for a new scorer and `/ff-develop`
for shared reward infrastructure.

- Local pointwise scorers default to R1; groupwise or remote scorers require focused escalation.
- Keep applicability, reconstruction, grouping, aggregation, and optimizer work-unit ownership in
  their existing framework layers.
- Changes to `abc.py`, `reward_processor.py`, `tile_plan.py`, canonical group identity, advantage,
  or overlap are R3/R4 candidates.
- Verify full and tail batches, output order/shape/finiteness, and cleanup/failure propagation where
  asynchronous work is involved.
