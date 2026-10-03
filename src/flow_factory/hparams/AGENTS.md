# Configuration Scope

Follow the repository root `AGENTS.md` and use `/ff-develop` or the selected extension skill.

- Algorithm arguments and trainer classes declare the same immutable execution contract.
- User-facing field changes update validation, every affected example, docs, and runtime access.
- Preserve early compatibility rejection before model or dataset loading.
- Test production parsing and defaults; do not rely on string presence in YAML alone.
