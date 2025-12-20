# AGENTS: Contribution Guardrails

- Run `cargo fmt` on every change set to keep style consistent.
- Run `cargo test` (and relevant package-specific tests/benches) before pushing; changes must ship green.
- Ensure the workspace builds: `cargo build --workspace` (plus `cargo check --all-targets` if unsure).
- Prefer clear, explicit, and highly performant code; avoid cleverness that obscures hot paths or allocations.
- Use Conventional Commits (`feat: ...`, `fix: ...`, `perf: ...`, `chore: ...`, etc.) for history hygiene.
- If behavior changes, add or update tests to cover it; no user-visible change should land untested.
