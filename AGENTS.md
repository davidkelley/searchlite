# AGENTS: Contribution Guardrails

- Run `cargo fmt --all` on every change set; CI enforces `cargo fmt --all -- --check`.
- Run `cargo clippy --all --all-features --all-targets -- -D warnings` locally to match CI.
- Ensure the workspace builds for all features: `cargo build --all --all-features` (plus `cargo check --all-targets` if unsure).
- Run tests with full coverage: `cargo test --all --all-features`; add/update tests when behavior changes.
- Run `cargo bench -p searchlite-core` before landing perf-sensitive changes; CI runs this suite.
- Keep compatibility with Rust 1.88.0+ (CI matrix: 1.88, 1.92, stable, beta, nightly).
- Prefer clear, explicit, and highly performant code; avoid cleverness that obscures hot paths or allocations.
- Use Conventional Commits (`feat: ...`, `fix: ...`, `perf: ...`, `chore: ...`, etc.) for history hygiene.
