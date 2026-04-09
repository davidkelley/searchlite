set shell := ["bash", "-cu"]

build:
  cargo build --all --all-features

test:
  cargo test --all --all-features

test-integration-quick:
  INTEGRATION_MODE=quick cargo test -p integration --all-features

test-integration-full:
  INTEGRATION_MODE=full cargo test -p integration --all-features

bench:
  cargo bench -p searchlite-core

fmt:
  cargo fmt --all

lint:
  cargo clippy --all --all-features -- -D warnings

build-node:
  cd searchlite-node && npm run build

test-node:
  cd searchlite-node && npm test

lint-node:
  cd searchlite-node && npm run lint

typecheck-node:
  cd searchlite-node && npm run typecheck

check-node:
  cd searchlite-node && npm run lint && npm run typecheck && npm test
