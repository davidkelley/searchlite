set shell := ["bash", "-cu"]

build:
  cargo build --all --all-features

test:
  cargo test --all --all-features

bench:
  cargo bench -p searchlite-core

fmt:
  cargo fmt --all

lint:
  cargo clippy --all --all-features -- -D warnings
