# syntax=docker/dockerfile:1.7

# Builder stage: compile the searchlite CLI binary.
FROM rust:1.92-bookworm AS builder
WORKDIR /workspace

# Install minimal build tooling (ca-certificates so released binary can validate TLS backends when used).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates pkg-config \
 && rm -rf /var/lib/apt/lists/*

# Cache dependencies separately to speed up iterative builds.
COPY Cargo.toml Cargo.lock ./
COPY searchlite-core/Cargo.toml searchlite-core/Cargo.toml
COPY searchlite-cli/Cargo.toml searchlite-cli/Cargo.toml
COPY searchlite-http/Cargo.toml searchlite-http/Cargo.toml
COPY searchlite-ffi/Cargo.toml searchlite-ffi/Cargo.toml
COPY searchlite-wasm/Cargo.toml searchlite-wasm/Cargo.toml
COPY searchlite-node/Cargo.toml searchlite-node/Cargo.toml
COPY integration/Cargo.toml integration/Cargo.toml

# Create dummy source files so cargo can resolve the workspace without pulling full sources yet.
RUN mkdir -p searchlite-core/src searchlite-cli/src searchlite-http/src searchlite-ffi/src searchlite-wasm/src searchlite-node/src integration/src \
 && echo "fn main() {}" > searchlite-cli/src/main.rs \
 && echo "pub fn placeholder() {}" > searchlite-core/src/lib.rs \
 && echo "pub fn placeholder() {}" > searchlite-http/src/lib.rs \
 && echo "pub fn placeholder() {}" > searchlite-ffi/src/lib.rs \
 && echo "pub fn placeholder() {}" > searchlite-wasm/src/lib.rs \
 && echo "pub fn placeholder() {}" > searchlite-node/src/lib.rs \
 && echo "pub fn placeholder() {}" > integration/src/lib.rs

RUN cargo fetch --locked

# Now copy the real sources and build the release binary.
COPY . .
RUN cargo build --locked --release -p searchlite-cli

# Runtime stage: minimal distroless image with only the binary and certificates.
FROM gcr.io/distroless/cc-debian12 AS runtime
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=builder /workspace/target/release/searchlite-cli /usr/local/bin/searchlite

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/searchlite"]
