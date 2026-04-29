# syntax=docker/dockerfile:1.7
#
# Build: docker build -f searchlite-adapter-elastic/Dockerfile -t searchlite-elastic .
# Run:   docker run --rm -p 9200:9200 \
#          -e SEARCHLITE_ELASTIC_UPSTREAM_URL=http://searchlite-http-host:8080 \
#          searchlite-elastic --bind 0.0.0.0:9200
#
# Build context must be the repo root so cargo can resolve the workspace.

# Builder stage: compile the searchlite-elastic adapter binary.
FROM rust:1.95-bookworm AS builder
WORKDIR /workspace

# Install minimal build tooling (ca-certificates so released binary can validate
# TLS backends if used to talk to an upstream over HTTPS).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates pkg-config \
 && rm -rf /var/lib/apt/lists/*

# Cache dependencies separately to speed up iterative builds. We copy every
# workspace member's Cargo.toml so cargo can resolve the workspace, then
# replace each one's source with a placeholder before doing the dependency
# fetch. The real sources are copied in afterwards.
COPY Cargo.toml Cargo.lock ./
COPY searchlite-core/Cargo.toml searchlite-core/Cargo.toml
COPY searchlite-cli/Cargo.toml searchlite-cli/Cargo.toml
COPY searchlite-http/Cargo.toml searchlite-http/Cargo.toml
COPY searchlite-ffi/Cargo.toml searchlite-ffi/Cargo.toml
COPY searchlite-wasm/Cargo.toml searchlite-wasm/Cargo.toml
COPY searchlite-node/Cargo.toml searchlite-node/Cargo.toml
COPY searchlite-adapter-elastic/Cargo.toml searchlite-adapter-elastic/Cargo.toml
COPY integration/Cargo.toml integration/Cargo.toml

RUN mkdir -p \
      searchlite-core/src searchlite-core/benches \
      searchlite-cli/src searchlite-http/src searchlite-ffi/src \
      searchlite-wasm/src searchlite-node/src \
      searchlite-adapter-elastic/src \
      integration/src \
 && echo "fn main() {}" > searchlite-cli/src/main.rs \
 && echo "fn main() {}" > searchlite-adapter-elastic/src/main.rs \
 && echo "pub fn placeholder() {}" > searchlite-core/src/lib.rs \
 && echo "fn main() {}" > searchlite-core/benches/aggs.rs \
 && echo "fn main() {}" > searchlite-core/benches/end_to_end.rs \
 && echo "fn main() {}" > searchlite-core/benches/scale.rs \
 && echo "pub fn placeholder() {}" > searchlite-http/src/lib.rs \
 && echo "pub fn placeholder() {}" > searchlite-ffi/src/lib.rs \
 && echo "pub fn placeholder() {}" > searchlite-wasm/src/lib.rs \
 && echo "pub fn placeholder() {}" > searchlite-node/src/lib.rs \
 && echo "pub fn placeholder() {}" > searchlite-adapter-elastic/src/lib.rs \
 && echo "pub fn placeholder() {}" > integration/src/lib.rs

RUN cargo fetch --locked

# Now copy the real sources and build the release binary.
COPY . .
RUN cargo build --locked --release -p searchlite-adapter-elastic --bin searchlite-elastic

# Runtime stage: minimal distroless image with only the binary and certificates.
FROM gcr.io/distroless/cc-debian12 AS runtime
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=builder /workspace/target/release/searchlite-elastic /usr/local/bin/searchlite-elastic

EXPOSE 9200
ENTRYPOINT ["/usr/local/bin/searchlite-elastic"]
