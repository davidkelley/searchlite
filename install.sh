#!/bin/sh
# install.sh: fetches the latest (or specified) Searchlite CLI release and installs it locally.
set -eu

OWNER="davidkelley"
REPO="searchlite"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command '$1' not found in PATH" >&2
    exit 1
  fi
}

detect_target() {
  os="$(uname -s)"
  arch="$(uname -m)"

  case "$os" in
    Darwin)
      case "$arch" in
        arm64|aarch64) target="aarch64-apple-darwin" ;;
        x86_64|amd64) target="x86_64-apple-darwin" ;;
        *) echo "error: unsupported macOS architecture: $arch" >&2; exit 1 ;;
      esac
      ;;
    Linux)
      case "$arch" in
        x86_64|amd64) target="x86_64-unknown-linux-gnu" ;;
        aarch64|arm64) target="aarch64-unknown-linux-gnu" ;;
        *) echo "error: unsupported Linux architecture: $arch" >&2; exit 1 ;;
      esac
      ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
      case "$arch" in
        x86_64|amd64) target="x86_64-pc-windows-msvc" ;;
        aarch64|arm64) target="aarch64-pc-windows-msvc" ;;
        *) echo "error: unsupported Windows architecture: $arch" >&2; exit 1 ;;
      esac
      ;;
    *)
      echo "error: unsupported OS: $os" >&2
      exit 1
      ;;
  esac

  echo "$target"
}

download() {
  url="$1"
  dest="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -fL "$url" -o "$dest"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$dest" "$url"
  else
    echo "error: need 'curl' or 'wget' to download artifacts" >&2
    exit 1
  fi
}

install_bin() {
  src="$1"
  dst="$2"

  mkdir -p "$(dirname "$dst")"

  if command -v install >/dev/null 2>&1; then
    install -m 0755 "$src" "$dst"
  else
    cp "$src" "$dst"
    chmod 755 "$dst"
  fi
}

choose_install_dir() {
  preferred="${SEARCHLITE_INSTALL_DIR:-/usr/local/bin}"
  fallback="$HOME/.local/bin"

  if [ -d "$preferred" ] && [ -w "$preferred" ]; then
    echo "$preferred"
    return
  fi

  if mkdir -p "$preferred" 2>/dev/null; then
    echo "$preferred"
    return
  fi

  if mkdir -p "$fallback" 2>/dev/null; then
    echo "$fallback"
    return
  fi

  echo "error: cannot create an install directory; set SEARCHLITE_INSTALL_DIR to a writable path" >&2
  exit 1
}

TARGET="$(detect_target)"
ARCHIVE="tar.gz"
BIN_EXT=""
case "$TARGET" in
  *windows*) ARCHIVE="zip"; BIN_EXT=".exe" ;;
esac

VERSION="${SEARCHLITE_VERSION:-latest}"
if [ "$VERSION" != "latest" ] && [ "${VERSION#v}" = "$VERSION" ]; then
  VERSION="v$VERSION"
fi

BASE_URL="https://github.com/$OWNER/$REPO/releases"
if [ "$VERSION" = "latest" ]; then
  BASE_URL="$BASE_URL/latest/download"
else
  BASE_URL="$BASE_URL/download/$VERSION"
fi

ASSET="searchlite-cli-${TARGET}.${ARCHIVE}"
URL="$BASE_URL/$ASSET"

TMPDIR="$(mktemp -d 2>/dev/null || mktemp -d -t searchlite)"
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT INT TERM HUP

ARCHIVE_PATH="$TMPDIR/$ASSET"
BIN_NAME="searchlite-cli$BIN_EXT"
INSTALL_NAME="${SEARCHLITE_BIN_NAME:-searchlite}$BIN_EXT"

echo "Detected target: $TARGET"
echo "Fetching: $URL"

download "$URL" "$ARCHIVE_PATH"

if [ "$ARCHIVE" = "zip" ]; then
  need_cmd unzip
  unzip -q "$ARCHIVE_PATH" -d "$TMPDIR"
else
  need_cmd tar
  tar -xzf "$ARCHIVE_PATH" -C "$TMPDIR"
fi

BIN_PATH="$TMPDIR/$BIN_NAME"
if [ ! -f "$BIN_PATH" ]; then
  echo "error: binary $BIN_NAME not found in archive" >&2
  exit 1
fi

DEST_DIR="$(choose_install_dir)"
DEST_PATH="$DEST_DIR/$INSTALL_NAME"

install_bin "$BIN_PATH" "$DEST_PATH"
echo "Installed to $DEST_PATH"

if printf '%s' ":$PATH:" | grep -q ":$DEST_DIR:"; then
  :
else
  echo "note: $DEST_DIR is not on PATH; add it to run '$INSTALL_NAME' directly"
fi

echo "Done. Run '$INSTALL_NAME --help' to get started."
