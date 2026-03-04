use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;

pub fn searchlite_bin() -> PathBuf {
  if let Ok(path) = std::env::var("CARGO_BIN_EXE_searchlite") {
    return PathBuf::from(path);
  }
  if let Ok(path) = std::env::var("CARGO_BIN_EXE_searchlite-cli") {
    return PathBuf::from(path);
  }

  let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .parent()
    .expect("workspace root")
    .to_path_buf();

  let candidates = [
    workspace_root
      .join("target")
      .join("debug")
      .join(if cfg!(windows) {
        "searchlite.exe"
      } else {
        "searchlite"
      }),
    workspace_root
      .join("target")
      .join("debug")
      .join(if cfg!(windows) {
        "searchlite-cli.exe"
      } else {
        "searchlite-cli"
      }),
  ];
  for candidate in candidates {
    if candidate.exists() {
      return candidate;
    }
  }

  let status = Command::new("cargo")
    .arg("build")
    .arg("-p")
    .arg("searchlite-cli")
    .current_dir(&workspace_root)
    .status()
    .expect("build searchlite binary");
  assert!(status.success(), "building searchlite-cli failed");

  workspace_root
    .join("target")
    .join("debug")
    .join(if cfg!(windows) {
      "searchlite-cli.exe"
    } else {
      "searchlite-cli"
    })
}

#[allow(dead_code)]
pub fn docs_to_ndjson(docs: &[Value]) -> String {
  let mut out = String::new();
  for doc in docs {
    out.push_str(&serde_json::to_string(doc).expect("serialize document"));
    out.push('\n');
  }
  out
}
