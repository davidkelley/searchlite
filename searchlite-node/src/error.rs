use std::panic::{catch_unwind, AssertUnwindSafe};

use napi::Status;

pub(crate) fn to_napi_error(err: anyhow::Error) -> napi::Error {
  napi::Error::new(Status::GenericFailure, format!("{err:#}"))
}

pub(crate) fn catch_panic<T>(name: &str, f: impl FnOnce() -> napi::Result<T>) -> napi::Result<T> {
  match catch_unwind(AssertUnwindSafe(f)) {
    Ok(result) => result,
    Err(payload) => {
      let msg = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(|s| s.as_str()))
        .unwrap_or("<unknown panic>");
      Err(napi::Error::new(
        Status::GenericFailure,
        format!("internal error in {name}: {msg}"),
      ))
    }
  }
}
