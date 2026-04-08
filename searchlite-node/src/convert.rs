use std::collections::BTreeMap;

use searchlite_core::api::types::Document;

pub(crate) fn value_to_document(value: serde_json::Value) -> napi::Result<Document> {
  let obj = value
    .as_object()
    .ok_or_else(|| napi::Error::new(napi::Status::InvalidArg, "document must be a JSON object"))?;
  let mut fields = BTreeMap::new();
  for (k, v) in obj.iter() {
    fields.insert(k.clone(), v.clone());
  }
  Ok(Document { fields })
}

pub(crate) fn value_to_documents(value: serde_json::Value) -> napi::Result<Vec<Document>> {
  match value {
    serde_json::Value::Array(items) => items.into_iter().map(value_to_document).collect(),
    obj @ serde_json::Value::Object(_) => Ok(vec![value_to_document(obj)?]),
    _ => Err(napi::Error::new(
      napi::Status::InvalidArg,
      "documents must be an object or array of objects",
    )),
  }
}
