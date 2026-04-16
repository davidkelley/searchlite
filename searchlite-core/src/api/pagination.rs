use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::api::reader::DocLookupMap;
use crate::api::scoring::score_sort_key;
use crate::api::types::SortOrder;
use crate::index::segment::SegmentReader;
use crate::query::sort::{SortKey, SortPlan, SortValue};
use crate::DocId;

const CURSOR_VERSION: u8 = 1;
const CURSOR_BYTES: usize = 21;
const CURSOR_HEX_LEN: usize = CURSOR_BYTES * 2;
const SORT_CURSOR_VERSION: u8 = 2;

/// Upper bound on the number of documents a cursor can advance past.
/// Imported from reader to enforce the limit during decoding.
use super::reader::MAX_CURSOR_ADVANCE;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PaginationCursor {
  pub(crate) version: u8,
  pub(crate) generation: u32,
  pub(crate) key: SortKey,
  pub(crate) returned: u32,
}

impl PaginationCursor {
  pub(crate) fn encode(&self) -> String {
    let score_bits = self
      .key
      .score_bits()
      .expect("score cursor missing score value");
    let mut buf = [0u8; CURSOR_BYTES];
    buf[0] = self.version;
    buf[1..5].copy_from_slice(&self.generation.to_be_bytes());
    buf[5..9].copy_from_slice(&score_bits.to_be_bytes());
    buf[9..13].copy_from_slice(&self.key.segment_ord.to_be_bytes());
    buf[13..17].copy_from_slice(&self.key.doc_id.to_be_bytes());
    buf[17..].copy_from_slice(&self.returned.to_be_bytes());
    let mut encoded = String::with_capacity(CURSOR_HEX_LEN);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in buf {
      encoded.push(HEX[(byte >> 4) as usize] as char);
      encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
  }

  pub(crate) fn decode(raw: &str) -> Result<Self> {
    if raw.len() != CURSOR_HEX_LEN {
      bail!(
        "invalid cursor length: expected {CURSOR_HEX_LEN} hex chars, got {}",
        raw.len()
      );
    }
    if !raw.is_ascii() {
      bail!("invalid cursor: must be ASCII hex");
    }
    let mut bytes = [0u8; CURSOR_BYTES];
    for (i, chunk) in raw.as_bytes().chunks_exact(2).enumerate() {
      // Report positions against the raw input string so diagnostics point the
      // caller at the offending character, not the decoded byte slot.
      let raw_offset = 2 * i;
      // `raw.is_ascii()` guarantees each byte is ASCII, so every two-byte
      // chunk is valid UTF-8. We propagate via `?` rather than `unwrap()` so a
      // future regression of the guard remains non-fatal.
      let hex = std::str::from_utf8(chunk).map_err(|_| {
        anyhow::anyhow!("cursor contains non-ASCII bytes at raw offset {raw_offset}")
      })?;
      let value = u8::from_str_radix(hex, 16)
        .with_context(|| format!("decoding cursor at raw offset {raw_offset}"))?;
      bytes[i] = value;
    }
    let version = bytes[0];
    if version != CURSOR_VERSION {
      bail!("unsupported cursor version {version}");
    }
    let generation = u32::from_be_bytes(bytes[1..5].try_into().unwrap());
    let score_bits = u32::from_be_bytes(bytes[5..9].try_into().unwrap());
    let segment_ord = u32::from_be_bytes(bytes[9..13].try_into().unwrap());
    let doc_id = u32::from_be_bytes(bytes[13..17].try_into().unwrap());
    let returned = u32::from_be_bytes(bytes[17..21].try_into().unwrap());
    if returned as usize > MAX_CURSOR_ADVANCE {
      bail!("cursor requests {returned} hits, which exceeds max supported {MAX_CURSOR_ADVANCE}");
    }
    Ok(Self {
      version,
      generation,
      key: score_sort_key(
        f32::from_bits(score_bits),
        segment_ord,
        doc_id,
        SortOrder::Desc,
      ),
      returned,
    })
  }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct SortCursorState {
  pub(crate) version: u8,
  pub(crate) generation: u32,
  pub(crate) returned: u32,
  pub(crate) plan_hash: u32,
  pub(crate) segment_ord: u32,
  pub(crate) doc_id: DocId,
  pub(crate) values: Vec<CursorValue>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "t", content = "v", rename_all = "lowercase")]
pub(crate) enum CursorValue {
  Score(u32),
  I64(i64),
  F64(f64),
  Str(String),
  Missing,
}

impl From<SortValue> for CursorValue {
  fn from(value: SortValue) -> Self {
    match value {
      SortValue::Score(score) => CursorValue::Score(score.to_bits()),
      SortValue::I64(v) => CursorValue::I64(v),
      SortValue::F64(v) => CursorValue::F64(v),
      SortValue::Str(v) => CursorValue::Str(v),
      SortValue::Missing => CursorValue::Missing,
    }
  }
}

impl From<CursorValue> for SortValue {
  fn from(value: CursorValue) -> Self {
    match value {
      CursorValue::Score(bits) => SortValue::Score(f32::from_bits(bits)),
      CursorValue::I64(v) => SortValue::I64(v),
      CursorValue::F64(v) => SortValue::F64(v),
      CursorValue::Str(v) => SortValue::Str(v),
      CursorValue::Missing => SortValue::Missing,
    }
  }
}

pub(crate) fn hex_encode(bytes: &[u8]) -> String {
  const HEX: &[u8; 16] = b"0123456789abcdef";
  let mut out = String::with_capacity(bytes.len() * 2);
  for byte in bytes {
    out.push(HEX[(byte >> 4) as usize] as char);
    out.push(HEX[(byte & 0x0f) as usize] as char);
  }
  out
}

pub(crate) fn hex_decode(raw: &str) -> Result<Vec<u8>> {
  if raw.len() & 1 != 0 {
    bail!("invalid cursor: expected even-length hex string");
  }
  if !raw.is_ascii() {
    bail!("invalid cursor: must be ASCII hex");
  }
  let mut bytes = Vec::with_capacity(raw.len() / 2);
  for (i, chunk) in raw.as_bytes().chunks_exact(2).enumerate() {
    // Report positions against the raw input so diagnostics point the caller
    // at the offending character, not the decoded byte slot.
    let raw_offset = 2 * i;
    // Guarded by `is_ascii()` above, so every two-byte chunk is valid UTF-8.
    // Propagate via `?` instead of `unwrap()` to avoid a panic on regression.
    let hex = std::str::from_utf8(chunk)
      .map_err(|_| anyhow::anyhow!("cursor contains non-ASCII bytes at raw offset {raw_offset}"))?;
    let value = u8::from_str_radix(hex, 16)
      .with_context(|| format!("decoding cursor at raw offset {raw_offset}"))?;
    bytes.push(value);
  }
  Ok(bytes)
}

pub(crate) fn sort_value_to_json(value: &SortValue) -> serde_json::Value {
  match value {
    SortValue::Score(v) => serde_json::Number::from_f64(*v as f64)
      .map(serde_json::Value::Number)
      .unwrap_or(serde_json::Value::Null),
    SortValue::I64(v) => serde_json::Value::Number((*v).into()),
    SortValue::F64(v) => serde_json::Number::from_f64(*v)
      .map(serde_json::Value::Number)
      .unwrap_or(serde_json::Value::Null),
    SortValue::Str(v) => serde_json::Value::String(v.clone()),
    SortValue::Missing => serde_json::Value::Null,
  }
}

fn parse_segment_ord(raw: &serde_json::Value) -> Result<u32> {
  match raw {
    serde_json::Value::Number(n) => n
      .as_u64()
      .and_then(|v| u32::try_from(v).ok())
      .ok_or_else(|| anyhow::anyhow!("search_after segment_ord must be a non-negative integer")),
    serde_json::Value::String(s) => {
      let trimmed = if let Some(rest) = s.strip_prefix("seg") {
        rest
      } else {
        s.as_str()
      };
      trimmed
        .trim()
        .parse::<u32>()
        .context("parsing search_after segment_ord")
    }
    _ => Err(anyhow::anyhow!(
      "search_after segment_ord must be string or integer"
    )),
  }
}

pub(crate) fn decode_search_after_token(
  token: &[serde_json::Value],
  sort_plan: &SortPlan,
  segments: &[SegmentReader],
  doc_lookup: &DocLookupMap,
) -> Result<SortKey> {
  if token.len() < sort_plan.len().saturating_add(2) {
    bail!(
      "search_after token length {} is less than expected {} values plus doc_id and segment_ord",
      token.len(),
      sort_plan.len()
    );
  }
  if token.len() != sort_plan.len().saturating_add(2) {
    bail!(
      "search_after token must contain {} sort values plus doc_id and segment_ord",
      sort_plan.len()
    );
  }
  let values = sort_plan.values_from_json(&token[..sort_plan.len()])?;
  let doc_id_value = token.get(sort_plan.len()).unwrap();
  let seg_value = token.get(sort_plan.len() + 1).unwrap();
  let segment_ord = parse_segment_ord(seg_value)?;
  let seg = segments
    .get(segment_ord as usize)
    .ok_or_else(|| anyhow::anyhow!("search_after segment_ord {segment_ord} out of range"))?;
  let doc_id_str = match doc_id_value {
    serde_json::Value::String(s) => s.clone(),
    serde_json::Value::Number(n) => n.to_string(),
    _ => {
      bail!("search_after doc_id must be string or number");
    }
  };
  let doc_id: DocId = doc_lookup
    .get(doc_id_str.as_str())
    .and_then(|entries| {
      entries
        .iter()
        .find(|(seg_idx, _)| *seg_idx == segment_ord)
        .map(|(_, doc_idx)| *doc_idx)
    })
    .or_else(|| seg.find_doc_id(&doc_id_str))
    .ok_or_else(|| {
      anyhow::anyhow!(
        "search_after doc_id `{}` not found in segment {}",
        doc_id_str,
        seg.meta.id
      )
    })?;
  if seg.is_deleted(doc_id) {
    bail!(
      "search_after doc_id `{}` in segment {} refers to a deleted document",
      doc_id_str,
      seg.meta.id
    );
  }
  sort_plan.key_from_values(&values, segment_ord, doc_id)
}

pub(crate) fn encode_search_after_token(
  sort_plan: &SortPlan,
  key: &SortKey,
  segments: &[SegmentReader],
) -> Result<Vec<serde_json::Value>> {
  let values = sort_plan.values_from_key(key)?;
  let mut out: Vec<serde_json::Value> = values.iter().map(sort_value_to_json).collect();
  let seg = segments
    .get(key.segment_ord as usize)
    .ok_or_else(|| anyhow::anyhow!("segment {} missing for search_after", key.segment_ord))?;
  let doc_id_str = seg
    .doc_id(key.doc_id)
    .ok_or_else(|| anyhow::anyhow!("doc_id {} missing in segment", key.doc_id))?;
  out.push(serde_json::Value::String(doc_id_str.to_string()));
  out.push(serde_json::Value::Number(key.segment_ord.into()));
  Ok(out)
}

#[derive(Clone, Debug)]
pub(crate) struct CursorState {
  pub(crate) key: SortKey,
  pub(crate) returned: u32,
}

pub(crate) fn decode_cursor(
  raw: &str,
  manifest_generation: u32,
  sort_plan: &SortPlan,
  score_fast_path: bool,
) -> Result<CursorState> {
  if score_fast_path {
    let cur = PaginationCursor::decode(raw)?;
    if cur.generation != manifest_generation {
      bail!(
        "stale cursor for this index generation: expected {}, got {}",
        manifest_generation,
        cur.generation
      );
    }
    return Ok(CursorState {
      key: cur.key,
      returned: cur.returned,
    });
  }
  let bytes = hex_decode(raw)?;
  let state: SortCursorState =
    serde_json::from_slice(&bytes).context("parsing sort cursor payload")?;
  if state.version != SORT_CURSOR_VERSION {
    bail!("unsupported sort cursor version {}", state.version);
  }
  if state.generation != manifest_generation {
    bail!(
      "stale cursor for this index generation: expected {}, got {}",
      manifest_generation,
      state.generation
    );
  }
  if state.plan_hash != sort_plan.hash() {
    bail!("cursor sort order does not match this request");
  }
  if state.returned as usize > MAX_CURSOR_ADVANCE {
    bail!(
      "cursor requests {} hits, which exceeds max supported {MAX_CURSOR_ADVANCE}",
      state.returned
    );
  }
  let values: Vec<SortValue> = state.values.into_iter().map(SortValue::from).collect();
  let key = sort_plan.key_from_values(&values, state.segment_ord, state.doc_id)?;
  Ok(CursorState {
    key,
    returned: state.returned,
  })
}

pub(crate) fn encode_cursor(
  manifest_generation: u32,
  returned: u32,
  key: &SortKey,
  sort_plan: &SortPlan,
  score_fast_path: bool,
) -> Result<String> {
  if score_fast_path {
    return Ok(
      PaginationCursor {
        version: CURSOR_VERSION,
        generation: manifest_generation,
        key: key.clone(),
        returned,
      }
      .encode(),
    );
  }
  let values = sort_plan.values_from_key(key)?;
  let state = SortCursorState {
    version: SORT_CURSOR_VERSION,
    generation: manifest_generation,
    returned,
    plan_hash: sort_plan.hash(),
    segment_ord: key.segment_ord,
    doc_id: key.doc_id,
    values: values.into_iter().map(CursorValue::from).collect(),
  };
  let data = serde_json::to_vec(&state)?;
  Ok(hex_encode(&data))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn pagination_cursor_roundtrips() {
    let cursor = PaginationCursor {
      version: CURSOR_VERSION,
      generation: 2,
      key: score_sort_key(1.5, 2, 3, SortOrder::Desc),
      returned: 42,
    };
    let encoded = cursor.encode();
    let decoded = PaginationCursor::decode(&encoded).unwrap();
    assert_eq!(decoded, cursor);
  }

  #[test]
  fn pagination_cursor_rejects_bad_length() {
    assert!(PaginationCursor::decode("deadbeef").is_err());
  }

  #[test]
  fn pagination_cursor_rejects_non_hex() {
    let invalid = "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"; // 42 chars, not hex
    assert!(PaginationCursor::decode(invalid).is_err());
  }

  #[test]
  fn pagination_cursor_rejects_excessive_advance() {
    let mut buf = [0u8; CURSOR_BYTES];
    buf[0] = CURSOR_VERSION;
    let returned = (MAX_CURSOR_ADVANCE as u32).saturating_add(1);
    buf[17..].copy_from_slice(&returned.to_be_bytes());
    let encoded = hex_encode(&buf);
    assert!(PaginationCursor::decode(&encoded).is_err());
  }

  #[test]
  fn pagination_cursor_rejects_multibyte_utf8_without_panic() {
    // 14 × U+4E16 ("世") = 42 bytes of valid UTF-8 that happens to match
    // `CURSOR_HEX_LEN`, so the length check passes. Every two-byte chunk
    // lands inside a three-byte UTF-8 sequence, which used to panic at
    // `from_utf8(..).unwrap()`. The fix must surface an error instead.
    let input: String = "\u{4E16}".repeat(14);
    assert_eq!(input.len(), CURSOR_HEX_LEN);
    let result = PaginationCursor::decode(&input);
    assert!(
      result.is_err(),
      "expected Err for non-ASCII cursor, got {result:?}"
    );
  }

  #[test]
  fn hex_decode_rejects_multibyte_utf8_without_panic() {
    // Two multi-byte characters produce 6 bytes — even-length, passes the
    // original gate, and every chunk straddles a UTF-8 boundary.
    let result = hex_decode("\u{4E16}\u{4E16}");
    assert!(
      result.is_err(),
      "expected Err for non-ASCII hex input, got {result:?}"
    );
  }

  #[test]
  fn hex_decode_accepts_valid_ascii_hex() {
    // Positive control: ensure the added ASCII guard doesn't break the happy path.
    let decoded = hex_decode("deadbeef").expect("valid ASCII hex decodes");
    assert_eq!(decoded, vec![0xde, 0xad, 0xbe, 0xef]);
  }
}
