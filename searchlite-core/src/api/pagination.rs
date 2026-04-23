use std::sync::Arc;

use anyhow::{bail, Context, Result};
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::api::scoring::score_sort_key;
use crate::api::types::SortOrder;
use crate::index::segment::SegmentReader;
use crate::query::sort::{SortKey, SortPlan, SortValue};
use crate::DocId;

/// Compact entry list for `DocLookupMap`. A given `doc_id` usually lives in a
/// single segment, so the inline capacity of `1` keeps the common case off the
/// heap while still supporting multi-segment tombstones for updated documents.
pub(crate) type DocLookupEntries = SmallVec<[(u32, DocId); 1]>;

/// Map from `doc_id` to the `(segment_ord, doc_idx)` pairs that currently host
/// it. Keys are cheaply shared `Arc<str>` clones of the segment-owned doc_ids.
pub(crate) type DocLookupMap = HashMap<Arc<str>, DocLookupEntries>;

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
    // The score bytes are reconstructed directly from user-supplied hex, so a
    // bit pattern like 0x7F800000 (+inf) or 0x7FC00000 (NaN) is indistinguishable
    // from a legitimate encoded score. Left unchecked these non-finite values
    // flow into `SortKey` and silently corrupt keyset pagination under
    // `total_cmp` (where NaN/+inf sort beyond all finite scores). Mirrors the
    // guard added for `search_after` _score in BUG-342.
    let score = f32::from_bits(score_bits);
    if !score.is_finite() {
      bail!("cursor contains non-finite score bits 0x{score_bits:08X}");
    }
    Ok(Self {
      version,
      generation,
      key: score_sort_key(score, segment_ord, doc_id, SortOrder::Desc),
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

impl TryFrom<CursorValue> for SortValue {
  type Error = anyhow::Error;

  fn try_from(value: CursorValue) -> Result<Self> {
    match value {
      // JSON-encoded cursors carry the score as a `u32` bit pattern. `serde`
      // happily accepts any u32, including 0x7F800000 (+inf) or 0x7FC00000
      // (NaN), so we must reject non-finite reconstructions here; otherwise
      // the poisoned value reaches `SortKey` and silently corrupts pagination
      // under `total_cmp`. Mirrors the hex-cursor guard above and the
      // `search_after` guard from BUG-342.
      CursorValue::Score(bits) => {
        let score = f32::from_bits(bits);
        if !score.is_finite() {
          bail!("cursor contains non-finite score bits 0x{bits:08X}");
        }
        Ok(SortValue::Score(score))
      }
      CursorValue::I64(v) => Ok(SortValue::I64(v)),
      // Same story as the Score variant above: a crafted JSON cursor can
      // deliver any `f64` bit pattern, including +/-inf or NaN, either
      // directly in the cursor or via a JSON literal that overflows to
      // `f64::INFINITY` during deserialization. Let those reach `SortKey`
      // and `total_cmp`-based pagination silently skips or duplicates
      // pages. Mirrors the `search_after` F64 guard (BUG-369) and the
      // Score guards from BUG-342 / BUG-345.
      CursorValue::F64(v) => {
        if !v.is_finite() {
          bail!("cursor contains non-finite F64 sort value ({v})");
        }
        Ok(SortValue::F64(v))
      }
      CursorValue::Str(v) => Ok(SortValue::Str(v)),
      CursorValue::Missing => Ok(SortValue::Missing),
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
  let values: Vec<SortValue> = state
    .values
    .into_iter()
    .map(SortValue::try_from)
    .collect::<Result<_>>()
    .context("decoding cursor sort values")?;
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

  /// Build a raw hex cursor whose score bits are set to `score_bits`.
  /// All other fields are zero, which is enough to exercise the decode-time
  /// score validation path. Other cursor validity checks depend on
  /// caller-supplied state and are enforced separately — the advance limit
  /// (`returned <= MAX_CURSOR_ADVANCE`) runs inside `decode` before this
  /// guard but is unaffected by zeroed bytes, and `generation` is checked
  /// by `decode_cursor` rather than `PaginationCursor::decode`.
  fn cursor_hex_with_score_bits(score_bits: u32) -> String {
    let mut buf = [0u8; CURSOR_BYTES];
    buf[0] = CURSOR_VERSION;
    buf[5..9].copy_from_slice(&score_bits.to_be_bytes());
    hex_encode(&buf)
  }

  #[test]
  fn pagination_cursor_rejects_positive_infinity_score_bits() {
    // 0x7F800000 is the IEEE-754 f32 positive infinity bit pattern. Any
    // legitimate encode path produces only finite scores, so surfacing this
    // as an error prevents a crafted cursor from silently corrupting
    // keyset pagination under `total_cmp`.
    let encoded = cursor_hex_with_score_bits(0x7F800000);
    let err = PaginationCursor::decode(&encoded).expect_err("expected non-finite score rejection");
    let msg = err.to_string();
    assert!(
      msg.contains("non-finite score"),
      "unexpected error message: {msg}"
    );
  }

  #[test]
  fn pagination_cursor_rejects_negative_infinity_score_bits() {
    let encoded = cursor_hex_with_score_bits(0xFF800000);
    let err = PaginationCursor::decode(&encoded).expect_err("expected non-finite score rejection");
    assert!(err.to_string().contains("non-finite score"));
  }

  #[test]
  fn pagination_cursor_rejects_nan_score_bits() {
    // 0x7FC00000 is an IEEE-754 f32 quiet NaN bit pattern.
    let encoded = cursor_hex_with_score_bits(0x7FC00000);
    let err = PaginationCursor::decode(&encoded).expect_err("expected non-finite score rejection");
    assert!(err.to_string().contains("non-finite score"));
  }

  #[test]
  fn pagination_cursor_accepts_finite_score_bits_at_boundary() {
    // Positive control: f32::MAX is the largest finite score and must still
    // decode successfully after the finitude guard.
    let encoded = cursor_hex_with_score_bits(f32::MAX.to_bits());
    let decoded = PaginationCursor::decode(&encoded).expect("finite score bits must decode");
    assert!(decoded.key.score_bits().is_some());
  }

  #[test]
  fn cursor_value_try_from_rejects_non_finite_score() {
    // JSON cursors deserialize `CursorValue::Score(u32)` straight from
    // serde, so the u32 can encode +inf/-inf/NaN. The TryFrom conversion
    // must reject those before they reach the sort key.
    for bits in [0x7F800000_u32, 0xFF800000, 0x7FC00000] {
      let err = SortValue::try_from(CursorValue::Score(bits))
        .expect_err("non-finite score bits must be rejected");
      assert!(
        err.to_string().contains("non-finite score"),
        "bits {bits:#X}: unexpected error: {err}"
      );
    }
  }

  #[test]
  fn cursor_value_try_from_accepts_finite_score() {
    let v = SortValue::try_from(CursorValue::Score(1.5_f32.to_bits()))
      .expect("finite score must convert");
    match v {
      SortValue::Score(s) => assert_eq!(s, 1.5_f32),
      other => panic!("expected Score, got {other:?}"),
    }
  }

  #[test]
  fn cursor_value_try_from_passes_through_non_score_variants() {
    // I64, Str, and Missing have no finitude guard and must continue to
    // round-trip unchanged. F64 is covered separately below because the
    // finite values still pass through, but non-finite values are
    // rejected (BUG-369).
    assert!(matches!(
      SortValue::try_from(CursorValue::I64(-7)).unwrap(),
      SortValue::I64(-7)
    ));
    assert!(matches!(
      SortValue::try_from(CursorValue::F64(2.5)).unwrap(),
      SortValue::F64(v) if v == 2.5
    ));
    assert!(matches!(
      SortValue::try_from(CursorValue::Missing).unwrap(),
      SortValue::Missing
    ));
    match SortValue::try_from(CursorValue::Str("k".into())).unwrap() {
      SortValue::Str(s) => assert_eq!(s, "k"),
      other => panic!("expected Str, got {other:?}"),
    }
  }

  #[test]
  fn cursor_value_try_from_rejects_non_finite_f64() {
    // A crafted JSON cursor can deliver any f64, including +/-inf or NaN,
    // either by embedding the bit pattern directly or by supplying a JSON
    // literal that overflows to f64::INFINITY during deserialization.
    // Those values must be rejected before they reach the sort key;
    // otherwise `total_cmp`-based keyset pagination silently skips or
    // duplicates pages.
    for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
      let err = SortValue::try_from(CursorValue::F64(value))
        .expect_err("non-finite F64 cursor value must be rejected");
      assert!(
        err.to_string().contains("non-finite F64 sort value"),
        "value {value}: unexpected error: {err}"
      );
    }
  }

  #[test]
  fn cursor_value_try_from_accepts_finite_f64_at_boundary() {
    // Positive control: finite boundary values (including signed zero)
    // must still round-trip after the guard so legitimate cursors are
    // not over-rejected. Bit-pattern equality keeps +0.0 and -0.0
    // distinguishable across the conversion.
    for value in [f64::MAX, f64::MIN, 0.0_f64, -0.0_f64] {
      let v = SortValue::try_from(CursorValue::F64(value)).expect("finite F64 must convert");
      match v {
        SortValue::F64(out) => {
          assert!(out.is_finite());
          assert_eq!(out.to_bits(), value.to_bits());
        }
        other => panic!("expected F64, got {other:?}"),
      }
    }
  }
}
