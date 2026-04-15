use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use hashbrown::HashMap;
use smallvec::SmallVec;

use crate::index::codec::{read_f32, read_u32, write_f32, write_u32};
use crate::util::varint::{read_u32_var, write_u32_var};
use crate::DocId;

pub const DEFAULT_BLOCK_SIZE: usize = 128;
const BLOCK_META_FLAG: u32 = 1u32 << 31;

/// Reject element counts that cannot possibly be backed by the bytes still
/// available in the segment file. The `count` is read from an untrusted
/// header field and would otherwise drive a `Vec::with_capacity` /
/// `Vec::reserve` of arbitrary size, allowing a tampered or corrupt segment
/// to commit a multi-gigabyte allocation before the per-entry read loop
/// surfaces the truncation. Mirrors the helper introduced by BUG-012 for
/// `fastfields::read_fields`, adapted for the `Read + Seek` byte budget.
fn checked_count(count: usize, min_stride: usize, remaining: u64) -> Result<usize> {
  let needed = (count as u64)
    .checked_mul(min_stride as u64)
    .ok_or_else(|| anyhow!("postings count {count} * stride {min_stride} overflows u64"))?;
  if needed > remaining {
    return Err(anyhow!(
      "postings count {count} would need {needed} bytes but only {remaining} remain in segment"
    ));
  }
  Ok(count)
}

fn stream_remaining<R: Seek>(file: &mut R, end: u64) -> Result<u64> {
  let cur = file.stream_position()?;
  Ok(end.saturating_sub(cur))
}

#[derive(Debug, Clone)]
pub struct PostingEntry {
  pub doc_id: DocId,
  pub term_freq: u32,
  pub positions: SmallVec<[u32; 4]>,
}

#[derive(Debug, Clone, Default)]
pub struct InvertedIndexBuilder {
  terms: HashMap<String, Vec<PostingEntry>>,
}

impl InvertedIndexBuilder {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn add_term(&mut self, term: &str, doc_id: DocId, position: u32, keep_positions: bool) {
    let entry = self.terms.entry(term.to_string()).or_default();
    if let Some(last) = entry.last_mut() {
      if last.doc_id == doc_id {
        last.term_freq += 1;
        if keep_positions {
          last.positions.push(position);
        }
        return;
      }
    }
    let mut positions = SmallVec::new();
    if keep_positions {
      positions.push(position);
    }
    entry.push(PostingEntry {
      doc_id,
      term_freq: 1,
      positions,
    });
  }

  pub fn into_terms(self) -> Vec<(String, Vec<PostingEntry>)> {
    let mut pairs: Vec<_> = self.terms.into_iter().collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    pairs
  }
}

pub struct PostingsWriter<'a, W: Write + Seek + ?Sized> {
  file: &'a mut W,
  keep_positions: bool,
}

pub fn read_doc_freq<R: Read + Seek>(file: &mut R, offset: u64) -> Result<u32> {
  file.seek(SeekFrom::Start(offset))?;
  read_u32(file)
}

impl<'a, W: Write + Seek + ?Sized> PostingsWriter<'a, W> {
  pub fn new(file: &'a mut W, keep_positions: bool) -> Self {
    Self {
      file,
      keep_positions,
    }
  }

  pub fn write_term(&mut self, postings: &[PostingEntry]) -> Result<u64> {
    let offset = self.file.stream_position()?;
    write_u32(self.file, postings.len() as u32)?;
    self.file.write_all(&[self.keep_positions as u8])?;
    let block_size = DEFAULT_BLOCK_SIZE;
    let block_count = postings.len().div_ceil(block_size).min(u32::MAX as usize) as u32;
    let block_flagged = if block_count > 0 {
      block_count | BLOCK_META_FLAG
    } else {
      0
    };
    write_u32(self.file, block_flagged)?;
    let max_doc_id = postings.last().map(|p| p.doc_id).unwrap_or(0);
    let max_tf = postings
      .iter()
      .map(|p| p.term_freq as f32)
      .fold(0.0_f32, f32::max);
    write_u32(self.file, max_doc_id)?;
    write_f32(self.file, max_tf)?;

    if block_count > 0 {
      write_u32(self.file, block_size as u32)?;
      for chunk in postings.chunks(block_size) {
        let max_doc = chunk.last().map(|p| p.doc_id).unwrap_or(0);
        write_u32(self.file, max_doc)?;
      }
      for chunk in postings.chunks(block_size) {
        let tf_max = chunk
          .iter()
          .map(|p| p.term_freq as f32)
          .fold(0.0_f32, f32::max);
        write_f32(self.file, tf_max)?;
      }
    }

    let mut buf = Vec::with_capacity(postings.len() * 8);
    for p in postings {
      write_u32_var(p.doc_id, &mut buf);
      write_u32_var(p.term_freq, &mut buf);
      if self.keep_positions {
        write_u32_var(p.positions.len() as u32, &mut buf);
        let mut prev = 0u32;
        for pos in p.positions.iter().copied() {
          // Positions must be non-decreasing so that delta encoding is
          // reversible. `PostingEntry` is a public type and its `positions`
          // field is directly constructible by external callers, so we
          // cannot rely on convention — reject out-of-order input here
          // rather than producing a segment that silently fails to decode
          // (debug: subtract-overflow panic; release: wrapped delta that
          // decodes into garbage positions — see BUG-003 / BUG-004).
          let delta = pos.checked_sub(prev).ok_or_else(|| {
            anyhow::anyhow!(
              "positions must be non-decreasing (doc {} pos {} after {})",
              p.doc_id,
              pos,
              prev
            )
          })?;
          write_u32_var(delta, &mut buf);
          prev = pos;
        }
      }
    }
    self.file.write_all(&buf)?;
    Ok(offset)
  }
}

/// Pre-computed per-block upper bounds, shared via `Arc` so that cloning
/// a `PostingsReader` (or extracting block metadata in the WAND loop)
/// is an O(1) reference-count bump instead of a full vector copy.
#[derive(Debug, Clone)]
pub struct BlockMeta {
  pub doc_ids: Vec<DocId>,
  pub tfs: Vec<f32>,
  pub block_size: usize,
}

#[derive(Debug, Clone)]
pub struct PostingsReader {
  data: Vec<PostingEntry>,
  pub max_tf: f32,
  block_meta: Arc<BlockMeta>,
}

impl PostingsReader {
  /// Block-max doc IDs (one per block).
  pub fn block_max_doc_ids(&self) -> &[DocId] {
    &self.block_meta.doc_ids
  }

  /// Block-max term frequencies (one per block).
  pub fn block_max_tfs(&self) -> &[f32] {
    &self.block_meta.tfs
  }

  /// Block size used when computing block metadata.
  pub fn block_size(&self) -> usize {
    self.block_meta.block_size
  }

  /// Cheaply share the block metadata via `Arc`.
  pub fn block_meta(&self) -> Arc<BlockMeta> {
    Arc::clone(&self.block_meta)
  }
}

impl PostingsReader {
  pub fn read_at<R: Read + Seek>(file: &mut R, offset: u64, keep_positions: bool) -> Result<Self> {
    file.seek(SeekFrom::Start(offset))?;
    // Capture the segment file length once so subsequent untrusted counts can
    // be validated against the remaining byte budget before being passed to
    // `Vec::with_capacity` / `Vec::reserve`. Otherwise a single 4-byte
    // `doc_freq` field could drive a 171 GiB allocation (BUG-205, mirrors
    // BUG-012 for `fastfields::read_fields`).
    let end = file.seek(SeekFrom::End(0))?;
    file.seek(SeekFrom::Start(offset))?;
    let doc_freq = read_u32(file)? as usize;
    // Track the on-disk flag independently of the caller's preference so that
    // we always consume the bytes that are actually present in the file.
    // Conflating these two concerns previously caused position bytes to be
    // left in the stream when `stored_positions == true` and
    // `keep_positions == false`, corrupting every subsequent entry.
    let stored_positions = {
      let mut flag = [0u8; 1];
      file.read_exact(&mut flag)?;
      flag[0] == 1
    };
    let keep_in_memory = stored_positions && keep_positions;
    let raw_block = read_u32(file)?;
    let has_block_meta = raw_block & BLOCK_META_FLAG != 0;
    let block_count = (raw_block & (!BLOCK_META_FLAG)) as usize;
    let max_doc_id = read_u32(file)?;
    let mut max_tf = read_f32(file)?;

    let mut block_size = DEFAULT_BLOCK_SIZE;
    let mut block_max_doc_ids = Vec::new();
    let mut block_max_tfs = Vec::new();
    if has_block_meta && block_count > 0 {
      block_size = read_u32(file)? as usize;
      // Each block-meta entry is 4 B (u32 max_doc_id) + 4 B (f32 max_tf),
      // read in two sequential loops. Validate the count against the
      // remaining byte budget so a crafted `block_count` near `2^31 - 1`
      // cannot reserve ~16 GiB before the read loop runs.
      let remaining = stream_remaining(file, end)?;
      let validated_block_count = checked_count(block_count, 8, remaining)?;
      block_max_doc_ids.reserve(validated_block_count);
      block_max_tfs.reserve(validated_block_count);
      for _ in 0..validated_block_count {
        block_max_doc_ids.push(read_u32(file)?);
      }
      for _ in 0..validated_block_count {
        block_max_tfs.push(read_f32(file)?);
      }
    }
    // Each posting entry encodes at minimum a 1-byte varint for `doc_id`
    // and a 1-byte varint for `term_freq` (plus a 1-byte varint for the
    // position count when `stored_positions`). Validate `doc_freq` against
    // the remaining byte budget before reserving capacity for a
    // 40-byte-element vector.
    let min_entry_stride = if stored_positions { 3 } else { 2 };
    let remaining = stream_remaining(file, end)?;
    let doc_freq = checked_count(doc_freq, min_entry_stride, remaining)?;
    let mut data = Vec::with_capacity(doc_freq);
    for _ in 0..doc_freq {
      let doc_id = read_u32_var(file)?;
      let term_freq = read_u32_var(file)?;
      let mut positions = SmallVec::new();
      if stored_positions {
        let count = read_u32_var(file)? as usize;
        if keep_in_memory {
          let mut acc = 0u32;
          for _ in 0..count {
            acc = acc
              .checked_add(read_u32_var(file)?)
              .ok_or_else(|| anyhow::anyhow!("position delta overflow while decoding postings"))?;
            positions.push(acc);
          }
        } else {
          // Caller doesn't want positions in memory, but the bytes are on
          // disk and the file cursor must still be advanced past them.
          for _ in 0..count {
            read_u32_var(file)?;
          }
        }
      }
      data.push(PostingEntry {
        doc_id,
        term_freq,
        positions,
      });
    }
    if block_max_doc_ids.is_empty() {
      block_size = DEFAULT_BLOCK_SIZE;
      for chunk in data.chunks(block_size) {
        let max_doc = chunk.last().map(|p| p.doc_id).unwrap_or(max_doc_id);
        let tf_max = chunk
          .iter()
          .map(|p| p.term_freq as f32)
          .fold(0.0_f32, f32::max);
        block_max_doc_ids.push(max_doc);
        block_max_tfs.push(tf_max);
      }
    }
    let computed_max = block_max_tfs.iter().copied().fold(0.0_f32, f32::max);
    if computed_max > max_tf {
      max_tf = computed_max;
    }
    Ok(Self {
      data,
      max_tf,
      block_meta: Arc::new(BlockMeta {
        doc_ids: block_max_doc_ids,
        tfs: block_max_tfs,
        block_size,
      }),
    })
  }

  pub fn iter(&self) -> impl Iterator<Item = &PostingEntry> {
    self.data.iter()
  }

  pub fn entry(&self, idx: usize) -> Option<&PostingEntry> {
    self.data.get(idx)
  }

  pub fn entries(&self) -> &[PostingEntry] {
    &self.data
  }

  pub fn len(&self) -> usize {
    self.data.len()
  }

  /// Drop all per-entry position data, freeing memory.
  ///
  /// Call this when only `doc_id` and `term_freq` are needed (e.g., BM25/WAND
  /// scoring). Positions are only required for phrase matching and are often
  /// the largest component of high-frequency postings lists.
  pub fn strip_positions(&mut self) {
    for entry in self.data.iter_mut() {
      if !entry.positions.is_empty() {
        entry.positions = SmallVec::new();
      }
    }
  }

  #[cfg(test)]
  pub fn from_entries_for_test(entries: Vec<PostingEntry>, block_size: usize) -> Self {
    let block_size = block_size.max(1);
    let max_tf = entries
      .iter()
      .map(|p| p.term_freq as f32)
      .fold(0.0_f32, f32::max);
    let mut block_max_doc_ids = Vec::new();
    let mut block_max_tfs = Vec::new();
    for chunk in entries.chunks(block_size) {
      if let Some(last) = chunk.last() {
        block_max_doc_ids.push(last.doc_id);
      }
      let tf_max = chunk
        .iter()
        .map(|p| p.term_freq as f32)
        .fold(0.0_f32, f32::max);
      block_max_tfs.push(tf_max);
    }
    Self {
      data: entries,
      max_tf,
      block_meta: Arc::new(BlockMeta {
        doc_ids: block_max_doc_ids,
        tfs: block_max_tfs,
        block_size,
      }),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use smallvec::smallvec;
  use tempfile::NamedTempFile;

  #[test]
  fn builder_merges_terms_per_doc() {
    let mut builder = InvertedIndexBuilder::new();
    builder.add_term("body:rust", 0, 0, true);
    builder.add_term("body:rust", 0, 1, true);
    builder.add_term("body:rust", 1, 0, true);
    let terms = builder.into_terms();
    assert_eq!(terms.len(), 1);
    let (_, postings) = &terms[0];
    assert_eq!(postings.len(), 2);
    assert_eq!(postings[0].term_freq, 2);
    assert_eq!(postings[0].positions.as_slice(), &[0, 1]);
    assert_eq!(postings[1].doc_id, 1);
  }

  #[test]
  fn writes_and_reads_postings() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();
    let mut file = tmp.reopen().unwrap();
    let postings = vec![
      PostingEntry {
        doc_id: 1,
        term_freq: 2,
        positions: smallvec![1, 3],
      },
      PostingEntry {
        doc_id: 2,
        term_freq: 1,
        positions: smallvec![4],
      },
    ];
    let mut writer = PostingsWriter::new(&mut file, true);
    let offset = writer.write_term(&postings).unwrap();

    let mut reader_file = std::fs::File::open(path).unwrap();
    let reader = PostingsReader::read_at(&mut reader_file, offset, true).unwrap();
    assert_eq!(reader.len(), 2);
    assert!(reader.max_tf >= 2.0);
    assert_eq!(reader.block_max_doc_ids().len(), 1);
    assert_eq!(reader.block_max_tfs().len(), 1);
    let collected: Vec<_> = reader
      .iter()
      .map(|p| (p.doc_id, p.positions.iter().copied().collect::<Vec<_>>()))
      .collect();
    assert_eq!(collected, vec![(1, vec![1, 3]), (2, vec![4])]);
  }

  /// Regression for BUG-001: when postings are written with positions stored on
  /// disk but read with `keep_positions = false`, the reader must still advance
  /// the file cursor past the position block. Otherwise position bytes bleed
  /// into subsequent entries and every posting after the first is corrupt.
  #[test]
  fn reads_positioned_postings_with_keep_positions_false() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();
    let mut file = tmp.reopen().unwrap();
    let postings = vec![
      PostingEntry {
        doc_id: 1,
        term_freq: 2,
        positions: smallvec![1, 3],
      },
      PostingEntry {
        doc_id: 2,
        term_freq: 1,
        positions: smallvec![4],
      },
      PostingEntry {
        doc_id: 5,
        term_freq: 3,
        positions: smallvec![7, 9, 12],
      },
    ];
    let mut writer = PostingsWriter::new(&mut file, true);
    let offset = writer.write_term(&postings).unwrap();

    let mut reader_file = std::fs::File::open(path).unwrap();
    let reader = PostingsReader::read_at(&mut reader_file, offset, false).unwrap();
    assert_eq!(reader.len(), 3);
    let collected: Vec<_> = reader
      .iter()
      .map(|p| (p.doc_id, p.term_freq, p.positions.len()))
      .collect();
    // doc_ids and term_freqs must survive intact; positions are skipped in
    // memory (hence `len == 0`) but must have been fully consumed from disk.
    assert_eq!(collected, vec![(1, 2, 0), (2, 1, 0), (5, 3, 0)]);
    assert!(
      reader.iter().all(|p| p.positions.is_empty()),
      "positions must not be materialised when keep_positions = false"
    );
  }

  /// Larger variant of the regression: with enough entries to force the
  /// corruption to surface as a varint decode error rather than a plausible
  /// (but wrong) doc_id. This guards against a future regression where an
  /// off-by-one on the skip logic looks correct on small inputs.
  #[test]
  fn reads_many_positioned_postings_with_keep_positions_false() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();
    let mut file = tmp.reopen().unwrap();
    let mut postings = Vec::new();
    for i in 0..64u32 {
      let base = i * 10;
      postings.push(PostingEntry {
        doc_id: i + 1,
        term_freq: (i % 5) + 1,
        positions: smallvec![base, base + 2, base + 4],
      });
    }
    let expected: Vec<(u32, u32)> = postings.iter().map(|p| (p.doc_id, p.term_freq)).collect();

    let mut writer = PostingsWriter::new(&mut file, true);
    let offset = writer.write_term(&postings).unwrap();

    let mut reader_file = std::fs::File::open(path).unwrap();
    let reader = PostingsReader::read_at(&mut reader_file, offset, false).unwrap();
    assert_eq!(reader.len(), expected.len());
    let collected: Vec<(u32, u32)> = reader.iter().map(|p| (p.doc_id, p.term_freq)).collect();
    assert_eq!(collected, expected);
    assert!(reader.iter().all(|p| p.positions.is_empty()));
  }

  /// Regression for BUG-004: writing a posting entry with non-monotonic
  /// positions must return an error rather than panicking in debug or
  /// silently producing a corrupt (un-decodable) segment in release.
  #[test]
  fn write_term_rejects_non_monotonic_positions() {
    let tmp = NamedTempFile::new().unwrap();
    let mut file = tmp.reopen().unwrap();
    let postings = vec![PostingEntry {
      doc_id: 0,
      term_freq: 2,
      positions: smallvec![5, 3], // out of order
    }];
    let mut writer = PostingsWriter::new(&mut file, true);
    let err = writer
      .write_term(&postings)
      .expect_err("non-monotonic positions must be rejected");
    let msg = err.to_string();
    assert!(
      msg.contains("non-decreasing"),
      "unexpected error message: {msg}"
    );
  }

  /// Regression for BUG-004: unchecked subtraction previously panicked in
  /// debug builds when the first position was smaller than the implicit
  /// starting value of zero — make sure a leading zero-prev is still fine,
  /// but a subsequent backward step at any offset is rejected.
  #[test]
  fn write_term_rejects_backward_step_mid_list() {
    let tmp = NamedTempFile::new().unwrap();
    let mut file = tmp.reopen().unwrap();
    let postings = vec![PostingEntry {
      doc_id: 7,
      term_freq: 3,
      positions: smallvec![0, 10, 4],
    }];
    let mut writer = PostingsWriter::new(&mut file, true);
    let err = writer
      .write_term(&postings)
      .expect_err("backward-step positions must be rejected");
    assert!(err.to_string().contains("non-decreasing"));
  }

  /// Equal consecutive positions (delta = 0) are legal; the encoder must
  /// accept them and the round-trip must preserve them.
  #[test]
  fn write_term_allows_equal_consecutive_positions() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();
    let mut file = tmp.reopen().unwrap();
    let postings = vec![PostingEntry {
      doc_id: 1,
      term_freq: 3,
      positions: smallvec![2, 2, 5],
    }];
    let mut writer = PostingsWriter::new(&mut file, true);
    let offset = writer.write_term(&postings).unwrap();

    let mut reader_file = std::fs::File::open(path).unwrap();
    let reader = PostingsReader::read_at(&mut reader_file, offset, true).unwrap();
    let round_tripped: Vec<_> = reader
      .iter()
      .map(|p| p.positions.iter().copied().collect::<Vec<_>>())
      .collect();
    assert_eq!(round_tripped, vec![vec![2, 2, 5]]);
  }

  /// Regression for BUG-205: a tampered or corrupt segment header that
  /// claims `u32::MAX` postings entries on a near-empty file must be
  /// rejected before `Vec::with_capacity` commits a multi-gigabyte
  /// reservation. The check happens *before* the per-entry read loop
  /// would otherwise surface the truncation as `failed to fill whole
  /// buffer`, so the assertion targets the bounds-check error string
  /// rather than an I/O error.
  #[test]
  fn read_at_rejects_oversized_doc_freq_on_short_file() {
    use std::io::Cursor;
    let mut buf = Vec::new();
    buf.extend_from_slice(&u32::MAX.to_le_bytes()); // doc_freq
    buf.push(0); // stored_positions
    buf.extend_from_slice(&0u32.to_le_bytes()); // raw_block (no block meta)
    buf.extend_from_slice(&0u32.to_le_bytes()); // max_doc_id
    buf.extend_from_slice(&0f32.to_le_bytes()); // max_tf
    let mut cur = Cursor::new(buf);
    let err =
      PostingsReader::read_at(&mut cur, 0, false).expect_err("oversized doc_freq must be rejected");
    let msg = err.to_string().to_lowercase();
    assert!(
      msg.contains("postings count") && msg.contains("remain"),
      "unexpected error: {err}"
    );
  }

  /// Same as above, with `stored_positions = 1` so the minimum entry stride
  /// becomes 3 bytes. The crafted file is still far too small to satisfy a
  /// `u32::MAX` claim.
  #[test]
  fn read_at_rejects_oversized_doc_freq_with_positions_on_short_file() {
    use std::io::Cursor;
    let mut buf = Vec::new();
    buf.extend_from_slice(&u32::MAX.to_le_bytes()); // doc_freq
    buf.push(1); // stored_positions
    buf.extend_from_slice(&0u32.to_le_bytes()); // raw_block (no block meta)
    buf.extend_from_slice(&0u32.to_le_bytes()); // max_doc_id
    buf.extend_from_slice(&0f32.to_le_bytes()); // max_tf
    let mut cur = Cursor::new(buf);
    let err =
      PostingsReader::read_at(&mut cur, 0, false).expect_err("oversized doc_freq must be rejected");
    let msg = err.to_string().to_lowercase();
    assert!(
      msg.contains("postings count") && msg.contains("remain"),
      "unexpected error: {err}"
    );
  }

  /// Regression for BUG-205: an oversized `block_count` (bottom 31 bits of
  /// the block header) must be rejected before the two block-meta `reserve`
  /// calls can commit ~16 GiB of capacity.
  #[test]
  fn read_at_rejects_oversized_block_count_on_short_file() {
    use std::io::Cursor;
    let mut buf = Vec::new();
    buf.extend_from_slice(&0u32.to_le_bytes()); // doc_freq = 0
    buf.push(0); // stored_positions
    buf.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // flag set + block_count = 2^31 - 1
    buf.extend_from_slice(&0u32.to_le_bytes()); // max_doc_id
    buf.extend_from_slice(&0f32.to_le_bytes()); // max_tf
    buf.extend_from_slice(&128u32.to_le_bytes()); // block_size
    let mut cur = Cursor::new(buf);
    let err = PostingsReader::read_at(&mut cur, 0, false)
      .expect_err("oversized block_count must be rejected");
    let msg = err.to_string().to_lowercase();
    assert!(
      msg.contains("postings count") && msg.contains("remain"),
      "unexpected error: {err}"
    );
  }

  /// Sanity: when the segment was written without positions, reading with
  /// `keep_positions = true` must not attempt to decode a position block.
  #[test]
  fn reads_unpositioned_postings_with_keep_positions_true() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();
    let mut file = tmp.reopen().unwrap();
    let postings = vec![
      PostingEntry {
        doc_id: 1,
        term_freq: 2,
        positions: smallvec![],
      },
      PostingEntry {
        doc_id: 7,
        term_freq: 4,
        positions: smallvec![],
      },
    ];
    let mut writer = PostingsWriter::new(&mut file, false);
    let offset = writer.write_term(&postings).unwrap();

    let mut reader_file = std::fs::File::open(path).unwrap();
    let reader = PostingsReader::read_at(&mut reader_file, offset, true).unwrap();
    assert_eq!(reader.len(), 2);
    let collected: Vec<(u32, u32, usize)> = reader
      .iter()
      .map(|p| (p.doc_id, p.term_freq, p.positions.len()))
      .collect();
    assert_eq!(collected, vec![(1, 2, 0), (7, 4, 0)]);
  }

  /// Regression for the unchecked u32 addition overflow in position delta
  /// accumulation. Two deltas each equal to u32::MAX / 2 + 1 would sum to
  /// 2^32 (u32::MAX + 1), overflowing in release builds (wrapping to 0) or
  /// panicking in debug builds. The fix uses checked_add to return an error
  /// instead.
  #[test]
  fn read_at_rejects_position_delta_overflow() {
    use crate::util::varint::write_u32_var;
    use std::io::Cursor;
    // Create a minimal valid postings segment header
    let mut buf = Vec::new();
    buf.extend_from_slice(&1u32.to_le_bytes()); // doc_freq = 1 (little-endian u32)
    buf.push(1); // stored_positions = true
    buf.extend_from_slice(&0u32.to_le_bytes()); // raw_block (no block meta)
    buf.extend_from_slice(&0u32.to_le_bytes()); // max_doc_id
    buf.extend_from_slice(&0f32.to_le_bytes()); // max_tf

    // Add a single posting with two positions that overflow when accumulated
    // Position 1: delta = u32::MAX / 2 + 1 = 2_147_483_648
    // Position 2: delta = u32::MAX / 2 + 1 = 2_147_483_648
    // Sum would be 2^32 = 4_294_967_296 (u32::MAX + 1), which overflows u32
    let delta: u32 = u32::MAX / 2 + 1;
    write_u32_var(1, &mut buf); // doc_id = 1 (1 byte varint)
    write_u32_var(2, &mut buf); // term_freq = 2 (1 byte varint)
    write_u32_var(2, &mut buf); // position count = 2
    write_u32_var(delta, &mut buf); // first delta
    write_u32_var(delta, &mut buf); // second delta - causes overflow

    let mut cur = Cursor::new(buf);
    let err = PostingsReader::read_at(&mut cur, 0, true).expect_err("overflow must be rejected");
    let msg = err.to_string();
    assert!(
      msg.contains("position delta overflow while decoding postings"),
      "unexpected error message: {msg}"
    );
  }
}
