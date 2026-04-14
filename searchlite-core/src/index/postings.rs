use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

use anyhow::Result;
use hashbrown::HashMap;
use smallvec::SmallVec;

use crate::index::codec::{read_f32, read_u32, write_f32, write_u32};
use crate::util::varint::{read_u32_var, write_u32_var};
use crate::DocId;

pub const DEFAULT_BLOCK_SIZE: usize = 128;
const BLOCK_META_FLAG: u32 = 1u32 << 31;

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
        let mut prev = 0;
        for pos in p.positions.iter().copied() {
          let delta = pos - prev;
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
      block_max_doc_ids.reserve(block_count);
      block_max_tfs.reserve(block_count);
      for _ in 0..block_count {
        block_max_doc_ids.push(read_u32(file)?);
      }
      for _ in 0..block_count {
        block_max_tfs.push(read_f32(file)?);
      }
    }
    let mut data = Vec::with_capacity(doc_freq);
    for _ in 0..doc_freq {
      let doc_id = read_u32_var(file)?;
      let term_freq = read_u32_var(file)?;
      let mut positions = SmallVec::new();
      if stored_positions {
        let count = read_u32_var(file)? as usize;
        let mut acc = 0u32;
        for _ in 0..count {
          acc += read_u32_var(file)?;
          if keep_in_memory {
            positions.push(acc);
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
}
