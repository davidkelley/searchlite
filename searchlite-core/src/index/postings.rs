use std::io::{Read, Seek, SeekFrom, Write};

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

#[derive(Debug, Clone)]
pub struct PostingsReader {
  data: Vec<PostingEntry>,
  pub max_tf: f32,
  pub block_max_doc_ids: Vec<DocId>,
  pub block_max_tfs: Vec<f32>,
  pub block_size: usize,
}

impl PostingsReader {
  pub fn read_at<R: Read + Seek>(file: &mut R, offset: u64, keep_positions: bool) -> Result<Self> {
    file.seek(SeekFrom::Start(offset))?;
    let doc_freq = read_u32(file)? as usize;
    let has_positions = {
      let mut flag = [0u8; 1];
      file.read_exact(&mut flag)?;
      flag[0] == 1 && keep_positions
    };
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
      if has_positions {
        let count = read_u32_var(file)? as usize;
        let mut acc = 0u32;
        for _ in 0..count {
          acc += read_u32_var(file)?;
          positions.push(acc);
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
      block_max_doc_ids,
      block_max_tfs,
      block_size,
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

  #[cfg(test)]
  pub fn from_entries_for_test(entries: Vec<PostingEntry>, block_size: usize) -> Self {
    let mut reader = Self {
      max_tf: 0.0,
      block_max_doc_ids: Vec::new(),
      block_max_tfs: Vec::new(),
      block_size: block_size.max(1),
      data: entries,
    };
    reader.max_tf = reader
      .data
      .iter()
      .map(|p| p.term_freq as f32)
      .fold(0.0_f32, f32::max);
    for chunk in reader.data.chunks(reader.block_size) {
      if let Some(last) = chunk.last() {
        reader.block_max_doc_ids.push(last.doc_id);
      }
      let tf_max = chunk
        .iter()
        .map(|p| p.term_freq as f32)
        .fold(0.0_f32, f32::max);
      reader.block_max_tfs.push(tf_max);
    }
    reader
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
    assert_eq!(reader.block_max_doc_ids.len(), 1);
    assert_eq!(reader.block_max_tfs.len(), 1);
    let collected: Vec<_> = reader
      .iter()
      .map(|p| (p.doc_id, p.positions.iter().copied().collect::<Vec<_>>()))
      .collect();
    assert_eq!(collected, vec![(1, vec![1, 3]), (2, vec![4])]);
  }
}
