#[derive(Debug, Clone, Copy)]
pub struct BlockMaxEntry {
  pub doc_id: u32,
  pub score: f32,
}

pub fn top_k(mut entries: Vec<BlockMaxEntry>, k: usize) -> Vec<BlockMaxEntry> {
  entries.sort_by(|a, b| {
    b.score
      .partial_cmp(&a.score)
      .unwrap_or(std::cmp::Ordering::Equal)
  });
  entries.truncate(k);
  entries
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn sorts_and_truncates_entries() {
    let entries = vec![
      BlockMaxEntry {
        doc_id: 1,
        score: 0.3,
      },
      BlockMaxEntry {
        doc_id: 2,
        score: 0.9,
      },
      BlockMaxEntry {
        doc_id: 3,
        score: 0.5,
      },
    ];
    let top = top_k(entries, 2);
    assert_eq!(top.len(), 2);
    assert_eq!(top[0].doc_id, 2);
    assert_eq!(top[1].doc_id, 3);
  }
}
