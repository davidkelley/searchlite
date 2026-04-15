use hashbrown::HashMap;
use std::collections::BTreeMap;

use smallvec::SmallVec;

use super::term_expansion::build_term_key;
use crate::analysis::analyzer::Analyzer;
use crate::index::manifest::{FieldKind, Schema, SchemaAnalyzers};
use crate::index::postings::PostingEntry;
use crate::index::segment::SegmentReader;
use crate::query::planner::PhraseSpec;
use crate::util::case_fold::fold_keyword;
use crate::DocId;

#[derive(Clone, Debug)]
pub(crate) struct TermMatchGroup {
  pub(crate) keys: Vec<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct PhraseFieldConfig {
  pub(crate) slop: u32,
  pub(crate) fields: Vec<(String, Vec<Vec<String>>)>,
}

#[derive(Clone, Debug)]
pub(crate) struct PhraseRuntime {
  pub(crate) slop: u32,
  pub(crate) variants: Vec<Vec<Vec<PostingEntry>>>,
}

pub(crate) struct TermDocLists {
  pub(crate) lists: Vec<Vec<DocId>>,
  pub(crate) group_lists: Vec<Vec<usize>>,
}

pub(crate) fn expand_phrase_fields(
  phrase_specs: &[PhraseSpec],
  analysis: &SchemaAnalyzers,
  schema: &Schema,
) -> Vec<PhraseFieldConfig> {
  phrase_specs
    .iter()
    .map(|phrase| {
      let fields = phrase
        .fields
        .iter()
        .filter_map(|field| match schema.field_kind(field) {
          FieldKind::Text => analysis.search_analyzer(field).and_then(|analyzer| {
            let phrase_body = phrase.terms.join(" ");
            let tokens = analyzer.analyze(&phrase_body);
            if tokens.is_empty() {
              return None;
            }
            let mut positions: Vec<Vec<String>> = Vec::new();
            for token in tokens {
              let pos = token.position as usize;
              if positions.len() <= pos {
                positions.resize(pos + 1, Vec::new());
              }
              if !positions[pos].contains(&token.text) {
                positions[pos].push(token.text);
              }
            }
            Some((field.clone(), positions))
          }),
          FieldKind::Keyword => {
            let joined = fold_keyword(&phrase.terms.join(" ")).into_owned();
            if joined.is_empty() {
              None
            } else {
              Some((field.clone(), vec![vec![joined]]))
            }
          }
          FieldKind::Numeric | FieldKind::Unknown => None,
        })
        .collect();
      PhraseFieldConfig {
        slop: phrase.slop,
        fields,
      }
    })
    .collect()
}

pub(crate) fn normalize_phrase_terms(
  phrases: &[Vec<String>],
  analyzer: Option<&Analyzer>,
) -> Vec<Vec<String>> {
  if let Some(analyzer) = analyzer {
    let mut out = Vec::new();
    for phrase in phrases.iter() {
      let mut seq = Vec::new();
      for term in phrase.iter() {
        for tok in analyzer.analyze(term) {
          seq.push(tok.text);
        }
      }
      if !seq.is_empty() {
        out.push(seq);
      }
    }
    if !out.is_empty() {
      return out;
    }
  }
  phrases.to_vec()
}

pub(crate) fn build_phrase_term_map(
  phrase_specs: &[PhraseSpec],
) -> BTreeMap<String, Vec<Vec<String>>> {
  let mut out = BTreeMap::new();
  for phrase in phrase_specs.iter() {
    for field in phrase.fields.iter() {
      out
        .entry(field.clone())
        .or_insert_with(Vec::new)
        .push(phrase.terms.clone());
    }
  }
  out
}

pub(crate) fn build_phrase_runtimes(
  seg: &SegmentReader,
  phrase_fields: &[PhraseFieldConfig],
) -> Vec<PhraseRuntime> {
  phrase_fields
    .iter()
    .map(|config| {
      let variants = config
        .fields
        .iter()
        .filter_map(|(field, positions)| {
          let mut per_position: Vec<Vec<PostingEntry>> = Vec::new();
          for alternatives in positions.iter() {
            let mut lists = Vec::new();
            for term in alternatives.iter() {
              let key = build_term_key(field, term);
              if let Some(posts) = seg.postings(&key) {
                lists.push(posts.iter().cloned().collect());
              }
            }
            if lists.is_empty() {
              return None;
            }
            per_position.push(merge_postings_lists(lists));
          }
          Some(per_position)
        })
        .collect::<Vec<Vec<Vec<PostingEntry>>>>();
      PhraseRuntime {
        slop: config.slop,
        variants,
      }
    })
    .collect()
}

pub(crate) fn build_term_doc_lists(
  seg: &SegmentReader,
  term_groups: &[TermMatchGroup],
) -> TermDocLists {
  let mut lists = Vec::new();
  let mut indices: HashMap<String, usize> = HashMap::new();
  let mut group_lists = Vec::with_capacity(term_groups.len());
  for group in term_groups.iter() {
    let mut group_indices = Vec::new();
    for key in group.keys.iter() {
      let idx = if let Some(idx) = indices.get(key) {
        *idx
      } else {
        let docs = seg
          .postings(key)
          .map(|p| p.iter().map(|e| e.doc_id).collect())
          .unwrap_or_default();
        let idx = lists.len();
        lists.push(docs);
        indices.insert(key.clone(), idx);
        idx
      };
      group_indices.push(idx);
    }
    group_lists.push(group_indices);
  }
  TermDocLists { lists, group_lists }
}

pub(crate) fn merge_postings_lists(lists: Vec<Vec<PostingEntry>>) -> Vec<PostingEntry> {
  let mut merged: HashMap<DocId, PostingEntry> = HashMap::new();
  for list in lists.into_iter() {
    for entry in list.into_iter() {
      let positions = entry.positions;
      let doc_id = entry.doc_id;
      let target = merged.entry(doc_id).or_insert_with(|| PostingEntry {
        doc_id,
        term_freq: 0,
        positions: SmallVec::new(),
      });
      target.positions.extend(positions.into_iter());
    }
  }
  let mut values: Vec<_> = merged.into_values().collect();
  for entry in values.iter_mut() {
    entry.positions.sort_unstable();
    entry.positions.dedup();
    entry.term_freq = entry.positions.len() as u32;
  }
  values.sort_by_key(|e| e.doc_id);
  values
}
