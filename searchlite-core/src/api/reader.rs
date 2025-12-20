use hashbrown::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::api::query::parse_query;
use crate::api::types::{IndexOptions, SearchRequest};
use crate::index::highlight::make_snippet;
use crate::index::manifest::Manifest;
use crate::index::segment::SegmentReader;
use crate::index::InnerIndex;
use crate::query::bm25::bm25;
use crate::query::filters::passes_filters;
use crate::query::phrase::matches_phrase;
use crate::query::planner::{expand_not_terms, expand_terms};
use crate::DocId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hit {
  pub doc_id: DocId,
  pub score: f32,
  pub fields: Option<serde_json::Value>,
  pub snippet: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
  pub total_hits_estimate: u64,
  pub hits: Vec<Hit>,
}

pub struct IndexReader {
  pub manifest: Manifest,
  pub segments: Vec<SegmentReader>,
  options: IndexOptions,
}

impl IndexReader {
  pub(crate) fn open(inner: Arc<InnerIndex>) -> Result<Self> {
    let manifest = inner.manifest.read().clone();
    let mut segments = Vec::new();
    for seg in manifest.segments.iter() {
      segments.push(SegmentReader::open(
        inner.storage.clone(),
        seg.clone(),
        inner.options.enable_positions,
      )?);
    }
    Ok(Self {
      manifest,
      segments,
      options: IndexOptions {
        path: inner.path.clone(),
        create_if_missing: inner.options.create_if_missing,
        enable_positions: inner.options.enable_positions,
        bm25_k1: inner.options.bm25_k1,
        bm25_b: inner.options.bm25_b,
        storage: inner.options.storage.clone(),
        #[cfg(feature = "vectors")]
        vector_defaults: inner.options.vector_defaults.clone(),
      },
    })
  }

  pub fn search(&self, req: &SearchRequest) -> Result<SearchResult> {
    let parsed = parse_query(&req.query);
    let default_fields: Vec<String> = if let Some(fields) = &req.fields {
      fields.clone()
    } else {
      self
        .manifest
        .schema
        .text_fields
        .iter()
        .map(|f| f.name.clone())
        .collect()
    };
    let term_keys = expand_terms(&parsed, &default_fields);
    let not_terms = expand_not_terms(&parsed, &default_fields);

    let qualified_terms: Vec<(String, String, String)> = term_keys
      .iter()
      .map(|(field, term)| {
        let mut key = String::with_capacity(field.len() + term.len() + 1);
        key.push_str(field);
        key.push(':');
        key.push_str(term);
        (field.clone(), term.clone(), key)
      })
      .collect();
    let qualified_not_terms: Vec<String> = not_terms
      .iter()
      .map(|(field, term)| {
        let mut key = String::with_capacity(field.len() + term.len() + 1);
        key.push_str(field);
        key.push(':');
        key.push_str(term);
        key
      })
      .collect();
    let highlight_terms: Vec<String> = {
      let mut dedup = HashSet::new();
      let mut terms = Vec::new();
      for (_, term, _) in qualified_terms.iter() {
        if dedup.insert(term) {
          terms.push(term.clone());
        }
      }
      terms
    };

    let phrase_fields: Vec<Vec<(String, Vec<String>)>> = parsed
      .phrases
      .iter()
      .map(|phrase| {
        let fields = if let Some(f) = &phrase.field {
          vec![f.clone()]
        } else {
          default_fields.clone()
        };
        fields
          .into_iter()
          .map(|field| {
            let term_keys = phrase
              .terms
              .iter()
              .map(|term| {
                let mut key = String::with_capacity(field.len() + term.len() + 1);
                key.push_str(&field);
                key.push(':');
                key.push_str(term);
                key
              })
              .collect();
            (field, term_keys)
          })
          .collect()
      })
      .collect();

    let mut hits: Vec<Hit> = Vec::new();
    for seg in self.segments.iter() {
      let mut scores: HashMap<DocId, f32> = HashMap::new();
      for (field, _, key) in qualified_terms.iter() {
        if let Some(postings) = seg.postings(key) {
          let df = postings.len() as f32;
          let docs = seg.meta.doc_count as f32;
          let avgdl = seg.avg_field_length(field);
          for entry in postings.iter() {
            let doc_len = if avgdl > 0.0 {
              avgdl
            } else {
              entry.term_freq as f32
            };
            let score = bm25(
              entry.term_freq as f32,
              df,
              doc_len,
              avgdl,
              docs,
              self.options.bm25_k1,
              self.options.bm25_b,
            );
            *scores.entry(entry.doc_id).or_insert(0.0) += score;
          }
        }
      }

      // Remove docs that match NOT terms.
      for key in qualified_not_terms.iter() {
        if let Some(postings) = seg.postings(key) {
          for entry in postings.iter() {
            scores.remove(&entry.doc_id);
          }
        }
      }

      // Phrase filtering if requested.
      scores.retain(|doc_id, _| {
        for (phrase_idx, phrase) in parsed.phrases.iter().enumerate() {
          let mut ok_any_field = false;
          for (_field, term_keys) in phrase_fields[phrase_idx].iter() {
            let mut per_term = Vec::with_capacity(term_keys.len());
            for key in term_keys.iter() {
              if let Some(postings) = seg.postings(key) {
                per_term.push(postings.iter().cloned().collect::<Vec<_>>());
              }
            }
            if per_term.len() == phrase.terms.len() && matches_phrase(&per_term, *doc_id) {
              ok_any_field = true;
              break;
            }
          }
          if !ok_any_field {
            return false;
          }
        }
        true
      });

      // Filters.
      scores.retain(|doc_id, _| passes_filters(seg.fast_fields(), *doc_id, &req.filters));

      let need_doc = req.return_stored || req.highlight_field.is_some();

      for (doc_id, score) in scores.into_iter() {
        let mut doc_cache = None;
        if need_doc {
          doc_cache = seg.get_doc(doc_id).ok();
        }

        let snippet = if let (Some(field), Some(doc)) = (&req.highlight_field, doc_cache.as_ref()) {
          if let Some(text_val) = doc.get(field).and_then(|v| v.as_str()) {
            make_snippet(text_val, &highlight_terms)
          } else {
            None
          }
        } else {
          None
        };

        let fields_val = if req.return_stored {
          doc_cache.clone()
        } else {
          None
        };
        hits.push(Hit {
          doc_id,
          score,
          fields: fields_val,
          snippet,
        });
      }
    }

    hits.sort_by(|a, b| {
      b.score
        .partial_cmp(&a.score)
        .unwrap_or(std::cmp::Ordering::Equal)
    });
    if hits.len() > req.limit {
      hits.truncate(req.limit);
    }
    Ok(SearchResult {
      total_hits_estimate: hits.len() as u64,
      hits,
    })
  }
}
