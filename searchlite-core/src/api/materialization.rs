use hashbrown::HashSet;
use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};

use crate::api::pagination::encode_search_after_token;
use crate::api::reader::{Hit, IndexReader, RankedHit};
use crate::api::types::{CollapseRequest, SearchRequest};
use crate::index::highlight::{highlight_fragments, HighlightOptions, make_snippet};
use crate::query::sort::SortPlan;

use super::phrase::normalize_phrase_terms;

impl IndexReader {
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn materialize_hit(
    &self,
    ranked: RankedHit,
    req: &SearchRequest,
    highlight_terms: &[String],
    phrase_terms: &BTreeMap<String, Vec<Vec<String>>>,
    sort_plan: &SortPlan,
    include_sort_key: bool,
    sort_key_json: Option<Vec<serde_json::Value>>,
  ) -> Option<Hit> {
    let seg = self.segments.get(ranked.key.segment_ord as usize)?;
    let doc_id_str = seg.doc_id(ranked.key.doc_id)?;
    let need_doc = req.return_stored || req.highlight_field.is_some() || req.highlight.is_some();

    // Fast path: when we don't need the stored document at all, skip docstore I/O
    // and return only the doc_id, score, and sort_key.
    if !need_doc {
      return Some(Hit {
        doc_id: doc_id_str.to_string(),
        score: ranked.score,
        vector_score: ranked.vector_score,
        sort_key: if include_sort_key {
          sort_key_json
            .or_else(|| encode_search_after_token(sort_plan, &ranked.key, &self.segments).ok())
        } else {
          None
        },
        fields: None,
        snippet: None,
        explanation: ranked.explanation,
        highlights: None,
        inner_hits: None,
      });
    }

    let doc_cache = seg.get_doc(ranked.key.doc_id).ok();

    let snippet = if let (Some(field), Some(doc)) = (&req.highlight_field, doc_cache.as_ref()) {
      if let Some(text_val) = doc.get(field).and_then(|v| v.as_str()) {
        let phrase_list = normalize_phrase_terms(
          phrase_terms.get(field).map(|v| v.as_slice()).unwrap_or(&[]),
          self.analysis.search_analyzer(field.as_str()),
        );
        make_snippet(text_val, highlight_terms, &phrase_list)
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

    let highlights = if let (Some(config), Some(doc)) = (req.highlight.as_ref(), doc_cache.as_ref())
    {
      let mut map = BTreeMap::new();
      for (field, opts) in config.fields.iter() {
        if let Some(text_val) = doc.get(field).and_then(|v| v.as_str()) {
          let terms: Vec<String> =
            if let Some(analyzer) = self.analysis.search_analyzer(field.as_str()) {
              let mut tokens = Vec::new();
              for term in highlight_terms {
                for tok in analyzer.analyze(term).into_iter() {
                  tokens.push(tok.text);
                }
              }
              let mut seen = HashSet::new();
              tokens
                .into_iter()
                .filter(|t| seen.insert(t.clone()))
                .collect()
            } else {
              highlight_terms.to_vec()
            };
          // If analysis strips everything (e.g., stopwords), keep the analyzed set (even if empty)
          // to avoid mixing analyzed and unanalyzed terms.
          let field_phrases = normalize_phrase_terms(
            phrase_terms.get(field).map(|v| v.as_slice()).unwrap_or(&[]),
            self.analysis.search_analyzer(field.as_str()),
          );
          let frags = highlight_fragments(
            text_val,
            &terms,
            &field_phrases,
            HighlightOptions {
              pre_tag: &opts.pre_tag,
              post_tag: &opts.post_tag,
              fragment_size: opts.fragment_size,
              number_of_fragments: opts.number_of_fragments,
            },
          );
          if !frags.is_empty() {
            map.insert(field.clone(), frags);
          }
        }
      }
      if map.is_empty() {
        None
      } else {
        Some(map)
      }
    } else {
      None
    };

    Some(Hit {
      doc_id: doc_id_str.to_string(),
      score: ranked.score,
      vector_score: ranked.vector_score,
      sort_key: if include_sort_key {
        sort_key_json
          .or_else(|| encode_search_after_token(sort_plan, &ranked.key, &self.segments).ok())
      } else {
        None
      },
      fields: fields_val,
      snippet,
      explanation: ranked.explanation,
      highlights,
      inner_hits: None,
    })
  }

  pub(crate) fn collapse_hits(
    &self,
    hits: Vec<RankedHit>,
    collapse: &CollapseRequest,
    sort_plan: &SortPlan,
  ) -> Result<Vec<(RankedHit, Vec<RankedHit>)>> {
    let mut groups: BTreeMap<String, Vec<RankedHit>> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();
    for hit in hits.into_iter() {
      let Some(key) = self.collapse_value(&hit, &collapse.field)? else {
        continue;
      };
      if !groups.contains_key(&key) {
        order.push(key.clone());
      }
      groups.entry(key).or_default().push(hit);
    }
    let inner_plan = if let Some(cfg) = collapse.inner_hits.as_ref() {
      SortPlan::from_request(&self.manifest.schema, &cfg.sort)
        .with_context(|| format!("invalid inner_hits sort for collapse on {}", collapse.field))?
    } else {
      sort_plan.clone()
    };
    let inner_from = collapse
      .inner_hits
      .as_ref()
      .and_then(|c| c.from)
      .unwrap_or(0);
    let mut out = Vec::with_capacity(order.len());
    let same_sort = inner_plan.hash() == sort_plan.hash();
    for key in order.into_iter() {
      if let Some(mut list) = groups.remove(&key) {
        // Ensure main-sort ordering for the representative hit.
        list.sort_by(|a, b| a.key.cmp(&b.key));
        let mut iter = list.into_iter();
        if let Some(top) = iter.next() {
          let mut inner: Vec<RankedHit> = iter.collect();
          if let Some(cfg) = collapse.inner_hits.as_ref() {
            if !inner.is_empty() && !same_sort {
              inner = self.resort_hits(&inner, &inner_plan)?;
            }
            if inner_from > 0 {
              if inner_from >= inner.len() {
                inner.clear();
              } else {
                inner.drain(0..inner_from);
              }
            }
            if let Some(size) = cfg.size {
              if size == 0 {
                inner.clear();
              } else if inner.len() > size {
                inner.truncate(size);
              }
            }
          } else {
            inner.clear();
          }
          out.push((top, inner));
        }
      }
    }
    Ok(out)
  }

  pub(crate) fn resort_hits(&self, hits: &[RankedHit], plan: &SortPlan) -> Result<Vec<RankedHit>> {
    let mut keyed = Vec::with_capacity(hits.len());
    for hit in hits.iter() {
      let seg = self
        .segments
        .get(hit.key.segment_ord as usize)
        .ok_or_else(|| anyhow::anyhow!("missing segment {}", hit.key.segment_ord))?;
      let key = plan.build_key(seg, hit.key.doc_id, hit.score, hit.key.segment_ord);
      keyed.push((key, hit.clone()));
    }
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(keyed.into_iter().map(|(_, hit)| hit).collect())
  }

  pub(crate) fn collapse_value(&self, hit: &RankedHit, field: &str) -> Result<Option<String>> {
    let seg = self
      .segments
      .get(hit.key.segment_ord as usize)
      .ok_or_else(|| anyhow::anyhow!("missing segment {}", hit.key.segment_ord))?;
    let values = seg.fast_fields().str_values(field, hit.key.doc_id);
    if values.is_empty() {
      return Ok(None);
    }
    if values.len() > 1 {
      let doc_id = seg.doc_id(hit.key.doc_id).unwrap_or("<unknown>");
      bail!(
        "collapse field `{field}` must be single-valued; document `{doc_id}` has {} values",
        values.len()
      );
    }
    Ok(values.first().map(|v| v.to_string()))
  }
}
