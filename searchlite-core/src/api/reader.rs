use hashbrown::{HashMap, HashSet};
use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::api::query::parse_query;
use crate::api::types::{IndexOptions, SearchRequest};
use crate::index::highlight::make_snippet;
use crate::index::manifest::Manifest;
use crate::index::postings::PostingEntry;
use crate::index::segment::SegmentReader;
use crate::index::InnerIndex;
use crate::query::facets::{AggregationOutput, AggregationPlan, AggregationState, FacetPlan};
use crate::query::filters::passes_filters;
use crate::query::phrase::matches_phrase;
use crate::query::planner::{expand_not_terms, expand_terms};
use crate::query::wand::{execute_top_k, ScoredTerm};
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
  pub facets: Vec<FacetResult>,
  pub aggregations: Vec<AggregationOutput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetResult {
  pub field: String,
  pub counts: BTreeMap<String, u64>,
}

struct SegmentSearchOutput {
  hits: Vec<Hit>,
  facets: Option<Vec<BTreeMap<String, u64>>>,
  aggregations: Option<Vec<AggregationState>>,
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

    let facet_plan = FacetPlan::new(&req.facets, &self.manifest.schema);
    let agg_plan = AggregationPlan::new(&req.aggregations, &self.manifest.schema);

    let mut facet_totals = facet_plan.as_ref().map(|p| vec![BTreeMap::new(); p.len()]);
    let mut agg_totals: Option<Vec<AggregationState>> = None;

    let mut hits: Vec<Hit> = Vec::new();
    for seg in self.segments.iter() {
      let seg_out = self.search_segment(
        seg,
        &qualified_terms,
        &qualified_not_terms,
        &phrase_fields,
        &highlight_terms,
        req,
        facet_plan.as_ref(),
        agg_plan.as_ref(),
      )?;
      hits.extend(seg_out.hits);
      if let (Some(global), Some(mut local)) = (facet_totals.as_mut(), seg_out.facets) {
        for (g, l) in global.iter_mut().zip(local.drain(..)) {
          for (k, v) in l.into_iter() {
            *g.entry(k).or_insert(0) += v;
          }
        }
      }
      if let (Some(global), Some(local)) = (agg_totals.as_mut(), seg_out.aggregations) {
        for (g, l) in global.iter_mut().zip(local.iter()) {
          g.merge_from(l);
        }
      } else if agg_totals.is_none() {
        agg_totals = seg_out.aggregations;
      }
    }

    hits.sort_by(|a, b| {
      b.score
        .total_cmp(&a.score)
        .then_with(|| a.doc_id.cmp(&b.doc_id))
    });
    if hits.len() > req.limit {
      hits.truncate(req.limit);
    }
    let facets = if let (Some(plan), Some(counts)) = (facet_plan, facet_totals) {
      counts
        .into_iter()
        .zip(plan.configs())
        .map(|(counts, cfg)| FacetResult {
          field: cfg.field_name().to_string(),
          counts,
        })
        .collect()
    } else {
      Vec::new()
    };
    let aggregations = agg_totals
      .unwrap_or_default()
      .into_iter()
      .map(|s| s.finalize())
      .collect();
    Ok(SearchResult {
      total_hits_estimate: hits.len() as u64,
      hits,
      facets,
      aggregations,
    })
  }

  fn search_segment(
    &self,
    seg: &SegmentReader,
    qualified_terms: &[(String, String, String)],
    qualified_not_terms: &[String],
    phrase_fields: &[Vec<(String, Vec<String>)>],
    highlight_terms: &[String],
    req: &SearchRequest,
    facet_plan: Option<&FacetPlan>,
    agg_plan: Option<&AggregationPlan>,
  ) -> Result<SegmentSearchOutput> {
    let mut term_counts: HashMap<String, (String, u32)> = HashMap::new();
    for (field, _, key) in qualified_terms.iter() {
      let entry = term_counts.entry(key.clone()).or_insert((field.clone(), 0));
      entry.1 += 1;
    }

    let docs = seg.meta.doc_count as f32;
    let mut terms: Vec<ScoredTerm> = Vec::new();
    for (key, (field, weight)) in term_counts.into_iter() {
      if let Some(postings) = seg.postings(&key) {
        terms.push(ScoredTerm {
          postings,
          weight,
          avgdl: seg.avg_field_length(&field),
          docs,
          k1: self.options.bm25_k1,
          b: self.options.bm25_b,
        });
      }
    }
    if terms.is_empty() {
      return Ok(SegmentSearchOutput {
        hits: Vec::new(),
        facets: facet_plan.map(|p| vec![BTreeMap::new(); p.len()]),
        aggregations: agg_plan.map(|p| p.collector(seg.fast_fields()).into_states()),
      });
    }

    let not_doc_lists: Vec<Vec<DocId>> = qualified_not_terms
      .iter()
      .filter_map(|key| {
        seg
          .postings(key)
          .map(|p| p.iter().map(|e| e.doc_id).collect())
      })
      .collect();

    let phrase_postings: Vec<Vec<Vec<Vec<PostingEntry>>>> = phrase_fields
      .iter()
      .map(|fields| {
        fields
          .iter()
          .filter_map(|(_field, term_keys)| {
            let per_term: Vec<Vec<PostingEntry>> = term_keys
              .iter()
              .filter_map(|key| seg.postings(key).map(|p| p.iter().cloned().collect()))
              .collect();
            if per_term.len() == term_keys.len() {
              Some(per_term)
            } else {
              None
            }
          })
          .collect::<Vec<Vec<Vec<PostingEntry>>>>()
      })
      .collect();

    let mut facet_collector = facet_plan.map(|p| p.collector(seg.fast_fields()));
    let mut agg_collector = agg_plan.map(|p| p.collector(seg.fast_fields()));

    let mut accept = |doc_id: DocId, _score: f32| -> bool {
      if !passes_filters(seg.fast_fields(), doc_id, &req.filters) {
        return false;
      }
      for list in not_doc_lists.iter() {
        if list.binary_search(&doc_id).is_ok() {
          return false;
        }
      }
      for variants in phrase_postings.iter() {
        if variants.is_empty() {
          return false;
        }
        let mut ok_any_field = false;
        for per_term in variants.iter() {
          if matches_phrase(per_term.as_slice(), doc_id) {
            ok_any_field = true;
            break;
          }
        }
        if !ok_any_field {
          return false;
        }
      }
      if let Some(facets) = facet_collector.as_mut() {
        facets.collect(doc_id);
      }
      if let Some(aggs) = agg_collector.as_mut() {
        aggs.collect(doc_id);
      }
      true
    };

    let ranked = execute_top_k(
      terms,
      req.limit,
      req.execution.clone(),
      req.bmw_block_size,
      &mut accept,
    );

    let need_doc = req.return_stored || req.highlight_field.is_some();
    let mut hits = Vec::with_capacity(ranked.len());
    for rd in ranked.into_iter() {
      let mut doc_cache = None;
      if need_doc {
        doc_cache = seg.get_doc(rd.doc_id).ok();
      }

      let snippet = if let (Some(field), Some(doc)) = (&req.highlight_field, doc_cache.as_ref()) {
        if let Some(text_val) = doc.get(field).and_then(|v| v.as_str()) {
          make_snippet(text_val, highlight_terms)
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
        doc_id: rd.doc_id,
        score: rd.score,
        fields: fields_val,
        snippet,
      });
    }
    let facets = facet_collector.map(|c| c.into_counts());
    let aggregations = agg_collector.map(|c| c.into_states());
    Ok(SegmentSearchOutput {
      hits,
      facets,
      aggregations,
    })
  }
}
