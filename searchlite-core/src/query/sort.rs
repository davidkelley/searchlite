use anyhow::{bail, Context, Result};
use crc32fast::Hasher;
use smallvec::SmallVec;

use crate::api::types::{SortOrder, SortSpec};
use crate::index::manifest::{FieldKind, Schema};
use crate::index::segment::SegmentReader;
use crate::DocId;

#[derive(Clone, Debug)]
enum SortField {
  Score,
  Keyword(String),
  I64(String),
  F64(String),
}

#[derive(Clone, Copy, Debug)]
enum ValueSelector {
  Min,
  Max,
}

impl From<SortOrder> for ValueSelector {
  fn from(order: SortOrder) -> Self {
    match order {
      SortOrder::Asc => ValueSelector::Min,
      SortOrder::Desc => ValueSelector::Max,
    }
  }
}

#[derive(Clone, Debug)]
struct ResolvedSortField {
  field: SortField,
  order: SortOrder,
  selector: ValueSelector,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SortValue {
  Score(f32),
  I64(i64),
  F64(f64),
  Str(String),
  Missing,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SortKeyPart {
  pub order: SortOrder,
  pub value: SortValue,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SortKey {
  pub parts: SmallVec<[SortKeyPart; 4]>,
  pub segment_ord: u32,
  pub doc_id: DocId,
}

impl SortKey {
  pub fn score_bits(&self) -> Option<u32> {
    if let Some(SortValue::Score(score)) = self.parts.first().map(|p| &p.value) {
      Some(score.to_bits())
    } else {
      None
    }
  }
}

impl Eq for SortKey {}

impl PartialOrd for SortKey {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    Some(self.cmp(other))
  }
}

impl Ord for SortKey {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    for (a, b) in self.parts.iter().zip(other.parts.iter()) {
      let ord = a.cmp(b);
      if !ord.is_eq() {
        return ord;
      }
    }
    self
      .segment_ord
      .cmp(&other.segment_ord)
      .then_with(|| self.doc_id.cmp(&other.doc_id))
  }
}

impl SortKeyPart {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    match (&self.value, &other.value) {
      (SortValue::Missing, SortValue::Missing) => std::cmp::Ordering::Equal,
      (SortValue::Missing, _) => std::cmp::Ordering::Greater,
      (_, SortValue::Missing) => std::cmp::Ordering::Less,
      (SortValue::Score(a), SortValue::Score(b)) => compare_f32(*a, *b, self.order),
      (SortValue::I64(a), SortValue::I64(b)) => compare_ord(*a, *b, self.order),
      (SortValue::F64(a), SortValue::F64(b)) => compare_f64(*a, *b, self.order),
      (SortValue::Str(a), SortValue::Str(b)) => compare_ord(a, b, self.order),
      _ => std::cmp::Ordering::Equal,
    }
  }
}

fn compare_ord<T: Ord>(a: T, b: T, order: SortOrder) -> std::cmp::Ordering {
  match a.cmp(&b) {
    std::cmp::Ordering::Less => match order {
      SortOrder::Asc => std::cmp::Ordering::Less,
      SortOrder::Desc => std::cmp::Ordering::Greater,
    },
    std::cmp::Ordering::Greater => match order {
      SortOrder::Asc => std::cmp::Ordering::Greater,
      SortOrder::Desc => std::cmp::Ordering::Less,
    },
    std::cmp::Ordering::Equal => std::cmp::Ordering::Equal,
  }
}

fn compare_f32(a: f32, b: f32, order: SortOrder) -> std::cmp::Ordering {
  match a.total_cmp(&b) {
    std::cmp::Ordering::Less => match order {
      SortOrder::Asc => std::cmp::Ordering::Less,
      SortOrder::Desc => std::cmp::Ordering::Greater,
    },
    std::cmp::Ordering::Greater => match order {
      SortOrder::Asc => std::cmp::Ordering::Greater,
      SortOrder::Desc => std::cmp::Ordering::Less,
    },
    std::cmp::Ordering::Equal => std::cmp::Ordering::Equal,
  }
}

fn compare_f64(a: f64, b: f64, order: SortOrder) -> std::cmp::Ordering {
  match a.total_cmp(&b) {
    std::cmp::Ordering::Less => match order {
      SortOrder::Asc => std::cmp::Ordering::Less,
      SortOrder::Desc => std::cmp::Ordering::Greater,
    },
    std::cmp::Ordering::Greater => match order {
      SortOrder::Asc => std::cmp::Ordering::Greater,
      SortOrder::Desc => std::cmp::Ordering::Less,
    },
    std::cmp::Ordering::Equal => std::cmp::Ordering::Equal,
  }
}

#[derive(Clone, Debug)]
pub struct SortPlan {
  fields: Vec<ResolvedSortField>,
  hash: u32,
}

impl SortPlan {
  pub fn from_request(schema: &Schema, specs: &[SortSpec]) -> Result<Self> {
    let resolved_specs: Vec<SortSpec> = if specs.is_empty() {
      vec![SortSpec {
        field: "_score".to_string(),
        order: None,
      }]
    } else {
      specs.to_vec()
    };
    let mut fields = Vec::with_capacity(resolved_specs.len());
    for spec in resolved_specs.into_iter() {
      let order = spec
        .order
        .unwrap_or_else(|| default_order_for_field(&spec.field));
      if spec.field == "_score" {
        fields.push(ResolvedSortField {
          field: SortField::Score,
          order,
          selector: ValueSelector::from(order),
        });
        continue;
      }
      let meta = schema
        .field_meta(&spec.field)
        .with_context(|| format!("unknown sort field `{}`", spec.field))?;
      match meta.kind {
        FieldKind::Keyword => {
          if !meta.fast {
            bail!("sort field `{}` must be marked as fast", spec.field);
          }
          fields.push(ResolvedSortField {
            field: SortField::Keyword(spec.field),
            order,
            selector: ValueSelector::from(order),
          });
        }
        FieldKind::Numeric => {
          if !meta.fast {
            bail!("sort field `{}` must be marked as fast", spec.field);
          }
          let is_i64 = meta.numeric_i64.unwrap_or(false);
          fields.push(ResolvedSortField {
            field: if is_i64 {
              SortField::I64(spec.field)
            } else {
              SortField::F64(spec.field)
            },
            order,
            selector: ValueSelector::from(order),
          });
        }
        _ => bail!(
          "sort field `{}` must be a fast keyword or numeric field",
          spec.field
        ),
      }
    }
    let hash = compute_hash(&fields);
    Ok(Self { fields, hash })
  }

  pub fn is_score_only(&self) -> bool {
    self.fields.len() == 1 && matches!(self.fields[0].field, SortField::Score)
  }

  pub fn hash(&self) -> u32 {
    self.hash
  }

  pub fn primary_order(&self) -> Option<SortOrder> {
    self.fields.first().map(|f| f.order)
  }

  pub fn uses_score(&self) -> bool {
    self
      .fields
      .iter()
      .any(|f| matches!(f.field, SortField::Score))
  }

  pub fn build_key(
    &self,
    segment: &SegmentReader,
    doc_id: DocId,
    score: f32,
    segment_ord: u32,
  ) -> SortKey {
    let mut parts: SmallVec<[SortKeyPart; 4]> = SmallVec::with_capacity(self.fields.len());
    for field in self.fields.iter() {
      let value = field.value(segment, doc_id, score);
      parts.push(SortKeyPart {
        order: field.order,
        value,
      });
    }
    SortKey {
      parts,
      segment_ord,
      doc_id,
    }
  }

  pub fn values_from_key(&self, key: &SortKey) -> Result<Vec<SortValue>> {
    if key.parts.len() != self.fields.len() {
      bail!(
        "cursor sort key length {} does not match plan {}",
        key.parts.len(),
        self.fields.len()
      );
    }
    Ok(key.parts.iter().map(|p| p.value.clone()).collect())
  }

  pub fn key_from_values(
    &self,
    values: &[SortValue],
    segment_ord: u32,
    doc_id: DocId,
  ) -> Result<SortKey> {
    if values.len() != self.fields.len() {
      bail!(
        "cursor contained {} sort values but plan expects {}",
        values.len(),
        self.fields.len()
      );
    }
    let mut parts: SmallVec<[SortKeyPart; 4]> = SmallVec::with_capacity(values.len());
    for (field, value) in self.fields.iter().zip(values.iter()) {
      parts.push(SortKeyPart {
        order: field.order,
        value: value.clone(),
      });
    }
    Ok(SortKey {
      parts,
      segment_ord,
      doc_id,
    })
  }
}

impl ResolvedSortField {
  fn value(&self, segment: &SegmentReader, doc_id: DocId, score: f32) -> SortValue {
    match &self.field {
      SortField::Score => SortValue::Score(score),
      SortField::Keyword(field) => {
        let values = segment.fast_fields().str_values(field, doc_id);
        let selected = match self.selector {
          ValueSelector::Min => values.iter().min().copied(),
          ValueSelector::Max => values.iter().max().copied(),
        };
        selected
          .map(|value| SortValue::Str(value.to_owned()))
          .unwrap_or(SortValue::Missing)
      }
      SortField::I64(field) => {
        let values = segment.fast_fields().i64_values(field, doc_id);
        pick_numeric(values, self.selector)
      }
      SortField::F64(field) => {
        let values = segment.fast_fields().f64_values(field, doc_id);
        pick_numeric(values, self.selector)
      }
    }
  }
}

fn pick_numeric<T>(values: Vec<T>, selector: ValueSelector) -> SortValue
where
  T: Into<SortValue> + Copy + PartialOrd,
{
  if values.is_empty() {
    return SortValue::Missing;
  }
  match selector {
    ValueSelector::Min => values
      .into_iter()
      .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
      .map(|v| v.into())
      .unwrap_or(SortValue::Missing),
    ValueSelector::Max => values
      .into_iter()
      .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
      .map(|v| v.into())
      .unwrap_or(SortValue::Missing),
  }
}

impl From<i64> for SortValue {
  fn from(value: i64) -> Self {
    SortValue::I64(value)
  }
}

impl From<f64> for SortValue {
  fn from(value: f64) -> Self {
    SortValue::F64(value)
  }
}

fn compute_hash(fields: &[ResolvedSortField]) -> u32 {
  let mut hasher = Hasher::new();
  for field in fields.iter() {
    let mut kind_byte = [0u8];
    match &field.field {
      SortField::Score => {
        kind_byte[0] = 0;
        hasher.update(&kind_byte);
      }
      SortField::Keyword(name) => {
        kind_byte[0] = 1;
        hasher.update(&kind_byte);
        hasher.update(name.as_bytes());
      }
      SortField::I64(name) => {
        kind_byte[0] = 2;
        hasher.update(&kind_byte);
        hasher.update(name.as_bytes());
      }
      SortField::F64(name) => {
        kind_byte[0] = 3;
        hasher.update(&kind_byte);
        hasher.update(name.as_bytes());
      }
    }
    hasher.update(&[match field.order {
      SortOrder::Asc => 0,
      SortOrder::Desc => 1,
    }]);
  }
  hasher.finalize()
}

fn default_order_for_field(field: &str) -> SortOrder {
  if field == "_score" {
    SortOrder::Desc
  } else {
    SortOrder::Asc
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::index::manifest::{KeywordField, NumericField, Schema, TextField};
  use smallvec::smallvec;

  #[test]
  fn sort_key_orders_with_direction() {
    let key_asc = SortKey {
      parts: smallvec![SortKeyPart {
        order: SortOrder::Asc,
        value: SortValue::I64(1),
      }],
      segment_ord: 0,
      doc_id: 0,
    };
    let mut key_desc = key_asc.clone();
    key_desc.parts[0].order = SortOrder::Desc;
    let higher_asc = SortKey {
      parts: smallvec![SortKeyPart {
        order: SortOrder::Asc,
        value: SortValue::I64(2),
      }],
      segment_ord: 0,
      doc_id: 1,
    };
    let higher_desc = SortKey {
      parts: smallvec![SortKeyPart {
        order: SortOrder::Desc,
        value: SortValue::I64(2),
      }],
      segment_ord: 0,
      doc_id: 1,
    };
    assert!(higher_asc > key_asc);
    assert!(higher_desc < key_desc);
  }

  #[test]
  fn sort_plan_defaults_score_desc() {
    let schema = Schema {
      doc_id_field: crate::index::manifest::default_doc_id_field(),
      analyzers: Vec::new(),
      text_fields: vec![TextField {
        name: "body".into(),
        analyzer: "default".into(),
        search_analyzer: None,
        stored: true,
        indexed: true,
        nullable: false,
      }],
      keyword_fields: vec![KeywordField {
        name: "tag".into(),
        stored: true,
        indexed: true,
        fast: true,
        nullable: false,
      }],
      numeric_fields: vec![NumericField {
        name: "year".into(),
        i64: true,
        fast: true,
        stored: false,
        nullable: false,
      }],
      nested_fields: Vec::new(),
      #[cfg(feature = "vectors")]
      vector_fields: Vec::new(),
    };
    let plan = SortPlan::from_request(&schema, &[]).unwrap();
    assert!(plan.is_score_only());
    assert!(plan.hash() != 0);
  }
}
