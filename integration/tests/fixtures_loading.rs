use std::collections::BTreeMap;

use integration::fixtures::{load_example_fixtures, DatasetName};

#[test]
fn fixtures_loading_parses_example_datasets() {
  let fixtures = load_example_fixtures().expect("load fixtures");
  let mut expected = BTreeMap::new();
  expected.insert(DatasetName::Recipes, 300usize);
  expected.insert(DatasetName::VideoGames, 1500usize);

  for (dataset, line_count) in expected {
    let loaded = fixtures.datasets.get(&dataset).expect("dataset present");
    assert_eq!(loaded.seed_docs.len(), line_count);
    assert!(!loaded.queries.is_empty(), "queries should be loaded");
  }
}

#[test]
fn fixtures_loading_parses_queries_as_search_requests() {
  let fixtures = load_example_fixtures().expect("load fixtures");
  let recipes = fixtures
    .datasets
    .get(&DatasetName::Recipes)
    .expect("recipes fixture");
  let sample = recipes
    .queries
    .iter()
    .find(|q| q.name == "fuzzy-weeknight-orzo")
    .expect("query fixture");

  assert_eq!(sample.request.limit, 5);
  assert!(sample.raw.get("fuzzy").is_some());
}

#[test]
fn fixtures_loading_derives_mutation_payloads() {
  let fixtures = load_example_fixtures().expect("load fixtures");
  let recipes = fixtures
    .datasets
    .get(&DatasetName::Recipes)
    .expect("recipes fixture");

  assert!(!recipes.mutations.insert_docs.is_empty());
  assert!(!recipes.mutations.update_docs.is_empty());
  assert!(!recipes.mutations.delete_ids.is_empty());
  assert!(!recipes.mutations.mget_ids.is_empty());
  assert!(recipes
    .mutations
    .mget_ids
    .iter()
    .any(|id| id == "missing-doc-id"));
}
