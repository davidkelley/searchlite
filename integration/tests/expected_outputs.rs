mod common;

use anyhow::Result;
use tempfile::tempdir;

use integration::assertions::assert_normalized_search_parity;
use integration::fixtures::load_example_fixtures;
use integration::surfaces::core::CoreHarness;
use integration::surfaces::http::HttpHarness;
use integration::surfaces::SurfaceHarness;

#[test]
fn expected_outputs_match_between_core_and_http() -> Result<()> {
  let fixtures = load_example_fixtures()?;
  let searchlite_bin = common::searchlite_bin();

  for (dataset_name, dataset) in fixtures.datasets {
    let dir = tempdir()?;
    let mut core = CoreHarness::new(dir.path().join(format!("idx-core-{dataset_name:?}")));
    let mut http = HttpHarness::new(
      searchlite_bin.clone(),
      dir.path().join(format!("idx-http-{dataset_name:?}")),
    )?;

    let schema_json = serde_json::to_value(&dataset.schema)?;
    let ndjson = common::docs_to_ndjson(&dataset.seed_docs);

    core.init(&schema_json)?;
    core.add_ndjson(ndjson.as_str())?;
    core.commit()?;

    http.init(&schema_json)?;
    http.add_ndjson(ndjson.as_str())?;
    http.commit()?;

    for query in dataset.queries {
      let request = serde_json::to_value(&query.request)?;
      let core_body = core.search(&request);
      let http_body = http.search(&request);
      match (core_body, http_body) {
        (Ok(core_body), Ok(http_body)) => {
          assert_normalized_search_parity(
            &core_body,
            &http_body,
            dataset_name,
            query.name.as_str(),
          )?;
        }
        (Err(_), Err(_)) => {
          // Some fixture queries intentionally exercise unsupported combinations.
        }
        (Err(err), Ok(_)) => {
          anyhow::bail!(
            "core failed but http succeeded for {dataset_name:?}/{}: {err}",
            query.name
          );
        }
        (Ok(_), Err(err)) => {
          anyhow::bail!(
            "http failed but core succeeded for {dataset_name:?}/{}: {err}",
            query.name
          );
        }
      }
    }
  }

  Ok(())
}
