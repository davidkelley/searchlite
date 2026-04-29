use searchlite_adapter_elastic::translate::schema_to_es;
use serde_json::json;

#[test]
fn text_field_maps_to_es_text_with_analyzer() {
  let schema = json!({
    "type": "object",
    "properties": {
      "title": {
        "type": "string",
        "searchlite:analyzer": "english",
      }
    }
  });
  let es = schema_to_es("books", &schema).unwrap();
  let mapping = es.get("books").unwrap().get("mappings").unwrap();
  let title = mapping.get("properties").unwrap().get("title").unwrap();
  assert_eq!(title.get("type").unwrap(), &json!("text"));
  assert_eq!(title.get("analyzer").unwrap(), &json!("english"));
}

#[test]
fn keyword_field_maps_to_es_keyword() {
  let schema = json!({
    "type": "object",
    "properties": {
      "tag": {
        "type": "string",
        "searchlite:kind": "keyword",
      }
    }
  });
  let es = schema_to_es("idx", &schema).unwrap();
  let tag = es
    .get("idx")
    .unwrap()
    .get("mappings")
    .unwrap()
    .get("properties")
    .unwrap()
    .get("tag")
    .unwrap();
  assert_eq!(tag.get("type").unwrap(), &json!("keyword"));
}

#[test]
fn integer_maps_to_long() {
  let schema = json!({
    "type": "object",
    "properties": {
      "count": { "type": "integer" }
    }
  });
  let es = schema_to_es("idx", &schema).unwrap();
  let count = es
    .get("idx")
    .unwrap()
    .get("mappings")
    .unwrap()
    .get("properties")
    .unwrap()
    .get("count")
    .unwrap();
  assert_eq!(count.get("type").unwrap(), &json!("long"));
}

#[test]
fn number_maps_to_double() {
  let schema = json!({
    "type": "object",
    "properties": {
      "price": { "type": "number" }
    }
  });
  let es = schema_to_es("idx", &schema).unwrap();
  let price = es
    .get("idx")
    .unwrap()
    .get("mappings")
    .unwrap()
    .get("properties")
    .unwrap()
    .get("price")
    .unwrap();
  assert_eq!(price.get("type").unwrap(), &json!("double"));
}

#[test]
fn nested_array_translates_to_es_nested_with_properties() {
  let schema = json!({
    "type": "object",
    "properties": {
      "comments": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "author": { "type": "string", "searchlite:kind": "keyword" },
            "votes": { "type": "integer" }
          }
        }
      }
    }
  });
  let es = schema_to_es("idx", &schema).unwrap();
  let comments = es
    .get("idx")
    .unwrap()
    .get("mappings")
    .unwrap()
    .get("properties")
    .unwrap()
    .get("comments")
    .unwrap();
  assert_eq!(comments.get("type").unwrap(), &json!("nested"));
  let inner = comments.get("properties").unwrap();
  assert_eq!(
    inner.get("author").unwrap().get("type").unwrap(),
    &json!("keyword")
  );
  assert_eq!(
    inner.get("votes").unwrap().get("type").unwrap(),
    &json!("long")
  );
}
