use std::sync::Arc;

use crate::client::SearchliteClient;
use crate::AdapterArgs;

#[derive(Clone)]
pub struct AppState {
  client: Arc<SearchliteClient>,
  args: AdapterArgs,
}

impl AppState {
  pub fn new(client: Arc<SearchliteClient>, args: AdapterArgs) -> Self {
    Self { client, args }
  }

  pub fn client(&self) -> &SearchliteClient {
    &self.client
  }

  pub fn args(&self) -> &AdapterArgs {
    &self.args
  }
}
