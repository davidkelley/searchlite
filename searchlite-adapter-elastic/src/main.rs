use clap::Parser;
use searchlite_adapter_elastic::{init_tracing, run, AdapterArgs};
use tracing::error;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  init_tracing();
  let args = AdapterArgs::parse();
  if let Err(err) = run(args).await {
    error!("{err:?}");
    std::process::exit(1);
  }
  Ok(())
}
