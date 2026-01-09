use clap::Parser;
use searchlite_http::{init_tracing, run, ServeArgs};
use tracing::error;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  init_tracing();
  let args = ServeArgs::parse();
  if let Err(err) = run(args).await {
    error!("{err:?}");
    std::process::exit(1);
  }
  Ok(())
}
