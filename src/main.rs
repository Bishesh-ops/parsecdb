// src/main.rs
use parsecdb::core::types::DistanceMetric;
use parsecdb::index::hnsw::{HnswConfig, HnswIndex};

#[tokio::main]
async fn main() {
    println!("Starting ParsecDB...");

    let dim = 8;
    let capacity = 100_000;
    let config = HnswConfig::default();

    let index = HnswIndex::new(dim, capacity, DistanceMetric::Cosine, config);

    parsecdb::api::server::start(index, 8000).await;
}
