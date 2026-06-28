use parsecdb::core::types::DistanceMetric;
use parsecdb::index::hnsw::{HnswConfig, HnswIndex};

#[tokio::main]
async fn main() {
    println!("🚀 Starting ParsecDB...");

    let dim = 8;
    let capacity = 100_000;
    let config = HnswConfig::default();
    let wal_path = "parsecdb.wal";

    let index = HnswIndex::boot(dim, capacity, DistanceMetric::Cosine, config, wal_path)
        .expect("Failed to boot database from WAL");

    parsecdb::api::server::start(index, 8000).await;
}
