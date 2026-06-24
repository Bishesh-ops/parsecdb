use serde::{Deserialize, Serialize};

/// JSON Payload for inserting a new vector
#[derive(Debug, Deserialize)]
pub struct InsertRequest {
    pub id: u64,
    pub vector: Vec<f32>,
}

/// JSON Payload for querying the database
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    pub k: usize,
}

/// JSON Payload returned after a search
#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultItem>,
}

/// A single matched vector in the search response
#[derive(Debug, Serialize)]
pub struct SearchResultItem {
    pub id: u64,
    pub distance: f32,
}

/// Generic success/failure message
#[derive(Debug, Serialize)]
pub struct StatusResponse {
    pub status: String,
    pub message: String,
}
