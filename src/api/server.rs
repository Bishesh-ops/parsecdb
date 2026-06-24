use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::api::protocol::{
    InsertRequest, SearchRequest, SearchResponse, SearchResultItem, StatusResponse,
};
use crate::index::hnsw::HnswIndex;

/// The shared state accessible by all HTTP worker threads
struct AppState {
    db: Arc<RwLock<HnswIndex>>,
}

/// Starts the Axum web server and binds it to the given port.
pub async fn start(index: HnswIndex, port: u16) {
    let shared_state = Arc::new(AppState {
        db: Arc::new(RwLock::new(index)),
    });

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/insert", post(handle_insert))
        .route("/search", post(handle_search))
        .route("/save", post(handle_save))
        .with_state(shared_state);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    println!("ParsecDB API Server listening on http://{}", addr);

    let listner = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listner, app.into_make_service()).await.unwrap();
}

/// Simple health check to verify the server is running.
async fn health_check() -> Json<StatusResponse> {
    Json(StatusResponse {
        status: "success".to_string(),
        message: "ParsecDB is alive and running".to_string(),
    })
}

/// Handles incoming vectors, normalizes them, and inserts them into the graph.
async fn handle_insert(
    State(state): State<Arc<AppState>>,
    Json(mut payload): Json<InsertRequest>,
) -> Result<Json<StatusResponse>, (StatusCode, String)> {
    crate::math::distance::normalize_in_place(&mut payload.vector);

    let mut db = state.db.write().await;

    match db.insert(payload.id, &payload.vector) {
        Ok(_) => Ok(Json(StatusResponse {
            status: "success".to_string(),
            message: format!("Inserted vector ID {}", payload.id),
        })),
        Err(e) => Err((StatusCode::BAD_REQUEST, e.to_string())),
    }
}

/// Performs a highly concurrent read-only search of the graph.
async fn handle_search(
    State(state): State<Arc<AppState>>,
    Json(mut payload): Json<SearchRequest>,
) -> Json<SearchResponse> {
    crate::math::distance::normalize_in_place(&mut payload.vector);

    let db = state.db.read().await;

    let results = db.search(&payload.vector, payload.k);

    let mapped_results = results
        .into_iter()
        .map(|r| SearchResultItem {
            id: r.id,
            distance: r.distance,
        })
        .collect();

    Json(SearchResponse {
        results: mapped_results,
    })
}

/// Triggers a binary memory dump to the SSD.
async fn handle_save(
    State(state): State<Arc<AppState>>,
) -> Result<Json<StatusResponse>, (StatusCode, String)> {
    let db = state.db.read().await;

    match db.save_to_disk("parsecdb_data.bin") {
        Ok(_) => Ok(Json(StatusResponse {
            status: "success".to_string(),
            message: "Database successfully flushed to disk".to_string(),
        })),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}
