use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::core::types::{DistanceMetric, Scalar, VectorId};
use crate::math::distance::calculate_distance;
use crate::storage::buffer::SoABuffer;

/// Represents a scored vector result from a search query.
#[derive(Debug, Clone, Copy)]
pub struct SearchResult {
    pub id: VectorId,
    pub distance: f32,
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub struct FlatIndex {
    buffer: SoABuffer,
    metric: DistanceMetric,
}

impl FlatIndex {
    pub fn new(dimension: usize, capacity: usize, metric: DistanceMetric) -> Self {
        Self {
            buffer: SoABuffer::new(dimension, capacity),
            metric,
        }
    }

    pub fn insert(&mut self, id: VectorId, vector: &[Scalar]) -> crate::core::error::Result<()> {
        self.buffer.insert(id, vector)
    }

    pub fn search(&self, query: &[Scalar], k: usize) -> Vec<SearchResult> {
        if self.buffer.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut heap: BinaryHeap<SearchResult> = BinaryHeap::with_capacity(k);

        for i in 0..self.buffer.len() {
            let vec = self.buffer.get_vector(i).unwrap();

            let distance = calculate_distance(query, vec, self.metric);

            let result = SearchResult {
                id: self.buffer.get_id(i).unwrap(),
                distance,
            };

            if heap.len() < k {
                heap.push(result);
            } else if let Some(mut worst_of_top_k) = heap.peek_mut() {
                if distance < worst_of_top_k.distance {
                    *worst_of_top_k = result;
                }
            }
        }

        let mut results = heap.into_vec();

        results.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        results
    }
}
