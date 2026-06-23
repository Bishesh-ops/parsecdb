use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::core::types::{Scalar, VectorId};
use crate::math::distance::cosine_similarity;
use crate::storage::buffer::SoABuffer;

#[derive(Debug, Clone, Copy)]
pub struct SearchResult {
    pub id: VectorId,
    pub score: f32,
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub struct FlatIndex {
    buffer: SoABuffer,
}

impl FlatIndex {
    pub fn new(dimension: usize, capacity: usize) -> Self {
        Self {
            buffer: SoABuffer::new(dimension, capacity),
        }
    }

    /// Performs a linear scan over the entire buffer, maintaining a Min-Heap
    /// to return the Top-K highest scoring vectors.
    pub fn search(&self, query: &[Scalar], k: usize) -> Vec<SearchResult> {
        if self.buffer.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut heap: BinaryHeap<Reverse<SearchResult>> = BinaryHeap::with_capacity(k);

        for i in 0..self.buffer.len() {
            let vec = self.buffer.get_vector(i).unwrap();
            let score = cosine_similarity(query, vec);

            let result = SearchResult {
                id: self.buffer.get_id(i).unwrap(),
                score,
            };

            if heap.len() < k {
                heap.push(Reverse(result));
            } else if let Some(mut min_of_top_k) = heap.peek_mut() {
                if score > min_of_top_k.0.score {
                    *min_of_top_k = Reverse(result);
                }
            }
        }

        let mut results: Vec<SearchResult> = heap.into_iter().map(|r| r.0).collect();
        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        results
    }
}
