use rand::Rng;
use std::collections::BinaryHeap;
use std::io::{Read, Write};

use crate::core::types::{DistanceMetric, Scalar, VectorId};
use crate::index::flat::SearchResult;
use crate::math::distance::calculate_distance;
use crate::storage::buffer::SoABuffer;
use std::cmp::Reverse;
use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use std::fs::File;

/// The configuration parameters that dictate the shape of the HNSW graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Max connextions per layer
    pub m: usize,
    /// Max connections for layer 0
    pub m_max_zero: usize,
    /// Size of the priority queue during insertion
    pub ef_construction: usize,
    /// Probability multiplier for layer generation
    pub m_l: f32,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m_max_zero: m * 2,
            ef_construction: 100,
            m_l: 1.0 / (m as f32).ln(),
        }
    }
}

/// A node in our flattened graph.
#[derive(Debug, Serialize, Deserialize)]
pub struct HnswNode {
    /// The external VectorId assigned by the user
    pub id: VectorId,
    /// The highest layer this node exists on
    pub max_layer: usize,
    /// Adjacency List: connections[layer][neighbor_index]
    pub connections: Vec<Vec<usize>>,
}

impl HnswNode {
    pub fn new(id: VectorId, max_layer: usize) -> Self {
        let mut connections = Vec::with_capacity(max_layer + 1);
        for _ in 0..=max_layer {
            connections.push(Vec::new());
        }
        Self {
            id,
            max_layer,
            connections,
        }
    }
}

pub struct HnswIndex {
    buffer: SoABuffer,
    metric: DistanceMetric,
    config: HnswConfig,
    /// The contiguous array of all graph nodes.
    nodes: Vec<HnswNode>,
    /// The absolute top layer currently existing in the graph
    max_layer: usize,
    /// The index of the node that serves as the entry point to the graph
    entry_point: Option<usize>,
}

impl HnswIndex {
    pub fn new(
        dimension: usize,
        capacity: usize,
        metric: DistanceMetric,
        config: HnswConfig,
    ) -> Self {
        Self {
            buffer: SoABuffer::new(dimension, capacity),
            metric,
            config,
            nodes: Vec::with_capacity(capacity),
            max_layer: 0,
            entry_point: None,
        }
    }

    /// Generates a random layer based on the exponentially decaying probability.
    fn generate_random_layer(&self) -> usize {
        let mut rng = rand::thread_rng();
        let uniform: f32 = rng.gen_range(0.0000001..1.0);

        let r = -uniform.ln() * self.config.m_l;
        r.floor() as usize
    }

    /// Exposes the current node count
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Explores a single graph layer to find the 'ef' closest nodes to query.
    fn search_layer(
        &self,
        query: &[Scalar],
        entry_point: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<SearchResult> {
        let mut visited = HashSet::new();
        visited.insert(entry_point);

        let mut candidates: BinaryHeap<Reverse<SearchResult>> = BinaryHeap::new();
        let mut top_results: BinaryHeap<SearchResult> = BinaryHeap::new();

        let entry_vec = self.buffer.get_vector(entry_point).unwrap();
        let entry_dist = calculate_distance(query, entry_vec, self.metric);

        let initial_result = SearchResult {
            id: entry_point as u64,
            distance: entry_dist,
        };

        candidates.push(Reverse(initial_result));
        top_results.push(initial_result);

        while let Some(Reverse(current)) = candidates.pop() {
            if let Some(worst) = top_results.peek() {
                if current.distance > worst.distance && top_results.len() == ef {
                    break;
                }
            }

            let neighbors = &self.nodes[current.id as usize].connections[layer];

            for &neighbor_idx in neighbors {
                if visited.insert(neighbor_idx) {
                    let neighbor_vec = self.buffer.get_vector(neighbor_idx).unwrap();
                    let dist = calculate_distance(query, neighbor_vec, self.metric);

                    let result = SearchResult {
                        id: neighbor_idx as u64,
                        distance: dist,
                    };

                    if top_results.len() < ef {
                        top_results.push(result);
                        candidates.push(Reverse(result));
                    } else if let Some(mut worst) = top_results.peek_mut() {
                        if dist < worst.distance {
                            *worst = result;
                            candidates.push(Reverse(result));
                        }
                    }
                }
            }
        }
        top_results.into_vec()
    }
    /// Filters a list of candidates to ensure spatial diversity.
    fn select_neighbors(&self, mut candidates: Vec<SearchResult>, m: usize) -> Vec<usize> {
        candidates.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        let mut selected = Vec::with_capacity(m);

        for candidate in candidates {
            if selected.len() >= m {
                break;
            }

            let candidate_vec = self.buffer.get_vector(candidate.id as usize).unwrap();
            let mut is_good = true;

            for &selected_idx in &selected {
                let selected_vec = self.buffer.get_vector(selected_idx).unwrap();
                let dist_to_selected = calculate_distance(candidate_vec, selected_vec, self.metric);

                if dist_to_selected < candidate.distance {
                    is_good = false;
                    break;
                }
            }

            if is_good {
                selected.push(candidate.id as usize);
            }
        }

        selected
    }

    pub fn insert(
        &mut self,
        external_id: VectorId,
        vector: &[Scalar],
    ) -> crate::core::error::Result<()> {
        self.buffer.insert(external_id, vector)?;

        let internal_idx = self.nodes.len();
        let insert_layer = self.generate_random_layer();

        let mut new_node = HnswNode::new(external_id, insert_layer);

        if self.entry_point.is_none() {
            self.entry_point = Some(internal_idx);
            self.max_layer = insert_layer;
            self.nodes.push(new_node);
            return Ok(());
        }

        let mut curr_obj = self.entry_point.unwrap();
        let mut curr_layer = self.max_layer;

        while curr_layer > insert_layer {
            let candidates = self.search_layer(vector, curr_obj, 1, curr_layer);
            curr_obj = candidates[0].id as usize; // Move to the closest node found

            if curr_layer == 0 {
                break;
            }
            curr_layer -= 1;
        }

        let max_level_to_process = std::cmp::min(insert_layer, self.max_layer);

        for lc in (0..=max_level_to_process).rev() {
            let candidates = self.search_layer(vector, curr_obj, self.config.ef_construction, lc);
            curr_obj = candidates
                .iter()
                .min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap())
                .unwrap()
                .id as usize;

            let m_limit = if lc == 0 {
                self.config.m_max_zero
            } else {
                self.config.m
            };
            let neighbors = self.select_neighbors(candidates, m_limit);

            new_node.connections[lc] = neighbors.clone();

            for neighbor_idx in neighbors {
                let (needs_pruning, current_connections) = {
                    let neighbor = &mut self.nodes[neighbor_idx];
                    neighbor.connections[lc].push(internal_idx);

                    let conns = &neighbor.connections[lc];
                    (conns.len() > m_limit, conns.clone())
                };

                if needs_pruning {
                    let mut neighbor_candidates = Vec::with_capacity(current_connections.len());
                    let neighbor_vec = self.buffer.get_vector(neighbor_idx).unwrap();

                    for n_idx in current_connections {
                        let n_vec = self.buffer.get_vector(n_idx).unwrap();
                        let dist = calculate_distance(neighbor_vec, n_vec, self.metric);
                        neighbor_candidates.push(SearchResult {
                            id: n_idx as u64,
                            distance: dist,
                        });
                    }

                    let pruned_connections = self.select_neighbors(neighbor_candidates, m_limit);

                    self.nodes[neighbor_idx].connections[lc] = pruned_connections;
                }
            }
        }

        self.nodes.push(new_node);

        if insert_layer > self.max_layer {
            self.max_layer = insert_layer;
            self.entry_point = Some(internal_idx);
        }

        Ok(())
    }

    /// Searches the HNSW graph for the Top-K closest vectors to the query.
    pub fn search(&self, query: &[Scalar], k: usize) -> Vec<SearchResult> {
        if self.entry_point.is_none() || k == 0 {
            return Vec::new();
        }

        let mut curr_obj = self.entry_point.unwrap();
        let mut curr_layer = self.max_layer;

        while curr_layer > 0 {
            let candidates = self.search_layer(query, curr_obj, 1, curr_layer);
            curr_obj = candidates[0].id as usize;
            curr_layer -= 1;
        }
        let ef_search = std::cmp::max(k, 50);
        let mut candidates = self.search_layer(query, curr_obj, ef_search, 0);

        candidates.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        let mut final_results = Vec::with_capacity(k);
        for candidate in candidates.into_iter().take(k) {
            let internal_idx = candidate.id as usize;
            let external_id = self.nodes[internal_idx].id;

            final_results.push(SearchResult {
                id: external_id,
                distance: candidate.distance,
            });
        }

        final_results
    }
    /// Serializes the entire database into a binary file.
    pub fn save_to_disk(&self, path: &str) -> std::io::Result<()> {
        let mut file = File::create(path)?;

        let graph_meta = (
            &self.metric,
            &self.config,
            &self.nodes,
            &self.max_layer,
            &self.entry_point,
        );
        let encoded_graph = bincode::serialize(&graph_meta).unwrap();

        let graph_len = encoded_graph.len() as u64;
        file.write_all(&graph_len.to_le_bytes())?;
        file.write_all(&encoded_graph)?;

        self.buffer.save(&mut file)?;
        Ok(())
    }
    /// Loads the database from the binary file, restoring SIMD alignment.
    pub fn load_from_disk(path: &str) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut graph_len_buf = [0u8; 8];
        file.read_exact(&mut graph_len_buf)?;

        let graph_len = u64::from_le_bytes(graph_len_buf) as usize;

        let mut graph_buf = vec![0u8; graph_len];
        file.read_exact(&mut graph_buf)?;

        let (metric, config, nodes, max_layer, entry_point) =
            bincode::deserialize(&graph_buf).unwrap();
        let buffer = SoABuffer::load(&mut file)?;

        Ok(Self {
            buffer,
            metric,
            config,
            nodes,
            max_layer,
            entry_point,
        })
    }
}
