pub type VectorId = u64;
pub type Scalar = f32;

/// Defines the mathematical space used to calculate distances between vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Cosine distance
    /// Best for normalized embeddings.
    Cosine,

    /// Squared Euclidean distance.
    /// Best for raw spatial data or computer vison embeddings.
    L2Squared,
}
