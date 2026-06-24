use crate::core::types::DistanceMetric;
use crate::core::types::Scalar;

#[cfg(target_arch = "x86_64")]
use super::simd;

/// Calculaes the Cosine Similarity between two normalized vectors.
/// Higher values mean more similar.
pub fn cosine_similarity(a: &[Scalar], b: &[Scalar]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must be of equal length");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd::dot_product_avx2(a, b) };
        }
    }

    fallback_dot_product(a, b)
}
/// A standard iterator-based dot product calculation.
fn fallback_dot_product(a: &[Scalar], b: &[Scalar]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Helper function to normalize a vector in-place.
/// Required before inserting into the database for our Cosine optimisation to work.
pub fn normalize_in_place(vector: &mut [Scalar]) {
    let magnitude_sq: f32 = vector.iter().map(|v| v * v).sum();
    if magnitude_sq > 0.0 {
        let magnitude = magnitude_sq.sqrt();
        for v in vector.iter_mut() {
            *v /= magnitude;
        }
    }
}

/// Calculates Squared Eculidean Distance (L2^2).
/// Lower values mean the vectors are closer together.
pub fn euclidean_squared(a: &[Scalar], b: &[Scalar]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must be of equal length");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd::l2_squared_avx2(a, b) };
        }
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

#[inline(always)]
pub fn calculate_distance(a: &[Scalar], b: &[Scalar], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine => {
            let similarity = cosine_similarity(a, b);
            1.0 - similarity
        }
        DistanceMetric::L2Squared => euclidean_squared(a, b),
    }
}
