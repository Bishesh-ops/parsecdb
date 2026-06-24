#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Calculates the dot product of two aligned f32 slices using AVX2.
///
/// # Safety
/// 1. The CPU executing this code MUST support the AVX2 feature set.
/// 2. `a` and `b` must be perfectly 32-byte aligned in memory (which our SoABuffer guarantees).
/// 3. `a` and `b` must be the exact same length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let mut result: f32;

    unsafe {
        let mut sum_v = _mm256_setzero_ps();

        for i in 0..chunks {
            let offset = i * 8;

            let a_v = _mm256_loadu_ps(a.as_ptr().add(offset));
            let b_v = _mm256_loadu_ps(b.as_ptr().add(offset));

            let prod_v = _mm256_mul_ps(a_v, b_v);
            sum_v = _mm256_add_ps(sum_v, prod_v);
        }

        let sum_128 = _mm_add_ps(
            _mm256_castps256_ps128(sum_v),
            _mm256_extractf128_ps(sum_v, 1),
        );

        let sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
        let sum_32 = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 0x55));

        result = _mm_cvtss_f32(sum_32);
    }

    for i in (chunks * 8)..len {
        result += a[i] * b[i];
    }

    result
}

/// Calculates the Squared Euclidean Distance (L2^2) using AVX2.
///
/// # Safety
/// Requires AVX2 CPU support and 32-byte aligned arrays of equal length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let mut result: f32;

    unsafe {
        let mut sum_v = _mm256_setzero_ps();

        for i in 0..chunks {
            let offset = i * 8;

            let a_v = _mm256_loadu_ps(a.as_ptr().add(offset));
            let b_v = _mm256_loadu_ps(b.as_ptr().add(offset));

            let diff_v = _mm256_sub_ps(a_v, b_v);
            let sq_v = _mm256_mul_ps(diff_v, diff_v);
            sum_v = _mm256_add_ps(sum_v, sq_v);
        }

        let sum_128 = _mm_add_ps(
            _mm256_castps256_ps128(sum_v),
            _mm256_extractf128_ps(sum_v, 1),
        );
        let sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
        let sum_32 = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 0x55));

        result = _mm_cvtss_f32(sum_32);
    }
    for i in (chunks * 8)..len {
        let diff = a[i] - b[i];
        result += diff * diff;
    }
    result
}
