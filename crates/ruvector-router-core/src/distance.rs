//! Distance calculations for the router's vector metrics.
//!
//! The default path is scalar. The loops below are manually unrolled eight
//! elements at a time, which helps the autovectorizer but does not emit vector
//! instructions on its own; nothing in this module has ever called SimSIMD,
//! despite what this file's header used to say.
//!
//! With the `lattice-simd` feature, Euclidean, cosine, and dot product route
//! through `lattice-embed`'s runtime-dispatched kernels (AVX-512F, AVX2, NEON,
//! wasm32 SIMD128, each with its own scalar fallback). Manhattan stays scalar:
//! lattice-embed 0.7.0 has no L1 kernel to route it to.
//!
//! Both paths return the same values. Every metric keeps the sign and
//! similarity-to-distance conversion this module already defined, and the
//! degenerate-input branches are unchanged.

use crate::error::{Result, VectorDbError};
use crate::types::DistanceMetric;

/// Calculate distance between two vectors using specified metric
pub fn calculate_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> Result<f32> {
    if a.len() != b.len() {
        return Err(VectorDbError::InvalidDimensions {
            expected: a.len(),
            actual: b.len(),
        });
    }

    match metric {
        DistanceMetric::Euclidean => Ok(euclidean_distance(a, b)),
        DistanceMetric::Cosine => Ok(cosine_similarity(a, b)),
        DistanceMetric::DotProduct => Ok(dot_product(a, b)),
        DistanceMetric::Manhattan => Ok(manhattan_distance(a, b)),
    }
}

/// Euclidean distance (L2).
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "lattice-simd")]
    {
        // Equal lengths only. The scalar path below indexes `b` by `a`'s
        // length and panics on a short `b`, where lattice returns f32::MAX;
        // routing only equal lengths keeps enabling the feature from turning
        // a panic into a silent value.
        if a.len() == b.len() {
            return lattice_embed::simd::euclidean_distance(a, b);
        }
    }

    euclidean_distance_scalar(a, b)
}

#[inline]
fn euclidean_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    // Process in chunks for better SIMD utilization
    let len = a.len();
    let mut i = 0;

    // Main loop - process 8 elements at a time for AVX2
    while i + 8 <= len {
        for j in 0..8 {
            let diff = a[i + j] - b[i + j];
            sum += diff * diff;
        }
        i += 8;
    }

    // Handle remaining elements
    while i < len {
        let diff = a[i] - b[i];
        sum += diff * diff;
        i += 1;
    }

    sum.sqrt()
}

/// Cosine distance.
/// Returns 1 - cosine_similarity to convert similarity to distance
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "lattice-simd")]
    {
        // lattice returns 0.0 for a zero-magnitude operand, which lands on the
        // same 1.0 this function's own zero check returns, so the degenerate
        // case needs no special handling here.
        if a.len() == b.len() {
            return 1.0 - lattice_embed::simd::cosine_similarity(a, b);
        }
    }

    cosine_similarity_scalar(a, b)
}

#[inline]
fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    let len = a.len();
    let mut i = 0;

    // Process in chunks
    while i + 8 <= len {
        for j in 0..8 {
            let ai = a[i + j];
            let bi = b[i + j];
            dot += ai * bi;
            norm_a += ai * ai;
            norm_b += bi * bi;
        }
        i += 8;
    }

    // Handle remaining
    while i < len {
        let ai = a[i];
        let bi = b[i];
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
        i += 1;
    }

    let norm_a = norm_a.sqrt();
    let norm_b = norm_b.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0; // Maximum distance
    }

    // Convert similarity to distance
    1.0 - (dot / (norm_a * norm_b))
}

/// Dot product, negated so that a larger similarity is a smaller distance.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "lattice-simd")]
    {
        // The negation stays here rather than in the backend, so both paths
        // agree on the similarity-to-distance convention.
        if a.len() == b.len() {
            return -lattice_embed::simd::dot_product(a, b);
        }
    }

    dot_product_scalar(a, b)
}

#[inline]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    let len = a.len();
    let mut i = 0;

    // Process in chunks
    while i + 8 <= len {
        for j in 0..8 {
            sum += a[i + j] * b[i + j];
        }
        i += 8;
    }

    // Handle remaining
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }

    -sum // Negate to convert similarity to distance
}

/// Manhattan distance (L1).
///
/// Not routed through lattice: lattice-embed 0.7.0 exposes no L1 kernel.
#[inline]
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    let len = a.len();
    let mut i = 0;

    // Process in chunks
    while i + 8 <= len {
        for j in 0..8 {
            sum += (a[i + j] - b[i + j]).abs();
        }
        i += 8;
    }

    // Handle remaining
    while i < len {
        sum += (a[i] - b[i]).abs();
        i += 1;
    }

    sum
}

/// Batch distance calculation for multiple queries
pub fn batch_distance(
    query: &[f32],
    vectors: &[Vec<f32>],
    metric: DistanceMetric,
) -> Result<Vec<f32>> {
    use rayon::prelude::*;

    vectors
        .par_iter()
        .map(|v| calculate_distance(query, v, metric))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dist = euclidean_distance(&a, &b);
        assert!((dist - 5.196).abs() < 0.01);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 0.01); // Same vectors = distance 0
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dot = dot_product(&a, &b);
        assert!((dot - (-32.0)).abs() < 0.01); // Negated
    }

    #[test]
    fn test_manhattan_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dist = manhattan_distance(&a, &b);
        assert!((dist - 9.0).abs() < 0.01);
    }

    fn pair(dim: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
        let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        (
            (0..dim).map(|_| next()).collect(),
            (0..dim).map(|_| next()).collect(),
        )
    }

    fn reference_euclidean(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = f64::from(*x) - f64::from(*y);
            acc += d * d;
        }
        acc.sqrt() as f32
    }

    fn reference_cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += f64::from(*x) * f64::from(*y);
            na += f64::from(*x) * f64::from(*x);
            nb += f64::from(*y) * f64::from(*y);
        }
        if na == 0.0 || nb == 0.0 {
            return 1.0;
        }
        (1.0 - dot / (na.sqrt() * nb.sqrt())) as f32
    }

    fn reference_dot(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            acc += f64::from(*x) * f64::from(*y);
        }
        -acc as f32
    }

    fn reference_manhattan(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            acc += (f64::from(*x) - f64::from(*y)).abs();
        }
        acc as f32
    }

    /// Whichever backend is compiled must agree with an f64 reference.
    ///
    /// Dimensions straddle the manual 8-wide chunk boundary and the 4/8/16-lane
    /// widths a SIMD backend uses, so both remainder paths are exercised rather
    /// than assumed. Sign and the similarity-to-distance conversion are part of
    /// what is compared, not just magnitude.
    #[test]
    fn backends_match_reference() {
        for dim in [
            1usize, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 384, 768,
        ] {
            for seed in 0..4u32 {
                let (a, b) = pair(dim, seed);

                for (name, got, want) in [
                    (
                        "euclidean",
                        euclidean_distance(&a, &b),
                        reference_euclidean(&a, &b),
                    ),
                    (
                        "cosine",
                        cosine_similarity(&a, &b),
                        reference_cosine(&a, &b),
                    ),
                    ("dot", dot_product(&a, &b), reference_dot(&a, &b)),
                    (
                        "manhattan",
                        manhattan_distance(&a, &b),
                        reference_manhattan(&a, &b),
                    ),
                ] {
                    assert!(
                        (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                        "{name} dim={dim} seed={seed}: {got} vs {want}"
                    );
                }
            }
        }
    }

    /// A zero-magnitude operand must give maximum cosine distance, not NaN.
    #[test]
    fn cosine_zero_vector_is_max_distance() {
        let zero = vec![0.0f32; 64];
        let (v, _) = pair(64, 21);
        assert_eq!(cosine_similarity(&zero, &v), 1.0);
        assert_eq!(cosine_similarity(&v, &zero), 1.0);
        assert_eq!(cosine_similarity(&zero, &zero), 1.0);
    }

    /// Dimension mismatch is rejected before any metric runs.
    #[test]
    fn calculate_distance_rejects_mismatched_dimensions() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.0f32, 2.0];
        for metric in [
            DistanceMetric::Euclidean,
            DistanceMetric::Cosine,
            DistanceMetric::DotProduct,
            DistanceMetric::Manhattan,
        ] {
            assert!(calculate_distance(&a, &b, metric).is_err(), "{metric:?}");
        }
    }
}
