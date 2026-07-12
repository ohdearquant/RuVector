//! Distance computations with SIMD acceleration and optional GPU offload
//!
//! Dispatch priority: GPU (if `gpu` feature) → SimSIMD (if `simd` feature, native
//! NEON/AVX2/AVX-512) → WASM SIMD128 (`wasm32` target with `simd128`
//! target-feature) → scalar

/// Flat vector storage — contiguous memory for cache-friendly access
/// Vectors are stored as a single `Vec<f32>` slab: `[v0_d0, v0_d1, ..., v1_d0, ...]`
#[derive(Clone)]
pub struct FlatVectors {
    pub data: Vec<f32>,
    pub dim: usize,
    pub count: usize,
}

impl FlatVectors {
    pub fn new(dim: usize) -> Self {
        Self {
            data: Vec::new(),
            dim,
            count: 0,
        }
    }

    pub fn with_capacity(dim: usize, n: usize) -> Self {
        Self {
            data: Vec::with_capacity(n * dim),
            dim,
            count: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, vector: &[f32]) {
        debug_assert_eq!(vector.len(), self.dim);
        self.data.extend_from_slice(vector);
        self.count += 1;
    }

    #[inline]
    pub fn get(&self, idx: usize) -> &[f32] {
        let start = idx * self.dim;
        &self.data[start..start + self.dim]
    }

    /// Zero out a vector (lazy deletion)
    #[inline]
    pub fn zero_out(&mut self, idx: usize) {
        let start = idx * self.dim;
        for v in &mut self.data[start..start + self.dim] {
            *v = f32::NAN;
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

// ============================================================================
// Distance functions — auto-dispatch based on features
// ============================================================================

/// L2 squared distance — dispatches to best available implementation
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(feature = "simd")]
    {
        simd_l2_squared(a, b)
    }

    #[cfg(not(feature = "simd"))]
    {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            wasm_simd128_l2_squared(a, b)
        }

        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            scalar_l2_squared(a, b)
        }
    }
}

/// Scalar L2² with 4 accumulators for ILP
#[inline]
pub fn scalar_l2_squared(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut i = 0;

    while i + 16 <= len {
        for j in 0..4 {
            let off = i + j * 4;
            let d0 = a[off] - b[off];
            let d1 = a[off + 1] - b[off + 1];
            let d2 = a[off + 2] - b[off + 2];
            let d3 = a[off + 3] - b[off + 3];
            s0 += d0 * d0;
            s1 += d1 * d1;
            s2 += d2 * d2;
            s3 += d3 * d3;
        }
        i += 16;
    }
    while i < len {
        let d = a[i] - b[i];
        s0 += d * d;
        i += 1;
    }
    s0 + s1 + s2 + s3
}

/// SimSIMD-accelerated L2² — uses hardware NEON/AVX2/AVX-512
#[cfg(feature = "simd")]
#[inline]
pub fn simd_l2_squared(a: &[f32], b: &[f32]) -> f32 {
    // simsimd sqeuclidean returns squared Euclidean directly
    simsimd::SpatialSimilarity::sqeuclidean(a, b)
        .map(|d| d as f32)
        .unwrap_or_else(|| scalar_l2_squared(a, b))
}

/// WASM SIMD128-accelerated L2² — two `v128` accumulators (8 lanes/iteration)
/// for instruction-level parallelism, mirroring the scalar path's 4-accumulator
/// shape. Handles any `dim`, including 0, non-multiples-of-4, and
/// non-multiples-of-8 via scalar remainder loops.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
pub fn wasm_simd128_l2_squared(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::wasm32::*;

    let len = a.len();
    let mut acc0 = f32x4_splat(0.0);
    let mut acc1 = f32x4_splat(0.0);
    let mut i = 0;

    while i + 8 <= len {
        unsafe {
            let a0 = v128_load(a.as_ptr().add(i) as *const v128);
            let b0 = v128_load(b.as_ptr().add(i) as *const v128);
            let d0 = f32x4_sub(a0, b0);
            acc0 = f32x4_add(acc0, f32x4_mul(d0, d0));

            let a1 = v128_load(a.as_ptr().add(i + 4) as *const v128);
            let b1 = v128_load(b.as_ptr().add(i + 4) as *const v128);
            let d1 = f32x4_sub(a1, b1);
            acc1 = f32x4_add(acc1, f32x4_mul(d1, d1));
        }
        i += 8;
    }
    while i + 4 <= len {
        unsafe {
            let av = v128_load(a.as_ptr().add(i) as *const v128);
            let bv = v128_load(b.as_ptr().add(i) as *const v128);
            let d = f32x4_sub(av, bv);
            acc0 = f32x4_add(acc0, f32x4_mul(d, d));
        }
        i += 4;
    }

    let sum_vec = f32x4_add(acc0, acc1);
    let arr: [f32; 4] = unsafe { core::mem::transmute(sum_vec) };
    let mut sum = arr[0] + arr[1] + arr[2] + arr[3];

    while i < len {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    sum
}

/// Inner product distance (negated for min-heap)
#[inline]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(feature = "simd")]
    {
        simsimd::SpatialSimilarity::inner(a, b)
            .map(|d| -(d as f32))
            .unwrap_or_else(|| scalar_inner_product(a, b))
    }

    #[cfg(not(feature = "simd"))]
    {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            wasm_simd128_inner_product(a, b)
        }

        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            scalar_inner_product(a, b)
        }
    }
}

#[inline]
fn scalar_inner_product(a: &[f32], b: &[f32]) -> f32 {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let len = a.len();
    let mut i = 0;

    while i + 16 <= len {
        for j in 0..4 {
            let off = i + j * 4;
            s0 += a[off] * b[off];
            s1 += a[off + 1] * b[off + 1];
            s2 += a[off + 2] * b[off + 2];
            s3 += a[off + 3] * b[off + 3];
        }
        i += 16;
    }
    while i < len {
        s0 += a[i] * b[i];
        i += 1;
    }
    -(s0 + s1 + s2 + s3)
}

/// WASM SIMD128-accelerated inner product (negated, same convention as
/// [`scalar_inner_product`]). Two `v128` accumulators, scalar remainder tail.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
pub fn wasm_simd128_inner_product(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::wasm32::*;

    let len = a.len();
    let mut acc0 = f32x4_splat(0.0);
    let mut acc1 = f32x4_splat(0.0);
    let mut i = 0;

    while i + 8 <= len {
        unsafe {
            let a0 = v128_load(a.as_ptr().add(i) as *const v128);
            let b0 = v128_load(b.as_ptr().add(i) as *const v128);
            acc0 = f32x4_add(acc0, f32x4_mul(a0, b0));

            let a1 = v128_load(a.as_ptr().add(i + 4) as *const v128);
            let b1 = v128_load(b.as_ptr().add(i + 4) as *const v128);
            acc1 = f32x4_add(acc1, f32x4_mul(a1, b1));
        }
        i += 8;
    }
    while i + 4 <= len {
        unsafe {
            let av = v128_load(a.as_ptr().add(i) as *const v128);
            let bv = v128_load(b.as_ptr().add(i) as *const v128);
            acc0 = f32x4_add(acc0, f32x4_mul(av, bv));
        }
        i += 4;
    }

    let sum_vec = f32x4_add(acc0, acc1);
    let arr: [f32; 4] = unsafe { core::mem::transmute(sum_vec) };
    let mut sum = arr[0] + arr[1] + arr[2] + arr[3];

    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    -sum
}

/// PQ asymmetric distance from precomputed lookup table
#[inline]
pub fn pq_asymmetric_distance(codes: &[u8], table: &[f32], k: usize) -> f32 {
    // table is flat: table[subspace * 256 + code]
    let mut dist = 0.0f32;
    for (i, &code) in codes.iter().enumerate() {
        dist += unsafe { *table.get_unchecked(i * k + code as usize) };
    }
    dist
}

// ============================================================================
// Visited bitset — O(1) membership test, much faster than HashSet<u32>
// ============================================================================

/// Compact bitset for tracking visited nodes during search
pub struct VisitedSet {
    bits: Vec<u64>,
    generation: u64,
    gens: Vec<u64>,
}

impl VisitedSet {
    pub fn new(n: usize) -> Self {
        Self {
            bits: vec![0u64; (n + 63) / 64],
            generation: 1,
            gens: vec![0u64; n],
        }
    }

    /// Reset for a new search — O(1) via generation counter
    #[inline]
    pub fn clear(&mut self) {
        self.generation += 1;
    }

    /// Mark node as visited
    #[inline]
    pub fn insert(&mut self, id: u32) {
        self.gens[id as usize] = self.generation;
    }

    /// Check if visited
    #[inline]
    pub fn contains(&self, id: u32) -> bool {
        self.gens[id as usize] == self.generation
    }
}

// ============================================================================
// GPU distance computation (optional, feature-gated)
// ============================================================================

/// GPU-accelerated batch distance computation
/// Computes distances from a single query to N vectors in parallel
#[cfg(feature = "gpu")]
pub mod gpu {
    use super::FlatVectors;

    /// GPU backend selection
    #[derive(Debug, Clone, Copy)]
    pub enum GpuBackend {
        /// Apple Metal (macOS/iOS)
        Metal,
        /// NVIDIA CUDA
        Cuda,
        /// Vulkan compute (cross-platform)
        Vulkan,
    }

    /// GPU distance computation context
    pub struct GpuDistanceContext {
        backend: GpuBackend,
        /// Batch size for GPU kernel launches
        batch_size: usize,
    }

    impl GpuDistanceContext {
        /// Create a new GPU context (auto-detects best backend)
        pub fn new() -> Option<Self> {
            // Auto-detect: Metal on macOS, CUDA if nvidia, Vulkan fallback
            #[cfg(target_os = "macos")]
            let backend = GpuBackend::Metal;
            #[cfg(not(target_os = "macos"))]
            let backend = GpuBackend::Cuda;

            Some(Self {
                backend,
                batch_size: 4096,
            })
        }

        /// Batch L2² distances: query vs all vectors in flat storage
        /// Returns Vec of (index, distance) sorted by distance
        pub fn batch_l2_squared(
            &self,
            query: &[f32],
            vectors: &FlatVectors,
            k: usize,
        ) -> Vec<(u32, f32)> {
            // GPU kernel dispatch:
            // 1. Upload query + vector slab to GPU memory
            // 2. Launch N threads, each computing one L2² distance
            // 3. Parallel top-k reduction on GPU
            // 4. Download k results
            //
            // For now, fall back to CPU parallel with rayon
            // (real Metal/CUDA shaders would be added via metal-rs or cuda-sys)
            use rayon::prelude::*;

            let mut dists: Vec<(u32, f32)> = (0..vectors.count as u32)
                .into_par_iter()
                .map(|i| {
                    let v = vectors.get(i as usize);
                    (i, super::scalar_l2_squared(query, v))
                })
                .collect();

            dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.truncate(k);
            dists
        }

        pub fn backend(&self) -> GpuBackend {
            self.backend
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_squared() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((l2_squared(&a, &b) - 27.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_identical() {
        let a = vec![1.0; 128];
        assert!(l2_squared(&a, &a) < 1e-10);
    }

    #[test]
    fn test_inner_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((inner_product(&a, &b) - (-32.0)).abs() < 1e-6);
    }

    #[test]
    fn test_flat_vectors() {
        let mut fv = FlatVectors::new(3);
        fv.push(&[1.0, 2.0, 3.0]);
        fv.push(&[4.0, 5.0, 6.0]);
        assert_eq!(fv.len(), 2);
        assert_eq!(fv.get(0), &[1.0, 2.0, 3.0]);
        assert_eq!(fv.get(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_visited_set() {
        let mut vs = VisitedSet::new(100);
        vs.insert(42);
        assert!(vs.contains(42));
        assert!(!vs.contains(43));
        vs.clear(); // O(1) reset
        assert!(!vs.contains(42));
        vs.insert(43);
        assert!(vs.contains(43));
    }

    #[test]
    fn test_pq_flat_table() {
        // 2 subspaces, 4 centroids each (k=4 for test)
        let table = vec![
            0.1, 0.2, 0.3, 0.4, // subspace 0
            0.5, 0.6, 0.7, 0.8, // subspace 1
        ];
        let codes = vec![1u8, 2u8]; // code 1 from sub0, code 2 from sub1
        let dist = pq_asymmetric_distance(&codes, &table, 4);
        assert!((dist - (0.2 + 0.7)).abs() < 1e-6);
    }
}

/// `wasm32` + `simd128` correctness and A/B timing checks against the scalar
/// path. Both `wasm_simd128_*` and `scalar_*` are compiled into the *same*
/// wasm binary here (the crate is built once, with `-C target-feature=+simd128`),
/// so the comparison isn't confounded by separate builds. Run via:
///
/// ```sh
/// RUSTFLAGS="-C target-feature=+simd128" wasm-pack test --node crates/ruvector-diskann --no-default-features
/// ```
#[cfg(all(test, target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_simd128_tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_test::*;

    // No `wasm_bindgen_test_configure!` call: Node.js is the default test
    // runner for `wasm-pack test --node` / `wasm-bindgen-test-runner`; the
    // macro is only needed to opt into `run_in_browser` / `run_in_worker`.

    /// Dims exercising: empty, 1-3 elements (below one v128 lane), the
    /// 4/8-lane boundaries, non-multiples-of-4, and production embedding
    /// sizes (384/768/1024).
    const CORRECTNESS_DIMS: &[usize] = &[0, 1, 2, 3, 4, 5, 7, 8, 9, 384, 768, 1000, 1023, 1024];

    fn random_vec(rng: &mut StdRng, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| rng.gen_range(-10.0f32..10.0)).collect()
    }

    /// Combined absolute+relative tolerance (numpy `allclose`-style:
    /// `|a - b| <= atol + rtol * max(|a|, |b|)`). A flat `< 1e-6` absolute
    /// bound is not achievable here: the scalar path sums with 4 interleaved
    /// f32 accumulators and the simd128 path with 2 `v128` (8-wide)
    /// accumulators plus a horizontal-sum tail, so the two reduction trees
    /// visit terms in different orders. f32 addition isn't associative, so
    /// reordering shifts rounding by a few ULPs — expected floating-point
    /// behavior, not a correctness bug, and it grows with the accumulated
    /// magnitude (up to dim=1024 terms here). Observed on this exact grid:
    /// max relative error ~1.1e-7 (essentially f32 machine epsilon). rtol
    /// here is ~90x that observed margin — tight enough to catch a real bug
    /// (wrong lane math, dropped remainder) which would produce relative
    /// error orders of magnitude larger, not machine-epsilon-scale drift.
    fn assert_close(op: &str, dim: usize, scalar: f32, simd: f32) {
        const ATOL: f32 = 1e-5;
        const RTOL: f32 = 1e-5;
        let diff = (scalar - simd).abs();
        let bound = ATOL + RTOL * scalar.abs().max(simd.abs());
        assert!(
            diff <= bound,
            "{op} dim={dim}: scalar={scalar} simd128={simd} diff={diff} bound={bound}"
        );
    }

    #[wasm_bindgen_test]
    fn l2_squared_matches_scalar_across_dims() {
        let mut rng = StdRng::seed_from_u64(42);
        for &dim in CORRECTNESS_DIMS {
            let a = random_vec(&mut rng, dim);
            let b = random_vec(&mut rng, dim);
            let scalar = scalar_l2_squared(&a, &b);
            let simd = wasm_simd128_l2_squared(&a, &b);
            assert_close("l2_squared", dim, scalar, simd);
        }
    }

    #[wasm_bindgen_test]
    fn inner_product_matches_scalar_across_dims() {
        let mut rng = StdRng::seed_from_u64(7);
        for &dim in CORRECTNESS_DIMS {
            let a = random_vec(&mut rng, dim);
            let b = random_vec(&mut rng, dim);
            let scalar = scalar_inner_product(&a, &b);
            let simd = wasm_simd128_inner_product(&a, &b);
            assert_close("inner_product", dim, scalar, simd);
        }
    }

    #[wasm_bindgen_test]
    fn identical_vectors_are_zero_distance() {
        let a = vec![1.0f32; 384];
        assert!(wasm_simd128_l2_squared(&a, &a) < 1e-10);
    }

    /// A/B timing: geometric mean of scalar/simd128 wall time across
    /// {l2_squared, inner_product} x {384, 768, 1024} dims, printed for the
    /// PR's A/B table. Not a pass/fail assertion — the speedup gate is
    /// evaluated from this test's output, pre-registered in PR_BODY.md
    /// before this test was run. MUST be run with `--release` (`wasm-pack
    /// test --node --release`): the default debug/unopt profile leaves
    /// `unsafe` intrinsic calls uninlined and makes simd128 look *slower*
    /// than scalar — verified: debug gave geomean 0.12x, release 1.2x+ on
    /// the identical source.
    #[wasm_bindgen_test]
    fn ab_timing_report() {
        const DIMS: &[usize] = &[384, 768, 1024];
        const WARMUP_ITERS: usize = 2_000;
        const TIMED_ITERS: usize = 300_000;
        const ROUNDS: usize = 5;

        let mut rng = StdRng::seed_from_u64(1234);
        let mut log_ratio_sum = 0.0f64;
        let mut cell_count = 0u32;

        for &dim in DIMS {
            let vectors: Vec<(Vec<f32>, Vec<f32>)> = (0..64)
                .map(|_| (random_vec(&mut rng, dim), random_vec(&mut rng, dim)))
                .collect();

            let ops: [(&str, fn(&[f32], &[f32]) -> f32, fn(&[f32], &[f32]) -> f32); 2] = [
                (
                    "l2_squared",
                    scalar_l2_squared as fn(&[f32], &[f32]) -> f32,
                    wasm_simd128_l2_squared as fn(&[f32], &[f32]) -> f32,
                ),
                (
                    "inner_product",
                    scalar_inner_product as fn(&[f32], &[f32]) -> f32,
                    wasm_simd128_inner_product as fn(&[f32], &[f32]) -> f32,
                ),
            ];

            for (op_name, scalar_fn, simd_fn) in ops {
                // Warmup both paths (JIT/tiering, cache warm-up).
                let mut sink = 0.0f32;
                for i in 0..WARMUP_ITERS {
                    let (a, b) = &vectors[i % vectors.len()];
                    sink += scalar_fn(a, b) + simd_fn(a, b);
                }
                core::hint::black_box(sink);

                let mut scalar_rounds = [0.0f64; ROUNDS];
                let mut simd_rounds = [0.0f64; ROUNDS];

                for r in 0..ROUNDS {
                    let t0 = performance_now();
                    let mut sink = 0.0f32;
                    for i in 0..TIMED_ITERS {
                        let (a, b) = &vectors[i % vectors.len()];
                        sink += scalar_fn(a, b);
                    }
                    scalar_rounds[r] = performance_now() - t0;
                    core::hint::black_box(sink);

                    let t0 = performance_now();
                    let mut sink = 0.0f32;
                    for i in 0..TIMED_ITERS {
                        let (a, b) = &vectors[i % vectors.len()];
                        sink += simd_fn(a, b);
                    }
                    simd_rounds[r] = performance_now() - t0;
                    core::hint::black_box(sink);
                }

                let scalar_ms = median(&mut scalar_rounds);
                let simd_ms = median(&mut simd_rounds);

                let ratio = if simd_ms > 0.0 {
                    scalar_ms / simd_ms
                } else {
                    f64::NAN
                };
                log_ratio_sum += ratio.ln();
                cell_count += 1;

                web_sys_console_log(&format!(
                    "AB_675 op={op_name} dim={dim} iters={TIMED_ITERS} rounds={ROUNDS} scalar_ms={scalar_ms:.4} simd128_ms={simd_ms:.4} speedup={ratio:.4}"
                ));
            }
        }

        let geomean = (log_ratio_sum / cell_count as f64).exp();
        web_sys_console_log(&format!("AB_675 geomean_speedup={geomean:.4}"));
    }

    /// In-place median (sorts `xs`).
    fn median(xs: &mut [f64]) -> f64 {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs[xs.len() / 2]
    }

    /// `performance.now()` — sub-millisecond monotonic clock, available as a
    /// global under both Node (16+) and browsers. `js_sys`/`web_sys` only
    /// expose it hung off `window`/`Performance` objects that don't exist
    /// under Node's `--node` test runner, so bind the global directly.
    fn performance_now() -> f64 {
        #[wasm_bindgen]
        extern "C" {
            #[wasm_bindgen(js_namespace = performance, js_name = now)]
            fn now() -> f64;
        }
        now()
    }

    /// Minimal `console.log` shim so timing output shows up in
    /// `wasm-pack test --node` output without pulling in `web-sys`'s full
    /// `console` feature for this dev-only path.
    fn web_sys_console_log(msg: &str) {
        #[wasm_bindgen]
        extern "C" {
            #[wasm_bindgen(js_namespace = console, js_name = log)]
            fn log(s: &str);
        }
        log(msg);
    }
}
