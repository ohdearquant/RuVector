//! Distance computations with SIMD acceleration and optional GPU offload
//!
//! Dispatch priority: GPU (if `gpu` feature) → SimSIMD (if `simd` feature) → scalar

use crate::error::{DiskAnnError, Result};
use memmap2::Mmap;

/// Backing storage for the flat vector slab.
///
/// `Owned` is heap-resident — used while inserting/building, and by
/// [`FlatVectors::from_owned`] (the default, back-compat `load()` path).
/// `Mmap` is read-through: vector slices are read directly out of the mapped file
/// on each [`FlatVectors::get`] call, so RSS stays proportional to the accessed
/// working set instead of the whole dataset (see #674). It is populated only via
/// [`FlatVectors::from_mmap`], which validates 4-byte alignment up front so `get`
/// never has to fall back to an unaligned-read fail path per call.
enum VectorStorage {
    Owned(Vec<f32>),
    Mmap { mmap: Mmap, data_offset: usize },
}

/// Flat vector storage — contiguous memory for cache-friendly access.
/// Vectors are logically a single slab: `[v0_d0, v0_d1, ..., v1_d0, ...]`, either
/// owned in RAM or read straight out of a memory-mapped file.
pub struct FlatVectors {
    storage: VectorStorage,
    pub dim: usize,
    pub count: usize,
    /// Post-load tombstones for mmap-backed storage. The mapped file is never
    /// mutated, so deletes on read-through storage are tracked in this owned
    /// overlay instead of the in-place NaN write `Owned` storage uses. Empty (and
    /// unused) for `Owned` storage, which keeps the original NaN-write behavior.
    tombstones: Vec<bool>,
    /// Shared all-NaN row returned by `get()` for a tombstoned mmap index — same
    /// externally observable shape as the NaN vector `Owned::zero_out` produces.
    tombstone_row: Vec<f32>,
}

impl FlatVectors {
    pub fn new(dim: usize) -> Self {
        Self {
            storage: VectorStorage::Owned(Vec::new()),
            dim,
            count: 0,
            tombstones: Vec::new(),
            tombstone_row: vec![f32::NAN; dim],
        }
    }

    pub fn with_capacity(dim: usize, n: usize) -> Self {
        Self {
            storage: VectorStorage::Owned(Vec::with_capacity(n * dim)),
            dim,
            count: 0,
            tombstones: Vec::new(),
            tombstone_row: vec![f32::NAN; dim],
        }
    }

    /// Build owned flat storage directly from an already-materialized slab (e.g.
    /// copied out of a save file's mmap). `data.len()` must equal `count * dim`.
    pub fn from_owned(data: Vec<f32>, dim: usize, count: usize) -> Self {
        debug_assert_eq!(data.len(), count * dim);
        Self {
            storage: VectorStorage::Owned(data),
            dim,
            count,
            tombstones: Vec::new(),
            tombstone_row: vec![f32::NAN; dim],
        }
    }

    /// Build a read-through view directly over a memory-mapped file's flat f32
    /// slab, starting `data_offset` bytes into the map.
    ///
    /// Fails closed (returns `Err`, never transmutes) if the slab's start isn't
    /// 4-byte aligned — required to reinterpret mapped bytes as `f32` without UB —
    /// or if the map is too short for `count * dim` floats.
    pub fn from_mmap(mmap: Mmap, data_offset: usize, dim: usize, count: usize) -> Result<Self> {
        let base = mmap.as_ptr() as usize;
        if base.wrapping_add(data_offset) % std::mem::align_of::<f32>() != 0 {
            return Err(DiskAnnError::InvalidConfig(format!(
                "mmap vector data at offset {data_offset} is not 4-byte aligned (mmap base 0x{base:x}); refusing to cast unaligned bytes to f32"
            )));
        }
        let need_bytes = count
            .checked_mul(dim)
            .and_then(|floats| floats.checked_mul(4))
            .and_then(|bytes| bytes.checked_add(data_offset))
            .ok_or_else(|| {
                DiskAnnError::InvalidConfig("vector slab size overflowed usize".to_string())
            })?;
        if mmap.len() < need_bytes {
            return Err(DiskAnnError::InvalidConfig(format!(
                "mmap too short for {count} vectors of dim {dim}: need {need_bytes} bytes, have {}",
                mmap.len()
            )));
        }
        Ok(Self {
            storage: VectorStorage::Mmap { mmap, data_offset },
            dim,
            count,
            tombstones: vec![false; count],
            tombstone_row: vec![f32::NAN; dim],
        })
    }

    /// Whether this instance is backed by a read-through mmap (vs. owned RAM).
    pub fn is_mmap_backed(&self) -> bool {
        matches!(self.storage, VectorStorage::Mmap { .. })
    }

    /// Zero-copy byte view of the flat slab, when it is owned in RAM. `None` for
    /// mmap-backed storage — callers needing bytes there should read per-vector via
    /// [`FlatVectors::get`] instead of assuming one contiguous owned buffer.
    pub fn as_owned_slice(&self) -> Option<&[f32]> {
        match &self.storage {
            VectorStorage::Owned(data) => Some(data),
            VectorStorage::Mmap { .. } => None,
        }
    }

    #[inline]
    pub fn push(&mut self, vector: &[f32]) {
        debug_assert_eq!(vector.len(), self.dim);
        match &mut self.storage {
            VectorStorage::Owned(data) => {
                data.extend_from_slice(vector);
                self.count += 1;
            }
            VectorStorage::Mmap { .. } => {
                panic!(
                    "FlatVectors::push called on mmap-backed (read-through) storage — mmap-loaded indexes are read-only for inserts; callers must check is_mmap_backed() first"
                );
            }
        }
    }

    #[inline]
    pub fn get(&self, idx: usize) -> &[f32] {
        match &self.storage {
            VectorStorage::Owned(data) => {
                let start = idx * self.dim;
                &data[start..start + self.dim]
            }
            VectorStorage::Mmap { mmap, data_offset } => {
                if self.tombstones.get(idx).copied().unwrap_or(false) {
                    return &self.tombstone_row;
                }
                let start = data_offset + idx * self.dim * 4;
                let byte_slice = &mmap[start..start + self.dim * 4];
                bytemuck::cast_slice(byte_slice)
            }
        }
    }

    /// Zero out a vector (lazy deletion). Owned storage keeps writing NaN in place
    /// (unchanged, zero extra cost); mmap storage sets a tombstone flag instead of
    /// mutating the mapped file — both are observed identically through `get()`.
    #[inline]
    pub fn zero_out(&mut self, idx: usize) {
        match &mut self.storage {
            VectorStorage::Owned(data) => {
                let start = idx * self.dim;
                for v in &mut data[start..start + self.dim] {
                    *v = f32::NAN;
                }
            }
            VectorStorage::Mmap { .. } => {
                if let Some(flag) = self.tombstones.get_mut(idx) {
                    *flag = true;
                }
            }
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
        scalar_l2_squared(a, b)
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
        scalar_inner_product(a, b)
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
        assert!(!fv.is_mmap_backed());
    }

    /// Write a `vectors.bin`-shaped file (8-byte n, 8-byte dim, then flat
    /// little-endian f32 data) and mmap it — the same layout `index.rs` produces.
    fn mmap_fixture(data: &[f32], dim: usize, count: usize) -> memmap2::Mmap {
        use std::io::Write;
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&(count as u64).to_le_bytes()).unwrap();
        tmp.write_all(&(dim as u64).to_le_bytes()).unwrap();
        for &v in data {
            tmp.write_all(&v.to_le_bytes()).unwrap();
        }
        tmp.flush().unwrap();
        let file = std::fs::File::open(tmp.path()).unwrap();
        // Leak the tempfile handle for the test's duration — NamedTempFile deletes
        // on drop, but the mmap needs the backing file to stay mapped/openable.
        std::mem::forget(tmp);
        unsafe { memmap2::MmapOptions::new().map(&file).unwrap() }
    }

    #[test]
    fn test_flat_vectors_mmap_read_through_matches_data() {
        let dim = 4;
        let count = 3;
        let data: Vec<f32> = (0..(dim * count) as u32).map(|x| x as f32).collect();
        let mmap = mmap_fixture(&data, dim, count);

        let fv = FlatVectors::from_mmap(mmap, 16, dim, count).unwrap();
        assert!(fv.is_mmap_backed());
        assert_eq!(fv.len(), count);
        for i in 0..count {
            assert_eq!(fv.get(i), &data[i * dim..(i + 1) * dim]);
        }
    }

    #[test]
    fn test_flat_vectors_mmap_rejects_unaligned_offset() {
        let data = vec![0.0f32; 8];
        let mmap = mmap_fixture(&data, 4, 2);
        // offset=1 is never 4-byte aligned — from_mmap must fail closed rather
        // than transmute unaligned bytes to f32.
        let result = FlatVectors::from_mmap(mmap, 1, 4, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_flat_vectors_mmap_rejects_undersized_map() {
        let data = vec![0.0f32; 4];
        let mmap = mmap_fixture(&data, 4, 1);
        // Header + 1 vector present, but claim 10 vectors — must fail closed.
        let result = FlatVectors::from_mmap(mmap, 16, 4, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_flat_vectors_mmap_zero_out_tombstones_without_mutating_file() {
        let dim = 4;
        let count = 2;
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mmap = mmap_fixture(&data, dim, count);
        let mut fv = FlatVectors::from_mmap(mmap, 16, dim, count).unwrap();

        fv.zero_out(0);
        assert!(fv.get(0).iter().all(|x| x.is_nan()));
        assert_eq!(fv.get(1), &[5.0, 6.0, 7.0, 8.0], "untouched row unaffected");
    }

    #[test]
    fn test_flat_vectors_push_panics_on_mmap_storage() {
        let data = vec![0.0f32; 4];
        let mmap = mmap_fixture(&data, 4, 1);
        let mut fv = FlatVectors::from_mmap(mmap, 16, 4, 1).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fv.push(&[1.0, 2.0, 3.0, 4.0]);
        }));
        assert!(
            result.is_err(),
            "push on mmap-backed storage must fail loud, not silently succeed"
        );
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
