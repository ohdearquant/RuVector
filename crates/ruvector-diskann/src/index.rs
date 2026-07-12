//! DiskANN index — ties together Vamana graph, PQ, and mmap persistence

use crate::distance::{l2_squared, FlatVectors, VisitedSet};
use crate::error::{DiskAnnError, Result};
use crate::graph::VamanaGraph;
use crate::pq::ProductQuantizer;
use memmap2::MmapOptions;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
}

/// Configuration for DiskANN index
#[derive(Debug, Clone)]
pub struct DiskAnnConfig {
    /// Vector dimension
    pub dim: usize,
    /// Maximum out-degree for Vamana graph (R)
    pub max_degree: usize,
    /// Search beam width during construction (L_build)
    pub build_beam: usize,
    /// Search beam width during query (L_search)
    pub search_beam: usize,
    /// Alpha parameter for robust pruning (>= 1.0)
    pub alpha: f32,
    /// Number of PQ subspaces (M). 0 = no PQ.
    pub pq_subspaces: usize,
    /// PQ training iterations
    pub pq_iterations: usize,
    /// Storage directory for persistence
    pub storage_path: Option<PathBuf>,
}

impl Default for DiskAnnConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            max_degree: 64,
            build_beam: 128,
            search_beam: 64,
            alpha: 1.2,
            pq_subspaces: 0,
            pq_iterations: 10,
            storage_path: None,
        }
    }
}

/// DiskANN index with Vamana graph + optional PQ + mmap persistence
pub struct DiskAnnIndex {
    config: DiskAnnConfig,
    /// Flat contiguous vector storage (cache-friendly)
    vectors: FlatVectors,
    /// ID mapping: internal index -> external string ID
    id_map: Vec<String>,
    /// Reverse mapping: external ID -> internal index
    id_reverse: HashMap<String, u32>,
    /// Vamana graph
    graph: Option<VamanaGraph>,
    /// Product quantizer (optional)
    pq: Option<ProductQuantizer>,
    /// PQ codes for all vectors
    pq_codes: Vec<Vec<u8>>,
    /// Whether index has been built
    built: bool,
    /// Reusable visited set for search (avoids per-query allocation)
    visited: Option<VisitedSet>,
}

impl DiskAnnIndex {
    /// Create a new DiskANN index
    pub fn new(config: DiskAnnConfig) -> Self {
        let dim = config.dim;
        Self {
            config,
            vectors: FlatVectors::new(dim),
            id_map: Vec::new(),
            id_reverse: HashMap::new(),
            graph: None,
            pq: None,
            pq_codes: Vec::new(),
            built: false,
            visited: None,
        }
    }

    /// Insert a vector with a string ID
    pub fn insert(&mut self, id: String, vector: Vec<f32>) -> Result<()> {
        if self.vectors.is_mmap_backed() {
            return Err(DiskAnnError::InvalidConfig(
                "cannot insert into an mmap-loaded index (read-only, read-through storage); use DiskAnnIndex::load() instead of load_mmap() if further inserts are needed".to_string(),
            ));
        }
        if vector.len() != self.config.dim {
            return Err(DiskAnnError::DimensionMismatch {
                expected: self.config.dim,
                actual: vector.len(),
            });
        }
        if self.id_reverse.contains_key(&id) {
            return Err(DiskAnnError::InvalidConfig(format!("Duplicate ID: {id}")));
        }

        let idx = self.vectors.len() as u32;
        self.id_reverse.insert(id.clone(), idx);
        self.id_map.push(id);
        self.vectors.push(&vector);
        self.built = false;
        Ok(())
    }

    /// Insert a batch of vectors
    pub fn insert_batch(&mut self, entries: Vec<(String, Vec<f32>)>) -> Result<()> {
        for (id, vector) in entries {
            self.insert(id, vector)?;
        }
        Ok(())
    }

    /// Build the index (must be called after all inserts, before search)
    pub fn build(&mut self) -> Result<()> {
        let n = self.vectors.len();
        if n == 0 {
            return Err(DiskAnnError::Empty);
        }

        // Train PQ if configured
        if self.config.pq_subspaces > 0 {
            // Collect vectors for PQ training
            let vecs: Vec<Vec<f32>> = (0..n).map(|i| self.vectors.get(i).to_vec()).collect();
            let mut pq = ProductQuantizer::new(self.config.dim, self.config.pq_subspaces)?;
            pq.train(&vecs, self.config.pq_iterations)?;

            self.pq_codes = vecs
                .iter()
                .map(|v| pq.encode(v))
                .collect::<Result<Vec<_>>>()?;

            self.pq = Some(pq);
        }

        // Build Vamana graph on flat storage
        let mut graph = VamanaGraph::new(
            n,
            self.config.max_degree,
            self.config.build_beam,
            self.config.alpha,
        );
        graph.build(&self.vectors)?;
        self.graph = Some(graph);

        // Pre-allocate visited set for search
        self.visited = Some(VisitedSet::new(n));
        self.built = true;

        if let Some(ref path) = self.config.storage_path {
            self.save(path)?;
        }

        Ok(())
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if !self.built {
            return Err(DiskAnnError::NotBuilt);
        }
        if query.len() != self.config.dim {
            return Err(DiskAnnError::DimensionMismatch {
                expected: self.config.dim,
                actual: query.len(),
            });
        }

        let graph = self.graph.as_ref().unwrap();
        let beam = self.config.search_beam.max(k);

        let (candidates, _) = graph.greedy_search(&self.vectors, query, beam);

        // Re-rank candidates with exact distance
        let mut scored: Vec<(u32, f32)> = candidates
            .into_iter()
            .map(|id| (id, l2_squared(self.vectors.get(id as usize), query)))
            .collect();
        scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored
            .into_iter()
            .take(k)
            .map(|(id, dist)| SearchResult {
                id: self.id_map[id as usize].clone(),
                distance: dist,
            })
            .collect())
    }

    /// Get the number of vectors in the index
    pub fn count(&self) -> usize {
        self.vectors.len()
    }

    /// Delete a vector by ID (marks as deleted, doesn't rebuild graph)
    pub fn delete(&mut self, id: &str) -> Result<bool> {
        if let Some(&idx) = self.id_reverse.get(id) {
            self.vectors.zero_out(idx as usize);
            self.id_reverse.remove(id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Save index to disk
    pub fn save(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir)?;

        // Save vectors as flat binary (already contiguous — mmap-friendly)
        let vec_path = dir.join("vectors.bin");
        let mut f = BufWriter::new(File::create(&vec_path)?);
        let n = self.vectors.len() as u64;
        let dim = self.config.dim as u64;
        f.write_all(&n.to_le_bytes())?;
        f.write_all(&dim.to_le_bytes())?;
        if let Some(data) = self.vectors.as_owned_slice() {
            // Owned storage: write the flat slab directly — zero copy.
            let byte_slice =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
            f.write_all(byte_slice)?;
        } else {
            // Mmap-backed storage: no single contiguous owned buffer to cast, so
            // re-serialize per vector instead (this also folds in any post-load
            // tombstones as NaN rows, matching owned storage's on-disk encoding).
            for i in 0..self.vectors.len() {
                for &value in self.vectors.get(i) {
                    f.write_all(&value.to_le_bytes())?;
                }
            }
        }
        f.flush()?;

        // Save graph adjacency
        let graph_path = dir.join("graph.bin");
        let mut f = BufWriter::new(File::create(&graph_path)?);
        if let Some(ref graph) = self.graph {
            f.write_all(&(graph.medoid as u64).to_le_bytes())?;
            f.write_all(&(graph.neighbors.len() as u64).to_le_bytes())?;
            for neighbors in &graph.neighbors {
                f.write_all(&(neighbors.len() as u32).to_le_bytes())?;
                for &n in neighbors {
                    f.write_all(&n.to_le_bytes())?;
                }
            }
        }
        f.flush()?;

        // Save ID map
        let ids_path = dir.join("ids.json");
        let ids_json = serde_json::to_string(&self.id_map)
            .map_err(|e| DiskAnnError::Serialization(e.to_string()))?;
        fs::write(&ids_path, ids_json)?;

        // Save PQ if present
        if let Some(ref pq) = self.pq {
            let pq_path = dir.join("pq.bin");
            let pq_bytes = bincode::encode_to_vec(pq, bincode::config::standard())
                .map_err(|e| DiskAnnError::Serialization(e.to_string()))?;
            fs::write(&pq_path, pq_bytes)?;

            // Save PQ codes
            let codes_path = dir.join("pq_codes.bin");
            let mut f = BufWriter::new(File::create(&codes_path)?);
            for codes in &self.pq_codes {
                f.write_all(codes)?;
            }
            f.flush()?;
        }

        // Save config
        let config_path = dir.join("config.json");
        let config_json = serde_json::json!({
            "dim": self.config.dim,
            "max_degree": self.config.max_degree,
            "build_beam": self.config.build_beam,
            "search_beam": self.config.search_beam,
            "alpha": self.config.alpha,
            "pq_subspaces": self.config.pq_subspaces,
            "count": self.vectors.len(),
            "built": self.built,
        });
        fs::write(
            &config_path,
            serde_json::to_string_pretty(&config_json).unwrap(),
        )?;

        Ok(())
    }

    /// Load index from disk with vectors fully materialized into RAM (default,
    /// back-compat behavior — unchanged from prior releases).
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_impl(dir, false)
    }

    /// Load index from disk with vectors served read-through from an mmap: the OS
    /// pages in only accessed vectors, so RSS stays proportional to the query
    /// working set instead of the full dataset. Graph adjacency, PQ codes, and IDs
    /// are still heap-resident (small relative to the vector slab). See #674.
    ///
    /// Opt-in — `load()` remains the default and is unaffected. Further inserts on
    /// the returned index fail with `DiskAnnError::InvalidConfig`; the read-through
    /// map is not writable. Deletes still work identically to `load()` (see
    /// [`FlatVectors::zero_out`]).
    pub fn load_mmap(dir: &Path) -> Result<Self> {
        Self::load_impl(dir, true)
    }

    fn load_impl(dir: &Path, mmap_vectors: bool) -> Result<Self> {
        // Load config
        let config_json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.join("config.json"))?)
                .map_err(|e| DiskAnnError::Serialization(e.to_string()))?;

        let dim = config_json["dim"].as_u64().unwrap() as usize;
        let max_degree = config_json["max_degree"].as_u64().unwrap() as usize;
        let build_beam = config_json["build_beam"].as_u64().unwrap() as usize;
        let search_beam = config_json["search_beam"].as_u64().unwrap() as usize;
        let alpha = config_json["alpha"].as_f64().unwrap() as f32;
        let pq_subspaces = config_json["pq_subspaces"].as_u64().unwrap_or(0) as usize;

        let config = DiskAnnConfig {
            dim,
            max_degree,
            build_beam,
            search_beam,
            alpha,
            pq_subspaces,
            storage_path: Some(dir.to_path_buf()),
            ..Default::default()
        };

        // Map the vectors file (needed either way to read the 16-byte header; in
        // mmap mode the map is retained and read through, in owned mode it is
        // copied out below and then dropped).
        let vec_file = File::open(dir.join("vectors.bin"))?;
        let mmap = unsafe { MmapOptions::new().map(&vec_file)? };

        let n = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let file_dim = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        assert_eq!(file_dim, dim);

        let data_start = 16;
        let vectors = if mmap_vectors {
            FlatVectors::from_mmap(mmap, data_start, dim, n)?
        } else {
            // Copy the flat slab out of the map into an owned Vec — unchanged
            // from the pre-#674 behavior.
            let total_floats = n * dim;
            let mut flat_data = Vec::with_capacity(total_floats);
            let byte_slice = &mmap[data_start..data_start + total_floats * 4];
            // Safe: f32 from le bytes
            for chunk in byte_slice.chunks_exact(4) {
                flat_data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            FlatVectors::from_owned(flat_data, dim, n)
        };

        // Load IDs
        let ids_json = fs::read_to_string(dir.join("ids.json"))?;
        let id_map: Vec<String> = serde_json::from_str(&ids_json)
            .map_err(|e| DiskAnnError::Serialization(e.to_string()))?;

        let mut id_reverse = HashMap::new();
        for (i, id) in id_map.iter().enumerate() {
            id_reverse.insert(id.clone(), i as u32);
        }

        // Load graph
        let graph_bytes = fs::read(dir.join("graph.bin"))?;
        let medoid = u64::from_le_bytes(graph_bytes[0..8].try_into().unwrap()) as u32;
        let graph_n = u64::from_le_bytes(graph_bytes[8..16].try_into().unwrap()) as usize;

        let mut neighbors = Vec::with_capacity(graph_n);
        let mut offset = 16;
        for _ in 0..graph_n {
            let deg =
                u32::from_le_bytes(graph_bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            let mut nbrs = Vec::with_capacity(deg);
            for _ in 0..deg {
                let nbr = u32::from_le_bytes(graph_bytes[offset..offset + 4].try_into().unwrap());
                offset += 4;
                nbrs.push(nbr);
            }
            neighbors.push(nbrs);
        }

        let graph = VamanaGraph {
            neighbors,
            medoid,
            max_degree,
            build_beam,
            alpha,
        };

        // Load PQ if present
        let pq_path = dir.join("pq.bin");
        let (pq, pq_codes) = if pq_path.exists() {
            let pq_bytes = fs::read(&pq_path)?;
            let (pq, _): (ProductQuantizer, usize) =
                bincode::decode_from_slice(&pq_bytes, bincode::config::standard())
                    .map_err(|e| DiskAnnError::Serialization(e.to_string()))?;

            let codes_bytes = fs::read(dir.join("pq_codes.bin"))?;
            let m = pq.m;
            let mut codes = Vec::with_capacity(n);
            for i in 0..n {
                codes.push(codes_bytes[i * m..(i + 1) * m].to_vec());
            }
            (Some(pq), codes)
        } else {
            (None, Vec::new())
        };

        Ok(Self {
            config,
            vectors,
            id_map,
            id_reverse,
            graph: Some(graph),
            pq,
            pq_codes,
            built: true,
            visited: Some(VisitedSet::new(n)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn random_vectors(n: usize, dim: usize) -> Vec<(String, Vec<f32>)> {
        use rand::prelude::*;
        // Seeded so tests are deterministic across CI runs — random data made
        // basic-search assertions (nearest of vec-X is vec-X) flake when the
        // ANN graph traversal happened to land on an unrelated near-duplicate.
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xD15CA77);
        (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
                (format!("vec-{i}"), v)
            })
            .collect()
    }

    fn random_data(n: usize, dim: usize) -> Vec<(String, Vec<f32>)> {
        random_vectors(n, dim)
    }

    #[test]
    fn test_diskann_basic() {
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 32,
            max_degree: 16,
            build_beam: 32,
            search_beam: 32,
            alpha: 1.2,
            ..Default::default()
        });

        let data = random_vectors(500, 32);
        let query = data[42].1.clone();

        index.insert_batch(data).unwrap();
        index.build().unwrap();

        let results = index.search(&query, 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "vec-42"); // Should find itself
        assert!(results[0].distance < 1e-6); // Exact match
    }

    #[test]
    fn test_diskann_with_pq() {
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 32,
            max_degree: 16,
            build_beam: 32,
            search_beam: 32,
            alpha: 1.2,
            pq_subspaces: 4,
            pq_iterations: 5,
            ..Default::default()
        });

        let data = random_vectors(200, 32);
        let query = data[10].1.clone();

        index.insert_batch(data).unwrap();
        index.build().unwrap();

        let results = index.search(&query, 5).unwrap();
        assert_eq!(results[0].id, "vec-10");
    }

    #[test]
    fn test_diskann_save_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("diskann_test");

        let data = random_vectors(100, 16);
        let query = data[7].1.clone();

        // Build and save
        {
            let mut index = DiskAnnIndex::new(DiskAnnConfig {
                dim: 16,
                max_degree: 8,
                build_beam: 16,
                search_beam: 16,
                alpha: 1.2,
                storage_path: Some(path.clone()),
                ..Default::default()
            });
            index.insert_batch(data).unwrap();
            index.build().unwrap();
        }

        // Load and search
        let loaded = DiskAnnIndex::load(&path).unwrap();
        let results = loaded.search(&query, 3).unwrap();
        assert_eq!(results[0].id, "vec-7");
    }

    #[test]
    fn test_diskann_load_mmap_basic() {
        // Same fixture as test_diskann_save_load, but exercised through the new
        // read-through load_mmap() entry point (#674).
        let dir = tempdir().unwrap();
        let path = dir.path().join("diskann_mmap_test");

        let data = random_vectors(100, 16);
        let query = data[7].1.clone();

        {
            let mut index = DiskAnnIndex::new(DiskAnnConfig {
                dim: 16,
                max_degree: 8,
                build_beam: 16,
                search_beam: 16,
                alpha: 1.2,
                storage_path: Some(path.clone()),
                ..Default::default()
            });
            index.insert_batch(data).unwrap();
            index.build().unwrap();
        }

        let loaded = DiskAnnIndex::load_mmap(&path).unwrap();
        assert!(loaded.vectors.is_mmap_backed());
        let results = loaded.search(&query, 3).unwrap();
        assert_eq!(results[0].id, "vec-7");
        assert!(results[0].distance < 1e-6);
    }

    #[test]
    fn test_load_and_load_mmap_return_identical_results() {
        // Parity contract: both storage backends must be indistinguishable to
        // callers for the same on-disk index — the core requirement of #674.
        use rand::prelude::*;

        let dir = tempdir().unwrap();
        let path = dir.path().join("diskann_parity_test");
        let data = random_vectors(300, 24);

        {
            let mut index = DiskAnnIndex::new(DiskAnnConfig {
                dim: 24,
                max_degree: 16,
                build_beam: 32,
                search_beam: 32,
                alpha: 1.2,
                storage_path: Some(path.clone()),
                ..Default::default()
            });
            index.insert_batch(data.clone()).unwrap();
            index.build().unwrap();
        }

        let owned = DiskAnnIndex::load(&path).unwrap();
        let mmapped = DiskAnnIndex::load_mmap(&path).unwrap();
        assert!(!owned.vectors.is_mmap_backed());
        assert!(mmapped.vectors.is_mmap_backed());

        let mut rng = StdRng::seed_from_u64(0xACED);
        for _ in 0..20 {
            let qi = rng.gen_range(0..data.len());
            let query = &data[qi].1;
            let owned_results: Vec<(String, f32)> = owned
                .search(query, 5)
                .unwrap()
                .into_iter()
                .map(|r| (r.id, r.distance))
                .collect();
            let mmap_results: Vec<(String, f32)> = mmapped
                .search(query, 5)
                .unwrap()
                .into_iter()
                .map(|r| (r.id, r.distance))
                .collect();
            assert_eq!(owned_results, mmap_results);
        }
    }

    #[test]
    fn test_delete_owned_zero_out_is_nan() {
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 4,
            ..Default::default()
        });
        index.insert("a".to_string(), vec![1.0; 4]).unwrap();
        index.insert("b".to_string(), vec![2.0; 4]).unwrap();
        index.build().unwrap();

        assert!(index.delete("a").unwrap());
        assert!(index.vectors.get(0).iter().all(|x| x.is_nan()));
        assert_eq!(index.vectors.get(1), &[2.0; 4]);
    }

    #[test]
    fn test_load_mmap_delete_matches_owned_tombstone() {
        // Delete/tombstone representation must be identical (externally: an
        // all-NaN row) whether storage is Owned or mmap-backed.
        let dir = tempdir().unwrap();
        let path = dir.path().join("diskann_mmap_delete_test");
        let data = random_vectors(50, 8);

        {
            let mut index = DiskAnnIndex::new(DiskAnnConfig {
                dim: 8,
                max_degree: 8,
                build_beam: 16,
                search_beam: 16,
                alpha: 1.2,
                storage_path: Some(path.clone()),
                ..Default::default()
            });
            index.insert_batch(data).unwrap();
            index.build().unwrap();
        }

        let mut loaded = DiskAnnIndex::load_mmap(&path).unwrap();
        assert!(loaded.delete("vec-3").unwrap());
        // "vec-3" was the 4th insert (0-indexed) — insert() assigns internal ids
        // in insertion order, and random_vectors() preserves that same order.
        assert!(loaded.vectors.get(3).iter().all(|x| x.is_nan()));
        assert!(
            !loaded.delete("vec-3").unwrap(),
            "second delete of the same id must report not-found"
        );
    }

    #[test]
    fn test_insert_rejected_on_mmap_loaded_index() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("diskann_mmap_insert_reject_test");
        let data = random_vectors(20, 4);

        {
            let mut index = DiskAnnIndex::new(DiskAnnConfig {
                dim: 4,
                max_degree: 4,
                build_beam: 8,
                search_beam: 8,
                alpha: 1.2,
                storage_path: Some(path.clone()),
                ..Default::default()
            });
            index.insert_batch(data).unwrap();
            index.build().unwrap();
        }

        let mut loaded = DiskAnnIndex::load_mmap(&path).unwrap();
        let result = loaded.insert("new-vec".to_string(), vec![1.0; 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_recall_at_10() {
        // Measure recall@10: what fraction of true top-10 neighbors does DiskANN find?
        use rand::prelude::*;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xD15CA77);
        let n = 2000;
        let dim = 64;
        let k = 10;

        let data: Vec<(String, Vec<f32>)> = (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
                (format!("v{i}"), v)
            })
            .collect();

        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim,
            max_degree: 32,
            build_beam: 64,
            search_beam: 64,
            alpha: 1.2,
            ..Default::default()
        });
        index.insert_batch(data.clone()).unwrap();
        index.build().unwrap();

        // Test 50 random queries
        let num_queries = 50;
        let mut total_recall = 0.0;

        for _ in 0..num_queries {
            let qi = rng.gen_range(0..n);
            let query = &data[qi].1;

            // Brute-force ground truth
            let mut brute: Vec<(usize, f32)> = data
                .iter()
                .enumerate()
                .map(|(i, (_, v))| (i, crate::distance::l2_squared(v, query)))
                .collect();
            brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let gt: std::collections::HashSet<String> =
                brute[..k].iter().map(|(i, _)| data[*i].0.clone()).collect();

            // DiskANN search
            let results = index.search(query, k).unwrap();
            let found: std::collections::HashSet<String> =
                results.iter().map(|r| r.id.clone()).collect();

            let recall = gt.intersection(&found).count() as f64 / k as f64;
            total_recall += recall;
        }

        let avg_recall = total_recall / num_queries as f64;
        println!("Recall@{k} = {avg_recall:.3} (n={n}, dim={dim}, queries={num_queries})");
        assert!(
            avg_recall >= 0.85,
            "Recall@{k} = {avg_recall:.3}, expected >= 0.85"
        );
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 16,
            ..Default::default()
        });

        // Wrong dimension on insert
        let result = index.insert("bad".to_string(), vec![1.0; 32]);
        assert!(result.is_err());

        // Wrong dimension on search
        index.insert("ok".to_string(), vec![1.0; 16]).unwrap();
        index.build().unwrap();
        let result = index.search(&[1.0; 32], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_id_rejected() {
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 4,
            ..Default::default()
        });
        index.insert("a".to_string(), vec![1.0; 4]).unwrap();
        let result = index.insert("a".to_string(), vec![2.0; 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_before_build_fails() {
        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim: 4,
            ..Default::default()
        });
        index.insert("a".to_string(), vec![1.0; 4]).unwrap();
        let result = index.search(&[1.0; 4], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_scale_5k() {
        // 5000 vectors, 128-dim — should build in under 5 seconds
        use rand::prelude::*;
        use std::time::Instant;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xD15CA77);

        let n = 5000;
        let dim = 128;
        let data: Vec<(String, Vec<f32>)> = (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
                (format!("v{i}"), v)
            })
            .collect();

        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim,
            max_degree: 48,
            build_beam: 96,
            search_beam: 48,
            alpha: 1.2,
            ..Default::default()
        });
        index.insert_batch(data.clone()).unwrap();

        let t0 = Instant::now();
        index.build().unwrap();
        let build_ms = t0.elapsed().as_millis();
        println!("Build {n} vectors ({dim}d): {build_ms}ms");

        // Search latency
        let query = &data[0].1;
        let t0 = Instant::now();
        let iters = 100;
        for _ in 0..iters {
            let _ = index.search(query, 10).unwrap();
        }
        let search_us = t0.elapsed().as_micros() / iters;
        println!("Search latency (k=10): {search_us}µs avg over {iters} queries");

        assert!(
            search_us < 10_000,
            "Search took {search_us}µs, expected <10ms"
        );
    }
}
