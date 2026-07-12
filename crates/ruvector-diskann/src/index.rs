//! DiskANN index — ties together Vamana graph, PQ, and mmap persistence

use crate::distance::{l2_squared, FlatVectors, VisitedSet};
use crate::error::{DiskAnnError, Result};
use crate::graph::VamanaGraph;
use crate::pq::ProductQuantizer;
use memmap2::{Mmap, MmapOptions};
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
    /// Memory-mapped vector data (for large datasets)
    mmap: Option<Mmap>,
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
            mmap: None,
        }
    }

    /// Insert a vector with a string ID
    pub fn insert(&mut self, id: String, vector: Vec<f32>) -> Result<()> {
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

        // PQ-guided traversal (#673): when build() trained a quantizer, score
        // candidate hops via the per-query asymmetric distance table instead
        // of exact L2, then re-rank the returned beam with exact distance
        // below — per ADR-144's own search design. No PQ configured
        // (pq_subspaces == 0, the default) means `self.pq` stays `None` and
        // this takes the exact-only path unchanged.
        //
        // Both arms route through the `_fast` graph entry points with a
        // locally-owned `VisitedSet`, the same shape `search_with`-style
        // reusable-visited-state callers use, rather than the self-allocating
        // `greedy_search`/`greedy_search_pq` wrappers.
        let candidates = if let Some(ref pq) = self.pq {
            let table = pq.build_distance_table(query)?;
            let mut visited = VisitedSet::new(self.pq_codes.len());
            graph
                .greedy_search_pq_fast(&self.pq_codes, &table, beam, &mut visited)
                .0
        } else {
            let mut visited = VisitedSet::new(self.vectors.len());
            graph
                .greedy_search_fast(&self.vectors, query, beam, &mut visited)
                .0
        };

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
        // Write flat slab directly — zero copy
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                self.vectors.data.as_ptr() as *const u8,
                self.vectors.data.len() * 4,
            )
        };
        f.write_all(byte_slice)?;
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

    /// Load index from disk with memory-mapped vectors
    pub fn load(dir: &Path) -> Result<Self> {
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

        // Load vectors via mmap
        let vec_file = File::open(dir.join("vectors.bin"))?;
        let mmap = unsafe { MmapOptions::new().map(&vec_file)? };

        let n = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let file_dim = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        assert_eq!(file_dim, dim);

        // Load vectors directly into flat slab from mmap
        let data_start = 16;
        let total_floats = n * dim;
        let mut flat_data = Vec::with_capacity(total_floats);
        let byte_slice = &mmap[data_start..data_start + total_floats * 4];
        // Safe: f32 from le bytes
        for chunk in byte_slice.chunks_exact(4) {
            flat_data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        let vectors = FlatVectors {
            data: flat_data,
            dim,
            count: n,
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
            mmap: Some(mmap),
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
    fn test_recall_at_10_with_pq_guided_search() {
        // Same seeded 2k x 64d harness as test_recall_at_10, but pq_subspaces
        // is configured (#673) so search() takes the PQ-guided traversal path
        // instead of exact L2 during the graph walk. Recall must clear the
        // same bar as the exact path — the final exact re-rank absorbs most
        // of the approximation loss from PQ-scored hops.
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
            pq_subspaces: 8,
            pq_iterations: 10,
            ..Default::default()
        });
        index.insert_batch(data.clone()).unwrap();
        index.build().unwrap();

        let num_queries = 50;
        let mut total_recall = 0.0;

        for _ in 0..num_queries {
            let qi = rng.gen_range(0..n);
            let query = &data[qi].1;

            let mut brute: Vec<(usize, f32)> = data
                .iter()
                .enumerate()
                .map(|(i, (_, v))| (i, crate::distance::l2_squared(v, query)))
                .collect();
            brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let gt: std::collections::HashSet<String> =
                brute[..k].iter().map(|(i, _)| data[*i].0.clone()).collect();

            let results = index.search(query, k).unwrap();
            let found: std::collections::HashSet<String> =
                results.iter().map(|r| r.id.clone()).collect();

            let recall = gt.intersection(&found).count() as f64 / k as f64;
            total_recall += recall;
        }

        let avg_recall = total_recall / num_queries as f64;
        println!(
            "PQ-guided Recall@{k} = {avg_recall:.3} (n={n}, dim={dim}, queries={num_queries}, pq_subspaces=8)"
        );
        assert!(
            avg_recall >= 0.85,
            "PQ-guided Recall@{k} = {avg_recall:.3}, expected >= 0.85"
        );
    }

    #[test]
    fn test_search_without_pq_matches_exact_reference_path() {
        // #673: search() now branches on self.pq.is_some(). With
        // pq_subspaces == 0 (the default), pq stays None and search() must
        // still take exactly the pre-#673 exact-L2 path. Verified by
        // replicating that path by hand (graph.greedy_search + the same
        // re-rank loop search() runs) and asserting identical ids and
        // distances on a seeded fixture.
        use rand::prelude::*;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xD15CA77);
        let n = 500;
        let dim = 32;
        let k = 10;

        let data: Vec<(String, Vec<f32>)> = (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
                (format!("v{i}"), v)
            })
            .collect();

        let mut index = DiskAnnIndex::new(DiskAnnConfig {
            dim,
            max_degree: 16,
            build_beam: 32,
            search_beam: 32,
            alpha: 1.2,
            ..Default::default() // pq_subspaces: 0 — no PQ trained
        });
        index.insert_batch(data.clone()).unwrap();
        index.build().unwrap();
        assert!(
            index.pq.is_none(),
            "pq_subspaces=0 must not train a quantizer"
        );

        let query = &data[3].1;
        let via_search = index.search(query, k).unwrap();

        // Hand-replicated pre-#673 exact path (identical to the `else`
        // branch search() runs internally).
        let graph = index.graph.as_ref().unwrap();
        let beam = index.config.search_beam.max(k);
        let (candidates, _) = graph.greedy_search(&index.vectors, query, beam);
        let mut scored: Vec<(u32, f32)> = candidates
            .into_iter()
            .map(|id| {
                (
                    id,
                    crate::distance::l2_squared(index.vectors.get(id as usize), query),
                )
            })
            .collect();
        scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let reference: Vec<(String, f32)> = scored
            .into_iter()
            .take(k)
            .map(|(id, dist)| (index.id_map[id as usize].clone(), dist))
            .collect();

        assert_eq!(via_search.len(), reference.len());
        for (a, (rid, rdist)) in via_search.iter().zip(reference.iter()) {
            assert_eq!(&a.id, rid, "id order must match the pre-#673 exact path");
            assert_eq!(
                a.distance, *rdist,
                "distance must match the pre-#673 exact path exactly"
            );
        }
    }

    #[test]
    fn test_pq_guided_search_degrades_with_corrupted_distance_table() {
        // Mutation check for #673: PQ-guided traversal must actually depend
        // on the distance table's content, not just its shape. Build a
        // normal PQ index, confirm recall clears the usual bar with the
        // correct per-query table, then corrupt the table (rotate each
        // subspace's 256-entry block by one slot — same shape, wrong
        // content) and confirm recall collapses. A traversal that silently
        // ignored the table would pass with either table; this fails that.
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
            pq_subspaces: 8,
            pq_iterations: 10,
            ..Default::default()
        });
        index.insert_batch(data.clone()).unwrap();
        index.build().unwrap();

        let pq = index.pq.as_ref().unwrap().clone();
        let graph = index.graph.as_ref().unwrap();
        let pq_codes = &index.pq_codes;
        let beam = index.config.search_beam.max(k);

        let num_queries = 30;
        let mut recall_correct = 0.0;
        let mut recall_corrupted = 0.0;
        let mut qrng = rand::rngs::StdRng::seed_from_u64(0xBEEF);

        for _ in 0..num_queries {
            let qi = qrng.gen_range(0..n);
            let query = &data[qi].1;

            let mut brute: Vec<(usize, f32)> = data
                .iter()
                .enumerate()
                .map(|(i, (_, v))| (i, crate::distance::l2_squared(v, query)))
                .collect();
            brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let gt: std::collections::HashSet<u32> =
                brute[..k].iter().map(|(i, _)| *i as u32).collect();

            // Exact re-rank of the traversal's candidate set — mirrors what
            // search() does with the beam it gets back from greedy_search_pq.
            let rerank = |cands: Vec<u32>| -> std::collections::HashSet<u32> {
                let mut scored: Vec<(u32, f32)> = cands
                    .into_iter()
                    .map(|id| (id, crate::distance::l2_squared(&data[id as usize].1, query)))
                    .collect();
                scored.sort_unstable_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                scored.into_iter().take(k).map(|(id, _)| id).collect()
            };

            let table = pq.build_distance_table(query).unwrap();
            let (cands, _) = graph.greedy_search_pq(pq_codes, &table, beam);
            let found = rerank(cands);
            recall_correct += gt.intersection(&found).count() as f64 / k as f64;

            let mut corrupted = table.clone();
            for sub in 0..pq.m {
                let base = sub * 256;
                corrupted[base..base + 256].rotate_left(1);
            }
            let (cands_bad, _) = graph.greedy_search_pq(pq_codes, &corrupted, beam);
            let found_bad = rerank(cands_bad);
            recall_corrupted += gt.intersection(&found_bad).count() as f64 / k as f64;
        }

        let avg_correct = recall_correct / num_queries as f64;
        let avg_corrupted = recall_corrupted / num_queries as f64;
        println!(
            "PQ table-integrity check: correct-table recall={avg_correct:.3}, corrupted-table recall={avg_corrupted:.3}"
        );

        assert!(
            avg_correct >= 0.85,
            "correct-table recall {avg_correct:.3} should clear the usual 0.85 bar"
        );
        assert!(
            avg_corrupted < avg_correct - 0.2,
            "corrupted-table recall {avg_corrupted:.3} should collapse well below \
             correct-table recall {avg_correct:.3} if traversal genuinely depends on \
             distance-table content"
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

    #[test]
    #[ignore] // Slow (N=100k, PQ training) — run explicitly:
              // cargo test -p ruvector-diskann --release -- --ignored --nocapture bench_pq_vs_exact_100k
    fn bench_pq_vs_exact_100k() {
        // #673 A/B evidence: exact-traversal search vs PQ-guided search at
        // N=100_000, dim=128 (this crate's own test_scale_5k convention,
        // scaled up), M=16 PQ subspaces. Reports recall@10 against a shared
        // brute-force ground truth, median/p95 query latency over >=200
        // timed queries after warmup, and the PQ-arm's build-time overhead
        // vs the exact arm.
        use rand::prelude::*;
        use std::time::Instant;

        let n = 100_000;
        let dim = 128;
        let k = 10;
        let m = 16; // PQ subspaces

        let mut rng = rand::rngs::StdRng::seed_from_u64(0xD15CA77);
        let data: Vec<(String, Vec<f32>)> = (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
                (format!("v{i}"), v)
            })
            .collect();

        // Ground truth: brute-force top-k for a fixed, shared query set.
        let num_gt_queries = 50;
        let mut gt_rng = rand::rngs::StdRng::seed_from_u64(0xBEEF);
        let query_indices: Vec<usize> = (0..num_gt_queries)
            .map(|_| gt_rng.gen_range(0..n))
            .collect();
        let ground_truth: Vec<std::collections::HashSet<String>> = query_indices
            .iter()
            .map(|&qi| {
                let query = &data[qi].1;
                let mut brute: Vec<(usize, f32)> = data
                    .iter()
                    .enumerate()
                    .map(|(i, (_, v))| (i, crate::distance::l2_squared(v, query)))
                    .collect();
                brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                brute[..k].iter().map(|(i, _)| data[*i].0.clone()).collect()
            })
            .collect();

        let measure_arm = |label: &str, pq_subspaces: usize| -> (u128, f64, u128, u128) {
            let mut index = DiskAnnIndex::new(DiskAnnConfig {
                dim,
                max_degree: 48,
                build_beam: 96,
                search_beam: 64,
                alpha: 1.2,
                pq_subspaces,
                pq_iterations: 8,
                ..Default::default()
            });
            index.insert_batch(data.clone()).unwrap();

            let t0 = Instant::now();
            index.build().unwrap();
            let build_ms = t0.elapsed().as_millis();

            // Recall@10 over the shared ground-truth query set.
            let mut total_recall = 0.0;
            for (qi, gt) in query_indices.iter().zip(ground_truth.iter()) {
                let query = &data[*qi].1;
                let results = index.search(query, k).unwrap();
                let found: std::collections::HashSet<String> =
                    results.iter().map(|r| r.id.clone()).collect();
                total_recall += gt.intersection(&found).count() as f64 / k as f64;
            }
            let avg_recall = total_recall / num_gt_queries as f64;

            // Latency: warmup + >=200 timed queries.
            let warmup = 20;
            let timed = 200;
            let mut lrng = rand::rngs::StdRng::seed_from_u64(0xFACEB00C);
            let query_pool: Vec<usize> = (0..(warmup + timed))
                .map(|_| lrng.gen_range(0..n))
                .collect();

            for &qi in query_pool.iter().take(warmup) {
                let _ = index.search(&data[qi].1, k).unwrap();
            }
            let mut latencies_us: Vec<u128> = Vec::with_capacity(timed);
            for &qi in query_pool.iter().skip(warmup) {
                let t0 = Instant::now();
                let _ = index.search(&data[qi].1, k).unwrap();
                latencies_us.push(t0.elapsed().as_micros());
            }
            latencies_us.sort_unstable();
            let median_us = latencies_us[latencies_us.len() / 2];
            let p95_us = latencies_us[(latencies_us.len() as f64 * 0.95) as usize];

            println!(
                "[{label}] build={build_ms}ms recall@{k}={avg_recall:.3} \
                 median={median_us}us p95={p95_us}us (n={n} dim={dim} pq_subspaces={pq_subspaces})"
            );
            (build_ms, avg_recall, median_us, p95_us)
        };

        let exact = measure_arm("exact-traversal (pq_subspaces=0)", 0);
        let pq_guided = measure_arm("PQ-guided (pq_subspaces=16)", m);

        println!(
            "\n=== #673 A/B summary (n={n}, dim={dim}, M={m}) ===\n\
             exact:      build={}ms recall@10={:.3} median={}us p95={}us\n\
             PQ-guided:  build={}ms recall@10={:.3} median={}us p95={}us\n\
             PQ arm build-time overhead vs exact arm: {}ms",
            exact.0,
            exact.1,
            exact.2,
            exact.3,
            pq_guided.0,
            pq_guided.1,
            pq_guided.2,
            pq_guided.3,
            pq_guided.0 as i64 - exact.0 as i64
        );
    }
}
