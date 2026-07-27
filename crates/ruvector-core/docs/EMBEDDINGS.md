# Text Embeddings for AgenticDB

This guide explains how to use real text embeddings with AgenticDB in ruvector-core.

## Quick Start

### Default (Hash-based - Testing Only)

```rust
use ruvector_core::{AgenticDB, types::DbOptions};

let mut options = DbOptions::default();
options.dimensions = 128;
options.storage_path = "agenticdb.db".to_string();

// Uses hash-based embeddings by default (fast but not semantic)
let db = AgenticDB::new(options)?;

// Store and retrieve episodes
let episode_id = db.store_episode(
    "Solve math problem".to_string(),
    vec!["read".to_string(), "calculate".to_string()],
    vec!["got 42".to_string()],
    "Should show work".to_string(),
)?;
```

⚠️ **Warning**: Hash-based embeddings don't understand semantic meaning!
- "dog" and "cat" will NOT be similar
- "dog" and "god" WILL be similar (same characters)

## Production: API-based Embeddings (Recommended)

### OpenAI

```rust
use ruvector_core::{AgenticDB, ApiEmbedding, types::DbOptions};
use std::sync::Arc;

let mut options = DbOptions::default();
options.dimensions = 1536; // text-embedding-3-small
options.storage_path = "agenticdb.db".to_string();

let api_key = std::env::var("OPENAI_API_KEY")?;
let provider = Arc::new(ApiEmbedding::openai(&api_key, "text-embedding-3-small"));

let db = AgenticDB::with_embedding_provider(options, provider)?;

// Now you have semantic embeddings!
let episodes = db.retrieve_similar_episodes("mathematics", 5)?;
```

**OpenAI Models:**
- `text-embedding-3-small` - 1536 dims, $0.02/1M tokens (recommended)
- `text-embedding-3-large` - 3072 dims, $0.13/1M tokens (best quality)

### Cohere

```rust
let api_key = std::env::var("COHERE_API_KEY")?;
let provider = Arc::new(ApiEmbedding::cohere(&api_key, "embed-english-v3.0"));

let mut options = DbOptions::default();
options.dimensions = 1024; // Cohere embedding size

let db = AgenticDB::with_embedding_provider(options, provider)?;
```

### Voyage AI

```rust
let api_key = std::env::var("VOYAGE_API_KEY")?;
let provider = Arc::new(ApiEmbedding::voyage(&api_key, "voyage-2"));

let mut options = DbOptions::default();
options.dimensions = 1024; // voyage-2 size

let db = AgenticDB::with_embedding_provider(options, provider)?;
```

## Custom Embedding Provider

Implement the `EmbeddingProvider` trait for any embedding system:

```rust
use ruvector_core::embeddings::EmbeddingProvider;
use ruvector_core::error::Result;

struct MyCustomEmbedding {
    // Your model here
}

impl EmbeddingProvider for MyCustomEmbedding {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Your embedding logic
        todo!()
    }

    fn dimensions(&self) -> usize {
        384 // Your embedding dimensions
    }

    fn name(&self) -> &str {
        "MyCustomEmbedding"
    }
}
```

## ONNX Runtime (Local, No API Costs)

For production use without API costs, use ONNX Runtime with pre-exported models:

```rust
// See examples/onnx-embeddings for complete implementation
use ort::{Session, Environment, Value};

struct OnnxEmbedding {
    session: Session,
    dimensions: usize,
}

impl EmbeddingProvider for OnnxEmbedding {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Tokenize text
        // Run ONNX inference
        // Return embeddings
        todo!()
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        "OnnxEmbedding"
    }
}
```

### Exporting Models to ONNX

```python
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model_id = "sentence-transformers/all-MiniLM-L6-v2"
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained("./onnx-model")
tokenizer.save_pretrained("./onnx-model")
```

## Native Pure-Rust (Local, No API Costs, No Export Step)

`LatticeEmbedding` runs BERT-family embedding models in-process with no Python, no ONNX
export step, and no network calls after the first model download. It is already in this
crate behind the `lattice-embeddings` feature.

```toml
[dependencies]
ruvector-core = { version = "2.3", features = ["lattice-embeddings"] }
```

```rust
use ruvector_core::{AgenticDB, embeddings::LatticeEmbedding, types::DbOptions};
use std::sync::Arc;

let provider = Arc::new(LatticeEmbedding::from_pretrained("bge-small-en-v1.5")?);

let mut options = DbOptions::default();
options.dimensions = 384; // bge-small-en-v1.5
options.storage_path = "agenticdb.db".to_string();

let db = AgenticDB::with_embedding_provider(options, provider)?;
```

BERT-family models (BGE, E5, MiniLM) download from HuggingFace into `~/.lattice/models`
on first use. Vectors are L2-normalized, so a dot-product index works as well as cosine.

A runnable example is in `crates/ruvector-core/examples/lattice_embedding_example.rs`.

**Requires Rust 1.93.** `lattice-embed` 0.6.1 declares `rust-version = 1.93` (edition 2024),
and cargo enforces that before compiling, so enabling this feature raises the toolchain
floor for the whole build. That is why it is opt-in rather than a default.

### Asymmetric models and query embedding

This is worth knowing before choosing a model, because it is invisible from the call site.

BGE, E5, and Qwen3-Embedding are *asymmetric* retrievers: the query side is prefixed with a
retrieval instruction, the document side is not. `LatticeEmbedding` follows that —
`EmbeddingProvider::embed` is the passage side, and the inherent `LatticeEmbedding::embed_query`
applies the query instruction.

`AgenticDB` reaches its provider through the `EmbeddingProvider` trait, which has one
role-neutral `embed`. Its search entry points (`retrieve_similar_episodes`, `search_skills`,
`find_relevant_turns`, `search`) call the same method its indexing paths call, so queries
issued through `AgenticDB` are embedded with the **passage** side and the query instruction
is not applied.

What that means in practice:

- **MiniLM is unaffected.** It is genuinely symmetric — contrastive training on raw text, no
  prefix — so its query and passage sides are equivalent and nothing is lost.
- **On BGE, E5, or Qwen3-Embedding**, retrieval quality through `AgenticDB` will be lower than
  the same model scores in a harness that embeds queries with `embed_query`. Stored documents
  are correct either way; it is the query side that differs.

If you want asymmetric retrieval at full strength today, embed queries yourself with
`LatticeEmbedding::embed_query` and query the vector index directly rather than through
`AgenticDB`'s text-search helpers. Whether `AgenticDB` should offer a query-side path is a
design question for this project, not something this provider can decide from underneath.

## Feature Flags

### `real-embeddings` (Optional)

This feature flag enables the `CandleEmbedding` type (currently a stub):

```toml
[dependencies]
ruvector-core = { version = "0.1", features = ["real-embeddings"] }
```

However, we recommend using API-based providers instead of implementing Candle integration yourself.

## Complete Example

```rust
use ruvector_core::{AgenticDB, ApiEmbedding, types::DbOptions};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup
    let api_key = std::env::var("OPENAI_API_KEY")?;
    let provider = Arc::new(ApiEmbedding::openai(&api_key, "text-embedding-3-small"));

    let mut options = DbOptions::default();
    options.dimensions = 1536;
    options.storage_path = "agenticdb.db".to_string();

    let db = AgenticDB::with_embedding_provider(options, provider)?;

    println!("Using: {}", db.embedding_provider_name());

    // Store reflexion episodes
    let ep1 = db.store_episode(
        "Debug memory leak in Rust".to_string(),
        vec!["profile".to_string(), "find leak".to_string()],
        vec!["fixed with Arc".to_string()],
        "Should explain reference counting".to_string(),
    )?;

    let ep2 = db.store_episode(
        "Optimize Python performance".to_string(),
        vec!["profile".to_string(), "vectorize".to_string()],
        vec!["10x speedup".to_string()],
        "Should mention NumPy".to_string(),
    )?;

    // Semantic search - will find Rust episode for memory-related query
    let episodes = db.retrieve_similar_episodes("memory management", 5)?;
    for episode in episodes {
        println!("Task: {}", episode.task);
        println!("Critique: {}", episode.critique);
    }

    // Create skills
    db.create_skill(
        "Memory Profiling".to_string(),
        "Profile application memory usage to find leaks".to_string(),
        Default::default(),
        vec!["valgrind".to_string(), "massif".to_string()],
    )?;

    // Search skills semantically
    let skills = db.search_skills("finding memory leaks", 3)?;
    for skill in skills {
        println!("Skill: {} - {}", skill.name, skill.description);
    }

    Ok(())
}
```

## Performance Considerations

### API-based (OpenAI, Cohere, Voyage)
- **Pros**: Always up-to-date, no model storage, easy to use
- **Cons**: Network latency, API costs, requires internet
- **Best for**: Production apps with internet access

### ONNX Runtime (Local)
- **Pros**: No API costs, offline support, fast inference
- **Cons**: Model storage (~100MB), setup complexity
- **Best for**: Edge deployment, high-volume apps

### Native Pure-Rust (`LatticeEmbedding`)
- **Pros**: No API costs, offline after first download, no ONNX export step, no Python
- **Cons**: Model storage (~130MB for bge-small), raises the build's Rust floor to 1.93
- **Best for**: Local and offline deployments that would rather not maintain an export pipeline

### Hash-based (Default)
- **Pros**: Zero dependencies, instant, no setup
- **Cons**: Not semantic, only for testing
- **Best for**: Development, unit tests

## Recommendations

1. **Development/Testing**: Use hash-based (default)
2. **Production (Cloud)**: Use `ApiEmbedding::openai()`
3. **Production (Edge/Offline)**: Use `LatticeEmbedding` (no export step), or implement an ONNX provider
4. **Custom Models**: Implement `EmbeddingProvider` trait

## Migration Path

```rust
// Start with hash for development
let db = AgenticDB::new(options)?;

// Switch to API for staging
let provider = Arc::new(ApiEmbedding::openai(&api_key, "text-embedding-3-small"));
let db = AgenticDB::with_embedding_provider(options, provider)?;

// Move to ONNX for production scale
let provider = Arc::new(OnnxEmbedding::from_file("model.onnx")?);
let db = AgenticDB::with_embedding_provider(options, provider)?;
```

The beauty is: **your AgenticDB code doesn't change**, just the provider!

## Error Handling

```rust
use ruvector_core::error::RuvectorError;

match AgenticDB::with_embedding_provider(options, provider) {
    Ok(db) => {
        // Use db
    }
    Err(RuvectorError::InvalidDimension(msg)) => {
        eprintln!("Dimension mismatch: {}", msg);
    }
    Err(RuvectorError::ModelLoadError(msg)) => {
        eprintln!("Failed to load model: {}", msg);
    }
    Err(RuvectorError::ModelInferenceError(msg)) => {
        eprintln!("Inference failed: {}", msg);
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

## See Also

- [AgenticDB API Documentation](../src/agenticdb.rs)
- [Embedding Provider Trait](../src/embeddings.rs)
- [ONNX Examples](../../examples/onnx-embeddings/)
- [Integration Tests](../tests/embeddings_test.rs)
