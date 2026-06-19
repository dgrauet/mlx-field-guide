# Embeddings & RAG: CUDA vs MLX

> **Status:** 🟢 [CUDA](../glossary.md#cuda) (Mature -- sentence-transformers, every vector DB, full RAG stack) | 🟡 MLX (Emerging -- embedding generation works; the surrounding tooling is thin but the vector stores are platform-agnostic)

## What This Domain Covers

Turning text (or images) into vectors that capture meaning, and then using those vectors to retrieve relevant information. This is the foundation of semantic search, recommendation, deduplication, and -- most visibly -- **Retrieval-Augmented Generation (RAG)**: the pattern where an [LLM](03-llms.md) answers a question using documents fetched from a vector store rather than from its frozen weights.

A RAG system has two model-bearing halves. The **embedding model** converts both your documents and the user's query into vectors in the same [latent space](../glossary.md#latent-space), so that "closeness" in that space means "related in meaning." The **generation model** (the LLM) reads the retrieved chunks and writes the answer. Everything in between -- chunking, the vector index, the similarity search -- is plain data infrastructure with no GPU framework dependency.

That split is the key to understanding the MLX story here. The LLM half is well covered (see [LLMs](03-llms.md)). The embedding half is where the ecosystem gap lives. And the infrastructure half doesn't care whether you are on CUDA or Apple Silicon at all.

---

## The CUDA Ecosystem

### Embedding Models

**sentence-transformers** (SBERT) is the de facto standard. It wraps BERT-class encoders with mean-pooling and a clean `model.encode(texts)` API, ships hundreds of pre-trained models, and handles batching, normalization, and pooling automatically. The strong open embedding models are almost all distributed as sentence-transformers checkpoints:

```
COMMON OPEN EMBEDDING MODELS (CUDA-native via sentence-transformers)

  Model family     Dim     Notes
  ------------     ---     -----
  BGE (BAAI)       384-1024  bge-small/base/large; strong MTEB scores
  GTE (Alibaba)    384-1024  gte-small/base/large; multilingual variants
  E5 (Microsoft)   384-1024  instruction-prefixed ("query:" / "passage:")
  Nomic Embed      768       long-context (8k), open weights + data
  mxbai-embed      1024      mixedbread; competitive small-model scores
  Jina Embeddings  512-1024  long-context, multilingual

  All evaluated on MTEB (Massive Text Embedding Benchmark).
```

For high-throughput embedding generation, **Text Embeddings Inference (TEI)** from Hugging Face plays the role vLLM plays for generation: batched, server-class, CUDA-optimized embedding at thousands of texts per second.

### Vector Stores & Orchestration

This layer is **framework-agnostic** -- it stores and searches float arrays and never touches CUDA or MLX:

```
THE RAG INFRASTRUCTURE STACK (platform-independent)

  Vector indexes:
    FAISS          in-process, C++ core, CPU or CUDA index
    hnswlib        in-process HNSW graph index, CPU
    Chroma         embedded or server, batteries-included
    LanceDB        embedded, columnar, on-disk, Rust core
    Qdrant         server, production-grade filtering
    Milvus, Weaviate, pgvector ...

  Orchestration frameworks:
    LlamaIndex     ingestion -> index -> query engines
    LangChain      chains, retrievers, agents
    Haystack       pipelines for search + RAG
```

The important fact: **FAISS (CPU), hnswlib, Chroma, LanceDB, and Qdrant all run natively on macOS.** The orchestration frameworks are pure Python. None of them require CUDA. The only piece that benefits from a GPU is generating the embedding vectors -- and that is exactly where MLX can step in.

---

## The MLX Ecosystem

### Embedding Generation

**mlx-embeddings** (community, Prince Canuma) is the closest analogue to sentence-transformers. It loads encoder models -- BERT, XLM-RoBERTa, ModernBERT and related architectures -- in MLX format and produces pooled embeddings on Apple Silicon. You call the model directly and read the pooled vector off its output:

```python
from mlx_embeddings.utils import load

model, tokenizer = load("mlx-community/all-MiniLM-L6-v2-4bit")

input_ids = tokenizer.encode("how does attention work", return_tensors="mlx")
outputs = model(input_ids)
text_embeds = outputs.text_embeds          # mean-pooled, normalized vector
```

Because the strong open embedding models -- BGE, GTE, E5 -- are themselves BERT-class encoders, they fall within the architecture families mlx-embeddings supports, but availability is per-*checkpoint*: a model is usable only once someone has converted those specific weights to MLX. Track the repo for the current list of supported architectures and pooling modes; it is the fastest-moving part of this domain.

If a model you need is not yet ported, you have two fallbacks, both viable:

1. **Run the encoder through mlx-lm-style conversion.** Encoder models are simpler than decoder LLMs (no [KV cache](../glossary.md#kv-cache), no causal masking); porting a new BERT-class encoder is one of the most accessible MLX porting tasks (see [Porting Guide](../03-contributing/02-porting-guide.md)).
2. **Generate embeddings on CPU or with a non-MLX library and feed the vectors into the same vector store.** Because the infrastructure layer is platform-agnostic, nothing forces the embedding step to be MLX.

### Vector Stores on Apple Silicon

There is no "MLX vector database" and there does not need to be one. FAISS (CPU build via `faiss-cpu`), Chroma, LanceDB, and Qdrant install and run on macOS with no changes. A fully local RAG stack on a Mac looks like:

```
LOCAL MLX RAG STACK (entirely on-device)

  Documents
     | chunk (LlamaIndex / manual)
     v
  mlx-embeddings  ->  vectors  ->  LanceDB / Chroma (on-disk)
                                        |
  User query --> mlx-embeddings --> query vector
                                        |
                                   similarity search (top-k chunks)
                                        |
                                   mlx-lm  -->  grounded answer

  Every box runs on the Mac. No CUDA, no cloud, no API key.
```

This is genuinely a place where Apple Silicon's [unified memory](../glossary.md#unified-memory) is an advantage: the embedding model, the vector index, and a sizeable LLM can coexist in the same memory pool without the host/device copies a CUDA box pays for.

### Orchestration

LlamaIndex and LangChain are pure Python and run on macOS, but their *default* embedding and LLM backends assume PyTorch/CUDA or a hosted API. To use MLX you wire in a custom embedding function (wrapping `mlx-embeddings`) and a custom LLM (wrapping `mlx-lm`). This glue is straightforward but not yet packaged as a first-class integration -- which is itself a contribution opportunity (below).

---

## Gap Analysis

| Capability | CUDA | MLX | Gap |
|-----------|------|-----|-----|
| **Embedding model coverage** | 🟢 Every MTEB model via sentence-transformers | 🟡 Popular BGE/GTE/E5 ported; long tail missing | Medium -- mainstream models available; niche/new ones need a port |
| **Embedding API ergonomics** | 🟢 `model.encode()` one-liner, auto pooling/batching | 🟡 mlx-embeddings covers the common path; fewer pooling options | Small -- the 90% case is one function call |
| **High-throughput batch embedding** | 🟢 TEI: thousands/sec, batched server | 🟡 Single-process generation; no batched embedding server | Medium -- fine for indexing a personal corpus; slow for millions of docs |
| **Vector stores** | 🟢 All, incl. CUDA FAISS GPU index | 🟢 FAISS-CPU, Chroma, LanceDB, Qdrant all run on macOS | None for typical corpora -- only the GPU-accelerated FAISS index is CUDA-only |
| **RAG orchestration** | 🟢 LlamaIndex/LangChain default integrations | 🟡 Works via custom embedding/LLM wrappers; no turnkey MLX backend | Small -- needs glue code, not new infrastructure |
| **Multimodal embeddings (CLIP)** | 🟢 open_clip, full ecosystem | 🟡 CLIP image/text encoders portable to MLX; less packaged | Medium -- CLIP-class models work but tooling is thin |
| **Reranking (cross-encoders)** | 🟢 sentence-transformers cross-encoders, Cohere/BGE rerankers | 🔴 Little MLX-native reranker tooling | Medium -- rerankers meaningfully improve RAG quality; largely unported |

The headline: **the gap is narrow and lives almost entirely in the embedding-generation layer.** Retrieval infrastructure is a solved, portable problem. For a developer building a local RAG app on a Mac, the practical workflow already exists today; it just has rougher edges than `pip install sentence-transformers`.

---

## What the Gap Actually Costs You

If you are building a **personal or single-user RAG app** (chat-with-your-docs, a local knowledge base): almost no real cost. mlx-embeddings + LanceDB + mlx-lm gives you a fully on-device, private RAG pipeline that a CUDA laptop cannot match for privacy or zero-API-cost operation.

If you are **indexing millions of documents** where embedding throughput dominates: CUDA's TEI batched server is meaningfully faster. You would embed on a CUDA box (or CPU, patiently) and then serve queries locally.

If you need **reranking or exotic embedding models** for top-tier retrieval quality: you will hit unported-model walls and may fall back to CPU PyTorch or a hosted API for those components.

---

## Contributing Here

This is one of the highest-leverage, most-accessible corners of the MLX ecosystem, because encoder models are simpler to port than generative ones:

1. **Port more embedding models to MLX.** Each new BGE/GTE/E5/Nomic checkpoint converted and uploaded to `mlx-community` directly expands what RAG builders can use. Encoders have no KV cache and no autoregressive loop -- a good first port.
2. **A first-class LlamaIndex / LangChain MLX backend.** A maintained `MLXEmbedding` and `MLXLLM` integration would turn "wire up glue code" into `pip install`. High impact, mostly Python.
3. **Cross-encoder rerankers in MLX.** Reranking is the cheapest large quality win in RAG and is almost entirely unported.
4. **A batched embedding server** (the TEI role for MLX) for people indexing large corpora locally.

See [Open Opportunities](../03-contributing/03-open-opportunities.md) for how these rank against other gaps.

---

## Sources

- sentence-transformers: [sbert.net](https://www.sbert.net/)
- MTEB leaderboard: [huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- Text Embeddings Inference (TEI): [github.com/huggingface/text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference)
- mlx-embeddings: [github.com/Blaizzy/mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)
- FAISS: [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
- LanceDB: [github.com/lancedb/lancedb](https://github.com/lancedb/lancedb)
- Chroma: [github.com/chroma-core/chroma](https://github.com/chroma-core/chroma)
- LlamaIndex: [github.com/run-llama/llama_index](https://github.com/run-llama/llama_index)
- LangChain: [github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)
- BGE (FlagEmbedding): [github.com/FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)

---

## See Also

- [LLMs](03-llms.md) -- the generation half of RAG; mlx-lm is the model that consumes retrieved chunks
- [Latent Space](../01-foundations/07-latent-space.md) -- why "nearby vectors mean related meaning"; the conceptual basis of semantic search
- [Tokenization](../01-foundations/09-tokenization.md) -- how text becomes the token IDs an embedding model ingests
- [Vision-Language Models](11-vision-language-models.md) -- multimodal embeddings (CLIP) live at the intersection of this page and VLMs
- [Porting Guide](../03-contributing/02-porting-guide.md) -- encoder models are among the most approachable MLX ports
