# Modern LLM Inference TODO

## Core Framework
- [ ] Define transformer configuration structs (layers, heads, dims, vocab, rotary bases) and wire them to a graph builder.
- [ ] Flesh out `internal/dag` with executable graph nodes supporting embeddings, attention blocks, MLPs, and output heads.
- [ ] Extend `internal/tensor` with reshape/transpose, broadcast add, bias views, and GPU-backed matmul helpers.
- [ ] Implement KV-cache data structures for autoregressive decoding (per-layer ring buffers, mixed-precision views, eviction policy).

## Metal Kernels
- [ ] Add optimized GEMM kernels (column-major variants, fused bias/activation, batched projections) for Q/K/V/O and MLP weights.
- [ ] Implement rotary embedding application and Q/K/V packing kernels that produce interleaved layouts expected by attention.
- [ ] Create scaled dot-product attention kernels with causal masking, softmax (log-sum-exp), and value aggregation fused where possible.
- [ ] Provide elementwise kernels for RMSNorm, LayerNorm, SwiGLU/GEGLU, bias add, residual add, and dropout stubs (no-op during inference).
- [ ] Introduce int8/int4 matmul + dequant kernels with per-channel scale/zero-point buffers to support Gemma/Qwen quantization.

## Model Assets & Loading
- [ ] Implement safetensors and GGUF readers that stream weights directly into Metal buffers without staging full copies in RAM.
- [ ] Build conversion scripts to ingest official Gemma3/Qwen3 checkpoints (sharding, quantization, rope bases) into repository format.
- [ ] Define model manifests capturing tensor metadata, tokenizer info, and weight shards for deterministic loading.
- [ ] Add integrity verification (hashing) and caching to avoid redundant host→GPU transfers across runs.

## Runtime & UX
- [ ] Develop an autoregressive decode loop with prompt ingestion, iterative token generation, and configurable batch size.
- [ ] Implement sampling strategies (greedy, temperature, top-k, top-p, repetition penalty) operating on GPU logits with minimal host sync.
- [ ] Integrate SentencePiece/BPE tokenizers for Gemma/Qwen along with detokenization and special token handling.
- [ ] Upgrade `cmd/fastgo` CLI into a model loader that warms kernels, streams tokens, and surfaces telemetry (tokens/s, latency).

## Testing & Tooling
- [ ] Add deterministic fixture generators and golden-output tests for each kernel (small tensors with precomputed expected results).
- [ ] Create integration tests that run a minimal transformer layer stack entirely on GPU and compare against stored goldens.
- [ ] Build benchmarking harnesses tracking per-layer latency and end-to-end tokens/s to guide optimization.
- [ ] Set up CI automation for `gofmt`, `go vet`, `go test`, Metal compilation smoke tests, and sanitizer runs on macOS agents.

## Performance & Stability
- [ ] Implement GPU memory pooling, command-buffer reuse, and async execution to minimize per-token overhead.
- [ ] Add telemetry/logging for kernel timings, pipeline cache hits, and memory usage to aid profiling and regression detection.
- [ ] Handle hardware capability detection (Metal feature sets) and fallbacks for unsupported features while keeping execution on GPU.
- [ ] Document system requirements, weight download workflow, and provide scripts for verifying model asset integrity.