# Project Memory

## Codebase: tekit

### CuTile / TileGym
- Full cutile programming reference saved in [cutile_reference.md]
- CuTile (`cuda.tile` / `ct`) = high-level Python CUDA kernel DSL with JIT compilation
- TileGym = dispatch framework wrapping cutile ops for LLM inference
- Key source paths:
  - TileGym source: `tekit/TileGym/src/tilegym/`
  - CuTile kernel implementations: `tekit/TileGym/src/tilegym/ops/cutile/`
  - Backend dispatch: `tekit/TileGym/src/tilegym/backend/dispatcher.py`
  - Real-world sparse attention kernel: `tekit/tensorrt_llm/_torch/attention_backend/sparse/mewtwo/kernel_cutile.py`
- 6 canonical patterns: element-wise, tiled GEMM, row-reduction, fused ops, online softmax (FMHA), static persistent
- Target GPUs: sm_90 (H100), sm_100 (Blackwell), sm_120 (B200)
