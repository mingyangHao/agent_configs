# CuTile Programming — Domain Knowledge Reference

## Overview

**CuTile** (`cuda.tile` / `ct`) is a high-level Python-embedded CUDA kernel DSL. TileGym wraps it with a dispatch/backend system for LLM inference kernels. Kernels are written in Python with decorators and compiled JIT for NVIDIA GPUs (sm_90 Hopper, sm_100 Blackwell, sm_120 B200).

---

## 1. Kernel Definition

```python
import cuda.tile as ct

@ct.kernel                                    # basic
@ct.kernel(occupancy=2)                       # occupancy hint
@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))    # per-GPU CTA count
def my_kernel(A, B, C, PARAM: ct.Constant[int]):
    ...
```

- `ct.Constant[T]` — compile-time constant (enables JIT specialization)
- `occupancy` — controls thread-block scheduling density
- `num_ctas` — CTAs per SM (for persistent kernels); `ct.ByTarget(...)` for per-arch values

---

## 2. Block / Thread Indexing

| API | Purpose |
|-----|---------|
| `ct.bid(axis)` | Block index along axis |
| `ct.tid(axis)` | Thread index along axis |
| `ct.num_blocks(axis)` | Total blocks along axis |
| `ct.cdiv(a, b)` | Ceiling division |
| `ct.num_tiles(tensor, axis, shape)` | Number of tiles along axis |

---

## 3. Data Movement

### Load / Store (structured, 2D tile access)
```python
tile = ct.load(A, index=(row_tile, col_tile), shape=(TILE_M, TILE_K),
               padding_mode=ct.PaddingMode.ZERO)
ct.store(C, index=(row_tile, col_tile), tile=result)
```

### Gather / Scatter (1D or irregular access)
```python
offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
x_tile = ct.gather(x, offsets, padding_value=0)
ct.scatter(y, offsets, y_tile)

# 2D gather/scatter
a = ct.gather(input, (row_idx, col_indices), check_bounds=True)
ct.scatter(output, (row_idx, col_indices), result, check_bounds=True)
```

### Padding Modes
- `ct.PaddingMode.ZERO` — zero-fill OOB
- `ct.PaddingMode.NEG_INF` — negative infinity (for softmax/attention masks)
- `ct.PaddingMode.REPLICATE` — edge replication

---

## 4. Computation Primitives

### Tensor Core MMA
```python
acc = ct.full((TILE_M, TILE_N), 0, dtype=ct.float32)
acc = ct.mma(a, b, acc)   # matrix multiply-accumulate
```

### Element-wise
```python
ct.add(a, b)    ct.sub(a, b)    ct.mul(a, b, flush_to_zero=True)
ct.truediv(a, b, flush_to_zero=True, rounding_mode=RMd.APPROX)
ct.exp(x)       ct.exp2(x)      ct.rsqrt(x)
ct.maximum(a, b)   ct.where(cond, a, b)
```

### Reductions
```python
ct.sum(tensor, axis=1, keepdims=True)
ct.max(tensor, axis=1, keepdims=True)
ct.min(tensor, axis=0)
```

### Shape / Type
```python
ct.astype(tensor, ct.float32)     # type cast
ct.reshape(tensor, (M, N))        # reshape (no data move)
ct.transpose(tensor)              # transpose
ct.permute(tensor, dims)          # permute
ct.broadcast_to(tensor, shape)    # broadcast
ct.expand_dims(tensor, axis)      # add dimension
ct.extract(tensor, ...)           # extract sub-tile
```

### Creation
```python
ct.full(shape, value, dtype)
ct.zeros(shape, dtype)
ct.arange(n, dtype)
```

### Bitwise
```python
ct.bitwise_xor(a, b)   ct.bitwise_and(a, b)
ct.bitwise_lshift(a, n)  ct.bitwise_rshift(a, n)
```

---

## 5. Kernel Launch

### Simple launch
```python
ct.launch(torch.cuda.current_stream(), grid, kernel_fn, (arg1, arg2, CONST1, CONST2))
```

### Autotune launch
```python
from cuda.tile_experimental import autotune_launch
autotune_launch(
    stream,
    grid_fn=lambda cfg: (ct.cdiv(M, cfg.TILE_M), ct.cdiv(N, cfg.TILE_N), 1),
    kernel=my_kernel,
    args_fn=lambda cfg: (A, B, C, cfg.TILE_M, cfg.TILE_N, cfg.TILE_K),
    hints_fn=lambda cfg: {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy},
    search_space=config_iterator,
)
```

---

## 6. Common Kernel Patterns

### Pattern A: Element-wise (1D gather/scatter)
```python
@ct.kernel
def relu_kernel(x, y, n: ct.Constant[int], BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    x_tile = ct.gather(x, offsets, padding_value=0)
    x_f32 = ct.astype(x_tile, ct.float32)
    y_f32 = ct.maximum(x_f32, ct.zeros((BLOCK,), dtype=ct.float32))
    ct.scatter(y, offsets, ct.astype(y_f32, x_tile.dtype))
```

### Pattern B: Tiled GEMM (2D load/store + MMA loop)
```python
@ct.kernel
def matmul(A, B, C, TILE_M: ct.Constant[int], TILE_N: ct.Constant[int], TILE_K: ct.Constant[int]):
    bid_m, bid_n = swizzle_2d(...)           # 2D block mapping
    acc = ct.full((TILE_M, TILE_N), 0, dtype=ct.float32)
    for k in range(num_k_tiles):
        a = ct.load(A, (bid_m, k), (TILE_M, TILE_K), padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, (k, bid_n), (TILE_K, TILE_N), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(ct.astype(a, ct.tfloat32), ct.astype(b, ct.tfloat32), acc)
    ct.store(C, (bid_m, bid_n), ct.astype(acc, C.dtype))
```

### Pattern C: Row-wise reduction (RMSNorm)
```python
@ct.kernel
def rms_norm(x, w, out, N: ct.Constant[int], eps: ct.Constant[float], TILE: ct.Constant[int]):
    row = ct.bid(0)
    sq_sum = ct.full((TILE,), 0.0, dtype=ct.float32)
    for t in range(ct.cdiv(N, TILE)):
        offsets = t * TILE + ct.arange(TILE, dtype=ct.int32)
        x_tile = ct.gather(x, (row, offsets), check_bounds=True)
        sq_sum += x_tile * x_tile
    rms = ct.rsqrt(ct.sum(sq_sum) / N + eps)
    for t in range(ct.cdiv(N, TILE)):
        ...
        ct.scatter(out, (row, offsets), x_tile * rms * w_tile)
```

### Pattern D: Fused SiLU-and-Mul
```python
@ct.kernel
def silu_and_mul(input, output, TILE: ct.Constant[int], H: ct.Constant[int]):
    row = ct.bid(0)
    offs = ct.arange(TILE, dtype=ct.int32)
    a = ct.gather(input, (row, offs), check_bounds=True)
    b = ct.gather(input, (row, offs + H), check_bounds=True)
    a, b = ct.astype(a, ct.float32), ct.astype(b, ct.float32)
    sigmoid_a = ct.truediv(1.0, 1.0 + ct.exp(-a), flush_to_zero=True)
    result = a * sigmoid_a * b
    ct.scatter(output, (row, offs), ct.astype(result, input.dtype))
```

### Pattern E: Online Softmax in FMHA
```python
m_prev = ct.full(..., float('-inf'))
l_prev = ct.zeros(...)
acc = ct.zeros(...)
for k_tile in range(...):
    qk = ct.mma(q, k_T, ...)
    m_new = ct.maximum(m_prev, ct.max(qk, axis=1, keepdims=True))
    correction = ct.exp(m_prev - m_new)
    p = ct.exp(qk - m_new)
    l_new = correction * l_prev + ct.sum(p, axis=1, keepdims=True)
    acc = correction * acc + ct.mma(p, v, ...)
    m_prev, l_prev = m_new, l_new
acc = acc / l_new
```

### Pattern F: Static Persistent Kernel
```python
@ct.kernel(occupancy=2, num_ctas=2)
def persistent_kernel(A, B, C, ...):
    start_bid = ct.bid(0)
    num_programs = ct.num_blocks(0)
    for tile_id in range(start_bid, total_tiles, num_programs):
        bid_m, bid_n = decompose(tile_id, ...)
        ...
```

---

## 7. TileGym Dispatch Architecture

```python
@dispatch("matmul", fallback_backend="pytorch")
def matmul(a, b, ...): raise NotImplementedError

@register_impl("matmul", "cutile")
def matmul_cutile(a, b, ...): ct.launch(...)

# Usage
tilegym.set_backend("cutile")
tilegym.ops.matmul(a, b)
```

---

## 8. PyTorch Autograd Integration

```python
class SiLUAndMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ct.launch(stream, grid, fwd_kernel, (...))
        ctx.save_for_backward(input)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        ct.launch(stream, grid, bwd_kernel, (...))
        return grad_input
```

---

## 9. Key Idioms

1. **Accumulate in fp32** — `ct.full(..., 0, dtype=ct.float32)`, cast back at store
2. **TF32 for fp32 inputs** — `ct.tfloat32` for MMA
3. **2D swizzling** — GROUP_SIZE_M for L2 cache locality
4. **Persistent threads** — occupancy + num_ctas + for-loop; grid = SM_count x occupancy
5. **flush_to_zero** — on `ct.mul`/`ct.truediv` for perf
6. **rounding_mode=RMd.APPROX** — fast approximate division
7. **Padding guards** — `padding_mode` or `check_bounds=True`
8. **Autotuning** — config search space with TILE_M/N/K, GROUP_SIZE_M, occupancy

---

## 10. Key Files

- TileGym source: `tekit/TileGym/src/tilegym/`
- CuTile ops: `tekit/TileGym/src/tilegym/ops/cutile/`
- Dispatch system: `tekit/TileGym/src/tilegym/backend/dispatcher.py`
- Real-world kernel: `tekit/tensorrt_llm/_torch/attention_backend/sparse/mewtwo/kernel_cutile.py`
