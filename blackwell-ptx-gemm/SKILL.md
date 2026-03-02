---
name: blackwell-ptx-gemm
description: Manual PTX patterns for production Blackwell (SM100a) GEMM and FMHA kernels using raw tcgen05.mma, TMA, SMEM/instruction descriptors, warp specialization, and inline PTX masking. Covers createSmemDesc, make_utcmma_desc, incrSmemAddr, gather TMA, TMEM load/store, warp role assignment, dynamic register allocation, pipeline wrappers, and per-element attention masking. Use when writing hand-tuned Blackwell kernels, FMHA, or fused operators that need manual descriptor management beyond CuTe abstractions.
---

# Blackwell Manual PTX GEMM & FMHA Patterns

## When to Use Manual PTX

CuTe abstractions (`make_tiled_mma`, `gemm()`, `copy()`) handle standard GEMM pipelines cleanly. Switch to manual PTX when:
- Building FMHA or fused kernels where operand layouts change dynamically
- Needing fine-grained control over descriptor construction and advancement
- Implementing paged/gather TMA access patterns
- Applying data-dependent masks (causal, variable seqlen)
- Performing warp-specialized kernels with heterogeneous roles

## Architecture Quick Reference

```
GMEM ──TMA──▶ SMEM ──descriptor──▶ tcgen05.mma ──▶ TMEM (accumulator)
                                                       │
                                                  tcgen05.ld
                                                       │
                                                       ▼
                                                     RMEM ──store──▶ GMEM/SMEM
```

**TMEM**: 128 rows x 512 columns per SM, accessible only via `tcgen05.ld`/`tcgen05.st` (32x32b tiles).

**tcgen05.mma**: Requires a 64-bit SMEM descriptor per operand and a 64-bit instruction descriptor. Only one thread per CTA (or CTA-group) issues the instruction.

## SMEM Descriptor Construction

### createSmemDesc

Builds the 64-bit descriptor that tells the MMA where to find operands in SMEM:

```cpp
uint64_t descA = trtllm::dev::createSmemDesc(
    &smemQ[stage][0],       // SMEM base pointer
    uint32_t{0x2000000},    // packed: leading_byte_offset=512, stride_byte_offset=0
    uint32_t{0x40004040});  // packed: swizzle pattern (SW128B)
```

64-bit layout:
```
[start_address:14b | leading_byte_offset:14b | stride_byte_offset:14b | base_offset:3b | layout_type:3b]
```

`layout_type` values:
| Value | Swizzle | Constant |
|---|---|---|
| 0 | None (interleave) | `LayoutType::SWIZZLE_NONE` |
| 2 | 128B | `LayoutType::SWIZZLE_128B` |
| 4 | 64B | `LayoutType::SWIZZLE_64B` |
| 6 | 32B | `LayoutType::SWIZZLE_32B` |

The packed uint32 arguments:
- `leadStride`: `(leading_offset << 16) | stride_offset`
- `swizzle`: `(stride_byte_offset << 16) | (base_offset << 8) | layout_type_bits`

**Critical**: The swizzle encoded here must match the SMEM buffer layout and the TMA descriptor swizzle exactly.

### Instruction Descriptor (make_utcmma_desc)

Encodes MMA shape, data types, and operand configuration:

```cpp
uint64_t desc = trtllm::dev::make_utcmma_desc(
    /*a_major=*/1,      // 0=K-major, 1=MN-major
    /*b_major=*/0,      // 0=K-major
    /*sparse=*/0,       // no sparsity
    /*neg_a=*/false,    // negate A
    /*neg_b=*/true,     // negate B (useful for attention: S = Q*K^T − mask)
    /*m_dim=*/128,      // MMA M dimension
    /*n_dim=*/256,      // MMA N dimension
    /*k_dim=*/32,       // MMA K dimension per issue
    /*trans_a=*/true);  // transpose A
```

## Issuing MMA

```cpp
if (cute::elect_one_sync()) {
    cuda_ptx::tcgen05_mma(
        cuda_ptx::kind_f8f6f4,    // MMA data type kind
        cuda_ptx::cta_group_2,     // 2SM mode (use cta_group_1 for 1SM)
        tmemPtrD,                   // TMEM accumulator base address
        smemDescA,                  // 64-bit SMEM descriptor for A
        smemDescB,                  // 64-bit SMEM descriptor for B
        utcmmaDesc,                 // 64-bit instruction descriptor
        /*accumulate=*/readD);      // false=zero-init, true=accumulate
}
```

MMA kind options: `kind_f16`, `kind_tf32`, `kind_f8f6f4`, `kind_i8`, `kind_mxf4nvf4`.

For 2SM mode, use `cta_group_2` and only the leader CTA issues the instruction.

## Advancing Descriptors Along K

After each MMA issue, advance the SMEM descriptor by one K-block:

```cpp
trtllm::dev::incrSmemAddr(smemDescA, /*delta=*/2);
trtllm::dev::incrSmemAddr(smemDescB, /*delta=*/256);
```

The delta is in units of the descriptor's addressing granularity:
`delta = K_block_elements * element_bytes / descriptor_granularity`

## TMA Load Patterns

### Standard TMA with Coordinate Arrays

```cpp
int32_t coords[4];
coords[0] = 0;              // innermost dim offset
coords[1] = 0;              // next dim
coords[2] = headIdx;        // head dimension
coords[3] = seqOffset;      // sequence position

uint64_t* leadCtaMbar = cuda_ptx::mapa(
    cuda_ptx::space_cluster_t{},
    barrier,
    int32_t{trtllm::dev::getLeadCtaRank()});

if (cute::elect_one_sync() && warpIdx == 0) {
    cuda_ptx::cp_async_bulk_tensor(
        cuda_ptx::space_cluster_t{},
        cuda_ptx::space_global_t{},
        cuda_ptx::cta_group_2_t{},
        &smemDst[stage][0],         // SMEM destination
        &params.tmaDescriptor,       // tensor map descriptor
        coords,                      // GMEM coordinates
        leadCtaMbar,                 // mbarrier for completion
        ctaMask);                    // CTA multicast mask
}

coords[0] += tileSize;  // advance for next tile
```

### Gather TMA (Paged/Indirect Access)

For paged KV cache in FMHA — gather 4 pages from different memory offsets in one TMA call:

```cpp
cuda_ptx::cp_async_bulk_tensor_tile_gather4(
    cuda_ptx::space_cluster_t{},
    cuda_ptx::space_global_t{},
    cuda_ptx::cta_group_2_t{},
    &smemDst[stage][0],
    &params.tmaKV,
    coords,            // base coordinates (per-page offsets set externally)
    leadCtaMbar,
    ctaMask);
```

### TMA OOB Fill for K-Residue

Set during host-side tensor map creation:
```cpp
CUtensorMapFloatOOBfill oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA;
```
TMA loads beyond tensor bounds produce NaN values that become zero under FMA — partial K-tiles contribute nothing to the accumulator automatically.

## TMEM Load/Store

```cpp
// TMEM → Registers (for epilogue / post-processing)
uint32_t regs[N];
trtllm::dev::tcgen05_ld_32x32b(regs, tmemAddr);
// Wider variants: tcgen05_ld_32x32b_x4, tcgen05_ld_32x32b_x8

// Registers → TMEM (e.g., writing quantized P back for V accumulation)
trtllm::dev::tcgen05_st_32x32b(tmemAddr, regs);
```

Each 32x32b operation transfers a 32-row x 32-byte tile. TMEM address format: `TmemAddr{col_id[0:16), row_id[16:32)}`.

## Warp Specialization

### Role Assignment

```cpp
int warp_idx = threadIdx.x / 32;

if (warp_idx < 4) {
    // Epilogue warp group: TMEM → RMEM → GMEM writeback
    epilogue_task();
} else if (warp_idx < 8) {
    // SMEM ↔ TMEM shuffle (format conversion if needed)
    shuffle_task();
} else if (warp_idx == 8) {
    // Tensor Core warp: issues all tcgen05_mma
    mma_task();
} else if (warp_idx == 9) {
    // TMA warp for weight/B matrix
    tma_load_B_task();
} else if (warp_idx == 10) {
    // TMA warp for activation/A matrix
    tma_load_A_task();
}
```

### Dynamic Register Allocation

```cpp
// Load task: reduce registers (mostly waiting for TMA)
asm volatile("setmaxnreg.dec.sync.aligned.u32 72;\n" ::: "memory");

// Compute task: increase registers for complex math
asm volatile("setmaxnreg.inc.sync.aligned.u32 152;\n" ::: "memory");
```

### Pipeline Wrappers (FMHA Style)

```cpp
using Pipeline = CutlassTmaUmmaAsyncPipeline</*Stages=*/2, ClusterShape>;
Pipeline pipeline;

// Producer (TMA warp)
pipeline.producer_try_acquire(stage);
pipeline.producer_acquire(stage);
// ... issue TMA loads ...
pipeline.producer_commit(stage);

// Consumer (MMA warp)
pipeline.consumer_try_wait(stage);
pipeline.consumer_wait(stage);
// ... issue MMA ...
pipeline.consumer_release(stage);
```

Inter-task sync via `mbarrier` pipelines: `producer_acquire → work → producer_commit`; `consumer_wait → work → consumer_release`.

## Dense Masking (FMHA Attention Pattern)

For causal masks and variable-length sequences, apply masks element-wise after loading from TMEM:

### Fast Path vs Slow Path

```cpp
bool allTilesAreCompleteK = (seqLenKv % 128) == 0;
bool isFullTileK = (tileOffsetK + 128) <= seqLenKv;

if (allTilesAreCompleteK || isFullTileK) {
    // Fast path: no masking, process all elements
    tcgen05_ld_32x32b(regs, tmemAddr);
    // ... compute max/softmax directly ...
} else {
    // Slow path: load then apply mask
    tcgen05_ld_32x32b(regs, tmemAddr);
    // Compute per-thread distance to boundary
    int uniformDist = seqLenKv - (tileOffsetK + threadColIdx);
    int clampedDist = clamp(uniformDist, 0, 64);
    // Apply mask via inline PTX
}
```

### Inline PTX Masking

```cpp
asm volatile(
    ".reg .pred p<64>;\n"
    ".reg .s32 offset<2>;\n"
    "mov.s32 offset0, %66;\n"          // clampedDist
    "sub.s32 offset1, %66, 32;\n"      // clampedDist - 32
    "max.s32 offset1, offset1, 0;\n"
    ".reg .u32 cond<2>;\n"
    "shl.b32 cond0, 0xffffffff, offset0;\n"  // mask for first 32 elements
    "shl.b32 cond1, 0xffffffff, offset1;\n"  // mask for next 32 elements
    // ... branch table to apply predicates to register array ...
    // Elements beyond clampedDist set to -inf (for softmax)
    : /* outputs */ : /* inputs */ : "memory");
```

Generates a bitmask from `clampedDist` and uses PTX predicates to conditionally set OOB elements to `-inf` for softmax stability.

## Dynamic SMEM (FMHA Style)

```cpp
extern __shared__ char shared_memory[];
char* smemQ    = shared_memory;                              // align 1024
char* smemK    = smemQ + alignUp(sizeQ, 128);               // align 128
char* smemV    = smemK + alignUp(sizeK, 128);
char* barriers = smemV + alignUp(sizeV, 16);                // align 16
```

Compute total via a host-side `GetSmemSize()` function:
```cpp
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
```

## Predication Summary

| Context | Mechanism | Who handles |
|---|---|---|
| Mainloop K-residue | TMA OOB zero-fill (`NAN_REQUEST_ZERO_FMA`) | Hardware |
| Epilogue M/N partial | `make_identity_tensor` + `elem_less` + residue | Software |
| Epilogue TMA store | TMA hardware bounds checking | Hardware |
| FMHA attention mask | Inline PTX predicates + bitmask | Software |
| CuTeDSL K boundary | `predicate_k` + `make_rmem_tensor` + `elem_less` | Software |

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Swizzle mismatch between SMEM desc and buffer | Silent wrong results | Ensure `layout_type_` in desc matches SMEM allocation swizzle |
| Wrong `incrSmemAddr` delta | Garbled MMA output | Verify delta = K_elements * elem_bytes / granularity |
| Missing `elect_one_sync()` before MMA | Multiple threads issue MMA | Always guard with `elect_one_sync()` |
| Wrong `cta_group` for 1SM vs 2SM | Hang or wrong results | `cta_group_1` for 1SM, `cta_group_2` for 2SM |
| Not resetting descriptor to stage base | Stale descriptor after K-loop | Reconstruct `createSmemDesc` for each new SMEM stage |
| TMA barrier bytes mismatch | Barrier never completes | `mbarrier_arrive_expect_tx` bytes must match actual TMA transfer |
| Missing `mapa` for 2-CTA barriers | Barrier targets wrong CTA | Use `cuda_ptx::mapa` to get lead CTA's barrier address |
| `setmaxnreg` without `sync.aligned` | Warp divergence | Always use `.sync.aligned` variant |
| Applying mask on fast path | Unnecessary perf loss | Check `isFullTileK` before entering mask path |
| Not `alignas(128)` SMEM buffers | TMA/MMA errors | Ensure 128B alignment on all SMEM allocations |
| Forgetting `release_allocation_lock` | TMEM deadlock in persistent kernels | Always release before `free()` |
| OOB K-residue without zero-fill | Accumulator poisoned with NaN | Set `CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA` on TMA descriptor |
