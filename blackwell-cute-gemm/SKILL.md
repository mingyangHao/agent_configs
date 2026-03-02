---
name: blackwell-cute-gemm
description: Complete guide to building Blackwell (SM100a) GEMM kernels using CuTe abstractions, TMA, tcgen05.mma, and TMEM. Covers host setup, SMEM swizzle derivation, TMA load/store, MMA mainloop, epilogue predication, 2SM clusters, multi-stage pipelines, and all supported data types. Use when writing CuTe-based GEMM kernels for Blackwell, deriving SMEM layouts from MMA atoms, choosing swizzle patterns, handling boundary tiles, or integrating TMA with tcgen05.mma.
---

# Blackwell CuTe GEMM: TMA + tcgen05.mma on SM100a

## Architecture Foundations

### Memory Hierarchy

```
GMEM ──TMA──▶ SMEM ──descriptor──▶ tcgen05.mma ──▶ TMEM (accumulator)
                                                       │
                                                  tcgen05.ld
                                                       │
                                                       ▼
                                                     RMEM ──store──▶ GMEM/SMEM
```

**TMEM**: Dedicated on-SM scratchpad (128 rows x 512 columns per SM). Accumulators live exclusively here. Only accessible via `tcgen05.ld`/`tcgen05.st` (32x32-byte tiles) between TMEM and registers.

**TMA**: Hardware DMA engine that copies tensor tiles between GMEM and SMEM using tensor map descriptors (`CUtensorMap`). A single thread issues the instruction; hardware handles address calculation, swizzling, and multicast.

**tcgen05.mma**: 5th-gen tensor core instruction. Reads A from SMEM (or TMEM for TS variants), B from SMEM, accumulates into TMEM. Requires SMEM descriptors for operand addressing and an instruction descriptor encoding shapes/types.

### MMA Instruction Dimensions

| MMA instruction | M | N | K (per issue) | Notes |
|---|---|---|---|---|
| F16/BF16 1SM | 128 | 256 | 16 | `SM100_MMA_F16BF16_SS` |
| F16/BF16 2SM | 256 | 256 | 16 | `SM100_MMA_F16BF16_2x1SM_SS` |
| FP8 (f8f6f4) | 128/256 | up to 256 | 32 | `kind_f8f6f4` |
| TF32 | 128 | 256 | 8 | `kind_tf32` |
| INT8 | 128 | 256 | 32 | `kind_i8` |

Typical tile: accumulate `bK = MmaK * num_k_blocks` per mainloop iteration (e.g., K=64 for F16 with 4 K-blocks of 16).

## SMEM Swizzle Patterns

### Why Swizzle

SMEM is organized in 32 banks of 4 bytes each. Without swizzling, sequential accesses by different threads to the same MMA operand column would hit the same bank, causing conflicts. Swizzle XORs high address bits into low bits so that adjacent threads access different banks.

### Swizzle<B,M,S> Parameters

CuTe encodes swizzle as `Swizzle<B, M, S>` composing a `ComposedLayout`:
- **B** (num_bits): number of swizzle bits — controls the XOR span
- **M** (num_base): log2 of the row count in the atom
- **S** (num_shft): starting bit position of the XOR source

The swizzle function XORs bits from a "source" range into a "destination" range of the linear SMEM address:

```
Linear address:  ... | bits [M+S-1 : S] | ... | bits [M+B-1 : M] | ...
                      ─────────┬──────           ─────────┬──────
                          XOR source                XOR destination

Swizzled address = linear_addr ^ ((linear_addr >> S) & mask(B)) << M
  where mask(B) = (1 << B) - 1
```

For `Swizzle<3,4,3>` (SW128B):
- Source bits: `[6:3]` — address within 128B cache line
- Destination bits: `[6:4]` — which 16B bank within 128B
- 3-bit XOR eliminates bank conflicts for strided access patterns up to stride 128B

### Swizzle Types and Atom Definitions

| Swizzle name | `Swizzle<B,M,S>` | LayoutType in descriptor |
|---|---|---|
| INTER (none) | `Swizzle<0,4,3>` | `SWIZZLE_NONE` (0) |
| SW32B | `Swizzle<1,4,3>` | `SWIZZLE_32B` (6) |
| SW64B | `Swizzle<2,4,3>` | `SWIZZLE_64B` (4) |
| SW128B | `Swizzle<3,4,3>` | `SWIZZLE_128B` (2) |
| SW128B_BASE32B | `Swizzle<2,5,2>` | `SWIZZLE_128B_BASE32B` |

Atom definitions in bits (from `mma_traits_sm100.hpp`):

```cpp
// K-major atoms (K as innermost dimension)
Layout_K_INTER_Atom_Bits = Swizzle<0,4,3> o smem_ptr o Shape<_8, _128>:Stride<_128,_1>
Layout_K_SW32_Atom_Bits  = Swizzle<1,4,3> o smem_ptr o Shape<_8, _256>:Stride<_256,_1>
Layout_K_SW64_Atom_Bits  = Swizzle<2,4,3> o smem_ptr o Shape<_8, _512>:Stride<_512,_1>
Layout_K_SW128_Atom_Bits = Swizzle<3,4,3> o smem_ptr o Shape<_8,_1024>:Stride<_1024,_1>

// MN-major atoms (M or N as innermost dimension)
Layout_MN_INTER_Atom_Bits = Swizzle<0,4,3> o smem_ptr o Shape< _128,_8>:Stride<_1, _128>
Layout_MN_SW32_Atom_Bits  = Swizzle<1,4,3> o smem_ptr o Shape< _256,_8>:Stride<_1, _256>
Layout_MN_SW64_Atom_Bits  = Swizzle<2,4,3> o smem_ptr o Shape< _512,_8>:Stride<_1, _512>
Layout_MN_SW128_Atom_Bits = Swizzle<3,4,3> o smem_ptr o Shape<_1024,_8>:Stride<_1,_1024>
```

`upcast<sizeof_bits<Type>>` converts bit atoms to element units:
```cpp
Layout_K_SW128_Atom<half_t>  → Shape<_8,_64>:Stride<_64,_1>  with Sw<3,4,3>
Layout_K_SW128_Atom<float>   → Shape<_8,_32>:Stride<_32,_1>  with Sw<3,4,3>
Layout_K_SW128_Atom<fp8>     → Shape<_8,_128>:Stride<_128,_1> with Sw<3,4,3>
```

### Choosing the Right Swizzle

**Rule of thumb**: Use `Layout_K_SW128_Atom<Type>` for K-major operands and `Layout_MN_SW128_Atom<Type>` for MN-major. Drop to SW64/SW32 only when the tile's contiguous dimension is too small for 128B.

### Deriving SMEM Layout from MMA

Never hardcode SMEM layouts. Always derive from the MMA instruction:

```cpp
// Step 1: Get post-partitioned shape from TiledMMA
auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(bM, bK));
// e.g. ((_128,_16), _1, _4) for 128x64 tile with MmaK=16

// Step 2: Apply swizzle atom and tile to that shape
auto sA_layout = UMMA::tile_to_mma_shape(
    UMMA::Layout_K_SW128_Atom<TypeA>{}, mma_shape_A);
// Result: Sw<3,4,3> o smem_ptr[16b] o ((_128,_16),_1,_4):((_64,_1),_0,_16)
```

`tile_to_mma_shape` internally calls `tile_to_shape` then `tiled_divide` to produce `((MmaM,MmaK), NumMma_M, NumMma_K)` with correct swizzle.

### Canonical SMEM Descriptor Forms

When CuTe constructs the `SmemDescriptor` via `make_umma_desc`, it validates the tensor matches canonical forms (in uint128_t units):

**Major::K canonical layout** (most common for GEMM):
```
SWIZZLE_NONE: Sw<0,4,3> o ptr o ((8,n), 2):((1,SBO), LBO)
SWIZZLE_32B:  Sw<1,4,3> o ptr o ((8,n), 2):((2,SBO), 1)
SWIZZLE_64B:  Sw<2,4,3> o ptr o ((8,n), 2):((4,SBO), 1)
SWIZZLE_128B: Sw<3,4,3> o ptr o ((8,n), 2):((8,SBO), 1)
```

**Major::MN canonical layout**:
```
SWIZZLE_NONE:         Sw<0,4,3> o ptr o ((1,n),(8,k)):((X,SBO),(1,LBO))
SWIZZLE_32B:          Sw<1,4,3> o ptr o ((2,n),(8,k)):((1,LBO),(2,SBO))
SWIZZLE_64B:          Sw<2,4,3> o ptr o ((4,n),(8,k)):((1,LBO),(4,SBO))
SWIZZLE_128B:         Sw<3,4,3> o ptr o ((8,n),(8,k)):((1,LBO),(8,SBO))
SWIZZLE_128B_BASE32B: Sw<2,5,2> o ptr o ((8,n),(4,k)):((1,LBO),(4,SBO))
```

### TMA-SMEM-MMA Swizzle Consistency

```
                    ┌──────────────┐
                    │ GMEM tensor  │
                    └──────┬───────┘
                           │ make_tma_atom(... sA_layout ...)
                           │  TMA descriptor encodes swizzle from sA_layout
                    ┌──────▼───────┐
                    │ SMEM buffer  │ ← allocated with cosize_v<sA_layout> bytes
                    │ (swizzled)   │   128B-aligned
                    └──────┬───────┘
                           │ make_fragment_A(tCsA)  or  make_umma_desc(tensor)
                           │  SMEM descriptor extracts layout_type from swizzle
                    ┌──────▼───────┐
                    │ tcgen05.mma  │ ← descriptor.layout_type_ must match SMEM layout
                    └──────────────┘
```

**Critical invariant**: The swizzle in the TMA descriptor, the SMEM buffer layout, and the MMA SMEM descriptor must all agree. Using `tile_to_mma_shape` + `make_tma_atom` + `make_fragment_A` guarantees this automatically.

## The Complete GEMM Pipeline

### Stage 0: Host Setup

```cpp
// 1. Create TiledMMA from instruction atom
TiledMMA tiled_mma = make_tiled_mma(
    SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeAcc,
                         128, 256,
                         UMMA::Major::K, UMMA::Major::K>{});

// 2. Define MMA tile shape
auto bM = tile_size<0>(tiled_mma);          // 128
auto bN = tile_size<1>(tiled_mma);          // 256
auto bK = tile_size<2>(tiled_mma) * _4{};   // 16 * 4 = 64
auto mma_tiler = make_shape(bM, bN, bK);

// 3. Derive SMEM layouts (swizzled) — see Swizzle section above
auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(bM, bK));
auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(bN, bK));
auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeA>{}, mma_shape_A);
auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeB>{}, mma_shape_B);

// 4. Create TMA atoms (host-side tensor map descriptors)
auto tma_atom_A = make_tma_atom<SM100_TMA_LOAD>(
    tma_atom_A_params, mA, sA_layout, mma_tiler, cluster_shape);
auto tma_atom_B = make_tma_atom<SM100_TMA_LOAD>(
    tma_atom_B_params, mB, sB_layout, mma_tiler, cluster_shape);
```

### Stage 1: Prologue (Device)

```cpp
// Partition GMEM tensors for this CTA's tile
Tensor gA = local_tile(mA, mma_tiler, mma_coord, Step<_1,X,_1>{});
Tensor gB = local_tile(mB, mma_tiler, mma_coord, Step<X,_1,_1>{});

// Create SMEM tensors & MMA fragments
Tensor tCsA = shared_storage.tensor_sA();
Tensor tCsB = shared_storage.tensor_sB();
ThrMMA cta_mma = tiled_mma.get_slice(mma_v);
Tensor tCrA = cta_mma.make_fragment_A(tCsA);  // SMEM descriptor iterators
Tensor tCrB = cta_mma.make_fragment_B(tCsB);  // SMEM descriptor iterators
Tensor tCtAcc = cta_mma.make_fragment_C(tCgC); // TMEM accumulator

// Allocate TMEM (warp 0 only)
TMEM::Allocator1Sm tmem_alloc{};  // or Allocator2Sm for 2SM
if (elect_one_warp) {
    tmem_alloc.allocate(Allocator::Sm100TmemCapacityColumns, &smem.tmem_ptr);
}
__syncthreads();
tCtAcc.data() = smem.tmem_ptr;

// Setup TMA partitions
auto [tAgA, tAsA] = tma_partition(tma_atom_A, ...);
auto [tBgB, tBsB] = tma_partition(tma_atom_B, ...);

// Init barriers
initialize_barrier(smem.tma_barrier, /*threads=*/1);
initialize_barrier(smem.mma_barrier, /*ctas=*/1);
```

### Stage 2: Mainloop (K-loop)

```cpp
tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;  // first iteration clears acc

for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
    // --- TMA Load: GMEM → SMEM ---
    if (elect_one_warp && elect_one_thr) {
        set_barrier_transaction_bytes(smem.tma_barrier, tma_bytes);
        copy(tma_atom_A.with(smem.tma_barrier), tAgA(_, k_tile), tAsA);
        copy(tma_atom_B.with(smem.tma_barrier), tBgB(_, k_tile), tBsB);
    }

    // Wait TMA completion
    wait_barrier(smem.tma_barrier, tma_phase);
    tma_phase ^= 1;

    // --- MMA: SMEM → TMEM ---
    if (elect_one_warp) {
        for (int k_blk = 0; k_blk < num_k_blocks; ++k_blk) {
            gemm(tiled_mma, tCrA(_,_,k_blk), tCrB(_,_,k_blk), tCtAcc);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;  // accumulate after first
        }
        umma_arrive(&smem.mma_barrier);
    }

    // Wait MMA before overwriting SMEM
    wait_barrier(smem.mma_barrier, mma_phase);
    mma_phase ^= 1;
}
```

### Stage 3: Epilogue

**Simple (register-based store)**:
```cpp
TiledCopy t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
ThrCopy thr = t2r.get_slice(threadIdx.x);
copy(t2r, thr.partition_S(tCtAcc), tDrAcc);   // TMEM → RMEM
axpby(alpha, tDrAcc, beta, tDrC);             // D = alpha*acc + beta*C
copy(tDrC, tDgD);                              // RMEM → GMEM
```

**TMA epilogue (through SMEM)** — better for large tiles:
```cpp
for (int epi = 0; epi < num_epi_tiles; ++epi) {
    copy(tma_atom_C.with(smem.tma_barrier, 0), tGS_gC(_,epi), tGS_sC);
    wait_barrier(smem.tma_barrier, phase);

    copy_aligned(tTR_sC, tTR_rC);                   // C: SMEM → RMEM
    copy(t2r_copy, tTR_tAcc(_,_,epi), tTR_rD);      // Acc: TMEM → RMEM
    axpby(beta, tTR_rC, alpha, tTR_rD);

    copy_aligned(tTR_rD, tTR_sD);                   // D: RMEM → SMEM
    tma_store_fence();
    __syncthreads();
    copy(tma_atom_D, tSG_sD, tSG_gD(_,epi));        // TMA store
    tma_store_arrive();
    tma_store_wait<0>();
}
```

### Stage 4: TMEM Cleanup

```cpp
if (elect_one_warp) {
    tmem_alloc.release_allocation_lock();
    tmem_alloc.free(smem.tmem_ptr, Allocator::Sm100TmemCapacityColumns);
}
```

## Predication and Boundary Handling

### Strategy 1: TMA Built-in OOB Fill (Mainloop K-Residue)

TMA tensor map descriptors support automatic OOB handling:

```cpp
CUtensorMapFloatOOBfill tma_oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA;
```

With `NAN_REQUEST_ZERO_FMA`, TMA loads beyond tensor bounds produce a special NaN in SMEM that becomes zero under FMA. Partial K-tiles contribute nothing to the accumulator. **No kernel-side predication needed for the mainloop.**

### Strategy 2: Coordinate Tensor Predication (Epilogue M/N)

For partial output tiles, CUTLASS constructs coordinate tensors and computes residues:

```cpp
// Create identity tensor matching full problem dimensions
Tensor mD_crd = make_identity_tensor(make_shape(M, N));
// Slice to this CTA's tile
Tensor cD_mn = local_tile(mD_crd, take<0,2>(cta_tile_mnk),
                          make_coord(m_coord, n_coord));

// Partition with same layout as T2R copy
Tensor tTR_cD_mn = thread_t2r.partition_D(
    flat_divide(cD_mn, EpilogueTile{}));

// Compute residue = distance from CTA origin to problem boundary
auto residue_cD = make_coord(M, N) - cD_mn(_0{});

// Per-element bounds check: coordinate < residue means in-bounds
bool in_bounds = elem_less(tTR_cD(epi_v), residue_cD);
if (in_bounds) { /* store */ }
```

### Strategy 3: TMA Store Predication

TMA store handles OOB tiles transparently when enabled — hardware skips writes for coordinates beyond tensor bounds. Preferred approach for Blackwell GEMM epilogues.

### Decision Tree

```
┌─ K-residue (mainloop) ──▶ TMA OOB zero-fill (hardware, preferred)
│
├─ M/N-residue (epilogue) ─┬─▶ TMA store enabled? → hardware handles it
│                           └─▶ No TMA store → identity_tensor + elem_less predication
│
└─ Attention mask (FMHA) ──▶ Explicit per-element inline PTX masking
```

## 2SM (Two-CTA Cluster) Mode

| Aspect | 1SM | 2SM |
|---|---|---|
| Cluster shape | `(1,1,1)` | `(2,1,1)` or `(2,N,1)` |
| MMA atom | `SM100_MMA_F16BF16_SS` | `SM100_MMA_F16BF16_2x1SM_SS` |
| TMEM allocator | `Allocator1Sm` | `Allocator2Sm` |
| MMA arrival | `umma_arrive(&bar)` | `umma_arrive_multicast_2x1SM(&bar, mcast_mask)` |
| MMA M dim | 128 | 256 (doubled) |
| Only leader CTA issues MMA | N/A | `if (elect_one_cta)` |

With multicast TMA, one TMA load populates SMEM on both peer CTAs simultaneously, halving bandwidth:

```cpp
auto [tAgA, tAsA] = tma_partition(tma_atom_A,
    get<2>(cta_coord),
    make_layout(size<2>(cluster_layout_vmnk)),
    group_modes<0,3>(tCsA), group_modes<0,3>(tCgA));

uint16_t mcast_mask = create_tma_multicast_mask<2>(cluster_layout, cta_coord);
copy(tma_atom_A.with(smem.tma_barrier, mcast_mask), tAgA(_, k), tAsA);
```

## Multi-Stage Pipeline

```cpp
constexpr int STAGES = 2;
// Prologue: fill first stage
issue_tma_load(stage=0);

for (int k = 0; k < num_tiles; ++k) {
    int curr = k % STAGES;
    int next = (k + 1) % STAGES;

    wait_barrier(tma_bar[curr], phase[curr]);
    phase[curr] ^= 1;

    if (k + 1 < num_tiles) {
        issue_tma_load(stage=next);
    }

    issue_mma(curr);
    wait_mma(curr);
}
```

Phase bit flips every iteration (`phase ^= 1`). `wait_barrier(bar, phase)` blocks until barrier completes that phase.

### Barrier Protocol

```
      TMA warp                    MMA warp                 All threads
    ┌──────────┐              ┌──────────────┐          ┌──────────────┐
    │set_barrier│              │              │          │              │
    │_tx_bytes  │              │              │          │              │
    │copy(tma_A)│              │              │          │              │
    │copy(tma_B)│              │              │          │              │
    └──────┬───┘              │  wait_barrier│          │              │
           │  tma_barrier ──▶ │  (tma)       │          │              │
           │                  │  gemm(...)   │          │              │
           │                  │  umma_arrive │          │              │
           │                  └──────┬───────┘          │  wait_barrier│
           │                  mma_barrier ──────────▶   │  (mma)       │
           │                                            └──────────────┘
```

## SharedStorage Layout

```cpp
struct SharedStorage {
    alignas(128) union {
        struct {
            alignas(128) ArrayEngine<TypeA, ...> A;
            alignas(128) ArrayEngine<TypeB, ...> B;
        } mainloop;
        ArrayEngine<TypeC, ...> C;
        ArrayEngine<TypeD, ...> D;
    } tensors;

    alignas(16) uint64_t mma_barrier;
    alignas(16) uint64_t tma_barrier;
    alignas(16) uint32_t tmem_base_ptr;
};
```

The union lets mainloop A/B overlap with epilogue C/D since they're never live simultaneously.

## Performance Optimization Checklist

- **Prefetch tensor maps** at kernel start: `prefetch_tensormap(&params.tmaA)`
- **Grid dependency sync**: `cudaGridDependencySynchronize()` before first TMA if prior kernel wrote the data
- **Multi-stage pipeline**: 2+ SMEM stages for TMA/MMA overlap
- **TMA multicast**: share loads across cluster to halve GMEM bandwidth
- **SMEM union**: overlap mainloop A/B buffers with epilogue C/D buffers
- **First MMA clears accumulator**: `ScaleOut::Zero`, then switch to `ScaleOut::One`
- **Unroll K-blocks** within a tile for instruction-level parallelism
- **128B alignment** on all SMEM buffers for TMA and MMA
- **Persistent kernels** with tile schedulers for large problems

## Data Type Reference

| Input A/B | Accumulator | MMA Kind | CuTe Atom |
|---|---|---|---|
| FP16 | FP32 | `f16` | `SM100_MMA_F16BF16_SS` |
| BF16 | FP32 | `f16` | `SM100_MMA_F16BF16_SS` |
| TF32 | FP32 | `tf32` | `SM100_MMA_TF32_SS` |
| FP8 E4M3 | FP32 | `f8f6f4` | `kind_f8f6f4` PTX |
| FP8 E5M2 | FP32 | `f8f6f4` | `kind_f8f6f4` PTX |
| FP4 E2M1 | FP32 | `mxf4nvf4` | Block-scaled variants |
| INT8 | INT32 | `i8` | `SM100_MMA_I8_SS` |

Block-scaled formats (MXFP): per-block UE8M0 scale factors stored in TMEM, consumed by `block_scale.scale_vec::4X` MMA variants.

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Wrong swizzle pattern | Garbled output | Use `UMMA::tile_to_mma_shape()` to auto-derive |
| Swizzle mismatch TMA vs MMA | Silent wrong results | Derive both from same `sA_layout` |
| Missing `tma_store_fence()` | TMA reads stale SMEM | Always fence before TMA store |
| Not setting `ScaleOut::Zero` first | Accumulator has garbage | Set on first MMA, then `ScaleOut::One` |
| Forgetting `release_allocation_lock` | Deadlock on TMEM | Always release before `free()` |
| Barrier phase mismatch | Hang or race condition | Always `^= 1` after each wait |
| Missing `__syncthreads` after alloc | Stale tmem_ptr | Sync after TMEM allocation |
| TMA bytes mismatch | Barrier never completes | `set_barrier_transaction_bytes` must match actual bytes sent |
| Non-128B-aligned SMEM | TMA/MMA errors | Use `alignas(128)` on all buffers |
| No TMA store + OOB tiles | Wrong epilogue output | Either enable TMA store or ensure M,N divide tile shape |
| Missing OOB fill for K-residue | Accumulator poisoned | Set `CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA` |
| Predicate not matching T2R layout | Elements at wrong positions | Partition identity tensor with same `thread_t2r` as data |
