# Metal Kernel Batch Size Optimization Analysis

This document summarizes findings from analyzing Metal kernel performance for batch sizes 64+ on Apple Silicon (M2 Ultra).

## Benchmark Command

```bash
# Build the benchmark tool
cmake --build build --target llama-verify-bench

# Run verification cost benchmark
./build/bin/llama-verify-bench \
    -m <model.gguf> \
    -fa 1 \
    -d 1024 \
    -nv 4,8,16,32,64,128,256 \
    -r 10
```

Options:
- `-fa 1`: Enable flash attention
- `-d 1024`: Context depth before verification
- `-nv`: Batch sizes to test (verification tokens)
- `-r 10`: Number of repetitions per measurement

## Sample Results (Devstral 123B Q6_K, M2 Ultra)

| depth | n_verify | avg_ms   | t/s   |
|------:|--------:|---------:|------:|
| 1024  | 4       | 302.891  | 13.21 |
| 1024  | 8       | 538.138  | 14.87 |
| 1024  | 16      | 605.293  | 26.43 |
| 1024  | 32      | 604.523  | 52.93 |
| 1024  | 64      | 1033.468 | 61.93 |
| 1024  | 128     | 1886.991 | 67.83 |
| 1024  | 256     | 3610.276 | 70.91 |

Note: Throughput improvement slows significantly after batch 64.

## Key Code Locations

### Flash Attention Kernels

| File | Line | Description |
|------|------|-------------|
| `ggml/src/ggml-metal/ggml-metal-ops.cpp` | 2154-2161 | Kernel selection threshold (batch < 20 uses vec kernel) |
| `ggml/src/ggml-metal/ggml-metal-impl.h` | 83-87 | Tile size constants (NQPTG, NCPSG) |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp` | 2459 | Simdgroups hardcoded to 4 |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp` | 2513 | Threadgroup dispatch for regular kernel |
| `ggml/src/ggml-metal/ggml-metal.metal` | 5309-5549 | `kernel_flash_attn_ext_impl` implementation |
| `ggml/src/ggml-metal/ggml-metal.metal` | 6144-6590 | `kernel_flash_attn_ext_vec_impl` implementation |

### Matrix Multiplication Kernels

| File | Line | Description |
|------|------|-------------|
| `ggml/src/ggml-metal/ggml-metal-ops.cpp` | 1681-1900 | `ggml_metal_op_mul_mat` dispatch logic |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp` | 1706 | Break-even point `ne11_mm_min = 8` |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp` | 1850 | Threadgroup dispatch: `(ne11+31)/32, (ne01+63)/64` |
| `ggml/src/ggml-metal/ggml-metal.metal` | 8786-9090 | `kernel_mul_mm` implementation |
| `ggml/src/ggml-metal/ggml-metal.metal` | 8801-8802 | Tile sizes: `NR0=64, NR1=32` |

## Kernel Selection Logic

### Flash Attention (`ggml-metal-ops.cpp:2154-2161`)

```cpp
bool ggml_metal_op_flash_attn_ext_use_vec(const ggml_tensor * op) {
    const int64_t ne00 = op->src[0]->ne[0]; // head size
    const int64_t ne01 = op->src[0]->ne[1]; // batch size
    // use vec kernel if batch < 20 and head size divisible by 32
    return (ne01 < 20) && (ne00 % 32 == 0);
}
```

### Matrix Multiplication (`ggml-metal-ops.cpp:1705-1810`)

Three kernel paths:
1. **`mul_mv_ext`**: Batch 2-8, optimized small-batch mat-vec
2. **`mul_mm`**: Batch > 8, uses simdgroup matrix multiply (64x32 tiles)
3. **`mul_mv`**: Fallback for other cases

## Constants and Tile Sizes

### Flash Attention (`ggml-metal-impl.h:83-87`)

```cpp
#define OP_FLASH_ATTN_EXT_NQPTG 8      // queries per threadgroup (regular)
#define OP_FLASH_ATTN_EXT_NCPSG 64     // cache items per simdgroup (regular)

#define OP_FLASH_ATTN_EXT_VEC_NQPTG 1  // queries per threadgroup (vec)
#define OP_FLASH_ATTN_EXT_VEC_NCPSG 32 // cache items per simdgroup (vec)
```

### Matrix Multiplication (`ggml-metal.metal:8801-8802`)

```cpp
constexpr int NR0 = 64;  // rows from weight matrix per threadgroup
constexpr int NR1 = 32;  // batch dimension per threadgroup
constexpr int NK  = 32;  // K dimension tile
```

## Threadgroup Utilization Analysis

### Flash Attention Dispatch (`ggml-metal-ops.cpp:2513`)

```cpp
ggml_metal_encoder_dispatch_threadgroups(enc,
    (ne01 + nqptg - 1)/nqptg,  // batch dimension: ceil(batch/8)
    ne02,                      // heads
    ne03,                      // layers
    32, nsg, 1);               // threads: 32 per simdgroup, 4 simdgroups
```

| Batch | Threadgroups (dim 0) |
|-------|---------------------|
| 64    | 8                   |
| 128   | 16                  |
| 256   | 32                  |

### Matrix Multiplication Dispatch (`ggml-metal-ops.cpp:1850`)

```cpp
ggml_metal_encoder_dispatch_threadgroups(enc,
    ((ne11 + 31)/32),   // batch dimension: ceil(batch/32)
    ((ne01 + 63)/64),   // output rows: ceil(rows/64)
    ne12*ne13,          // batches
    128, 1, 1);         // 128 threads per threadgroup
```

| Batch | Threadgroups (batch dim) |
|-------|-------------------------|
| 64    | 2                       |
| 128   | 4                       |
| 256   | 8                       |

## Potential Optimization Opportunities

### 1. Flash Attention Simdgroups Fixed at 4

**Location**: `ggml-metal-ops.cpp:2459`

```cpp
int32_t nsg = 4;  // Currently hardcoded
```

There is commented-out code (lines 2443-2458) suggesting dynamic adjustment was considered. For larger batch sizes, increasing `nsg` could improve parallelism.

### 2. Matrix Multiplication Tile Size

**Location**: `ggml-metal.metal:8801-8802`

The fixed 32-wide batch tile means batch 64 only launches 2 threadgroups in that dimension. For large batches on M2 Ultra (24 GPU cores), this may leave cores idle.

Possible investigation:
- Larger batch tiles (e.g., NR1=64) for better GPU utilization
- Dynamic tile selection based on batch size

### 3. Flash Attention Query Block Size

**Location**: `ggml-metal-impl.h:83`

`NQPTG=8` means 8 queries processed per threadgroup. For very large batch sizes, larger blocks might amortize memory access overhead better.

### 4. Vec Kernel Threshold

**Location**: `ggml-metal-ops.cpp:2161`

The threshold of 20 for switching between vec and regular kernels may not be optimal for all hardware configurations.

## Profiling Recommendations

1. Use Xcode Instruments Metal System Trace to identify:
   - GPU occupancy per kernel
   - Memory bandwidth utilization
   - Threadgroup execution time distribution

2. Compare kernel execution times:
   ```bash
   # Enable Metal debug output
   export GGML_METAL_DEBUG=1
   ./build/bin/llama-verify-bench -m model.gguf -fa 1 -d 1024 -nv 64 -r 1 -v
   ```

3. Test with different context depths to isolate attention vs FFN bottlenecks:
   ```bash
   ./build/bin/llama-verify-bench -m model.gguf -fa 1 -d 256,512,1024,2048 -nv 64 -r 5
   ```

## Experimental Results

### NQPTG (Queries Per Threadgroup) Tuning

Tested on M2 Ultra with Devstral 123B Q6_K, context depth 1024:

| Config | n=64 (t/s) | n=128 (t/s) | n=256 (t/s) |
|--------|------------|-------------|-------------|
| Original (NQPTG=8) | 61.93 | 67.83 | 70.91 |
| **NQPTG=16** | 62.30 | **68.00** | **71.38** |
| NQPTG=32 | 61.69 | 67.35 | 70.69 |

**Finding**: NQPTG=16 is optimal. Going to 32 causes regression, likely due to:
- Increased shared memory pressure per threadgroup
- Reduced parallelism (fewer threadgroups)

### NCPSG (Cache Items Per Simdgroup) Tuning

With NQPTG=16:

| Config | n=64 (t/s) | n=128 (t/s) | n=256 (t/s) |
|--------|------------|-------------|-------------|
| NCPSG=64 (original) | 62.30 | 68.00 | 71.38 |
| NCPSG=128 | 61.92 | 67.80 | 70.99 |

**Finding**: NCPSG=64 remains optimal. Doubling to 128 causes slight regression.

### Recommended Change

```cpp
// ggml-metal-impl.h line 83
#define OP_FLASH_ATTN_EXT_NQPTG 16  // was 8
```

Expected improvement: ~0.5% for batch sizes 128+.

## Related Files

- `tools/verify-bench/verify-bench.cpp` - Benchmark implementation
- `ggml/src/ggml-metal/ggml-metal-device.cpp` - Pipeline selection
- `ggml/src/ggml-metal/ggml-metal-device.h` - Device properties
- `ggml/src/ggml-metal/ggml-metal-impl.h` - Constants and structs
