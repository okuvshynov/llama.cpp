# llama-verify-bench

Benchmark tool for measuring verification cost as a function of batch size. This simulates the target model verification phase in speculative decoding, where the target model evaluates N draft tokens at once with existing KV cache context.

## Build

```bash
# Build only this tool
cmake -B build
cmake --build build --target llama-verify-bench

# Binary location
./build/bin/llama-verify-bench
```

## Usage

```bash
llama-verify-bench -m <model.gguf> [options]
```

### Basic Examples

```bash
# Test default batch sizes (1,2,4,8,16,32,64) at context depth 512
llama-verify-bench -m model.gguf

# Test specific batch sizes with flash attention
llama-verify-bench -m model.gguf -fa 1 -nv 1,2,4,8,16,32,64,128

# Test all batch sizes from 1 to 64 at depth 1024
llama-verify-bench -m model.gguf -d 1024 -nv 1-64+1 -r 10

# Multiple context depths
llama-verify-bench -m model.gguf -d 256,512,1024,2048 -nv 64

# Output as JSON
llama-verify-bench -m model.gguf -nv 1-32*2 -o json > results.json

# Show progress during benchmarking
llama-verify-bench -m model.gguf -nv 1-64+1 --progress
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-m, --model` | (required) | Path to GGUF model file |
| `-d, --n-depth` | 512 | Context depth before verification (tokens in KV cache) |
| `-nv, --n-verify` | 1,2,4,8,16,32,64 | Batch sizes to test |
| `-b, --batch-size` | 2048 | Batch size for context filling |
| `-ub, --ubatch-size` | 512 | Micro-batch size |
| `-t, --threads` | auto | Number of CPU threads |
| `-ngl, --n-gpu-layers` | 99 | Number of layers to offload to GPU |
| `-fa, --flash-attn` | 0 | Enable flash attention (0 or 1) |
| `-r, --repetitions` | 10 | Number of repetitions per measurement |
| `-o, --output` | md | Output format: `md`, `csv`, `json`, `jsonl` |
| `-v, --verbose` | off | Show model loading and debug output |
| `--progress` | off | Show progress during benchmarking |

## Range Syntax

Both `-d` and `-nv` support flexible range specifications:

| Syntax | Example | Expands to |
|--------|---------|------------|
| Single value | `64` | 64 |
| Comma-separated | `1,2,4,8` | 1, 2, 4, 8 |
| Range with step | `1-64+1` | 1, 2, 3, ..., 64 |
| Range with multiplier | `1-64*2` | 1, 2, 4, 8, 16, 32, 64 |
| Mixed | `1-8+1,16,32,64` | 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64 |

## Output Formats

### Markdown (default)

```
# Verification Cost Benchmark

Model: model.gguf
Description: llama 7B Q4_0
GPU layers: 99, Flash attention: yes, Threads: 8

|  depth | n_verify |     avg_ms |   stdev_ms |        t/s |
|------:|--------:|----------:|----------:|----------:|
|   512 |        1 |     15.234 |      0.312 |      65.64 |
|   512 |        2 |     15.891 |      0.287 |     125.86 |
```

### CSV

```csv
n_depth,n_verify,avg_ms,stdev_ms,avg_ts,samples_ns
512,1,15.234,0.312,65.64,"15234123;15198456;..."
512,2,15.891,0.287,125.86,"15891234;15876543;..."
```

### JSON

```json
{
  "model": "model.gguf",
  "model_desc": "llama 7B Q4_0",
  "n_gpu_layers": 99,
  "flash_attn": true,
  "n_threads": 8,
  "results": [
    {"n_depth": 512, "n_verify": 1, "avg_ms": 15.234, "stdev_ms": 0.312, "avg_ts": 65.64, "samples_ns": [15234123, 15198456]}
  ]
}
```

### JSONL

One JSON object per line, useful for streaming/appending:

```jsonl
{"n_depth": 512, "n_verify": 1, "avg_ms": 15.234, "stdev_ms": 0.312, "avg_ts": 65.64, "samples_ns": [15234123, 15198456]}
{"n_depth": 512, "n_verify": 2, "avg_ms": 15.891, "stdev_ms": 0.287, "avg_ts": 125.86, "samples_ns": [15891234, 15876543]}
```

## How It Works

1. **Context filling**: The tool first fills the KV cache with random tokens up to the specified depth
2. **Verification simulation**: For each batch size, it decodes N tokens at once (with logits requested for all positions, simulating speculation verification)
3. **Measurement**: After a warmup run, it measures decode time across multiple repetitions
4. **Context restoration**: After each measurement, tokens are removed from KV cache to restore the original context depth

This accurately simulates what happens during speculative decoding when the target model verifies a batch of draft tokens proposed by the draft model.

## Use Cases

- **Speculative decoding analysis**: Understand how verification cost scales with draft length to optimize speculation strategy
- **Hardware comparison**: Compare verification throughput across different GPUs or configurations
- **Kernel optimization**: Measure impact of Metal/CUDA kernel changes on batch processing efficiency
- **Flash attention evaluation**: Compare performance with and without flash attention at various batch sizes
