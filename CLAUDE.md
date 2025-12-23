# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Standard build (CPU-only, uses Metal on macOS)
cmake -B build
cmake --build build --config Release -j $(nproc)

# Build specific target
cmake --build build --target llama-cli

# CUDA build
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)

# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

Built binaries go to `build/bin/`. The root Makefile is deprecated; always use CMake.

## Testing

```bash
# Run all tests
ctest --test-dir build --output-on-failure -j $(nproc)

# Server tests (requires .venv)
cd tools/server/tests
source ../../../.venv/bin/activate
./tests.sh
```

## Code Formatting

```bash
# Format C++ before committing
git clang-format

# Python (activate venv first)
source .venv/bin/activate
```

Key formatting rules: 4-space indentation, 120 column limit, pointer/reference style `void * ptr`, `int & a`.

## Architecture Overview

### Core Libraries
- **`ggml/`**: Tensor computation library with backend-specific implementations in `ggml/src/ggml-{cpu,cuda,metal,vulkan,...}/`
- **`src/`**: Main llama library implementation (`llama-*.cpp` files)
- **`include/llama.h`**: Public C API header
- **`common/`**: Shared utilities used across tools and examples

### Key Source Files in `src/`
- `llama.cpp`: Core library entry point
- `llama-model.cpp`: Model loading and architecture-specific code (~450KB)
- `llama-context.cpp`: Inference context management
- `llama-sampling.cpp`: Token sampling implementations
- `llama-vocab.cpp`: Tokenizer implementations
- `llama-kv-cache.cpp`: Key-value cache for transformer attention
- `llama-grammar.cpp`: GBNF grammar constraint parsing
- `llama-arch.cpp`: Model architecture definitions

### Primary Tools (in `tools/`)
- `cli/`: Main inference tool (`llama-cli`)
- `server/`: OpenAI-compatible HTTP server (`llama-server`)
- `quantize/`: Model quantization utility
- `perplexity/`: Model evaluation
- `llama-bench/`: Performance benchmarking

### GGML Backends
Located in `ggml/src/`: `ggml-cpu`, `ggml-cuda`, `ggml-metal`, `ggml-vulkan`, `ggml-sycl`, `ggml-hip`, `ggml-opencl`, `ggml-rpc`

## Coding Guidelines

- Use `snake_case` for all identifiers
- Naming pattern: `<class>_<method>` (e.g., `llama_model_init`, `llama_sampler_get_seed`)
- Enum values: uppercase with prefix (e.g., `LLAMA_VOCAB_TYPE_BPE`)
- Avoid adding external dependencies
- Use sized integers (`int32_t`) in public API
- Struct declarations: `struct foo {}` not `typedef struct foo {} foo`
- Matrix multiplication is unconventional: `C = ggml_mul_mat(ctx, A, B)` means C^T = A * B^T

## CI/Validation

```bash
# Local CI validation
mkdir tmp
bash ./ci/run.sh ./tmp/results ./tmp/mnt

# Add "ggml-ci" to commit message to trigger heavy CI
```

When modifying ggml operators, run `test-backend-ops` to verify backend consistency.
