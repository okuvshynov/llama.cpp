# MoE Expert Selection Logging Guide

## Overview

This guide provides instructions for instrumenting llama.cpp to log expert selection patterns in Mixture-of-Experts (MoE) models. This is useful for studying how different experts are activated for various types of inputs.

## Understanding Your MoE Architecture

Based on typical MoE model tensor layouts:

- **`blk.*.ffn_gate_inp.weight`** - Router network: Maps hidden states to expert logits
- **`blk.*.exp_probs_b.bias`** - Expert selection bias (DeepSeek V3 style)
- **Expert tensors** - Stored as 3D tensors: `[output_dim, hidden_dim, n_experts]`

### Key Tensors in the Routing Process

The MoE routing happens in `src/llama-graph.cpp` in the `build_moe_ffn()` function (around line 877):

```cpp
// Router logits (raw scores) - around line 905
logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
cb(logits, "ffn_moe_logits", il);

// Router probabilities (after softmax/sigmoid) - around line 932
cb(probs, "ffn_moe_probs", il);

// Selection probabilities with bias - around line 939
cb(selection_probs, "ffn_moe_probs_biased", il);

// Selected expert indices (top-k) - around line 979-981
selected_experts = ggml_top_k(ctx0, selection_probs, n_expert_used);
cb(selected_experts->src[0], "ffn_moe_argsort", il);  // argsort indices
cb(selected_experts, "ffn_moe_topk", il);             // top-k indices

// Expert weights (final routing weights) - around line 993
weights = ggml_get_rows(ctx0, probs, selected_experts);
cb(weights, "ffn_moe_weights", il);
```

**Note:** Additional callbacks have been added for debugging:
- `ffn_moe_logits_biased` - logits with input bias applied
- `ffn_moe_group_topk` - for models with expert groups
- `ffn_moe_probs_masked` - probabilities after group masking
- `ffn_moe_weights_softmax`, `ffn_moe_weights_sum`, `ffn_moe_weights_norm`, `ffn_moe_weights_scaled` - weight normalization stages

## Implementation Strategy

### Option 1: Using eval_callback (Recommended for Research)

This approach intercepts tensors during graph execution without modifying llama.cpp core.

#### Step 1: Create Example Directory

```bash
cd /path/to/your/llama.cpp/fork
mkdir -p examples/moe-logger
```

#### Step 2: Create Source File

Create `examples/moe-logger/moe-logger.cpp`:

```cpp
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include <fstream>
#include <sstream>
#include <map>
#include <mutex>

struct moe_logging_data {
    std::ofstream log_file;
    int current_layer = -1;
    int token_idx = 0;
    std::string prompt_context;
    std::mutex log_mutex;

    // Store expert selection per layer per token
    struct expert_selection {
        std::vector<int32_t> expert_ids;
        std::vector<float> expert_weights;
        std::vector<float> all_logits;  // All expert scores
    };

    std::map<int, std::vector<expert_selection>> layer_selections;
};

static bool moe_expert_logger(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * data = (moe_logging_data *) user_data;

    if (ask) {
        // Only interested in MoE-related tensors
        std::string name(t->name);
        return name.find("ffn_moe") != std::string::npos;
    }

    std::string tensor_name(t->name);

    // Extract layer number from tensor name (e.g., "blk.10.ffn_moe_logits")
    int layer = -1;
    if (sscanf(tensor_name.c_str(), "blk.%d.", &layer) != 1) {
        return true; // Not a layer-specific tensor
    }

    // Copy tensor data from GPU if needed
    std::vector<uint8_t> buffer;
    uint8_t * tensor_data = (uint8_t *) t->data;

    if (!ggml_backend_buffer_is_host(t->buffer)) {
        size_t n_bytes = ggml_nbytes(t);
        buffer.resize(n_bytes);
        ggml_backend_tensor_get(t, buffer.data(), 0, n_bytes);
        tensor_data = buffer.data();
    }

    std::lock_guard<std::mutex> lock(data->log_mutex);

    // Parse different tensor types
    if (tensor_name.find("ffn_moe_logits") != std::string::npos) {
        // Shape: [n_expert, n_tokens]
        int n_experts = t->ne[0];
        int n_tokens = t->ne[1];

        float * logits = (float *) tensor_data;

        data->log_file << "Layer " << layer << " - Raw Logits:\n";
        data->log_file << "  n_experts=" << n_experts << ", n_tokens=" << n_tokens << "\n";
        for (int tok = 0; tok < n_tokens; tok++) {
            data->log_file << "  Token " << tok << ": [";
            for (int exp = 0; exp < std::min(10, n_experts); exp++) {
                data->log_file << logits[tok * n_experts + exp];
                if (exp < std::min(10, n_experts) - 1) data->log_file << ", ";
            }
            if (n_experts > 10) data->log_file << ", ...";
            data->log_file << "]\n";
        }

    } else if (tensor_name.find("ffn_moe_probs") != std::string::npos &&
               tensor_name.find("biased") == std::string::npos) {
        // Shape: [n_expert, n_tokens] or [1, n_expert, n_tokens]
        int n_experts = t->ne[1] > 1 ? t->ne[1] : t->ne[0];
        int n_tokens = t->ne[2] > 1 ? t->ne[2] : (t->ne[1] > 1 ? t->ne[2] : t->ne[1]);

        float * probs = (float *) tensor_data;

        data->log_file << "Layer " << layer << " - Router Probabilities:\n";
        for (int tok = 0; tok < std::min(n_tokens, 3); tok++) {
            data->log_file << "  Token " << tok << ": [";
            for (int exp = 0; exp < std::min(10, n_experts); exp++) {
                data->log_file << probs[tok * n_experts + exp];
                if (exp < std::min(10, n_experts) - 1) data->log_file << ", ";
            }
            if (n_experts > 10) data->log_file << ", ...";
            data->log_file << "]\n";
        }

    } else if (tensor_name.find("ffn_moe_topk") != std::string::npos) {
        // Shape: [n_expert_used, n_tokens]
        // This contains the INDICES of selected experts
        int n_expert_used = t->ne[0];
        int n_tokens = t->ne[1];

        int32_t * indices = (int32_t *) tensor_data;

        data->log_file << "Layer " << layer << " - Selected Expert IDs (top-"
                      << n_expert_used << "):\n";
        for (int tok = 0; tok < n_tokens; tok++) {
            data->log_file << "  Token " << tok << ": [";
            for (int i = 0; i < n_expert_used; i++) {
                data->log_file << indices[tok * n_expert_used + i];
                if (i < n_expert_used - 1) data->log_file << ", ";
            }
            data->log_file << "]\n";
        }

    } else if (tensor_name.find("ffn_moe_weights") != std::string::npos &&
               tensor_name.find("_sum") == std::string::npos &&
               tensor_name.find("_norm") == std::string::npos &&
               tensor_name.find("_scaled") == std::string::npos) {
        // Shape: [1, n_expert_used, n_tokens] or [n_expert_used, n_tokens]
        int n_expert_used = t->ne[1] > 1 ? t->ne[1] : t->ne[0];
        int n_tokens = t->ne[2] > 1 ? t->ne[2] : (t->ne[1] > 1 ? t->ne[2] : t->ne[1]);

        float * weights = (float *) tensor_data;

        data->log_file << "Layer " << layer << " - Expert Routing Weights:\n";
        for (int tok = 0; tok < n_tokens; tok++) {
            data->log_file << "  Token " << tok << ": [";
            float sum = 0.0f;
            for (int i = 0; i < n_expert_used; i++) {
                float w = weights[tok * n_expert_used + i];
                data->log_file << w;
                if (i < n_expert_used - 1) data->log_file << ", ";
                sum += w;
            }
            data->log_file << "] (sum=" << sum << ")\n";
        }
    }

    data->log_file.flush();
    return true;
}

int main(int argc, char ** argv) {
    moe_logging_data moe_data;
    moe_data.log_file.open("moe_expert_selection.log");

    if (!moe_data.log_file.is_open()) {
        fprintf(stderr, "Failed to open log file\n");
        return 1;
    }

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    // CRITICAL: Set the callback
    params.cb_eval = moe_expert_logger;
    params.cb_eval_user_data = &moe_data;
    params.warmup = false;  // Important: warmup skips callbacks

    // Initialize model
    common_init_result llama_init = common_init_from_params(params);
    llama_context * ctx = llama_init.context.get();
    llama_model * model = llama_init.model.get();

    if (!ctx || !model) {
        fprintf(stderr, "Failed to initialize model\n");
        return 1;
    }

    // Process prompt
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);

    if (tokens.empty()) {
        fprintf(stderr, "No tokens to process\n");
        return 1;
    }

    moe_data.log_file << "=== MoE Expert Selection Log ===\n";
    moe_data.log_file << "Prompt: " << params.prompt << "\n";
    moe_data.log_file << "Tokens: " << tokens.size() << "\n\n";

    // Run inference
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        fprintf(stderr, "Failed to decode\n");
        return 1;
    }

    moe_data.log_file << "\n=== Inference Complete ===\n";
    moe_data.log_file.close();

    LOG("\nExpert selection logged to: moe_expert_selection.log\n");

    llama_perf_context_print(ctx);
    llama_backend_free();

    return 0;
}
```

#### Step 3: Create CMakeLists.txt

Create `examples/moe-logger/CMakeLists.txt`:

```cmake
set(TARGET llama-moe-logger)
add_executable(${TARGET} moe-logger.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
```

#### Step 4: Register in Parent CMakeLists

Add to `examples/CMakeLists.txt` (within the existing structure):

```cmake
# examples

if (EMSCRIPTEN)
else()
    # ... existing examples ...
    add_subdirectory(moe-logger)  # Add this line
    # ... more examples ...
endif()
```

#### Step 5: Build

```bash
mkdir -p build
cd build
cmake .. -DLLAMA_BUILD_EXAMPLES=ON
make moe-logger -j$(nproc)
```

#### Step 6: Run

```bash
./bin/llama-moe-logger \
    -m /path/to/your/model.gguf \
    -p "Explain quantum computing" \
    -n 50 \
    --temp 0.7

# Check output
cat moe_expert_selection.log
```

### Option 2: Core Modification (Deep Integration)

For production use or when you need more control, modify `src/llama-graph.cpp`:

Add after the `cb(weights, "ffn_moe_weights", il);` line (around line 993) in `build_moe_ffn()`:

```cpp
cb(weights, "ffn_moe_weights", il);

// ADD THIS BLOCK:
if (const char* log_path = std::getenv("LLAMA_MOE_LOG")) {
    static std::ofstream log_file(log_path, std::ios::app);
    static std::mutex log_mutex;
    static bool initialized = false;

    if (!initialized) {
        log_file << "=== MoE Expert Selection Log ===\n";
        log_file << "Layer,Token,ExpertIDs,ExpertWeights\n";
        initialized = true;
    }

    // Copy weights from GPU if needed
    std::vector<float> weights_cpu(n_expert_used * n_tokens);
    if (!ggml_backend_buffer_is_host(weights->buffer)) {
        ggml_backend_tensor_get(weights, weights_cpu.data(), 0,
                               sizeof(float) * n_expert_used * n_tokens);
    } else {
        memcpy(weights_cpu.data(), weights->data,
               sizeof(float) * n_expert_used * n_tokens);
    }

    // Copy selected expert indices
    std::vector<int32_t> expert_ids(n_expert_used * n_tokens);
    if (!ggml_backend_buffer_is_host(selected_experts->buffer)) {
        ggml_backend_tensor_get(selected_experts, expert_ids.data(), 0,
                               sizeof(int32_t) * n_expert_used * n_tokens);
    } else {
        memcpy(expert_ids.data(), selected_experts->data,
               sizeof(int32_t) * n_expert_used * n_tokens);
    }

    std::lock_guard<std::mutex> lock(log_mutex);

    for (int tok = 0; tok < n_tokens; tok++) {
        log_file << il << "," << tok << ",\"[";
        for (int i = 0; i < n_expert_used; i++) {
            log_file << expert_ids[tok * n_expert_used + i];
            if (i < n_expert_used - 1) log_file << " ";
        }
        log_file << "]\",\"[";
        for (int i = 0; i < n_expert_used; i++) {
            log_file << weights_cpu[tok * n_expert_used + i];
            if (i < n_expert_used - 1) log_file << " ";
        }
        log_file << "]\"\n";
    }
    log_file.flush();
}
```

Then rebuild and run with:

```bash
export LLAMA_MOE_LOG=expert_log.csv
./bin/llama-cli -m model.gguf -p "test prompt"
```

## Output Format Examples

### Sample Log Output

```
=== MoE Expert Selection Log ===
Prompt: Explain quantum computing
Tokens: 15

Layer 10 - Selected Expert IDs (top-8):
  Token 0: [42, 105, 17, 88, 134, 3, 99, 12]
  Token 1: [42, 17, 88, 105, 12, 99, 3, 67]
  Token 2: [105, 42, 88, 17, 12, 3, 99, 134]

Layer 10 - Expert Routing Weights:
  Token 0: [0.245, 0.189, 0.156, 0.112, 0.089, 0.078, 0.067, 0.064] (sum=1.000)
  Token 1: [0.267, 0.201, 0.145, 0.128, 0.091, 0.073, 0.058, 0.037] (sum=1.000)
  Token 2: [0.234, 0.198, 0.167, 0.143, 0.102, 0.084, 0.051, 0.021] (sum=1.000)
```

## Analysis Scripts

### Python Analysis Example

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Parse log file
def parse_moe_log(log_path):
    data = []
    current_layer = None

    with open(log_path) as f:
        for line in f:
            if 'Layer' in line and 'Selected Expert IDs' in line:
                current_layer = int(line.split()[1])
            elif 'Token' in line and ':' in line and '[' in line:
                # Parse token line
                parts = line.split(':')
                token_idx = int(parts[0].split()[-1])
                experts = [int(x.strip()) for x in parts[1].strip()[1:-1].split(',')]

                for exp in experts:
                    data.append({
                        'layer': current_layer,
                        'token': token_idx,
                        'expert': exp
                    })

    return pd.DataFrame(data)

# Analyze expert usage
df = parse_moe_log('moe_expert_selection.log')

# Expert frequency across all tokens
expert_freq = df['expert'].value_counts().sort_index()
plt.figure(figsize=(15, 5))
plt.bar(expert_freq.index, expert_freq.values)
plt.xlabel('Expert ID')
plt.ylabel('Selection Count')
plt.title('Expert Selection Frequency')
plt.savefig('expert_frequency.png')
plt.close()

# Heatmap: experts vs tokens
pivot = df.pivot_table(index='expert', columns='token', aggfunc='size', fill_value=0)
plt.figure(figsize=(15, 10))
plt.imshow(pivot.values, aspect='auto', cmap='hot', interpolation='nearest')
plt.colorbar(label='Selection Count')
plt.xlabel('Token Position')
plt.ylabel('Expert ID')
plt.title('Expert Selection Heatmap')
plt.savefig('expert_heatmap.png')
plt.close()

print("Analysis complete. Charts saved.")
```

## Debugging Tips

### 1. Check Tensor Names

If tensors aren't being captured, first identify the actual tensor names in your model:

```cpp
// Temporary debug version
static bool moe_expert_logger(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) return true;  // Get everything
    fprintf(stderr, "Tensor: %s, shape: [%ld, %ld, %ld, %ld]\n",
            t->name, t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    return true;
}
```

### 2. Inspect Model Structure

```bash
python3 -c "
import gguf
reader = gguf.GGUFReader('your-model.gguf')
tensors = [t.name for t in reader.tensors]
moe_tensors = [t for t in tensors if 'ffn' in t and '0' in t]
for t in sorted(moe_tensors)[:20]:
    print(t)
"
```

### 3. Common Issues

**Empty Logs:**
- Verify `params.cb_eval` is set before `common_init_from_params`
- Ensure `params.warmup = false`
- Check tensor name matching in callback

**Shape Mismatches:**
- Print `t->ne[0], t->ne[1], t->ne[2], t->ne[3]` to debug dimensions
- Different models may transpose dimensions

**GPU Memory Issues:**
- Test with `--n-gpu-layers 0` first (CPU-only)
- Gradually increase GPU offloading

**Performance:**
- Callback has overhead; use judiciously
- For production, use Option 2 with environment variable guard

## Usage Examples

### Example 1: Code Generation

```bash
./bin/llama-moe-logger \
    -m deepseek-model.gguf \
    -p "def fibonacci(n):" \
    -n 100 \
    --temp 0.2
```

### Example 2: Math Problem

```bash
./bin/llama-moe-logger \
    -m deepseek-model.gguf \
    -p "Solve: 3x + 7 = 22" \
    -n 50 \
    --temp 0.1
```

### Example 3: Creative Writing

```bash
./bin/llama-moe-logger \
    -m deepseek-model.gguf \
    -p "Once upon a time in a distant galaxy" \
    -n 200 \
    --temp 0.8
```

### Example 4: Batch Processing

```bash
#!/bin/bash
for prompt in "code" "math" "story"; do
    ./bin/llama-moe-logger \
        -m model.gguf \
        -p "$(cat prompts/${prompt}.txt)" \
        -n 100
    mv moe_expert_selection.log logs/${prompt}_experts.log
done
```

## Research Applications

1. **Expert Specialization**: Identify which experts activate for different task types
2. **Token-level Analysis**: Track expert routing patterns across sequence positions
3. **Layer-wise Patterns**: Compare early vs late layer routing behavior
4. **Load Balancing**: Detect expert imbalance or underutilization
5. **Routing Diversity**: Measure routing entropy and consistency

## Performance Considerations

- **Callback Overhead**: ~10-20% slowdown during inference
- **File I/O**: Can be significant; consider buffering for production
- **GPU Transfers**: Copying tensors from GPU to CPU adds latency
- **Log Size**: Can grow quickly for long sequences; implement rotation

## Next Steps

After collecting data:

1. **Visualize patterns** using the Python scripts provided
2. **Correlate with token types** (code vs text vs math)
3. **Compare across models** (different sizes, architectures)
4. **Analyze routing stability** (same prompt, different runs)
5. **Study expert semantics** (what does each expert "specialize" in?)

## References

- llama.cpp MoE implementation: `src/llama-graph.cpp:877-1026`
- GGML callback system: `ggml/include/ggml-backend.h:303`
- eval-callback example: `examples/eval-callback/eval-callback.cpp`

## Contributing

If you develop improvements or analysis tools, consider contributing back to the llama.cpp community!
