// Layer extraction utility for llama.cpp
// Runs a single transformer layer on input hidden states and produces output.
// Useful for cross-validation with other inference engines.

#include "llama.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

// Internal headers for direct layer access
#include "../../src/llama-model.h"
#include "../../src/llama-hparams.h"

#include "npy-io.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

struct layer_extract_params {
    std::string model_path;
    std::string input_path;
    std::string output_path;
    std::string pos_path;
    int layer_idx = -1;
    int n_threads = 1;
    bool causal = false;
    bool verbose = false;
};

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Extract and run a single transformer layer from a GGUF model.\n");
    fprintf(stderr, "Useful for cross-validation with other inference engines.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Required options:\n");
    fprintf(stderr, "  -m, --model <path>     Path to GGUF model file\n");
    fprintf(stderr, "  -l, --layer <n>        Layer index (0-based)\n");
    fprintf(stderr, "  -i, --input <path>     Input hidden states .npy file [n_tokens, n_embd]\n");
    fprintf(stderr, "  -o, --output <path>    Output hidden states .npy file\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Optional:\n");
    fprintf(stderr, "  --pos <path>           Position IDs .npy file [n_tokens] (default: 0,1,2,...)\n");
    fprintf(stderr, "  --causal               Use causal attention mask\n");
    fprintf(stderr, "  -t, --threads <n>      Number of threads (default: 1)\n");
    fprintf(stderr, "  -v, --verbose          Verbose output\n");
    fprintf(stderr, "  -h, --help             Show this help\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Example:\n");
    fprintf(stderr, "  %s -m model.gguf -l 0 -i input.npy -o output.npy --causal\n", prog);
    fprintf(stderr, "\n");
}

static bool parse_args(int argc, char ** argv, layer_extract_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) { fprintf(stderr, "Error: missing model path\n"); return false; }
            params.model_path = argv[i];
        } else if (arg == "-l" || arg == "--layer") {
            if (++i >= argc) { fprintf(stderr, "Error: missing layer index\n"); return false; }
            params.layer_idx = atoi(argv[i]);
        } else if (arg == "-i" || arg == "--input") {
            if (++i >= argc) { fprintf(stderr, "Error: missing input path\n"); return false; }
            params.input_path = argv[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) { fprintf(stderr, "Error: missing output path\n"); return false; }
            params.output_path = argv[i];
        } else if (arg == "--pos") {
            if (++i >= argc) { fprintf(stderr, "Error: missing pos path\n"); return false; }
            params.pos_path = argv[i];
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) { fprintf(stderr, "Error: missing threads\n"); return false; }
            params.n_threads = atoi(argv[i]);
        } else if (arg == "--causal") {
            params.causal = true;
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else {
            fprintf(stderr, "Error: unknown argument '%s'\n", arg.c_str());
            return false;
        }
    }

    if (params.model_path.empty()) { fprintf(stderr, "Error: model path required (-m)\n"); return false; }
    if (params.layer_idx < 0) { fprintf(stderr, "Error: layer index required (-l)\n"); return false; }
    if (params.input_path.empty()) { fprintf(stderr, "Error: input path required (-i)\n"); return false; }
    if (params.output_path.empty()) { fprintf(stderr, "Error: output path required (-o)\n"); return false; }

    return true;
}

// Build attention mask (causal or non-causal)
static ggml_tensor * build_attn_mask(ggml_context * ctx, int64_t n_tokens, bool /*causal*/) {
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_set_name(mask, "attn_mask");
    return mask;
}

// Fill attention mask data
static void fill_attn_mask(float * data, int64_t n_tokens, bool causal) {
    for (int64_t i = 0; i < n_tokens; i++) {
        for (int64_t j = 0; j < n_tokens; j++) {
            if (causal && j > i) {
                // Causal: mask future tokens with -inf
                data[i * n_tokens + j] = -INFINITY;
            } else {
                data[i * n_tokens + j] = 0.0f;
            }
        }
    }
}

int main(int argc, char ** argv) {
    layer_extract_params params;
    if (!parse_args(argc, argv, params)) {
        print_usage(argv[0]);
        return 1;
    }

    // Load dynamic backends (required)
    ggml_backend_load_all();

    // Load model
    if (params.verbose) {
        fprintf(stderr, "Loading model: %s\n", params.model_path.c_str());
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU only for determinism
    model_params.use_mmap = true;

    llama_model * model = llama_model_load_from_file(params.model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Error: failed to load model\n");
        return 1;
    }

    // Get model parameters
    const int32_t n_layer = llama_model_n_layer(model);
    const int32_t n_embd = llama_model_n_embd(model);
    const int32_t n_head = llama_model_n_head(model);
    const int32_t n_head_kv = llama_model_n_head_kv(model);
    const int32_t n_embd_head = n_embd / n_head;
    const llama_rope_type rope_type = llama_model_rope_type(model);

    // Access internal model structure
    const llama_hparams & hparams = model->hparams;
    const int32_t n_rot = hparams.n_rot;
    const float rope_freq_base = hparams.rope_freq_base_train;
    const float rope_freq_scale = hparams.rope_freq_scale_train;
    const float f_norm_rms_eps = hparams.f_norm_rms_eps;

    if (params.verbose) {
        fprintf(stderr, "Model: n_layer=%d, n_embd=%d, n_head=%d, n_head_kv=%d, n_rot=%d\n",
                n_layer, n_embd, n_head, n_head_kv, n_rot);
        fprintf(stderr, "RoPE: type=%d, freq_base=%.1f, freq_scale=%.4f\n",
                (int)rope_type, rope_freq_base, rope_freq_scale);
    }

    if (params.layer_idx >= n_layer) {
        fprintf(stderr, "Error: layer index %d out of range (0-%d)\n", params.layer_idx, n_layer - 1);
        llama_model_free(model);
        return 1;
    }

    const llama_layer & layer = model->layers[params.layer_idx];

    // Read input tensor
    std::vector<float> input_data;
    std::vector<size_t> input_shape;
    if (!npy::read_f32(params.input_path, input_data, input_shape)) {
        fprintf(stderr, "Error: failed to read input from '%s'\n", params.input_path.c_str());
        llama_model_free(model);
        return 1;
    }

    if (input_shape.size() != 2) {
        fprintf(stderr, "Error: input must be 2D [n_tokens, n_embd], got %zu dimensions\n", input_shape.size());
        llama_model_free(model);
        return 1;
    }

    const int64_t n_tokens = (int64_t)input_shape[0];
    const int64_t input_n_embd = (int64_t)input_shape[1];

    if (input_n_embd != n_embd) {
        fprintf(stderr, "Error: input n_embd=%lld doesn't match model n_embd=%d\n",
                (long long)input_n_embd, n_embd);
        llama_model_free(model);
        return 1;
    }

    if (params.verbose) {
        fprintf(stderr, "Input: n_tokens=%lld, n_embd=%lld\n", (long long)n_tokens, (long long)input_n_embd);
    }

    // Read or generate position IDs
    std::vector<int32_t> positions(n_tokens);
    if (!params.pos_path.empty()) {
        std::vector<size_t> pos_shape;
        if (!npy::read_i32(params.pos_path, positions, pos_shape)) {
            fprintf(stderr, "Error: failed to read positions from '%s'\n", params.pos_path.c_str());
            llama_model_free(model);
            return 1;
        }
        if (pos_shape.size() != 1 || pos_shape[0] != (size_t)n_tokens) {
            fprintf(stderr, "Error: positions shape mismatch\n");
            llama_model_free(model);
            return 1;
        }
    } else {
        for (int64_t i = 0; i < n_tokens; i++) {
            positions[i] = (int32_t)i;
        }
    }

    // Create CPU backend
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "Error: failed to create CPU backend\n");
        llama_model_free(model);
        return 1;
    }

    // Set number of threads
    ggml_backend_cpu_set_n_threads(backend, params.n_threads);

    // Create GGML context for computation graph
    // Estimate memory needed for the graph
    const size_t mem_size = 256 * 1024 * 1024;  // 256 MB should be enough for one layer's graph
    struct ggml_init_params ctx_params = {
        .mem_size   = mem_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,  // We'll use ggml_gallocr for allocation
    };
    ggml_context * ctx = ggml_init(ctx_params);
    if (!ctx) {
        fprintf(stderr, "Error: failed to create GGML context\n");
        ggml_backend_free(backend);
        llama_model_free(model);
        return 1;
    }

    // ==================== Build Computation Graph ====================

    // Input tensor
    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(inp, "input");
    ggml_set_input(inp);

    // Position IDs tensor
    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "positions");
    ggml_set_input(inp_pos);

    // Attention mask
    ggml_tensor * attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_set_name(attn_mask, "attn_mask");
    ggml_set_input(attn_mask);

    ggml_tensor * cur = inp;
    ggml_tensor * inpSA = cur;

    // 1. Attention normalization (RMSNorm)
    cur = ggml_rms_norm(ctx, cur, f_norm_rms_eps);
    cur = ggml_mul(ctx, cur, layer.attn_norm);
    ggml_set_name(cur, "attn_norm_out");

    // 2. Q, K, V projections
    ggml_tensor * Qcur = ggml_mul_mat(ctx, layer.wq, cur);
    ggml_tensor * Kcur = ggml_mul_mat(ctx, layer.wk, cur);
    ggml_tensor * Vcur = ggml_mul_mat(ctx, layer.wv, cur);

    // Add biases if present
    if (layer.bq) Qcur = ggml_add(ctx, Qcur, layer.bq);
    if (layer.bk) Kcur = ggml_add(ctx, Kcur, layer.bk);
    if (layer.bv) Vcur = ggml_add(ctx, Vcur, layer.bv);

    // Reshape for multi-head attention: [n_embd, n_tokens] -> [n_embd_head, n_head, n_tokens]
    Qcur = ggml_reshape_3d(ctx, Qcur, n_embd_head, n_head, n_tokens);
    Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head_kv, n_tokens);

    // 3. Apply RoPE
    // ggml_rope_ext params: (ctx, a, b=pos, c=freq_factors, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
    Qcur = ggml_rope_ext(ctx, Qcur, inp_pos, nullptr,
                         n_rot, (int)rope_type, 0, rope_freq_base, rope_freq_scale,
                         0.0f, 1.0f, 0.0f, 0.0f);
    Kcur = ggml_rope_ext(ctx, Kcur, inp_pos, nullptr,
                         n_rot, (int)rope_type, 0, rope_freq_base, rope_freq_scale,
                         0.0f, 1.0f, 0.0f, 0.0f);

    ggml_set_name(Qcur, "Q_rope");
    ggml_set_name(Kcur, "K_rope");
    ggml_set_name(Vcur, "V");

    // 4. Self-attention using flash attention (handles GQA internally)
    // Q: [n_embd_head, n_head, n_tokens]
    // K: [n_embd_head, n_head_kv, n_tokens]
    // V: [n_embd_head, n_head_kv, n_tokens]

    // Permute for flash attention:
    // Q: [n_embd_head, n_head, n_tokens] -> [n_embd_head, n_tokens, n_head]
    // K: [n_embd_head, n_head_kv, n_tokens] -> [n_embd_head, n_tokens, n_head_kv]
    // V: [n_embd_head, n_head_kv, n_tokens] -> [n_embd_head, n_tokens, n_head_kv]
    Qcur = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
    Kcur = ggml_permute(ctx, Kcur, 0, 2, 1, 3);
    Vcur = ggml_permute(ctx, Vcur, 0, 2, 1, 3);

    // Make K and V contiguous (required for flash attention)
    Kcur = ggml_cont(ctx, Kcur);
    Vcur = ggml_cont(ctx, Vcur);

    // Scale for attention
    const float kq_scale = 1.0f / sqrtf((float)n_embd_head);

    // Use flash attention which handles GQA internally
    // ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias, logit_softcap)
    ggml_tensor * kqv = ggml_flash_attn_ext(ctx, Qcur, Kcur, Vcur, attn_mask, kq_scale, 0.0f, 0.0f);
    ggml_set_name(kqv, "kqv");

    // Output shape: [n_embd_head, n_tokens, n_head, 1]
    // Permute back to [n_embd_head, n_head, n_tokens, 1]
    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    kqv = ggml_cont(ctx, kqv);

    // Reshape to [n_embd, n_tokens]
    kqv = ggml_reshape_2d(ctx, kqv, n_embd, n_tokens);

    // 5. Output projection
    cur = ggml_mul_mat(ctx, layer.wo, kqv);
    if (layer.bo) cur = ggml_add(ctx, cur, layer.bo);
    ggml_set_name(cur, "attn_out");

    // 6. Residual connection
    ggml_tensor * ffn_inp = ggml_add(ctx, cur, inpSA);
    ggml_set_name(ffn_inp, "ffn_inp");

    // 7. FFN normalization (RMSNorm)
    cur = ggml_rms_norm(ctx, ffn_inp, f_norm_rms_eps);
    cur = ggml_mul(ctx, cur, layer.ffn_norm);
    ggml_set_name(cur, "ffn_norm_out");

    // 8. FFN (SwiGLU: gate * silu(up))
    ggml_tensor * ffn_up = ggml_mul_mat(ctx, layer.ffn_up, cur);
    ggml_tensor * ffn_gate = ggml_mul_mat(ctx, layer.ffn_gate, cur);

    if (layer.ffn_up_b) ffn_up = ggml_add(ctx, ffn_up, layer.ffn_up_b);
    if (layer.ffn_gate_b) ffn_gate = ggml_add(ctx, ffn_gate, layer.ffn_gate_b);

    ffn_gate = ggml_silu(ctx, ffn_gate);
    cur = ggml_mul(ctx, ffn_up, ffn_gate);
    cur = ggml_mul_mat(ctx, layer.ffn_down, cur);
    if (layer.ffn_down_b) cur = ggml_add(ctx, cur, layer.ffn_down_b);
    ggml_set_name(cur, "ffn_out");

    // 9. Final residual connection
    cur = ggml_add(ctx, cur, ffn_inp);
    ggml_set_name(cur, "layer_out");
    ggml_set_output(cur);

    // Build computation graph
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);

    if (params.verbose) {
        fprintf(stderr, "Graph: %d nodes\n", ggml_graph_n_nodes(gf));
    }

    // ==================== Allocate and Compute ====================

    // Create graph allocator
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    // Pre-allocate graph memory
    if (!ggml_gallocr_reserve(allocr, gf)) {
        fprintf(stderr, "Error: failed to reserve graph memory\n");
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        ggml_backend_free(backend);
        llama_model_free(model);
        return 1;
    }

    // Allocate graph tensors
    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "Error: failed to allocate graph\n");
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        ggml_backend_free(backend);
        llama_model_free(model);
        return 1;
    }

    // Set input data
    ggml_backend_tensor_set(inp, input_data.data(), 0, n_tokens * n_embd * sizeof(float));
    ggml_backend_tensor_set(inp_pos, positions.data(), 0, n_tokens * sizeof(int32_t));

    // Fill and set attention mask
    std::vector<float> mask_data(n_tokens * n_tokens);
    fill_attn_mask(mask_data.data(), n_tokens, params.causal);
    ggml_backend_tensor_set(attn_mask, mask_data.data(), 0, n_tokens * n_tokens * sizeof(float));

    // Compute
    if (params.verbose) {
        fprintf(stderr, "Computing layer %d...\n", params.layer_idx);
    }

    enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "Error: graph compute failed with status %d\n", (int)status);
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        ggml_backend_free(backend);
        llama_model_free(model);
        return 1;
    }

    // Get output
    std::vector<float> output_data(n_tokens * n_embd);
    ggml_backend_tensor_get(cur, output_data.data(), 0, n_tokens * n_embd * sizeof(float));

    // Write output
    if (!npy::write_f32(params.output_path, output_data.data(), {(size_t)n_tokens, (size_t)n_embd})) {
        fprintf(stderr, "Error: failed to write output to '%s'\n", params.output_path.c_str());
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        ggml_backend_free(backend);
        llama_model_free(model);
        return 1;
    }

    if (params.verbose) {
        fprintf(stderr, "Output written to: %s\n", params.output_path.c_str());
    }

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_free(backend);
    llama_model_free(model);

    fprintf(stderr, "Done.\n");
    return 0;
}
