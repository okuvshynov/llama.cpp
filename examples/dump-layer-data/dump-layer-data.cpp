// dump-layer-data: Dump layer weights and activations to .npy files for external verification
//
// This tool dequantizes weights from a GGUF model and captures layer input/output
// activations during inference. Output is saved as NumPy .npy files.
//
// Supports:
//   - MLP weights and activations (gate, up, down projections)
//   - Attention weights and activations (Q, K, V, O projections)
//
// Usage:
//   ./dump-layer-data -m model.gguf -p "Hello" --layer 0 --output-dir ./layer_dump
//

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>

// Include internal headers for model structure access
#include "llama-model.h"

// ============================================================================
// NPY File Writer
// ============================================================================

static void save_npy_2d(const std::string & path, const float * data,
                        int64_t rows, int64_t cols, bool transpose = false) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        LOG_ERR("Failed to open %s for writing\n", path.c_str());
        return;
    }

    const char magic[] = "\x93NUMPY";
    fwrite(magic, 1, 6, f);

    uint8_t version[2] = {1, 0};
    fwrite(version, 1, 2, f);

    char header[256];
    if (transpose) {
        snprintf(header, sizeof(header),
                 "{'descr': '<f4', 'fortran_order': False, 'shape': (%lld, %lld), }",
                 (long long)cols, (long long)rows);
    } else {
        snprintf(header, sizeof(header),
                 "{'descr': '<f4', 'fortran_order': False, 'shape': (%lld, %lld), }",
                 (long long)rows, (long long)cols);
    }

    size_t header_len = strlen(header);
    size_t total_header = 10 + header_len + 1;
    size_t padding = (64 - (total_header % 64)) % 64;

    for (size_t i = 0; i < padding; i++) {
        header[header_len + i] = ' ';
    }
    header[header_len + padding] = '\n';
    header_len += padding + 1;

    uint16_t header_len_u16 = (uint16_t)header_len;
    fwrite(&header_len_u16, 2, 1, f);
    fwrite(header, 1, header_len, f);

    if (transpose) {
        for (int64_t j = 0; j < cols; j++) {
            for (int64_t i = 0; i < rows; i++) {
                float val = data[i * cols + j];
                fwrite(&val, sizeof(float), 1, f);
            }
        }
    } else {
        fwrite(data, sizeof(float), rows * cols, f);
    }

    fclose(f);
    LOG_INF("Saved %s: shape=(%lld, %lld)%s\n", path.c_str(),
            transpose ? (long long)cols : (long long)rows,
            transpose ? (long long)rows : (long long)cols,
            transpose ? " (transposed)" : "");
}

// ============================================================================
// Tensor Dequantization
// ============================================================================

static std::vector<float> dequantize_tensor(const ggml_tensor * t) {
    const int64_t n_elements = ggml_nelements(t);
    std::vector<float> result(n_elements);

    if (t->type == GGML_TYPE_F32) {
        memcpy(result.data(), t->data, n_elements * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = (const ggml_fp16_t *)t->data;
        for (int64_t i = 0; i < n_elements; i++) {
            result[i] = ggml_fp16_to_fp32(src[i]);
        }
    } else {
        const ggml_type_traits * traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) {
            traits->to_float(t->data, result.data(), n_elements);
        } else {
            LOG_ERR("Unsupported tensor type: %s\n", ggml_type_name(t->type));
            result.clear();
        }
    }

    return result;
}

// ============================================================================
// Callback Data Structure
// ============================================================================

struct dump_config {
    int target_layer;
    std::string output_dir;
    bool dump_activations;

    // Track what we've dumped
    bool dumped_l_inp;      // Layer input (before RMSNorm)
    bool dumped_attn_norm;
    bool dumped_ffn_norm;
    bool dumped_ffn_out;
    bool dumped_l_out;      // Layer output (after second residual)

    // Temporary buffer for GPU->CPU copy
    std::vector<uint8_t> data_buf;
};

// ============================================================================
// Activation Callback
// ============================================================================

static bool dump_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cfg = (dump_config *)user_data;

    if (!cfg->dump_activations) {
        return true;
    }

    std::string name(t->name);

    // Build target names for the layer we care about
    std::string l_inp_name = "l_inp-" + std::to_string(cfg->target_layer);
    std::string attn_norm_name = "attn_norm-" + std::to_string(cfg->target_layer);
    std::string ffn_norm_name = "ffn_norm-" + std::to_string(cfg->target_layer);
    std::string ffn_out_name = "ffn_out-" + std::to_string(cfg->target_layer);
    std::string l_out_name = "l_out-" + std::to_string(cfg->target_layer);

    bool is_l_inp = (name == l_inp_name);
    bool is_attn_norm = (name == attn_norm_name);
    bool is_ffn_norm = (name == ffn_norm_name);
    bool is_ffn_out = (name == ffn_out_name);
    bool is_l_out = (name == l_out_name);

    if (ask) {
        return is_l_inp || is_attn_norm || is_ffn_norm || is_ffn_out || is_l_out;
    }

    // Copy data from GPU if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    uint8_t * data_ptr;

    if (!is_host) {
        size_t n_bytes = ggml_nbytes(t);
        cfg->data_buf.resize(n_bytes);
        ggml_backend_tensor_get(t, cfg->data_buf.data(), 0, n_bytes);
        data_ptr = cfg->data_buf.data();
    } else {
        data_ptr = (uint8_t *)t->data;
    }

    // Convert to FP32 if needed
    std::vector<float> fp32_data;
    const float * float_ptr;

    if (t->type == GGML_TYPE_F32) {
        float_ptr = (const float *)data_ptr;
    } else if (t->type == GGML_TYPE_F16) {
        fp32_data.resize(ggml_nelements(t));
        const ggml_fp16_t * src = (const ggml_fp16_t *)data_ptr;
        for (int64_t i = 0; i < ggml_nelements(t); i++) {
            fp32_data[i] = ggml_fp16_to_fp32(src[i]);
        }
        float_ptr = fp32_data.data();
    } else {
        LOG_ERR("Unexpected activation type: %s\n", ggml_type_name(t->type));
        return true;
    }

    int64_t cols = t->ne[0];
    int64_t rows = t->ne[1];
    if (t->ne[2] > 1 || t->ne[3] > 1) {
        LOG_WRN("Tensor %s has more than 2 dimensions: [%lld, %lld, %lld, %lld]\n",
                name.c_str(), (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->ne[2], (long long)t->ne[3]);
    }

    // Save to file
    if (is_l_inp && !cfg->dumped_l_inp) {
        std::string path = cfg->output_dir + "/" + name + ".npy";
        save_npy_2d(path, float_ptr, rows, cols, false);
        cfg->dumped_l_inp = true;
    } else if (is_attn_norm && !cfg->dumped_attn_norm) {
        std::string path = cfg->output_dir + "/" + name + ".npy";
        save_npy_2d(path, float_ptr, rows, cols, false);
        cfg->dumped_attn_norm = true;
    } else if (is_ffn_norm && !cfg->dumped_ffn_norm) {
        std::string path = cfg->output_dir + "/" + name + ".npy";
        save_npy_2d(path, float_ptr, rows, cols, false);
        cfg->dumped_ffn_norm = true;
    } else if (is_ffn_out && !cfg->dumped_ffn_out) {
        std::string path = cfg->output_dir + "/" + name + ".npy";
        save_npy_2d(path, float_ptr, rows, cols, false);
        cfg->dumped_ffn_out = true;
    } else if (is_l_out && !cfg->dumped_l_out) {
        std::string path = cfg->output_dir + "/" + name + ".npy";
        save_npy_2d(path, float_ptr, rows, cols, false);
        cfg->dumped_l_out = true;
    }

    return true;
}

// ============================================================================
// Dump Layer Weights
// ============================================================================

static void dump_layer_weights(const llama_model * model, int layer_idx, const std::string & output_dir) {
    if (layer_idx < 0 || layer_idx >= (int)model->layers.size()) {
        LOG_ERR("Invalid layer index %d (model has %zu layers)\n", layer_idx, model->layers.size());
        return;
    }

    const llama_layer & layer = model->layers[layer_idx];
    std::string prefix = output_dir + "/layer" + std::to_string(layer_idx) + "_";

    // ========== RMSNorm Weights ==========
    LOG_INF("Dumping RMSNorm weights for layer %d\n", layer_idx);

    // Attention input RMSNorm (pre-attention normalization)
    if (layer.attn_norm) {
        LOG_INF("  attn_norm type: %s, shape: [%lld]\n",
                ggml_type_name(layer.attn_norm->type),
                (long long)layer.attn_norm->ne[0]);
        std::vector<float> data = dequantize_tensor(layer.attn_norm);
        if (!data.empty()) {
            // 1D tensor - save as (1, hidden)
            int64_t hidden = layer.attn_norm->ne[0];
            save_npy_2d(prefix + "attn_norm.npy", data.data(), 1, hidden, false);
        }
    }

    // FFN input RMSNorm (post-attention normalization)
    if (layer.ffn_norm) {
        LOG_INF("  ffn_norm type: %s, shape: [%lld]\n",
                ggml_type_name(layer.ffn_norm->type),
                (long long)layer.ffn_norm->ne[0]);
        std::vector<float> data = dequantize_tensor(layer.ffn_norm);
        if (!data.empty()) {
            // 1D tensor - save as (1, hidden)
            int64_t hidden = layer.ffn_norm->ne[0];
            save_npy_2d(prefix + "ffn_norm_weight.npy", data.data(), 1, hidden, false);
        }
    }

    // ========== Attention Weights ==========
    LOG_INF("\nDumping attention weights for layer %d\n", layer_idx);

    // Q projection: wq
    if (layer.wq) {
        LOG_INF("  wq type: %s, shape: [%lld, %lld]\n",
                ggml_type_name(layer.wq->type),
                (long long)layer.wq->ne[0], (long long)layer.wq->ne[1]);
        std::vector<float> data = dequantize_tensor(layer.wq);
        if (!data.empty()) {
            // llama.cpp: (out_features, in_features) -> transpose to (in_features, out_features)
            int64_t rows = layer.wq->ne[1];  // num_heads * head_dim (or hidden for some models)
            int64_t cols = layer.wq->ne[0];  // hidden_size
            save_npy_2d(prefix + "wq.npy", data.data(), rows, cols, true);
        }
    }

    // K projection: wk
    if (layer.wk) {
        LOG_INF("  wk type: %s, shape: [%lld, %lld]\n",
                ggml_type_name(layer.wk->type),
                (long long)layer.wk->ne[0], (long long)layer.wk->ne[1]);
        std::vector<float> data = dequantize_tensor(layer.wk);
        if (!data.empty()) {
            int64_t rows = layer.wk->ne[1];
            int64_t cols = layer.wk->ne[0];
            save_npy_2d(prefix + "wk.npy", data.data(), rows, cols, true);
        }
    }

    // V projection: wv
    if (layer.wv) {
        LOG_INF("  wv type: %s, shape: [%lld, %lld]\n",
                ggml_type_name(layer.wv->type),
                (long long)layer.wv->ne[0], (long long)layer.wv->ne[1]);
        std::vector<float> data = dequantize_tensor(layer.wv);
        if (!data.empty()) {
            int64_t rows = layer.wv->ne[1];
            int64_t cols = layer.wv->ne[0];
            save_npy_2d(prefix + "wv.npy", data.data(), rows, cols, true);
        }
    }

    // O projection: wo
    if (layer.wo) {
        LOG_INF("  wo type: %s, shape: [%lld, %lld]\n",
                ggml_type_name(layer.wo->type),
                (long long)layer.wo->ne[0], (long long)layer.wo->ne[1]);
        std::vector<float> data = dequantize_tensor(layer.wo);
        if (!data.empty()) {
            int64_t rows = layer.wo->ne[1];
            int64_t cols = layer.wo->ne[0];
            save_npy_2d(prefix + "wo.npy", data.data(), rows, cols, true);
        }
    }

    // ========== MLP Weights ==========
    if (layer.ffn_gate && layer.ffn_up && layer.ffn_down) {
        LOG_INF("\nDumping MLP weights for layer %d\n", layer_idx);

        // ffn_gate
        LOG_INF("  ffn_gate type: %s, shape: [%lld, %lld]\n",
                ggml_type_name(layer.ffn_gate->type),
                (long long)layer.ffn_gate->ne[0], (long long)layer.ffn_gate->ne[1]);
        {
            std::vector<float> data = dequantize_tensor(layer.ffn_gate);
            if (!data.empty()) {
                int64_t rows = layer.ffn_gate->ne[1];
                int64_t cols = layer.ffn_gate->ne[0];
                save_npy_2d(prefix + "ffn_gate.npy", data.data(), rows, cols, true);
            }
        }

        // ffn_up
        LOG_INF("  ffn_up type: %s, shape: [%lld, %lld]\n",
                ggml_type_name(layer.ffn_up->type),
                (long long)layer.ffn_up->ne[0], (long long)layer.ffn_up->ne[1]);
        {
            std::vector<float> data = dequantize_tensor(layer.ffn_up);
            if (!data.empty()) {
                int64_t rows = layer.ffn_up->ne[1];
                int64_t cols = layer.ffn_up->ne[0];
                save_npy_2d(prefix + "ffn_up.npy", data.data(), rows, cols, true);
            }
        }

        // ffn_down
        LOG_INF("  ffn_down type: %s, shape: [%lld, %lld]\n",
                ggml_type_name(layer.ffn_down->type),
                (long long)layer.ffn_down->ne[0], (long long)layer.ffn_down->ne[1]);
        {
            std::vector<float> data = dequantize_tensor(layer.ffn_down);
            if (!data.empty()) {
                int64_t rows = layer.ffn_down->ne[1];
                int64_t cols = layer.ffn_down->ne[0];
                save_npy_2d(prefix + "ffn_down.npy", data.data(), rows, cols, true);
            }
        }
    }

    LOG_INF("\nDone dumping layer weights\n");
}

// ============================================================================
// Main
// ============================================================================

static void print_usage(const char * prog) {
    LOG("\nUsage: %s [options]\n\n", prog);
    LOG("Options:\n");
    LOG("  -m, --model FILE     Model file (required)\n");
    LOG("  -p, --prompt TEXT    Prompt for activation capture (optional)\n");
    LOG("  --layer N            Layer index to dump (default: 0)\n");
    LOG("  --output-dir DIR     Output directory (default: ./layer_dump)\n");
    LOG("  --weights-only       Only dump weights, skip activation capture\n");
    LOG("  -h, --help           Show this help\n");
    LOG("\n");
    LOG("Output files:\n");
    LOG("  Weights:\n");
    LOG("    layer<N>_attn_norm.npy      Input RMSNorm weight (1, hidden)\n");
    LOG("    layer<N>_ffn_norm_weight.npy Post-attention RMSNorm weight (1, hidden)\n");
    LOG("    layer<N>_wq.npy             Q projection weights (hidden, num_heads*head_dim)\n");
    LOG("    layer<N>_wk.npy             K projection weights (hidden, num_kv_heads*head_dim)\n");
    LOG("    layer<N>_wv.npy             V projection weights (hidden, num_kv_heads*head_dim)\n");
    LOG("    layer<N>_wo.npy             O projection weights (num_heads*head_dim, hidden)\n");
    LOG("    layer<N>_ffn_gate.npy       Gate projection weights (hidden, intermediate)\n");
    LOG("    layer<N>_ffn_up.npy         Up projection weights (hidden, intermediate)\n");
    LOG("    layer<N>_ffn_down.npy       Down projection weights (intermediate, hidden)\n");
    LOG("  Activations:\n");
    LOG("    l_inp-<N>.npy               Layer input before RMSNorm (M, hidden)\n");
    LOG("    attn_norm-<N>.npy           Attention input after RMSNorm (M, hidden)\n");
    LOG("    ffn_norm-<N>.npy            MLP input after RMSNorm (M, hidden)\n");
    LOG("    ffn_out-<N>.npy             MLP output (M, hidden)\n");
    LOG("    l_out-<N>.npy               Layer output after second residual (M, hidden)\n");
    LOG("\n");
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt;
    int layer_idx = 0;
    std::string output_dir = "./layer_dump";
    bool weights_only = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (++i >= argc) { LOG_ERR("Missing argument for %s\n", arg.c_str()); return 1; }
            model_path = argv[i];
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i >= argc) { LOG_ERR("Missing argument for %s\n", arg.c_str()); return 1; }
            prompt = argv[i];
        } else if (arg == "--layer") {
            if (++i >= argc) { LOG_ERR("Missing argument for %s\n", arg.c_str()); return 1; }
            layer_idx = std::atoi(argv[i]);
        } else if (arg == "--output-dir") {
            if (++i >= argc) { LOG_ERR("Missing argument for %s\n", arg.c_str()); return 1; }
            output_dir = argv[i];
        } else if (arg == "--weights-only") {
            weights_only = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            LOG_ERR("Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty()) {
        LOG_ERR("Model path is required\n");
        print_usage(argv[0]);
        return 1;
    }

    mkdir(output_dir.c_str(), 0755);

    LOG_INF("dump-layer-data\n");
    LOG_INF("  Model: %s\n", model_path.c_str());
    LOG_INF("  Layer: %d\n", layer_idx);
    LOG_INF("  Output: %s\n", output_dir.c_str());
    if (!prompt.empty()) {
        LOG_INF("  Prompt: %s\n", prompt.c_str());
    }
    LOG_INF("\n");

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU only

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        LOG_ERR("Failed to load model: %s\n", model_path.c_str());
        llama_backend_free();
        return 1;
    }

    // Dump weights
    dump_layer_weights(model, layer_idx, output_dir);

    // Capture activations if prompt provided
    if (!weights_only && !prompt.empty()) {
        LOG_INF("\nCapturing activations...\n");

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 512;
        ctx_params.n_batch = 512;

        dump_config cfg;
        cfg.target_layer = layer_idx;
        cfg.output_dir = output_dir;
        cfg.dump_activations = true;
        cfg.dumped_l_inp = false;
        cfg.dumped_attn_norm = false;
        cfg.dumped_ffn_norm = false;
        cfg.dumped_ffn_out = false;
        cfg.dumped_l_out = false;

        ctx_params.cb_eval = dump_callback;
        ctx_params.cb_eval_user_data = &cfg;

        llama_context * ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) {
            LOG_ERR("Failed to create context\n");
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const bool add_bos = llama_vocab_get_add_bos(vocab);

        std::vector<llama_token> tokens(prompt.size() + 16);
        int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                       tokens.data(), tokens.size(), add_bos, true);
        if (n_tokens < 0) {
            LOG_ERR("Failed to tokenize prompt\n");
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        tokens.resize(n_tokens);

        LOG_INF("Tokenized prompt: %d tokens\n", n_tokens);

        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("Failed to decode\n");
        } else {
            LOG_INF("Decode complete\n");
        }

        if (!cfg.dumped_l_inp) {
            LOG_WRN("Did not capture l_inp for layer %d\n", layer_idx);
        }
        if (!cfg.dumped_attn_norm) {
            LOG_WRN("Did not capture attn_norm for layer %d\n", layer_idx);
        }
        if (!cfg.dumped_ffn_norm) {
            LOG_WRN("Did not capture ffn_norm for layer %d\n", layer_idx);
        }
        if (!cfg.dumped_ffn_out) {
            LOG_WRN("Did not capture ffn_out for layer %d\n", layer_idx);
        }
        if (!cfg.dumped_l_out) {
            LOG_WRN("Did not capture l_out for layer %d\n", layer_idx);
        }

        llama_free(ctx);
    }

    llama_model_free(model);
    llama_backend_free();

    LOG_INF("\nDone!\n");
    return 0;
}
