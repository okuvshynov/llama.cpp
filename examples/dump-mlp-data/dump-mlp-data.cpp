// dump-mlp-data: Dump MLP weights and activations to .npy files for external verification
//
// This tool dequantizes MLP weights from a GGUF model and captures MLP input/output
// activations during inference. Output is saved as NumPy .npy files.
//
// Usage:
//   ./dump-mlp-data -m model.gguf -p "Hello" --layer 0 --output-dir ./mlp_dump
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

    // NPY v1.0 format
    // Magic: \x93NUMPY
    // Version: 1.0
    // Header length: 2 bytes (little-endian)
    // Header: Python dict as ASCII string, padded to 64-byte boundary
    // Data: raw binary

    const char magic[] = "\x93NUMPY";
    fwrite(magic, 1, 6, f);

    uint8_t version[2] = {1, 0};
    fwrite(version, 1, 2, f);

    // Build header dict
    // '<f4' means little-endian 32-bit float
    char header[256];
    if (transpose) {
        // Swap dimensions in header (data will be written transposed)
        snprintf(header, sizeof(header),
                 "{'descr': '<f4', 'fortran_order': False, 'shape': (%lld, %lld), }",
                 (long long)cols, (long long)rows);
    } else {
        snprintf(header, sizeof(header),
                 "{'descr': '<f4', 'fortran_order': False, 'shape': (%lld, %lld), }",
                 (long long)rows, (long long)cols);
    }

    // Pad header to 64-byte boundary (including magic + version + header_len)
    size_t header_len = strlen(header);
    size_t total_header = 10 + header_len + 1;  // 10 = magic(6) + version(2) + len(2), +1 for newline
    size_t padding = (64 - (total_header % 64)) % 64;

    // Pad with spaces
    for (size_t i = 0; i < padding; i++) {
        header[header_len + i] = ' ';
    }
    header[header_len + padding] = '\n';
    header_len += padding + 1;

    uint16_t header_len_u16 = (uint16_t)header_len;
    fwrite(&header_len_u16, 2, 1, f);
    fwrite(header, 1, header_len, f);

    // Write data
    size_t n_elements = rows * cols;
    if (transpose) {
        // Write transposed: iterate output in row-major order
        // output[j, i] = input[i, j]
        // Output has shape (cols, rows), input has shape (rows, cols)
        for (int64_t j = 0; j < cols; j++) {
            for (int64_t i = 0; i < rows; i++) {
                float val = data[i * cols + j];
                fwrite(&val, sizeof(float), 1, f);
            }
        }
    } else {
        fwrite(data, sizeof(float), n_elements, f);
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
        // Use type traits for quantized types
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
    bool dumped_ffn_inp;
    bool dumped_ffn_out;

    // Temporary buffer for GPU->CPU copy
    std::vector<uint8_t> data_buf;
};

// ============================================================================
// Activation Callback
// ============================================================================

static bool dump_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cfg = (dump_config *)user_data;

    if (!cfg->dump_activations) {
        return true;  // Continue but don't capture
    }

    std::string name(t->name);

    // Build target names for the layer we care about
    std::string ffn_inp_name = "ffn_inp-" + std::to_string(cfg->target_layer);
    std::string ffn_out_name = "ffn_out-" + std::to_string(cfg->target_layer);

    bool is_ffn_inp = (name == ffn_inp_name);
    bool is_ffn_out = (name == ffn_out_name);

    if (ask) {
        // Return true only for tensors we want to capture
        return is_ffn_inp || is_ffn_out;
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

    // Determine shape - activations are typically (n_tokens, hidden_size)
    // ggml uses ne[0] for innermost dimension
    int64_t cols = t->ne[0];  // hidden_size
    int64_t rows = t->ne[1];  // n_tokens
    if (t->ne[2] > 1 || t->ne[3] > 1) {
        LOG_WRN("Tensor %s has more than 2 dimensions: [%lld, %lld, %lld, %lld]\n",
                name.c_str(), (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->ne[2], (long long)t->ne[3]);
    }

    // Save to file
    if (is_ffn_inp && !cfg->dumped_ffn_inp) {
        std::string path = cfg->output_dir + "/" + name + ".npy";
        save_npy_2d(path, float_ptr, rows, cols, false);
        cfg->dumped_ffn_inp = true;
    } else if (is_ffn_out && !cfg->dumped_ffn_out) {
        std::string path = cfg->output_dir + "/" + name + ".npy";
        save_npy_2d(path, float_ptr, rows, cols, false);
        cfg->dumped_ffn_out = true;
    }

    return true;
}

// ============================================================================
// Dump MLP Weights
// ============================================================================

static void dump_mlp_weights(const llama_model * model, int layer_idx, const std::string & output_dir) {
    if (layer_idx < 0 || layer_idx >= (int)model->layers.size()) {
        LOG_ERR("Invalid layer index %d (model has %zu layers)\n", layer_idx, model->layers.size());
        return;
    }

    const llama_layer & layer = model->layers[layer_idx];

    // Check if this is a standard FFN layer (not MoE)
    if (!layer.ffn_gate || !layer.ffn_up || !layer.ffn_down) {
        LOG_ERR("Layer %d does not have standard FFN weights (might be MoE or other architecture)\n", layer_idx);
        return;
    }

    LOG_INF("Dumping MLP weights for layer %d\n", layer_idx);
    LOG_INF("  ffn_gate type: %s, shape: [%lld, %lld]\n",
            ggml_type_name(layer.ffn_gate->type),
            (long long)layer.ffn_gate->ne[0], (long long)layer.ffn_gate->ne[1]);
    LOG_INF("  ffn_up type: %s, shape: [%lld, %lld]\n",
            ggml_type_name(layer.ffn_up->type),
            (long long)layer.ffn_up->ne[0], (long long)layer.ffn_up->ne[1]);
    LOG_INF("  ffn_down type: %s, shape: [%lld, %lld]\n",
            ggml_type_name(layer.ffn_down->type),
            (long long)layer.ffn_down->ne[0], (long long)layer.ffn_down->ne[1]);

    // Dequantize and save each weight
    // Note: llama.cpp stores weights as (out_features, in_features)
    // Lyrae expects (in_features, out_features), so we transpose

    // ffn_gate: (intermediate_size, hidden_size) -> transpose to (hidden_size, intermediate_size)
    {
        std::vector<float> gate_data = dequantize_tensor(layer.ffn_gate);
        if (!gate_data.empty()) {
            int64_t rows = layer.ffn_gate->ne[1];  // intermediate_size
            int64_t cols = layer.ffn_gate->ne[0];  // hidden_size
            std::string path = output_dir + "/layer" + std::to_string(layer_idx) + "_ffn_gate.npy";
            save_npy_2d(path, gate_data.data(), rows, cols, true);  // transpose
        }
    }

    // ffn_up: (intermediate_size, hidden_size) -> transpose to (hidden_size, intermediate_size)
    {
        std::vector<float> up_data = dequantize_tensor(layer.ffn_up);
        if (!up_data.empty()) {
            int64_t rows = layer.ffn_up->ne[1];  // intermediate_size
            int64_t cols = layer.ffn_up->ne[0];  // hidden_size
            std::string path = output_dir + "/layer" + std::to_string(layer_idx) + "_ffn_up.npy";
            save_npy_2d(path, up_data.data(), rows, cols, true);  // transpose
        }
    }

    // ffn_down: (hidden_size, intermediate_size) -> transpose to (intermediate_size, hidden_size)
    {
        std::vector<float> down_data = dequantize_tensor(layer.ffn_down);
        if (!down_data.empty()) {
            int64_t rows = layer.ffn_down->ne[1];  // hidden_size
            int64_t cols = layer.ffn_down->ne[0];  // intermediate_size
            std::string path = output_dir + "/layer" + std::to_string(layer_idx) + "_ffn_down.npy";
            save_npy_2d(path, down_data.data(), rows, cols, true);  // transpose
        }
    }

    LOG_INF("Done dumping MLP weights\n");
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
    LOG("  --output-dir DIR     Output directory (default: ./mlp_dump)\n");
    LOG("  --weights-only       Only dump weights, skip activation capture\n");
    LOG("  -h, --help           Show this help\n");
    LOG("\n");
}

int main(int argc, char ** argv) {
    // Parse arguments manually for simplicity
    std::string model_path;
    std::string prompt;
    int layer_idx = 0;
    std::string output_dir = "./mlp_dump";
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

    // Create output directory
    mkdir(output_dir.c_str(), 0755);

    LOG_INF("dump-mlp-data\n");
    LOG_INF("  Model: %s\n", model_path.c_str());
    LOG_INF("  Layer: %d\n", layer_idx);
    LOG_INF("  Output: %s\n", output_dir.c_str());
    if (!prompt.empty()) {
        LOG_INF("  Prompt: %s\n", prompt.c_str());
    }
    LOG_INF("\n");

    // Initialize llama
    llama_backend_init();

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU only

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        LOG_ERR("Failed to load model: %s\n", model_path.c_str());
        llama_backend_free();
        return 1;
    }

    // Dump weights
    dump_mlp_weights(model, layer_idx, output_dir);

    // Capture activations if prompt provided
    if (!weights_only && !prompt.empty()) {
        LOG_INF("\nCapturing activations...\n");

        // Create context with callback
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 512;
        ctx_params.n_batch = 512;

        dump_config cfg;
        cfg.target_layer = layer_idx;
        cfg.output_dir = output_dir;
        cfg.dump_activations = true;
        cfg.dumped_ffn_inp = false;
        cfg.dumped_ffn_out = false;

        ctx_params.cb_eval = dump_callback;
        ctx_params.cb_eval_user_data = &cfg;

        llama_context * ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) {
            LOG_ERR("Failed to create context\n");
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        // Tokenize prompt
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

        // Run inference
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("Failed to decode\n");
        } else {
            LOG_INF("Decode complete\n");
        }

        if (!cfg.dumped_ffn_inp) {
            LOG_WRN("Did not capture ffn_inp for layer %d\n", layer_idx);
        }
        if (!cfg.dumped_ffn_out) {
            LOG_WRN("Did not capture ffn_out for layer %d\n", layer_idx);
        }

        llama_free(ctx);
    }

    llama_model_free(model);
    llama_backend_free();

    LOG_INF("\nDone!\n");
    return 0;
}
