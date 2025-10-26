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
        // Skip reshaped variants (they're duplicates)
        if (name.find("(reshaped)") != std::string::npos) {
            return false;
        }
        return name.find("ffn_moe") != std::string::npos;
    }

    std::string tensor_name(t->name);

    // Extract layer number from tensor name
    // Supports both formats:
    // - "blk.10.ffn_moe_logits" (DeepSeek V3 style)
    // - "ffn_moe_logits-10" (GLM-4.6 style with hyphen suffix)
    int layer = -1;

    // Try blk.X. format first
    if (sscanf(tensor_name.c_str(), "blk.%d.", &layer) != 1) {
        // Try hyphen suffix format
        const char* hyphen = strrchr(tensor_name.c_str(), '-');
        if (hyphen && sscanf(hyphen, "-%d", &layer) == 1) {
            // Successfully parsed layer from hyphen suffix
        } else {
            // Not a layer-specific tensor, skip
            return true;
        }
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
    moe_data.log_file << "Prompt tokens: " << tokens.size() << "\n";
    moe_data.log_file << "Max tokens to generate: " << params.n_predict << "\n\n";

    // Initialize sampler
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.sampling.seed));

    // Print the prompt
    LOG("Prompt: %s\n", params.prompt.c_str());
    for (auto id : tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            LOG("%.*s", n, buf);
        }
    }
    LOG("\n");

    // Process the prompt (prefill)
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

    moe_data.log_file << "--- Prompt Processing (Prefill) ---\n";
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "Failed to decode prompt\n");
        llama_sampler_free(smpl);
        return 1;
    }

    // Main generation loop
    moe_data.log_file << "\n--- Token Generation ---\n";
    int n_decode = 0;
    const int n_predict = params.n_predict;

    for (int i = 0; i < n_predict; i++) {
        // Sample the next token
        llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);

        // Check for end of generation
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            LOG("\n[EOS]\n");
            break;
        }

        // Print the token
        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            LOG("%.*s", n, buf);
        }

        // Decode the token
        batch = llama_batch_get_one(&new_token_id, 1);
        moe_data.log_file << "\n=== Decoding token " << (i + 1) << " ===\n";

        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "Failed to decode token %d\n", i);
            break;
        }

        n_decode++;
    }

    LOG("\n\n");

    moe_data.log_file << "\n=== Generation Complete ===\n";
    moe_data.log_file << "Generated tokens: " << n_decode << "\n";
    moe_data.log_file.close();

    LOG("\nExpert selection logged to: moe_expert_selection.log\n");
    LOG("Generated %d tokens\n\n", n_decode);

    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);

    llama_sampler_free(smpl);
    llama_backend_free();

    return 0;
}
