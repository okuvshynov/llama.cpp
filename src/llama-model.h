#pragma once

#include "llama.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-hparams.h"
#include "llama-memory.h"
#include "llama-vocab.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_ubatch;
struct llama_model_loader;

// Minimal model types for Devstral models
enum llm_type {
    LLM_TYPE_UNKNOWN,
    LLM_TYPE_3B,    // 26 layers (mistral3)
    LLM_TYPE_8B,    // 34 layers (mistral3)
    LLM_TYPE_14B,   // 40 layers (mistral3)
    LLM_TYPE_123B,  // llama architecture (Devstral-2-123B)
};

std::string llama_rope_scaling_type_name(llama_rope_scaling_type rope_scaling_type);

// NOTE: llama_layer_posnet, llama_layer_convnext, llama_layer_shortconv, llama_layer_nextn
//       removed for minimal build (only MISTRAL3 architecture supported)

// Minimal llama_layer for MISTRAL3 architecture only
struct llama_layer {
    // normalization
    struct ggml_tensor * attn_norm = nullptr;

    // attention weights
    struct ggml_tensor * wq = nullptr;
    struct ggml_tensor * wk = nullptr;
    struct ggml_tensor * wv = nullptr;
    struct ggml_tensor * wo = nullptr;

    // attention bias
    struct ggml_tensor * bq = nullptr;
    struct ggml_tensor * bk = nullptr;
    struct ggml_tensor * bv = nullptr;
    struct ggml_tensor * bo = nullptr;

    // ffn normalization
    struct ggml_tensor * ffn_norm = nullptr;

    // ff weights (dense layers)
    struct ggml_tensor * ffn_gate = nullptr; // w1
    struct ggml_tensor * ffn_down = nullptr; // w2
    struct ggml_tensor * ffn_up   = nullptr; // w3

    // ff bias
    struct ggml_tensor * ffn_gate_b = nullptr;
    struct ggml_tensor * ffn_down_b = nullptr;
    struct ggml_tensor * ffn_up_b   = nullptr;

    // ff MoE
    struct ggml_tensor * ffn_gate_inp  = nullptr;
    struct ggml_tensor * ffn_gate_exps = nullptr;
    struct ggml_tensor * ffn_down_exps = nullptr;
    struct ggml_tensor * ffn_up_exps   = nullptr;

    // rope factors
    struct ggml_tensor * rope_long  = nullptr;
    struct ggml_tensor * rope_short = nullptr;
    struct ggml_tensor * rope_freqs = nullptr;
};

struct llama_model {
    llm_type type = LLM_TYPE_UNKNOWN;
    llm_arch arch = LLM_ARCH_UNKNOWN;

    std::string name = "n/a";

    llama_hparams hparams = {};
    llama_vocab   vocab;

    // for classifier models
    std::vector<std::string> classifier_labels;

    struct ggml_tensor * tok_embd   = nullptr;
    struct ggml_tensor * type_embd  = nullptr;
    struct ggml_tensor * pos_embd   = nullptr;
    struct ggml_tensor * tok_norm   = nullptr;
    struct ggml_tensor * tok_norm_b = nullptr;

    struct ggml_tensor * output_norm     = nullptr;
    struct ggml_tensor * output_norm_b   = nullptr;
    struct ggml_tensor * output          = nullptr;
    struct ggml_tensor * output_b        = nullptr;
    struct ggml_tensor * output_norm_enc = nullptr;

    // classifier
    struct ggml_tensor * cls       = nullptr;
    struct ggml_tensor * cls_b     = nullptr;
    struct ggml_tensor * cls_out   = nullptr;
    struct ggml_tensor * cls_out_b = nullptr;

    struct ggml_tensor * conv1d   = nullptr;
    struct ggml_tensor * conv1d_b = nullptr;

    // gemma3n altup
    struct ggml_tensor * tok_embd_per_layer   = nullptr;
    struct ggml_tensor * altup_proj           = nullptr;
    struct ggml_tensor * altup_unembd_proj    = nullptr;
    struct ggml_tensor * per_layer_model_proj = nullptr;
    struct ggml_tensor * per_layer_proj_norm  = nullptr;

    std::vector<llama_layer> layers;

    //Dense linear projections for SentenceTransformers models like embeddinggemma
    // For Sentence Transformers models structure see
    // https://sbert.net/docs/sentence_transformer/usage/custom_models.html#structure-of-sentence-transformer-models
    struct ggml_tensor * dense_2_out_layers = nullptr;
    struct ggml_tensor * dense_3_out_layers = nullptr;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // list of devices used in this model
    std::vector<ggml_backend_dev_t> devices;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;

    // for keeping track of extra nodes used by lora adapters
    uint32_t n_lora_nodes = 0;

    int64_t t_load_us  = 0;
    int64_t t_start_us = 0;

    explicit llama_model(const struct llama_model_params & params);
    ~llama_model();

    void load_stats  (llama_model_loader & ml);
    void load_arch   (llama_model_loader & ml);
    void load_hparams(llama_model_loader & ml);
    void load_vocab  (llama_model_loader & ml);
    bool load_tensors(llama_model_loader & ml); // returns false if cancelled by progress_callback

    std::string arch_name() const;
    std::string type_name() const;

    std::string desc() const;

    size_t size() const; // file size
    size_t n_tensors() const;
    size_t n_devices() const;

    uint32_t n_gpu_layers() const;
    llama_split_mode split_mode() const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // total number of parameters in the model
    uint64_t n_elements() const;

    void print_info() const;

    ggml_backend_dev_t dev_layer(int il) const;
    ggml_backend_dev_t dev_output() const;

    ggml_backend_buffer_type_t select_buft(int il) const;

    bool has_tensor_overrides() const;

    const struct ggml_tensor * get_tensor(const char * name) const;

    float get_rope_freq_base (const llama_cparams & cparams, int il) const;
    float get_rope_freq_scale(const llama_cparams & cparams, int il) const;

    ggml_tensor * get_rope_factors(const llama_cparams & cparams, int il) const;

    // TODO: move this to new llm_arch_model_i interface
    llama_memory_i * create_memory(const llama_memory_params & params, const llama_cparams & cparams) const;

    // TODO: move this to new llm_arch_model_i interface
    ggml_cgraph * build_graph(const llm_graph_params & params) const;

private:
    llama_model_params params;

    struct impl;
    std::unique_ptr<impl> pimpl;
};

const char * llm_type_name(llm_type type);

// For internal test use
// TODO: remove
const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model);
