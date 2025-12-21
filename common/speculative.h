#pragma once

#include "llama.h"
#include "common.h"

#include <vector>
#include <string>

struct common_speculative;

struct common_speculative_params {
    int n_draft = 16;  // max drafted tokens
    int n_reuse = 256;

    float p_min = 0.75f; // min probability required to accept a token in the draft
};

// Log data for a single draft token position
struct common_speculative_draft_token {
    int            pos;      // position in draft sequence
    llama_token    id;       // selected token id
    float          prob;     // probability of selected token
    float          entropy;  // entropy of distribution (top-k)
    std::vector<std::pair<llama_token, float>> top_k; // top-k candidates: (id, prob)
};

// Log data for an entire draft generation round
struct common_speculative_draft_log {
    std::vector<common_speculative_draft_token> tokens;
    std::string stop_reason; // "p_min", "n_max", or "complete"

    // Timing data (microseconds)
    int64_t t_draft_us  = 0;  // time spent generating draft tokens
    int64_t t_verify_us = 0;  // time spent verifying (target model decode)

    void clear() {
        tokens.clear();
        stop_reason.clear();
        t_draft_us  = 0;
        t_verify_us = 0;
    }
};

struct common_speculative * common_speculative_init(
        struct llama_context * ctx_tgt,
        struct llama_context * ctx_dft
);

void common_speculative_free(struct common_speculative * spec);

bool common_speculative_are_compatible(
        const struct llama_context * ctx_tgt,
        const struct llama_context * ctx_dft);

void common_speculative_add_replacement_tgt_dft(
        struct common_speculative * spec,
        const char *source, const char *dest);

// sample up to n_draft tokens and add them to the batch using the draft model
// if draft_log is non-null, detailed draft data will be collected for logging
llama_tokens common_speculative_gen_draft(
               struct common_speculative * spec,
        struct common_speculative_params   params,
                      const llama_tokens & prompt,
                             llama_token   id_last,
       struct common_speculative_draft_log * draft_log = nullptr);
