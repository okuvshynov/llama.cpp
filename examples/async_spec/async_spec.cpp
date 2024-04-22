#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <mutex>

struct linear_speculative_context {
  std::vector<llama_token> speculation;
  std::mutex mut_speculation;
  bool done;
};

static std::vector<llama_token> greedy_tokens(llama_model* model, llama_context* ctx, int from_idx, int to_idx) {
  auto   n_vocab = llama_n_vocab(model);
  std::vector<llama_token_data> candidates;
  candidates.resize(n_vocab);
  std::vector<llama_token> res;

  for (int idx = from_idx; idx < to_idx; idx++) {
    auto * logits  = llama_get_logits_ith(ctx, idx);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        candidates[token_id] = llama_token_data{ token_id, logits[token_id], 0.0f };
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

    // sample the most likely token
    const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);
    res.push_back(new_token_id);
  }
  return res;
} 

// what we do next
// - try llama3 8B & 70B
// - measure how they work on GPU/CPU at different quant level
// - for example, if we were to run 8b 70 on top of 4b 8? Can we make it work?
static int main_loop(llama_model* model, linear_speculative_context* spec_ctx, gpt_params params) {
    const int n_len = 256;
    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.seed  = 1234;
    ctx_params.n_ctx = 2048;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize(ctx, params.prompt, true);

    const int n_ctx    = llama_n_ctx(ctx);
    const int n_kv_req = tokens_list.size() + (n_len - tokens_list.size());

    LOG_TEE("\n%s: n_len = %d, n_ctx = %d, n_kv_req = %d\n", __func__, n_len, n_ctx, n_kv_req);

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx) {
        LOG_TEE("%s: error: n_kv_req > n_ctx, the required KV cache size is not big enough\n", __func__);
        LOG_TEE("%s:        either reduce n_len or increase n_ctx\n", __func__);
        return 1;
    }

    llama_batch batch = llama_batch_init(512, 0, 1);

    // evaluate the initial prompt
    for (size_t i = 0; i < tokens_list.size(); i++) {
        llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        LOG_TEE("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    // how many tokens are currently accepted
    int n_cur  = batch.n_tokens;

    std::vector<llama_token> input_seq;
    input_seq.push_back(tokens_list.back());

    int logits_from = n_cur - 1;
    int logits_to = n_cur;

    while (n_cur <= n_len) {
        auto next_tokens = greedy_tokens(model, ctx, logits_from, logits_to);
        if (next_tokens.size() != input_seq.size()) {
          fprintf(stderr, "invalid next tokens\n");
          return 1;
        }

        // this is where next_tokens start
        int next_tokens_pos = n_cur;
        // we always accept at least one new token
        n_cur += 1;
        for (size_t i = 0; i + 1 < input_seq.size(); i++) {
          if (next_tokens[i] == input_seq[i + 1]) {
            n_cur += 1;
          } else {
            // reject. next_tokens[i] is the last 'correct' one.
            // removing all rejected next tokens
            next_tokens.erase(next_tokens.begin() + i + 1, next_tokens.end());
            break;
          }
        }
        //printf("%zu %zu\n", input_seq.size(), next_tokens.size());

        llama_kv_cache_seq_rm(ctx, 0, n_cur - 1, -1);

        bool done = false;
        for (llama_token new_token_id: next_tokens) {
          LOG_TEE("%s", llama_token_to_piece(ctx, new_token_id).c_str());
          fflush(stdout);
          if (new_token_id == llama_token_eos(model)) {
              done = true;
              break;
          }
        }
        if (n_cur >= n_len || done) {
          break;
        }

        {
          std::lock_guard<std::mutex> _lock(spec_ctx->mut_speculation);
          // here we need to match the currently available speculation and what we have accepted
          // speculation is input_seq + whatever draft process appended to it.
          auto& spec = spec_ctx->speculation;
          size_t n_match = 0;
          for (size_t i = 0; i < next_tokens.size() && i + next_tokens_pos < spec.size(); i++) {
            if (next_tokens[i] == spec[i + next_tokens_pos]) {
              n_match++;
            } else {
              break;
            }
          }
          if (n_match != next_tokens.size()) {
            // need to modify speculation
            spec.erase(spec.begin() + next_tokens_pos, spec.end());
            for (const auto tok: next_tokens) {
              spec.push_back(tok);
            }
          }
          input_seq.assign(spec.begin() + n_cur - 1, spec.end());
        }

        llama_batch_clear(batch);

        for (size_t i = 0; i < input_seq.size(); i++) {
          llama_batch_add(batch, input_seq[i], n_cur - 1 + i, { 0 }, true);
        }
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
        logits_from = 0;
        logits_to = input_seq.size();
    }

    {
      std::lock_guard<std::mutex> _lock(spec_ctx->mut_speculation);
      spec_ctx->done = true;
    }

    llama_print_timings(ctx);
    llama_batch_free(batch);
    llama_free(ctx);
    return 0;
}

static int draft_loop(llama_model* model, linear_speculative_context* spec_ctx, gpt_params params) {
    const int n_len = 256;
    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.seed  = 1234;
    ctx_params.n_ctx = 2048;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize(ctx, params.prompt, true);

    // TODO: this needs to be done in critical section/outside?
    spec_ctx->speculation = tokens_list;
    spec_ctx->done = false;


    const int n_ctx    = llama_n_ctx(ctx);
    const int n_kv_req = tokens_list.size() + (n_len - tokens_list.size());

    LOG_TEE("\n%s: n_len = %d, n_ctx = %d, n_kv_req = %d\n", __func__, n_len, n_ctx, n_kv_req);

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx) {
        LOG_TEE("%s: error: n_kv_req > n_ctx, the required KV cache size is not big enough\n", __func__);
        LOG_TEE("%s:        either reduce n_len or increase n_ctx\n", __func__);
        return 1;
    }

    llama_batch batch = llama_batch_init(512, 0, 1);

    // evaluate the initial prompt
    for (size_t i = 0; i < tokens_list.size(); i++) {
        llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        LOG_TEE("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    // main loop
    int n_cur    = batch.n_tokens;
    int logit_idx = batch.n_tokens - 1;
    std::vector<llama_token> local_spec = tokens_list;
    size_t match_len;

    return 0;
    while (true) {
        // sample the next token
        auto next_tokens = greedy_tokens(model, ctx, logit_idx, logit_idx + 1);
        if (next_tokens.size() != 1) {
          fprintf(stderr, "invalid next tokens\n");
          return 1;
        }
        // new_token_id will be written to position n_cur

        local_spec.push_back(next_tokens[0]);
        //LOG_TEE("draft: %s\n", llama_token_to_piece(ctx, next_tokens[0]).c_str());
        //fflush(stdout);

        {
          std::lock_guard<std::mutex> _lock(spec_ctx->mut_speculation);
          if (spec_ctx->done) {
            break;
          } 
          auto& spec = spec_ctx->speculation;
          // if all global is contained in local, we reuse the whole local
          // if either there's a mismatch or global is longer we reuse all global while still 
          // requesting logits from the last one only
          bool match = true;
          match_len = local_spec.size() - 1;
          for (size_t i = 0; i < std::min(spec.size(), local_spec.size()); i++) {
            if (spec[i] != local_spec[i]) {
              match = false;
              match_len = i;
              llama_kv_cache_seq_rm(ctx, 0, i, -1);
              break;
            }
          }
          if (match) {
            spec = local_spec;
          } else {
            local_spec = spec;
          }
        }

        llama_batch_clear(batch);
        //printf("IN LOCAL: %zu %ld\n", match_len, local_spec.size());

        for (size_t i = match_len; i < local_spec.size(); i++) {
          llama_batch_add(batch, local_spec[i], i, { 0 }, true);
        }

        logit_idx = batch.n_tokens - 1;
        n_cur += 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
    }

    llama_print_timings(ctx);
    llama_batch_free(batch);
    llama_free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    gpt_params params;

    if (argc >= 2) {
        params.model = argv[1];
    }

    if (params.prompt.empty()) {
        params.prompt = "Here's a list of main characters in Pulp Fiction movie:";
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    llama_model * gpu_model = llama_load_model_from_file(params.model.c_str(), model_params);

    if (argc >= 3) {
        params.model = argv[2];
    }

    model_params.n_gpu_layers = 0;
    llama_model * cpu_model = llama_load_model_from_file(params.model.c_str(), model_params);

    linear_speculative_context spec_ctx;

    //main_loop(gpu_model, &spec_ctx, params);
    //draft_loop(gpu_model, &spec_ctx, params);

    const auto t_main_start = ggml_time_us();
    std::thread t_cpu(draft_loop, cpu_model, &spec_ctx, params);
    std::thread t_gpu(main_loop, gpu_model, &spec_ctx, params);
    t_gpu.join();
    t_cpu.join();
    const auto t_main_end = ggml_time_us();

    printf("Total time: %.3lf\n", (t_main_end - t_main_start) / 1000000.0);

    llama_free_model(gpu_model);
    llama_free_model(cpu_model);
    llama_backend_free();

    return 0;
}
