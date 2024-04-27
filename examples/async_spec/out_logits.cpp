#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// greedy sampling
static llama_token greedy_token(llama_model *model, llama_context *ctx, int idx) {
  auto n_vocab = llama_n_vocab(model);
  std::vector<llama_token_data> candidates;
  candidates.resize(n_vocab);

  auto *logits  = llama_get_logits_ith(ctx, idx);
  for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
    std::cout << logits[token_id] << " ";
    candidates[token_id] = llama_token_data{ token_id, logits[token_id], 0.0f };
  }

  llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

  // sample the most likely token
  return llama_sample_token_greedy(ctx, &candidates_p);
}

static int main_loop(
    llama_model *model,
    llama_context *ctx,
    std::vector<llama_token> tokens_list /* copy here */) {
  const int n_len = 1024;

  llama_batch batch = llama_batch_init(1024, 0, 1);

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

  while (n_cur <= n_len) {
    llama_token new_token_id = greedy_token(model, ctx, batch.n_tokens - 1);
    // this is where next_tokens start
    if (new_token_id == llama_token_eos(model)) {
      break;
    }
    if (n_cur >= n_len) {
      break;
    }
    std::cout << llama_token_to_piece(ctx, new_token_id) << std::flush;

    llama_batch_clear(batch);
    llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);
    if (llama_decode(ctx, batch)) {
      fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
      return 1;
    }
    n_cur += 1;
  }

  llama_batch_free(batch);
  return 0;
}

int main(int argc, char ** argv) {
    gpt_params params;

    llama_backend_init();
    llama_numa_init(params.numa);

    // init context params
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.seed  = 1234;
    ctx_params.n_ctx = 2048;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;

    // Init main model and context
    if (argc >= 2) {
        params.model = argv[1];
    }
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    llama_model *main_model = llama_load_model_from_file(params.model.c_str(), model_params);

    llama_context *main_ctx = llama_new_context_with_model(main_model, ctx_params);
    std::ifstream t(argv[2]);
    std::stringstream buffer;
    buffer << t.rdbuf();
    params.prompt = buffer.str();

    if (params.prompt.empty()) {
        params.prompt = "What's the difference between instruction cache and data cache?";
    }
    std::cout << params.prompt << std::flush;
    std::vector<llama_token> tokens_list = llama_tokenize(main_ctx, params.prompt, true);

    main_loop(main_model, main_ctx, tokens_list);

    llama_free_model(main_model);
    llama_free(main_ctx);
    llama_backend_free();

    return 0;
}
