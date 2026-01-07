#pragma once

#include "../llama-model.h"
#include "../llama-graph.h"

struct llm_build_mistral3 : public llm_graph_context {
    llm_build_mistral3(const llama_model & model, const llm_graph_params & params);
};
