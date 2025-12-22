// Benchmark for measuring verification cost as a function of batch size
// This simulates the verification phase of speculative decoding where the
// target model evaluates N draft tokens at once.

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "common.h"
#include "ggml.h"
#include "llama.h"

// utils

static uint64_t get_time_ns() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::nanoseconds(clock::now().time_since_epoch()).count();
}

template <class T>
static std::string join(const std::vector<T> & values, const std::string & delim) {
    std::ostringstream str;
    for (size_t i = 0; i < values.size(); i++) {
        str << values[i];
        if (i < values.size() - 1) {
            str << delim;
        }
    }
    return str.str();
}

template <typename T>
static T avg(const std::vector<T> & v) {
    if (v.empty()) {
        return 0;
    }
    T sum = std::accumulate(v.begin(), v.end(), T(0));
    return sum / (T) v.size();
}

template <typename T>
static T stdev(const std::vector<T> & v) {
    if (v.size() <= 1) {
        return 0;
    }
    T mean   = avg(v);
    T sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), T(0));
    T stdev  = std::sqrt(sq_sum / (T) (v.size() - 1) - mean * mean * (T) v.size() / (T) (v.size() - 1));
    return stdev;
}

static std::string get_cpu_info() {
    std::vector<std::string> cpu_list;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto * dev      = ggml_backend_dev_get(i);
        auto   dev_type = ggml_backend_dev_type(dev);
        if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU || dev_type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            cpu_list.push_back(ggml_backend_dev_description(dev));
        }
    }
    return join(cpu_list, ", ");
}

static std::string get_gpu_info() {
    std::vector<std::string> gpu_list;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto * dev      = ggml_backend_dev_get(i);
        auto   dev_type = ggml_backend_dev_type(dev);
        if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            gpu_list.push_back(ggml_backend_dev_description(dev));
        }
    }
    return join(gpu_list, ", ");
}

// output formats
enum output_formats { NONE, CSV, JSON, JSONL, MARKDOWN };

static const char * output_format_str(output_formats format) {
    switch (format) {
        case NONE:     return "none";
        case CSV:      return "csv";
        case JSON:     return "json";
        case JSONL:    return "jsonl";
        case MARKDOWN: return "md";
        default:       GGML_ABORT("invalid output format");
    }
}

static bool output_format_from_str(const std::string & s, output_formats & format) {
    if (s == "none") {
        format = NONE;
    } else if (s == "csv") {
        format = CSV;
    } else if (s == "json") {
        format = JSON;
    } else if (s == "jsonl") {
        format = JSONL;
    } else if (s == "md") {
        format = MARKDOWN;
    } else {
        return false;
    }
    return true;
}

// parse range like "1-64" or "1-64+1" or "1-64*2"
static std::vector<int> parse_int_range(const std::string & s) {
    std::regex range_regex(R"(^(\d+)(?:-(\d+)(?:([\+|\*])(\d+))?)?(?:,|$))");

    std::smatch match;
    std::string::const_iterator search_start(s.cbegin());
    std::vector<int> result;

    while (std::regex_search(search_start, s.cend(), match, range_regex)) {
        int  first = std::stoi(match[1]);
        int  last  = match[2].matched ? std::stoi(match[2]) : first;
        char op    = match[3].matched ? match[3].str()[0] : '+';
        int  step  = match[4].matched ? std::stoi(match[4]) : 1;

        for (int i = first; i <= last;) {
            result.push_back(i);

            int prev_i = i;

            if (op == '+') {
                i += step;
            } else if (op == '*') {
                i *= step;
            } else {
                throw std::invalid_argument("invalid range format");
            }

            if (i <= prev_i) {
                throw std::invalid_argument("invalid range");
            }
        }
        search_start = match.suffix().first;
    }

    if (search_start != s.cend()) {
        throw std::invalid_argument("invalid range format");
    }

    return result;
}

// command line params
struct cmd_params {
    std::string      model;
    std::vector<int> n_depth;     // context depth before verification
    std::vector<int> n_verify;    // batch sizes to test
    int              n_batch;
    int              n_ubatch;
    int              n_threads;
    int              n_gpu_layers;
    bool             flash_attn;
    int              reps;
    bool             verbose;
    bool             progress;
    output_formats   output_format;
};

static const cmd_params cmd_params_defaults = {
    /* model         */ "",
    /* n_depth       */ { 512 },
    /* n_verify      */ { 1, 2, 4, 8, 16, 32, 64 },
    /* n_batch       */ 2048,
    /* n_ubatch      */ 512,
    /* n_threads     */ -1,
    /* n_gpu_layers  */ 99,
    /* flash_attn    */ false,
    /* reps          */ 10,
    /* verbose       */ false,
    /* progress      */ false,
    /* output_format */ MARKDOWN,
};

static void print_usage(int /* argc */, char ** argv) {
    printf("usage: %s [options]\n", argv[0]);
    printf("\n");
    printf("Benchmark verification cost as a function of batch size.\n");
    printf("Simulates the target model verification phase in speculative decoding.\n");
    printf("\n");
    printf("options:\n");
    printf("  -h, --help                        show this help message and exit\n");
    printf("  -m, --model <filename>            model path (required)\n");
    printf("  -d, --n-depth <n>                 context depth before verification (default: %s)\n",
           join(cmd_params_defaults.n_depth, ",").c_str());
    printf("  -nv, --n-verify <n>               batch sizes to test (default: %s)\n",
           join(cmd_params_defaults.n_verify, ",").c_str());
    printf("  -b, --batch-size <n>              batch size for prompt processing (default: %d)\n",
           cmd_params_defaults.n_batch);
    printf("  -ub, --ubatch-size <n>            ubatch size (default: %d)\n",
           cmd_params_defaults.n_ubatch);
    printf("  -t, --threads <n>                 number of threads (default: auto)\n");
    printf("  -ngl, --n-gpu-layers <n>          number of GPU layers (default: %d)\n",
           cmd_params_defaults.n_gpu_layers);
    printf("  -fa, --flash-attn <0|1>           enable flash attention (default: %d)\n",
           cmd_params_defaults.flash_attn);
    printf("  -r, --repetitions <n>             number of repetitions (default: %d)\n",
           cmd_params_defaults.reps);
    printf("  -o, --output <csv|json|jsonl|md>  output format (default: %s)\n",
           output_format_str(cmd_params_defaults.output_format));
    printf("  -v, --verbose                     verbose output\n");
    printf("  --progress                        show progress\n");
    printf("\n");
    printf("Ranges can be specified as 'first-last' or 'first-last+step' or 'first-last*mult'\n");
    printf("\n");
    printf("Example:\n");
    printf("  %s -m model.gguf -d 256,512 -nv 1-64+1 -r 5\n", argv[0]);
}

static cmd_params parse_cmd_params(int argc, char ** argv) {
    cmd_params params = cmd_params_defaults;
    std::string arg;
    bool invalid_param = false;

    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        try {
            if (arg == "-h" || arg == "--help") {
                print_usage(argc, argv);
                exit(0);
            } else if (arg == "-m" || arg == "--model") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.model = argv[i];
            } else if (arg == "-d" || arg == "--n-depth") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.n_depth = parse_int_range(argv[i]);
            } else if (arg == "-nv" || arg == "--n-verify") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.n_verify = parse_int_range(argv[i]);
            } else if (arg == "-b" || arg == "--batch-size") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.n_batch = std::stoi(argv[i]);
            } else if (arg == "-ub" || arg == "--ubatch-size") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.n_ubatch = std::stoi(argv[i]);
            } else if (arg == "-t" || arg == "--threads") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.n_threads = std::stoi(argv[i]);
            } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.n_gpu_layers = std::stoi(argv[i]);
            } else if (arg == "-fa" || arg == "--flash-attn") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.flash_attn = std::stoi(argv[i]) != 0;
            } else if (arg == "-r" || arg == "--repetitions") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                params.reps = std::stoi(argv[i]);
            } else if (arg == "-o" || arg == "--output") {
                if (++i >= argc) {
                    invalid_param = true;
                    break;
                }
                if (!output_format_from_str(argv[i], params.output_format)) {
                    invalid_param = true;
                    break;
                }
            } else if (arg == "-v" || arg == "--verbose") {
                params.verbose = true;
            } else if (arg == "--progress") {
                params.progress = true;
            } else {
                invalid_param = true;
                break;
            }
        } catch (const std::exception & e) {
            fprintf(stderr, "error: %s\n", e.what());
            invalid_param = true;
            break;
        }
    }

    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        print_usage(argc, argv);
        exit(1);
    }

    if (params.model.empty()) {
        fprintf(stderr, "error: model path is required\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (params.n_threads < 0) {
        params.n_threads = cpu_get_num_math();
    }

    return params;
}

// test result for a single (depth, verify_size) combination
struct test_result {
    int                   n_depth;
    int                   n_verify;
    std::vector<uint64_t> samples_ns;

    uint64_t avg_ns() const { return ::avg(samples_ns); }
    uint64_t stdev_ns() const { return ::stdev(samples_ns); }
    double   avg_ms() const { return avg_ns() / 1e6; }
    double   stdev_ms() const { return stdev_ns() / 1e6; }

    // tokens per second
    double avg_ts() const {
        if (avg_ns() == 0) return 0;
        return 1e9 * n_verify / avg_ns();
    }
};

// printers

static std::string escape_csv(const std::string & field) {
    std::string escaped = "\"";
    for (auto c : field) {
        if (c == '"') {
            escaped += "\"";
        }
        escaped += c;
    }
    escaped += "\"";
    return escaped;
}

static std::string escape_json(const std::string & value) {
    std::string escaped;
    for (auto c : value) {
        if (c == '"') {
            escaped += "\\\"";
        } else if (c == '\\') {
            escaped += "\\\\";
        } else if (c <= 0x1f) {
            char buf[8];
            snprintf(buf, sizeof(buf), "\\u%04x", c);
            escaped += buf;
        } else {
            escaped += c;
        }
    }
    return escaped;
}

struct printer {
    FILE * fout = stdout;

    virtual ~printer() = default;
    virtual void print_header(const cmd_params & params, const std::string & model_desc) = 0;
    virtual void print_result(const test_result & r) = 0;
    virtual void print_footer() {}
};

struct csv_printer : public printer {
    void print_header(const cmd_params & params, const std::string & model_desc) override {
        fprintf(fout, "n_depth,n_verify,avg_ms,stdev_ms,avg_ts,samples_ns\n");
        (void) params;
        (void) model_desc;
    }

    void print_result(const test_result & r) override {
        fprintf(fout, "%d,%d,%.3f,%.3f,%.2f,\"%s\"\n",
                r.n_depth, r.n_verify, r.avg_ms(), r.stdev_ms(), r.avg_ts(),
                join(r.samples_ns, ";").c_str());
    }
};

struct json_printer : public printer {
    bool first = true;

    void print_header(const cmd_params & params, const std::string & model_desc) override {
        fprintf(fout, "{\n");
        fprintf(fout, "  \"model\": \"%s\",\n", escape_json(params.model).c_str());
        fprintf(fout, "  \"model_desc\": \"%s\",\n", escape_json(model_desc).c_str());
        fprintf(fout, "  \"n_gpu_layers\": %d,\n", params.n_gpu_layers);
        fprintf(fout, "  \"flash_attn\": %s,\n", params.flash_attn ? "true" : "false");
        fprintf(fout, "  \"n_threads\": %d,\n", params.n_threads);
        fprintf(fout, "  \"results\": [\n");
    }

    void print_result(const test_result & r) override {
        if (!first) {
            fprintf(fout, ",\n");
        }
        first = false;
        fprintf(fout, "    {\"n_depth\": %d, \"n_verify\": %d, \"avg_ms\": %.3f, \"stdev_ms\": %.3f, \"avg_ts\": %.2f, \"samples_ns\": [%s]}",
                r.n_depth, r.n_verify, r.avg_ms(), r.stdev_ms(), r.avg_ts(),
                join(r.samples_ns, ", ").c_str());
    }

    void print_footer() override {
        fprintf(fout, "\n  ]\n}\n");
    }
};

struct jsonl_printer : public printer {
    void print_header(const cmd_params & params, const std::string & model_desc) override {
        (void) params;
        (void) model_desc;
    }

    void print_result(const test_result & r) override {
        fprintf(fout, "{\"n_depth\": %d, \"n_verify\": %d, \"avg_ms\": %.3f, \"stdev_ms\": %.3f, \"avg_ts\": %.2f, \"samples_ns\": [%s]}\n",
                r.n_depth, r.n_verify, r.avg_ms(), r.stdev_ms(), r.avg_ts(),
                join(r.samples_ns, ", ").c_str());
    }
};

struct markdown_printer : public printer {
    void print_header(const cmd_params & params, const std::string & model_desc) override {
        fprintf(fout, "# Verification Cost Benchmark\n\n");
        fprintf(fout, "Model: %s\n", params.model.c_str());
        fprintf(fout, "Description: %s\n", model_desc.c_str());
        fprintf(fout, "GPU layers: %d, Flash attention: %s, Threads: %d\n\n",
                params.n_gpu_layers, params.flash_attn ? "yes" : "no", params.n_threads);
        fprintf(fout, "| %6s | %8s | %10s | %10s | %10s |\n",
                "depth", "n_verify", "avg_ms", "stdev_ms", "t/s");
        fprintf(fout, "|%7s:|%9s:|%11s:|%11s:|%11s:|\n",
                "------", "--------", "----------", "----------", "----------");
    }

    void print_result(const test_result & r) override {
        fprintf(fout, "| %6d | %8d | %10.3f | %10.3f | %10.2f |\n",
                r.n_depth, r.n_verify, r.avg_ms(), r.stdev_ms(), r.avg_ts());
    }

    void print_footer() override {
        fprintf(fout, "\n");
    }
};

static std::unique_ptr<printer> create_printer(output_formats format) {
    switch (format) {
        case NONE:     return nullptr;
        case CSV:      return std::make_unique<csv_printer>();
        case JSON:     return std::make_unique<json_printer>();
        case JSONL:    return std::make_unique<jsonl_printer>();
        case MARKDOWN: return std::make_unique<markdown_printer>();
    }
    GGML_ABORT("invalid output format");
}

static void llama_null_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

// fill context with random tokens to reach depth
static bool fill_context(llama_context * ctx, int n_depth, int n_batch, int n_threads) {
    llama_set_n_threads(ctx, n_threads, n_threads);

    const llama_model * model   = llama_get_model(ctx);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> tokens(n_batch);

    int n_processed = 0;

    while (n_processed < n_depth) {
        int n_tokens = std::min(n_depth - n_processed, n_batch);
        tokens[0]    = n_processed == 0 && llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : std::rand() % n_vocab;
        for (int i = 1; i < n_tokens; i++) {
            tokens[i] = std::rand() % n_vocab;
        }
        int res = llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tokens));
        if (res != 0) {
            fprintf(stderr, "%s: failed to fill context, res = %d\n", __func__, res);
            return false;
        }
        n_processed += n_tokens;
    }

    llama_synchronize(ctx);
    return true;
}

// run a single verification test: decode n_verify tokens at once
static uint64_t test_verify(llama_context * ctx, int n_depth, int n_verify, int n_threads) {
    llama_set_n_threads(ctx, n_threads, n_threads);

    const llama_model * model   = llama_get_model(ctx);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    // create batch of n_verify random tokens starting at position n_depth
    std::vector<llama_token> tokens(n_verify);
    for (int i = 0; i < n_verify; i++) {
        tokens[i] = std::rand() % n_vocab;
    }

    // build batch with logits requested for all positions (like in speculation verification)
    llama_batch batch = llama_batch_init(n_verify, 0, 1);
    for (int i = 0; i < n_verify; i++) {
        common_batch_add(batch, tokens[i], n_depth + i, {0}, true);
    }

    uint64_t t_start = get_time_ns();

    int res = llama_decode(ctx, batch);
    if (res != 0) {
        fprintf(stderr, "%s: failed to decode verification batch, res = %d\n", __func__, res);
        llama_batch_free(batch);
        return 0;
    }

    llama_synchronize(ctx);

    uint64_t t_end = get_time_ns();

    // remove the tokens we just added to restore context to n_depth
    llama_memory_seq_rm(llama_get_memory(ctx), 0, n_depth, -1);

    llama_batch_free(batch);

    return t_end - t_start;
}

int main(int argc, char ** argv) {
    setlocale(LC_CTYPE, ".UTF-8");

#if !defined(NDEBUG)
    fprintf(stderr, "warning: asserts enabled, performance may be affected\n");
#endif

#if (defined(_MSC_VER) && defined(_DEBUG)) || (!defined(_MSC_VER) && !defined(__OPTIMIZE__))
    fprintf(stderr, "warning: debug build, performance may be affected\n");
#endif

    ggml_backend_load_all();

    cmd_params params = parse_cmd_params(argc, argv);

    if (!params.verbose) {
        llama_log_set(llama_null_log_callback, NULL);
    }

    llama_backend_init();

    // load model
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = params.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(params.model.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "error: failed to load model: %s\n", params.model.c_str());
        return 1;
    }

    char model_desc[128];
    llama_model_desc(model, model_desc, sizeof(model_desc));

    // find max depth and max verify size to determine context size
    int max_depth  = *std::max_element(params.n_depth.begin(), params.n_depth.end());
    int max_verify = *std::max_element(params.n_verify.begin(), params.n_verify.end());

    // create context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = max_depth + max_verify + 64;  // some buffer
    cparams.n_batch         = params.n_batch;
    cparams.n_ubatch        = params.n_ubatch;
    cparams.flash_attn_type = params.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // setup printer
    auto p = create_printer(params.output_format);
    if (p) {
        p->fout = stdout;
        p->print_header(params, model_desc);
    }

    if (params.progress) {
        fprintf(stderr, "CPU: %s\n", get_cpu_info().c_str());
        fprintf(stderr, "GPU: %s\n", get_gpu_info().c_str());
        fprintf(stderr, "Model: %s\n", model_desc);
        fprintf(stderr, "Context size: %d\n", cparams.n_ctx);
    }

    // sort depths so we can potentially reuse context state
    std::vector<int> depths = params.n_depth;
    std::sort(depths.begin(), depths.end());

    int current_depth = 0;

    for (int depth : depths) {
        // fill context up to this depth (incrementally if possible)
        if (depth > current_depth) {
            if (params.progress) {
                fprintf(stderr, "Filling context to depth %d...\n", depth);
            }

            // if we need to go deeper, we need to add more tokens
            // but first, check if we need to start from scratch
            if (current_depth == 0) {
                llama_memory_clear(llama_get_memory(ctx), false);
            }

            if (!fill_context(ctx, depth - current_depth, params.n_batch, params.n_threads)) {
                fprintf(stderr, "error: failed to fill context to depth %d\n", depth);
                break;
            }
            current_depth = depth;
        }

        // test each verify size
        for (int n_verify : params.n_verify) {
            if (params.progress) {
                fprintf(stderr, "Testing depth=%d, n_verify=%d...\n", depth, n_verify);
            }

            test_result result;
            result.n_depth  = depth;
            result.n_verify = n_verify;

            // warmup
            test_verify(ctx, depth, n_verify, params.n_threads);

            // actual measurements
            for (int rep = 0; rep < params.reps; rep++) {
                uint64_t t = test_verify(ctx, depth, n_verify, params.n_threads);
                if (t == 0) {
                    fprintf(stderr, "error: verification test failed\n");
                    break;
                }
                result.samples_ns.push_back(t);
            }

            if (p && !result.samples_ns.empty()) {
                p->print_result(result);
            }
        }
    }

    if (p) {
        p->print_footer();
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
