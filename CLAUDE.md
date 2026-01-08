IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## Project: Minimal llama.cpp Fork

This is an experimental minimal fork of llama.cpp for Apple Silicon (Metal) performance study.
Goal: Reduce complexity to a "single file implementation" feel while maintaining identical inference.

### Supported Architectures

- **llama**: Devstral-2-123B (LLM_TYPE_123B)
- **mistral3**: Devstral-Small-2-24B (LLM_TYPE_3B, LLM_TYPE_8B, LLM_TYPE_14B)

### Completed Phases

| Phase | Description | Lines Removed |
|-------|-------------|---------------|
| 1 | Strip to Metal-only + Mistral3 architecture | ~213K LoC |
| 2 | Remove non-ARM CPU backends and tests | ~86K LoC |
| 3 | Remove unused architecture definitions | ~8.5K LoC |
| 4 | Remove mtmd multimodal support | ~12.9K LoC |
| 5 | Remove recurrent/hybrid memory (SSM/RWKV support) | ~1.8K LoC |
| 6 | Simplify llama_layer struct (Devstral-only members) | ~0.4K LoC |
| 7 | Simplify llm_type enum (Devstral-only types) | ~0.1K LoC |

### Potential Future Phases

- Remove unused llama_model struct members
- Further simplify llama-graph.h/cpp (remove cross-attention, encoder code)

## Testing

Test models for this minimal build:
```
~/projects/llms/Devstral-Small-2-24B-Instruct-2512-UD-Q8_K_XL.gguf
~/projects/llms/Devstral-2-123B-Instruct-2512-UD-Q8_K_XL-00001-of-00003.gguf
```

Quick inference test (24B):
```bash
./build/bin/llama-cli -m ~/projects/llms/Devstral-Small-2-24B-Instruct-2512-UD-Q8_K_XL.gguf \
    -ngl 99 --single-turn -p "The capital of France is" -n 16 -c 1024
```

Quick inference test (123B):
```bash
./build/bin/llama-cli -m ~/projects/llms/Devstral-2-123B-Instruct-2512-UD-Q8_K_XL-00001-of-00003.gguf \
    -ngl 99 --single-turn -p "Hi" -n 8 -c 2048
```

Expected performance on M2 Ultra:
- 24B: ~513 t/s prompt, ~21 t/s generation
- 123B: ~95 t/s prompt, ~4 t/s generation
