IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## Project: Minimal llama.cpp Fork

This is an experimental minimal fork of llama.cpp for Apple Silicon (Metal) performance study.
Goal: Reduce complexity to a "single file implementation" feel while maintaining identical inference.

### Completed Phases

| Phase | Description | Lines Removed |
|-------|-------------|---------------|
| 1 | Strip to Metal-only + Mistral3 architecture | ~213K LoC |
| 2 | Remove non-ARM CPU backends and tests | ~86K LoC |
| 3 | Remove unused architecture definitions | ~8.5K LoC |
| 4 | Remove mtmd multimodal support | ~12.9K LoC |

### Next Phases

- **Phase 5**: Clean up remaining dead code in llama-model.cpp (~500 LoC)

## Testing

Test model for this minimal build:
```
~/projects/llms/Devstral-Small-2-24B-Instruct-2512-UD-Q8_K_XL.gguf
```

Quick inference test:
```bash
./build/bin/llama-cli -m ~/projects/llms/Devstral-Small-2-24B-Instruct-2512-UD-Q8_K_XL.gguf \
    -ngl 99 --single-turn -p "The capital of France is" -n 16 -c 1024
```

Expected performance on M2 Ultra: ~513 t/s prompt, ~21 t/s generation
