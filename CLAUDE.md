IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

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
