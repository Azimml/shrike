# vLLM head-to-head protocol (same hardware, same harness, same workload)

Both engines serve Qwen2.5-0.5B-Instruct bf16 from the same local snapshot on
the same RTX 3050 Laptop 4GB. The identical load generator drives both
(`bench/load_gen.py`; `--api openai` for vLLM): 256 requests x 64 forced
tokens, burst-open, concurrency 256.

## shrike
```bash
.venv/bin/python -m shrike.server.api --max-running 256
.venv/bin/python -m bench.load_gen --concurrency 256 --num-requests 256 \
  --out bench/results/h2h_shrike.json
```

## vLLM (0.10.2, --enforce-eager to fit 4GB; noted in results)
```bash
.venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model models_cache/qwen2.5-0.5b-instruct --served-model-name qwen \
  --max-model-len 2048 --gpu-memory-utilization 0.8 --enforce-eager \
  --disable-log-requests --port 8001
.venv/bin/python -m bench.load_gen --api openai --model qwen \
  --url http://127.0.0.1:8001 --concurrency 256 --num-requests 256 \
  --out bench/results/h2h_vllm.json
```

Run each pair back-to-back (same thermal conditions); report aggregate tok/s,
TTFT p50/p99, inter-token p50/p99. If VRAM allows, also try vLLM without
--enforce-eager (CUDA graphs) and report both.
