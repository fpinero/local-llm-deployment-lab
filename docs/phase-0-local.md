# Phase 0: truly local on the Mac

Step-by-step guide for running Qwen3-4B with llama.cpp on an Apple Silicon Mac, querying it
through the OpenAI-compatible API and measuring prefill and decode speed. Every command
below was run on the reference machine (Apple M1 Pro, 16 GB, macOS 26) and the results are
recorded at the end.

## Step 1: install llama.cpp

```bash
brew install llama.cpp
llama-server --version
```

Homebrew installs the `ggml` library and around forty binaries. The ones used in this lab:

| Binary | Purpose |
|---|---|
| `llama-server` | HTTP server with an OpenAI-compatible API and a built-in web chat |
| `llama-cli` | Interactive chat in the terminal |
| `llama-bench` | Synthetic benchmark of prefill (`pp`) and decode (`tg`) |
| `llama-quantize` | Converts a GGUF to another quantization level |
| `llama-tokenize` | Shows how a text is split into tokens |

On Apple Silicon the build uses the Metal backend, so the model runs on the integrated GPU
and shares the unified memory with the CPU.

## Step 2: create a local API key

Even on localhost, get used to protecting every server. Generate a random key in a file
that Git ignores:

```bash
mkdir -p .secrets && chmod 700 .secrets
openssl rand -hex 24 > .secrets/llama-api-key
chmod 600 .secrets/llama-api-key
```

## Step 3: start the server

```bash
llama-server -hf Qwen/Qwen3-4B-GGUF:Q4_K_M -c 32768 --port 8080 --api-key-file .secrets/llama-api-key
```

| Flag | Meaning |
|---|---|
| `-hf Qwen/Qwen3-4B-GGUF:Q4_K_M` | Download the Q4_K_M file from the official Qwen repository (first run only) |
| `-c 32768` | Context window of 32k tokens, enough for the long benchmark prompt. The KV cache memory grows with this value |
| `--port 8080` | Listen on `http://localhost:8080` |
| `--api-key-file` | Require `Authorization: Bearer <key>` on every API call |

The model is cached under `~/.cache/huggingface/hub/`, so later starts skip the download.
Wait until the health check answers:

```bash
curl -s localhost:8080/health
```

`{"status":"ok"}` means the model is loaded. The built-in chat UI is at
`http://localhost:8080` (it asks for the API key).

## Step 4: first request with curl

```bash
KEY=$(cat .secrets/llama-api-key)
curl -s http://localhost:8080/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Explain prefill vs decode in two sentences."}]}'
```

The response follows the OpenAI format (`choices[0].message.content`) and adds a `timings`
object specific to llama.cpp:

| Field | Meaning |
|---|---|
| `prompt_n` / `prompt_per_second` | Prompt tokens processed and prefill speed |
| `predicted_n` / `predicted_per_second` | Tokens generated and decode speed |

Qwen3 is a reasoning model: by default it thinks first and the reasoning arrives in
`reasoning_content`. To skip it, add `"chat_template_kwargs": {"enable_thinking": false}`
to the request body.

## Step 5: benchmark with the lab script

`scripts/bench.py` uses streaming to measure time to first token (TTFT) from the client
side, and prints the server timings too. It only needs the Python standard library.

```bash
python3 scripts/bench.py --key-file .secrets/llama-api-key --prompt short --no-think
python3 scripts/bench.py --key-file .secrets/llama-api-key --prompt long --no-think --max-tokens 200
```

The long prompt is about 30,000 tokens, the size of a coding agent's system prompt plus a
few files. Its TTFT is the "silence" described in the video.

## Step 6: synthetic benchmark with llama-bench

`llama-bench` loads the model on its own (stop the server first to free memory) and reports
prefill (`pp512`) and decode (`tg128`) speed:

```bash
llama-bench -m ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-GGUF/snapshots/*/Qwen3-4B-Q4_K_M.gguf
```

## Results on the reference machine

Measured on 2026-10-21. Apple M1 Pro (16 GB), llama.cpp 0.7.0 (Homebrew, Metal), Qwen3-4B
Q4_K_M (2.32 GiB), context 32k, thinking disabled for the benchmark runs.

### Server runs (`scripts/bench.py`)

| Run | Prompt tokens | TTFT | Prefill (server) | Decode |
|---|---|---|---|---|
| First curl request (thinking on) | 18 | 0.3 s | 61 tok/s | 38.8 tok/s |
| Short prompt | 64 | 0.45 s | 155 tok/s | 39.1 tok/s |
| Long prompt, cold | 30,363 | **208 s** | 146 tok/s (average) | 17.0 tok/s |
| Long prompt, repeated (prefix cache hit) | 30,363 | **0.17 s** | 1 new token | 17.3 tok/s |

### Synthetic runs (`llama-bench`, 2 repetitions)

| Flash Attention | pp512 | tg128 | pp512 at 16k depth | tg128 at 16k depth |
|---|---|---|---|---|
| off | 398 tok/s | 40.8 tok/s | 109 tok/s | 9.4 tok/s |
| on | 414 tok/s | 42.4 tok/s | 140 tok/s | 24.1 tok/s |

### What the numbers teach

- **Decode is bandwidth-bound.** Ceiling: about 200 GB/s / 2.5 GB, roughly 80 tok/s. Measured
  about 42 tok/s, around half the theoretical ceiling, which is typical.
- **Prefill speed is not constant.** The server log shows the long prompt starting at
  517 tok/s on the first 2,048 tokens and falling to about 70 tok/s near the end, because
  every new token attends to all the previous ones. The 146 tok/s figure is an average.
- **A 30k-token prompt takes 208 s on this Mac.** Behind a Cloudflare Quick Tunnel this
  request would be cut at 100 s with a 524, exactly the failure the video describes for the
  T4s, only worse.
- **Decode also slows down with context.** 39 tok/s with an empty context, 17 tok/s at 30k
  tokens: each generated token must read the whole KV cache as well as the weights.
- **Prefix caching changes everything for agents.** Resending the same 30k-token prefix
  drops TTFT from 208 s to 0.17 s. Coding agents repeat the same system prompt on every
  turn, so only the first request pays the full prefill.
- **Flash Attention matters at long context.** Little difference with an empty context,
  but 2.5x faster decode at 16k depth. `llama-server` uses `auto`, which enabled it here.
- **The API key works.** A request without the key gets HTTP 401.
