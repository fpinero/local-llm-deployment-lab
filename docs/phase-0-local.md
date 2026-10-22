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

## Step 7: a bigger model and a lighter quantization (Qwen3-8B)

Run by the user in guided mode on 2026-10-21. The server is restarted for each file:

```bash
llama-server -hf Qwen/Qwen3-8B-GGUF:Q4_K_M -c 16384 --port 8080 --api-key-file .secrets/llama-api-key
llama-server -hf Qwen/Qwen3-8B-GGUF:Q8_0 -c 4096 --port 8080 --api-key-file .secrets/llama-api-key
```

The context is reduced for Q8_0 on purpose: macOS gives the GPU roughly 10 to 11 GB of the
16 GB, and 8.7 GB of weights plus a 16k KV cache (about 2.3 GB) do not fit. Same dilemma as
on the T4s: either the model fits or the context does.

Synthetic comparison (stop the server first):

```bash
D=$(echo ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B-GGUF/snapshots/*)
llama-bench -m $D/Qwen3-8B-Q4_K_M.gguf -m $D/Qwen3-8B-Q8_0.gguf -p 512 -n 128 -r 3
```

### Results

| Model | File | Decode (`bench.py`, short) | Effective bandwidth | pp512 | tg128 |
|---|---|---|---|---|---|
| Qwen3-4B Q4_K_M | 2.50 GB | 39.1 tok/s | ~98 GB/s | 414 tok/s | 42.4 tok/s |
| Qwen3-8B Q4_K_M | 5.03 GB | 23.6 tok/s | ~119 GB/s | 224 tok/s | 24.5 tok/s |
| Qwen3-8B Q8_0 | 8.71 GB | 20.1 tok/s | ~175 GB/s | 245 tok/s | 20.4 tok/s |

Long prompt with Qwen3-8B Q4_K_M (14,190 tokens): TTFT 91.7 s, decode 17.2 tok/s. The same
size took about 60 s on the 4B.

### What the numbers teach

- **Twice the size is not half the decode speed.** Every token has a fixed cost (kernel
  launches, sampling, HTTP) on top of reading the weights. Small models waste more of the
  memory bus, so bigger models get closer to the bandwidth ceiling.
- **The quantization format changes how well the bus is used.** Q8_0 is trivial to
  dequantize and reaches about 87% of the ~200 GB/s ceiling. Q4_K_M needs more arithmetic
  to unpack and stays near 60% on the M1 Pro. Result: Q8_0 weighs 73% more but is only 15%
  slower at decode.
- **Each format wins one phase.** Prefill is compute-bound, so the cheaper dequantization
  of Q8_0 wins (245 vs 224 tok/s). Decode is memory-bound, so fewer bytes win (Q4_K_M,
  24.5 vs 20.4 tok/s).
- **Prefill scales with parameters at short context, less at long context.** pp512 ratio
  4B/8B is 0.54, close to the parameter ratio of 0.49. With 14k tokens the 8B is only 1.5x
  slower than the 4B, because the attention part costs the same in both: same layer count,
  same KV heads and head size.
- **The KV cache is read on every generated token.** 8B decode goes from 42.4 ms per token
  (empty context) to 58.3 ms (14k context). The extra 16 ms match reading about 2 GB of KV
  cache (14,190 tokens x ~144 KB) at ~120 GB/s.
- **91.7 s for a 14k prompt** leaves 8 s of margin before the 100 s Quick Tunnel cut, the
  same margin the video describes for the T4s, with a prompt half the size.

### Web chat observations

- The built-in chat at `http://localhost:8080` asks for the API key. Copy it without
  printing it: `pbcopy < .secrets/llama-api-key`.
- With thinking on, a trivial question produced 281 output tokens, of which only about 60
  were the visible answer.
- In a conversation, the server log shows `selected slot by LCP similarity ... f_keep`. The
  history is reused from the cache and only the new message is prefilled. The Qwen3 template
  drops previous reasoning from the history, so that part of the cache is discarded.
- llama.cpp log levels: `I` info, `W` warning (shown in red by the terminal), `E` error. The
  `control-looking token` warning is harmless metadata cleanup.

## Step 8: expose the server with a Cloudflare Quick Tunnel

Run by the user in guided mode on 2026-10-22, with Qwen3-4B Q4_K_M and `-c 32768` (the 30k
prompt is known to take about 208 s to prefill, well beyond the 100 s limit).

```bash
# Terminal 1: the server
llama-server -hf Qwen/Qwen3-4B-GGUF:Q4_K_M -c 32768 --port 8080 --api-key-file .secrets/llama-api-key

# Terminal 2: the tunnel (anonymous, random *.trycloudflare.com URL, no account)
cloudflared tunnel --url http://localhost:8080

# Terminal 3: the client
export TUNNEL=https://<random-words>.trycloudflare.com
curl -s -o /dev/null -w "%{http_code}\n" $TUNNEL/v1/models     # expect 401
python3 scripts/bench.py --base-url $TUNNEL/v1 --key-file .secrets/llama-api-key --prompt short --no-think
python3 scripts/bench.py --base-url $TUNNEL/v1 --key-file .secrets/llama-api-key --prompt long --no-think --max-tokens 200
python3 scripts/bench.py --base-url $TUNNEL/v1 --key-file .secrets/llama-api-key --prompt long --no-think --max-tokens 200 --no-stream --no-cache
```

`cloudflared` opens an outbound connection to Cloudflare (QUIC over UDP, HTTP/2 over TCP as
fallback), so it works behind any NAT or firewall. That is why the Kaggle recipes use it: a
notebook cannot accept inbound connections, but it can connect out. This version of the
Quick Tunnel opens a single connection (`ha-connections:1`), here to the Madrid data center.

### Results

| Request through the tunnel | Outcome |
|---|---|
| `/v1/models` without API key | 401, the key protects the public URL |
| Short prompt, streaming | OK. TTFT 1.6 s (1.4 s of it was the cold server, not the tunnel), decode 41 tok/s, tokens arrive progressively |
| Long prompt (30,363 tokens), streaming | **OK after 205 s of prefill**, no 524 |
| Long prompt, no streaming, no prompt cache | **HTTP 524**. `cloudflared` logs `context canceled`; llama-server cancels the task at about 124 s, 74% into the prefill |

### What the experiment teaches

- **The 100 s limit is about when the response starts, not how long it lasts.** With
  streaming, llama-server sends the HTTP headers immediately and Cloudflare stops counting.
  Without streaming nothing is sent until the full answer is ready, so Cloudflare cuts at
  100 s with a 524.
- **Whether a setup breaks depends on the client and the engine.** Does the client stream,
  and does the engine send headers early? llama-server does. Ollama, used by the viral T4
  tutorials, is still to be tested (phase 1).
- **Streaming through a Quick Tunnel worked**, even though Cloudflare documents that Quick
  Tunnels do not support Server-Sent Events. Observed behavior, not a guarantee.
- **A cut request still burns compute.** The cancellation travels Cloudflare, then
  `cloudflared`, then llama-server; the server kept prefilling for about 20 to 25 s after
  the cut. On Kaggle that is quota.
- **Warm up before measuring.** The first request after starting the server pays for Metal
  kernel and graph preparation.
- `bench.py` options added for this step: `--no-stream`, `--no-cache` (llama.cpp
  `cache_prompt: false`) and a `headers received` line that shows when the response headers
  arrived. On HTTP errors it prints how many seconds it took to fail.
- Close the tunnel (`Ctrl+C`) as soon as the experiment ends. The URL is gone for good and the
  next run gets a new one.

## Step 9: the same model through Ollama

Run by the user in guided mode on 2026-10-22, with Ollama 0.40.8 (Homebrew).

```bash
brew install ollama
OLLAMA_CONTEXT_LENGTH=32768 ollama serve                       # Terminal 1, port 11434
ollama pull hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M                     # same GGUF as llama.cpp
ollama run hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M --verbose           # chat with stats, /bye to exit
ollama ps                                                      # what is loaded, where, how big
python3 scripts/bench.py --base-url http://localhost:11434/v1 --model hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M --reasoning-effort none --prompt short
python3 scripts/bench.py --base-url http://localhost:11434/v1 --model hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M --reasoning-effort none --prompt long --max-tokens 200
```

The tunnel test was done locally on purpose: Ollama has no authentication, so the lab rules
forbid exposing it, and step 8 showed that the 524 depends only on when the headers arrive,
which can be measured on localhost.

### Ollama runs llama-server inside

The process list shows Ollama starting its own bundled `llama-server`:

```
.../libexec/lib/ollama/llama-server --model ~/.ollama/models/blobs/sha256-7485fe6f11af...
    -c 40960 -np 1 --no-jinja --chat-template chatml --flash-attn auto
    -b 512 -ub 512 --context-shift --keep 4
```

- The blob hash is the same file llama.cpp downloaded to `~/.cache/huggingface`, stored a
  second time in `~/.ollama/models` (content-addressed blobs plus manifests).
- Ollama renders the chat template itself (Go template layer) and passes plain text.
- `--context-shift --keep 4`: when the context fills up, old tokens are dropped silently
  instead of returning an error.
- The context is 40,960, not the requested 32,768: the `params` layer published in the Qwen
  repository sets `num_ctx: 40960` and model parameters take precedence over
  `OLLAMA_CONTEXT_LENGTH`, which is only the default for models without one.
- The same layer sets Qwen's recommended sampling (temperature 0.6, top_p 0.95, top_k 20,
  presence_penalty 1.5). `ollama run` uses them; the OpenAI-compatible endpoint used
  temperature 1.0 and top_p 1.0, so the same model behaves differently depending on the API.
- `ollama ps` reported 8.8 GB, 100% GPU: 2.5 GB of weights plus about 6 GB of KV cache for
  40,960 tokens. Ollama measured the GPU budget at 11.8 GiB of the 16 GB.
- The model is unloaded after 5 minutes idle (`OLLAMA_KEEP_ALIVE`); the next request paid
  3.5 s of reload.
- Ollama enables its cloud features by default (`-cloud` models run on ollama.com, not
  locally). `OLLAMA_NO_CLOUD=1` disables them.

### Results

| Engine | Decode | Long prompt prefill | Long prompt TTFT | Headers received (long, streaming) |
|---|---|---|---|---|
| llama-server | 39.1 tok/s | 146 to 148 tok/s | 205 to 208 s | immediately |
| Ollama | 40.0 tok/s | 148.3 tok/s | 205 s | **205 s, together with the first token** |

### What the numbers teach

- **Same engine, same speed.** Ollama adds no measurable overhead to decode or prefill.
- **Ollama holds the HTTP headers until the first token, even when streaming.** Behind a
  Quick Tunnel a 205 s prefill would get a 524 at 100 s, while llama-server passed. This is
  the concrete cause of the 524 errors in the viral T4 tutorials, which use Ollama.
- **What Ollama costs is control:** no authentication, context and sampling decided by model
  parameters and by the API used, silent context shift, automatic unloading and a duplicate
  copy of every model. What it gives: easy multi-model management, automatic GPU/CPU split
  and many ready-made integrations. Good for personal desktop use; llama-server (or vLLM)
  for serving.
- **Thinking is not reasoning.** Asked the classic trick question (wash the car at a car
  wash 150 m away: walk or drive?), Qwen3-4B thought for 531 tokens and recommended walking.
  Fluent formatting hides weak reasoning in small models.

## Step 10: a small reasoning evaluation (the car wash question)

Run by the user in guided mode on 2026-10-22. `scripts/ask.py` repeats a question N times
with Qwen's recommended sampling for thinking mode (temperature 0.6, top_p 0.95, top_k 20)
and prints every final answer, so a human can judge them. Thinking stays on.

```bash
llama-server -hf Qwen/Qwen3-8B-GGUF:Q4_K_M -c 8192 -np 1 --port 8080 --api-key-file .secrets/llama-api-key
python3 scripts/ask.py --key-file .secrets/llama-api-key --runs 5
```

The question: *"I have to wash the car; the car wash is 150 meters from home. Would you
recommend going by car or on foot?"* The correct answer is by car: the car is what needs
washing.

### Results

| Model | Correct | Output tokens per answer | Time per answer |
|---|---|---|---|
| Qwen3-4B Q4_K_M (Ollama, 1 run) | 0/1 | 531 | 14 s |
| Qwen3-4B Q4_K_M (llama-server from the external disk, 5 runs) | 0/5 (run 5 could not be judged, see below) | 1,390 to 2,224 | 35 to 59 s |
| Qwen3-8B Q4_K_M | 2/5 | 693 to 3,498 | 31 to 156 s |
| Qwen3-8B Q8_0 | 0/5 to 1/5 (run 5 states driving is the only option, then calls it inefficient) | 1,017 to 2,960 | 52 to 150 s |

### What the numbers teach

- **One answer proves nothing; a rate does.** Run 1 of the Q4_K_M alone would suggest the 8B
  reasons well; runs 2 or 5 alone would suggest it fails like the 4B. With 5 runs, the true
  rate could still be anywhere between roughly 10% and 75%.
- **Longer thinking helps but does not guarantee.** Both correct answers came from long
  reasoning (3,498 and 1,946 tokens), the shortest runs failed, yet a 3,092-token run also
  failed.
- **Q4_K_M versus Q8_0 is not distinguishable with 5 runs.** The gap between 2/5 and 1/5 is
  within chance. Quantization effects on quality are small compared to sampling noise here.
- **Quality costs tokens, tokens cost time.** Up to 2.5 minutes per common-sense question on
  this Mac, for a 40% hit rate.
- **Same model, different paths.** Tiny numerical differences between quantizations change
  token probabilities; one different token early in the reasoning sends the rest down a
  different path.
- **Size shows, within the limits of the sample.** 8B: 2 or 3 correct out of 10. 4B: 0 out
  of 6. Consistent with theory, not a firm proof. The 4B also reasons shorter (never above
  about 2,200 tokens) and invents facts with confidence (150 m "about 1/4 of a mile").
- **Look at the end of the answer.** These models list pros and cons first and conclude
  last. The first version of `ask.py` showed only the first 400 characters, so one 4B answer
  could not be judged. It now prints both ends of long answers.

## Model storage

The internal disk filled up quickly: llama.cpp caches downloads in `~/.cache/huggingface`
and Ollama keeps its own copy in `~/.ollama/models`. The kept models now live on the
external disk and are loaded with `-m` instead of `-hf`:

```bash
# Copy the real files, not the snapshot symlinks of the Hugging Face cache
cp -L ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-GGUF/snapshots/*/Qwen3-4B-Q4_K_M.gguf /Volumes/Ext1Tb/models/
# Verify: the SHA-256 must match the blob name in the cache
shasum -a 256 /Volumes/Ext1Tb/models/Qwen3-4B-Q4_K_M.gguf
# Run from the external disk
llama-server -m /Volumes/Ext1Tb/models/Qwen3-4B-Q4_K_M.gguf -c 8192 --port 8080 --api-key-file .secrets/llama-api-key
# Download new models straight to the external disk
curl -L -o /Volumes/Ext1Tb/models/<file>.gguf https://huggingface.co/<org>/<repo>/resolve/main/<file>.gguf
```

Notes:

- In the Hugging Face cache, `snapshots/` only holds symlinks to `blobs/`, whose names are
  the SHA-256 of their content. `cp` without `-L` would copy a dangling link.
- The GGUF is memory-mapped: once loaded, inference does not read the USB disk again unless
  memory pressure evicts pages.
- Freed space may not show up immediately in `df` while an APFS (Time Machine) local
  snapshot still references the deleted blocks. macOS releases it on its own.
- Stop the server before deleting its model. `rm` removes the name, not the data: a running
  llama-server kept 8.7 GB of a deleted Q8_0 file allocated until it exited. `lsof -p <pid>`
  shows such open files.
