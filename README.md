# Local model deployment lab

A hands-on lab to learn the full circuit of self-hosting a large language model: picking
weights, quantization, inference engines (llama.cpp, Ollama, vLLM), measuring prefill and
decode, exposing an OpenAI-compatible endpoint, and wiring a coding agent to it.

The lab avoids buying a GPU. It combines the local machine (Apple M1 Pro, 16 GB) for small
models with Kaggle's free accelerators (2x NVIDIA T4 and TPU v5e-8) for a 27B model.

## Origin and framing

This lab is inspired by the video
[No compres una GPU: monta este servidor de IA gratis de 32 GB](https://youtu.be/o5ubV1tnBmM)
by Jorge Marfil (2026-10-16). The video reviews the viral "free 32 GB AI server" recipe
(Kaggle 2x T4 + Ollama + Qwen 27B + Cloudflare Quick Tunnel + coding agent) and concludes:

- It is **not local AI**. Prompts and code travel through Cloudflare to Google machines.
  If privacy is the reason for self-hosting, only your own hardware qualifies.
- It is **not a daily server**. Weekly quota, session caps, idle timeouts, cold starts of
  15 to 30 minutes and a new tunnel URL every session. The video estimates that a full week
  of the T4 setup, used perfectly, produces about the same tokens as roughly 4 USD of the
  same model on a hosted provider (OpenRouter).
- It **is an excellent sandbox to learn**: install the engine, watch it break, fix it,
  with nothing heating up at home.

This lab takes that last point literally. Every session is a bounded experiment with a
goal, a measurement and a written result. Nothing here is meant to run 24/7.

### A correction worth keeping in mind

The Cloudflare Quick Tunnel cut at 100 seconds does **not** go away on the TPU. What
changes is the prefill speed: according to the video, a 30,000-token agent system prompt
takes about 92 s to prefill on the T4s (8 s of margin before the 100 s cut) and about 3 s
on the TPU. The tunnel limit is the same; the TPU simply stays far away from it.

## Key concepts

| Concept | Plain meaning | Why it matters here |
|---|---|---|
| Token | A chunk of a word, the unit models read and write | All speeds are measured in tokens per second (tok/s) |
| Parameters | The numbers the model learned. 27B = 27 billion | More parameters, more memory, usually smarter |
| Quantization | Storing weights with fewer bits (like saving a photo as JPG) | Qwen 27B is 55.6 GB in bf16 and about 17 GB at 4 bits |
| VRAM | Memory attached to the accelerator | The model (plus KV cache) must fit in it |
| Prefill | Reading the whole prompt before the first output token | Compute-bound (FLOPs). Dominates with long agent prompts |
| Decode | Generating output tokens one by one | Memory-bandwidth-bound. Every token reads all the weights |
| TTFT | Time to first token | Roughly prompt tokens / prefill speed |
| KV cache and context window | Memory that holds the conversation; max text the model sees at once | Long contexts need extra memory on top of the weights |
| Layer split | Model cut by layers across devices; they work in turns | 2x T4 gives the capacity of two cards and the speed of one |
| Tensor parallelism | Every layer split across devices; they work at the same time | What vLLM does across the 8 TPU chips |
| Speculative decoding (MTP) | Predict several tokens ahead, verify in one pass | Raises decode speed without changing the output |

### Back-of-envelope formulas (hypotheses to verify in the lab)

```
decode ceiling (tok/s)  ~ memory bandwidth (GB/s) / bytes of weights read per token (GB)
TTFT (s)                ~ prompt tokens / prefill speed (tok/s)
```

| Hardware | Bandwidth | Weights | Decode ceiling | Video / repo figure |
|---|---|---|---|---|
| 1x T4 (layer split behaves as one) | 320 GB/s | 17 GB (Q4) | ~19 tok/s | 16 to 18 tok/s |
| TPU v5e-8 (8 chips, tensor parallel) | 8 x ~800 GB/s | 55.6 GB (bf16) | ~115 tok/s | ~130 tok/s with MTP |
| M1 Pro (this Mac) | ~200 GB/s | ~5 GB (Qwen3-8B Q4) | ~40 tok/s | to be measured |

These numbers come from the video, the kaggle-tpu-lab README and public spec sheets. The
lab's job is to measure them, not to trust them.

## Tools and services

| Tool or service | Role in the lab | Notes |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Inference engine for GGUF models. `llama-server` exposes an OpenAI-compatible API; `llama-bench` measures prefill and decode | Runs on Apple Metal and on CUDA (T4). `brew install llama.cpp` |
| [Ollama](https://ollama.com) | Friendly wrapper around llama.cpp: downloads, model management, API | It is what the viral tutorials use. No authentication by default. `brew install ollama` |
| [vLLM](https://docs.vllm.ai) | High-throughput engine (PagedAttention, batching, tensor parallelism) | No Apple Silicon backend. On TPU it runs through [tpu-inference](https://github.com/vllm-project/tpu-inference) |
| [Kaggle](https://www.kaggle.com/docs/notebooks) | Free notebooks with accelerators: "GPU T4 x2" and "TPU VM v5e-8" | Phone verification. Check your exact weekly quota in Settings |
| Cloudflare Quick Tunnel (`cloudflared`) | Public `trycloudflare.com` URL for a server inside the notebook | No account needed. Already installed on this Mac |
| [kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab) | Open source (MIT) recipe that serves Qwen3.8-27B with vLLM on the free TPU | Published Kaggle notebook plus a `launch.py` CLI. Read the code before running it |
| [Hugging Face](https://huggingface.co) | Source of the weights | Public models download without a token |
| [OpenRouter](https://openrouter.ai) | Hosted baseline to compare speed and cost | Optional, a few USD of credit |
| Clients | `curl`, Python `openai` SDK, a coding agent (Claude Code, opencode, Codex CLI) | Any OpenAI- or Anthropic-compatible client works |

Note on naming: there is no "ollama.cpp". **llama.cpp** is the engine; **Ollama** is a
separate product built on top of it. In this lab both are used, so the difference becomes
tangible.

### Models used

| Phase | Model | Repository | License |
|---|---|---|---|
| 0 (Mac) | Qwen3-4B and Qwen3-8B, GGUF Q4_K_M | [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF), [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) (official) | Apache-2.0 |
| 1 (2x T4) | Qwen3.8-27B, GGUF UD-Q4_K_M | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) (Qwen publishes no official GGUF of this model; Unsloth is a well-known quantizer) | Apache-2.0 |
| 2 (TPU) | Qwen3.8-27B, bf16 | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (official, mirrored by the recipe) | Apache-2.0 |

Avoid the "uncensored", "abliterated" or merged re-uploads that dominate search results.
The video points out that most T4 tutorials download one of those instead of a faithful
quantization of the official weights.

## Lab roadmap

Each phase uses the same two benchmark prompts so results are comparable:

- **Short prompt**: about 200 tokens in, about 500 tokens out (chat-like).
- **Long prompt**: about 30,000 tokens in (simulates a coding agent's system prompt plus
  files), about 500 tokens out.

### Phase 0: truly local on the Mac

Goal: learn the engines and the measurements for free, with no network in the loop.

1. Run `llama-server` with Qwen3-4B Q4_K_M, protected by an API key, and query it with
   `curl` through the OpenAI-compatible `/v1/chat/completions` endpoint.
2. Measure prefill and decode with `llama-bench`, and with the `timings` field that
   `llama-server` returns on every response.
3. Repeat with Qwen3-8B Q4_K_M and Q8_0. Observe memory use and speed versus model size and
   quantization. Compare against the bandwidth formula.
4. Run the same model through Ollama and compare ergonomics and speed with raw llama.cpp.
5. Run the long prompt and compute TTFT. This is the "silence" the video talks about.
6. Optional: download a small model in safetensors (for example Qwen3-1.7B), convert it to
   GGUF with `convert_hf_to_gguf.py` and quantize it with `llama-quantize`.
7. Expose the local server with a Quick Tunnel and call it from outside (phone on mobile
   data). First contact with `cloudflared`, with nothing at stake.

What you learn: GGUF, quantization levels, prefill versus decode, TTFT, API keys, the
OpenAI-compatible API, the difference between Ollama and llama.cpp.

### Phase 1: Kaggle 2x T4 (replicating the viral recipe, critically)

Goal: reproduce the setup the video criticizes and confirm its numbers.

1. Create a notebook with Accelerator = GPU T4 x2 and Internet = ON. Run `nvidia-smi`.
2. Serve Qwen3.8-27B UD-Q4_K_M with Ollama, as the tutorials do. Note the cold start time.
3. Serve the same GGUF with `llama-server` (CUDA build), with `--api-key`. Check how the
   layers land on each card (`--split-mode layer`, the default) and watch `nvidia-smi` while
   it decodes: one card busy, the other idle.
4. Try `--split-mode row` and compare. Does tensor-style splitting help on T4s?
5. Run both benchmark prompts. Compare with the video: about 16 tok/s decode, about 325 tok/s
   prefill, about 92 s TTFT for 30k tokens.
6. Expose `llama-server` (never bare Ollama) with a Quick Tunnel. Then, deliberately:
   - Trigger a 524 by sending a prompt long enough to exceed 100 s of prefill.
   - Check whether streaming responses survive the tunnel. Cloudflare's docs say Quick
     Tunnels do not support Server-Sent Events, yet the recipes report working agents.
     Find out what actually happens.
7. Optional: try vLLM with tensor parallelism 2 on a model that fits in fp16 (for example
   Qwen3-8B). The T4 is an old architecture (2018, no bf16), so check vLLM's current support
   first and record whatever breaks.
8. Stop the session as soon as the measurements are done.

What you learn: CUDA, multi-GPU placement, layer split versus row split, why bandwidth caps
decode, why long prompts kill the experience behind a 100 s proxy timeout.

### Phase 2: Kaggle TPU v5e-8 with vLLM (the option nobody picked)

Goal: run the same model unquantized with tensor parallelism and see the difference.

1. Read the [kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab) code, mainly
   `qwen38-27b/kernel/` and `launch.py`. Understand what it downloads, patches and exposes.
2. First run via the notebook route (Copy and Edit of the
   [published notebook](https://www.kaggle.com/code/rahim3/qwen3-8-27b-bf16-on-kaggle-tpu-130-tok-s-api),
   Accelerator = TPU VM v5e-8). No API token needed on your Mac.
3. Second run via the terminal route (`python launch.py serve --keepalive-min 90`), which
   needs the Kaggle API token. The `--keepalive-min` flag puts a hard stop on the session.
4. Run both benchmark prompts. Compare with the repo figures: about 130 tok/s decode,
   about 10,300 tok/s prefill, about 22 min from Run to URL.
5. Connect a coding agent to the endpoint for one short, non-sensitive task. The README has
   the exact environment variables for each agent.
6. Stop the session (`python launch.py stop`).

What you learn: vLLM, tensor parallelism, bf16 versus 4-bit, XLA compilation and warm-up,
compile caches, serving a model with tool calling and reasoning parsers.

### Phase 3: comparison and conclusions

1. Run the same two prompts against the same model on a hosted provider (OpenRouter).
2. Fill in the results table and compute the effective cost per million tokens of each
   option, counting cold starts and idle time against the quota.
3. Write the conclusions in `avances.md`: when each option makes sense.

## Metrics to record

Copy this table into `avances.md` (or a `results/` file) for every run:

| Field | Example |
|---|---|
| Date and phase | 2026-10-21, phase 0 |
| Hardware | M1 Pro 16 GB |
| Engine and version | llama.cpp (brew) |
| Model and quantization | Qwen3-4B Q4_K_M |
| Memory used | GB of RAM or VRAM |
| Cold start | Minutes from Run to first answer |
| Prefill speed | tok/s (short and long prompt) |
| Decode speed | tok/s |
| TTFT, long prompt | seconds |
| Errors observed | 524, 502, OOM, timeouts |
| Quota consumed | Kaggle hours |

## Rules of engagement

- **Kaggle acceptable use.** Read the [Kaggle AUP](https://www.kaggle.com/aup). According to
  the video, the version in force since June 2025 lists "server farming" and activity
  unrelated to machine learning or data science as abuse. Keep every session a bounded
  experiment (benchmark, evaluate, stop). No scripts to dodge the idle timeout, no
  always-on endpoint, no sharing the URL with others.
- **Always authenticate.** `llama-server --api-key`, or the API key the TPU recipe generates.
  Never expose Ollama directly: it has no authentication, so the URL becomes the password.
- **Nothing sensitive through the tunnel.** No private code, credentials or personal data.
  Use throwaway tasks and public code.
- **Secrets stay out of Git.** `kaggle.json` lives in `~/.kaggle/` with `chmod 600`.
  Inside notebooks, use Kaggle Secrets. Never paste tokens into notebook cells you publish.
- **Trusted weights only.** Official repositories or well-known quantizers.
- **Read before you run.** Any third-party notebook or script, the TPU recipe included.

## Prerequisites and first steps

1. **Kaggle account.** Create it, verify the phone number under Settings and check the GPU
   and TPU weekly quota shown there (public sources quote about 30 GPU hours and about
   20 TPU hours a week, with session caps between 9 and 12 hours; trust your account page).
   Some accounts also need an identity check (Persona) before TPU sessions are granted.
2. **Hugging Face account** (optional). Not required for public models, useful to browse
   files and model cards.
3. **Mac tooling** for phase 0:

   ```bash
   brew install llama.cpp ollama
   ```

   `cloudflared` is already installed (check with `cloudflared --version`).
4. **Python 3.9+ and the Kaggle CLI** for phase 2's terminal route, in a virtual
   environment (`pip install kaggle`). Create the API token in Kaggle Settings only when you
   get there.
5. **OpenRouter account** (optional, phase 3) with a small credit.
6. **This repository.** Track progress in `TODO.md` (queue) and `avances.md` (log).

### First command to try (phase 0, step 1)

```bash
llama-server -hf Qwen/Qwen3-4B-GGUF:Q4_K_M --port 8080 --api-key change-me
```

Then, from another terminal:

```bash
curl -s http://localhost:8080/v1/chat/completions -H "Authorization: Bearer change-me" -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Explain prefill vs decode in two sentences."}]}'
```

Look at the `timings` object in the response: `prompt_per_second` is prefill speed,
`predicted_per_second` is decode speed.

## Sources

- Video: [No compres una GPU: monta este servidor de IA gratis de 32 GB](https://youtu.be/o5ubV1tnBmM), Jorge Marfil, 2026-10-16.
- [kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab) and its [Qwen3.8-27B recipe](https://github.com/ARahim3/kaggle-tpu-lab/tree/main/qwen38-27b).
- [Cloudflare Quick Tunnels documentation](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/) (testing only, no uptime guarantee, 200 in-flight requests, no SSE).
- [Cloudflare error 524](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-524/) (100 s proxy read timeout, raised only on Enterprise).
- [Kaggle acceptable use policy](https://www.kaggle.com/aup) and [Kaggle notebooks documentation](https://www.kaggle.com/docs/notebooks).
- [llama.cpp](https://github.com/ggml-org/llama.cpp), [Ollama](https://ollama.com), [vLLM](https://docs.vllm.ai), [tpu-inference](https://github.com/vllm-project/tpu-inference).
