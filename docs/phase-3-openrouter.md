# Phase 3: the same model on a hosted provider (OpenRouter)

Goal: run the same prompts against Qwen3.8-27B on paid hosted inference and compare it with
the Mac (phase 0) and the 2x T4 (phase 1): speed, time to first token, availability, and
the real cost per request. Phase 2 (free TPU) was closed because no TPU was ever assigned,
so this phase is also the reference for what "unquantized 27B at datacenter speed" looks
like.

[OpenRouter](https://openrouter.ai) is a router, not an inference provider. It exposes one
OpenAI-compatible API (`https://openrouter.ai/api/v1`) and forwards each request to one of
the providers that serve the model. That makes it a good lab instrument: the same model
name runs on different hardware and at different precisions, and you can pin the provider
per request.

## What the catalog shows (snapshot of 2026-10-26)

Taken from the public endpoint `https://openrouter.ai/api/v1/models/qwen/qwen3.8-27b/endpoints`
(no key needed). 20 providers serve `qwen/qwen3.8-27b`. A selection, in US dollars per
million tokens:

| Provider (slug) | Precision | Context | Input | Output | Cached input |
|---|---|---|---|---|---|
| DeepInfra (`deepinfra`) | bf16 | 262,144 | 0.15 | 1.875 | 0.0375 |
| Cerebras (`cerebras`) | fp16 | 65,536 | 0.99 | 1.49 | 0.99 |
| Chutes (`chutes`) | fp8 | 262,144 | 0.039 | 1.25 | 0.0039 |
| Darkbloom (`darkbloom`) | fp4 | 262,144 | 0.035 | 2.20 | 0.0175 |
| Alibaba (`alibaba`) | not stated | 1,000,000 | 0.425 | 2.55 | 0.085 |

Lessons before running anything:

- **The same model name hides different models in practice.** fp4, fp8, bf16 and fp16
  variants of the same weights are sold under one id. Default routing picks one for you,
  so quality and speed can change between two identical requests. Pinning the provider
  (below) is what makes a measurement reproducible.
- **Prices span more than 20x on input.** Cheap providers usually run lower precision or
  older hardware. Cerebras charges more for input because its wafer-scale chips are built
  for speed.
- **Cached input is much cheaper** on most providers: the prefix caching seen in phases 0
  and 1, now with a price tag.
- There is no free variant (`:free`) of this model.

## How the scripts talk to OpenRouter

`scripts/bench.py` and `scripts/ask.py` need no new dependencies. For this phase they
gained:

- `--provider deepinfra` (or `deepinfra,chutes`): sends
  `"provider": {"order": [...], "allow_fallbacks": false}`, so OpenRouter uses only those
  providers in that order and fails instead of silently switching.
- They print the `provider` that served the request, the `cost` in dollars from `usage`,
  cached input tokens and reasoning tokens.
- `bench.py` counts the `reasoning` field of OpenRouter's stream for the time to first
  token (llama.cpp calls it `reasoning_content`).

OpenRouter returns no `timings` object, so prefill is estimated from TTFT on the client.
That estimate includes network time, routing and any queue at the provider: it is a lower
bound on the real prefill speed.

## Step 0: account, credit and key (done by you)

1. Create an account at openrouter.ai and add a small credit (a few dollars is far more
   than this phase needs; the estimate is below).
2. In **Keys** (API keys), create a key for the lab with a **credit limit** (for example
   2 dollars), so a mistake cannot spend more. Not in **Management Keys**: those manage
   other keys and cannot call models (the first attempt in this lab made that mistake, so
   delete one if you created it by accident).
3. Optional: in the privacy settings, decide whether providers that may train on prompts
   are allowed. The lab prompts are synthetic, so it does not matter here, but it does for
   real code.
4. Save the key without it appearing on screen or in the shell history (copy it, then
   paste it from the clipboard into the file):

```bash
pbpaste > .secrets/openrouter-api-key
```

`.secrets/` is in `.gitignore`. Never paste the key in a chat or a notebook cell.

## Steps (guided)

1. Check the key and its limit (`GET /api/v1/key`).
2. Short prompt with default routing: which provider does OpenRouter pick?
3. Short prompt pinned to three providers: DeepInfra (bf16), Cerebras (fp16), Chutes
   (fp8). Compare decode speed and cost.
4. Long prompt (32k tokens, `--nonce`) on DeepInfra and Cerebras: TTFT and prefill against
   the 72 seconds of the 2x T4.
5. Prefix caching with a price: the same long prompt twice without `--nonce`, looking at
   `cached tokens` and `cost`.
6. The car wash question with `ask.py` (5 runs) on DeepInfra bf16, to compare the 27B with
   the 4B and 8B of phase 0.
7. Cost: total spent according to the key, cost per million tokens, and the comparison
   table of all phases.

Expected spend for the whole phase: a 32k-token prompt costs about half a cent on DeepInfra
and about three cents on Cerebras. Everything together should stay well under 0.25 dollars.

## Results

### Steps 1-3: key check and short prompt (2026-10-26)

Key check: `limit` 2, `usage` 0, `is_management_key` false, expires in six months. A first
attempt returned 401 "Missing Authentication header" because the command ran from inside
`.secrets/`: `cat` failed, `$(...)` expanded to nothing and the header was sent empty. An
error inside a command substitution does not stop the outer command.

Short prompt (about 123 tokens in), `--nonce`, streaming, from the Mac:

| Run | Provider (precision) | Output tokens | Headers | TTFT | Decode (client) | Cost |
|---|---|---|---|---|---|---|
| Default routing | DekaLLM (unknown) | 500 (all reasoning) | 1.15 s | 1.15 s | 77.2 tok/s | $0.001255 |
| `--provider deepinfra` | DeepInfra (bf16) | 500 (391 reasoning) | 0.44 s | 1.77 s | 60.6 tok/s | $0.000956 |
| `--provider cerebras` | Cerebras (fp16) | 500 (all reasoning) | 0.45 s | 0.45 s | 896.7 tok/s | $0.000868 |
| `--provider chutes` | Chutes (fp8) | 500 (not reported) | 1.49 s | 1.62 s | 54.0 tok/s | $0.000630 |
| `--provider cerebras --max-tokens 4000` | Cerebras (fp16) | 1,414 (483 reasoning) | 0.82 s | 0.82 s | 1,266.6 tok/s | $0.002230 |

Phase 1 reference, same model in Q4_K_M on 2x T4 with `-sm tensor`: 16.5 to 17.3 tok/s.

What the numbers show:

- Every cost matches the catalog prices exactly (input tokens x input price plus output
  tokens x output price). On this model an output token costs 12 to 60 times an input
  token, depending on the provider: decode is the expensive phase. Reasoning tokens are
  billed as output even when the model never reaches the answer.
- Default routing is not reproducible: it picked DekaLLM, which is neither the cheapest
  nor states its precision. Pin the provider for any measurement.
- DeepInfra in bf16 (60.6 tok/s) sits right at the bandwidth ceiling of one H100 for 54 GB
  of weights (3.35 TB/s / 54 GB, about 62 tok/s). The hardware is not published, so this is
  consistent with the theory, not proof.
- Chutes in fp8 is slower than DeepInfra in bf16. Hosted per-user speed depends on how many
  users the provider batches on each GPU, a business choice; Chutes uses fp8 to be cheap.
- Cerebras keeps the weights in on-chip SRAM, so the memory-bandwidth limit of every other
  system in this lab almost disappears: 15 to 20 times DeepInfra at the same precision and
  more than 50 times the 2x T4. Its output price is also the lowest of the three; its input
  price is the highest.
- Short generations underestimate very fast servers. The client measures from the first
  token to the end of the stream, which includes a fixed tail (final usage event, stream
  close). With 500 tokens that tail is a large share of about half a second, so Cerebras
  measured 897 tok/s; with 1,414 tokens it measured 1,267 tok/s, and the real figure is
  probably higher still.
- OpenRouter sends the response headers at once and keeps the connection alive while the
  model works (headers at 0.44 s, first token at 1.77 s on DeepInfra), the opposite of
  Ollama, so a proxy's 100-second cut would not apply.
- With a 123-token prompt the TTFT-based prefill estimate (70 to 276 tok/s) only measures
  network, routing and queueing. Step 4 measures prefill with a long prompt.

### Step 4: long prompt, 32k tokens (2026-10-26)

Same long prompt as phase 1 (`--prompt long`, 32,615 tokens with the nonce line), `--nonce`,
streaming, pinned providers:

| Provider (precision) | TTFT | Prefill (client, lower bound) | Decode (client) | Output tokens | Cost |
|---|---|---|---|---|---|
| DeepInfra (bf16) | 3.55 s | 9,193 tok/s | 64.2 tok/s | 259 | $0.005378 |
| Cerebras (fp16) | 2.45 s | 13,331 tok/s | 1,148.6 tok/s | 215 | $0.032609 |
| Phase 1: 2x T4, Q4_K_M, `-sm layer` | 72.2 s | 453.5 tok/s | 12.2 tok/s | | quota |

Predictions: the user said 5 s and 3 s; Claude said 3 to 6 s and 1 to 3 s, from about
1.8 x 10^15 FLOP for the prompt on one H100 at half its peak.

What the numbers show:

- **Prefill is compute-bound, so Cerebras's lead shrinks** from 15 to 20 times in decode to
  about 1.5 times here. Its SRAM removes the memory-bandwidth limit, which is what decode
  hits, but prefill is limited by arithmetic, where a datacenter GPU is much closer.
  This is the prefill/decode split of phase 0, measured on two very different machines.
- Both TTFT values include about 0.9 s of connection, upload and routing (headers arrived
  at 0.88 s in both runs), so the real prefill speeds are higher than the table shows.
- A datacenter GPU reads the 32k-token prompt about 20 times faster than the 2x T4 (3.55 s
  against 72.2 s). The kaggle-tpu-lab recipe claimed 10,300 tok/s on the free TPU, in the
  same range as DeepInfra.
- Decode does not slow down at 32k tokens on DeepInfra (64.2 tok/s against 60.6 with the
  short prompt). Qwen3.8's hybrid attention keeps the KV cache small, as seen in phase 1.
  The Cerebras decode figure comes from only 215 tokens and underestimates it.
- **With long prompts the bill is the input.** Input is 91% of the DeepInfra cost and 99%
  of the Cerebras cost. Cerebras costs six times more to save 1.1 seconds. Coding agents
  resend tens of thousands of tokens of context on every turn, so input price and prefix
  caching (step 5) decide what they cost.

### Step 5: prefix caching with a price (2026-10-26)

The same long prompt twice in a row, without `--nonce`, pinned to DeepInfra:

| Run | Prompt tokens | Cached tokens | TTFT | Decode (client) | Output tokens | Cost |
|---|---|---|---|---|---|---|
| 1 (cold) | 32,596 | 0 | 3.48 s | 42.9 tok/s | 261 | $0.005379 |
| 2 (warm) | 32,596 | 32,144 | 0.71 s | 41.5 tok/s | 349 | $0.001928 |

Predictions: the user said 50% of the first run for both TTFT and cost; Claude said about
30% of the cost and 1 to 1.5 s, if the second request reached the replica holding the cache.

What the numbers show:

- The cache hit covered 98.6% of the prompt. The 452 uncached tokens are the tail: the
  cached count (32,144) is a multiple of 16, consistent with a KV cache managed in
  fixed-size blocks where only full blocks of the shared prefix are reused.
- TTFT fell to 20% of the cold run (0.71 s against 3.48 s), better than both predictions.
  Headers arrived at 0.50 s, so the provider spent about 0.2 s on the lookup and the 452
  new tokens.
- Cost: input fell from $0.00489 to $0.00127 (26%), exactly 452 tokens at $0.15/M plus
  32,144 at $0.0375/M. The total is 36% of the cold run only because the second answer was
  longer (349 output tokens against 261).
- The TTFT-based prefill estimate of run 2 (45,945 tok/s) is meaningless: the provider did
  not compute those tokens, it read them from the cache. This is why every speed run uses
  `--nonce`.
- Decode on the same provider dropped from about 62 tok/s (step 4, minutes earlier) to
  about 42 tok/s in both runs. Hosted per-user speed changes with the provider's load: a
  single measurement is a sample, not a property of the model.
- For coding agents, which resend the same growing context on every turn, prefix caching
  is what makes hosted inference fast and cheap: one fifth of the waiting time and one
  quarter of the input cost on every turn after the first.

### Step 6: the car wash question, 5 runs (2026-10-26)

`scripts/ask.py --provider deepinfra --runs 5` (bf16, Qwen's thinking-mode sampling, up to
4,096 tokens per answer, no streaming):

| Model | Correct | Output tokens per answer | Time per answer | Cost of the 5 runs |
|---|---|---|---|---|
| Qwen3.8-27B bf16 (DeepInfra) | **5/5** | 177 to 416 | 5.6 to 12.9 s | $0.003162 |
| Qwen3-8B Q4_K_M (Mac, phase 0) | 2/5 | 693 to 3,498 | 31 to 156 s | electricity |
| Qwen3-4B Q4_K_M (Mac, phase 0) | 0/5 | 1,390 to 2,224 | 35 to 59 s | electricity |

Predictions: the user said 2 at most; Claude said 3 or 4.

What the answers show:

- All five recommend the car because the car has to be at the car wash. Runs 1 to 3 add a
  hedge for a reading where only the person goes there; runs 4 and 5 answer directly.
- The 27B reasons far less (541 to 1,363 characters of reasoning) and is right every time,
  while the 8B needed up to 3,498 tokens and was right 2 times out of 5. A stronger model
  does not need to think longer; it thinks better.
- Caveats: Qwen3.8 is a newer generation than the Qwen3 4B and 8B of phase 0, so this
  mixes size and generation. The question is a well-known trick that circulates online and
  may be in the training data. Five runs bound the rate loosely. It is consistent with
  "bigger and newer reasons better", not a benchmark.

### Step 7: cost (2026-10-26)

`GET /api/v1/key` after all the runs: `usage` 0.054393583 dollars, `limit_remaining`
1.945606417. The sum of the `cost` lines printed by the scripts was 0.054395 (each rounded
to six decimals), so OpenRouter billed exactly what each response reported.

The whole phase processed about 131,000 input tokens and 6,000 output tokens (plus the five
short car wash prompts) for 5.4 cents. The Cerebras long prompt alone was 60% of that.

Cost of one typical coding agent turn (32,600 tokens of context in, 300 tokens out), at the
prices measured here:

| Option | Cost per turn | TTFT |
|---|---|---|
| DeepInfra bf16, cold | $0.0055 | 3.5 s |
| DeepInfra bf16, prefix cache hit | $0.0018 | 0.7 s |
| Cerebras fp16 (no cache discount) | $0.0327 | 2.5 s |
| 2x T4 on Kaggle, Q4_K_M | $0 and quota | 72 to 77 s |

What the free Kaggle quota is worth at these prices: 30 GPU hours a week on the 2x T4,
spent entirely on decode at 17 tok/s, produce about 1.8 million output tokens, which cost
about $3.40 on DeepInfra. Spent entirely on prefill at 450 tok/s, they read about 49
million input tokens: about $7.30 on DeepInfra, or $1.80 with cache hits. The full weekly
quota is worth a few dollars of hosted inference that runs 4 to 75 times faster, with no
setup time and no queue.

## Comparison of all phases

Same long prompt (about 30,000 to 32,600 tokens) where available:

| Phase | Where | Model | Decode | TTFT, long prompt | Startup | Money | Availability |
|---|---|---|---|---|---|---|---|
| 0 | M1 Pro 16 GB, llama-server | Qwen3-4B Q4_K_M | 39 to 42 tok/s | 208 s (30k) | seconds | electricity | always |
| 0 | M1 Pro 16 GB, llama-server | Qwen3-8B Q4_K_M | 23.6 tok/s | 91.7 s (14k) | seconds | electricity | always |
| 1 | Kaggle 2x T4, `-sm tensor` | Qwen3.8-27B Q4_K_M | 16.5 to 17.3 tok/s | 76.2 s (32k) | about 7 minutes per session | free, 30 h a week | seconds to get a session |
| 2 | Kaggle TPU v5e-8, vLLM | Qwen3.8-27B bf16 | not measured | not measured | 22 minutes claimed | free, 20 h a week | none in 7.5 hours |
| 3 | OpenRouter, DeepInfra | Qwen3.8-27B bf16 | 42 to 64 tok/s | 3.5 s (0.7 s cached) | none | $0.0055 per 32k turn | immediate |
| 3 | OpenRouter, Cerebras | Qwen3.8-27B fp16 | 897 to 1,267 tok/s | 2.5 s | none | $0.033 per 32k turn | immediate |

Quality on the car wash question: 4B 0/5, 8B 2/5 (Mac), 27B 5/5 (DeepInfra bf16).

## Conclusions

- **The Mac** is the place to learn and to keep private data private: instant start, no
  network, no quota. With 16 GB it is limited to small models, which reason poorly and
  read long prompts slowly (minutes for 30k tokens).
- **Kaggle 2x T4** runs a 27B model for free and teaches multi-GPU serving, tunnels and
  their limits. It is too slow for agent work (more than a minute before the first token
  of every 32k turn), every session starts from zero, and the quota is worth only a few
  dollars of hosted inference a week.
- **The free TPU** is a theoretical option: the recipe is sound and the claimed speed is
  in the range of a datacenter GPU, but no machine was assigned in 7.5 hours over two days.
- **Hosted inference through OpenRouter** is the practical choice for real work with an
  open model: same weights, 20 times faster prefill than the T4, prefix caching, and cents
  per session. The price is that prompts leave your machine and that the provider, the
  precision and the speed vary unless you pin them.
- **The prefill/decode split explains every number in this lab.** Decode follows memory
  bandwidth (Cerebras's SRAM: 20 times a GPU), prefill follows compute (Cerebras: 1.5
  times a GPU), and long prompts make input price and caching dominate the bill.
- **What the video got right and what it left out.** Kaggle plus a tunnel is a good way to
  learn and not "local AI". It does not mention that the free TPU may never start, that
  the tunnel holds headers until the first token, or how little the free hours are worth
  compared with a cheap hosted endpoint.
