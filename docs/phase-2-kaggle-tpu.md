# Phase 2: Kaggle TPU v5e-8 with vLLM

**Status: closed as not viable.** Two attempts (about 7.5 hours of waiting in total) never
got a TPU assigned, so nothing below Step 3 was run. See the availability section. The
recipe review and the prepared steps stay here in case a TPU ever becomes available.

Goal: serve the same model family unquantized (Qwen3.8-27B in bf16, 55.6 GB) on the free
TPU v5e-8 with vLLM and tensor parallelism across 8 chips, and compare it with the 2x T4
results of phase 1. The serving recipe is the open source
[kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab), reviewed before use (below).

## Review of the recipe (commit `9a31b8519a80`, 2026-10-06)

The code was read before running it, as the lab rules require. What it does:

- Builds a Python virtual environment with `uv` and installs `vllm-tpu==0.28.0` (other
  packages resolved as of the environment dataset's build date), applies an embedded patch
  to `tpu_inference` (a port of upstream PR #3178 so multi-token prediction rolls back the
  linear-attention state correctly; byte-identical to `patches/mtp-rollback-v0280.diff`),
  and starts vLLM's OpenAI server on port 8000 with `--api-key`.
- Expects two public Kaggle datasets: `rahim3/qwen3-8-27b-bf16` (a mirror of
  `Qwen/Qwen3.8-27B`, 55.6 GB) and `rahim3/qwen38-tpu-env-v5e8` (a precompiled XLA cache,
  a `cloudflared` binary and a manifest). Without the weights dataset it downloads the
  official weights from Hugging Face.
- Opens a Cloudflare Quick Tunnel to port 8000 and prints the URL and the API key in a
  `READY` banner. Writes only to `/tmp` and `/kaggle/working`. No sudo, no system changes.
- Keeps serving for `keepalive_min` minutes after `READY` (90 in the notebook), then stops
  vLLM. There are no tricks to dodge the idle timeout.

Findings that shape how this lab uses it:

| Finding | Decision |
|---|---|
| The terminal route (`launch.py`) always publishes the endpoint URL **and the API key** to a random topic on the public service ntfy.sh, which keeps messages server-side | **Use the notebook route only**, where this happens only if `ntfy_topic` is set (it is not) |
| `launch.py stop` deletes the whole Kaggle kernel (all versions, logs and outputs), not just the session | Not used |
| The API key can be supplied through `"api_key"` in `serve_config.json` | Use the lab key from the `LLAMA_API_KEY` Kaggle secret, so `bench.py --key-file` works from the Mac |
| vLLM's `--api-key` normally protects only `/v1/*`; `/metrics`, `/docs`, `/openapi.json`, `/tokenize` and `/version` are probably public through the tunnel (not verified) | Verify it from the Mac, keep the tunnel short-lived, never share the URL |
| `cloudflared` (unpinned) and the XLA cache are third-party binaries without checksums | Accepted for a learning run inside the Kaggle sandbox; noted as a supply-chain risk |
| The published notebook on Kaggle was not reviewed; the repository `.ipynb` was (its script cell is identical to `serve_qwen38.py`) | Import the repository notebook at the reviewed commit instead of "Copy and Edit" |
| When `keepalive_min` ends, the cell exits but the TPU session stays allocated | Stop the session manually |

Claimed performance (from the recipe's README): about 130 tok/s single-stream decode with
multi-token prediction (78 without), 10,300 tok/s prefill, about 22 minutes from Run to a
live URL with both datasets attached. Only the single-stream decode figure has a method in
the code; the others are to be measured here.

## Step 0: account

1. In Kaggle **Settings**, check the **TPU** weekly quota (separate from the GPU quota;
   public sources say about 20 hours) and write it in `avances.md`.
2. If Kaggle asks for an identity check (Persona) before granting TPU sessions, complete it.

## Availability: the main finding of this phase

Free TPU v5e-8 machines are scarce. The 2x T4 of phase 1 were assigned in seconds. The TPU
never was:

| Attempt | When (Spain) | Wait | Result |
|---|---|---|---|
| 1 | Saturday 2026-10-24, midday to evening | 6 hours | No TPU, stopped by hand |
| 2 | Sunday 2026-10-25, 8:30 to 10:05 (US night) | about 1.5 hours | No TPU, stopped by hand |

The second attempt was made in the US off-peak window on purpose, so the wait does not
depend only on the time of day.

- Kaggle shows no queue position, no estimated wait and no session metrics while it waits.
- The stop button stays disabled and reloading the page does not change the state.
- Switching the accelerator to None stops the attempt; switching back to TPU starts a new
  one at the end of the queue. The switch also saved a private notebook version (no code
  ran, no key in it).
- Waiting does not consume TPU quota.

Lesson: a free accelerator you cannot get when you need it is a theoretical option. The
video compares speed and does not mention availability. Treat the TPU as opportunistic at
best and keep the T4 path as the dependable one. The claimed figures of the recipe (130
tok/s decode, 10,300 tok/s prefill, 22 minutes to a live URL) remain unverified here.

## Step 1: import the reviewed notebook

On the Mac:

```bash
curl -sSL -o /Volumes/Ext1Tb/models/qwen38-tpu-serve.ipynb https://raw.githubusercontent.com/ARahim3/kaggle-tpu-lab/9a31b8519a80/qwen38-27b/notebook/qwen38-tpu-serve.ipynb
```

In Kaggle: **Create > New Notebook**, then **File > Import Notebook** and upload that file.

## Step 2: session options and inputs

- **Accelerator: TPU VM v5e-8.**
- **Internet: On.**
- **Add Input:** the datasets `rahim3/qwen3-8-27b-bf16` and `rahim3/qwen38-tpu-env-v5e8`.
- **Add-ons > Secrets:** tick `LLAMA_API_KEY`.

## Step 3: configuration with the lab key

Replace the first code cell (`%%writefile serve_config.json`) with a Python cell, so the
key comes from Kaggle Secrets:

```python
import json
from kaggle_secrets import UserSecretsClient

config = {
    "max_model_len": 262144,
    "max_num_seqs": 4,
    "mtp_tokens": 3,
    "reasoning_effort_default": "xhigh",
    "keepalive_min": 60,
    "api_key": UserSecretsClient().get_secret("LLAMA_API_KEY"),
}
json.dump(config, open("serve_config.json", "w"))
print("config written")
```

The values match one of the precompiled configurations in the environment dataset
(262,144 tokens, 4 sequences, MTP 3), so startup uses the cache. `keepalive_min` is lowered
to 60 to protect the quota. The `READY` banner will print the key in the cell output: the
notebook is private, but do not save a version with that output.

## Steps not run (no TPU was assigned)

1. Run the script cell and the launch cell; follow the progress lines (about 22 minutes).
2. From the Mac: confirm `/v1/models` returns 401 without the key, and check which other
   paths answer without it (`/metrics`, `/docs`, `/version`).
3. Measure with `bench.py` (`--model qwen3.8-27b`, `--nonce` to defeat vLLM's prefix cache;
   vLLM returns no `timings`, so prefill is estimated from TTFT on the client).
4. Compare with phase 1: decode, prefill, TTFT for 32k tokens, and a 100k-token prompt.
5. Stop the session.
