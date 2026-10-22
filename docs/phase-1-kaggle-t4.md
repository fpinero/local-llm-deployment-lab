# Phase 1: Kaggle 2x T4

Goal: reproduce the setup the video criticizes (Qwen 27B at 4 bits on two NVIDIA T4 cards,
exposed through a Cloudflare Quick Tunnel) and confirm or refute its numbers with the same
methods used in phase 0. The engine is llama-server, not Ollama: phase 0 showed that Ollama
holds the HTTP headers until the first token, which is what makes long prompts fail behind
the tunnel.

This guide is written step by step as the lab advances. Steps 0 to 2 are defined; the
following ones depend on what the reconnaissance in step 2 reports.

## Rules for Kaggle sessions

- **Every session is a bounded experiment.** Open it with a goal, measure, write the results,
  stop it.
- **Stop the session when you finish.** GPU quota is consumed while the session is running,
  even if nothing is executing. Use the power button or "Stop session" in the notebook.
- **No tricks to dodge the idle timeout.** The Kaggle acceptable use policy treats serving
  infrastructure and non-ML activity as abuse; benchmarking and evaluating a model is what
  the platform is for.
- **Write down your real quota.** Public sources disagree (about 30 GPU hours a week,
  sessions of 9 to 12 hours). The numbers that count are the ones in your account settings.
- **Nothing sensitive goes through the tunnel**, and the server always requires an API key.

## Step 0: account preparation (web interface)

1. Sign in to Kaggle, open Settings and verify your phone number. Without it, notebooks have
   no internet access and no GPU.
2. If Kaggle asks for an identity check (Persona), complete it. Some accounts need it before
   accelerators are granted.
3. In Settings, write down the weekly GPU quota and the hours already used.
4. Store the lab API key as a Kaggle secret, so the same key works on the Mac and in the
   notebook:
   - On the Mac, copy it without printing it: `pbcopy < .secrets/llama-api-key`.
   - In a notebook, open **Add-ons > Secrets**, add a secret with label `LLAMA_API_KEY` and
     paste the value. Secrets are stored outside the notebook and never appear in saved
     versions.

With the same key on both sides, `scripts/bench.py --key-file .secrets/llama-api-key` works
unchanged from the Mac through the tunnel.

## Step 1: create the notebook

1. **Create > New Notebook.**
2. In the right panel (**Session options**):
   - **Accelerator: GPU T4 x2.**
   - **Internet: On.**
   - Look at the **Persistence** option and note what it offers. Whether it can keep a 16 GB
     model between sessions will be decided after step 2.
3. Attach the `LLAMA_API_KEY` secret to this notebook (**Add-ons > Secrets**, tick it).

How a notebook runs commands:

- A cell runs Python. A line starting with `!` runs a shell command, the same idea as the
  `!` prefix in the Claude Code CLI. A cell starting with `%%bash` runs entirely in bash.
- A running cell blocks the notebook until it finishes. A server never finishes, so it will
  be started in the background later (`nohup ... > server.log 2>&1 &`) and followed through
  its log.

## Step 2: reconnaissance

Run this in the first cell:

```bash
!nvidia-smi
!nproc
!free -h
!df -h
!cat /etc/os-release | head -4
```

What each output decides:

| Command | What to read | What it decides |
|---|---|---|
| `nvidia-smi` | Two "Tesla T4" lines with ~15 GB each, and **CUDA Version** in the header | That both GPUs are there and free. The CUDA version is the maximum the driver supports, and it picks the llama.cpp build: `ubuntu-cuda-12.8-x64` if it is 12.8 or higher (below 13), `ubuntu-cuda-13.4-x64` if it is 13.4 or higher |
| `nproc` | Number of CPU cores | How fast anything CPU-bound runs (extracting archives, tokenizing) |
| `free -h` | Total RAM | The 16.46 GB GGUF is memory-mapped; if RAM is smaller, the system pages the file while loading it to the GPUs |
| `df -h` | Free space on `/`, `/kaggle/working` and any scratch area | Where the 16.46 GB GGUF can go. `/kaggle/working` is saved output with a cap (around 20 GB according to public sources), so a scratch area is preferred |
| `/etc/os-release` | Ubuntu version | Compatibility of the prebuilt binaries (built on Ubuntu 24.04 images) |

### Reconnaissance results (2026-10-22)

| Item | Value | Consequence |
|---|---|---|
| Quota | 30 h a week (0 used) | About 30 sessions of one hour; every idle minute counts |
| GPUs | 2x Tesla T4, 15,360 MiB each, idle | 30 GB of VRAM in total for weights plus KV cache |
| Driver | 580.178.04, **CUDA Version 13.0** | Use the **CUDA 12.8** build (see step 3) |
| CPU | 4 cores | Archive extraction and tokenizing are slow-ish |
| RAM | 31 GiB, no swap | The 16.46 GB GGUF fits in the page cache |
| Disk | `/` overlay with 1.1 TB free; `/dev/shm` 14 GB; `/kaggle/working` not a separate mount | Download the model to `/tmp` (root overlay), never to `/kaggle/working`, which is saved output |
| OS | Ubuntu 24.04.1 | Same base as the llama.cpp release builds |
| Persistence options | No persistence, Variables and Files, Variables only, Files only | Keep **No persistence** for now; re-downloading costs minutes, persisting 16 GB of output is not worth it until download time is measured |

## Step 3: install llama.cpp with CUDA

```bash
%%bash
set -e
TAG=b11819
BASE=https://github.com/ggml-org/llama.cpp/releases/download/$TAG
cd /tmp
curl -sSL -o llama.tar.gz  $BASE/llama-$TAG-bin-ubuntu-cuda-12.8-x64.tar.gz
curl -sSL -o cudart.tar.gz $BASE/cudart-llama-$TAG-bin-ubuntu-cuda-12.8-x64.tar.gz
mkdir -p /tmp/llama
tar -xzf llama.tar.gz  -C /tmp/llama --strip-components=1
tar -xzf cudart.tar.gz -C /tmp/llama --strip-components=1
/tmp/llama/llama-server --version
/tmp/llama/llama-server --list-devices
```

Why the CUDA 12.8 build and not 13.4, with a driver that reports CUDA 13.0:

- The release binaries carry the Turing architecture (sm_75, the T4) only as **PTX**, an
  intermediate GPU code that the driver compiles (JIT) on first use.
- A driver can only JIT-compile PTX produced by a toolkit version it knows. The 13.0 driver
  cannot compile PTX from the 13.4 toolkit; PTX from 12.8 is older and works.
- The `cudart` archive ships the CUDA runtime and cuBLAS libraries (594 MB) next to the
  binaries, which find them through their `$ORIGIN` rpath. Only the driver is taken from the
  system.

The tag is pinned (`b11819`) so the lab is reproducible; newer tags work the same way.

Result: done in under a minute. `--list-devices` reports `CUDA0` and `CUDA1`, Tesla T4,
14,911 MiB total and 14,806 MiB free each (`nvidia-smi` shows 15,360 MiB; the difference is
memory the driver reserves, ECC is enabled on these cards). The version line appears in red
because llama.cpp prints it to stderr, not because of an error.

## Step 4: download the model

```bash
%%bash
mkdir -p /tmp/models
cd /tmp/models
time curl -sSL -o Qwen3.8-27B-UD-Q4_K_M.gguf https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_M.gguf
ls -l
time sha256sum Qwen3.8-27B-UD-Q4_K_M.gguf
```

Expected: 16,464,440,224 bytes and SHA-256
`322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` (the LFS object id
published by Hugging Face). The download time decides whether notebook persistence is worth
it.

Result: download in 4 min 2 s (about 68 MB/s), size and SHA-256 match, hashing took 1 min
4 s on the 4 cores. Four minutes per session is about 0.2% of the weekly quota, so the lab
keeps downloading instead of persisting 16 GB. A private Kaggle Dataset holding the GGUF,
attached to the notebook, would remove the download entirely (to evaluate later).

## Step 5: start the server in the background

A Python cell, so the secret goes from Kaggle Secrets into the environment of the server
process without ever appearing on a command line:

```python
import os, subprocess
from kaggle_secrets import UserSecretsClient

os.environ["LLAMA_API_KEY"] = UserSecretsClient().get_secret("LLAMA_API_KEY")
os.environ["CUDA_CACHE_MAXSIZE"] = str(4 * 1024**3)  # keep JIT-compiled kernels for restarts

log = open("/tmp/server.log", "w")
server = subprocess.Popen(
    ["/tmp/llama/llama-server",
     "-m", "/tmp/models/Qwen3.8-27B-UD-Q4_K_M.gguf",
     "-ngl", "99", "-sm", "layer",
     "-c", "32768", "-np", "1",
     "--host", "127.0.0.1", "--port", "8080"],
    stdout=log, stderr=subprocess.STDOUT)
print("started, pid", server.pid)
```

Then wait for the health check, failing fast if the process dies:

```python
import time, urllib.request
start = time.time()
while True:
    if server.poll() is not None:
        print("server exited with code", server.returncode); break
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2)
        print(f"ready after {time.time() - start:.0f} s"); break
    except Exception:
        time.sleep(2)
```

And look at what happened:

```bash
!grep -iE "CUDA|offload|buffer size|split|n_ctx|error" /tmp/server.log | head -40
!nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

Result: the model was loaded 1 min 56 s after the process started (log timestamp of
`load_model: initializing`). With the default log verbosity the server does not print the
per-device buffers, so the split is read from `nvidia-smi`:

| GPU | Memory used |
|---|---|
| 0 | 8,389 MiB |
| 1 | 9,401 MiB |

About 17.4 GiB in total: 15.3 GiB of weights plus KV cache for 32k tokens, compute buffers
and the CUDA context on each card. GPU 1 holds more because the last layers and the output
projection (large, because of the vocabulary) live there. About 12 GiB remain free.

## Step 6: warm-up request and GPU activity

Start a background monitor that samples both GPUs every second:

```python
dmon = subprocess.Popen(["nvidia-smi", "dmon", "-s", "um", "-d", "1"],
                        stdout=open("/tmp/dmon.log", "w"), stderr=subprocess.STDOUT)
```

Send a first request (it pays for the JIT compilation of the GPU kernels) and time it:

```python
import json, time, urllib.request

def chat(prompt, max_tokens=300, think=False):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + os.environ["LLAMA_API_KEY"]})
    t = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    tm = r["timings"]
    print(f"{time.time() - t:.1f} s | prefill {tm['prompt_per_second']:.1f} tok/s "
          f"({tm['prompt_n']} tok) | decode {tm['predicted_per_second']:.1f} tok/s "
          f"({tm['predicted_n']} tok)")
    return r["choices"][0]["message"]

chat("Explain in three sentences why decode speed depends on memory bandwidth.")
chat("Explain in three sentences why decode speed depends on memory bandwidth.")
```

Then read the monitor:

```bash
!tail -n 25 /tmp/dmon.log
```

### What happened on the first attempt (and the fix)

The first request never reached the GPUs: the server log shows the task cancelled after
37 s, then `cleaning up before exit...`, and the server hung half-way through its shutdown
with the connection open, so the client waited until its 15-minute timeout. The most likely
trigger is a notebook interrupt: Jupyter sends the interrupt signal to the whole process
group of the kernel, and a server started with plain `Popen` belongs to it. The fix is
`start_new_session=True` in `Popen`, which puts the server in its own session (like `setsid`).

The PTX JIT compilation happened during the first model load (1 min 56 s, about 98 MB of
`/root/.nv/ComputeCache`), not during the request. After the relaunch the model loaded in
about 20 s (file already in the page cache) and the warm-up took 14 s.

Lesson on secrets: `env | grep -i proxy` printed a Kaggle session token
(`KAGGLE_DATA_PROXY_TOKEN`). Filter environment dumps by exact variable names, clear such
outputs and never save a notebook version that contains them.

### Results, `-sm layer`

| Request | Decode |
|---|---|
| Short prompt, 100 tokens | 13.6 tok/s |
| Same prompt, 95 tokens | 13.7 tok/s |

13.7 tok/s x 16.46 GB is about 225 GB/s of effective bandwidth, 70% of the 320 GB/s of one
T4. The video reports 16.2 tok/s **with multi-token prediction** (speculative decoding);
without it, 13.7 tok/s is consistent with that figure.

GPU activity during decode (`nvidia-smi dmon`, one sample per second): both GPUs at about
43 to 55% `sm` and 42 to 58% `mem`, never both near 100%. Each card works on its half of
the layers while the other waits: the relay race described in the video, in numbers.

## Step 7: the long prompt inside the notebook

`bench.py` is pasted into a cell whose first line is `%%writefile /tmp/bench.py` (running the
cell writes the file), and the key is passed through the environment:

```python
os.environ["BENCH_API_KEY"] = os.environ["LLAMA_API_KEY"]
```

```bash
!python3 /tmp/bench.py --prompt long --no-think --max-tokens 200
```

| Metric | Video | Measured |
|---|---|---|
| Prompt tokens | ~30,000 | 32,560 (the Qwen3.8 tokenizer and template produce more tokens than Qwen3-4B for the same text: 30,363 on the Mac) |
| Headers received | | 0.42 s (llama-server sends them immediately) |
| Prefill | ~325 tok/s | **453.5 tok/s** |
| TTFT | ~92 s | **72.2 s** |
| Decode at 32k context | | 12.2 tok/s (13.7 with an empty context) |

- The video's prefill figure is an estimate scaled from the T4 FLOPs; the measurement is 40%
  faster. Possible reasons: pipelining of prompt batches across both GPUs, the int8 tensor
  cores used for quantized matrix multiplications, and the hybrid attention (only 16 of 64
  layers use full attention), which degrades less with context than Qwen3-4B on the Mac.
- 72 s leaves 28 s of margin before the 100 s Quick Tunnel cut. With llama-server the cut
  would not apply anyway while streaming, because the headers arrive at once.
- Decode barely drops with 32k tokens of context (13.7 to 12.2 tok/s): the KV cache of the
  hybrid model is small compared to the 4B on the Mac (39 to 17 tok/s).
- The context was almost full (32,560 + 180 of 32,768 tokens). Larger prompts need a larger
  `-c`, and there is plenty of free VRAM for it.

GPU activity during this run confirms both regimes:

| Phase | GPU 0 `sm` / `mem` | GPU 1 `sm` / `mem` | Reading |
|---|---|---|---|
| Prefill | ~100% / ~20% | ~100% / ~20% | **Both GPUs busy at the same time**: prompt batches are pipelined across the cards. Compute-bound (`sm` high, `mem` low) |
| Decode | 40 to 57% / 37 to 56% | 41 to 60% / 41 to 63% | Relay race: each card idle half the time. Memory-bound (`mem` as high as `sm`) |

So layer split costs nothing in prefill and halves the hardware in decode. The video's
"capacity of two cards, speed of one" is true for decode only.

## Step 8: tensor split (`-sm tensor`)

Restart the server with the only change `-sm tensor` (each layer's matrices are split
between the cards, and both work on every token at the same time):

```python
server.terminate()
server.wait(timeout=60)
log = open("/tmp/server-tensor.log", "w")
server = subprocess.Popen([... same arguments, "-sm", "tensor" ...],
                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
```

The Qwen3.8 hybrid architecture supports it in llama.cpp `b11819`, and no noticeable extra
JIT compilation was needed (warm-up 1.1 s).

### Results, layer versus tensor

| Metric | `-sm layer` | `-sm tensor` | Change |
|---|---|---|---|
| Memory per GPU | 8,409 / 9,411 MiB | 8,821 / 8,821 MiB | Perfectly balanced |
| Decode, short prompt | 13.6 to 13.7 tok/s | 16.5 to 17.3 tok/s | **+23%** |
| Prefill, 32,560 tokens | 453.5 tok/s | 429.6 tok/s | -5% |
| TTFT, 32,560 tokens | 72.2 s | 76.2 s | +4 s |
| Decode at 32k context | 12.2 tok/s | 15.9 tok/s | **+30%** |
| Total time, long prompt + ~170 tokens | 86.9 s | 85.9 s | Even |
| GPU `sm` during decode | ~50% each, in turns | ~97% both at once | |
| GPU `mem` during decode | ~50% | 65 to 72% | |

### What the numbers teach

- **The second T4 does add decode speed with tensor parallelism**, but far from double:
  about 280 GB/s effective out of 640 GB/s, because after every one of the 64 layers both
  cards must combine their partial results (all-reduce) over PCIe, with no NVLink.
- **`sm` near 100% does not mean useful work.** Communication and waiting kernels count as
  busy time. The metric that matters is tokens per second.
- **Prefill barely suffers** (-5%), less than expected: the pipelining of layer split was
  already using both cards, and tensor split uses them too, paying some communication.
- **Which one to use depends on the workload (see below).** Prefill-heavy and one-shot: layer is
  marginally better. Generation-heavy, or agents whose long prefix is cached after the first
  request: tensor wins (+23 to 30% decode).

## Next steps

The plan, to be turned into exact commands once step 2 reports its numbers:

1. **First model load:** the PTX JIT compilation of the GPU kernels happens here and can
   take a few minutes. It is not a hang.
2. **Download the model** explicitly to `/tmp`:
   `curl -L -o /tmp/models/Qwen3.8-27B-UD-Q4_K_M.gguf https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_M.gguf`
   (16.46 GB). Not `-hf`, which caches on the root filesystem. To check later: whether the
   model is available as a Kaggle Model or Dataset that can be attached without downloading.
3. **Serve with `-sm layer`** (default), API key from the secret through the `LLAMA_API_KEY`
   environment variable so it never appears on the command line, and watch `nvidia-smi`
   while it decodes: one GPU busy, the other idle.
4. **Compare `-sm row` and `-sm tensor`.** This llama.cpp version has a real tensor-parallel
   mode; it may change the "capacity of two, speed of one" conclusion of the video.
5. **Measure twice:** once inside the notebook against localhost (`bench.py` pasted with
   `%%writefile`, since the repository is not on GitHub yet) and once from the Mac through a
   Quick Tunnel. That separates compute from network, the same discipline as phase 0.
6. **Expose with `cloudflared`** (Linux binary from the Cloudflare releases) and repeat the
   streaming versus no-streaming test.
7. **Confirm the Ollama header behavior** on the T4s, if quota allows.
8. **Stop the session.**

Reference values to confirm or refute (from the video): decode about 16 tok/s, prefill about
325 tok/s, about 92 s to the first token for a 30,000-token prompt.

## Step 9: expose the server and measure from the Mac

Install the same `cloudflared` version as on the Mac (single 40 MB static binary):

```bash
!curl -sSL -o /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/download/2026.10.0/cloudflared-linux-amd64 && chmod +x /tmp/cloudflared && /tmp/cloudflared --version
```

Start the Quick Tunnel in its own session and wait for the public URL in its log:

```python
import re
tunnel = subprocess.Popen(["/tmp/cloudflared", "tunnel", "--url", "http://127.0.0.1:8080"],
                          stdout=open("/tmp/tunnel.log", "w"), stderr=subprocess.STDOUT,
                          start_new_session=True)
url = None
for _ in range(60):
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open("/tmp/tunnel.log").read())
    if m:
        url = m.group(0); break
    time.sleep(1)
print(url)
```

From the Mac (the key in `.secrets/llama-api-key` is the same as the Kaggle secret):

```bash
export TUNNEL=https://<random-words>.trycloudflare.com
curl -s -o /dev/null -w "%{http_code}\n" $TUNNEL/v1/models     # expect 401
python3 scripts/bench.py --base-url $TUNNEL/v1 --key-file .secrets/llama-api-key --prompt short --no-think
python3 scripts/bench.py --base-url $TUNNEL/v1 --key-file .secrets/llama-api-key --prompt long --no-think --max-tokens 200 --no-cache
```

Close the tunnel when done: `tunnel.terminate()` in the notebook.

### Results through the tunnel (server with `-sm tensor`, client on the Mac in Madrid)

| Metric | Inside the notebook | From the Mac through the tunnel |
|---|---|---|
| `/v1/models` without key | | 401 |
| Short prompt: headers / TTFT | | 3.83 s / 3.83 s |
| Short prompt: decode | 16.5 to 17.3 tok/s | 16.0 tok/s (server 15.9) |
| Long prompt (32,560 tokens): headers | 0.41 s | **31.97 s** |
| Long prompt: TTFT | 76.2 s | 77.1 s |
| Long prompt: prefill (server) | 429.6 tok/s | 433.3 tok/s |
| Long prompt: decode | 15.9 tok/s | 15.7 tok/s (server 15.6) |

- **The tunnel adds about 2 to 3 seconds per request** (77.1 s of TTFT against 75.1 s of
  server prefill at 433 tok/s; 17.2 s against 13.9 s in the 6,544-token test below), and
  decode arrives at the generation rate: streaming works end to end through the Quick Tunnel.
- **A 32k-token agent prompt from home: first token after 77 s, then about 16 tok/s.** No
  524, because the response started before 100 s.
- **The headers no longer arrive immediately.** Inside the notebook llama-server sends them
  at 0.4 s; through the tunnel the client receives them at about 32 s (long prompt) or with
  the first token (short prompt). Something in the tunnel path holds the response start.
  Hypothesis: the response is released when the first body bytes arrive or after roughly
  30 s, whichever comes first.

Test of the hypothesis, with a prompt whose prefill stays well below 30 s:

```bash
python3 scripts/bench.py --base-url $TUNNEL/v1 --key-file .secrets/llama-api-key --prompt long --long-tokens 6000 --no-think --max-tokens 50 --no-cache
```

| Prompt | Server prefill time | Headers received | TTFT |
|---|---|---|---|
| 64 tokens | ~0.6 s | 3.83 s | 3.83 s |
| 6,544 tokens | ~13.9 s (470 tok/s) | 17.24 s | 17.24 s |
| 32,560 tokens | ~75 s | **31.97 s** | 77.08 s |

Below 30 s the headers arrive together with the first token; above 30 s they arrive at about
30 s plus the network time. The data supports the hypothesis (three points, the exact
threshold is not measured). Consequences:

- **Through a Quick Tunnel the effective rule is: the client sees nothing until the first
  token or about 30 s.** With llama-server the response always starts within that window,
  so the 100 s cut never triggers while streaming, whatever the prefill time.
- **Ollama would still be cut at 100 s**, because it holds even the headers until the first
  token (measured on the Mac in phase 0): there is nothing for the tunnel to release.
- Clients with their own short connection timeouts may give up before 30 s on long prompts,
  even though the server is working.
