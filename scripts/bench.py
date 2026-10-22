#!/usr/bin/env python3
"""Measure TTFT, prefill and decode speed against an OpenAI-compatible endpoint.

Standard library only, so it runs unchanged on the Mac, in a Kaggle notebook or
anywhere else. Uses streaming to measure time to first token from the client side,
and also prints the server-side `timings` object when the server is llama.cpp.

Usage:
    python3 scripts/bench.py --base-url http://localhost:8080/v1 --key-file .secrets/llama-api-key
    python3 scripts/bench.py --prompt long --max-tokens 200
    python3 scripts/bench.py --prompt long --no-stream --no-cache   # reproduce a proxy timeout
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# Filler used to build the long prompt. Repeated until the target size is reached.
FILLER = (
    "def process(record):\n"
    "    # Normalize the record and compute derived fields.\n"
    "    total = sum(item['price'] * item['qty'] for item in record['items'])\n"
    "    record['total'] = round(total, 2)\n"
    "    record['status'] = 'paid' if record.get('paid') else 'pending'\n"
    "    return record\n\n"
)

SHORT_PROMPT = (
    "You are helping a developer learn how LLM inference works. "
    "Explain the difference between the prefill phase and the decode phase, "
    "why prefill is compute-bound and decode is memory-bandwidth-bound, "
    "and how quantization changes decode speed. Use a short example."
)


def build_messages(kind, long_tokens):
    if kind == "short":
        return [{"role": "user", "content": SHORT_PROMPT}]
    # Roughly 4 characters per token for English text and code.
    target_chars = long_tokens * 4
    body = FILLER * (target_chars // len(FILLER) + 1)
    return [
        {"role": "system", "content": "You are a senior code reviewer.\n\n" + body[:target_chars]},
        {"role": "user", "content": "Summarize what the code above does in three bullet points."},
    ]


def run(args):
    key = args.key
    if args.key_file:
        with open(args.key_file) as f:
            key = f.readline().strip()

    stream = not args.no_stream
    payload = {
        "messages": build_messages(args.prompt, args.long_tokens),
        "max_tokens": args.max_tokens,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if args.no_cache:
        # llama.cpp specific: do not reuse the KV cache of a previous prompt.
        payload["cache_prompt"] = False
    if args.model:
        payload["model"] = args.model
    if args.no_think:
        # Qwen3 chat templates honor this flag to skip the thinking block.
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )

    start = time.perf_counter()
    first = None
    chunks = 0
    usage = timings = None
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        # urlopen returns as soon as the status line and headers arrive.
        headers_at = time.perf_counter()
        if not stream:
            body = json.loads(resp.read())
            usage, timings = body.get("usage"), body.get("timings")
        for raw in resp if stream else []:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage") or usage
            timings = event.get("timings") or timings
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    if first is None:
                        first = time.perf_counter()
                    chunks += 1
    end = time.perf_counter()

    print(f"prompt kind        : {args.prompt} ({'streaming' if stream else 'no streaming'})")
    print(f"headers received   : {headers_at - start:.2f} s")
    if usage:
        print(f"prompt tokens      : {usage.get('prompt_tokens')}")
        print(f"completion tokens  : {usage.get('completion_tokens')}")
    if first is not None:
        print(f"TTFT (client)      : {first - start:.2f} s")
        gen = usage.get("completion_tokens") if usage else chunks
        if gen and end > first:
            print(f"decode (client)    : {gen / (end - first):.1f} tok/s")
    print(f"total time         : {end - start:.2f} s")
    if timings:
        print(f"prefill (server)   : {timings.get('prompt_per_second', 0):.1f} tok/s")
        print(f"decode (server)    : {timings.get('predicted_per_second', 0):.1f} tok/s")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", default=os.environ.get("BENCH_BASE_URL", "http://localhost:8080/v1"))
    p.add_argument("--key", default=os.environ.get("BENCH_API_KEY", ""))
    p.add_argument("--key-file", help="file whose first line is the API key")
    p.add_argument("--model", help="model name, required by some servers (vLLM, OpenRouter)")
    p.add_argument("--prompt", choices=["short", "long"], default="short")
    p.add_argument("--long-tokens", type=int, default=30000, help="approximate size of the long prompt")
    p.add_argument("--max-tokens", type=int, default=500)
    p.add_argument("--no-think", action="store_true", help="disable Qwen3 thinking")
    p.add_argument("--no-stream", action="store_true", help="wait for the full response instead of streaming")
    p.add_argument("--no-cache", action="store_true", help="ask llama.cpp not to reuse a cached prompt prefix")
    p.add_argument("--timeout", type=float, default=600)
    start = time.perf_counter()
    try:
        run(p.parse_args())
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - start
        sys.exit(f"HTTP {e.code} after {elapsed:.1f} s: {e.read().decode(errors='replace')[:500]}")


if __name__ == "__main__":
    main()
