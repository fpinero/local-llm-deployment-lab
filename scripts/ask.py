#!/usr/bin/env python3
"""Ask the same question several times and print every answer, to judge a model by hand.

Sampling is random, so one answer proves little. This script repeats the question with
the same sampling settings (Qwen3 recommendations for thinking mode by default) and shows
each final answer plus how many tokens the model spent. Standard library only.

Usage:
    python3 scripts/ask.py --key-file .secrets/llama-api-key --runs 5
    python3 scripts/ask.py --question "Which is heavier, 1 kg of iron or 1 kg of feathers?"
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

CAR_WASH = (
    "I have to wash the car; the car wash is 150 meters from home. "
    "Would you recommend going by car or on foot?"
)


def ask(args, key):
    payload = {
        "messages": [{"role": "user", "content": args.question}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": 0.0,
    }
    if args.model:
        payload["model"] = args.model
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        return json.loads(resp.read())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", default=os.environ.get("BENCH_BASE_URL", "http://localhost:8080/v1"))
    p.add_argument("--key", default=os.environ.get("BENCH_API_KEY", ""))
    p.add_argument("--key-file", help="file whose first line is the API key")
    p.add_argument("--model", help="model name, required by some servers (Ollama, vLLM)")
    p.add_argument("--question", default=CAR_WASH, help="defaults to the car wash trick question")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--answer-chars", type=int, default=400, help="characters of each answer to show")
    p.add_argument("--timeout", type=float, default=600)
    args = p.parse_args()

    key = args.key
    if args.key_file:
        with open(args.key_file) as f:
            key = f.readline().strip()

    print(f"question: {args.question}")
    print(f"sampling: temperature={args.temperature} top_p={args.top_p} top_k={args.top_k}\n")
    for i in range(1, args.runs + 1):
        start = time.perf_counter()
        try:
            body = ask(args, key)
        except urllib.error.HTTPError as e:
            sys.exit(f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")
        elapsed = time.perf_counter() - start
        msg = body["choices"][0]["message"]
        answer = " ".join((msg.get("content") or "").split())
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        tokens = body.get("usage", {}).get("completion_tokens")
        print(f"--- run {i}/{args.runs}: {tokens} output tokens, {len(reasoning)} reasoning chars, {elapsed:.1f} s")
        print(answer[: args.answer_chars] + ("..." if len(answer) > args.answer_chars else ""))
        print()


if __name__ == "__main__":
    main()
