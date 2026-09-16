#!/usr/bin/env python3
"""Quality sanity battery: identical prompts to Swift and Qwen, capture
side-by-side outputs with mechanical checks where cheap (maths answers,
JSON validity, tool-call shape, key code tokens) and raw text for human
reading. Not a benchmark: correctness evidence for the token reduction.

Usage (run once per backend, against its /v1 base):
  KEY=... BASE=http://127.0.0.1:8092/v1 LABEL=swift THINKING=1 \
    python3 quality_ab.py results/swift-ab/swift-quality.json
"""
import json, os, sys, time, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8092/v1")
KEY = os.environ.get("KEY", "")
LABEL = os.environ.get("LABEL", "backend")
MODEL = os.environ.get("MODEL", "qwen3.8-27b")
THINKING = os.environ.get("THINKING", "1") == "1"
# Qwen's instruct recommendation drives both arms (identical sampling);
# Swift's own card recommends T=1.0/top_p 0.95/top_k 20 — set via env for a
# second Swift arm if wanted.
TEMP = float(os.environ.get("TEMP", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.8"))

TASKS = [
    {
        "id": "rust_lru",
        "prompt": "Write a Rust struct `LruCache<K, V>` with `new(cap)`, `get(&K) -> Option<&V>` and `put(K, V)`, using only std (HashMap + VecDeque or your own list). Include one doctest that inserts 3 items into a cap-2 cache and shows the oldest evicted.",
        "check": "rust code with struct LruCache and fn get/put",
        "must_contain": ["struct LruCache", "fn get", "fn put"],
    },
    {
        "id": "ts_debounce",
        "prompt": "Write a TypeScript function `debounce<A extends unknown[]>(fn: (...args: A) => void, ms: number): (...args: A) => void` with a `cancel()` method attached to the returned function, and a correct `this`-free implementation. No libraries.",
        "check": "ts code with debounce generic and cancel",
        "must_contain": ["debounce", "cancel", "setTimeout"],
    },
    {
        "id": "debug_race",
        "prompt": "This Go function is meant to count unique visitors per hour, but under load it sometimes panics with 'concurrent map writes' and sometimes undercounts. Diagnose both bugs precisely and show the minimal fix:\n\n```go\nvar visitors = map[int64]int{}\nvar unique = map[int64]bool{}\nfunc Visit(userID int64) {\n    h := time.Now().Unix() / 3600\n    visitors[h]++\n    unique[userID] = true\n}\n```",
        "check": "identifies unsynchronised map access; fix uses mutex or sync.Map",
        "must_contain": [],
    },
    {
        "id": "math_series",
        "prompt": "Sum all positive three-digit numbers whose digits are strictly decreasing (e.g. 931, 520). Show the working briefly, then give the sum as a single integer on the last line.",
        "check": "correct sum 92610",
        "answer_contains": "92610",
    },
    {
        "id": "math_prob",
        "prompt": "A fair coin is flipped until two consecutive heads appear. What is the expected number of flips? Show the derivation briefly, then the exact value as a fraction on the last line.",
        "check": "expected value 6",
        "answer_contains": "6",
    },
    {
        "id": "instr_json",
        "prompt": "Return ONLY a JSON object (no prose, no code fence) with keys \"name\" (string), \"ports\" (array of exactly 3 integers), and \"active\" (boolean true). The name must be \"acme\".",
        "check": "parses as JSON with the exact schema",
        "json_schema": {"name": "acme", "ports_len": 3, "active": True},
    },
    {
        "id": "instr_neg",
        "prompt": "List exactly five words that each contain the letter 'q' but not the letter 'u'. No explanations, no numbering — just the five words, one per line.",
        "check": "five lines, each word contains q not u",
    },
]

TOOLS_SPEC = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the repository",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "start_line": {"type": "integer"}},
                       "required": ["path"]},
    },
}]

def chat(body_extra, messages, max_tokens=1024):
    body = {
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": TEMP, "top_p": TOP_P,
        "chat_template_kwargs": {"enable_thinking": THINKING},
        **body_extra,
    }
    req = urllib.request.Request(
        BASE + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    msg = out["choices"][0]["message"]
    usage = out.get("usage", {})
    return {
        "content": msg.get("content") or "",
        "reasoning": (msg.get("reasoning_content") or msg.get("reasoning") or ""),
        "tool_calls": [{"name": tc["function"]["name"],
                        "args": tc["function"]["arguments"]}
                       for tc in (msg.get("tool_calls") or [])],
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "wall_s": round(time.time() - t0, 1),
        "finish": out["choices"][0].get("finish_reason"),
    }
def check(task, r):
    errs = []
    low = r["content"].lower()
    for s in task.get("must_contain", []):
        if s.lower() not in low:
            errs.append(f"missing {s!r}")
    if "answer_contains" in task:
        want = task["answer_contains"]
        tail = " ".join(r["content"].split()[-12:]) + " " + (r["reasoning"] or "")[-200:]
        if want not in tail and want not in tail.replace(",", ""):
            errs.append(f"answer {want!r} not found near end")
    if "json_schema" in task:
        try:
            j = json.loads(r["content"].strip().strip("`").removeprefix("json").strip())
            if j.get("name") != "acme": errs.append("json name wrong")
            if not isinstance(j.get("ports"), list) or len(j["ports"]) != 3: errs.append("ports != 3")
            if j.get("active") is not True: errs.append("active not true")
        except Exception as e:
            errs.append(f"json parse: {e}")
    if task["id"] == "instr_neg":
        lines = [l.strip().strip(".").strip() for l in r["content"].strip().splitlines() if l.strip()]
        words = [l for l in lines if l.isalpha()]
        if len(words) != 5: errs.append(f"{len(words)} words, not 5")
        for w in words:
            if "q" not in w.lower() or "u" in w.lower(): errs.append(f"bad word {w!r}")
    return errs

def main(out_path):
    results = {"label": LABEL, "base": BASE, "thinking": THINKING,
               "sampling": {"temperature": TEMP, "top_p": TOP_P}, "tasks": []}
    # warm-up
    try:
        chat({}, [{"role": "user", "content": "Say OK."}], 8)
    except Exception as e:
        print("warm-up failed:", e); sys.exit(1)

    for t in TASKS:
        budget = 8192 if t["id"] in ("rust_lru", "ts_debounce", "debug_race", "math_series") else 1536
        r = chat({}, [{"role": "user", "content": t["prompt"]}], budget)
        errs = check(t, r)
        row = {"id": t["id"], "check": t["check"], "errors": errs,
               "pass": not errs, **r}
        results["tasks"].append(row)
        print(f"  {t['id']}: {'PASS' if not errs else 'FAIL ' + '; '.join(errs)} "
              f"(reasoning {r['reasoning_tokens']} tok, {r['wall_s']}s)", file=sys.stderr)

    # tool calling
    r = chat({"tools": TOOLS_SPEC, "tool_choice": "auto"},
             [{"role": "user", "content": "Read the file src/main.rs from line 10 for me."}], 512)
    ok = (len(r["tool_calls"]) == 1 and r["tool_calls"][0]["name"] == "read_file"
          and "path" in json.loads(r["tool_calls"][0]["args"]))
    results["tasks"].append({"id": "tool_call", "check": "read_file(path) parsed",
                             "errors": [] if ok else ["tool call wrong"], "pass": ok, **r})
    print(f"  tool_call: {'PASS' if ok else 'FAIL'}", file=sys.stderr)

    # streaming
    body = {"model": MODEL, "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": 64, "temperature": TEMP, "top_p": TOP_P,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": "Count 1 to 5, digits only, one per line."}]}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    chunks, usage, ttft = 0, {}, None
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            p = line[5:].strip()
            if p == "[DONE]": break
            c = json.loads(p)
            if c.get("usage"): usage = c["usage"]
            if any(ch.get("delta", {}).get("content") for ch in c.get("choices") or []):
                chunks += 1
                if ttft is None: ttft = time.time() - t0
    ok = chunks >= 3 and usage.get("completion_tokens")
    results["tasks"].append({"id": "streaming", "check": ">=3 content chunks + usage",
                             "errors": [] if ok else [f"chunks={chunks} usage={usage}"],
                             "pass": ok, "content": f"{chunks} chunks, ttft {ttft and round(ttft,3)}s",
                             "completion_tokens": usage.get("completion_tokens")})
    print(f"  streaming: {'PASS' if ok else 'FAIL'}", file=sys.stderr)

    npass = sum(1 for t in results["tasks"] if t["pass"])
    results["summary"] = f"{npass}/{len(results['tasks'])} PASS"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"{results['summary']} -> {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else f"results/swift-ab/{LABEL}-quality.json")
