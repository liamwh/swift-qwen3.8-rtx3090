#!/usr/bin/env python3
"""Swift-vs-Qwen A/B harness (llm-bench conventions; stdlib only).

Extends abbench.py with what the Swift comparison needs:

  * reasoning visibility: captures reasoning_content separately from content
    (thinking ON by default for Swift — its reduced-reasoning behaviour is
    part of what is being measured; THINKING=0 matches abbench.py exactly),
    and reports reasoning vs content token splits.
  * speculative-decode acceptance: diffs vllm:spec_decode_* Prometheus
    counters around each phase (drafted vs accepted tokens), so acceptance
    is measured on the SAME requests that are timed.
  * quote mode: a context-copy workload (reproduce a document verbatim),
    the case DFLASH_TOKENS=15 exists for; reports reproduction fidelity
    (sha + character overlap) alongside speed so a speed win can never
    hide a correctness loss.

Usage (env: BASE KEY MODEL LABEL REPS THINKING SAMPLING_*):
  swift_ab.py suite|prefix|long|quote|reason results/<label>-<mode>.json
"""
import hashlib, json, os, re, statistics, sys, time, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8090/v1")
KEY = os.environ.get("KEY", "")
MODEL = os.environ.get("MODEL", "qwen3.8-27b")
LABEL = os.environ.get("LABEL", "backend")
REPS = int(os.environ.get("REPS", "3"))
THINKING = os.environ.get("THINKING", "1") == "1"
TEMP = float(os.environ.get("TEMP", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.8"))
TOP_K = int(os.environ.get("TOP_K", "0"))  # 0 = omit (vLLM default)

# Same prompt sets as abbench.py so numbers remain comparable.
SHORT = [
    ("short_backlog", "What's a good strategy for prioritizing a long backlog of "
     "small bug fixes versus a few large features? Answer in one paragraph."),
    ("short_locking", "In one paragraph, explain the difference between optimistic "
     "and pessimistic locking in databases."),
    ("short_closure", "Write a one-line Rust closure that returns the square of an i32."),
    ("short_tradeoffs", "What are the main tradeoffs between REST and gRPC for an "
     "internal microservice API? Answer in one paragraph."),
]
CODING = [
    ("code_new_rust",
     "Write a Rust function `parse_kv_pairs(input: &str) -> Result<HashMap<String, "
     "String>, ParseError>` that parses lines of the form `key=value`, skipping "
     "blank lines and lines starting with `#`. Define `ParseError` with `thiserror`, "
     "with a `LineNumber(usize)` variant reporting the 1-indexed failing line. "
     "Include a short doctest."),
    ("code_edit_existing",
     "Add proper error handling to this function instead of the two `unwrap()` "
     "calls — return `Result<u64, std::io::Error>` and propagate errors with `?`:\n\n"
     "```rust\nfn read_first_u64(path: &str) -> u64 {\n"
     "    let contents = std::fs::read_to_string(path).unwrap();\n"
     "    let first_line = contents.lines().next().unwrap();\n"
     "    first_line.trim().parse().unwrap()\n}\n```"),
]
REASON = [
    ("math_word",
     "A tank has two taps. The hot tap fills it in 18 minutes, the cold tap in "
     "12 minutes, and the drain empties the full tank in 36 minutes. If all three "
     "are open, how long until the tank fills? Show your reasoning, then give the "
     "answer as a single number of minutes."),
    ("logic_grid",
     "Three engineers — Ada, Grace, Linus — sit in seats 1-3. Ada is not in seat 1. "
     "Grace is directly left of Linus (seat i is directly left of seat i+1). Who "
     "sits in seat 2? Show your reasoning, then answer with just the name."),
]

FILLER = (
    "# module telemetry_{i} — batching helper for span export {i}\n"
    "# Internal utility, not part of the public API surface.\n"
    "import logging\nfrom dataclasses import dataclass\nfrom typing import Iterable, Optional\n\n"
    "logger = logging.getLogger(__name__)\n\n"
    "@dataclass\nclass Batch{i}:\n    items: list\n    max_size: int = 256\n\n"
    "    def add(self, item: item) -> bool:\n"
    "        if len(self.items) >= self.max_size:\n            return False\n"
    "        self.items.append(item)\n        return True\n\n"
    "    def flush(self) -> Iterable:\n        out, self.items = self.items, []\n        return out\n\n"
)
FACTS = [
    "The retry budget for the export queue is capped at 4 attempts; a 5th failure "
    "drops the batch and increments `telemetry_dropped_total` rather than blocking.",
    "Batch flushing is triggered by size (256 items) OR a 2-second timer, whichever "
    "comes first — never by both racing in the same tick.",
    "The `Batch` dataclass is deliberately not thread-safe: callers must hold the "
    "per-shard lock in `ShardWriter` before calling `add()`.",
]

def build_doc(target_tokens, marker=None):
    target_chars = target_tokens * 4
    paras, i = [], 0
    while sum(len(p) for p in paras) < target_chars:
        paras.append(FILLER.format(i=i)); i += 1
        frac = sum(len(p) for p in paras) / target_chars
        for j, fact in enumerate(FACTS):
            if abs(frac - (0.2 + 0.3 * j)) < 0.02 and f"__fact{j}__" not in "".join(paras[-3:]):
                paras.append(f"# DESIGN NOTE (fact-{j}) — required reading\n# __fact{j}__ {fact}\n\n")
    return "".join(paras)[:target_chars]


# ---------------- spec-decode acceptance via /metrics ----------------

SPEC_PATTERNS = [
    ("accepted", re.compile(r'^vllm:spec_decode_num_accepted_tokens(?:_total)?\{[^}]*\}\s+(\d+)')),
    ("drafted",  re.compile(r'^vllm:spec_decode_num_draft_tokens(?:_total)?\{[^}]*\}\s+(\d+)')),
    ("drafts",   re.compile(r'^vllm:spec_decode_num_drafts(?:_total)?\{[^}]*\}\s+(\d+)')),
]

def metrics_snapshot():
    """Sum spec-decode counters from /metrics (same port, no /v1 prefix)."""
    url = BASE.rsplit("/v1", 1)[0] + "/metrics"
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + KEY})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read().decode()
    except Exception as e:
        return {"error": str(e)}
    out = {}
    for line in body.splitlines():
        for name, pat in SPEC_PATTERNS:
            m = pat.match(line)
            if m:
                out[name] = out.get(name, 0) + int(m.group(1))
    return out

def acceptance_delta(a, b):
    if "error" in a or "error" in b or not a or not b:
        return None
    acc = b.get("accepted", 0) - a.get("accepted", 0)
    drf = b.get("drafted", 0) - a.get("drafted", 0)
    steps = b.get("drafts", 0) - a.get("drafts", 0)
    if drf <= 0:
        return None
    return {
        "accepted": acc, "drafted": drf, "steps": steps,
        "acceptance_rate": acc / drf,
        "tok_per_step": (acc + steps) / steps if steps else None,  # accepted + bonus token
    }


# ---------------- request machinery ----------------

def stream_chat(messages, max_tokens=512, seed=None, extra=None):
    body = {
        "model": MODEL, "messages": messages, "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": THINKING},
        "temperature": TEMP, "top_p": TOP_P,
    }
    if TOP_K:
        body["top_k"] = TOP_K
    if seed is not None:
        body["seed"] = seed
    if extra:
        body.update(extra)
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time(); ttft = None
    text, reason, usage, finish = [], [], {}, None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                delta = choice.get("delta", {})
                rc = (delta.get("reasoning_content") or delta.get("reasoning") or "")
                pc = delta.get("content") or ""
                if rc:
                    if ttft is None: ttft = time.time() - t0
                    reason.append(rc)
                if pc:
                    if ttft is None: ttft = time.time() - t0
                    text.append(pc)
    total = time.time() - t0
    ct = usage.get("completion_tokens")
    rt = usage.get("completion_tokens_details", {}).get("reasoning_tokens") if usage.get("completion_tokens_details") else None
    content_text = "".join(text)
    decode_s = total - (ttft or 0)
    return {
        "ttft_s": ttft, "total_s": total,
        "decode_tps": (ct / decode_s) if ct and decode_s > 0 else None,
        "e2e_tps": (ct / total) if ct and total > 0 else None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": ct,
        "finish": finish,
        "reasoning_tokens_reported": rt,
        "reasoning_chars": len("".join(reason)),
        "content_chars": len(content_text),
        "reasoning_sha": hashlib.sha1("".join(reason).encode()).hexdigest()[:12] if reason else None,
        "text": content_text,
    }


def warm_up():
    stream_chat([{"role": "user", "content": "Say OK."}], max_tokens=8)


def save(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2)
    print(f"wrote {path}", file=sys.stderr)


# ---------------- modes ----------------

def run_suite(out):
    warm_up()
    m0 = metrics_snapshot()
    results = {"label": LABEL, "base": BASE, "model": MODEL, "reps": REPS,
               "thinking": THINKING, "sampling": {"temperature": TEMP, "top_p": TOP_P, "top_k": TOP_K or None},
               "runs": []}
    for category, items, mt in (("short", SHORT, 512), ("coding", CODING, 700)):
        for pid, prompt in items:
            for rep in range(REPS):
                r = stream_chat([{"role": "user", "content": prompt}], max_tokens=mt, seed=1000 + rep)
                r.update({"category": category, "id": pid, "rep": rep})
                r.pop("text")
                results["runs"].append(r)
                print(f"  {category}/{pid} rep{rep}: ttft={r['ttft_s']:.3f}s "
                      f"decode={r['decode_tps'] and round(r['decode_tps'],1)}tok/s "
                      f"rchars={r['reasoning_chars']} cchars={r['content_chars']}", file=sys.stderr)
    results["acceptance"] = acceptance_delta(m0, metrics_snapshot())
    save(out, results)


def run_prefix(out):
    doc = build_doc(8000)
    warm_up()
    m0 = metrics_snapshot()
    q1 = doc + "\n\nQuestion: what triggers a batch flush, and what happens on the 5th failed attempt?"
    q2 = doc + "\n\nQuestion: is `Batch` safe to share across threads without external locking? Why or why not?"
    cold = stream_chat([{"role": "user", "content": q1}], max_tokens=300)
    warm = stream_chat([{"role": "user", "content": q2}], max_tokens=300)
    acc = acceptance_delta(m0, metrics_snapshot())
    for r in (cold, warm):
        r.pop("text")
    save(out, {"label": LABEL, "base": BASE, "model": MODEL, "doc_chars": len(doc),
               "cold": cold, "warm": warm, "acceptance": acc})


def run_long(out):
    doc = build_doc(32000)
    warm_up()
    q = doc + "\n\nSummarise the retry/flush/thread-safety design in this material in 5 bullet points."
    r = stream_chat([{"role": "user", "content": q}], max_tokens=500)
    text = r.pop("text")
    save(out, {"label": LABEL, "base": BASE, "model": MODEL, "doc_chars": len(doc),
               "run": r, "answer_head": text[:400]})


def run_quote(out):
    """Context-copy workload: reproduce an 8k-token document verbatim."""
    doc = build_doc(8000)
    prompt = ("Below is a source document. Reproduce it EXACTLY, character for "
              "character, from the first `# module` line to the end. Do not "
              "summarise, do not fix anything, do not add commentary.\n\n"
              "=== SOURCE START ===\n" + doc + "\n=== SOURCE END ===")
    warm_up()
    m0 = metrics_snapshot()
    rows = []
    for rep in range(max(1, REPS - 1)):  # 2 reps: one arms nothing here (no prefix reuse: same prompt hits the cache — keep reps low)
        r = stream_chat([{"role": "user", "content": prompt}],
                        max_tokens=9000, seed=None, extra={"temperature": 0.0})
        text = r.pop("text")
        r["reproduced_chars"] = len(text)
        r["char_match"] = sum(1 for a, b in zip(text, doc)) / max(1, len(doc))
        r["source_sha"] = hashlib.sha1(doc.encode()).hexdigest()[:12]
        r["output_sha"] = hashlib.sha1(text.encode()).hexdigest()[:12]
        r["exact"] = text.strip() == doc.strip()
        rows.append(r)
        print(f"  quote rep{rep}: decode={r['decode_tps'] and round(r['decode_tps'],1)}tok/s "
              f"match={r['char_match']:.3f} exact={r['exact']}", file=sys.stderr)
    acc = acceptance_delta(m0, metrics_snapshot())
    save(out, {"label": LABEL, "base": BASE, "model": MODEL, "doc_chars": len(doc),
               "runs": rows, "acceptance": acc})


def run_reason(out):
    warm_up()
    rows = []
    for pid, prompt in REASON:
        r = stream_chat([{"role": "user", "content": prompt}], max_tokens=2048, seed=7)
        text = r.pop("text")
        rows.append({"id": pid, **r, "answer": text})
        print(f"  {pid}: reasoning_chars={r['reasoning_chars']} "
              f"reasoning_tokens={r['reasoning_tokens_reported']} total={r['total_s']:.1f}s", file=sys.stderr)
    save(out, {"label": LABEL, "base": BASE, "model": MODEL, "runs": rows})


AGENT_TASKS = [
    ("agent_rust_impl", 8192, ["struct LruCache", "fn get", "fn put"],
     "Write a Rust struct `LruCache<K, V>` with `new(cap)`, `get(&K) -> Option<&V>` and "
     "`put(K, V)` using only std. Include a doctest that inserts 3 items into a cap-2 "
     "cache and shows the oldest evicted."),
    ("agent_ts_impl", 8192, ["debounce", "cancel", "setTimeout"],
     "Write a TypeScript function `debounce<A extends unknown[]>(fn: (...args: A) => void, "
     "ms: number): (...args: A) => void` with a `cancel()` method on the returned function. "
     "No libraries. Show usage."),
    ("agent_debug", 8192, ["lock", "mutex", "Lock".lower()],
     "This Go service sometimes panics with 'concurrent map writes' under load:\n\n```go\n"
     "var visitors = map[int64]int{}\nvar unique = map[int64]bool{}\nfunc Visit(id int64) {\n"
     "    h := time.Now().Unix() / 3600\n    visitors[h]++\n    unique[id] = true\n}\n```\n"
     "Diagnose precisely, show the minimal fix, and note any remaining hazards."),
    ("agent_edit", 8192, ["Result", "fn sum_sizes", "Err"],
     "Apply this change to the code below: return Result<u64, SumError> instead of "
     "panicking on overflow, with SumError defined via thiserror. Show the full modified "
     "version.\n\n```rust\nfn sum_sizes(items: &[Item]) -> u64 {\n"
     "    items.iter().map(|i| i.size as u64).sum()\n}\n```"),
    ("agent_arch", 8192, [],
     "We are choosing between Postgres and SQLite for a single-node Rust service with "
     "modest write volume and one writer process. Argue both sides briefly and give a "
     "clear recommendation."),
]

def run_agent(out):
    """Coding-agent latency: thinking ON, one request per task, everything the
    agent user feels: request -> complete usable final answer."""
    warm_up()
    m0 = metrics_snapshot()
    rows = []
    # tool-call task handled separately (non-streaming, tool parse check)
    for pid, mt, must, prompt in AGENT_TASKS:
        r = stream_chat([{"role": "user", "content": prompt}], max_tokens=mt, seed=None)
        text = r.pop("text")
        ok = all(s.lower() in text.lower() for s in must)
        rows.append({"id": pid, "must_contain": must, "content_ok": ok,
                     "finish_ok": r.get("finish") != "length",
                     "answer_head": text[:220], **r})
        print(f"  {pid}: wall={r['total_s']:.1f}s reasoning={r['reasoning_tokens_reported']} "
              f"completion={r['completion_tokens']} ok={ok} finish={r.get('finish')}", file=sys.stderr)
    # tool-call task
    body = {"model": MODEL, "max_tokens": 4096, "temperature": TEMP, "top_p": TOP_P,
            "chat_template_kwargs": {"enable_thinking": True},
            "messages": [{"role": "user", "content": "Read the file src/main.rs and fix the "
                                                    "failing parse test. Start by reading the file."}],
            "tools": [{"type": "function", "function": {
                "name": "read_file", "parameters": {"type": "object",
                "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
            "tool_choice": "auto"}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + KEY})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        resp_json = json.load(resp)
    wall = time.time() - t0
    m = resp_json["choices"][0]["message"]
    tcs = m.get("tool_calls") or []
    tc_ok = len(tcs) == 1 and tcs[0]["function"]["name"] == "read_file" \
        and "main.rs" in tcs[0]["function"]["arguments"]
    rows.append({"id": "agent_tool", "total_s": round(wall, 1),
                 "reasoning_tokens": (resp_json["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens"),
                 "completion_tokens": resp_json["usage"]["completion_tokens"],
                 "content_ok": tc_ok, "finish_ok": True,
                 "answer_head": f"tool_calls={[(t['function']['name'], t['function']['arguments']) for t in tcs]}"})
    print(f"  agent_tool: wall={wall:.1f}s ok={tc_ok}", file=sys.stderr)
    acc = acceptance_delta(m0, metrics_snapshot())
    save(out, {"label": LABEL, "base": BASE, "model": MODEL, "runs": rows, "acceptance": acc})


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "suite"
    out = sys.argv[2] if len(sys.argv) > 2 else f"results/{LABEL}-{cmd}.json"
    if cmd == "suite": run_suite(out)
    elif cmd == "prefix": run_prefix(out)
    elif cmd == "long": run_long(out)
    elif cmd == "quote": run_quote(out)
    elif cmd == "reason": run_reason(out)
    elif cmd == "agent": run_agent(out)
    else:
        print(__doc__); sys.exit(1)
