#!/usr/bin/env python3
"""Generate a workload-weighted self-distillation prompt corpus for a target
model (output: JSONL consumable by the syv stack's drafter/gen_data.py).

The corpus is the calibration source for everything downstream: the model's
generated outputs feed the draft vocabulary (vocab_build.py), the captured
hidden states calibrate GPTQ (capture.py + train_mtp.py --dump-hessians).
A corpus that does not look like your real workload produces a draft vocab
that does not cover your real outputs. Customise the mix — see
config/workload.example.json.

The default mix approximates a single-user coding-agent workload:
  - Rust implementation / review / refactor / unsafe / async   ~35%
  - TypeScript implementation / types / refactor               ~15%
  - Debugging (Go, Python, bash, C, Rust) with symptoms        ~12%
  - Code-edit tasks ("apply this change to the code")          ~10%
  - Agent / tool-use transcripts (tool results as context)      ~8%
  - Architecture / design / tradeoff questions                 ~10%
  - General technical reasoning                                 ~7%
  - Ordinary non-code requests                                  ~3%

Thinking ~60% on (a coding agent reasons on hard tasks; the target's
reasoning is cheap). Deterministic for a fixed seed.

Usage:
  python make_corpus.py --out prompts.jsonl [--config workload.json]
                        [--n 5500] [--seed 20260915]
"""
import argparse, json, os, random, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "..", "config", "workload.example.json")
R = random.Random(20260915)  # reassigned in main() from --seed

CRATES = ["tokio", "axum", "serde", "clap", "reqwest", "sqlx", "rayon", "crossbeam", "anyhow", "thiserror",
          "tracing", "hyper", "tonic", "prost", "rand", "chrono", "uuid", "tarpc", "dashmap", "parking_lot"]
RUST_DOMAINS = ["a CLI tool", "an HTTP API service", "a background worker", "a parser", "a small embedded daemon",
                "a library crate", "a test harness", "a data pipeline stage", "a websocket server", "a scheduler"]
TS_DOMAINS = ["a React component library", "a Node.js API", "a CLI tool", "a VS Code extension backend",
              "a state management layer", "a fetch wrapper", "a form validation module", "a streaming client"]
ITEMS = ["configuration", "retry logic", "error handling", "pagination", "caching layer", "rate limiting",
         "health checks", "graceful shutdown", "streaming responses", "background jobs", "idempotency keys",
         "connection pooling", "schema migration", "input validation", "observability hooks"]
LANGS_DEBUG = ["Go", "Python", "Bash", "C", "Rust", "TypeScript"]
BUG_KINDS = ["an off-by-one error", "a nil/None dereference", "a race condition", "a resource leak",
             "an unhandled error path", "an infinite loop on empty input", "a wrong comparison operator",
             "a missing lock", "a shadowed variable", "a incorrect integer division",
             "a deadlock between two mutexes", "a stale cache entry"]

def rust_impl():
    crate = R.choice(CRATES); dom = R.choice(RUST_DOMAINS); item = R.choice(ITEMS)
    style = R.random()
    if style < 0.4:
        return (f"Implement {item} for {dom} in Rust using {crate}. Define the public types and function "
                f"signatures first, then the implementation with unit tests. Use idiomatic error handling.")
    if style < 0.7:
        sig = f"pub fn {R.choice(['process','handle','load','validate','encode','merge','diff'])}_" \
              f"{R.choice(['batch','stream','entries','records'])}(input: &{R.choice(['str','[u8]','Path','Vec<Record>'])}) -> Result<{R.choice(['Vec<Output>','Output','HashMap<String,u64>','()'])}>"
        return (f"Write a Rust function `{sig}` for {dom}. Include the type definitions it needs, "
                f"doc comments, and a doctest. {R.choice(['Use iterators, avoid indexing.', 'Keep it allocation-free where possible.', 'Make it work for empty input.', 'Handle non-UTF8 input gracefully.'])}")
    return (f"Design and implement a small Rust module for {item} in {dom}. Show the file layout, the public "
            f"API, and one integration test. Assume {crate} is available.")

def rust_review():
    n = R.randint(2, 5)
    code = f"```rust\nfn {R.choice(['sum_sizes','collect_ids','fold_counts','scan_lines'])}(items: &Vec<Item>) -> u64 {{\n"
    code += "    let mut total = 0;\n"
    for _ in range(n):
        code += f"    total += items.iter().filter(|i| i.{R.choice(['size','count','len','weight'])} > {R.randint(0, 100)}).count() as u64;\n"
    code += "    total\n}\n```"
    asks = ["point out correctness bugs", "suggest idiomatic improvements", "review the API design",
            "rewrite it with iterators", "explain what it does, then critique it"]
    return f"Review this Rust code and {R.choice(asks)}:\n\n{code}"

def rust_refactor():
    pat = R.choice(["repeated Vec::contains (O(n^2))", "unwrap() everywhere", "clone-heavy ownership",
                    "manual retry loops", "string-based dispatch", "one giant function"])
    return (f"Refactor a Rust function that suffers from {pat} so it is idiomatic and efficient. "
            f"First show a representative 'before' sketch, then the 'after' version, and explain each change briefly.")

def rust_snippet_task():
    tasks = [
        "Write a Rust macro that logs a struct's fields with tracing at debug level.",
        "Implement a bounded async channel wrapper in Rust that drops the oldest item when full.",
        "Write a Rust function that walks a directory and streams every file's first line as an iterator.",
        "Implement a type-safe builder in Rust that only allows build() after all required fields are set.",
        "Write a Rust integration test that spins up a temporary HTTP server and exercises a client against it.",
        "Implement Display for a tree structure in Rust with indentation and connector characters.",
        "Write a zero-copy parser for a simple `key=value;` header format in Rust using &[u8].",
        "Implement a small LRU cache in Rust using only std collections, with O(1) get and put.",
    ]
    return R.choice(tasks) + " Include tests."

def rust_prompt():
    r = R.random()
    if r < 0.45: return rust_impl()
    if r < 0.6: return rust_review()
    if r < 0.75: return rust_refactor()
    return rust_snippet_task()

def ts_prompt():
    r = R.random()
    if r < 0.4:
        return (f"Implement {R.choice(ITEMS)} for {R.choice(TS_DOMAINS)} in TypeScript. Use strict types, "
                f"no `any`, and include a couple of Vitest tests.")
    if r < 0.6:
        g = R.choice(["<T extends object>", "<K extends string, V>", "<A extends unknown[]>",
                      "<T extends { id: string }>", "<E extends Error>"])
        return (f"Write a TypeScript generic function {g} that {R.choice(['debounces','memoizes','retries with backoff','validates','deep-clones','partially applies'])} "
                f"its input. Explain the type-level decisions.")
    if r < 0.8:
        return ("Migrate this JavaScript to strictly-typed TypeScript, preserving behaviour:\n\n"
                "```js\n" + R.choice([
                    "function groupBy(list, key) { return list.reduce((m, x) => { (m[x[key]] ||= []).push(x); return m; }, {}); }",
                    "async function fetchAll(urls, limit) { const out = []; let i = 0; async function next() { if (i >= urls.length) return; const u = urls[i++]; out.push(await fetch(u).then(r => r.json()).catch(() => null)); await next(); } await Promise.all(Array.from({length: limit}, next)); return out; }",
                    "const throttle = (fn, ms) => { let last = 0; return (...a) => { const now = Date.now(); if (now - last > ms) { last = now; fn(...a); } }; }",
                ]) + "\n```")
    return ("Design the TypeScript types for an event bus that supports typed payloads per event name, "
            "compile-time checking of subscriptions, and async handlers. Show usage examples.")

def debug_prompt():
    lang = R.choice(LANGS_DEBUG); bug = R.choice(BUG_KINDS)
    symptom = R.choice([
        "under load it sometimes panics", "it intermittently returns wrong results",
        "it hangs after a few hours", "the CI test flakes about 1 run in 10",
        "it works locally but fails in production", "memory grows steadily until the OOM killer fires",
    ])
    if lang == "Bash":
        code = ("```bash\n#!/bin/bash\nset -e\nfor f in $(find . -name '*.log'); do\n  grep ERROR $f | "
                "mail -s \"errors\" team@example.com\ndone\n```")
        return f"This bash script has {bug} and {symptom}. Diagnose it precisely and show the minimal fix:\n\n{code}"
    if lang == "Go":
        code = ("```go\nvar mu sync.Mutex\nvar cache = map[string]*Entry{}\nfunc Get(k string) *Entry {\n"
                "    mu.Lock()\n    e := cache[k]\n    if e == nil {\n        e = load(k)\n        cache[k] = e\n"
                "    }\n    mu.Unlock()\n    return e\n}\nfunc load(k string) *Entry { /* hits network */ }\n```")
        return f"This Go code has {bug}; {symptom}. Identify every problem precisely and show the fix:\n\n{code}"
    if lang == "Python":
        code = ("```python\ndef process(items):\n    results = []\n    for i in range(len(items)):\n"
                "        if items[i] and items[i-1].get('ok'):\n            results.append(items[i]['value'] / items[i].get('div', 0))\n    return results\n```")
        return f"This Python function contains {bug} and {symptom}. Find the defects, explain them, and fix:\n\n{code}"
    if lang == "C":
        code = ("```c\nchar *join(const char *a, const char *b) {\n    char *out = malloc(strlen(a) + strlen(b));\n"
                "    strcpy(out, a); strcat(out, b); return out;\n}\n```")
        return f"This C function has {bug}; {symptom}. Diagnose and fix it, explaining the memory semantics:\n\n{code}"
    if lang == "Rust":
        code = ("```rust\nlet (tx, rx) = mpsc::channel();\nfor w in 0..4 {\n    let tx = tx.clone();\n"
                "    std::thread::spawn(move || { while let Ok(j) = rx.recv() { tx.send(process(j)).ok(); } });\n}\n```")
        return f"This Rust snippet exhibits {bug} and {symptom}. Explain the root cause precisely and fix it:\n\n{code}"
    code = ("```ts\nconst users = await Promise.all(ids.map(id => db.user.findFirst({ where: { id } })));\n"
            "const names = users.map(u => u.profile.displayName);\n```")
    return f"This TypeScript has {bug}; {symptom}. Diagnose and fix:\n\n{code}"

def edit_prompt():
    fn = R.choice(['sum_sizes', 'collect_ids', 'fold_counts', 'scan_lines', 'merge_flags'])
    change = R.choice([
        "return Result<u64, SumError> instead of panicking on overflow",
        "take a slice instead of a Vec reference",
        "also accept an optional limit parameter",
        "make it generic over the item type using a trait",
        "log each skipped item at trace level",
    ])
    code = (f"```rust\nfn {fn}(items: &[Item]) -> u64 {{\n    items.iter().map(|i| i.size as u64).sum()\n}}\n```")
    return f"Apply this change to the code below: {change}. Show the full modified version and nothing else.\n\n{code}"

def agent_prompt():
    tool = R.choice(["read_file", "grep_search", "run_tests", "list_dir", "edit_file", "bash"])
    n = R.randint(1, 3)
    results = []
    for i in range(n):
        results.append({"role": "tool", "content": R.choice([
            f"[{tool}] src/main.rs lines 1-40:\n```rust\nfn main() {{\n    let args: Vec<String> = std::env::args().collect();\n    let cfg = Config::parse(&args)?;\n    run(cfg)\n}}\n```",
            f"[{tool}] 3 matches for 'unwrap()' in src/db.rs:\n  41: let row = stmt.query_row(...).unwrap();\n  88: let conn = pool.get().unwrap();\n  130: let v: i64 = row.get(0).unwrap();",
            f"[{tool}] cargo test --lib: 14 passed, 2 failed:\n  failures:\n    test_parse_empty ... panicked at 'index out of bounds'",
            f"[{tool}] src/ contains: main.rs, config.rs, db.rs, errors.rs, tests/",
            f"[{tool}] git diff --stat: 4 files changed, 51 insertions(+), 12 deletions(-)",
        ])})
    goal = R.choice([
        "Continue fixing the failing tests. What is your next step and why?",
        "Diagnose why the tests fail and propose the minimal fix.",
        "Summarise the current state of the task and what remains.",
        "Refactor the unwraps you found into proper error handling. Show the diff.",
        "Decide which file to read next to understand the config flow, and explain.",
    ])
    msgs = [{"role": "system", "content": "You are a coding agent working in a Rust repository. Use tools when needed; be concise."}]
    msgs.append({"role": "user", "content": "Please " + R.choice([
        "fix the failing tests in the parser module.",
        "make the database layer stop panicking on connection errors.",
        "clean up the unwrap calls before we ship.",
        "add a config validation step at startup.",
    ])})
    msgs.extend(results)
    msgs.append({"role": "user", "content": goal})
    return msgs

def arch_prompt():
    qs = [
        "We need to choose between Postgres and SQLite for a single-node Rust service with modest write volume. Compare them for this use case and recommend one.",
        "Design the versioning and rollout strategy for a public HTTP API that will have breaking changes twice a year.",
        "Should a background job queue live in the same Postgres as the app data, or in Redis? Argue both sides, then decide.",
        "How should a CLI tool structure configuration precedence (flags, env, config file, defaults)? Show a design.",
        "Compare REST, gRPC and a simple JSON-over-HTTP for an internal service called by two other services. Recommend.",
        "Design a schema for multi-tenant invoice storage with per-tenant custom fields. Discuss trade-offs.",
        "When is a monorepo the wrong choice? What breaks first?",
        "How would you add idempotency to a payment-taking endpoint? Cover retries, duplicates and auditing.",
        "Argue for and against feature flags over long-lived branches for a 4-person team.",
        "Design a graceful-degradation strategy for a service that depends on three flaky upstreams.",
        "What belongs in the application layer vs the database (constraints, triggers, views)? Take a position.",
        "Sketch a deployment topology for a hobby-scale but observable stack: one app, one worker, one DB, metrics.",
    ]
    return R.choice(qs) + " Be concrete; prefer a recommendation over a survey."

def reasoning_prompt():
    qs = [
        "Explain what speculative decoding loses and gains, precisely. When is it a net loss?",
        "Why do B-trees remain competitive with LSM trees for read-heavy workloads? Explain the mechanics.",
        "Walk through what happens, step by step, when a TLS connection is established to an HTTP/2 server.",
        "Explain the difference between memory ordering guarantees in Rust's atomics as if to a senior engineer.",
        "How does a write-ahead log make crash recovery possible? Trace a torn write through recovery.",
        "Explain tail latency: why does p99 behave so differently from the mean, and what actually fixes it?",
        "What exactly does 'zero-copy' mean in the context of network file transfer? Trace the buffers.",
        "Explain how incremental compilation can be correct — what invalidation guarantees are needed?",
        "Why is quicksort's worst case rare in practice but a real problem for adversarial input?",
        "Explain the trade-offs of per-request allocation vs arena allocation in a request handler.",
    ]
    return R.choice(qs)

def general_prompt():
    qs = [
        "Draft a short release note for a CLI tool update that adds a --json flag and fixes two crashes.",
        "Summarise the pros and cons of working async-first as a small team, in three bullets.",
        "Write a friendly rejection reply to a feature request that will not be implemented.",
        "Give three concrete tips for keeping a personal knowledge base useful over years.",
        "Explain this to a non-technical manager: why did the deploy take the site down for 40 seconds?",
        "Rewrite this sentence three ways, more politely each time: 'your patch broke the build'.",
        "What is a good weekly review structure for a solo developer? Keep it under 150 words.",
        "Recommend a naming scheme for build servers in a small lab, with examples.",
    ]
    return R.choice(qs)

GENERATORS = {  # category -> prompt generator (kept in sync with the config)
    "rust": rust_prompt, "ts": ts_prompt, "debug": debug_prompt, "edit": edit_prompt,
    "agent": agent_prompt, "arch": arch_prompt, "reason": reasoning_prompt,
    "general": general_prompt,
}

def make(i, weights, think_p):
    r = R.random()
    acc = 0.0
    cat = None
    for c, w in weights:              # weighted dispatch over the config order
        acc += w
        if r < acc:
            cat = c
            break
    if cat is None:
        cat = weights[-1][0]
    msgs = GENERATORS[cat]()
    think = R.random() < think_p.get(cat, 0.5)
    if isinstance(msgs, str):
        msgs = [{"role": "user", "content": msgs}]
    return {"id": f"{cat}-{i}", "src": cat, "messages": msgs, "think": bool(think)}

def main():
    ap = argparse.ArgumentParser(description="workload-weighted prompt corpus generator")
    ap.add_argument("--out", required=True, help="output prompts.jsonl path")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="workload mix JSON")
    ap.add_argument("--n", type=int, default=5500, help="number of prompts")
    ap.add_argument("--seed", type=int, default=20260915, help="RNG seed")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    weights = [(c["category"], c["weight"]) for c in cfg["categories"]]
    tot = sum(w for _, w in weights)
    if abs(tot - 1.0) > 1e-6:
        sys.exit(f"config error: category weights sum to {tot}, expected 1.0")
    think_p = {c["category"]: c.get("think", 0.5) for c in cfg["categories"]}
    unknown = [c for c, _ in weights if c not in GENERATORS]
    if unknown:
        sys.exit(f"config error: unknown categories {unknown}; "
                 f"available: {sorted(GENERATORS)}")

    global R
    R = random.Random(a.seed)
    prompts = [make(i, weights, think_p) for i in range(a.n)]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    c = Counter(p["src"] for p in prompts)
    t = sum(p["think"] for p in prompts)
    print("total", len(prompts), "think", t, f"({t/len(prompts):.0%})")
    for k, v in c.most_common():
        print(f"  {k:8s} {v:5d} ({v/len(prompts):.0%})")
    print("wrote", a.out)

if __name__ == "__main__":
    main()
