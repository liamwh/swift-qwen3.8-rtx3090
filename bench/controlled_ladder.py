#!/usr/bin/env python3
"""Controlled, interleaved Swift 1.0 / 1.5 ladder.

Why this exists: the 2026-10-02 03:12 ladder ran its legs once each in a
fixed order (1.5 int8, 1.5 fast, 1.0 fast, ...), so leg order and night drift
were confounded with the 1.0-vs-1.5 gap it was meant to measure. This runner
boots one configuration per leg, in an order given on the command line, so the
caller can interleave (A B B A, B A A B) and run no-speculation controls and
vocabulary variants in the same session.

Per leg (one `docker compose up --force-recreate` boot):
  1. boot, wait for /health, record the boot time
  2. warm-up: one pass of the suite (discarded)
  3. measured suite at T=0.7 / top_p 0.8 (the existing protocol, seeds
     1000+rep come from swift_ab.py)
  4. measured suite at T=0 (deterministic sampling, same prompts)
  5. quote workload (--quote only; legs where acceptance on verbatim copying
     matters)
A 1 Hz nvidia-smi sampler (SM clock, memory clock, temperature, power, VRAM,
utilisation, throttle reasons) runs through steps 3 to 5.

Rows go to results/profiles/controlled-ladder.jsonl (one row per measured
suite, label "<leg>-t07" or "<leg>-t0", same shape as the older ladders so
bench/analyze_runs.py reads them), raw per-request JSON to
results/profiles/controlled/.

The API key is read with `sops -d` inside this process and handed to docker
compose and the harness through the environment. It is never printed.

The daily deployment is restored in a `finally` block with
`systemctl --user restart swift-qwen.service` unless --keep is given (use
--keep on all but the last invocation of a multi-stage session).

Leg spec: <leg-id>:<config>[+quote], e.g. c01:t10 c02:t15+quote
Configs: see CONFIGS below.
"""
import argparse
import csv
import datetime
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

HOME = os.path.expanduser("~")
REPO = f"{HOME}/git/qwen38-27b-rtx3090"
INFRA = f"{HOME}/git/infra"
OVERRIDE = f"{INFRA}/hosts/zeus/swift-qwen/compose.override.yaml"
BENCH = f"{INFRA}/hosts/zeus/llm-bench"
PROFILES = f"{BENCH}/results/profiles"
RAW = f"{PROFILES}/controlled"
LADDER = f"{PROFILES}/controlled-ladder.jsonl"
KEYFILE = f"{INFRA}/hosts/zeus/secrets/syv-vllm/api-key"
BASE = "http://127.0.0.1:8092"
MODEL_ROOT = "/app/models"

M10 = "Swift-Qwen3.8-27B-W4A16-syv-fast"
M15 = "Swift-1.5-Qwen3.8-27B-W4A16-syv-fast"
M15_INT8 = "Swift-1.5-Qwen3.8-27B-W4A16-syv"

# config -> (model dir, extra container environment). Everything else comes
# from compose.override.yaml (SPEC=mtp, CTX=long, MAX_LEN=114688, GPU_UTIL=0.85).
CONFIGS = {
    "t10": (M10, {}),                       # 1.0 target, 1.0 list (published build)
    "t15": (M15, {}),                       # 1.5 target, 1.5 list (current artefact)
    "t10off": (M10, {"SPEC": "off"}),
    "t15off": (M15, {"SPEC": "off"}),
    "i15": (M15_INT8, {}),                  # 1.5 int8 heads, base-Qwen list
    "v10": (f"{M15}-vocab10", {}),          # 1.5 target, 1.0 list
    "vu": (f"{M15}-vocabunion", {}),        # 1.5 target, union list
    "x15": (f"{M10}-vocab15", {}),          # 1.0 target, 1.5 list
    "xu": (f"{M10}-vocabunion", {}),        # 1.0 target, union list
}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sh(cmd, **kw):
    return subprocess.run(cmd, check=False, text=True, capture_output=True, **kw)


def get_key():
    r = sh(["sops", "-d", "--output-type", "binary", KEYFILE])
    if r.returncode != 0 or not r.stdout:
        sys.exit("could not decrypt the API key with sops (stderr suppressed on purpose)")
    return r.stdout.strip()


def http(path, key, body=None, timeout=10):
    req = urllib.request.Request(BASE + path, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    if body is not None:
        req.data = json.dumps(body).encode()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def healthy(key, port_base=BASE):
    try:
        return http("/health", key, timeout=4)[0] == 200
    except Exception:
        return False


def wait_healthy(key, seconds):
    end = time.time() + seconds
    while time.time() < end:
        if healthy(key):
            return True
        time.sleep(5)
    return False


def boot(key, model, env):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, prefix="controlled-ladder-")
    tmp.write("services:\n  swift:\n    environment:\n")
    full = {"MODEL": f"{MODEL_ROOT}/{model}", "VERIFY": "0", **env}
    for k, v in full.items():
        tmp.write(f'      {k}: "{v}"\n')
    tmp.close()
    t0 = time.time()
    e = {**os.environ, "VLLM_API_KEY": key}
    sh(["docker", "compose", "--project-name", "qwen38-swift", "--project-directory", ".", "--file", "docker-compose.yml",
        "--file", OVERRIDE, "--file", tmp.name, "--profile", "swift", "up", "-d", "--force-recreate"], cwd=REPO, env=e)
    os.unlink(tmp.name)
    ok = wait_healthy(key, 20 * 60)
    return ok, time.time() - t0


class GpuSampler:
    FIELDS = "timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu,clocks_throttle_reasons.active"

    def __init__(self, path):
        self.path = path
        self.proc = None

    def start(self):
        self.fh = open(self.path, "w")
        self.proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits", "-l", "1"], stdout=self.fh)

    def stop(self):
        self.proc.terminate()
        self.proc.wait()
        self.fh.close()
        rows = []
        for r in csv.reader(open(self.path)):
            try:
                rows.append([x.strip() for x in r])
            except Exception:
                pass
        busy = [r for r in rows if float(r[6]) >= 50] or rows
        f = lambda i, rs: [float(r[i]) for r in rs]
        return {
            "samples": len(rows),
            "busy_samples": len(busy),
            "sm_clock_mhz_median": statistics.median(f(1, busy)),
            "sm_clock_mhz_min": min(f(1, busy)),
            "mem_clock_mhz_median": statistics.median(f(2, busy)),
            "temp_c_max": max(f(3, rows)),
            "temp_c_median_busy": statistics.median(f(3, busy)),
            "power_w_mean_busy": statistics.mean(f(4, busy)),
            "power_w_max": max(f(4, rows)),
            "vram_mib_peak": max(f(5, rows)),
            "throttle_reasons_seen": sorted({r[7] for r in busy}),
        }


def harness(key, label, mode, out, reps, temp):
    env = {**os.environ, "LABEL": label, "BASE": BASE + "/v1", "KEY": key, "MODEL": "qwen3.8-27b", "REPS": str(reps), "THINKING": "0", "TEMP": str(temp)}
    r = subprocess.run([sys.executable, f"{BENCH}/swift_ab.py", mode, out], env=env, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"swift_ab.py {mode} failed: {r.stderr[-400:]}")
    return json.load(open(out))


def med(runs, k):
    v = [r[k] for r in runs if r.get(k) is not None]
    return round(statistics.median(v), 2) if v else None


def run_leg(key, order, leg, config, quote, reps):
    model, env = CONFIGS[config]
    os.makedirs(RAW, exist_ok=True)
    meta = {"leg": leg, "order": order, "config": config, "model_dir": model, "env": env, "start_utc": now()}
    ok, boot_s = boot(key, model, env)
    meta["boot_s"] = round(boot_s, 1)
    if not ok:
        meta["error"] = "server did not become healthy"
        with open(LADDER, "a") as f:
            f.write(json.dumps({"label": f"{leg}-boot", **meta}) + "\n")
        print(f"[{leg}] boot failed", flush=True)
        return False
    digest = sh(["docker", "inspect", "--format", "{{.Image}}", "qwen38-swift-swift-1"]).stdout.strip()
    meta["container_image_id"] = digest
    # warm-up pass (discarded), identical for every leg
    harness(key, f"{leg}-warm", "suite", "/tmp/controlled-warm.json", 1, 0.7)
    sampler = GpuSampler(f"{RAW}/{leg}-gpu.csv")
    sampler.start()
    try:
        res = {}
        for tag, temp in (("t07", 0.7), ("t0", 0.0)):
            res[tag] = harness(key, f"{leg}-{tag}", "suite", f"{PROFILES}/{leg}-{tag}-suite.json", reps, temp)
        q = harness(key, f"{leg}-quote", "quote", f"{PROFILES}/{leg}-quote.json", reps, 0.0) if quote else None
    finally:
        gpu = sampler.stop()
    meta.update({"end_utc": now(), "gpu": gpu})
    for tag in ("t07", "t0"):
        s = res[tag]
        row = {"label": f"{leg}-{tag}", **meta, "sampling": s.get("sampling"), "reps": s["reps"],
               "ttft_s_med": med(s["runs"], "ttft_s"), "decode_tps_med": med(s["runs"], "decode_tps"),
               "e2e_tps_med": med(s["runs"], "e2e_tps"), "acceptance_suite": s.get("acceptance")}
        if q and tag == "t07":
            row.update({"quote_decode": [round(r["decode_tps"], 1) for r in q["runs"]],
                        "quote_char_match": [round(r["char_match"], 4) for r in q["runs"]],
                        "quote_exact": [r.get("exact") for r in q["runs"]],
                        "acceptance_quote": q.get("acceptance")})
        with open(LADDER, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{leg}] {tag}: median {row['decode_tps_med']} tok/s acceptance {(s.get('acceptance') or {}).get('acceptance_rate')}", flush=True)
    return True


def restore(key):
    print("restoring swift-qwen.service ...", flush=True)
    sh(["systemctl", "--user", "restart", "swift-qwen.service"])
    if not wait_healthy(key, 25 * 60):
        print("RESTORE FAILED: swift-qwen.service not healthy; check journalctl --user -u swift-qwen", flush=True)
        return False
    status, body = http("/v1/chat/completions", key, {"model": "qwen3.8-27b", "max_tokens": 24, "messages": [{"role": "user", "content": "Reply with the single word ready."}], "chat_template_kwargs": {"enable_thinking": False}}, timeout=60)
    ok = status == 200 and json.loads(body)["choices"][0]["message"].get("content") is not None
    print(f"restore smoke chat: {'ok' if ok else 'FAILED'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("legs", nargs="+", help="<leg-id>:<config>[+quote]")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--keep", action="store_true", help="do not restore the daily deployment afterwards")
    args = ap.parse_args()
    key = get_key()
    done = set()
    if os.path.exists(LADDER):
        done = {json.loads(l)["label"].rsplit("-", 1)[0] for l in open(LADDER) if l.strip()}
    used = sh(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]).stdout.strip()
    print(f"[{now()}] GPU utilisation {used}% at start", flush=True)
    try:
        for i, spec in enumerate(args.legs, 1):
            leg_cfg, _, flag = spec.partition("+")
            leg, config = leg_cfg.split(":")
            if leg in done:
                print(f"[{leg}] already measured, skipping", flush=True)
                continue
            print(f"[{now()}] leg {i}/{len(args.legs)}: {leg} = {config}{' +quote' if flag else ''}", flush=True)
            order = len(done) + 1
            try:
                run_leg(key, order, leg, config, bool(flag), args.reps)
            except Exception as e:  # keep going: a bad variant must not strand the session
                print(f"[{leg}] FAILED: {type(e).__name__}: {e}", flush=True)
            done.add(leg)
    finally:
        if not args.keep:
            restore(key)


if __name__ == "__main__":
    main()
