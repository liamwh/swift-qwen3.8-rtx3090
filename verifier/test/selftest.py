#!/usr/bin/env python3
"""Verifier selftest: builds toy model dirs (fixtures.py, stdlib only) and
asserts verify_model.py passes the good fixture and fails each corruption.

The stale-collision fixture is the regression test for the historical
fast-variant bug: a hardlinked model-nonquant shard still carrying int8 MTP
packed tensors collided with the int4 ones and passed every existence check.

Run: python3 verifier/test/selftest.py
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
VERIFY = os.path.join(HERE, "..", "verify_model.py")
from fixtures import N_DRAFT, build_dir, clean  # noqa: E402



def run_verify(D, geo, *extra):
    r = subprocess.run([sys.executable, VERIFY, D, "--mtp-geometry", geo, *extra],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main():
    root = tempfile.mkdtemp(prefix="verify-selftest-")
    failures = 0
    try:
        cases = [
            ("good int4 fast layout passes", dict(), 0),
            ("good int8 layout passes (auto-detect)", dict(lm_bits=8, mtp_bits=8), 0),
            ("stale duplicate packed tensor fails (the historical bug)",
             dict(stale_collision=True), 1),
            ("config/tensor bit-width mismatch fails", dict(bits_mismatch=True), 1),
            ("draft head rows != ids count fails", dict(draft_rows=N_DRAFT + 8), 1),
            ("duplicate draft ids fail", dict(dup_ids=True), 1),
            ("missing mapped shard fails", dict(drop_shard=True), 1),
            ("--expect int8 against an int4 dir fails", dict(), 1, ["--expect", "int8"]),
        ]
        for case in cases:
            name, kwargs, want_rc = case[0], case[1], case[2]
            extra = case[3] if len(case) > 3 else []
            sub = os.path.join(root, name.replace(" ", "_").replace("/", "-"))
            os.makedirs(sub, exist_ok=True)
            D, geo = build_dir(sub, **kwargs)
            rc, out = run_verify(D, geo, *extra)
            got = "PASS-exit0" if rc == 0 else f"FAIL-exit{rc}"
            verdict = "ok " if rc == want_rc else "BAD"
            if rc != want_rc:
                failures += 1
            print(f"  {verdict}  {name}: verifier exit {rc} (want {want_rc})")
            if rc != want_rc:
                print("\n".join("      " + l for l in out.splitlines()[-12:]))
    finally:
        clean(root)
    if failures:
        print(f"selftest: {failures} case(s) misbehaved")
        sys.exit(1)
    print("selftest: all cases behaved")


if __name__ == "__main__":
    main()
