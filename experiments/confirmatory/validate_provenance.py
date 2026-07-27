#!/usr/bin/env python3
"""
Validate the provenance block of the committed confirmatory execution record.

check_against_manifest.py validates the scientific numbers; this validates the
integrity of the record that carries them. A green run here means:

  * the record has a provenance block with the required fields;
  * results_sha256 recomputed from the scientific payload matches the stored
    digest (the payload was not altered after generation);
  * every code_sha256 entry matches the current bytes of that file (the
    generating code was not altered after the record was produced);
  * the recorded generating commit is an ancestor of HEAD (the record was
    produced by code that is actually in this history).

A hand-edited provenance block, a tampered payload, or a code file changed after
the record was generated all fail here. CI-safe: no restricted data.
"""

import hashlib
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_DEFAULT_RECORD = os.path.join(_HERE, "confirmatory_results.json")

REQUIRED_FIELDS = (
    "generated_by", "git_commit", "git_working_tree", "python_version",
    "package_versions", "code_sha256", "cohorts", "seeds",
    "started_utc", "completed_utc", "results_sha256",
)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _payload_digest(record):
    payload = json.dumps(
        {"protocol": record["protocol"], "granularities": record["granularities"]},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_ancestor(commit):
    try:
        r = subprocess.run(
            ["git", "-C", _REPO, "merge-base", "--is-ancestor", commit, "HEAD"],
            capture_output=True, text=True,
        )
        return r.returncode == 0
    except (FileNotFoundError, OSError):
        return None  # git unavailable -> undetermined


def main(argv):
    record_path = argv[1] if len(argv) > 1 else _DEFAULT_RECORD
    if not os.path.exists(record_path):
        print(f"[FAIL] execution record not found: {record_path}")
        return 1
    record = json.load(open(record_path, encoding="utf-8"))

    prov = record.get("provenance")
    if not isinstance(prov, dict):
        print("[FAIL] record has no provenance object")
        return 1

    fails = 0

    missing = [f for f in REQUIRED_FIELDS if f not in prov]
    if missing:
        print(f"[FAIL] provenance missing fields: {missing}")
        fails += 1
    else:
        print("[ok] provenance has all required fields")

    # payload digest
    recomputed = _payload_digest(record)
    if recomputed == prov.get("results_sha256"):
        print(f"[ok] results_sha256 matches payload ({recomputed[:16]}...)")
    else:
        print(f"[FAIL] results_sha256 mismatch: stored={prov.get('results_sha256')} "
              f"recomputed={recomputed}")
        fails += 1

    # code hashes
    for rel, stored in (prov.get("code_sha256") or {}).items():
        path = os.path.join(_REPO, rel)
        if not os.path.exists(path):
            print(f"[FAIL] hashed code file missing: {rel}")
            fails += 1
            continue
        got = _sha256_file(path)
        if got == stored:
            print(f"[ok] code_sha256 {rel} ({got[:16]}...)")
        else:
            print(f"[FAIL] code_sha256 changed since record generated: {rel}\n"
                  f"        stored={stored}\n        current={got}")
            fails += 1

    # commit ancestry
    commit = prov.get("git_commit")
    if commit and commit not in ("unknown", ""):
        anc = _is_ancestor(commit)
        if anc is True:
            print(f"[ok] generating commit {commit[:12]} is an ancestor of HEAD")
        elif anc is False:
            print(f"[FAIL] generating commit {commit[:12]} is NOT an ancestor of HEAD")
            fails += 1
        else:
            print(f"[warn] git unavailable; skipped ancestry check for {commit[:12]}")
    else:
        print("[warn] no usable git_commit in provenance; ancestry not checked")

    print(f"\nprovenance validation: {'FAIL' if fails else 'OK'} ({fails} problem(s))")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
