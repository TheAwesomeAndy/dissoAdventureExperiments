#!/usr/bin/env python3
"""
Generate a release manifest that binds the frozen artifacts together.
====================================================================

At freeze time this records, in one signed-by-content document, the exact
state a reviewer must be able to retrieve: the git commit, and SHA-256
checksums of the dissertation source, the dissertation PDF, the results
manifest, the citation/license files, and a single digest over the entire
figure set. Regenerate it on the exact commit you tag (see RELEASE.md), then
commit the output so the tag carries its own integrity record.

    python scripts/make_release_manifest.py --out RELEASE_MANIFEST.json

The output is deterministic given the working tree, so re-running on the same
commit reproduces byte-identical checksums.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.check_output(["git", "-C", _REPO, *args],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def hash_tree(rel_dir, exts=None):
    """Return (combined_digest, file_count, total_bytes) over a directory.
    The combined digest hashes the sorted list of "relpath:sha256" lines, so it
    changes if any file's content or the set of files changes."""
    root = os.path.join(_REPO, rel_dir)
    entries = []
    total = 0
    for dirpath, _dirs, fnames in os.walk(root):
        for fn in sorted(fnames):
            if exts and os.path.splitext(fn)[1].lower() not in exts:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, _REPO)
            digest = sha256_file(full)
            entries.append(f"{rel}:{digest}")
            total += os.path.getsize(full)
    entries.sort()
    combined = hashlib.sha256("\n".join(entries).encode()).hexdigest()
    return combined, len(entries), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="RELEASE_MANIFEST.json")
    ap.add_argument("--pdf", default=None,
                    help="path to the final dissertation PDF to checksum")
    args = ap.parse_args()

    # Individual files to bind (skip any that are absent)
    single_files = [
        "results_manifest.json", "README.md", "CITATION.cff", "LICENSE",
        "LICENSES.md", "docs/REPRODUCTION_MAP.md",
        "docs/VERIFICATION_METHODOLOGY.md",
    ]
    # dissertation source
    diss_dir = os.path.join(_REPO, "dissertation")
    if os.path.isdir(diss_dir):
        for fn in sorted(os.listdir(diss_dir)):
            if fn.endswith((".tex", ".bib")):
                single_files.append(f"dissertation/{fn}")

    # candidate PDFs
    pdf_candidates = []
    if args.pdf:
        pdf_candidates.append(args.pdf)
    for c in ("ARSPI-Net_Defense.pdf",):
        if os.path.exists(os.path.join(_REPO, c)):
            pdf_candidates.append(c)

    files = {}
    for rel in single_files + pdf_candidates:
        full = os.path.join(_REPO, rel)
        if os.path.exists(full):
            files[rel] = {"sha256": sha256_file(full),
                          "bytes": os.path.getsize(full)}

    fig_digest, fig_count, fig_bytes = hash_tree("pictures")

    manifest = {
        "artifact": "ARSPI-Net frozen release manifest",
        "git_commit": git("rev-parse", "HEAD"),
        "git_describe": git("describe", "--tags", "--always", "--dirty"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "files": dict(sorted(files.items())),
        "figures": {
            "combined_sha256": fig_digest,
            "file_count": fig_count,
            "total_bytes": fig_bytes,
        },
        "note": ("Binds the dissertation source, PDF, results manifest, and the "
                 "entire figure set to one git commit. Regenerate on the tagged "
                 "commit; re-running on the same commit reproduces identical "
                 "checksums. Verifies integrity, not scientific validity."),
    }

    out = os.path.join(_REPO, args.out) if not os.path.isabs(args.out) else args.out
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {out}")
    print(f"  commit   : {manifest['git_commit']}")
    print(f"  files    : {len(files)} checksummed")
    print(f"  figures  : {fig_count} files, combined {fig_digest[:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
