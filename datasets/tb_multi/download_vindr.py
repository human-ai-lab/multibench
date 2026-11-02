"""Selective, parallel download of VinDr-CXR 1.0.0 (PhysioNet, credentialed access) for use as an external TB cohort.

The full release is 18,000 DICOMs; only ~2,100 are needed (all TB positives plus a sample of negatives).
Labels are radiologist read-outs, not microbiology: train images carry 3 independent reads, test images a consensus.
  positive : train with >=2 of 3 'Tuberculosis' reads, or test with the consensus label
  negative : zero 'Tuberculosis' reads, half 'No finding' and half other findings (hard negatives), seeded
  excluded : train images with exactly 1 of 3 TB reads (ambiguous)
The sample is enriched for TB; AUROC is unaffected but Brier/ECE/PPV need prevalence correction.

Usage:
  python -m datasets.tb_multi.download_vindr select                       # writes results/vindr_selection.csv (no login)
  PHYSIONET_USER=you python -m datasets.tb_multi.download_vindr fetch     # prompts for the password
Options: --root (default /data/physionet.org/files/vindr-cxr/1.0.0), --workers 4. Uses wget as transport: physionet.org returns 403 to python-requests/curl User-Agents.
"""
import argparse
import getpass
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

BASE = "https://physionet.org/files/vindr-cxr/1.0.0"
ROOT = "/data/physionet.org/files/vindr-cxr/1.0.0"
SEL = "results/vindr_selection.csv"
N_NEG_TRAIN, N_NEG_TEST = 1000, 500


def _neg_sample(ids, nofind, n, rng):
    half = n // 2
    a = rng.permutation(ids[nofind.loc[ids].values])[:half]
    b = rng.permutation(ids[~nofind.loc[ids].values])[: n - half]
    return list(a) + list(b)


def select(root, seed=0):
    ann = f"{root}/annotations"
    tr, te = pd.read_csv(f"{ann}/image_labels_train.csv"), pd.read_csv(f"{ann}/image_labels_test.csv").set_index("image_id")
    g = tr.groupby("image_id")
    votes, nf = g["Tuberculosis"].sum(), g["No finding"].sum() >= 2
    rng = np.random.default_rng(seed)
    rows = [(i, "train", 1, int(v)) for i, v in votes[votes >= 2].items()]
    rows += [(i, "test", 1, 1) for i in te.index[te.Tuberculosis == 1]]
    zero = votes.index[votes == 0].values
    rows += [(i, "train", 0, 0) for i in _neg_sample(zero, nf, N_NEG_TRAIN, rng)]
    tz = te.index[te.Tuberculosis == 0].values
    rows += [(i, "test", 0, 0) for i in _neg_sample(tz, te["No finding"].astype(bool), N_NEG_TEST, rng)]
    df = pd.DataFrame(rows, columns=["image_id", "split", "label", "tb_votes"])
    os.makedirs(os.path.dirname(SEL), exist_ok=True)
    df.to_csv(SEL, index=False)
    print(df.groupby(["split", "label"]).size().to_string(), f"\ntotal {len(df)} -> {SEL}")


def _sha_table(root):
    t = {}
    for line in open(f"{root}/SHA256SUMS.txt"):
        h, p = line.split(None, 1)
        t[p.strip()] = h
    return t


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _get(rc, rel, root, want, tries=6):
    """Fetch one file with wget (the transport PhysioNet accepts: python-requests got 401 with a valid login)."""
    dst = f"{root}/{rel}"
    if os.path.exists(dst) and _sha256(dst) == want:
        return "have"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for k in range(tries):
        try:  # -c resumes a partial file left by an interrupted run
            p = subprocess.run(["wget", f"--config={rc}", "-c", "-nv", "-nd", "--timeout=30", "--tries=1",
                                "-P", os.path.dirname(dst), f"{BASE}/{rel}"], capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError(f"wget exit {p.returncode}: {p.stderr.strip()[-100:]}")
            if _sha256(dst) != want:
                os.remove(dst)
                raise RuntimeError("checksum mismatch")
            return "ok"
        except Exception as e:  # noqa: BLE001 - retry any network/IO error
            err = e
            print(f"retry {k + 1}/{tries} {rel}: {type(e).__name__} {str(e)[:80]}", file=sys.stderr, flush=True)
            time.sleep(2 ** k)
    return f"FAIL {err}"


def fetch(root, workers):
    user = os.environ.get("PHYSIONET_USER") or input("PhysioNet user: ")
    pw = os.environ.get("PHYSIONET_PASSWORD") or getpass.getpass("PhysioNet password: ")
    df = pd.read_csv(SEL)
    sums = _sha_table(root)
    rels = [f"{s}/{i}.dicom" for i, s in zip(df.image_id, df.split)]
    print(f"{len(rels)} files, {workers} workers", flush=True)
    # credentials go in a private temp wgetrc (mode 600, removed on exit), not on the command line
    fd, rc = tempfile.mkstemp(prefix="wgetrc_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(f"user={user}\npassword={pw}\n")
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(_get, rc, r, root, sums[r]): r for r in rels}
            done = bad = 0
            for f in as_completed(futs):
                done += 1
                res = f.result()
                if res.startswith("FAIL"):
                    bad += 1
                    print(futs[f], res, file=sys.stderr)
                if done % 10 == 0 or done == len(rels):
                    print(f"{done}/{len(rels)} done, {bad} failed", flush=True)
    finally:
        os.remove(rc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["select", "fetch"])
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    select(a.root) if a.mode == "select" else fetch(a.root, a.workers)
