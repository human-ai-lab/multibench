"""Fetch and preprocess a subset of the Google + CIDRZ Health AI TB dataset from Kaggle.

https://www.kaggle.com/datasets/googlehealthai/google-health-ai (83.9GB total: chest X-ray
DICOMs for 1828 participants, cough audio from 4 devices for 664 of them, and a metadata
table with symptoms/history and TB reference-standard labels keyed by participant barcode).

This is a one-time preprocessing step, not part of the training-time dataloader (see
`get_data.py`) - it downloads a small stratified sample (audio + image + label all present),
extracts fixed-size features per modality (`features.py`), and writes a single pickle that
`get_data.get_dataloader` reads directly, mirroring how `datasets/avmnist` expects
already-materialized local arrays rather than downloading at train time.

Requires the `google_health` extra (`pip install -e .[google_health]`) and a Kaggle API
token at `~/.kaggle/kaggle.json` (see https://www.kaggle.com/docs/api#authentication).

Note on runtime: resolving each chest X-ray's participant requires one request per
candidate image (see `build_image_index`'s docstring), and Kaggle's per-file download
endpoint 404s nondeterministically on the large majority of attempts - empirically, getting
`max_samples` matches can mean scanning several hundred candidates at ~1 request/second.
Expect minutes, not seconds, even for a small `--max-samples`; `image_index.csv` caches
progress so a second run resumes rather than rescanning.

Usage:
    python -m datasets.google_health.download --output data/google_health/tb_dataset.pkl \
        --max-samples 80
"""
import argparse
import csv
import io
import os
import pickle
import time
from typing import Dict, List, Optional

import numpy as np

from .features import encode_label, extract_audio_features, extract_image_features, extract_text_features

DATASET_REF = "googlehealthai/google-health-ai"
# Pinning the version means kagglehub's dataset_download skips an extra "get current
# version" API call per file (a separate quota from the download endpoint itself, and one
# that got rate-limited on its own after repeated calls - one per file downloaded).
DATASET_VERSION = 18
DATASET_REF_VERSIONED = f"{DATASET_REF}/versions/{DATASET_VERSION}"
METADATA_PATH = "Metadata and Codebook/Metadata and Codebook/GHAI_Final_Data_2023.csv"
IMAGE_DIR = "Google_AI_Anonymized_images/Google_AI_Anonymized_images"
FACILITY_TO_AUDIO_DIR = {
    "Kan": "Audio-Recorder-Kanyama",
    "Cha": "Audio-Recorder-Chawama",
    "Chai": "Audio-Recorder-Chainda-South",
}

# The image files' names carry no relation to the participant they belong to - the only
# link is the `PatientID` DICOM tag, which equals `anon_id` in the metadata table. Reading
# just the header (a small byte-range request against the signed GCS URL Kaggle redirects
# to) is enough to resolve filename -> anon_id without downloading full pixel data. Results
# are cached to `image_index.csv` (gitignored - it's a local cache, not checked in) so
# repeated runs against the same sample don't re-scan.
IMAGE_INDEX_CACHE = os.path.join(os.path.dirname(__file__), "image_index.csv")
_HEADER_RANGE_BYTES = 24576


def _require(module_name: str, extra_hint: str = "google_health"):
    import importlib
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"'{module_name}' is required for datasets.google_health.download. "
            f"Install it with: pip install -e '.[{extra_hint}]'"
        ) from e


def _kaggle_auth():
    kagglehub = _require("kagglehub")
    return kagglehub


def _download_file(kagglehub, relative_path: str) -> str:
    """Download one file from the dataset and return its local path."""
    return kagglehub.dataset_download(DATASET_REF_VERSIONED, path=relative_path)


FILE_LIST_CACHE = os.path.join(os.path.dirname(__file__), "file_list_cache.txt")


def list_dataset_files(cache_path: str = FILE_LIST_CACHE) -> List[str]:
    """List every file in the dataset (paginated; a few thousand entries, no downloads).

    The file listing is static, so it's cached to disk - re-running this every restart of
    `build_dataset` was hitting Kaggle's rate limit on the listing endpoint (a separate
    quota from the per-file download endpoint).
    """
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return [line.rstrip("\n") for line in f]

    kaggle_api_mod = _require("kaggle.api.kaggle_api_extended", "google_health")
    api = kaggle_api_mod.KaggleApi()
    api.authenticate()
    names = []
    page_token = None
    while True:
        resp = api.dataset_list_files(DATASET_REF, page_token=page_token, page_size=200)
        if not resp.files:
            break
        names.extend(f.name for f in resp.files)
        page_token = getattr(resp, "next_page_token", None)
        if not page_token:
            break
    with open(cache_path, "w") as f:
        f.write("\n".join(names) + "\n")
    return names


def _fetch_patient_id(image_relative_path: str, creds: dict, retries: int = 1, timeout: float = 12.0) -> Optional[str]:
    import requests

    pydicom = _require("pydicom")
    for attempt in range(retries):
        try:
            r = requests.get(
                f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_REF}",
                params={"file_name": image_relative_path},
                auth=(creds["username"], creds["key"]),
                headers={"Range": f"bytes=0-{_HEADER_RANGE_BYTES - 1}"},
                allow_redirects=True,
                timeout=timeout,
            )
            if r.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {r.status_code}")
            ds = pydicom.dcmread(io.BytesIO(r.content), stop_before_pixels=True, force=True)
            return str(ds.PatientID)
        except Exception as e:
            if attempt == retries - 1:
                print(f"    WARN: giving up on {image_relative_path}: {e}")
                return None
            time.sleep(1.0)


def build_image_index(
    image_paths: List[str],
    target_anon_ids: Optional[set] = None,
    stop_after: Optional[int] = None,
    seed: int = 42,
    request_delay: float = 0.3,
    max_scan: int = 100_000,
    max_wall_seconds: float = 2 * 3600,
    cache_path: str = IMAGE_INDEX_CACHE,
) -> Dict[str, str]:
    """Map `anon_id -> image relative path` by reading each DICOM's PatientID header tag.

    One request at a time (a small delay between each) rather than concurrent - bursts of
    parallel requests against Kaggle's per-file download endpoint triggered a temporary
    block earlier. Separately, and unrelated to concurrency: individual per-file downloads
    against this endpoint 404 nondeterministically (~90% of the time, empirically - not a
    permissions or rate-limit issue, since the *same* file succeeds on a later, independent
    attempt). Because of that, this makes repeated passes over whatever's still unresolved
    (every successful decode - target-matching or not - is cached and excluded from later
    passes) rather than a single sweep, since one pass only resolves ~10% of files. Stops
    when `stop_after` matches are found, or `max_scan` total attempts or `max_wall_seconds`
    elapse, whichever comes first - `max_scan`/`max_wall_seconds` exist so a run asking for
    (near-)complete coverage still terminates, not to speed up a small-sample run.
    Progress is appended to `cache_path` as it goes, so an interrupted run can be resumed
    (already-cached filenames are skipped) - including by a fresh invocation of this
    function, which is exactly what looping passes here does internally.
    """
    import json as json_lib

    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        creds = json_lib.load(f)

    anon_id_to_path = load_cached_image_index(cache_path)

    def _matched_targets():
        if target_anon_ids is None:
            return 0
        return len(target_anon_ids & set(anon_id_to_path))

    def _goal_met():
        return target_anon_ids is not None and _matched_targets() >= (stop_after or len(target_anon_ids))

    file_exists = os.path.exists(cache_path)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    start_time = time.monotonic()
    total_scanned = 0
    n_pass = 0
    with open(cache_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["filename", "patient_id"])
        while not _goal_met():
            already_scanned = set(anon_id_to_path.values())
            remaining = [p for p in image_paths if p not in already_scanned]
            if not remaining:
                print("  every image has been resolved; nothing left to scan.")
                break
            n_pass += 1
            rng = np.random.default_rng(seed + n_pass)
            order = rng.permutation(len(remaining))
            print(f"  pass {n_pass}: {len(remaining)} unresolved images left to try...", flush=True)
            for idx in order:
                if _goal_met():
                    break
                total_scanned += 1
                if total_scanned > max_scan:
                    print(f"  reached max_scan={max_scan} total attempts; "
                          f"continuing with {_matched_targets()} matches found.")
                    return anon_id_to_path
                if time.monotonic() - start_time > max_wall_seconds:
                    print(f"  reached max_wall_seconds={max_wall_seconds:.0f}; "
                          f"continuing with {_matched_targets()} matches found.")
                    return anon_id_to_path
                path = remaining[idx]
                t0 = time.monotonic()
                patient_id = _fetch_patient_id(path, creds)
                elapsed = time.monotonic() - t0
                if patient_id is not None:
                    anon_id_to_path[patient_id] = path
                    writer.writerow([path, patient_id])
                    f.flush()
                if total_scanned % 25 == 0 or elapsed > 5.0:
                    print(f"  scanned {total_scanned} images total ({len(anon_id_to_path)} resolved, "
                          f"last took {elapsed:.1f}s), {_matched_targets()} target matches so far...",
                          flush=True)
                time.sleep(request_delay)
    return anon_id_to_path


def load_cached_image_index(cache_path: str = IMAGE_INDEX_CACHE) -> Dict[str, str]:
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, newline="") as f:
        return {row["patient_id"]: row["filename"] for row in csv.DictReader(f)}


def load_or_build_image_index(
    image_paths: List[str],
    target_anon_ids: Optional[set] = None,
    stop_after: Optional[int] = None,
    max_wall_seconds: float = 2 * 3600,
    cache_path: str = IMAGE_INDEX_CACHE,
) -> Dict[str, str]:
    cached = load_cached_image_index(cache_path)
    n_cached_matches = len(target_anon_ids & set(cached)) if target_anon_ids is not None else len(cached)
    if n_cached_matches >= (stop_after or (len(target_anon_ids) if target_anon_ids is not None else 0)):
        return cached

    print(f"Resolving chest X-ray filenames for {len(target_anon_ids) if target_anon_ids else 'all'} "
          f"participants ({n_cached_matches} already cached at {cache_path})...")
    return build_image_index(
        image_paths, target_anon_ids=target_anon_ids, stop_after=stop_after,
        max_wall_seconds=max_wall_seconds, cache_path=cache_path,
    )


def _stratified_split(labels: np.ndarray, val_frac: float, test_frac: float, seed: int):
    """Stratified train/valid/test split; falls back to a plain shuffle if a class is too
    small to stratify (train_test_split needs >=1 member per class per split)."""
    from sklearn.model_selection import train_test_split

    n = len(labels)
    idx = np.arange(n)
    min_class_count = np.bincount(labels).min()
    try:
        if min_class_count < 2:
            raise ValueError("a class has fewer than 2 members; can't stratify")
        train_idx, hold_idx = train_test_split(
            idx, test_size=val_frac + test_frac, stratify=labels, random_state=seed)
        val_idx, test_idx = train_test_split(
            hold_idx, test_size=test_frac / (val_frac + test_frac),
            stratify=labels[hold_idx], random_state=seed)
    except ValueError as e:
        print(f"WARNING: falling back to an unstratified split ({e}). With this few samples "
              "per class, treat any evaluation metrics as a pipeline smoke test, not a result.")
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        test_idx, val_idx, train_idx = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
    return train_idx, val_idx, test_idx


def build_dataset(
    output: str,
    max_samples: int = 80,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    image_size: int = 64,
    audio_bins: int = 64,
    max_wall_seconds: float = 2 * 3600,
):
    """Download a stratified sample and write the preprocessed pickle `get_data.py` reads."""
    pd = _require("pandas")
    pydicom = _require("pydicom")
    kagglehub = _kaggle_auth()
    from scipy.io import wavfile

    print("Listing dataset files...")
    all_files = list_dataset_files()
    audio_by_barcode: Dict[str, str] = {}
    for facility_dir in FACILITY_TO_AUDIO_DIR.values():
        prefix = f"{facility_dir}/"
        for name in all_files:
            if name.startswith(prefix) and name.lower().endswith(".wav"):
                barcode = os.path.splitext(os.path.basename(name))[0]
                audio_by_barcode.setdefault(barcode, name)
    image_paths = [n for n in all_files if n.startswith(IMAGE_DIR) and n.endswith(".dcm")]

    print("Downloading metadata table...")
    metadata_local = _download_file(kagglehub, METADATA_PATH)
    df = pd.read_csv(metadata_local)
    df = df[df["barcode"].isin(audio_by_barcode) & df["ground_truth_tb"].isin(["neg", "pos"])]

    print("Resolving participant <-> chest X-ray mapping (image_index.csv)...")
    target_anon_ids = set(df["anon_id"].astype(str))
    # In principle only ~36% of images belong to an audio-having participant, so a modest
    # oversampling margin would be enough to stratify by label from - but Kaggle's per-file
    # download endpoint 404s nondeterministically on ~90% of attempts (empirically; the same
    # file often succeeds on a later attempt), so getting even `max_samples` matches already
    # means scanning several hundred candidates. Don't inflate the target further - accept
    # whatever label balance results (build_dataset's stratified split falls back gracefully
    # if one class ends up tiny).
    pool_size = min(len(target_anon_ids), max_samples)
    anon_id_to_image = load_or_build_image_index(
        image_paths, target_anon_ids=target_anon_ids, stop_after=pool_size,
        max_wall_seconds=max_wall_seconds,
    )
    df = df[df["anon_id"].astype(str).isin(anon_id_to_image)]

    if df.empty:
        raise RuntimeError("No participants have audio + image + a valid label; check the dataset access.")

    rng = np.random.default_rng(seed)
    pos = df[df["ground_truth_tb"] == "pos"]
    neg = df[df["ground_truth_tb"] == "neg"]
    n_pos = min(len(pos), max(1, max_samples // 2))
    n_neg = min(len(neg), max_samples - n_pos)
    sampled = pd.concat([
        pos.iloc[rng.permutation(len(pos))[:n_pos]],
        neg.iloc[rng.permutation(len(neg))[:n_neg]],
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"Sampled {len(sampled)} participants ({n_pos} pos / {n_neg} neg).")

    checkpoint_path = output + ".checkpoint.pkl"
    done: Dict[str, dict] = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            done = pickle.load(f)
        print(f"Resuming from checkpoint: {len(done)}/{len(sampled)} samples already done.")

    def _save_checkpoint():
        with open(checkpoint_path, "wb") as f:
            pickle.dump(done, f)

    n_skipped = 0
    for i, (_, row) in enumerate(sampled.iterrows(), start=1):
        barcode = row["barcode"]
        if barcode in done:
            continue
        for attempt in range(3):
            try:
                audio_path = _download_file(kagglehub, audio_by_barcode[barcode])
                sr, waveform = wavfile.read(audio_path)
                image_path = _download_file(kagglehub, anon_id_to_image[str(row["anon_id"])])
                ds = pydicom.dcmread(image_path)

                done[barcode] = {
                    "text": extract_text_features(row.to_dict()),
                    "audio": extract_audio_features(waveform, n_bins=audio_bins),
                    "image": extract_image_features(ds.pixel_array, size=image_size),
                    "label": encode_label(row["ground_truth_tb"]),
                }
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  WARN: skipping {barcode} after 3 failed attempts: {e}")
                    done[barcode] = None  # remembered as permanently skipped, not retried
                    n_skipped += 1
                else:
                    time.sleep(5.0 * (attempt + 1))
        if i % 10 == 0:
            _save_checkpoint()
            print(f"  downloaded {i}/{len(sampled)} samples ({n_skipped} skipped this run)...", flush=True)
        time.sleep(0.3)
    _save_checkpoint()
    n_skipped_total = sum(1 for v in done.values() if v is None)
    if n_skipped_total:
        print(f"Skipped {n_skipped_total}/{len(sampled)} samples total after repeated download failures.")

    usable = [v for v in done.values() if v is not None]
    text_feats, audio_feats, image_feats, labels = map(
        np.stack, zip(*[(d["text"], d["audio"], d["image"], d["label"]) for d in usable])
    )
    labels = np.asarray(labels, dtype=np.int64)

    train_idx, val_idx, test_idx = _stratified_split(labels, val_frac, test_frac, seed)

    def _subset(idx):
        return {"text": text_feats[idx], "audio": audio_feats[idx], "image": image_feats[idx], "label": labels[idx]}

    data = {"train": _subset(train_idx), "valid": _subset(val_idx), "test": _subset(test_idx)}
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(data, f)
    print(f"Wrote {output}: train={len(train_idx)} valid={len(val_idx)} test={len(test_idx)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/google_health/tb_dataset.pkl")
    parser.add_argument("--max-samples", type=int, default=80,
                         help="Pass a number >= the audio-having cohort size (664) for 'all of them'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-wall-seconds", type=float, default=2 * 3600,
                         help="Wall-clock budget for resolving chest X-ray filenames before giving up "
                              "and continuing with whatever was found.")
    args = parser.parse_args()
    build_dataset(args.output, max_samples=args.max_samples, seed=args.seed,
                  max_wall_seconds=args.max_wall_seconds)


if __name__ == "__main__":
    main()
