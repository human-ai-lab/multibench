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
    return kagglehub.dataset_download(DATASET_REF, path=relative_path)


def list_dataset_files() -> List[str]:
    """List every file in the dataset (paginated; a few thousand entries, no downloads)."""
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
    request_delay: float = 0.5,
    max_scan: int = 1000,
    cache_path: str = IMAGE_INDEX_CACHE,
) -> Dict[str, str]:
    """Map `anon_id -> image relative path` by reading each DICOM's PatientID header tag.

    One request at a time (a small delay between each) rather than concurrent - bursts of
    parallel requests against Kaggle's per-file download endpoint triggered a temporary
    block earlier. Separately, and unrelated to concurrency: individual per-file downloads
    against this endpoint 404 nondeterministically (~90% of the time, empirically - not a
    permissions or rate-limit issue, since the *same* file succeeds on a later attempt) -
    so this scans more candidates than the naive "36% of images belong to an audio-having
    participant" estimate would suggest, and always stops after `max_scan` attempts so a
    bad run terminates rather than scanning indefinitely.
    Progress is appended to `cache_path` as it goes, so an interrupted run can be resumed
    (already-cached filenames are skipped).
    """
    import json as json_lib

    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        creds = json_lib.load(f)

    anon_id_to_path = load_cached_image_index(cache_path)
    already_scanned = set(anon_id_to_path.values())
    remaining = [p for p in image_paths if p not in already_scanned]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(remaining))

    def _matched_targets():
        if target_anon_ids is None:
            return 0
        return len(target_anon_ids & set(anon_id_to_path))

    file_exists = os.path.exists(cache_path)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["filename", "patient_id"])
        for n_scanned, idx in enumerate(order, start=1):
            if target_anon_ids is not None and _matched_targets() >= (stop_after or len(target_anon_ids)):
                break
            if n_scanned > max_scan:
                print(f"  reached max_scan={max_scan} without finding all requested matches; "
                      f"continuing with {_matched_targets()} found.")
                break
            path = remaining[idx]
            t0 = time.monotonic()
            patient_id = _fetch_patient_id(path, creds)
            elapsed = time.monotonic() - t0
            if patient_id is not None:
                anon_id_to_path[patient_id] = path
                writer.writerow([path, patient_id])
                f.flush()
            if n_scanned % 5 == 0 or elapsed > 5.0:
                print(f"  scanned {n_scanned}/{len(remaining)} images "
                      f"(last took {elapsed:.1f}s), {_matched_targets()} target matches so far...",
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
    cache_path: str = IMAGE_INDEX_CACHE,
) -> Dict[str, str]:
    cached = load_cached_image_index(cache_path)
    n_cached_matches = len(target_anon_ids & set(cached)) if target_anon_ids is not None else len(cached)
    if n_cached_matches >= (stop_after or (len(target_anon_ids) if target_anon_ids is not None else 0)):
        return cached

    print(f"Resolving chest X-ray filenames for {len(target_anon_ids) if target_anon_ids else 'all'} "
          f"participants ({n_cached_matches} already cached at {cache_path})...")
    return build_image_index(
        image_paths, target_anon_ids=target_anon_ids, stop_after=stop_after, cache_path=cache_path,
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

    text_feats, audio_feats, image_feats, labels = [], [], [], []
    for _, row in sampled.iterrows():
        audio_path = _download_file(kagglehub, audio_by_barcode[row["barcode"]])
        sr, waveform = wavfile.read(audio_path)
        image_path = _download_file(kagglehub, anon_id_to_image[str(row["anon_id"])])
        ds = pydicom.dcmread(image_path)

        text_feats.append(extract_text_features(row.to_dict()))
        audio_feats.append(extract_audio_features(waveform, n_bins=audio_bins))
        image_feats.append(extract_image_features(ds.pixel_array, size=image_size))
        labels.append(encode_label(row["ground_truth_tb"]))

    text_feats, audio_feats, image_feats, labels = map(np.stack, (text_feats, audio_feats, image_feats, labels))
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
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_dataset(args.output, max_samples=args.max_samples, seed=args.seed)


if __name__ == "__main__":
    main()
