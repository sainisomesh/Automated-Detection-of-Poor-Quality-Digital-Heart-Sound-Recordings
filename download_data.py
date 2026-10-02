#!/usr/bin/env python3
"""Download and extract the raw AST-QA source dataset from Zenodo.

Fetches ast-heart-quality-dataset.zip from record 10.5281/zenodo.19638493 and
extracts PhysioNet2022/, ICBHI2017/, ESC-50/ and UrbanSound8K/ into the given
directory. Does nothing if all four folders are already present and non-empty.
The download is verified against the checksum published by Zenodo.

Uses only the Python standard library, so it behaves the same on Linux, macOS
and Windows.

Usage:
    python download_data.py <dataset_dir>
"""

import hashlib
import json
import shutil
import ssl
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ZENODO_RECORD = "19638493"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
FILE_KEY = "ast-heart-quality-dataset.zip"
FOLDERS = ["PhysioNet2022", "ICBHI2017", "ESC-50", "UrbanSound8K"]

# Some Python installs (e.g. python.org builds on macOS) ship without a CA
# bundle, so use certifi's when it is available.
try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()


def has_wavs(folder):
    return folder.is_dir() and next(folder.rglob("*.wav"), None) is not None


def fetch_json(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60, context=SSL_CONTEXT) as r:
                return json.load(r)
        except Exception as e:
            if attempt == retries - 1:
                sys.exit(f"ERROR: could not reach Zenodo ({url}): {e}")
            time.sleep(5)


def download(url, dest, expected_md5):
    """Download url to dest, resuming a partial file, then check its md5."""
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=120, context=SSL_CONTEXT) as r:
        if have and r.status != 206:
            have = 0  # server ignored the range request, start over
        total = have + int(r.headers.get("Content-Length", 0))
        with open(part, "ab" if have else "wb") as f:
            done, last = have, 0.0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 2:
                    print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
                    last = time.time()
    print()

    if expected_md5:
        md5 = hashlib.md5()
        with open(part, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                md5.update(chunk)
        if md5.hexdigest() != expected_md5:
            part.unlink()
            sys.exit("ERROR: checksum mismatch, the partial download was removed. Please rerun.")
    part.replace(dest)


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python download_data.py <dataset_dir>")
    target = Path(sys.argv[1]).resolve()

    if all(has_wavs(target / name) for name in FOLDERS):
        print(f"Dataset already present in {target}, skipping download.")
        return

    record = fetch_json(ZENODO_API)
    entry = next((f for f in record.get("files", []) if f.get("key") == FILE_KEY), None)
    if entry is None:
        sys.exit(f"ERROR: {FILE_KEY} not found in Zenodo record {ZENODO_RECORD}")
    url = entry["links"]["self"]
    checksum = entry.get("checksum", "")
    expected_md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None

    target.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.gettempdir())
    zip_path = cache / FILE_KEY
    if not zip_path.exists():
        print(f"Downloading {FILE_KEY} ({entry.get('size', 0) / 1e9:.1f} GB) from Zenodo...")
        download(url, zip_path, expected_md5)

    print(f"Extracting into {target} ...")
    with tempfile.TemporaryDirectory(dir=target) as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        for name in FOLDERS:
            src = next((p for p in Path(tmp).rglob(name) if p.is_dir()), None)
            if src is None:
                sys.exit(f"ERROR: {name}/ not found inside {FILE_KEY}")
            dest = target / name
            if has_wavs(dest):
                continue
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            elif dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(src), str(dest))

    missing = [name for name in FOLDERS if not has_wavs(target / name)]
    if missing:
        sys.exit(f"ERROR: extraction incomplete, missing {', '.join(missing)}")
    zip_path.unlink()
    print(f"Dataset ready in {target}")


if __name__ == "__main__":
    main()
