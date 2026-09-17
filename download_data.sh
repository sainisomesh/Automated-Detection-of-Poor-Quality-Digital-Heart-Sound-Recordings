#!/bin/bash
# ============================================================================
# Downloads and extracts the raw AST-QA source dataset from Zenodo
# (DOI 10.5281/zenodo.19638493) into the given target directory. Idempotent --
# skips the download if PhysioNet2022/ already exists there. Shared by
# ../run_all.sh and revision/run_all.sh.
#
# The pre-mixed lambda sweep is deliberately NOT downloaded here (it's an
# ~19 GB archive): both run_all.sh scripts generate it locally instead, on
# demand, via src/generate_mixed_datasets.py -- same source data, no extra
# download.
#
# Usage: download_data.sh <dataset_dir>
#
# Only requires curl and python3 (already a hard dependency of every script in
# this package) -- extraction goes through Python's zipfile module rather than
# shelling out to `unzip`, since that isn't guaranteed to be on PATH on every
# platform (e.g. a bare Git Bash install on Windows).
# ============================================================================

set -e

DATASET_DIR="$1"
ZENODO_RECORD="19638493"
ZENODO_API="https://zenodo.org/api/records/$ZENODO_RECORD"

if [ -z "$DATASET_DIR" ]; then
    echo "Usage: download_data.sh <dataset_dir>" >&2
    exit 1
fi

download_and_extract() {
    file_key="$1"
    target_dir="$2"
    check_subdir="$3"

    if [ -d "$target_dir/$check_subdir" ]; then
        echo "✓ $target_dir/$check_subdir already present, skipping $file_key"
        return
    fi

    echo "Looking up download URL for $file_key (Zenodo record $ZENODO_RECORD)..."
    url=$(curl -s "$ZENODO_API" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for f in data['files']:
    if f['key'] == '$file_key':
        print(f['links']['self'])
        break
")
    if [ -z "$url" ]; then
        echo "ERROR: could not find '$file_key' in Zenodo record $ZENODO_RECORD" >&2
        exit 1
    fi

    mkdir -p "$target_dir"
    tmp_zip="${TMPDIR:-/tmp}/$file_key"
    echo "Downloading $file_key to $tmp_zip (resumes automatically if interrupted)..."
    curl -L -C - -o "$tmp_zip" "$url"

    echo "Extracting $file_key..."
    extract_dir="${TMPDIR:-/tmp}/zenodo_extract_$$"
    mkdir -p "$extract_dir"
    python3 -c "
import zipfile
with zipfile.ZipFile('$tmp_zip') as zf:
    zf.extractall('$extract_dir')
"
    rm -f "$tmp_zip"

    # The archive's contents may or may not sit inside an extra wrapper folder --
    # find the actual data folders wherever they landed and move them into place
    # rather than assuming a fixed nesting depth.
    python3 -c "
import shutil
from pathlib import Path

extract_dir = Path('$extract_dir')
target_dir = Path('$target_dir')
target_dir.mkdir(parents=True, exist_ok=True)
wanted = {'PhysioNet2022', 'ICBHI2017', 'ESC-50', 'UrbanSound8K'}
found_any = False
for p in extract_dir.rglob('*'):
    if p.is_dir() and (p.name in wanted or p.name.startswith('lambda_')):
        dest = target_dir / p.name
        if not dest.exists():
            shutil.move(str(p), str(dest))
        found_any = True
if not found_any:
    raise SystemExit(f'No expected folders found inside {extract_dir} -- archive layout may have changed.')
"
    rm -rf "$extract_dir"
    echo "✓ $file_key extracted to $target_dir"
}

download_and_extract "ast-heart-quality-dataset.zip" "$DATASET_DIR" "PhysioNet2022"
