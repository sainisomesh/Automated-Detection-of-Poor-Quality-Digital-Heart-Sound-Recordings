#!/usr/bin/env python3
"""Generate patient-level cross-validation fold assignments.

Splitting is done over unique patient identifiers rather than over individual
recordings, so that every recording from a given patient falls in exactly one
test fold. This is what prevents the same patient appearing in both the
training and test set of a fold, which would leak patient-specific acoustic
characteristics and inflate measured performance.

The patient identifier is the portion of the filename before the first
underscore: 13918_AV.wav, 13918_MV.wav, 13918_PV.wav and 13918_TV.wav are four
auscultation sites from patient 13918 and always travel together.

Usage:
    python generate_5fold_assignments.py [data_dir] [n_splits]

Writes patient_folds_<n_splits>fold.csv next to this script. Uses a fixed
random_state so the split is reproducible; the checked-in CSVs were produced
with the defaults below.
"""

import os
import sys
from pathlib import Path
from sklearn.model_selection import KFold
import pandas as pd

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../../dataset/"
    n_splits = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    data_root = Path(data_dir)
    heart_files = sorted(list(data_root.rglob("PhysioNet2022/**/*.wav")))

    patient_map = {}
    for f in heart_files:
        pid = f.name.split('_')[0]
        patient_map.setdefault(pid, []).append(f.name)
    pids = sorted(list(patient_map.keys()))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []
    for fold, (_, test_idx) in enumerate(kf.split(pids)):
        for i in test_idx:
            pid = pids[i]
            for fname in patient_map[pid]:
                rows.append({"patient_id": pid, "filename": fname, "fold": fold})

    df = pd.DataFrame(rows)
    out_dir = Path(__file__).parent
    out_path = out_dir / f"patient_folds_{n_splits}fold.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} file-fold assignments for {len(pids)} patients across {n_splits} folds -> {out_path}")

if __name__ == "__main__":
    main()
