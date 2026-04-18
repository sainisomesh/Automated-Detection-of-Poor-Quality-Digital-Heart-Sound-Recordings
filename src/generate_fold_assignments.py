#!/usr/bin/env python3
"""Generate fold assignments CSV for reproducibility verification."""

import os
import sys
from pathlib import Path
from sklearn.model_selection import KFold
import pandas as pd

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset/"
    data_root = Path(data_dir)
    heart_files = sorted(list(data_root.rglob("PhysioNet2022/**/*.wav")))

    patient_map = {}
    for f in heart_files:
        pid = f.name.split('_')[0]
        patient_map.setdefault(pid, []).append(f.name)
    pids = sorted(list(patient_map.keys()))

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    rows = []
    for fold, (_, test_idx) in enumerate(kf.split(pids)):
        for i in test_idx:
            pid = pids[i]
            for fname in patient_map[pid]:
                rows.append({"patient_id": pid, "filename": fname, "fold": fold})

    df = pd.DataFrame(rows)
    os.makedirs("fold_assignments", exist_ok=True)
    df.to_csv("fold_assignments/patient_folds.csv", index=False)
    print(f"Saved {len(df)} file-fold assignments for {len(pids)} patients")

if __name__ == "__main__":
    main()
