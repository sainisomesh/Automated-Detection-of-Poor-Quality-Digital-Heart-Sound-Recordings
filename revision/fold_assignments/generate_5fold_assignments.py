#!/usr/bin/env python3
"""Generate 5-fold patient-level assignments for PAPER_REVISIONS experiments.

Mirrors reproducibility/src/generate_fold_assignments.py exactly (same patient-level
grouping, same seed=42) but with n_splits=5 instead of 10, per the explicit decision
to trade fold count for compute budget on the revision experiments. See
PAPER_REVISIONS/README.md for why this is a deliberate, documented deviation from the
10-fold splits used in the published Table 2.
"""

import os
import sys
from pathlib import Path
from sklearn.model_selection import KFold
import pandas as pd

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../../data_processed/"
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
