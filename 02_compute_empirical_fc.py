#!/usr/bin/env python3
"""
Step 02 – Compute empirical functional connectivity matrices.

For each subject and condition, this script:
  1. Loads BOLD timeseries from ts_dict.npy
  2. Removes bad trials
  3. Computes FC: detrend → bandpass → z-score → Pearson correlation
  4. Averages FC across trials
  5. Saves per-subject FC and a combined dictionary

Run with:
    python 02_compute_empirical_fc.py
"""

import os
import yaml
import numpy as np
from signal_processing import compute_fc

# ---- Load configuration ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "config.yaml")) as f:
    CONFIG = yaml.safe_load(f)


def main():
    tr = CONFIG["model"]["time_repetition"]
    lp = CONFIG["filter"]["lowpass"]
    hp = CONFIG["filter"]["highpass"]
    order = CONFIG["filter"]["order"]
    subjects = CONFIG["subjects"]
    problem = CONFIG["problem_indices"]

    # Load timeseries dictionary
    ts_dict_path = os.path.join(SCRIPT_DIR, CONFIG["paths"]["timeseries_dict"])
    if not os.path.exists(ts_dict_path):
        print(f"ERROR: Timeseries file not found: {ts_dict_path}")
        return
    ts_dict = np.load(ts_dict_path, allow_pickle=True).item()
    print(f"Loaded timeseries from {ts_dict_path}")

    all_fcs = {}

    for subject, conditions in subjects.items():
        all_fcs[subject] = {}
        if subject not in ts_dict:
            print(f"  Warning: Subject '{subject}' not in ts_dict, skipping")
            continue

        for condition in conditions:
            if condition not in ts_dict[subject]:
                print(f"  Warning: {subject}/{condition} not in ts_dict, skipping")
                continue

            ts = ts_dict[subject][condition]
            print(f"\n{subject}/{condition}: shape {ts.shape}")

            # Remove bad trials
            bad = problem.get(subject, {}).get(condition, [])
            if bad:
                mask = np.ones(ts.shape[0], dtype=bool)
                mask[bad] = False
                ts = ts[mask]
                print(f"  Removed {len(bad)} bad trials, {ts.shape[0]} remaining")

            # Compute FC
            fcs = compute_fc(ts, time_repetition=tr, lowpass=lp,
                             highpass=hp, order=order)
            print(f"  FC shape: {fcs.shape}")

            # Average across trials
            fc_mean = fcs.mean(axis=0)
            all_fcs[subject][condition] = fc_mean

            # Save per-subject per-condition
            out_dir = os.path.join(SCRIPT_DIR, "outputs", "empirical_fc", subject)
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f"{condition}_fc.npy")
            np.save(out_file, fc_mean)
            print(f"  Saved → {out_file}")

    # Save combined dictionary
    out_path = os.path.join(SCRIPT_DIR, CONFIG["paths"]["empirical_fcs"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, all_fcs, allow_pickle=True)
    print(f"\nAll empirical FCs saved → {out_path}")


if __name__ == "__main__":
    main()
