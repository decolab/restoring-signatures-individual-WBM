#!/usr/bin/env python3
"""
Step 01 – Compute peak frequencies from empirical BOLD timeseries.

For each subject and condition, this script:
  1. Loads BOLD timeseries from ts_dict.npy
  2. Removes bad trials (listed in config.yaml)
  3. Computes the dominant oscillation frequency per brain region via FFT
  4. Saves the result as node_frequencies.npy

Run with:
    python 01_compute_frequencies.py
"""

import os
import yaml
import numpy as np
from signal_processing import compute_peak_frequencies

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
        print("       Place ts_dict.npy in the data/ folder.")
        return
    ts_dict = np.load(ts_dict_path, allow_pickle=True).item()
    print(f"Loaded timeseries from {ts_dict_path}")

    natural_frequencies = {}

    for subject, conditions in subjects.items():
        if subject not in ts_dict:
            print(f"  Warning: Subject '{subject}' not in ts_dict, skipping")
            continue
        natural_frequencies[subject] = {}

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

            # Compute peak frequencies: Data already filtered
            freqs = compute_peak_frequencies(
                ts, time_repetition=tr, lowpass=None, highpass=None,
                filter_order=order
            )
            natural_frequencies[subject][condition] = freqs

    # Save
    out_path = os.path.join(SCRIPT_DIR, CONFIG["paths"]["node_frequencies"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, natural_frequencies, allow_pickle=True)
    print(f"\nSaved node frequencies → {out_path}")


if __name__ == "__main__":
    main()
