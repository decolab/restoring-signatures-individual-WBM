#!/usr/bin/env python3
"""
Step 03 – Explore the global coupling parameter G.

Sweeps G from 0 to 2 and computes how well the model matches the
empirical data at each value (measured by SSIM). This finds the
optimal G for subsequent steps.

I recommend running several converging sweeps with increasing resolution 
(g_steps) around the best G value.

Run with:
    python 03_explore_global_coupling.py
"""

import os
import yaml
import numpy as np
from signal_processing import compute_fc, normalize_minmax
from hopf_model import run_simulations
from metrics import compute_ssim

# ---- Load configuration ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "config.yaml")) as f:
    CONFIG = yaml.safe_load(f)


def main(subject="taro", condition="stim_off", g_min=0.0, g_max=2.0,
         g_steps=21):
    """
    Parameters
    ----------
    subject : str
        Which subject to use (default: "taro").
    condition : str
        Which condition to fit (default: "stim_off").
    g_min, g_max : float
        Range of G values to explore.
    g_steps : int
        Number of G values in the grid.
    """
    mc = CONFIG["model"]
    fc = CONFIG["filter"]
    use_parallel = mc.get("use_parallel", False)
    max_workers = mc.get("max_workers", 4)

    # Load structural connectivity
    sc_path = os.path.join(SCRIPT_DIR, CONFIG["paths"]["structural_connectivity"])
    sc = np.loadtxt(sc_path)
    sc_max = mc.get("sc_normalisation_max", None)
    if sc_max is not None:
        sc = sc / sc.max() * sc_max
        print(f"SC normalised to max={sc_max}")

    # Load frequencies
    freq_path = os.path.join(SCRIPT_DIR, CONFIG["paths"]["node_frequencies"])
    freqs = np.load(freq_path, allow_pickle=True).item()[subject][condition]

    # Load empirical FC
    emp_fc_path = os.path.join(SCRIPT_DIR, "outputs", "empirical_fc",
                               subject, f"{condition}_fc.npy")
    if not os.path.exists(emp_fc_path):
        print(f"ERROR: Empirical FC not found: {emp_fc_path}")
        print("       Run step 02 first.")
        return
    emp_fc = np.load(emp_fc_path)
    emp_fc_norm = normalize_minmax(emp_fc[np.newaxis])[0]

    # Homogeneous bifurcation parameter (all zeros = purely noise-driven)
    bif = np.zeros((1, mc["num_parcels"]))

    # Grid search
    g_values = np.linspace(g_min, g_max, g_steps)
    ssim_values = np.zeros(g_steps)

    print(f"\nExploring G from {g_min} to {g_max} ({g_steps} steps)")
    print(f"Subject: {subject}, Condition: {condition}")
    print(f"Parallel: {use_parallel}, Workers: {max_workers}")
    print("-" * 50)

    for i, g in enumerate(g_values):
        print(f"G = {g:.4f} ({i+1}/{g_steps})", end=" ... ")

        # Simulate
        ts_batch = run_simulations(
            n=mc["num_subsimulations"],
            time_points=mc["time_points"],
            time_repetition=mc["time_repetition"],
            num_parcels=mc["num_parcels"],
            global_coupling=g,
            structural_connectivity=sc,
            bifurcation_parameters=bif,
            frequencies=freqs,
            dt=mc["dt"],
            sig=mc["sig"],
            use_parallel=use_parallel,
            max_workers=max_workers,
        )

        # Compute FC
        sim_fcs = compute_fc(
            ts_batch, time_repetition=mc["time_repetition"],
            lowpass=fc["lowpass"], highpass=fc["highpass"], order=fc["order"]
        )
        sim_fc_mean = sim_fcs.mean(axis=0)
        sim_fc_norm = normalize_minmax(sim_fc_mean[np.newaxis])[0]

        # SSIM
        ssim_val = compute_ssim(emp_fc_norm, sim_fc_norm)
        ssim_values[i] = ssim_val
        print(f"SSIM = {ssim_val:.4f}")

    # Save results
    out_dir = os.path.join(SCRIPT_DIR, "outputs", "g_exploration")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{subject}_{condition}_g_values.npy"), g_values)
    np.save(os.path.join(out_dir, f"{subject}_{condition}_ssim_values.npy"), ssim_values)

    best_idx = np.argmax(ssim_values)
    print(f"\nBest G = {g_values[best_idx]:.4f} (SSIM = {ssim_values[best_idx]:.4f})")
    print(f"Results saved → {out_dir}")


if __name__ == "__main__":
    main()
