#!/usr/bin/env python3
"""
Step 05 – Data augmentation (generate surrogate FC matrices).

For each condition, generates many surrogate FC matrices by running
the fitted Hopf model with different random seeds. These surrogate
FCs are used to train the VAE (step 07).

Run with:
    python 05_data_augmentation.py
"""

import os
import yaml
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from signal_processing import compute_fc
from hopf_model import _run_single_baseline, get_random_seeds

# ---- Load configuration ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "config_test.yaml")) as f:
    CONFIG = yaml.safe_load(f)


def main(subject="taro"):
    """
    Parameters
    ----------
    subject : str
        Which subject to generate surrogates for.
    """
    mc = CONFIG["model"]
    fc_cfg = CONFIG["filter"]
    aug_cfg = CONFIG["augmentation"]
    num_fcs = aug_cfg["num_fcs"]
    num_subsim = mc["num_subsimulations"]
    use_parallel = mc.get("use_parallel", False)
    max_workers = mc.get("max_workers", 4)

    # Load structural connectivity
    sc = np.loadtxt(os.path.join(SCRIPT_DIR, CONFIG["paths"]["structural_connectivity"]))
    sc_max = mc.get("sc_normalisation_max", None)
    if sc_max is not None:
        sc = sc / sc.max() * sc_max

    # Load frequencies (nested dict: subject → condition → array)
    freq_dict = np.load(os.path.join(SCRIPT_DIR, CONFIG["paths"]["node_frequencies"]),
                        allow_pickle=True).item()

    # Load fitted bifurcation parameters
    best_genome_path = os.path.join(SCRIPT_DIR, CONFIG["paths"]["ga_best_genomes"])
    if not os.path.exists(best_genome_path):
        print(f"ERROR: Fitted genome not found: {best_genome_path}")
        print("       Run step 04 first.")
        return
    genome = np.load(best_genome_path)
    bif = np.concatenate([genome, genome]).reshape(1, mc["num_parcels"])

    # Determine optimal G from step 03
    g_path = os.path.join(SCRIPT_DIR, "outputs", "g_exploration",
                          f"{subject}_stim_off_g_values.npy")
    ssim_path = os.path.join(SCRIPT_DIR, "outputs", "g_exploration",
                             f"{subject}_stim_off_ssim_values.npy")
    if os.path.exists(g_path) and os.path.exists(ssim_path):
        g_vals = np.load(g_path)
        ssim_vals = np.load(ssim_path)
        g = float(g_vals[np.argmax(ssim_vals)])
    else:
        print("ERROR: G exploration results not found. Run step 03 first.")
        return

    conditions = CONFIG["subjects"][subject]
    all_fcs = []
    all_labels = []

    # Total simulations needed: num_fcs * num_subsim per condition.
    # We dispatch ALL simulations to a single persistent pool, then
    # group results into sets of num_subsim to compute each FC.

    for condition in conditions:
        freqs = freq_dict[subject][condition]
        total_sims = num_fcs * num_subsim
        seeds = get_random_seeds(total_sims)

        # Build argument tuples for _run_single_baseline
        base_args = (mc["time_points"], mc["time_repetition"],
                     mc["num_parcels"], g, sc, bif, freqs,
                     mc["dt"], mc["sig"])
        arglist = [base_args + (s,) for s in seeds]

        print(f"\n{subject}/{condition}: dispatching {total_sims} simulations "
              f"({num_fcs} FCs × {num_subsim} sub-sims)")

        if use_parallel:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                sim_results = list(tqdm(
                    pool.map(_run_single_baseline, arglist, chunksize=max_workers),
                    total=total_sims,
                    desc=f"{subject}/{condition}",
                    unit="sim",
                ))
        else:
            sim_results = []
            for args in tqdm(arglist, desc=f"{subject}/{condition}", unit="sim"):
                sim_results.append(_run_single_baseline(args))

        sim_results = np.stack(sim_results)  # (total_sims, 82, 500)

        # Group into sets of num_subsim → compute one FC each
        for j in tqdm(range(num_fcs), desc="Computing FCs", unit="FC"):
            ts_group = sim_results[j * num_subsim : (j + 1) * num_subsim]
            fcs = compute_fc(ts_group,
                             time_repetition=mc["time_repetition"],
                             lowpass=fc_cfg["lowpass"],
                             highpass=fc_cfg["highpass"],
                             order=fc_cfg["order"])
            all_fcs.append(fcs.mean(axis=0))
            all_labels.append(condition)

    # Save
    all_fcs_arr = np.stack(all_fcs)
    all_labels_arr = np.array(all_labels)

    out_fc = os.path.join(SCRIPT_DIR, CONFIG["paths"]["augmented_fcs"])
    out_lb = os.path.join(SCRIPT_DIR, CONFIG["paths"]["augmented_labels"])
    os.makedirs(os.path.dirname(out_fc), exist_ok=True)
    np.save(out_fc, all_fcs_arr)
    np.save(out_lb, all_labels_arr)
    print(f"\nSaved {len(all_fcs)} augmented FCs → {out_fc}")
    print(f"Saved labels → {out_lb}")


if __name__ == "__main__":
    main()
