#!/usr/bin/env python3
"""
Step 04 – Fit bifurcation parameters with the genetic algorithm.

Uses a GA to find 41 homotopic bifurcation parameters that maximise
similarity (SSIM) between simulated and empirical FC.

The genome is symmetric: the 41 parameters are mirrored to create 82
(left hemisphere = right hemisphere).

Run with:
    python 04_fit_bifurcation_params.py
"""

import os
import yaml
import numpy as np
from signal_processing import compute_fc, normalize_minmax
from genetic_algorithm import generate_population, run_evolution
from hopf_model import run_simulations
from metrics import compute_ssim

# ---- Load configuration ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "config.yaml")) as f:
    CONFIG = yaml.safe_load(f)


def main(subject="taro", condition="stim_off", global_coupling=None):
    """
    Parameters
    ----------
    subject : str
        Which subject to fit (default: "taro").
    condition : str
        Which condition to fit (default: "stim_off").
    global_coupling : float or None
        If None, uses the optimal G from step 03.
    """
    mc = CONFIG["model"]
    fc_cfg = CONFIG["filter"]
    ga_cfg = CONFIG["ga"]
    num_parcels = mc["num_parcels"]
    genome_length = ga_cfg["genome_length"]

    # Load structural connectivity
    sc = np.loadtxt(os.path.join(SCRIPT_DIR, CONFIG["paths"]["structural_connectivity"]))
    sc_max = mc.get("sc_normalisation_max", None)
    if sc_max is not None:
        sc = sc / sc.max() * sc_max

    # Load frequencies for this subject/condition
    freq_dict = np.load(os.path.join(SCRIPT_DIR, CONFIG["paths"]["node_frequencies"]),
                        allow_pickle=True).item()
    freqs = freq_dict[subject][condition]

    # Load empirical FC
    emp_fc_path = os.path.join(SCRIPT_DIR, "outputs", "empirical_fc",
                               subject, f"{condition}_fc.npy")
    emp_fc = np.load(emp_fc_path)
    emp_fc_norm = normalize_minmax(emp_fc[np.newaxis])[0]

    # Determine G (from step 03 results if not given)
    if global_coupling is None:
        g_path = os.path.join(SCRIPT_DIR, "outputs", "g_exploration",
                              f"{subject}_{condition}_g_values.npy")
        ssim_path = os.path.join(SCRIPT_DIR, "outputs", "g_exploration",
                                 f"{subject}_{condition}_ssim_values.npy")
        if os.path.exists(g_path) and os.path.exists(ssim_path):
            g_vals = np.load(g_path)
            ssim_vals = np.load(ssim_path)
            global_coupling = float(g_vals[np.argmax(ssim_vals)])
            print(f"Using optimal G={global_coupling:.4f} from step 03")
        else:
            print("ERROR: No global coupling value found. Run step 03 first "
                  "or set global_coupling manually.")
            return

    def genome_to_bif(genome):
        """Expand 41-element genome → (1, 82) bifurcation array."""
        arr = np.array(genome)
        return np.concatenate([arr, arr]).reshape(1, num_parcels)

    def fitness_func(genome):
        """Fitness = SSIM between empirical and simulated FC."""
        bif = genome_to_bif(genome)

        # Run simulations
        ts = run_simulations(
            n=mc["num_subsimulations"],
            time_points=mc["time_points"],
            time_repetition=mc["time_repetition"],
            num_parcels=num_parcels,
            global_coupling=global_coupling,
            structural_connectivity=sc,
            bifurcation_parameters=bif,
            frequencies=freqs,
            dt=mc["dt"],
            sig=mc["sig"],
        )

        # Compute FC
        fcs = compute_fc(ts, time_repetition=mc["time_repetition"],
                         lowpass=fc_cfg["lowpass"], highpass=fc_cfg["highpass"],
                         order=fc_cfg["order"])
        fc_mean = fcs.mean(axis=0)
        fc_norm = normalize_minmax(fc_mean[np.newaxis])[0]

        return compute_ssim(emp_fc_norm, fc_norm)

    # Checkpoint path
    ckpt = os.path.join(SCRIPT_DIR, CONFIG["paths"]["ga_checkpoint"])

    print(f"\nFitting bifurcation parameters for {subject}/{condition}")
    print(f"Global coupling G = {global_coupling:.4f}")
    print(f"Genome length = {genome_length} (homotopic → {num_parcels} regions)")
    print("=" * 60)

    # Run GA
    population, gen, fitness_values = run_evolution(
        populate_func=lambda: generate_population(
            ga_cfg["population_size"], genome_length
        ),
        fitness_func=fitness_func,
        fitness_limit=ga_cfg["fitness_limit"],
        generation_limit=ga_cfg["generation_limit"],
        convergence_limit=ga_cfg["convergence_limit"],
        convergence_window=ga_cfg["convergence_window"],
        number_of_elites=ga_cfg["number_of_elites"],
        number_of_mutations=ga_cfg["number_of_mutations"],
        mutation_probability=ga_cfg["mutation_probability"],
        checkpoint_file=ckpt,
    )

    # Extract the best genome
    best_idx = int(np.argmax(fitness_values))
    best_genome = population[best_idx]
    best_fitness = fitness_values[best_idx]
    print(f"\nBest fitness: {best_fitness:.6f} (generation {gen})")

    # Save
    out_path = os.path.join(SCRIPT_DIR, CONFIG["paths"]["ga_best_genomes"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, np.array(best_genome))
    print(f"Best genome saved → {out_path}")

    # Also save the full (1, 82) bifurcation parameter array
    bif_full = genome_to_bif(best_genome)
    bif_path = out_path.replace("ga_best_genomes", "bifurcation_params_fitted")
    np.save(bif_path, bif_full)
    print(f"Full bifurcation params saved → {bif_path}")


if __name__ == "__main__":
    main()
