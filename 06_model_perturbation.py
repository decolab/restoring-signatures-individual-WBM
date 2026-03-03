#!/usr/bin/env python3
"""
Step 06 – Model perturbation (wave / noise / sync).

Simulates brain stimulation by perturbing specific brain regions in the
model. For each target region and perturbation amplitude, generates FC
matrices that show how the brain state changes under stimulation.

Three perturbation types:
  - wave:  periodic driving signal (cosine/sine)
  - noise: shift bifurcation parameter toward noise (negative)
  - sync:  shift bifurcation parameter toward oscillation (positive)

Run with:
    python 06_model_perturbation.py
"""

import os
import yaml
import numpy as np
from tqdm import tqdm
from signal_processing import compute_fc
from hopf_model import run_simulations_wave, run_simulations_noisesync

# ---- Load configuration ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "config.yaml")) as f:
    CONFIG = yaml.safe_load(f)


def main(subject="taro", pert_type="wave"):
    """
    Parameters
    ----------
    subject : str
        Which subject's model to perturb.
    pert_type : str
        Perturbation type: "wave", "noise", or "sync".
    """
    mc = CONFIG["model"]
    fc_cfg = CONFIG["filter"]
    pert_cfg = CONFIG["perturbation"]
    amplitudes = pert_cfg["amplitudes"]
    num_fcs = pert_cfg["num_fcs_per_amplitude"]
    num_subsim = mc["num_subsimulations"]
    use_parallel = mc.get("use_parallel", False)
    max_workers = mc.get("max_workers", 4)

    # Load data
    sc = np.loadtxt(os.path.join(SCRIPT_DIR, CONFIG["paths"]["structural_connectivity"]))
    sc_max = mc.get("sc_normalisation_max", None)
    if sc_max is not None:
        sc = sc / sc.max() * sc_max

    freq_dict = np.load(os.path.join(SCRIPT_DIR, CONFIG["paths"]["node_frequencies"]),
                        allow_pickle=True).item()
    freqs = freq_dict[subject]["stim_off"]  # baseline condition frequencies

    # Load fitted parameters
    genome = np.load(os.path.join(SCRIPT_DIR, CONFIG["paths"]["ga_best_genomes"]))
    bif = np.concatenate([genome, genome]).reshape(1, mc["num_parcels"])

    # Get optimal G
    g_path = os.path.join(SCRIPT_DIR, "outputs", "g_exploration",
                          f"{subject}_stim_off_g_values.npy")
    ssim_path = os.path.join(SCRIPT_DIR, "outputs", "g_exploration",
                             f"{subject}_stim_off_ssim_values.npy")
    g_vals = np.load(g_path)
    ssim_vals = np.load(ssim_path)
    g = float(g_vals[np.argmax(ssim_vals)])

    # Perturbation targets from config
    targets = CONFIG["perturbation_targets"]

    out_base = os.path.join(SCRIPT_DIR, CONFIG["paths"]["perturbation_fcs"])

    for target_name, target_idx in targets.items():
        target_idx = np.array(target_idx)

        for amp in amplitudes:
            print(f"\nPerturbation: type={pert_type}, target={target_name}, "
                  f"amplitude={amp:.2f}")

            fcs_list = []
            for i in tqdm(range(num_fcs), desc=f"{target_name} amp={amp:.2f}",
                         unit="FC"):
                if pert_type == "wave":
                    ts = run_simulations_wave(
                        n=num_subsim,
                        time_points=mc["time_points"],
                        time_repetition=mc["time_repetition"],
                        num_parcels=mc["num_parcels"],
                        global_coupling=g,
                        structural_connectivity=sc,
                        bifurcation_parameters=bif.copy(),
                        frequencies=freqs,
                        pert_strength=amp,
                        perturbation_indices=target_idx,
                        dt=mc["dt"],
                        sig=mc["sig"],
                        use_parallel=use_parallel,
                        max_workers=max_workers,
                    )
                elif pert_type in ("noise", "sync"):
                    # noise: negative shift; sync: positive shift
                    strength = -amp if pert_type == "noise" else amp
                    ts = run_simulations_noisesync(
                        n=num_subsim,
                        time_points=mc["time_points"],
                        time_repetition=mc["time_repetition"],
                        num_parcels=mc["num_parcels"],
                        global_coupling=g,
                        structural_connectivity=sc,
                        bifurcation_parameters=bif.copy(),
                        frequencies=freqs,
                        pert_strength=strength,
                        perturbation_indices=target_idx,
                        dt=mc["dt"],
                        sig=mc["sig"],
                        use_parallel=use_parallel,
                        max_workers=max_workers,
                    )
                else:
                    raise ValueError(f"Unknown perturbation type: {pert_type}")

                fcs = compute_fc(ts, time_repetition=mc["time_repetition"],
                                 lowpass=fc_cfg["lowpass"],
                                 highpass=fc_cfg["highpass"],
                                 order=fc_cfg["order"])
                fcs_list.append(fcs.mean(axis=0))

            fcs_arr = np.stack(fcs_list)

            out_dir = os.path.join(out_base, pert_type, target_name)
            os.makedirs(out_dir, exist_ok=True)
            fname = f"amp_{amp:.2f}.npy"
            np.save(os.path.join(out_dir, fname), fcs_arr)
            print(f"  Saved {num_fcs} FCs → {os.path.join(out_dir, fname)}")


if __name__ == "__main__":
    main()
