#!/usr/bin/env python3
"""
Step 08 – Latent-space analysis.

Encodes all augmented and perturbation FCs into the VAE latent space,
computes brain-state metrics, and saves everything for plotting.

Run with:
    python 08_latent_space_analysis.py

Note: Requires TensorFlow. Install with: pip install tensorflow
"""

import os
import yaml
import numpy as np
from vae import build_vae, encode
from metrics import compute_all_metrics

# ---- Load configuration ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "config.yaml")) as f:
    CONFIG = yaml.safe_load(f)


def main():
    vae_cfg = CONFIG["vae"]
    mc = CONFIG["model"]

    # Load trained VAE
    weights_path = os.path.join(SCRIPT_DIR, CONFIG["paths"]["vae_weights"])
    if not os.path.exists(weights_path):
        print(f"ERROR: VAE weights not found: {weights_path}")
        print("       Run step 07 first.")
        return

    n_regions = mc["num_parcels"]
    original_dim = n_regions * n_regions
    vae, encoder, decoder = build_vae(
        original_dim=original_dim,
        intermediate_dim=vae_cfg["intermediate_dim"],
        latent_dim=vae_cfg["latent_dim"],
    )
    vae.build((None, original_dim))
    vae.load_weights(weights_path)
    print(f"VAE weights loaded from {weights_path}")

    # Load SC for FS-delta metric
    sc = np.loadtxt(os.path.join(SCRIPT_DIR, CONFIG["paths"]["structural_connectivity"]))
    sc_max = mc.get("sc_normalisation_max", None)
    if sc_max is not None:
        sc = sc / sc.max() * sc_max

    out_dir = os.path.join(SCRIPT_DIR, "outputs", "latent_space")
    os.makedirs(out_dir, exist_ok=True)

    # --- Encode augmented FCs ---
    print("\nEncoding augmented FCs...")
    fcs = np.load(os.path.join(SCRIPT_DIR, CONFIG["paths"]["augmented_fcs"]))
    labels = np.load(os.path.join(SCRIPT_DIR, CONFIG["paths"]["augmented_labels"]),
                     allow_pickle=True)

    z_mean, z_log_var, z = encode(encoder, fcs)
    print(f"  Encoded {len(fcs)} FCs, latent shape: {z_mean.shape}")

    np.save(os.path.join(out_dir, "augmented_z_mean.npy"), z_mean)
    np.save(os.path.join(out_dir, "augmented_z.npy"), z)
    np.save(os.path.join(out_dir, "augmented_labels.npy"), labels)

    # Metrics for augmented FCs
    print("  Computing metrics...")
    metrics = compute_all_metrics(fcs, sc=sc)
    for name, vals in metrics.items():
        np.save(os.path.join(out_dir, f"augmented_{name}.npy"), vals)
    print("  Augmented metrics saved.")

    # --- Encode perturbation FCs ---
    pert_base = os.path.join(SCRIPT_DIR, CONFIG["paths"]["perturbation_fcs"])
    if os.path.exists(pert_base):
        print("\nEncoding perturbation FCs...")
        for pert_type in sorted(os.listdir(pert_base)):
            pert_type_dir = os.path.join(pert_base, pert_type)
            if not os.path.isdir(pert_type_dir):
                continue
            for target in sorted(os.listdir(pert_type_dir)):
                target_dir = os.path.join(pert_type_dir, target)
                if not os.path.isdir(target_dir):
                    continue
                for amp_file in sorted(os.listdir(target_dir)):
                    if not amp_file.endswith(".npy"):
                        continue
                    pfcs = np.load(os.path.join(target_dir, amp_file))
                    pz_mean, _, pz = encode(encoder, pfcs)

                    p_out = os.path.join(out_dir, "perturbation",
                                         pert_type, target)
                    os.makedirs(p_out, exist_ok=True)
                    stem = amp_file.replace(".npy", "")
                    np.save(os.path.join(p_out, f"{stem}_z_mean.npy"), pz_mean)
                    np.save(os.path.join(p_out, f"{stem}_z.npy"), pz)

                    p_metrics = compute_all_metrics(pfcs, sc=sc)
                    for name, vals in p_metrics.items():
                        np.save(os.path.join(p_out, f"{stem}_{name}.npy"), vals)

                    print(f"  {pert_type}/{target}/{stem}: {len(pfcs)} FCs encoded")
    else:
        print(f"\nNo perturbation FCs found at {pert_base}")

    print("\nLatent-space analysis complete!")


if __name__ == "__main__":
    main()
