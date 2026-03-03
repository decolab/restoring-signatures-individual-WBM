#!/usr/bin/env python3
"""
Step 07 – Train the Variational Autoencoder (VAE).

Trains a VAE on augmented FC matrices from selected conditions
(anesthesia, CT 5V, VL 5V by default) and saves the trained weights.

The VAE learns to compress 82×82 FC matrices into a 2D latent space,
which allows us to visualise and compare brain states.

Run with:
    python 07_train_vae.py

Note: Requires TensorFlow. Install with: pip install tensorflow
"""

import os
import yaml
import numpy as np
from vae import train_vae

# ---- Load configuration ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "config.yaml")) as f:
    CONFIG = yaml.safe_load(f)


def main():
    vae_cfg = CONFIG["vae"]

    # Load augmented FCs and labels
    fcs = np.load(os.path.join(SCRIPT_DIR, CONFIG["paths"]["augmented_fcs"]))
    labels = np.load(os.path.join(SCRIPT_DIR, CONFIG["paths"]["augmented_labels"]),
                     allow_pickle=True)

    training_conditions = vae_cfg["training_conditions"]
    print(f"Training conditions: {training_conditions}")

    # Filter to training conditions only
    mask = np.isin(labels, training_conditions)
    fcs_train = fcs[mask]
    labels_train = labels[mask]
    print(f"Training data: {fcs_train.shape[0]} samples "
          f"(from {fcs.shape[0]} total)")

    # Train
    vae, encoder, decoder, history, train_idx, test_idx = train_vae(
        fcs_train,
        labels=labels_train,
        latent_dim=vae_cfg["latent_dim"],
        intermediate_dim=vae_cfg["intermediate_dim"],
        batch_size=vae_cfg["batch_size"],
        epochs=vae_cfg["epochs"],
        train_split=vae_cfg["train_split"],
    )

    # Save weights
    weights_path = os.path.join(SCRIPT_DIR, CONFIG["paths"]["vae_weights"])
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    vae.save_weights(weights_path)
    print(f"\nVAE weights saved → {weights_path}")

    # Save encoder and decoder separately
    enc_path = weights_path.replace("vae_weights", "encoder_weights")
    dec_path = weights_path.replace("vae_weights", "decoder_weights")
    encoder.save_weights(enc_path)
    decoder.save_weights(dec_path)

    # Save training history
    hist_path = weights_path.replace("vae_weights.weights.h5", "training_history.npy")
    np.save(hist_path, history.history, allow_pickle=True)
    print(f"Training history saved → {hist_path}")


if __name__ == "__main__":
    main()
