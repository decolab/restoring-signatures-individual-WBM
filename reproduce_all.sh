#!/usr/bin/env bash
# ============================================================================
#  Run the complete pipeline end-to-end
#
#  Usage:
#    chmod +x reproduce_all.sh
#    ./reproduce_all.sh
#
#  Prerequisites:
#    1. Python >= 3.8 with dependencies from requirements.txt
#    2. Data files in data/ (see README.md)
#    3. TensorFlow installed (for steps 07-08)
# ============================================================================

set -euo pipefail

echo "============================================================"
echo "  Whole-Brain Model – Full Reproduction Pipeline"
echo "============================================================"

echo ""
echo "[1/8] Computing peak frequencies..."
python 01_compute_frequencies.py

echo ""
echo "[2/8] Computing empirical functional connectivity..."
python 02_compute_empirical_fc.py

echo ""
echo "[3/8] Exploring global coupling parameter G..."
python 03_explore_global_coupling.py

echo ""
echo "[4/8] Fitting bifurcation parameters (genetic algorithm)..."
python 04_fit_bifurcation_params.py

echo ""
echo "[5/8] Data augmentation (generating surrogate FCs)..."
python 05_data_augmentation.py

echo ""
echo "[6/8] Running model perturbations..."
python 06_model_perturbation.py

echo ""
echo "[7/8] Training the VAE..."
python 07_train_vae.py

echo ""
echo "[8/8] Latent-space analysis..."
python 08_latent_space_analysis.py

echo ""
echo "============================================================"
echo "  Pipeline complete! Results are in outputs/"
echo "============================================================"
