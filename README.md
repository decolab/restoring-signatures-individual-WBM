# Restoring Signatures of Consciousness – Whole-Brain Model

Whole-brain computational model of deep brain stimulation (DBS) effects on
consciousness in anaesthetised macaques.

> **Paper:** *Restoring signatures of consciousness by thalamic stimulation
> in a whole-brain model of an anesthetized nonhuman primate*
> — Pérez-Ordoyo E., …, Sanz-Perl Y. (2025)

## What this code does

1. **Models the brain** as 82 coupled oscillators (Stuart-Landau/Hopf model)
   connected through real anatomical wiring (structural connectivity).
2. **Fits the model** to real fMRI data using a genetic algorithm.
3. **Generates surrogate data** by running the fitted model many times.
4. **Simulates brain stimulation** by perturbing specific brain regions.
5. **Maps brain states** into a 2D space using a variational autoencoder (VAE).

## Quick start

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your data files in data/ (see below)

# 4. Run the pipeline step by step:
python 01_compute_frequencies.py
python 02_compute_empirical_fc.py
python 03_explore_global_coupling.py
python 04_fit_bifurcation_params.py
python 05_data_augmentation.py
python 06_model_perturbation.py
python 07_train_vae.py          # needs tensorflow
python 08_latent_space_analysis.py  # needs tensorflow

# Or run everything at once:
chmod +x reproduce_all.sh
./reproduce_all.sh
```

## File structure

```
├── config.yaml                 ← All parameters (THE place to change things)
│
├── hopf_model.py               ← Brain simulator (Stuart-Landau oscillators)
├── signal_processing.py        ← Filtering, FC computation, peak frequencies
├── metrics.py                  ← Brain-state metrics (mean FC, modularity, etc.)
├── genetic_algorithm.py        ← GA for fitting model parameters
├── vae.py                      ← Variational autoencoder
│
├── 01_compute_frequencies.py   ← Step 1: Extract oscillation frequencies
├── 02_compute_empirical_fc.py  ← Step 2: Compute empirical FC matrices
├── 03_explore_global_coupling.py ← Step 3: Find optimal coupling G
├── 04_fit_bifurcation_params.py  ← Step 4: Fit bifurcation parameters
├── 05_data_augmentation.py     ← Step 5: Generate surrogate FCs
├── 06_model_perturbation.py    ← Step 6: Simulate brain stimulation
├── 07_train_vae.py             ← Step 7: Train the VAE
├── 08_latent_space_analysis.py ← Step 8: Map to latent space
├── reproduce_all.sh            ← Run everything
│
├── notebooks/                  ← Jupyter notebooks for figures
│   ├── figure2_model_fitting.ipynb
│   ├── figure3_latent_space.ipynb
│   ├── figure4_v2_perturbation.ipynb
│   ├── figure5_cortical_perturbation.ipynb
│   ├── figure6_gnw_perturbation.ipynb
│   └── supplementary_figures.ipynb
│
├── data/                       ← Input data (not in git)
│   ├── SC.txt                  ← Structural connectivity (82×82)
│   ├── ts_dict.npy             ← BOLD timeseries (all subjects)
│   └── metadata_DBS.tsv        ← Region metadata
│
└── outputs/                    ← Generated results (not in git)
    ├── figures/
    ├── empirical_fc/
    ├── g_exploration/
    ├── perturbation_fcs/
    └── latent_space/
```

## How the configuration works

**Everything** is controlled by `config.yaml`. Open it — it's heavily
commented. Here's the key idea:

```python
# At the top of every script:
import yaml
with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

# Then use it:
tr = CONFIG["model"]["time_repetition"]   # 1.25 seconds
```

If you want to change a parameter (e.g., number of simulations, filter
frequencies, GA settings), change it in `config.yaml` and it takes effect
everywhere.

## Data you need

Place these files in `data/`:

| File | What it is |
|------|-----------|
| `SC.txt` | 82×82 structural connectivity matrix (text format) |
| `ts_dict.npy` | Dictionary of BOLD timeseries for all subjects/conditions |
| `metadata_DBS.tsv` | Brain region metadata (optional, for some analyses) |

The timeseries dictionary has this structure:
```
ts_dict["taro"]["stim_off"]  →  numpy array (n_trials, 82, 500)
ts_dict["jade"]["awake_bold"] →  numpy array (n_trials, 82, 500)
...
```

## Key parameters (in config.yaml)

| Parameter | Value | What it means |
|-----------|-------|--------------|
| `model.num_parcels` | 82 | Brain regions in the atlas |
| `model.time_repetition` | 1.25 s | fMRI sampling interval (TR) |
| `model.dt` | 0.01 s | Simulation time step |
| `model.sig` | 0.04 | Noise amplitude |
| `model.use_parallel` | false | Set to true for faster computation |
| `filter.lowpass` | 0.0025 Hz | Bandpass filter lower bound |
| `filter.highpass` | 0.05 Hz | Bandpass filter upper bound |
| `ga.population_size` | 20 | Genomes per generation |
| `ga.genome_length` | 41 | Parameters per hemisphere |
| `vae.latent_dim` | 2 | 2D latent space |

## Parallel processing

By default, simulations run sequentially (easier to debug). To speed
things up, set `model.use_parallel: true` in `config.yaml` and adjust
`model.max_workers` to the number of CPU cores you want to use.

## Troubleshooting

**"TensorFlow not installed"** — Steps 07-08 need TensorFlow. Install with
`pip install tensorflow`. Steps 01-06 work without it.

**"networkx not installed"** — The modularity metric needs it. Install with
`pip install networkx`.

**"File not found"** — Make sure you run the steps in order (01→02→03→...).
Each step depends on the outputs of previous steps.

**Out of memory** — Reduce `model.num_subsimulations` in `config.yaml`
or set `model.use_parallel: false`.

## Dependencies

- **Always needed:** numpy, scipy, pyyaml
- **For metrics:** scikit-image (SSIM), networkx (modularity)
- **For plots:** matplotlib, seaborn
- **For VAE (steps 07-08 only):** tensorflow

## Citation

If you use this code, please cite:

> Pérez-Ordoyo Bellido, E.L. et al. "Restoring signatures of consciousness
> by thalamic stimulation in a whole-brain model of an anesthetized nonhuman
> primate." (2024).

## Contact

eider(dot)perezordoyo(at)upf(dot)edu

## Acknowledgements

The code was developed by Eider Pérez-Ordoyo, following previous code developed 
by Yonatan Sanz-Perl and Gustavo Deco and with assistance from Wiep Stikvoort.
We thank the open-source community for the libraries we used.
Claude Opus 4.6 (Anthropic) was used for refactoring and cleaning the code.

## License

See [LICENSE](LICENSE).
