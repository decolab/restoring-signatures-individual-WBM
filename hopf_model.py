"""
Hopf (Stuart-Landau) Whole-Brain Model
=======================================

This file simulates the brain as a network of coupled oscillators.

Each brain region is modelled as a Stuart-Landau oscillator (a simplified
model that can switch between noisy and oscillatory behaviour). The
oscillators are connected through the structural connectivity (SC) matrix,
which represents the physical wiring of the brain.

There are 3 types of simulation:
    1. Baseline    — no external perturbation
    2. Wave        — a periodic driving signal applied to selected regions
    3. Noise/Sync  — the bifurcation parameter is shifted at selected regions

The maths (Euler-Maruyama integration):
    dz = [a*z + z_flip*omega - z*(|z|^2) + G*SC@z - G*rowsum*z] * dt + noise

where:
    z       = complex state of each oscillator (stored as 2 columns: x, y)
    a       = bifurcation parameter (-1 = noisy, +1 = oscillatory)
    omega   = natural frequency of each oscillator
    G       = global coupling strength
    SC      = structural connectivity matrix
    noise   = Gaussian white noise

Reference:
    Deco et al., "Perturbation of whole-brain dynamics in silico reveals
    mechanistic differences between brain states", NeuroImage (2018).
"""

import secrets
import numpy as np
from concurrent.futures import ProcessPoolExecutor


# ============================================================================
#  Helper functions
# ============================================================================

def get_random_seeds(n):
    """Generate n cryptographically-random seeds for reproducible simulations.

    Parameters
    ----------
    n : int
        Number of seeds to generate.

    Returns
    -------
    list of int
        Random 32-bit integer seeds.
    """
    return [secrets.randbelow(2**32 - 1) for _ in range(n)]


def _prepare_simulation(num_parcels, global_coupling, structural_connectivity,
                        bifurcation_parameters, frequencies, dt, sig, seed):
    """Set up all the arrays needed before running a simulation.

    This is shared by all three simulation types (baseline, wave, noisesync).

    Returns
    -------
    omega, weighted_conn, sum_conn, a, dt, dsig : arrays and scalars
    """
    # Set the random seed
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()

    # Noise scaling
    dsig = np.sqrt(dt) * sig

    # Omega: natural frequencies as a (num_parcels, 2) array
    # Column 0 = -omega (for the y->x coupling), Column 1 = +omega (for x->y)
    omega = np.tile(2 * np.pi * frequencies, (2, 1)).T
    omega[:, 0] *= -1

    # Weighted structural connectivity
    weighted_conn = global_coupling * structural_connectivity
    sum_conn = np.tile(weighted_conn.sum(axis=1, keepdims=True), (1, 2))

    # Bifurcation parameter: expand to (num_parcels, 2)
    a = np.tile(bifurcation_parameters.T, (1, 2))

    return omega, weighted_conn, sum_conn, a, dt, dsig


def _burn_in_transient(z, omega, weighted_conn, sum_conn, a, dt, dsig,
                       num_parcels, duration=2000.0):
    """Run the simulation for a transient period (to reach steady state).

    We throw away this initial period because the oscillators need time
    to settle into their natural dynamics after being initialised.

    Parameters
    ----------
    z : ndarray, shape (num_parcels, 2)
        Current state of all oscillators.
    duration : float
        Length of transient period in time-units (default: 2000).

    Returns
    -------
    z : ndarray
        Updated state after transient.
    """
    for _ in np.arange(0, duration + dt, dt):
        zz = z[:, ::-1]  # Swap columns (x↔y coupling)
        interaction = weighted_conn @ z - sum_conn * z
        bifur_freq = a * z + zz * omega
        amplitude_sq = z * (z * z + zz * zz)
        noise = dsig * np.random.normal(0.0, 1.0, (num_parcels, 2))
        z = z + dt * (bifur_freq - amplitude_sq + interaction) + noise
    return z


# ============================================================================
#  Simulation functions
# ============================================================================

def hopf_simulate(time_points, time_repetition, num_parcels, global_coupling,
                  structural_connectivity, bifurcation_parameters, frequencies,
                  dt=0.01, sig=0.04, seed=None):
    """Run a baseline Hopf simulation (no perturbation).

    Parameters
    ----------
    time_points : int
        Number of BOLD samples to record.
    time_repetition : float
        TR in seconds (sampling interval).
    num_parcels : int
        Number of brain regions.
    global_coupling : float
        Global coupling parameter G.
    structural_connectivity : ndarray, shape (N, N)
        Structural connectivity matrix.
    bifurcation_parameters : ndarray, shape (1, N)
        Per-region bifurcation parameter a.
    frequencies : ndarray, shape (N,)
        Natural oscillation frequency per region (Hz).
    dt : float
        Integration time step (default: 0.01 s).
    sig : float
        Noise amplitude (default: 0.04).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    ndarray, shape (N, time_points)
        Simulated BOLD-like signal.
    """
    omega, weighted_conn, sum_conn, a, dt, dsig = _prepare_simulation(
        num_parcels, global_coupling, structural_connectivity,
        bifurcation_parameters, frequencies, dt, sig, seed
    )

    # Initialise oscillators
    z = 0.1 * np.ones((num_parcels, 2))

    # Burn in transient
    z = _burn_in_transient(z, omega, weighted_conn, sum_conn, a, dt, dsig,
                           num_parcels)

    # Record the actual signal
    xs = np.zeros((time_points, num_parcels))
    sample_index = 0
    steps_since_last_sample = 0

    while sample_index < time_points:
        zz = z[:, ::-1]
        interaction = weighted_conn @ z - sum_conn * z
        bifur_freq = a * z + zz * omega
        amplitude_sq = z * (z * z + zz * zz)
        noise = dsig * np.random.normal(0.0, 1.0, (num_parcels, 2))
        z = z + dt * (bifur_freq - amplitude_sq + interaction) + noise

        steps_since_last_sample += 1
        if steps_since_last_sample >= time_repetition / dt:
            steps_since_last_sample = 0
            xs[sample_index, :] = z[:, 0]  # Record x-component
            sample_index += 1

    return xs.T  # shape: (num_parcels, time_points)


def hopf_simulate_wave(time_points, time_repetition, num_parcels,
                       global_coupling, structural_connectivity,
                       bifurcation_parameters, frequencies,
                       pert_strength=0.0, perturbation_indices=None,
                       pert_frequencies=None,
                       dt=0.01, sig=0.04, seed=None):
    """Run a Hopf simulation with periodic (wave) perturbation.

    A cosine/sine driving signal is applied to selected brain regions.

    Parameters
    ----------
    pert_strength : float
        Amplitude of the wave perturbation (0 = no perturbation).
    perturbation_indices : list or array
        Which brain regions receive the perturbation.
    pert_frequencies : ndarray or None
        Perturbation frequencies. If None, uses the natural frequencies.

    Other parameters are the same as hopf_simulate().

    Returns
    -------
    ndarray, shape (N, time_points)
    """
    omega, weighted_conn, sum_conn, a, dt, dsig = _prepare_simulation(
        num_parcels, global_coupling, structural_connectivity,
        bifurcation_parameters, frequencies, dt, sig, seed
    )

    z = 0.1 * np.ones((num_parcels, 2))
    z = _burn_in_transient(z, omega, weighted_conn, sum_conn, a, dt, dsig,
                           num_parcels)

    # Build the perturbation mask
    if perturbation_indices is None:
        perturbation_indices = []
    pert_term = np.zeros((num_parcels, 2))
    pert_term[perturbation_indices, :] = pert_strength

    # Perturbation frequency
    if pert_frequencies is None:
        omega_p = omega
    else:
        omega_p = np.tile(2 * np.pi * pert_frequencies, (2, 1)).T
        omega_p[:, 0] *= -1

    def stimulus(t):
        """Compute the periodic driving stimulus at time t."""
        return pert_term * np.append(
            np.cos(omega_p * t), np.sin(omega_p * t), axis=1
        )[:, ::-2]

    # Record
    xs = np.zeros((time_points, num_parcels))
    sample_index = 0
    steps_since_last_sample = 0
    t = 0.0

    while sample_index < time_points:
        zz = z[:, ::-1]
        interaction = weighted_conn @ z - sum_conn * z
        bifur_freq = a * z + zz * omega
        amplitude_sq = z * (z * z + zz * zz)
        noise = dsig * np.random.normal(0.0, 1.0, (num_parcels, 2))
        z = z + dt * (bifur_freq - amplitude_sq + interaction + stimulus(t)) + noise

        steps_since_last_sample += 1
        if steps_since_last_sample >= time_repetition / dt:
            steps_since_last_sample = 0
            xs[sample_index, :] = z[:, 0]
            sample_index += 1
        t += dt

    return xs.T


def hopf_simulate_noisesync(time_points, time_repetition, num_parcels,
                            global_coupling, structural_connectivity,
                            bifurcation_parameters, frequencies,
                            pert_strength=0.0, perturbation_indices=None,
                            dt=0.01, sig=0.04, seed=None):
    """Run a Hopf simulation with noise/synchronisation perturbation.

    The bifurcation parameter 'a' is shifted at selected regions.
    Positive shift → more oscillatory (synchronisation).
    Negative shift → more noisy.

    Parameters
    ----------
    pert_strength : float
        Shift applied to the bifurcation parameter at target regions.
    perturbation_indices : list or array
        Which brain regions receive the perturbation.

    Other parameters are the same as hopf_simulate().

    Returns
    -------
    ndarray, shape (N, time_points)
    """
    omega, weighted_conn, sum_conn, a, dt, dsig = _prepare_simulation(
        num_parcels, global_coupling, structural_connectivity,
        bifurcation_parameters, frequencies, dt, sig, seed
    )

    z = 0.1 * np.ones((num_parcels, 2))
    z = _burn_in_transient(z, omega, weighted_conn, sum_conn, a, dt, dsig,
                           num_parcels)

    # Apply the bifurcation shift
    if perturbation_indices is None:
        perturbation_indices = []
    a[perturbation_indices, :] += pert_strength

    # Record
    xs = np.zeros((time_points, num_parcels))
    sample_index = 0
    steps_since_last_sample = 0

    while sample_index < time_points:
        zz = z[:, ::-1]
        interaction = weighted_conn @ z - sum_conn * z
        bifur_freq = a * z + zz * omega
        amplitude_sq = z * (z * z + zz * zz)
        noise = dsig * np.random.normal(0.0, 1.0, (num_parcels, 2))
        z = z + dt * (bifur_freq - amplitude_sq + interaction) + noise

        steps_since_last_sample += 1
        if steps_since_last_sample >= time_repetition / dt:
            steps_since_last_sample = 0
            xs[sample_index, :] = z[:, 0]
            sample_index += 1

    return xs.T


# ============================================================================
#  Running multiple simulations
# ============================================================================

# These top-level functions are needed for parallel processing (pickle).
def _run_single_baseline(args):
    return hopf_simulate(*args)

def _run_single_wave(args):
    return hopf_simulate_wave(*args)

def _run_single_noisesync(args):
    return hopf_simulate_noisesync(*args)


def run_simulations(n, time_points, time_repetition, num_parcels,
                    global_coupling, structural_connectivity,
                    bifurcation_parameters, frequencies,
                    dt=0.01, sig=0.04, use_parallel=False, max_workers=4):
    """Run n baseline simulations (sequential or parallel).

    Parameters
    ----------
    n : int
        Number of simulations to run.
    use_parallel : bool
        If True, uses multiple CPU cores (faster but harder to debug).
    max_workers : int
        Number of CPU cores for parallel mode.

    Returns
    -------
    ndarray, shape (n, num_parcels, time_points)
    """
    seeds = get_random_seeds(n)

    if use_parallel:
        arglist = [
            (time_points, time_repetition, num_parcels, global_coupling,
             structural_connectivity, bifurcation_parameters, frequencies,
             dt, sig, s)
            for s in seeds
        ]
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_run_single_baseline, arglist))
        return np.stack(results)
    else:
        results = np.zeros((n, num_parcels, time_points))
        for i, seed in enumerate(seeds):
            results[i] = hopf_simulate(
                time_points, time_repetition, num_parcels, global_coupling,
                structural_connectivity, bifurcation_parameters, frequencies,
                dt, sig, seed
            )
            if (i + 1) % 5 == 0:
                print(f"  Simulation {i+1}/{n} done")
        return results


def run_simulations_wave(n, time_points, time_repetition, num_parcels,
                         global_coupling, structural_connectivity,
                         bifurcation_parameters, frequencies,
                         pert_strength=0.0, perturbation_indices=None,
                         pert_frequencies=None,
                         dt=0.01, sig=0.04, use_parallel=False, max_workers=4):
    """Run n wave-perturbation simulations.

    Returns
    -------
    ndarray, shape (n, num_parcels, time_points)
    """
    seeds = get_random_seeds(n)

    if use_parallel:
        arglist = [
            (time_points, time_repetition, num_parcels, global_coupling,
             structural_connectivity, bifurcation_parameters, frequencies,
             pert_strength, perturbation_indices, pert_frequencies,
             dt, sig, s)
            for s in seeds
        ]
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_run_single_wave, arglist))
        return np.stack(results)
    else:
        results = np.zeros((n, num_parcels, time_points))
        for i, seed in enumerate(seeds):
            results[i] = hopf_simulate_wave(
                time_points, time_repetition, num_parcels, global_coupling,
                structural_connectivity, bifurcation_parameters, frequencies,
                pert_strength, perturbation_indices, pert_frequencies,
                dt, sig, seed
            )
        return results


def run_simulations_noisesync(n, time_points, time_repetition, num_parcels,
                              global_coupling, structural_connectivity,
                              bifurcation_parameters, frequencies,
                              pert_strength=0.0, perturbation_indices=None,
                              dt=0.01, sig=0.04,
                              use_parallel=False, max_workers=4):
    """Run n noise/sync-perturbation simulations.

    Returns
    -------
    ndarray, shape (n, num_parcels, time_points)
    """
    seeds = get_random_seeds(n)

    if use_parallel:
        arglist = [
            (time_points, time_repetition, num_parcels, global_coupling,
             structural_connectivity, bifurcation_parameters, frequencies,
             pert_strength, perturbation_indices, dt, sig, s)
            for s in seeds
        ]
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_run_single_noisesync, arglist))
        return np.stack(results)
    else:
        results = np.zeros((n, num_parcels, time_points))
        for i, seed in enumerate(seeds):
            results[i] = hopf_simulate_noisesync(
                time_points, time_repetition, num_parcels, global_coupling,
                structural_connectivity, bifurcation_parameters, frequencies,
                pert_strength, perturbation_indices, dt, sig, seed
            )
        return results
