"""
Brain-State Metrics
===================

Functions that summarise a brain state (functional connectivity matrix)
with a single number. These metrics help compare different conditions
(e.g., anesthesia vs. awake) and evaluate model fits.

Available metrics:
    - mean_fc:         Average correlation strength
    - modularity:      How modular the brain network is (community structure)
    - fs_delta:        FC-SC dissimilarity (functional vs structural)
    - compute_ssim:    Structural similarity between two FC matrices
    - irreversibility: Time-reversal asymmetry (non-equilibrium dynamics)
    - cohen_d:         Effect size between two groups

Dependencies: numpy, networkx (for modularity), scikit-image (for SSIM)
"""

import math
import statistics
import numpy as np

# Optional imports — these libraries are only needed for specific metrics.
# The code won't crash if they're missing; it will just raise an error
# when you try to use the functions that need them.

try:
    import networkx as nx
    from networkx.algorithms.community import (
        greedy_modularity_communities,
        louvain_communities,
    )
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Note: networkx not installed. modularity() won't work.")
    print("      Install with: pip install networkx")

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Note: scikit-image not installed. compute_ssim() won't work.")
    print("      Install with: pip install scikit-image")


# ============================================================================
#  Basic FC metrics
# ============================================================================

def mean_fc(fc):
    """Average of the upper-triangular elements of an FC matrix.

    This gives a single number summarising the overall correlation
    strength in the brain network.

    Parameters
    ----------
    fc : ndarray, shape (N, N)
        Functional connectivity matrix.

    Returns
    -------
    float
        Mean upper-triangular FC value.
    """
    idx = np.triu_indices(fc.shape[0], k=1)
    return float(np.mean(fc[idx]))


def modularity(fc):
    """Newman modularity Q of a functional connectivity network.

    Uses greedy modularity maximisation on the positive-weight FC graph.
    Higher values = more modular (the brain has clearer communities).

    Parameters
    ----------
    fc : ndarray, shape (N, N)
        Functional connectivity matrix.

    Returns
    -------
    float
        Modularity Q value.

    Notes
    -----
    Requires networkx. Install with: pip install networkx
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx is needed for modularity. "
                          "Install with: pip install networkx")

    # Only keep positive weights, remove diagonal
    adj = fc.copy()
    np.fill_diagonal(adj, 0.0)
    adj[adj < 0] = 0.0

    G = nx.from_numpy_array(adj)
    communities = greedy_modularity_communities(G)
    return float(nx.community.modularity(G, communities))


def modularity_louvain(fc, n_runs=10, seed_start=0):
    """Mean Louvain modularity averaged over multiple random runs.

    The Louvain algorithm is stochastic — different seeds give different
    partitions. We average over n_runs for a more stable estimate.

    Parameters
    ----------
    fc : ndarray, shape (N, N)
    n_runs : int
        Number of Louvain runs to average (default: 10).
    seed_start : int
        First random seed.

    Returns
    -------
    float
        Mean modularity Q.
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx is needed for modularity.")

    adj = fc.copy()
    np.fill_diagonal(adj, 0.0)
    adj[adj < 0] = 0.0
    G = nx.from_numpy_array(adj)

    q_values = []
    for seed in range(seed_start, seed_start + n_runs):
        communities = louvain_communities(G, seed=seed)
        q = float(nx.community.modularity(G, communities))
        q_values.append(q)

    return float(np.mean(q_values))


def fs_delta(fc, sc):
    """Functional-Structural delta: mean |FC - SC| in the upper triangle.

    Measures how different the functional connectivity is from the
    structural (anatomical) connectivity.

    Parameters
    ----------
    fc : ndarray, shape (N, N)
        Functional connectivity matrix.
    sc : ndarray, shape (N, N)
        Structural connectivity matrix.

    Returns
    -------
    float
    """
    idx = np.triu_indices(fc.shape[0], k=1)
    return float(np.mean(np.abs(fc[idx] - sc[idx])))


# ============================================================================
#  SSIM (goodness of fit between two FC matrices)
# ============================================================================

def compute_ssim(fc_empirical, fc_simulated, data_range=None):
    """Structural Similarity Index (SSIM) between two FC matrices.

    SSIM measures how visually similar two images (matrices) are.
    Values range from -1 to 1, where 1 = identical.

    Parameters
    ----------
    fc_empirical : ndarray, shape (N, N)
    fc_simulated : ndarray, shape (N, N)
    data_range : float or None
        Dynamic range. If None, uses max-min of the empirical matrix.

    Returns
    -------
    float
        SSIM value (higher = more similar).

    Notes
    -----
    Requires scikit-image. Install with: pip install scikit-image
    """
    if not HAS_SKIMAGE:
        raise ImportError("scikit-image is needed for SSIM. "
                          "Install with: pip install scikit-image")
    if data_range is None:
        data_range = float(fc_empirical.max() - fc_empirical.min())
    return float(ssim(fc_empirical, fc_simulated, data_range=data_range))


# ============================================================================
#  Irreversibility (time-reversal asymmetry)
# ============================================================================

def irreversibility(timeseries, tau=1):
    """Time-reversal asymmetry of BOLD timeseries.

    Measures non-equilibrium dynamics by comparing the lagged
    cross-covariance matrix C(tau) with its transpose C(tau)^T.
    A system in equilibrium has C(tau) = C(tau)^T (detailed balance).

    The metric is:  ||C(tau) - C(tau)^T||_F / N

    where N = number of regions and ||.||_F = Frobenius norm.

    Parameters
    ----------
    timeseries : ndarray, shape (n_regions, n_timepoints)
        Single-trial BOLD timeseries.
    tau : int
        Lag in samples (default: 1).

    Returns
    -------
    float
        Irreversibility (0 = fully reversible/equilibrium).

    Reference
    ---------
    Deco G, Sanz Perl Y, et al. (2022). Nature Human Behaviour, 7, 1-11.
    """
    n_regions, n_tp = timeseries.shape

    # Demean
    ts = timeseries - timeseries.mean(axis=1, keepdims=True)

    # Lagged cross-covariance: C[i,j] = <x_i(t) * x_j(t+tau)>
    n_eff = n_tp - tau
    c_tau = (ts[:, :n_eff] @ ts[:, tau:].T) / n_eff

    # Asymmetry
    asymmetry = c_tau - c_tau.T
    return float(np.linalg.norm(asymmetry, "fro") / n_regions)


def batch_irreversibility(timeseries_batch, tau=1):
    """Compute irreversibility for a batch of timeseries.

    Parameters
    ----------
    timeseries_batch : ndarray, shape (n_trials, n_regions, n_timepoints)
    tau : int

    Returns
    -------
    ndarray, shape (n_trials,)
    """
    return np.array([
        irreversibility(timeseries_batch[i], tau)
        for i in range(timeseries_batch.shape[0])
    ])


# ============================================================================
#  Statistics helpers
# ============================================================================

def pooled_std(group1, group2):
    """Pooled standard deviation of two samples."""
    sd1 = statistics.stdev(group1)
    sd2 = statistics.stdev(group2)
    n1, n2 = len(group1), len(group2)
    return math.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))


def cohen_d(group1, group2):
    """Cohen's d effect size between two groups.

    Positive value means group1 > group2.

    Parameters
    ----------
    group1, group2 : array-like
        Two groups of measurements.

    Returns
    -------
    float
        Cohen's d.
    """
    return (np.mean(group1) - np.mean(group2)) / pooled_std(group1, group2)


# ============================================================================
#  Batch computation of all metrics
# ============================================================================

def compute_all_metrics(fcs, sc=None):
    """Compute all available metrics for a batch of FC matrices.

    Parameters
    ----------
    fcs : ndarray, shape (n, N, N)
        Batch of FC matrices.
    sc : ndarray, shape (N, N) or None
        Structural connectivity (needed for fs_delta).

    Returns
    -------
    dict
        Keys: "mean_fc", "modularity" (if networkx), "fs_delta" (if sc given).
        Values: ndarray of shape (n,).
    """
    n = fcs.shape[0]
    result = {}

    result["mean_fc"] = np.array([mean_fc(fcs[i]) for i in range(n)])

    if HAS_NETWORKX:
        result["modularity"] = np.array([modularity(fcs[i]) for i in range(n)])
        result["modularity_louvain"] = np.array(
            [modularity_louvain(fcs[i]) for i in range(n)]
        )

    if sc is not None:
        result["fs_delta"] = np.array([fs_delta(fcs[i], sc) for i in range(n)])

    return result
