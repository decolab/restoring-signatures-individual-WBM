"""
Signal Processing
=================

Functions for processing BOLD fMRI signals and computing functional
connectivity (FC) matrices.

The FC pipeline:
    1. Detrend (remove linear trends)
    2. Bandpass filter (Butterworth, keep 0.0025-0.05 Hz)
    3. Z-score normalise
    4. Pearson correlation → FC matrix

Also includes peak-frequency extraction via FFT.

Dependencies: numpy, scipy
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d


# ============================================================================
#  Z-score normalisation
# ============================================================================

def zscore(array):
    """Z-score normalisation along the last axis.

    Subtracts the mean and divides by the standard deviation, so that
    each time series has mean=0 and std=1.

    Parameters
    ----------
    array : ndarray
        Input array (any shape). Normalisation is along the last axis.

    Returns
    -------
    ndarray
        Z-scored array (same shape as input).
    """
    mean = array.mean(axis=-1, keepdims=True)
    std = array.std(axis=-1, keepdims=True)
    std[std == 0.0] = np.finfo(float).eps  # Avoid division by zero
    return (array - mean) / std


# ============================================================================
#  Butterworth bandpass filter
# ============================================================================

def bandpass_filter(data, time_repetition, lowpass=0.0025, highpass=0.05,
                    order=2, detrend_first=False):
    """Apply a bandpass Butterworth filter to BOLD timeseries.

    Parameters
    ----------
    data : ndarray, shape (n_trials, n_regions, n_timepoints)
        Input BOLD data.
    time_repetition : float
        TR in seconds (sampling interval).
    lowpass : float
        Lower frequency bound in Hz (default: 0.0025).
    highpass : float
        Upper frequency bound in Hz (default: 0.05).
    order : int
        Filter order (default: 2).
    detrend_first : bool
        If True, apply linear detrending before filtering.

    Returns
    -------
    ndarray
        Filtered data (same shape as input).
    """
    # Compute filter coefficients
    nyquist = 1.0 / (2.0 * time_repetition)
    wn = np.array([lowpass / nyquist, highpass / nyquist])
    b, a = signal.butter(order, wn, btype="band")

    # Demean
    filtered = data - np.nanmean(data, axis=-1, keepdims=True)

    # Optional detrending
    if detrend_first:
        filtered = signal.detrend(filtered, axis=-1, type="linear")

    # Apply zero-phase filtering (forward + backward)
    filtered = signal.filtfilt(b, a, filtered, axis=-1)
    return filtered


# ============================================================================
#  Functional connectivity (Pearson correlation)
# ============================================================================

def correlate_timeseries(timeseries):
    """Compute Pearson correlation matrices for a batch of timeseries.

    Parameters
    ----------
    timeseries : ndarray, shape (n_trials, n_regions, n_timepoints)

    Returns
    -------
    ndarray, shape (n_trials, n_regions, n_regions)
        Correlation (FC) matrices.
    """
    # Demean
    ts = timeseries - timeseries.mean(axis=2, keepdims=True)
    # Normalise
    std = timeseries.std(axis=2, keepdims=True) + 1e-8
    ts = ts / std
    # Pearson correlation via einsum
    fc = np.einsum("ijk,ilk->ijl", ts, ts) / ts.shape[2]
    return fc


def compute_fc(time_series, time_repetition, lowpass=0.0025, highpass=0.05,
               order=2):
    """Full FC pipeline: detrend → bandpass → z-score → correlate.

    This is the main function to compute functional connectivity from
    raw BOLD timeseries.

    Parameters
    ----------
    time_series : ndarray, shape (n_trials, n_regions, n_timepoints)
        Raw BOLD data.
    time_repetition : float
        TR in seconds.
    lowpass : float
        Lower frequency bound in Hz.
    highpass : float
        Upper frequency bound in Hz.
    order : int
        Butterworth filter order.

    Returns
    -------
    ndarray, shape (n_trials, n_regions, n_regions)
        Functional connectivity matrices.

    Example
    -------
    >>> fcs = compute_fc(bold_data, time_repetition=1.25)
    >>> mean_fc = fcs.mean(axis=0)  # Average across trials
    """
    # Step 1: Detrend
    ts = signal.detrend(time_series, axis=-1, type="linear")

    # Step 2: Bandpass filter
    ts = bandpass_filter(ts, time_repetition, lowpass, highpass, order)

    # Step 3: Z-score
    ts = zscore(ts)

    # Step 4: Correlate
    return correlate_timeseries(ts)


# ============================================================================
#  Min-max normalisation
# ============================================================================

def normalize_minmax(matrices):
    """Min-max normalise each matrix independently to [0, 1].

    Parameters
    ----------
    matrices : ndarray, shape (N, H, W)
        Batch of 2D matrices.

    Returns
    -------
    ndarray, shape (N, H, W)
        Each matrix scaled so min=0, max=1.
    """
    mins = matrices.min(axis=(1, 2), keepdims=True)
    maxs = matrices.max(axis=(1, 2), keepdims=True)
    rng = maxs - mins
    rng[rng == 0] = 1.0
    return (matrices - mins) / rng


# ============================================================================
#  Peak-frequency extraction (FFT)
# ============================================================================

def compute_peak_frequencies(timeseries, time_repetition, smooth_sigma=0.01,
                             lowpass=None, highpass=None, filter_order=2):
    """Extract the peak oscillation frequency per brain region using FFT.

    For each region, this function:
      1. Detrends and demeans each trial
      2. (Optionally) bandpass-filters
      3. Computes the power spectrum via FFT
      4. Averages power across trials
      5. Smooths the spectrum
      6. Finds the peak frequency

    Parameters
    ----------
    timeseries : ndarray, shape (n_trials, n_regions, n_timepoints)
        BOLD data.
    time_repetition : float
        TR in seconds.
    smooth_sigma : float
        Gaussian smoothing width in Hz (default: 0.01).
    lowpass, highpass : float or None
        If both given, apply bandpass filter before FFT.
    filter_order : int
        Butterworth filter order (only if filtering).

    Returns
    -------
    ndarray, shape (n_regions,)
        Peak frequency in Hz for each brain region.
    """
    n_trials, n_regions, n_timepoints = timeseries.shape

    # Detrend per-trial
    ts = signal.detrend(timeseries, axis=-1, type="linear")

    # Demean
    ts = ts - ts.mean(axis=-1, keepdims=True)

    # Optional bandpass filter
    if lowpass is not None and highpass is not None:
        nyquist = 1.0 / (2.0 * time_repetition)
        wn = np.array([lowpass / nyquist, highpass / nyquist])
        b, a = signal.butter(filter_order, wn, btype="band")
        ts = signal.filtfilt(b, a, ts, axis=-1)

    # FFT
    freqs = np.fft.rfftfreq(n_timepoints, d=time_repetition)
    power = np.abs(np.fft.rfft(ts, axis=-1)) ** 2  # (trials, regions, freq_bins)

    # Average across trials
    mean_power = power.mean(axis=0)  # (regions, freq_bins)

    # Convert sigma from Hz to frequency bins
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    sigma_bins = smooth_sigma / df if df > 0 else 1.0

    # Find peak per region
    peak_freqs = np.zeros(n_regions)
    for r in range(n_regions):
        smoothed = gaussian_filter1d(mean_power[r], sigma=sigma_bins)
        # Skip the DC component (index 0)
        peak_idx = np.argmax(smoothed[1:]) + 1
        peak_freqs[r] = freqs[peak_idx]

    print(f"  Peak frequencies: min={peak_freqs.min():.4f} Hz, "
          f"max={peak_freqs.max():.4f} Hz, mean={peak_freqs.mean():.4f} Hz")

    return peak_freqs
