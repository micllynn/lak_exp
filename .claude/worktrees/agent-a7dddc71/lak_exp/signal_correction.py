"""
Signal correction methods for dual-color two-photon imaging.

This module provides functions to correct a target signal (s2/green channel)
using a control signal (s1/red channel) to remove shared noise sources such
as motion artifacts, hemodynamic signals, or other common mode noise.

All functions expect input arrays of shape (T, X, Y) where:
- T: number of time frames
- X, Y: spatial dimensions (pixels)

Memory Optimization
-------------------
These functions are designed to work with memory-mapped arrays and minimize
RAM usage by streaming data in time batches. Key parameters:
- batch_size: Number of time frames to process at once (default: 500)
- dtype: Use np.int16 or np.float32 to reduce output size
- output_path: Write result directly to a memory-mapped TIFF on disk

Functions
---------
correct_linear_regression : Pixel-wise linear regression correction
correct_lms_adaptive : LMS adaptive filter correction
correct_pca_shared_variance : PCA-based shared variance removal
correct_ica_shared_components : ICA-based shared component removal
correct_nmf_shared_components : NMF-based shared component removal
"""

import numpy as np
import tifffile
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from sklearn.decomposition import IncrementalPCA, FastICA, MiniBatchNMF


# =============================================================================
# Helper functions
# =============================================================================

def _clip_to_dtype(arr, dtype):
    """
    Clip array values to fit within dtype range and convert.

    For integer dtypes, clips to [dtype.min, dtype.max].
    For float dtypes, just converts without clipping.

    Parameters
    ----------
    arr : np.ndarray
        Input array (typically float32)
    dtype : np.dtype
        Target dtype

    Returns
    -------
    np.ndarray
        Array converted to dtype, with values clipped if integer dtype
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(arr, info.min, info.max).astype(dtype)
    else:
        return arr.astype(dtype)


def _iter_pixel_chunks(n_pixels, chunk_size):
    """
    Yield (start, end, chunk_idx, n_chunks) for processing pixels in chunks.

    Parameters
    ----------
    n_pixels : int
        Total number of pixels
    chunk_size : int
        Number of pixels per chunk

    Yields
    ------
    tuple
        (start_idx, end_idx, chunk_idx, n_chunks)
    """
    n_chunks = (n_pixels + chunk_size - 1) // chunk_size
    chunk_idx = 0
    for start in range(0, n_pixels, chunk_size):
        end = min(start + chunk_size, n_pixels)
        yield start, end, chunk_idx, n_chunks
        chunk_idx += 1


def _huber_weights_vectorized(residuals, epsilon=1.35):
    """
    Compute Huber weights for robust regression (IRLS), vectorized.

    Parameters
    ----------
    residuals : np.ndarray
        Residual values, shape (T, n_pixels)
    epsilon : float
        Huber threshold parameter (default: 1.35)

    Returns
    -------
    np.ndarray
        Weights for each residual, shape (T, n_pixels)
    """
    abs_residuals = np.abs(residuals)
    # MAD scale estimate per pixel
    scale = np.median(abs_residuals, axis=0) * 1.4826
    scale = np.maximum(scale, 1e-10)

    scaled_residuals = abs_residuals / scale
    weights = np.where(
        scaled_residuals <= epsilon,
        np.ones_like(residuals),
        epsilon / scaled_residuals
    )
    return weights


def _compute_s1_loadings(components, n_s1_pixels):
    """
    Compute how much each component loads on s1 (control) pixels.

    Parameters
    ----------
    components : np.ndarray
        Component loadings, shape (n_components, n_total_pixels)
    n_s1_pixels : int
        Number of pixels from s1 (first n_s1_pixels columns)

    Returns
    -------
    np.ndarray
        Fraction of variance each component explains in s1 pixels
    """
    s1_loadings = np.sum(components[:, :n_s1_pixels]**2, axis=1)
    total_loadings = np.sum(components**2, axis=1)
    total_loadings = np.maximum(total_loadings, 1e-10)
    return s1_loadings / total_loadings


def _compute_pixel_means_chunked(arr, batch_size=500, dtype=np.float64):
    """
    Compute mean across time for each pixel in a memory-efficient way.

    Instead of loading the entire array, processes in time batches
    using an incremental mean calculation.

    Parameters
    ----------
    arr : np.ndarray or memmap
        Input array of shape (T, X, Y)
    batch_size : int
        Number of time frames to process at once
    dtype : np.dtype
        Dtype for accumulator (use float64 for numerical stability)

    Returns
    -------
    np.ndarray
        Mean values, shape (X*Y,) flattened
    """
    T, X, Y = arr.shape
    n_pixels = X * Y

    total_sum = np.zeros(n_pixels, dtype=np.float64)

    for t_start in range(0, T, batch_size):
        t_end = min(t_start + batch_size, T)
        batch = arr[t_start:t_end].reshape(t_end - t_start, n_pixels).astype(np.float64)
        total_sum += batch.sum(axis=0)

    return (total_sum / T).astype(np.float32)


def _extract_pixel_timeseries(arr, pixel_indices, Y, dtype=np.float32):
    """
    Extract time series for specific pixels from a (T, X, Y) array.

    Vectorized implementation using numpy fancy indexing.

    Parameters
    ----------
    arr : np.ndarray or memmap
        Input array of shape (T, X, Y)
    pixel_indices : np.ndarray
        Flat pixel indices to extract
    Y : int
        Number of columns (for converting flat index to i,j)
    dtype : np.dtype
        Output data type

    Returns
    -------
    np.ndarray
        Extracted data of shape (T, n_pixels)
    """
    i_indices = pixel_indices // Y
    j_indices = pixel_indices % Y

    return arr[:, i_indices, j_indices].astype(dtype)


def _vectorized_ols(s1_chunk, s2_chunk):
    """
    Vectorized OLS regression: s2 = beta0 + beta1*s1 + residual.

    Computes regression for all pixels simultaneously.

    Parameters
    ----------
    s1_chunk : np.ndarray
        Control signal chunk, shape (T, n_pixels)
    s2_chunk : np.ndarray
        Target signal chunk, shape (T, n_pixels)

    Returns
    -------
    beta0 : np.ndarray
        Intercepts, shape (n_pixels,)
    beta1 : np.ndarray
        Slopes, shape (n_pixels,)
    residuals : np.ndarray
        Residuals (corrected signal), shape (T, n_pixels)
    """
    # Compute means
    s1_mean = np.mean(s1_chunk, axis=0)
    s2_mean = np.mean(s2_chunk, axis=0)

    # Center the data
    s1_centered = s1_chunk - s1_mean
    s2_centered = s2_chunk - s2_mean

    # Compute covariance and variance (vectorized across pixels)
    # cov(s1, s2) = sum((s1 - mean_s1) * (s2 - mean_s2)) / (T-1)
    # var(s1) = sum((s1 - mean_s1)^2) / (T-1)
    cov_s1_s2 = np.sum(s1_centered * s2_centered, axis=0)
    var_s1 = np.sum(s1_centered ** 2, axis=0)

    # Add small regularization to avoid division by zero
    var_s1 = np.maximum(var_s1, 1e-10)

    # Compute coefficients
    beta1 = cov_s1_s2 / var_s1
    beta0 = s2_mean - beta1 * s1_mean

    # Compute residuals
    residuals = s2_chunk - (beta0 + beta1 * s1_chunk)

    return beta0, beta1, residuals


def _vectorized_robust_ols(
        s1_chunk, s2_chunk, huber_epsilon=1.35, max_iter=10, tol=1e-4):
    """
    Vectorized robust OLS regression with Huber loss via IRLS.

    Parameters
    ----------
    s1_chunk : np.ndarray
        Control signal chunk, shape (T, n_pixels)
    s2_chunk : np.ndarray
        Target signal chunk, shape (T, n_pixels)
    huber_epsilon : float
        Huber threshold parameter
    max_iter : int
        Maximum IRLS iterations
    tol : float
        Convergence tolerance

    Returns
    -------
    beta0 : np.ndarray
        Intercepts, shape (n_pixels,)
    beta1 : np.ndarray
        Slopes, shape (n_pixels,)
    residuals : np.ndarray
        Residuals (corrected signal), shape (T, n_pixels)
    """
    T, n_pixels = s1_chunk.shape

    # Initialize with OLS
    beta0, beta1, residuals = _vectorized_ols(s1_chunk, s2_chunk)

    # IRLS iterations
    for _ in range(max_iter):
        # Compute Huber weights
        weights = _huber_weights_vectorized(residuals, huber_epsilon)

        # Weighted means
        w_sum = np.sum(weights, axis=0)
        w_sum = np.maximum(w_sum, 1e-10)

        s1_wmean = np.sum(weights * s1_chunk, axis=0) / w_sum
        s2_wmean = np.sum(weights * s2_chunk, axis=0) / w_sum

        # Weighted covariance and variance
        s1_centered = s1_chunk - s1_wmean
        s2_centered = s2_chunk - s2_wmean

        wcov = np.sum(weights * s1_centered * s2_centered, axis=0)
        wvar = np.sum(weights * s1_centered ** 2, axis=0)
        wvar = np.maximum(wvar, 1e-10)

        # Update coefficients
        beta1_new = wcov / wvar
        beta0_new = s2_wmean - beta1_new * s1_wmean

        # Check convergence
        if np.max(np.abs(beta1_new - beta1)) < tol and \
           np.max(np.abs(beta0_new - beta0)) < tol:
            beta0, beta1 = beta0_new, beta1_new
            break

        beta0, beta1 = beta0_new, beta1_new
        residuals = s2_chunk - (beta0 + beta1 * s1_chunk)

    # Final residuals
    residuals = s2_chunk - (beta0 + beta1 * s1_chunk)

    return beta0, beta1, residuals


def _process_linear_chunk(
        chunk_info, s1, s2, Y, robust, huber_epsilon, max_iter, tol, dtype):
    """
    Process a single chunk for linear regression (for parallel execution).

    Parameters
    ----------
    chunk_info : tuple
        (chunk_start, chunk_end) pixel indices
    s1, s2 : np.ndarray
        Input signals
    Y : int
        Number of columns
    robust : bool
        Use robust regression
    huber_epsilon, max_iter, tol : float, int, float
        Robust regression parameters
    dtype : np.dtype
        Output dtype

    Returns
    -------
    tuple
        (pixel_indices, i_indices, j_indices, beta0, beta1, residuals)
    """
    chunk_start, chunk_end = chunk_info
    pixel_indices = np.arange(chunk_start, chunk_end)
    i_indices = pixel_indices // Y
    j_indices = pixel_indices % Y

    # Extract chunk data (use float32 internally for numerical stability)
    s1_chunk = _extract_pixel_timeseries(s1, pixel_indices, Y, np.float32)
    s2_chunk = _extract_pixel_timeseries(s2, pixel_indices, Y, np.float32)

    # Compute regression (in float32)
    if not robust:
        beta0, beta1, residuals_f = _vectorized_ols(s1_chunk, s2_chunk)
    else:
        beta0, beta1, residuals_f = _vectorized_robust_ols(
            s1_chunk, s2_chunk, huber_epsilon, max_iter, tol
        )

    # Clip and convert to output dtype
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        residuals = np.clip(residuals_f, info.min, info.max).astype(dtype)
    else:
        residuals = residuals_f.astype(dtype)

    return (pixel_indices, i_indices, j_indices, beta0, beta1, residuals)


def _process_lms_chunk(
        chunk_info, s1, s2, Y, filter_order, mu, normalized, dtype):
    """
    Process a single chunk for LMS adaptive filtering (for parallel).

    Parameters
    ----------
    chunk_info : tuple
        (chunk_start, chunk_end) pixel indices
    s1, s2 : np.ndarray
        Input signals
    Y : int
        Number of columns
    filter_order : int
        Filter order
    mu : float
        Step size
    normalized : bool
        Use NLMS
    dtype : np.dtype
        Output dtype

    Returns
    -------
    tuple
        (pixel_indices, i_indices, j_indices, corrected, noise, coeffs)
    """
    chunk_start, chunk_end = chunk_info
    pixel_indices = np.arange(chunk_start, chunk_end)
    i_indices = pixel_indices // Y
    j_indices = pixel_indices % Y

    T = s1.shape[0]
    chunk_len = len(pixel_indices)
    eps = 1e-10

    # Vectorized extraction for all pixels in chunk at once
    s1_chunk = s1[:, i_indices, j_indices].astype(np.float32)  # (T, chunk_len)
    s2_chunk = s2[:, i_indices, j_indices].astype(np.float32)  # (T, chunk_len)

    # w: (filter_order, chunk_len) - weights for all pixels simultaneously
    w = np.zeros((filter_order, chunk_len), dtype=np.float32)
    corrected_f = np.zeros((T, chunk_len), dtype=np.float32)
    estimated_noise_f = np.zeros((T, chunk_len), dtype=np.float32)

    # Time loop only - all pixels processed simultaneously at each step.
    # x_n: (filter_order, chunk_len) view into s1_chunk window (reversed)
    #
    # For NLMS, pre-compute ||x_n||^2 for all n outside the loop using a
    # cumsum-based sliding window sum of squares.  This replaces the per-step
    # einsum('fp,fp->p', x_n, x_n) with a single O(T*chunk_len) pass.
    #
    # x_norm_sq[k] = sum(s1_chunk[n-filter_order:n]**2) for n = filter_order+k,
    # computed as padded_cumsum[n] - padded_cumsum[n-filter_order].
    if normalized:
        s1_sq = s1_chunk ** 2                                  # (T, chunk_len)
        padded = np.empty((T + 1, chunk_len), dtype=np.float32)
        padded[0] = 0.0
        np.cumsum(s1_sq, axis=0, out=padded[1:])
        x_norm_sq = padded[filter_order:T] - padded[0:T - filter_order]  # (T-filter_order, chunk_len)
        x_norm_sq += eps                                       # in-place, avoids temp array

        for n in range(filter_order, T):
            x_n = s1_chunk[n-filter_order:n, :][::-1, :]      # (filter_order, chunk_len)
            y_n = (w * x_n).sum(0)                             # (chunk_len,)
            e_n = s2_chunk[n] - y_n                            # (chunk_len,)
            estimated_noise_f[n] = y_n
            corrected_f[n] = e_n
            w += x_n * ((mu / x_norm_sq[n - filter_order]) * e_n)
    else:
        for n in range(filter_order, T):
            x_n = s1_chunk[n-filter_order:n, :][::-1, :]      # (filter_order, chunk_len)
            y_n = (w * x_n).sum(0)                             # (chunk_len,)
            e_n = s2_chunk[n] - y_n                            # (chunk_len,)
            estimated_noise_f[n] = y_n
            corrected_f[n] = e_n
            w += x_n * (mu * e_n)

    filter_coeffs = w.T  # (chunk_len, filter_order)

    corrected = _clip_to_dtype(corrected_f, dtype)
    estimated_noise = _clip_to_dtype(estimated_noise_f, dtype)

    return (pixel_indices, i_indices, j_indices,
            corrected, estimated_noise, filter_coeffs)


def _prepare_inputs(s1, s2, batch_size, T, X, Y, verbose):
    """
    Optionally preload s1/s2 into RAM and auto-scale batch_size.

    If psutil is available and both arrays fit in 70% of free RAM,
    copies them from disk (memmap) into RAM so later passes read from
    memory instead of disk.  Then uses remaining free RAM to raise
    batch_size (capped at T) so each pass makes fewer loop iterations.

    Parameters
    ----------
    s1, s2 : np.ndarray or np.memmap
        Input signals, shape (T, X, Y).
    batch_size : int
        Caller-supplied batch size (lower bound after auto-scaling).
    T, X, Y : int
        Array dimensions.
    verbose : bool
        Print preload/scaling messages.

    Returns
    -------
    s1, s2 : np.ndarray
        Possibly preloaded copies.
    batch_size : int
        Possibly enlarged batch size.
    """
    try:
        import psutil as _psutil
        avail = int(_psutil.virtual_memory().available)
    except Exception:
        avail = 0

    # Only preload if the arrays are not already plain ndarrays in RAM
    _is_memmap = lambda a: isinstance(a, np.memmap)
    data_bytes = int(s1.nbytes) + int(s2.nbytes)

    if avail > 0 and (_is_memmap(s1) or _is_memmap(s2)) \
            and data_bytes < avail * 0.70:
        if verbose:
            print(f'\t\tpreloading {data_bytes / 2**30:.2f} GB into RAM ...')
        s1 = np.array(s1)
        s2 = np.array(s2)
        avail -= data_bytes

    if avail > 0:
        # Target 50% of free RAM shared across 5 float32 (X, Y) batch arrays
        auto_batch = int(avail * 0.50 / (5 * X * Y * 4))
        auto_batch = max(batch_size, min(T, auto_batch))
        if auto_batch > batch_size:
            if verbose:
                print(f'\t\tauto batch_size: {batch_size} → {auto_batch}')
            batch_size = auto_batch

    return s1, s2, batch_size


def _batched_f32(s1, s2, T, batch_size):
    """
    Yield (bi, t0, t1, s1_b, s2_b) as float32 with I/O prefetch.

    Reads the next batch in a background thread while the caller
    processes the current one, hiding disk latency.

    Parameters
    ----------
    s1, s2 : np.ndarray or np.memmap
        Input signals, shape (T, X, Y).
    T : int
        Number of time frames.
    batch_size : int
        Frames per batch.

    Yields
    ------
    bi : int
        Batch index (0-based).
    t0, t1 : int
        Slice [t0:t1] for this batch.
    s1_b, s2_b : np.ndarray, float32, shape (t1-t0, X, Y)
    """
    n_batches = (T + batch_size - 1) // batch_size

    def _read(t0, t1):
        return (s1[t0:t1].astype(np.float32),
                s2[t0:t1].astype(np.float32))

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_read, 0, min(batch_size, T))
        for bi in range(n_batches):
            t0 = bi * batch_size
            t1 = min(t0 + batch_size, T)
            if bi + 1 < n_batches:
                ns = (bi + 1) * batch_size
                next_future = pool.submit(_read, ns, min(ns + batch_size, T))
            s1_b, s2_b = future.result()
            if bi + 1 < n_batches:
                future = next_future
            yield bi, t0, t1, s1_b, s2_b


def _correct_ols_time_batched(s1, s2, batch_size, dtype, verbose, output_path):
    """
    Time-batch OLS correction: fits s2 = beta0 + beta1*s1 per pixel using
    two passes over contiguous temporal batches.

    Pass 1: accumulate Σs1, Σs2, Σs1², Σs1s2 in one read → beta (X, Y)
    Pass 2: write residuals                                → corrected (T, X, Y)

    The one-pass statistics formula avoids a separate centred-data pass:
      var(s1)    = Σs1²/T  − mean_s1²
      cov(s1,s2) = Σs1s2/T − mean_s1·mean_s2
      beta1      = cov / var,  beta0 = mean_s2 − beta1·mean_s1

    Parameters
    ----------
    s1, s2 : np.ndarray or memmap
        Input signals, shape (T, X, Y)
    batch_size : int
        Number of time frames per batch
    dtype : np.dtype
        Output dtype
    verbose : bool
        Print progress
    output_path : str or None
        Write output here if given

    Returns
    -------
    corrected : np.ndarray or memmap, shape (T, X, Y)
    coefficients : np.ndarray, shape (X, Y, 2)  — [beta0, beta1] per pixel
    """
    T, X, Y = s1.shape
    s1, s2, batch_size = _prepare_inputs(s1, s2, batch_size, T, X, Y, verbose)
    n_batches = (T + batch_size - 1) // batch_size

    # Pre-allocate output
    if output_path is not None:
        if verbose:
            print(f'\t\twriting output to: {output_path}')
        corrected = tifffile.memmap(
            output_path, shape=(T, X, Y), dtype=dtype, bigtiff=True
        )
    else:
        corrected = np.zeros((T, X, Y), dtype=dtype)
    coefficients = np.zeros((X, Y, 2), dtype=dtype)

    # ------------------------------------------------------------------
    # Pass 1 — one-pass statistics → beta
    # Accumulate in float64 for numerical stability with int16 inputs.
    # ------------------------------------------------------------------
    if verbose:
        print(f'\t\tpass 1/2: computing regression coefficients '
              f'({n_batches} time batches)...')

    s1_sum   = np.zeros((X, Y), dtype=np.float64)
    s2_sum   = np.zeros((X, Y), dtype=np.float64)
    s1sq_sum = np.zeros((X, Y), dtype=np.float64)
    s1s2_sum = np.zeros((X, Y), dtype=np.float64)

    for bi, t_start, t_end, s1_b, s2_b in _batched_f32(s1, s2, T, batch_size):
        if verbose:
            pct = (bi + 1) / n_batches * 100
            print(f'\t\t\tbatch {bi+1}/{n_batches} ({pct:.1f}%)...',
                  end='\r')
        s1_sum   += s1_b.sum(axis=0, dtype=np.float64)
        s2_sum   += s2_b.sum(axis=0, dtype=np.float64)
        s1sq_sum += (s1_b * s1_b).sum(axis=0, dtype=np.float64)
        s1s2_sum += (s1_b * s2_b).sum(axis=0, dtype=np.float64)

    if verbose:
        print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')

    mean_s1  = s1_sum / T
    mean_s2  = s2_sum / T
    var_s1   = np.maximum(s1sq_sum / T - mean_s1 ** 2, 1e-10)
    cov_s1s2 = s1s2_sum / T - mean_s1 * mean_s2
    beta1 = (cov_s1s2 / var_s1).astype(np.float32)            # (X, Y)
    beta0 = (mean_s2 - beta1 * mean_s1).astype(np.float32)    # (X, Y)
    del s1_sum, s2_sum, s1sq_sum, s1s2_sum, mean_s1, mean_s2, var_s1, cov_s1s2

    coefficients[:, :, 0] = _clip_to_dtype(beta0, dtype)
    coefficients[:, :, 1] = _clip_to_dtype(beta1, dtype)

    # ------------------------------------------------------------------
    # Pass 2 — write residuals
    # ------------------------------------------------------------------
    if verbose:
        print(f'\t\tpass 2/2: writing residuals ({n_batches} time batches)...')

    for bi, t_start, t_end, s1_b, s2_b in _batched_f32(s1, s2, T, batch_size):
        if verbose:
            pct = (bi + 1) / n_batches * 100
            print(f'\t\t\tbatch {bi+1}/{n_batches} ({pct:.1f}%)...',
                  end='\r')
        corrected[t_start:t_end] = _clip_to_dtype(
            s2_b - (beta0 + beta1 * s1_b), dtype
        )

    if verbose:
        print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')

    if output_path is not None:
        corrected.flush()
        if verbose:
            print('\t\tflushed output to disk')

    return corrected, coefficients


def _correct_robust_time_batched(
        s1, s2, batch_size, huber_epsilon, max_iter, tol, dtype, verbose,
        output_path):
    """
    Time-batched robust (Huber IRLS) correction: fits s2 = beta0 + beta1*s1
    per pixel using streaming temporal batches.

    Peak RAM: O(batch_T * X * Y * 4 bytes) + O(X * Y * 8 arrays).

    Passes
    ------
    Pass 1    : one-pass OLS init — accumulate Σs1, Σs2, Σs1², Σs2², Σs1s2
                → beta (X, Y) and initial scale (X, Y) with no extra read.
    Per IRLS iter (single pass):
                Accumulate w, ws1, ws2, ws1², ws1s2, Σresid, Σresid²
                → update beta via one-pass WLS formula + update scale.
    Final pass: write clipped residuals.

    The one-pass weighted least squares formula is mathematically equivalent
    to the two-pass centred form:
        beta1 = (W·Σwᵢs1ᵢs2ᵢ − Σwᵢs1ᵢ·Σwᵢs2ᵢ) / (W·Σwᵢs1ᵢ² − (Σwᵢs1ᵢ)²)
    where W = Σwᵢ.  This eliminates the need to stream the data a second time
    to compute centred weighted cov/var.

    Total passes: 1 (OLS init) + N (IRLS) + 1 (write) = N+2
    vs. old:      3 (OLS init) + 2N (IRLS) + 1 (write) = 2N+4
    Speedup: ~2× for N≥2 (I/O-bound workload).

    Parameters
    ----------
    s1, s2 : np.ndarray or memmap
        Input signals, shape (T, X, Y)
    batch_size : int
        Number of time frames per batch
    huber_epsilon : float
        Huber threshold (default: 1.35)
    max_iter : int
        Maximum IRLS iterations
    tol : float
        Convergence tolerance on beta
    dtype : np.dtype
        Output dtype
    verbose : bool
        Print progress
    output_path : str or None
        Write output here if given

    Returns
    -------
    corrected : np.ndarray or memmap, shape (T, X, Y)
    coefficients : np.ndarray, shape (X, Y, 2)  — [beta0, beta1] per pixel
    """
    T, X, Y = s1.shape
    s1, s2, batch_size = _prepare_inputs(s1, s2, batch_size, T, X, Y, verbose)
    n_batches = (T + batch_size - 1) // batch_size
    eps = huber_epsilon

    # ------------------------------------------------------------------
    # Pass 1 — one-pass OLS init + initial scale
    # Accumulate Σs1, Σs2, Σs1², Σs2², Σs1s2 in float64.
    # Derive beta analytically; derive scale = std(OLS residual) from:
    #   Var(residual) = Var(s2) − beta1·Cov(s1,s2)  (no extra pass needed)
    # ------------------------------------------------------------------
    if verbose:
        print(f'\t\tpass 1/{max_iter + 2} (OLS init): computing coefficients '
              f'and scale ({n_batches} time batches)...')

    s1_sum   = np.zeros((X, Y), dtype=np.float64)
    s2_sum   = np.zeros((X, Y), dtype=np.float64)
    s1sq_sum = np.zeros((X, Y), dtype=np.float64)
    s2sq_sum = np.zeros((X, Y), dtype=np.float64)
    s1s2_sum = np.zeros((X, Y), dtype=np.float64)

    for bi, t_start, t_end, s1_b, s2_b in _batched_f32(s1, s2, T, batch_size):
        if verbose:
            pct = (bi + 1) / n_batches * 100
            print(f'\t\t\tbatch {bi+1}/{n_batches} ({pct:.1f}%)...',
                  end='\r')
        s1_sum   += s1_b.sum(axis=0, dtype=np.float64)
        s2_sum   += s2_b.sum(axis=0, dtype=np.float64)
        s1sq_sum += (s1_b * s1_b).sum(axis=0, dtype=np.float64)
        s2sq_sum += (s2_b * s2_b).sum(axis=0, dtype=np.float64)
        s1s2_sum += (s1_b * s2_b).sum(axis=0, dtype=np.float64)

    if verbose:
        print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')

    mean_s1  = s1_sum / T
    mean_s2  = s2_sum / T
    var_s1   = np.maximum(s1sq_sum / T - mean_s1 ** 2, 1e-10)
    var_s2   = s2sq_sum / T - mean_s2 ** 2
    cov_s1s2 = s1s2_sum / T - mean_s1 * mean_s2
    beta1 = (cov_s1s2 / var_s1).astype(np.float32)            # (X, Y)
    beta0 = (mean_s2 - beta1 * mean_s1).astype(np.float32)    # (X, Y)

    # Var(s2 − beta0 − beta1·s1) = Var(s2) − beta1·Cov(s1,s2)
    resid_var = np.maximum(
        var_s2 - beta1.astype(np.float64) * cov_s1s2, 0.0
    )
    scale = np.maximum(np.sqrt(resid_var).astype(np.float32), 1e-10)
    del (s1_sum, s2_sum, s1sq_sum, s2sq_sum, s1s2_sum,
         mean_s1, mean_s2, var_s1, var_s2, cov_s1s2, resid_var)

    # ------------------------------------------------------------------
    # IRLS iterations — single pass per iteration
    #
    # Accumulate the five weighted sufficient statistics in one data read:
    #   w_sum, w_s1_sum, w_s2_sum, w_s1sq_sum, w_s1s2_sum
    # Then compute beta1_new and beta0_new via the one-pass WLS formula
    # (mathematically identical to the centred two-pass version).
    # Also accumulate sum_resid / sum_sq_resid for the scale update.
    # ------------------------------------------------------------------
    for irls_iter in range(max_iter):
        iter_label = f'{irls_iter+1}/{max_iter}'
        if verbose:
            print(f'\t\tIRLS iter {iter_label} '
                  f'({n_batches} time batches)...')

        w_sum      = np.zeros((X, Y), dtype=np.float64)
        w_s1_sum   = np.zeros((X, Y), dtype=np.float64)
        w_s2_sum   = np.zeros((X, Y), dtype=np.float64)
        w_s1sq_sum = np.zeros((X, Y), dtype=np.float64)
        w_s1s2_sum = np.zeros((X, Y), dtype=np.float64)
        sum_resid    = np.zeros((X, Y), dtype=np.float64)
        sum_sq_resid = np.zeros((X, Y), dtype=np.float64)

        for bi, t_start, t_end, s1_b, s2_b in _batched_f32(s1, s2, T, batch_size):
            if verbose:
                pct = (bi + 1) / n_batches * 100
                print(f'\t\t\tbatch {bi+1}/{n_batches} ({pct:.1f}%)...',
                      end='\r')
            resid_b = s2_b - (beta0 + beta1 * s1_b)
            scaled_b = np.abs(resid_b) / scale
            w_b = np.where(scaled_b <= eps, 1.0,
                           eps / np.maximum(scaled_b, 1e-10))  # float32

            # Compute ws1 once; reuse for both ws1sq and ws1s2
            ws1_b = w_b * s1_b                                 # float32

            w_sum      += w_b.sum(axis=0)
            w_s1_sum   += ws1_b.sum(axis=0)
            w_s2_sum   += (w_b * s2_b).sum(axis=0)
            w_s1sq_sum += (ws1_b * s1_b).sum(axis=0)
            w_s1s2_sum += (ws1_b * s2_b).sum(axis=0)
            sum_resid    += resid_b.sum(axis=0)
            sum_sq_resid += (resid_b * resid_b).sum(axis=0)

            del ws1_b

        if verbose:
            print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')

        # One-pass WLS:
        #   beta1 = (W·Σws1s2 − Σws1·Σws2) / (W·Σws1² − (Σws1)²)
        #   beta0 = (Σws2 − beta1·Σws1) / W
        w_sum = np.maximum(w_sum, 1e-10)
        denom = np.maximum(
            w_sum * w_s1sq_sum - w_s1_sum ** 2, 1e-10
        )
        beta1_new = (
            (w_sum * w_s1s2_sum - w_s1_sum * w_s2_sum) / denom
        ).astype(np.float32)
        beta0_new = (
            (w_s2_sum - beta1_new.astype(np.float64) * w_s1_sum) / w_sum
        ).astype(np.float32)

        new_scale = np.sqrt(
            np.maximum(sum_sq_resid / T - (sum_resid / T) ** 2, 0.0)
        ).astype(np.float32)
        new_scale = np.maximum(new_scale, 1e-10)

        converged = (np.max(np.abs(beta1_new - beta1)) < tol and
                     np.max(np.abs(beta0_new - beta0)) < tol)

        beta0, beta1 = beta0_new, beta1_new
        scale = new_scale

        if converged:
            if verbose:
                print(f'\t\tIRLS converged after {irls_iter+1} iteration(s)')
            break

    # ------------------------------------------------------------------
    # Pre-allocate output
    # ------------------------------------------------------------------
    if output_path is not None:
        if verbose:
            print(f'\t\twriting output to: {output_path}')
        corrected = tifffile.memmap(
            output_path, shape=(T, X, Y), dtype=dtype, bigtiff=True
        )
    else:
        corrected = np.zeros((T, X, Y), dtype=dtype)

    coefficients = np.zeros((X, Y, 2), dtype=dtype)
    coefficients[:, :, 0] = _clip_to_dtype(beta0, dtype)
    coefficients[:, :, 1] = _clip_to_dtype(beta1, dtype)

    # ------------------------------------------------------------------
    # Final pass — write residuals
    # ------------------------------------------------------------------
    if verbose:
        print(f'\t\tfinal pass: writing residuals ({n_batches} time batches)...')

    for bi, t_start, t_end, s1_b, s2_b in _batched_f32(s1, s2, T, batch_size):
        if verbose:
            pct = (bi + 1) / n_batches * 100
            print(f'\t\t\tbatch {bi+1}/{n_batches} ({pct:.1f}%)...',
                  end='\r')
        corrected[t_start:t_end] = _clip_to_dtype(
            s2_b - (beta0 + beta1 * s1_b), dtype
        )

    if verbose:
        print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')

    if output_path is not None:
        corrected.flush()
        if verbose:
            print('\t\tflushed output to disk')

    return corrected, coefficients


def detrend_linearly(arr, batch_size=500, verbose=False, output_path=None):
    """
    Remove a per-pixel linear trend from a (T, X, Y) array using two
    streaming temporal passes.

    Fits the model y_p(t) = a_p + b_p * t independently for every pixel p
    and returns y_p(t) - a_p - b_p * t as a float32 array.

    Two passes are made over contiguous temporal batches so the full array
    is never loaded into RAM at once:
      Pass 1 — accumulate sum(y) and sum(t*y) per pixel in float64 to
               derive a_p and b_p.
      Pass 2 — subtract the trend and write to the output array.

    Parameters
    ----------
    arr : np.ndarray or np.memmap
        Input array, shape (T, X, Y).
    batch_size : int
        Number of time frames to load per batch (default: 500).
    verbose : bool
        If True, print pass progress (default: False).
    output_path : str, optional
        If provided, write the detrended output to a raw numpy memmap file
        at this path instead of allocating T*X*Y*4 bytes in RAM.  The
        caller is responsible for deleting the file when no longer needed.
        (default: None)

    Returns
    -------
    out : np.ndarray or np.memmap
        Linearly detrended array, shape (T, X, Y), dtype float32.
    """
    T, X, Y = arr.shape
    n_pixels = X * Y
    n_batches = (T + batch_size - 1) // batch_size

    t_arr = np.arange(T, dtype=np.float64)
    mean_t = (T - 1) / 2.0                          # exact mean of 0..T-1
    sum_t2 = T * (T - 1) * (2 * T - 1) / 6.0       # exact sum of t^2
    denom = sum_t2 - T * mean_t ** 2                 # T * Var(t), scalar

    # ------------------------------------------------------------------
    # Pass 1: accumulate per-pixel sum(y) and sum(t*y)
    # ------------------------------------------------------------------
    if verbose:
        print(f'\t\t\tdetrend pass 1/2 ({n_batches} batches)...', end='\r')
    sum_y = np.zeros(n_pixels, dtype=np.float64)
    sum_ty = np.zeros(n_pixels, dtype=np.float64)
    for bi in range(n_batches):
        t0 = bi * batch_size
        t1 = min(t0 + batch_size, T)
        batch = np.asarray(
            arr[t0:t1], dtype=np.float64).reshape(t1 - t0, -1)  # (bt, P)
        sum_y += batch.sum(axis=0)
        sum_ty += t_arr[t0:t1] @ batch    # (P,) via BLAS DGEMV, no large tmp

    mean_y = sum_y / T
    b = ((sum_ty / T) - mean_t * mean_y) / (denom / T)   # slope (P,)
    a = mean_y - b * mean_t                               # intercept (P,)
    a = a.astype(np.float32).reshape(X, Y)
    b = b.astype(np.float32).reshape(X, Y)
    del sum_y, sum_ty, mean_y

    # ------------------------------------------------------------------
    # Pass 2: subtract per-pixel trend, write float32 output
    # ------------------------------------------------------------------
    if verbose:
        print(f'\t\t\tdetrend pass 2/2 ({n_batches} batches)...', end='\r')
    if output_path is not None:
        out = np.memmap(output_path, dtype=np.float32, mode='w+',
                        shape=(T, X, Y))
    else:
        out = np.empty((T, X, Y), dtype=np.float32)
    for bi in range(n_batches):
        t0 = bi * batch_size
        t1 = min(t0 + batch_size, T)
        batch = np.asarray(arr[t0:t1], dtype=np.float32)         # (bt, X, Y)
        t_b = t_arr[t0:t1].astype(np.float32)                    # (bt,)
        out[t0:t1] = batch - (a + b * t_b[:, np.newaxis, np.newaxis])
    if output_path is not None:
        out.flush()

    if verbose:
        print(f'\t\t\tdetrend done.                                ')
    return out


# =============================================================================
# Correction methods
# =============================================================================

def correct_linear_regression(
        s1, s2, robust=False, huber_epsilon=1.35,
        max_iter=10, tol=1e-4, batch_size=1000,
        dtype=np.int16, verbose=True, output_path=None):
    """
    Pixel-wise linear regression correction.

    Fits s2 = beta0 + beta1*s1 + residual per pixel and returns the
    residual as the corrected signal. This removes the linear component
    of s1 from s2.

    Parameters
    ----------
    s1 : np.ndarray
        Control signal (e.g., red channel), shape (T, X, Y)
    s2 : np.ndarray
        Target signal to correct (e.g., green channel), shape (T, X, Y)
    robust : bool
        If True, use Huber robust regression via IRLS (default: False)
    huber_epsilon : float
        Huber threshold parameter for robust regression (default: 1.35)
    max_iter : int
        Maximum iterations for IRLS (only used if robust=True)
    tol : float
        Convergence tolerance for IRLS (only used if robust=True)
    batch_size : int
        Number of time frames per batch (default: 500). Larger batches
        are faster but use more RAM. Both OLS and robust paths use this.
    dtype : np.dtype
        Data type for output (default: np.int16)
    verbose : bool
        If True, print progress updates (default: True)
    output_path : str, optional
        If provided, write corrected signal to this path as a memory-mapped
        TIFF file instead of storing in RAM. Significantly reduces memory
        usage for large movies. (default: None)

    Returns
    -------
    corrected : np.ndarray or np.memmap
        Corrected s2 signal, shape (T, X, Y). If output_path is provided,
        this is a memmap pointing to the file on disk.
    coefficients : np.ndarray
        Regression coefficients [beta0, beta1] per pixel, shape (X, Y, 2)
    """
    if s1.shape != s2.shape:
        raise ValueError(
            f"s1 and s2 must have same shape, "
            f"got {s1.shape} and {s2.shape}"
        )

    T, X, Y = s1.shape
    n_pixels = X * Y

    method_name = "robust regression" if robust else "linear regression"
    disk_str = " -> disk" if output_path else ""
    if verbose:
        print(f'\tcorrecting signal ({method_name}{disk_str})...')

    # Both OLS and robust use time-batched streaming passes
    if not robust:
        return _correct_ols_time_batched(
            s1, s2, batch_size, dtype, verbose, output_path
        )
    else:
        return _correct_robust_time_batched(
            s1, s2, batch_size, huber_epsilon, max_iter, tol, dtype,
            verbose, output_path
        )


def correct_lms_adaptive(
        s1, s2, filter_order=10, mu=0.01, normalized=True,
        chunk_size=1000, dtype=np.int16, verbose=True, n_jobs=-1,
        output_path=None):
    """
    LMS adaptive filter correction per pixel.

    Uses the Least Mean Squares (LMS) algorithm to adaptively filter
    the control signal s1 and subtract it from s2.

    Supports parallel processing across pixels.

    Parameters
    ----------
    s1 : np.ndarray
        Control signal (noise reference), shape (T, X, Y)
    s2 : np.ndarray
        Target signal to correct, shape (T, X, Y)
    filter_order : int
        Number of filter taps (default: 10)
    mu : float
        Step size / learning rate (default: 0.01)
    normalized : bool
        If True, use Normalized LMS for better convergence (default: True)
    chunk_size : int
        Number of pixels to process at once (default: 1000)
    dtype : np.dtype
        Data type for computation (default: np.float32)
    verbose : bool
        If True, print progress updates (default: True)
    n_jobs : int
        Number of parallel jobs. -1 uses all cores. (default: 1)
    output_path : str, optional
        If provided, write corrected signal to this path as a memory-mapped
        TIFF file instead of storing in RAM. (default: None)

    Returns
    -------
    corrected : np.ndarray or np.memmap
        Corrected s2 signal, shape (T, X, Y). If output_path is provided,
        this is a memmap pointing to the file on disk.
    estimated_noise : np.ndarray
        Estimated noise component removed from s2, shape (T, X, Y)
    filter_coefficients : np.ndarray
        Final filter coefficients per pixel, shape (X, Y, filter_order)
    """
    if s1.shape != s2.shape:
        raise ValueError(
            f"s1 and s2 must have same shape, "
            f"got {s1.shape} and {s2.shape}"
        )

    T, X, Y = s1.shape
    n_pixels = X * Y

    method_name = "NLMS" if normalized else "LMS"
    parallel_str = f", {n_jobs} jobs" if n_jobs != 1 else ""
    disk_str = " -> disk" if output_path else ""
    if verbose:
        print(
            f'\tcorrecting signal ({method_name} adaptive{parallel_str}'
            f'{disk_str})...'
        )

    # Pre-allocate output arrays
    if output_path is not None:
        if verbose:
            print(f'\t\twriting output to: {output_path}')
        corrected = tifffile.memmap(
            output_path, shape=(T, X, Y), dtype=dtype, bigtiff=True
        )
    else:
        corrected = np.zeros((T, X, Y), dtype=dtype)
    estimated_noise = np.zeros((T, X, Y), dtype=dtype)
    filter_coefficients = np.zeros((X, Y, filter_order), dtype=dtype)

    # Build list of chunks
    chunks = [
        (start, end)
        for start, end, _, _ in _iter_pixel_chunks(n_pixels, chunk_size)
    ]
    n_chunks = len(chunks)

    if n_jobs == 1:
        # Serial processing with progress
        if verbose:
            print(f'\t\tprocessing {n_pixels} pixels in {n_chunks} chunks...')

        for chunk_idx, chunk_info in enumerate(chunks):
            pixels_done = min((chunk_idx + 1) * chunk_size, n_pixels)
            if verbose:
                pct = (chunk_idx + 1) / n_chunks * 100
                print(
                    f'\t\t\t{pixels_done}/{n_pixels} pixels ({pct:.1f}%)...',
                    end='\r'
                )

            result = _process_lms_chunk(
                chunk_info, s1, s2, Y, filter_order, mu, normalized, dtype
            )
            (pixel_indices, i_indices, j_indices,
             corr_chunk, noise_chunk, coeff_chunk) = result

            # Vectorized write-back using numpy advanced indexing
            corrected[:, i_indices, j_indices] = corr_chunk
            estimated_noise[:, i_indices, j_indices] = noise_chunk
            filter_coefficients[i_indices, j_indices, :] = coeff_chunk

        if verbose:
            print(f'\t\t\t{n_pixels}/{n_pixels} pixels (100.0%)...done')

    else:
        # Parallel processing in batches to limit memory usage
        effective_jobs = n_jobs if n_jobs > 0 else 6
        parallel_batch_size = effective_jobs * 2
        n_batches = (n_chunks + parallel_batch_size - 1) // parallel_batch_size

        if verbose:
            print(
                f'\t\tprocessing {n_pixels} pixels in {n_batches} batches '
                f'({n_jobs} parallel jobs)...'
            )

        for batch_idx, batch_start in enumerate(
                range(0, n_chunks, parallel_batch_size)):
            batch_end = min(batch_start + parallel_batch_size, n_chunks)
            batch_chunks = chunks[batch_start:batch_end]
            pixels_done = min(batch_end * chunk_size, n_pixels)

            if verbose:
                pct = (batch_idx + 1) / n_batches * 100
                print(
                    f'\t\t\tbatch {batch_idx+1}/{n_batches}: '
                    f'{pixels_done}/{n_pixels} pixels ({pct:.1f}%)...',
                    end='\r'
                )

            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_process_lms_chunk)(
                    chunk_info, s1, s2, Y, filter_order, mu, normalized, dtype
                )
                for chunk_info in batch_chunks
            )

            # Copy results immediately and discard
            for result in results:
                (pixel_indices, i_indices, j_indices,
                 corr_chunk, noise_chunk, coeff_chunk) = result
                corrected[:, i_indices, j_indices] = corr_chunk
                estimated_noise[:, i_indices, j_indices] = noise_chunk
                filter_coefficients[i_indices, j_indices, :] = coeff_chunk

            del results

        if verbose:
            print(
                f'\t\t\tbatch {n_batches}/{n_batches}: '
                f'{n_pixels}/{n_pixels} pixels (100.0%)...done'
            )

    # Flush memmap to disk if applicable
    if output_path is not None:
        corrected.flush()
        if verbose:
            print('\t\tflushed output to disk')

    return corrected, estimated_noise, filter_coefficients


def correct_pca_shared_variance(
        s1, s2, n_components='auto',
        variance_threshold=0.8,
        s1_loading_threshold=0.5,
        batch_size=500,
        spatial_subsample=4,
        dtype=np.int16,
        verbose=True,
        n_jobs=-1,
        output_path=None):
    """
    PCA-based correction by removing shared variance components.

    Memory-optimized using IncrementalPCA and spatial subsampling.

    Parameters
    ----------
    s1 : np.ndarray
        Control signal, shape (T, X, Y)
    s2 : np.ndarray
        Target signal to correct, shape (T, X, Y)
    n_components : int or 'auto'
        Number of PCA components. If 'auto', uses min(50, n_features//10)
    variance_threshold : float
        Not used with IncrementalPCA (kept for API compatibility)
    s1_loading_threshold : float
        Components with s1 loading above this are removed (default: 0.5)
    batch_size : int
        Number of time frames to process at once (default: 500)
    spatial_subsample : int
        Subsample every N pixels for fitting PCA (default: 4)
    dtype : np.dtype
        Data type for computation (default: np.float32)
    verbose : bool
        If True, print progress updates (default: True)
    n_jobs : int
        Number of parallel jobs for regression phase (default: 1)
    output_path : str, optional
        If provided, write corrected signal to this path as a memory-mapped
        TIFF file instead of storing in RAM. (default: None)

    Returns
    -------
    corrected : np.ndarray or np.memmap
        Corrected s2 signal, shape (T, X, Y). If output_path is provided,
        this is a memmap pointing to the file on disk.
    noise_components : np.ndarray
        Indices of removed noise components
    explained_variance : np.ndarray
        Explained variance ratio for each component
    """
    if s1.shape != s2.shape:
        raise ValueError(
            f"s1 and s2 must have same shape, "
            f"got {s1.shape} and {s2.shape}"
        )

    T, X, Y = s1.shape
    n_pixels = X * Y

    disk_str = " -> disk" if output_path else ""
    if verbose:
        print(f'\tcorrecting signal (PCA shared variance{disk_str})...')
        if output_path:
            print(f'\t\twriting output to: {output_path}')

    # Subsample pixels for fitting
    fit_pixel_indices = np.arange(0, n_pixels, spatial_subsample)
    n_fit_pixels = len(fit_pixel_indices)
    n_total_fit = n_fit_pixels * 2  # s1 + s2

    # Determine number of components
    if n_components == 'auto':
        n_components = min(50, n_total_fit // 10, T)
        n_components = max(n_components, min(5, T))

    # Streaming PCA — never materialise s1[:, fit_pixels] or s2[:, fit_pixels]
    # up front (that would be T × n_fit_pixels × 4 bytes per channel).
    # Instead make four passes over contiguous temporal batches:
    #   Pass 1 — compute per-pixel means for the subsampled pixels
    #   Pass 2 — IncrementalPCA.partial_fit each batch
    #   Pass 3 — extract noise source time-courses
    #   Pass 4/5 — two-pass temporal regression (accumulate beta, write output)

    n_batches = (T + batch_size - 1) // batch_size

    # ------------------------------------------------------------------
    # Pass 1: stream-compute per-pixel means (subsampled pixels)
    # ------------------------------------------------------------------
    if verbose:
        print(f'\t\tpass 1/5: computing subsampled means '
              f'({n_batches} batches, {n_fit_pixels} pixels)...')
    sum_s1 = np.zeros(n_fit_pixels, dtype=np.float64)
    sum_s2 = np.zeros(n_fit_pixels, dtype=np.float64)
    for batch_idx in range(n_batches):
        t0 = batch_idx * batch_size
        t1 = min(t0 + batch_size, T)
        if verbose:
            pct = (batch_idx + 1) / n_batches * 100
            print(f'\t\t\tbatch {batch_idx+1}/{n_batches} ({pct:.1f}%)...',
                  end='\r')
        s1_b = np.asarray(s1[t0:t1], dtype=np.float32).reshape(t1 - t0, -1)
        s2_b = np.asarray(s2[t0:t1], dtype=np.float32).reshape(t1 - t0, -1)
        sum_s1 += s1_b[:, fit_pixel_indices].sum(axis=0).astype(np.float64)
        sum_s2 += s2_b[:, fit_pixel_indices].sum(axis=0).astype(np.float64)
    if verbose:
        print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')
    s1_mean = (sum_s1 / T).astype(np.float32)
    s2_mean = (sum_s2 / T).astype(np.float32)
    del sum_s1, sum_s2

    # ------------------------------------------------------------------
    # Pass 2: fit IncrementalPCA in temporal batches
    # ------------------------------------------------------------------
    if verbose:
        print(f'\t\tpass 2/5: fitting PCA ({n_components} components, '
              f'{n_batches} batches)...')
    ipca = IncrementalPCA(n_components=n_components)
    combined_buf = np.empty((batch_size, n_fit_pixels * 2), dtype=np.float32)
    batches_fit = 0
    for batch_idx in range(n_batches):
        t0 = batch_idx * batch_size
        t1 = min(t0 + batch_size, T)
        bt = t1 - t0
        if bt < n_components:   # IncrementalPCA requires batch >= n_components
            break
        if verbose:
            pct = (batch_idx + 1) / n_batches * 100
            print(f'\t\t\tbatch {batch_idx+1}/{n_batches} ({pct:.1f}%)...',
                  end='\r')
        s1_b = np.asarray(s1[t0:t1], dtype=np.float32).reshape(bt, -1)
        s2_b = np.asarray(s2[t0:t1], dtype=np.float32).reshape(bt, -1)
        combined_buf[:bt, :n_fit_pixels] = s1_b[:, fit_pixel_indices] - s1_mean
        combined_buf[:bt, n_fit_pixels:] = s2_b[:, fit_pixel_indices] - s2_mean
        ipca.partial_fit(combined_buf[:bt])
        batches_fit += 1
    if verbose:
        print(f'\t\t\t{batches_fit}/{n_batches} batches fitted (100.0%)...done')

    components = ipca.components_
    explained_variance = ipca.explained_variance_ratio_

    # Identify noise components
    s1_loadings = _compute_s1_loadings(components, n_fit_pixels)
    noise_mask = s1_loadings > s1_loading_threshold
    noise_component_indices = np.where(noise_mask)[0]
    n_noise_comp = int(np.sum(noise_mask))
    noise_components_T = components[noise_mask]   # (n_noise, n_total_fit)

    if verbose:
        print(f'\t\tidentified {n_noise_comp} noise components '
              f'(s1 loading > {s1_loading_threshold})')

    # ------------------------------------------------------------------
    # Pass 3: extract noise source time-courses (T, n_noise_comp)
    # ------------------------------------------------------------------
    noise_sources = np.empty((T, n_noise_comp), dtype=np.float32)
    if n_noise_comp > 0:
        if verbose:
            print(f'\t\tpass 3/5: extracting noise time courses '
                  f'({n_batches} batches)...')
        for batch_idx in range(n_batches):
            t0 = batch_idx * batch_size
            t1 = min(t0 + batch_size, T)
            bt = t1 - t0
            if verbose:
                pct = (batch_idx + 1) / n_batches * 100
                print(f'\t\t\tbatch {batch_idx+1}/{n_batches} ({pct:.1f}%)...',
                      end='\r')
            s1_b = np.asarray(s1[t0:t1], dtype=np.float32).reshape(bt, -1)
            s2_b = np.asarray(s2[t0:t1], dtype=np.float32).reshape(bt, -1)
            combined_buf[:bt, :n_fit_pixels] = (
                s1_b[:, fit_pixel_indices] - s1_mean)
            combined_buf[:bt, n_fit_pixels:] = (
                s2_b[:, fit_pixel_indices] - s2_mean)
            # (bt, n_total_fit) @ (n_total_fit, n_noise) -> (bt, n_noise)
            noise_sources[t0:t1] = combined_buf[:bt] @ noise_components_T.T
        if verbose:
            print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')
    del combined_buf

    # ------------------------------------------------------------------
    # Pre-allocate output
    # ------------------------------------------------------------------
    if output_path is not None:
        corrected = tifffile.memmap(
            output_path, shape=(T, X, Y), dtype=dtype, bigtiff=True)
    else:
        corrected = np.zeros((T, X, Y), dtype=dtype)

    if n_noise_comp == 0:
        if verbose:
            print('\t\tno noise components found, returning original signal')
        corrected[:] = s2
        if output_path is not None:
            corrected.flush()
        return corrected, noise_component_indices, explained_variance

    # ------------------------------------------------------------------
    # Passes 4/5: two-pass temporal regression (contiguous full-frame reads
    # and writes — no scatter I/O).
    # beta = (n_noise_comp, n_pixels): accumulated over time batches
    # corrected[t] = clip(s2[t] - noise_sources[t] @ beta + s2_mean)
    # ------------------------------------------------------------------
    XtX_inv = np.linalg.inv(
        noise_sources.T @ noise_sources
        + 1e-6 * np.eye(n_noise_comp, dtype=np.float32))
    projection_matrix = (XtX_inv @ noise_sources.T).astype(np.float32)

    s2_full_mean = _compute_pixel_means_chunked(s2, batch_size=batch_size)

    if verbose:
        print(f'\t\tpass 4/5: computing pixel coefficients '
              f'({n_batches} batches)...')
    beta = np.zeros((n_noise_comp, n_pixels), dtype=np.float32)
    for batch_idx in range(n_batches):
        t0 = batch_idx * batch_size
        t1 = min(t0 + batch_size, T)
        if verbose:
            pct = (batch_idx + 1) / n_batches * 100
            print(f'\t\t\tbatch {batch_idx+1}/{n_batches} ({pct:.1f}%)...',
                  end='\r')
        s2_flat = (np.asarray(s2[t0:t1], dtype=np.float32).reshape(t1-t0, -1)
                   - s2_full_mean)
        beta += projection_matrix[:, t0:t1] @ s2_flat
    if verbose:
        print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')

    if verbose:
        print(f'\t\tpass 5/5: writing residuals ({n_batches} batches)...')
    for batch_idx in range(n_batches):
        t0 = batch_idx * batch_size
        t1 = min(t0 + batch_size, T)
        if verbose:
            pct = (batch_idx + 1) / n_batches * 100
            print(f'\t\t\tbatch {batch_idx+1}/{n_batches} ({pct:.1f}%)...',
                  end='\r')
        s2_flat = (np.asarray(s2[t0:t1], dtype=np.float32).reshape(t1-t0, -1)
                   - s2_full_mean)
        residuals = s2_flat - noise_sources[t0:t1] @ beta
        corrected[t0:t1] = _clip_to_dtype(
            (residuals + s2_full_mean).reshape(t1-t0, X, Y), dtype)
    if verbose:
        print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')

    if output_path is not None:
        corrected.flush()
        if verbose:
            print('\t\tflushed output to disk')

    return corrected, noise_component_indices, explained_variance


def correct_ica_shared_components(
        s1, s2, n_components=None,
        s1_loading_threshold=0.5,
        max_iter=200, random_state=None,
        spatial_subsample=4,
        batch_size=500,
        n_fit_frames=5000,
        dtype=np.int16,
        verbose=True,
        n_jobs=-1,
        output_path=None):
    """
    ICA-based correction by removing shared independent components.

    Memory-optimized using spatial and temporal subsampling for fitting,
    followed by streaming temporal batch transform and two-pass regression.
    All intermediate computation is done in float32; only the final output
    is cast to ``dtype``.

    Parameters
    ----------
    s1 : np.ndarray
        Control signal, shape (T, X, Y)
    s2 : np.ndarray
        Target signal to correct, shape (T, X, Y)
    n_components : int or None
        Number of ICA components. If None, uses min(20, n_pixels//100)
    s1_loading_threshold : float
        Components with s1 loading above this are removed (default: 0.5)
    max_iter : int
        Maximum iterations for FastICA (default: 200)
    random_state : int or None
        Random seed for reproducibility
    spatial_subsample : int
        Subsample every N pixels for fitting (default: 4)
    batch_size : int
        Time frames to process at once (default: 500)
    n_fit_frames : int
        Number of time frames to subsample for ICA fitting.
        FastICA requires the full fit matrix in RAM; this caps its size.
        (default: 5000)
    dtype : np.dtype
        Output data type (default: np.int16)
    verbose : bool
        If True, print progress updates (default: True)
    n_jobs : int
        Number of parallel jobs (reserved, default: 1)
    output_path : str, optional
        If provided, write corrected signal to this path as a memory-mapped
        TIFF file instead of storing in RAM. (default: None)

    Returns
    -------
    corrected : np.ndarray or np.memmap
        Corrected s2 signal, shape (T, X, Y). If output_path is provided,
        this is a memmap pointing to the file on disk.
    noise_components : np.ndarray
        Indices of removed noise components
    mixing_matrix : np.ndarray
        ICA mixing matrix (n_total_fit, n_components)
    """
    if s1.shape != s2.shape:
        raise ValueError(
            f"s1 and s2 must have same shape, "
            f"got {s1.shape} and {s2.shape}"
        )

    T, X, Y = s1.shape
    n_pixels = X * Y

    disk_str = " -> disk" if output_path else ""
    if verbose:
        print(f'\tcorrecting signal (ICA shared components{disk_str})...')
        if output_path:
            print(f'\t\twriting output to: {output_path}')

    # Subsample pixels for fitting
    fit_pixel_indices = np.arange(0, n_pixels, spatial_subsample)
    n_fit_pixels = len(fit_pixel_indices)
    n_total_fit = n_fit_pixels * 2

    i_fit = fit_pixel_indices // Y
    j_fit = fit_pixel_indices % Y

    if n_components is None:
        n_components = min(20, n_fit_pixels // 50, T // 10)
        n_components = max(n_components, 3)

    n_batches = (T + batch_size - 1) // batch_size

    # Temporal subsampling indices for ICA fitting
    actual_fit_frames = min(n_fit_frames, T)
    fit_frame_indices = np.sort(
        np.random.default_rng(random_state).choice(
            T, size=actual_fit_frames, replace=False)
    ) if actual_fit_frames < T else np.arange(T)

    # ------------------------------------------------------------------
    # Pass 1: stream-compute per-pixel means (subsampled pixels only)
    # ------------------------------------------------------------------
    if verbose:
        print(f'\t\tpass 1/5: computing subsampled means '
              f'({n_batches} batches, {n_fit_pixels} pixels)...')
    sum_s1 = np.zeros(n_fit_pixels, dtype=np.float64)
    sum_s2 = np.zeros(n_fit_pixels, dtype=np.float64)
    for batch_idx in range(n_batches):
        t0 = batch_idx * batch_size
        t1 = min(t0 + batch_size, T)
        if verbose:
            pct = (batch_idx + 1) / n_batches * 100
            print(f'\t\t\tbatch {batch_idx+1}/{n_batches} ({pct:.1f}%)...',
                  end='\r')
        s1_b = np.asarray(s1[t0:t1], dtype=np.float32)
        s2_b = np.asarray(s2[t0:t1], dtype=np.float32)
        sum_s1 += s1_b[:, i_fit, j_fit].astype(np.float64).sum(axis=0)
        sum_s2 += s2_b[:, i_fit, j_fit].astype(np.float64).sum(axis=0)
    if verbose:
        print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')
    s1_mean = (sum_s1 / T).astype(np.float32)
    s2_mean = (sum_s2 / T).astype(np.float32)
    del sum_s1, sum_s2

    # ------------------------------------------------------------------
    # Pass 2: build combined_sub (actual_fit_frames, n_total_fit) for ICA.
    # Load contiguous temporal batches and pick only the fit-frame rows —
    # avoids T individual scatter reads (one per frame).
    # ------------------------------------------------------------------
    if verbose:
        print(f'\t\tpass 2/5: building fit matrix '
              f'({actual_fit_frames} frames x {n_total_fit} features)...')
    combined_sub = np.empty((actual_fit_frames, n_total_fit), dtype=np.float32)
    # combined_buf is also reused in pass 3
    combined_buf = np.empty((batch_size, n_total_fit), dtype=np.float32)

    for batch_idx in range(n_batches):
        t0 = batch_idx * batch_size
        t1 = min(t0 + batch_size, T)
        if verbose:
            pct = (batch_idx + 1) / n_batches * 100
            print(f'\t\t\tbatch {batch_idx+1}/{n_batches} ({pct:.1f}%)...',
                  end='\r')
        batch_fit_mask = (fit_frame_indices >= t0) & (fit_frame_indices < t1)
        if not np.any(batch_fit_mask):
            continue
        local_frames = fit_frame_indices[batch_fit_mask] - t0
        dest_rows = np.where(batch_fit_mask)[0]

        s1_b = np.asarray(s1[t0:t1], dtype=np.float32)
        s2_b = np.asarray(s2[t0:t1], dtype=np.float32)
        combined_sub[dest_rows, :n_fit_pixels] = (
            s1_b[local_frames][:, i_fit, j_fit] - s1_mean)
        combined_sub[dest_rows, n_fit_pixels:] = (
            s2_b[local_frames][:, i_fit, j_fit] - s2_mean)
    if verbose:
        print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')

    # Fit ICA on temporally subsampled data (float32 throughout — no int cast)
    if verbose:
        print(f'\t\tfitting ICA ({n_components} components, '
              f'{actual_fit_frames} frames)...')
    ica = FastICA(
        n_components=n_components, max_iter=max_iter,
        random_state=random_state, whiten='unit-variance'
    )
    try:
        ica.fit(combined_sub)
        if verbose:
            print('\t\t\tICA converged successfully')
    except Exception as e:
        print(f"\t\tWarning: ICA did not converge: {e}")
        if output_path is not None:
            corrected = tifffile.memmap(
                output_path, shape=(T, X, Y), dtype=dtype, bigtiff=True)
            corrected[:] = s2
            corrected.flush()
        else:
            corrected = np.asarray(s2, dtype=dtype).copy()
        return corrected, np.array([]), np.array([])

    del combined_sub

    mixing_matrix = ica.mixing_   # (n_total_fit, n_components)

    # Compute s1 loadings and identify noise components
    s1_loadings = np.sum(mixing_matrix[:n_fit_pixels, :]**2, axis=0)
    total_loadings = np.sum(mixing_matrix**2, axis=0)
    total_loadings = np.maximum(total_loadings, 1e-10)
    s1_loading_ratio = s1_loadings / total_loadings

    noise_mask = s1_loading_ratio > s1_loading_threshold
    noise_component_indices = np.where(noise_mask)[0]
    n_noise_comp = int(np.sum(noise_mask))

    if verbose:
        print(
            f'\t\tidentified {n_noise_comp} noise components '
            f'(s1 loading > {s1_loading_threshold})'
        )

    # Pre-allocate output
    if output_path is not None:
        corrected = tifffile.memmap(
            output_path, shape=(T, X, Y), dtype=dtype, bigtiff=True
        )
    else:
        corrected = np.zeros((T, X, Y), dtype=dtype)

    if n_noise_comp == 0:
        if verbose:
            print('\t\tno noise components found, returning original signal')
        corrected[:] = s2
        if output_path is not None:
            corrected.flush()
        return corrected, noise_component_indices, mixing_matrix

    # ------------------------------------------------------------------
    # Pass 3: transform full dataset in temporal batches.
    # ica.transform(batch) → (bt, n_components); keep only noise_mask cols.
    # ------------------------------------------------------------------
    if verbose:
        print(f'\t\tpass 3/5: extracting noise time courses '
              f'({n_batches} batches)...')
    noise_sources = np.empty((T, n_noise_comp), dtype=np.float32)

    for batch_idx in range(n_batches):
        t0 = batch_idx * batch_size
        t1 = min(t0 + batch_size, T)
        bt = t1 - t0
        if verbose:
            pct = (batch_idx + 1) / n_batches * 100
            print(f'\t\t\tbatch {batch_idx+1}/{n_batches} ({pct:.1f}%)...',
                  end='\r')
        s1_b = np.asarray(s1[t0:t1], dtype=np.float32)
        s2_b = np.asarray(s2[t0:t1], dtype=np.float32)
        combined_buf[:bt, :n_fit_pixels] = s1_b[:, i_fit, j_fit] - s1_mean
        combined_buf[:bt, n_fit_pixels:] = s2_b[:, i_fit, j_fit] - s2_mean
        all_sources = ica.transform(combined_buf[:bt])   # (bt, n_components)
        noise_sources[t0:t1] = all_sources[:, noise_mask]
    if verbose:
        print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')
    del combined_buf

    # ------------------------------------------------------------------
    # Passes 4/5: two-pass temporal regression (contiguous full-frame reads
    # and writes — no scatter I/O).
    # beta = (n_noise_comp, n_pixels): accumulated over time batches
    # corrected[t] = clip(s2[t] - noise_sources[t] @ beta + s2_mean)
    # ------------------------------------------------------------------
    XtX_inv = np.linalg.inv(
        noise_sources.T @ noise_sources
        + 1e-6 * np.eye(n_noise_comp, dtype=np.float32))
    projection_matrix = (XtX_inv @ noise_sources.T).astype(np.float32)

    s2_full_mean = _compute_pixel_means_chunked(s2, batch_size=batch_size)

    if verbose:
        print(f'\t\tpass 4/5: computing pixel coefficients '
              f'({n_batches} batches)...')
    beta = np.zeros((n_noise_comp, n_pixels), dtype=np.float32)
    for batch_idx in range(n_batches):
        t0 = batch_idx * batch_size
        t1 = min(t0 + batch_size, T)
        if verbose:
            pct = (batch_idx + 1) / n_batches * 100
            print(f'\t\t\tbatch {batch_idx+1}/{n_batches} ({pct:.1f}%)...',
                  end='\r')
        s2_flat = (np.asarray(s2[t0:t1], dtype=np.float32).reshape(t1-t0, -1)
                   - s2_full_mean)
        beta += projection_matrix[:, t0:t1] @ s2_flat
    if verbose:
        print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')

    if verbose:
        print(f'\t\tpass 5/5: writing residuals ({n_batches} batches)...')
    for batch_idx in range(n_batches):
        t0 = batch_idx * batch_size
        t1 = min(t0 + batch_size, T)
        if verbose:
            pct = (batch_idx + 1) / n_batches * 100
            print(f'\t\t\tbatch {batch_idx+1}/{n_batches} ({pct:.1f}%)...',
                  end='\r')
        s2_flat = (np.asarray(s2[t0:t1], dtype=np.float32).reshape(t1-t0, -1)
                   - s2_full_mean)
        residuals = s2_flat - noise_sources[t0:t1] @ beta
        corrected[t0:t1] = _clip_to_dtype(
            (residuals + s2_full_mean).reshape(t1-t0, X, Y), dtype)
    if verbose:
        print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')

    if output_path is not None:
        corrected.flush()
        if verbose:
            print('\t\tflushed output to disk')

    return corrected, noise_component_indices, mixing_matrix


def correct_nmf_shared_components(
        s1, s2, n_components=10,
        s1_loading_threshold=0.5,
        max_iter=200, random_state=None,
        batch_size=500,
        spatial_subsample=4,
        dtype=np.int16,
        verbose=True,
        n_jobs=-1,
        zscore_input=False,
        output_path=None):
    """
    NMF-based correction for non-negative fluorescence signals.

    Memory-optimized using MiniBatchNMF and spatial subsampling.

    Parameters
    ----------
    s1 : np.ndarray
        Control signal, shape (T, X, Y)
    s2 : np.ndarray
        Target signal to correct, shape (T, X, Y)
    n_components : int
        Number of NMF components (default: 10)
    s1_loading_threshold : float
        Components with s1 loading above this are removed (default: 0.5)
    max_iter : int
        Maximum iterations for NMF (default: 200)
    random_state : int or None
        Random seed for reproducibility
    batch_size : int
        Batch size for MiniBatchNMF (default: 500)
    spatial_subsample : int
        Subsample every N pixels for fitting (default: 4)
    dtype : np.dtype
        Data type (default: np.int16)
    verbose : bool
        If True, print progress updates (default: True)
    n_jobs : int
        Number of parallel jobs for regression phase (default: 1)
    zscore_input : bool
        If True, z-score each subsampled pixel's time series (zero mean,
        unit variance) before feeding to NMF. This equalises the
        contribution of dim and bright pixels and can improve component
        separation when channels differ in mean intensity. The z-scoring
        is applied only during NMF fitting and transform; the regression
        phase that produces the corrected output operates on the original
        unscaled signal so that the output amplitude is preserved.
        (default: False)
    output_path : str, optional
        If provided, write corrected signal to this path as a memory-mapped
        TIFF file instead of storing in RAM. (default: None)

    Returns
    -------
    corrected : np.ndarray or np.memmap
        Corrected s2 signal, shape (T, X, Y). If output_path is provided,
        this is a memmap pointing to the file on disk.
    W : np.ndarray
        NMF temporal components (basis), shape (T, n_components)
    H : np.ndarray
        NMF spatial components, shape (n_components, n_fit_pixels*2)
    """
    if s1.shape != s2.shape:
        raise ValueError(
            f"s1 and s2 must have same shape, "
            f"got {s1.shape} and {s2.shape}"
        )

    T, X, Y = s1.shape
    n_pixels = X * Y

    disk_str = " -> disk" if output_path else ""
    if verbose:
        print(f'\tcorrecting signal (NMF shared components{disk_str})...')
        if output_path:
            print(f'\t\twriting output to: {output_path}')

    # Subsample pixels
    fit_pixel_indices = np.arange(0, n_pixels, spatial_subsample)
    n_fit_pixels = len(fit_pixel_indices)
    n_total_fit = n_fit_pixels * 2

    i_fit = fit_pixel_indices // Y
    j_fit = fit_pixel_indices % Y

    # Streaming NMF: never build the full (T x n_total_fit) matrix in RAM.
    # Always 3 passes over contiguous temporal batches:
    #   Pass 1 — prep (either zscore stats OR raw min-scan)
    #   Pass 2 — fit NMF via partial_fit
    #   Pass 3 — transform each batch to obtain W (T x n_components)
    # Peak RAM per pass: ~2 x batch_size x n_fit_pixels x 4 bytes.

    n_batches = (T + batch_size - 1) // batch_size

    # ------------------------------------------------------------------
    # Pass 1: prep — zscore stats or raw min-scan
    # ------------------------------------------------------------------
    # zscore_input=True:  load float32, subsample, cast only subsampled
    #   portion to float64 for accumulation (avoids full-frame float64
    #   conversion).  After computing mean/std we use a fixed shift of 5.0
    #   for non-negativity — z-scored signals are bounded to ±5σ in
    #   practice, so no separate scan pass is needed.
    #
    # zscore_input=False: scan batches to find global minimum for shift.

    if zscore_input:
        if verbose:
            print(f'\t\tpass 1/3: computing per-pixel mean/std for z-scoring '
                  f'({n_batches} batches, {n_fit_pixels} subsampled pixels)...')
        sum_s1 = np.zeros(n_fit_pixels, dtype=np.float64)
        sum_s2 = np.zeros(n_fit_pixels, dtype=np.float64)
        sumsq_s1 = np.zeros(n_fit_pixels, dtype=np.float64)
        sumsq_s2 = np.zeros(n_fit_pixels, dtype=np.float64)
        for batch_idx in range(n_batches):
            t0 = batch_idx * batch_size
            t1 = min(t0 + batch_size, T)
            if verbose:
                pct = (batch_idx + 1) / n_batches * 100
                print(f'\t\t\tbatch {batch_idx+1}/{n_batches} ({pct:.1f}%)...',
                      end='\r')
            # Load as float32 first; cast only the subsampled portion to
            # float64.  This avoids converting the full (batch, X*Y) frame
            # to float64, which costs 2x memory for no benefit.
            s1_b = np.asarray(
                s1[t0:t1], dtype=np.float32).reshape(t1 - t0, -1)
            s2_b = np.asarray(
                s2[t0:t1], dtype=np.float32).reshape(t1 - t0, -1)
            s1_sub = s1_b[:, fit_pixel_indices].astype(np.float64)
            s2_sub = s2_b[:, fit_pixel_indices].astype(np.float64)
            sum_s1 += s1_sub.sum(axis=0)
            sum_s2 += s2_sub.sum(axis=0)
            sumsq_s1 += (s1_sub ** 2).sum(axis=0)
            sumsq_s2 += (s2_sub ** 2).sum(axis=0)
        if verbose:
            print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')

        mean_s1 = (sum_s1 / T).astype(np.float32)
        mean_s2 = (sum_s2 / T).astype(np.float32)
        std_s1 = np.sqrt(
            np.maximum(sumsq_s1 / T - (sum_s1 / T) ** 2, 0.0)
        ).astype(np.float32)
        std_s2 = np.sqrt(
            np.maximum(sumsq_s2 / T - (sum_s2 / T) ** 2, 0.0)
        ).astype(np.float32)
        # Guard against zero-variance pixels (constant signal)
        std_s1 = np.where(std_s1 > 0, std_s1, 1.0).astype(np.float32)
        std_s2 = np.where(std_s2 > 0, std_s2, 1.0).astype(np.float32)
        del sum_s1, sum_s2, sumsq_s1, sumsq_s2

        # Z-scored values are bounded to ±5σ for real signals: skip the
        # separate min-scan pass and use a fixed conservative shift.
        min_val = -5.0
        if verbose:
            print('\t\tz-scored: using fixed non-negativity shift of 5.0 '
                  '(skipping min-scan pass)')
    else:
        mean_s1 = mean_s2 = std_s1 = std_s2 = None
        if verbose:
            print(f'\t\tpass 1/3: scanning data range '
                  f'({n_batches} batches, {n_fit_pixels} subsampled pixels)...')
        min_val = 0.0
        for batch_idx in range(n_batches):
            t0 = batch_idx * batch_size
            t1 = min(t0 + batch_size, T)
            if verbose:
                pct = (batch_idx + 1) / n_batches * 100
                print(f'\t\t\tbatch {batch_idx+1}/{n_batches} ({pct:.1f}%)...',
                      end='\r')
            s1_b = np.asarray(
                s1[t0:t1], dtype=np.float32).reshape(t1 - t0, -1)
            s2_b = np.asarray(
                s2[t0:t1], dtype=np.float32).reshape(t1 - t0, -1)
            batch_min = min(
                float(s1_b[:, fit_pixel_indices].min()),
                float(s2_b[:, fit_pixel_indices].min()))
            if batch_min < min_val:
                min_val = batch_min
        if verbose:
            print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')
        if min_val < 0:
            if verbose:
                print(f'\t\tshifting data by {-min_val:.2f} for non-negativity')
        else:
            min_val = 0.0

    # ------------------------------------------------------------------
    # Pass 2: fit NMF via partial_fit over temporal batches
    # ------------------------------------------------------------------
    # max_iter controls how many full passes (epochs) to make through the
    # data.  One epoch is usually sufficient for signal correction; cap at
    # 10 to avoid spending excessive time on convergence.
    n_epochs = max(1, min(max_iter // max(n_batches, 1), 10))
    if verbose:
        zscore_str = ', z-scored' if zscore_input else ''
        print(f'\t\tpass 2/3: fitting NMF '
              f'({n_components} components{zscore_str}, '
              f'{n_epochs} epoch(s) x {n_batches} batches)...')

    nmf = MiniBatchNMF(
        n_components=n_components, max_iter=max_iter,
        random_state=random_state, batch_size=min(batch_size, T)
    )

    # Pre-allocate combined_b once at full batch size; slice a view for the
    # last (possibly shorter) batch rather than allocating on every iteration.
    combined_b_buf = np.empty((batch_size, n_total_fit), dtype=np.float32)

    try:
        for epoch in range(n_epochs):
            for batch_idx in range(n_batches):
                t0 = batch_idx * batch_size
                t1 = min(t0 + batch_size, T)
                bt = t1 - t0
                if verbose:
                    pct = (epoch * n_batches + batch_idx + 1) / \
                          (n_epochs * n_batches) * 100
                    print(f'\t\t\tepoch {epoch+1}/{n_epochs}, '
                          f'batch {batch_idx+1}/{n_batches} '
                          f'({pct:.1f}%)...',
                          end='\r')
                s1_b = np.asarray(
                    s1[t0:t1], dtype=np.float32).reshape(bt, -1)
                s2_b = np.asarray(
                    s2[t0:t1], dtype=np.float32).reshape(bt, -1)
                s1_sub = s1_b[:, fit_pixel_indices]
                s2_sub = s2_b[:, fit_pixel_indices]
                if zscore_input:
                    s1_sub = (s1_sub - mean_s1) / std_s1
                    s2_sub = (s2_sub - mean_s2) / std_s2
                combined_b = combined_b_buf[:bt]
                combined_b[:, :n_fit_pixels] = s1_sub - min_val
                combined_b[:, n_fit_pixels:] = s2_sub - min_val
                nmf.partial_fit(combined_b)
        H = nmf.components_
        if verbose:
            print('\t\t\tNMF fit completed                              ')
    except Exception as e:
        print(f"\t\tWarning: NMF did not converge: {e}")
        corrected = s2.astype(dtype).copy()
        return corrected, np.array([]), np.array([])

    # ------------------------------------------------------------------
    # Pass 3: transform each batch to obtain W (T x n_components)
    # ------------------------------------------------------------------
    if verbose:
        print(f'\t\tpass 3/3: transforming to get temporal activations '
              f'({n_batches} batches)...')
    W = np.empty((T, n_components), dtype=np.float32)
    for batch_idx in range(n_batches):
        t0 = batch_idx * batch_size
        t1 = min(t0 + batch_size, T)
        bt = t1 - t0
        if verbose:
            pct = (batch_idx + 1) / n_batches * 100
            print(f'\t\t\tbatch {batch_idx+1}/{n_batches} ({pct:.1f}%)...',
                  end='\r')
        s1_b = np.asarray(
            s1[t0:t1], dtype=np.float32).reshape(bt, -1)
        s2_b = np.asarray(
            s2[t0:t1], dtype=np.float32).reshape(bt, -1)
        s1_sub = s1_b[:, fit_pixel_indices]
        s2_sub = s2_b[:, fit_pixel_indices]
        if zscore_input:
            s1_sub = (s1_sub - mean_s1) / std_s1
            s2_sub = (s2_sub - mean_s2) / std_s2
        combined_b = combined_b_buf[:bt]
        combined_b[:, :n_fit_pixels] = s1_sub - min_val
        combined_b[:, n_fit_pixels:] = s2_sub - min_val
        W[t0:t1] = nmf.transform(combined_b)
    if verbose:
        print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')

    # Compute s1 loadings
    s1_loadings = np.sum(H[:, :n_fit_pixels]**2, axis=1)
    total_loadings = np.sum(H**2, axis=1)
    total_loadings = np.maximum(total_loadings, 1e-10)
    s1_loading_ratio = s1_loadings / total_loadings

    noise_mask = s1_loading_ratio > s1_loading_threshold
    noise_W = W[:, noise_mask]
    n_noise_comp = noise_W.shape[1]

    if verbose:
        n_noise = np.sum(noise_mask)
        print(
            f'\t\tidentified {n_noise} noise components '
            f'(s1 loading > {s1_loading_threshold})'
        )
        ratio_str = ', '.join(f'{r:.3f}' for r in sorted(
            s1_loading_ratio, reverse=True))
        print(f'\t\ts1 loading ratios (sorted): [{ratio_str}]')
        print(f'\t\t  -> set s1_loading_threshold below the highest value '
              f'to capture noise components')

    # Pre-allocate output
    if output_path is not None:
        corrected = tifffile.memmap(
            output_path, shape=(T, X, Y), dtype=dtype, bigtiff=True
        )
    else:
        corrected = np.zeros((T, X, Y), dtype=dtype)

    if n_noise_comp == 0:
        if verbose:
            print('\t\tno noise components found, returning original signal')
        corrected[:] = s2
        if output_path is not None:
            corrected.flush()
        return corrected, W, H

    # Regress noise temporal patterns out of every s2 pixel.
    #
    # Instead of spatial chunks (scatter-read s2[:, pixels] + scatter-write
    # corrected[:, pixels], both non-contiguous on a memmap), we do two
    # temporal passes that read and write contiguous full-frame blocks:
    #
    #   beta = projection_matrix @ s2_flat       (n_noise_comp, n_pixels)
    #        = sum_t  proj[:, t] * s2_flat[t, :]
    #   residuals[t] = s2_flat[t] - noise_W[t] @ beta
    #
    # Pass 1 accumulates beta; Pass 2 streams residuals to corrected[].
    if verbose:
        print('\t\tregressing out noise from all pixels...')

    XtX_inv = np.linalg.inv(
        noise_W.T @ noise_W + 1e-6 * np.eye(n_noise_comp, dtype=np.float32)
    )
    projection_matrix = (XtX_inv @ noise_W.T).astype(np.float32)  # (k, T)

    n_batches = (T + batch_size - 1) // batch_size

    # Pass 1: accumulate beta over temporal batches
    if verbose:
        print('\t\t\tpass 1/2: computing pixel coefficients...')
    beta = np.zeros((n_noise_comp, n_pixels), dtype=np.float32)
    for batch_idx in range(n_batches):
        t0 = batch_idx * batch_size
        t1 = min(t0 + batch_size, T)
        if verbose:
            pct = (batch_idx + 1) / n_batches * 100
            print(f'\t\t\t\tbatch {batch_idx+1}/{n_batches} ({pct:.1f}%)...',
                  end='\r')
        s2_batch = np.asarray(s2[t0:t1]).astype(np.float32)  # (bt, X, Y)
        s2_flat = s2_batch.reshape(t1 - t0, -1) - min_val    # (bt, n_pixels)
        beta += projection_matrix[:, t0:t1] @ s2_flat         # (k, n_pixels)
    if verbose:
        print(f'\t\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')

    # Pass 2: write residuals as contiguous full-frame slices
    if verbose:
        print('\t\t\tpass 2/2: writing residuals...')
    for batch_idx in range(n_batches):
        t0 = batch_idx * batch_size
        t1 = min(t0 + batch_size, T)
        if verbose:
            pct = (batch_idx + 1) / n_batches * 100
            print(f'\t\t\t\tbatch {batch_idx+1}/{n_batches} ({pct:.1f}%)...',
                  end='\r')
        s2_batch = np.asarray(s2[t0:t1]).astype(np.float32)
        s2_flat = s2_batch.reshape(t1 - t0, -1) - min_val    # (bt, n_pixels)
        residuals = s2_flat - noise_W[t0:t1, :] @ beta        # (bt, n_pixels)
        corrected[t0:t1] = _clip_to_dtype(
            (residuals + min_val).reshape(t1 - t0, X, Y), dtype
        )
    if verbose:
        print(f'\t\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')

    if verbose:
        print('\t\t\tprocessing pixels: 100.0% complete.      ')

    # Flush memmap to disk if applicable
    if output_path is not None:
        corrected.flush()
        if verbose:
            print('\t\tflushed output to disk')

    return corrected, W, H
