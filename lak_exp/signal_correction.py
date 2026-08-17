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
correct_full_regress : Martianova et al. (2019) full-regression
    photometric correction (the primary method); correct_full_regress_1d
    is its 1-D companion.
correct_pixel_spatial_subtr : per-pixel dF/F correction against a Gaussian
    spatial average of the surrounding high-SNR control pixels. Unlike
    the regression family it fits no coupling slope (unit gain) and
    returns true dF/F rather than zdFF; low-SNR pixels are NaN.
correct_lms_adaptive : LMS adaptive filter correction
correct_pca_shared_variance : PCA-based shared variance removal
correct_ica_shared_components : ICA-based shared component removal
correct_nmf_shared_components : NMF-based shared component removal
infer_field_bayesnf : field-inference *helper* (not a corrector) that
    fits an independent BayesNF spatiotemporal field to each channel and
    returns the two inferred (denoised) fields s1_real, s2_real.
"""

import os
import functools
import warnings
from types import SimpleNamespace

import numpy as np
import tifffile
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from sklearn.decomposition import IncrementalPCA, FastICA, MiniBatchNMF


# =============================================================================
# Primary correction: full-regression photometric
# =============================================================================

def _autocorr_time(trace, max_lag=None):
    """Integrated autocorrelation time tau of a 1-D trace, in samples.

    ``tau = 1 + 2·sum_k rho(k)`` with Bartlett triangular tapering and
    truncation at the first non-positive ``rho`` — the standard estimator
    for the variance inflation of a mean/correlation computed from
    serially dependent samples. The effective sample size of a length-T
    trace is then ``T / tau``.

    Haemodynamic regressors are heavily autocorrelated (tau ≈ 11 frames
    for the default ``DualColourSim``), so a significance floor built on
    the raw frame count rather than ``T / tau`` is several times too
    permissive.

    Parameters
    ----------
    trace : array-like, shape (T,)
        Trace to characterise. Non-finite samples are dropped.
    max_lag : int or None
        Largest lag summed. Defaults to ``min(T // 4, 500)``.

    Returns
    -------
    tau : float
        Integrated autocorrelation time in samples, floored at 1.0.
    """
    x = np.asarray(trace, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return 1.0
    x = x - x.mean()
    _var = float(np.dot(x, x))
    if _var <= 0:
        return 1.0
    if max_lag is None:
        max_lag = int(min(n // 4, 500))
    max_lag = int(max(1, min(max_lag, n - 1)))
    _ac = np.correlate(x, x, mode='full')[n - 1:n + max_lag] / _var
    _tau = 1.0
    for _k in range(1, len(_ac)):
        if _ac[_k] <= 0.0:
            break
        _tau += 2.0 * _ac[_k] * (1.0 - _k / n)
    return float(max(_tau, 1.0))


def correct_full_regress(
        s1, s2, batch_size=1000, dtype=np.float16, verbose=True,
        output_path=None, eps=1e-6,
        smooth_window=10,
        airpls_lam=1e5, airpls_porder=2, airpls_max_iter=50,
        trim_initial=0, nn_slope=False,
        aggregates_only=None,
        fit_mode='global',
        clean_reg_level='frame', min_pixel_corr=0.0,
        f0_level='frame', f0_n_sectors=8,
        spatial_avg_sigma_px=4.0,
        beta_loss='huber', beta_f_scale=1.0, beta_scale='mad',
        regress_type='ols'):
    """Full-regression photometry-style correction (zdFF).

    Adapted from the zdFF pipeline of Martianova, Aronson & Proulx
    (2019, J. Vis. Exp. 152, e60278): smooth both channels, remove the
    airPLS baseline, standardise, fit a non-negative linear coupling of
    the control to the signal, and subtract in normalised (z) space.
    This differs from GuPPy (Sherathiya, Schaid, Litts & Lerner, 2021,
    Sci. Rep. 11:24212), which fits the raw control to the raw signal
    and divides by the fitted control to obtain ΔF/F; here the output
    stays in z units and nothing is divided by a fitted baseline.

    Coupling-slope fit — 1-D, on the spatial-mean traces s̄1(t), s̄2(t):
        1. moving-average lowpass (``smooth_window``)
        2. airPLS baseline removal per channel
        3. drop the first ``trim_initial`` warm-up frames (fit only)
        4. median-subtract + std-divide normalisation
        5. non-negative slope β of normalised s̄2 on normalised s̄1
           (``beta_loss`` sets the loss, ``regress_type`` the solver:
           trust-region least-squares or IRLS)

    Per-voxel correction:
        ``zdFF(t,x,y) = s2_norm(t,x,y) − β·r(t,x,y)``, where the signal
        channel is normalised by its own (pooled) session mean m_xy and
        scale σ_xy, and ``r`` is the **control regressor averaged at the
        same spatial scale β was fit at**:

        =====================  ==================================
        ``fit_mode``           regressor ``r(t,x,y)``
        =====================  ==================================
        'global'               s̄1_norm(t) — the whole-frame mean
                               control trace, identical for every
                               pixel
        'per_sector'           s̄1_norm(t, sector) — that pixel's
                               sector-mean control trace
        'per_pixel'            s1_norm(t,x,y) — the pixel's own
                               control trace
        'per_pixel_spatial_avg' s̃1_norm(t,x,y) — a Gaussian spatial
                               average of the control channel around
                               that pixel (σ = ``spatial_avg_sigma_px``)
        'per_pixel_clean_reg'  the SAME spatially-averaged trace as
                               'global' / 'per_sector' (selected by
                               ``clean_reg_level``), but with β fit
                               independently at every pixel
        =====================  ==================================

        Matching the regressor to the fit scale matters: haemodynamics
        are spatially coherent while shot noise is not, so a spatially
        averaged control is a far cleaner estimate of the artefact than
        a single pixel's trace. Regressing on the per-pixel control
        (what 'global'/'per_sector' previously subtracted, despite
        fitting β on averaged traces) removes little artefact and
        injects that pixel's noise into the output — badly so wherever
        the control fluorophore is weakly expressed. For 'global' /
        'per_sector', ``r`` is built from the **raw** (unsmoothed,
        un-airPLS'd) mean control trace, debased by the same baseline and
        re-normalised by its own median/scale — the ``smooth_window``
        moving average is a denoising step for the β *fit* only; applying
        it to what actually gets subtracted from the raw per-frame data
        would blur out the fast-varying part of a genuinely coherent
        artefact, leaving it in the residual.

        A subtraction in normalised space (no division by F̂): features
        unique to s2 survive at full magnitude, shared features are
        attenuated by (1 − β). Output ≈ N(0, √(1−β²)).

    The coupling fit is cheap (length-T 1-D). The correction streams the
    stack in two passes — pass 1 collects spatial-mean traces + per-pixel
    session moments; pass 2 computes and writes/aggregates zdFF.

    Parameters
    ----------
    s1, s2 : (T, X, Y) ndarray or memmap
        Control / isosbestic-like reference (s1, e.g. green) and the
        real signal to correct (s2, e.g. red).
    batch_size : int
        Frames per streaming batch. Default 1000.
    dtype : np.dtype
        Output dtype. Default np.float16.
    verbose : bool
        Print progress, fitted β, and airPLS info.
    output_path : str or None
        If given, stream the corrected stack to a memmapped TIFF instead
        of a RAM buffer. Mutually exclusive with ``aggregates_only``.
    eps : float
        Floor protecting per-pixel divides.
    smooth_window : int
        Moving-average window before airPLS (paper: 10; 1 disables).
    airpls_lam, airpls_porder, airpls_max_iter : float, int, int
        airPLS baseline: smoothness penalty (1e5), difference order
        (2 ⇒ penalise curvature → piecewise-linear baseline), IRLS
        iteration cap (50).
    trim_initial : int
        Leading frames dropped from the 1-D fit only; the full T-frame
        output is still written. Default 0.
    nn_slope : bool
        Enforce β ≥ 0 (paper constraint: the control can only *add*).
    fit_mode : {'per_pixel', 'per_pixel_spatial_avg', 'per_pixel_clean_reg', \
'per_sector', 'global'}
        How β couples the channels, and — see the regressor table above —
        what gets subtracted at each pixel. 'global': one scalar OLS β
        from the spatial-mean traces, applied everywhere against the
        whole-frame mean control trace (paper-faithful). 'per_pixel'
        (default): the per-pixel Pearson r (the local coupling *pattern*),
        rescaled per sector so each sector's mean matches that sector's
        denoised β, applied to that pixel's own control trace — this
        keeps the per-pixel spatial structure while restoring the
        global-comparable magnitude that a raw per-pixel correlation
        loses to shot noise (a single pixel cannot spatially average, so
        its raw r is strongly attenuated). 'per_pixel_spatial_avg':
        identical to 'per_pixel' in every respect (per-pixel coupling
        pattern, per-sector β calibration, per-voxel s2 normalisation),
        EXCEPT that the control channel is replaced everywhere it enters —
        both the per-pixel Pearson r that sets the coupling pattern and
        the trace subtracted in pass 2 — by a Gaussian spatial average of
        s1 around each pixel (σ = ``spatial_avg_sigma_px``). Averaging a
        coherent artefact over a neighbourhood recovers far more of the
        true coupling than a single pixel's shot-noise-attenuated trace
        (the ``correct_pixel_spatial_subtr`` idea, but with a fitted β
        rather than unit gain), while keeping the per-pixel spatial
        structure that 'global'/'per_sector' discard. 'per_sector': one β
        per sector block, the denoised fit on each sector's spatially-
        averaged mean trace, applied against that same sector-mean trace,
        broadcast to every pixel of the block. All per-pixel/per-sector
        modes derive the denoised per-sector β the same way 'global'
        derives its scalar (normalised non-negative OLS slope = Pearson r
        of the smoothed, baseline-removed mean trace), just per sector.
        The sector grid is the F0 grid (``f0_n_sectors``, or
        ``aggregates_only['n_sectors']`` when streaming QC aggregates).
        'per_pixel_clean_reg': subtracts the SAME clean spatially-averaged
        regressor as 'global' / 'per_sector' (which of the two is set by
        ``clean_reg_level``), but fits β independently at every pixel by
        OLS against it. This fixes a scale mismatch the other modes share:
        β fit on a spatial mean measures the artefact fraction of a trace
        that is almost pure artefact (~0.998 of the whole-frame mean's
        variance in sim), whereas the per-voxel s2 it is subtracted from
        is only ~0.88 artefact once shot noise and real signal are folded
        into σ_xy. Subtracting the larger coefficient OVER-corrects, and
        the residual's correlation with the artefact flips sign rather
        than vanishing (sim: +0.999 raw -> −0.768 for 'global'). Fitting
        β per pixel against the same regressor removes exactly the
        coherent part present at that pixel.

        Unlike 'per_pixel', this needs no per-sector recalibration: the
        errors-in-variables attenuation that shrinks a raw per-pixel
        Pearson r comes from noise in the REGRESSOR, and a spatial mean
        is essentially noise-free. In sim the fitted β map matches the
        true per-pixel artefact fraction at r = 0.9999.

        ASSUMPTION: the real signal is uncorrelated with the control
        regressor. Where it is not — stimulus-locked designs in which
        neuromodulation and haemodynamics share a drive, or spectral
        bleed-through of the sensor into the control — β absorbs real
        signal, and does so preferentially in the most responsive pixels.
        The effect is graceful rather than catastrophic (in a toy model
        with ρ(signal, artefact) = 0.8 it still recovers more than
        'global' does), but note that ``DualColourSim`` CANNOT test this:
        it builds the two fields independently and puts exactly zero
        signal into the control channel. Use
        :func:`xval_control_residual` on real recordings.
    clean_reg_level : {'frame', 'sector'}
        Spatial scale of the clean regressor used by
        ``fit_mode='per_pixel_clean_reg'``; ignored by every other mode.
        'frame' (default) subtracts the whole-frame mean control trace,
        as 'global' does; 'sector' subtracts each pixel's sector-mean
        trace, as 'per_sector' does. 'sector' tracks spatially varying
        artefact timing at the cost of a noisier regressor (fewer pixels
        averaged), so prefer 'frame' unless the artefact is known to
        differ in *timing*, not just amplitude, across the FOV.
    min_pixel_corr : float
        Coupling-significance floor for ``fit_mode='per_pixel_clean_reg'``.
        β is zeroed at pixels whose |Pearson r| with the regressor falls
        below ``max(min_pixel_corr, 3/sqrt(N_eff))``, leaving those pixels
        uncorrected rather than subtracting a coefficient consistent with
        noise. The gate is on the CORRELATION, not on β itself: β is an
        OLS slope, and unless ``f0_level='pixel'`` the two differ by the
        pixel's brightness, so gating β would zero dim pixels and spare
        bright ones irrespective of coupling. ``N_eff = T / tau`` uses
        the regressor's integrated
        autocorrelation time tau, not the raw frame count — haemodynamic
        regressors are strongly autocorrelated (tau ≈ 11 frames in sim,
        so N_eff ≈ 224 at T = 2500), and a raw-T floor would be several
        times too permissive. Default 0.0: the 3/sqrt(N_eff) noise floor
        still applies. Mirrors ``correct_two_stage_regress``'s
        ``min_block_corr``.
    f0_level : {'sector', 'pixel', 'frame'}
        Granularity of the normalisation stats (m, σ) and the airPLS
        baseline. 'sector' (default) pools within f0_n_sectors² blocks
        (avoids per-pixel over-normalisation; baseline fit per sector to
        track spatially varying drift); 'pixel' is per-voxel; 'frame' is
        one scalar over the FOV. Variance is pooled by the law of total
        variance, so 'pixel' reduces to the per-voxel values. The global
        β fit always uses the whole-frame baseline.
    f0_n_sectors : int
        Blocks per axis for f0_level='sector'. Default 8; overridden by
        ``aggregates_only['n_sectors']`` when present so the F0 grid
        matches the QC sector grid.
    spatial_avg_sigma_px : float
        Gaussian sigma (pixels) of the spatial average used as the
        control regressor when ``fit_mode='per_pixel_spatial_avg'``.
        Ignored by every other fit_mode. Default 2.0 (matches
        ``correct_pixel_spatial_subtr``'s ``sigma_px``: the centre pixel
        then carries ≈ 4% of the kernel weight, so this really is a
        neighbourhood average). A normalised convolution with
        ``mode='reflect'`` handles the FOV edges, and sigma acts on the
        two spatial axes only (never across time), so the streamed
        per-batch average equals filtering the whole stack. 0 disables
        the averaging and reduces the mode to plain 'per_pixel'.
    aggregates_only : dict or None
        If e.g. ``{'n_sectors': 8}``, skip the (T, X, Y) buffer and
        stream-compute the whole-frame + per-sector mean of zdFF
        directly (mirrors ``QCMixin._compute_sectors``); returns those
        aggregates instead of the full stack, avoiding the big alloc +
        write for the QC pipeline. Default None.
    beta_loss : {'huber', 'cauchy', 'soft_l1', 'arctan', 'linear'}
        Loss for the 1-D coupling-slope fit. 'huber' (default) is a robust
        M-estimator fit via ``scipy.optimize.least_squares`` that down-
        weights outlier frames (motion / z-drift / saturation); 'linear'
        recovers the closed-form non-negative OLS slope. Affects the global
        and per-sector β only (≤ n_sectors² + 1 scalar fits); per-pixel β
        inherits the robustness through its per-sector calibration. Never
        run per pixel.
    beta_f_scale : float
        Soft threshold of the robust loss. With ``regress_type='ols'`` it is
        absolute, in normalised-trace units (≈ trace scale); with
        ``'irls'`` it multiplies the residual MAD re-estimated at each
        iteration. Only used when ``beta_loss != 'linear'``. Default 1.
    regress_type : {'ols', 'irls'}
        Solver used for the robust coupling-slope fit. 'ols' (default)
        minimises the robustified residual with
        ``scipy.optimize.least_squares`` (trust-region reflective, seeded
        at the analytic OLS slope) — the previous behaviour. 'irls' runs
        iteratively reweighted least squares instead: repeated closed-form
        weighted OLS with weights w = ρ'(u²) recomputed from the current
        residuals, u = r / (``beta_f_scale`` × MAD(r)). Both target the
        same M-estimator defined by ``beta_loss``; IRLS is a few NumPy
        reductions per iteration (so noticeably cheaper across many
        sectors), and its MAD-rescaled threshold makes it
        scale-equivariant — the same *fraction* of frames is down-weighted
        regardless of a sector's absolute noise level, which the fixed
        ``beta_f_scale`` of the 'ols' path does not guarantee across a
        dim FOV. Ignored when ``beta_loss='linear'`` (unit weights ⇒ IRLS
        collapses to the analytic OLS slope). Like ``beta_loss``, it
        affects the global and per-sector β only; per-pixel β inherits it
        through the per-sector calibration.
    beta_scale : {'mad', 'std'}
        Scale estimator for the trace normalisation that precedes the β
        fit. 'mad' (default) is the median absolute deviation × 1.4826 —
        equal to σ for Gaussian data but not inflated by outlier frames,
        which would otherwise attenuate the normalised correlation (β).
        'std' is the classical standard deviation. Pass
        ``beta_loss='linear', beta_scale='std'`` to recover the exact
        pre-robust behaviour.

    Returns
    -------
    corrected : (T, X, Y) ndarray / memmap, or dict
        Per-pixel zdFF. With ``aggregates_only`` set, instead returns
        ``{'frame_f': (T,), 'sector_f': (n_sec², T), 'n_sectors': int,
        'is_aggregates': True}``.
    coefficients : (X, Y, 2) float32
        ``[:, :, 0]`` = per-pixel session mean of (s2 − b2), a raw-unit
        intercept diagnostic; ``[:, :, 1]`` = β (scalar in 'global',
        per-pixel in 'per_pixel').
    """
    if s1.shape != s2.shape:
        raise ValueError(
            f"s1 and s2 must have same shape, "
            f"got {s1.shape} and {s2.shape}")
    if fit_mode not in ('global', 'per_pixel', 'per_pixel_spatial_avg',
                        'per_pixel_clean_reg', 'per_sector'):
        raise ValueError(
            "fit_mode must be 'global', 'per_pixel', "
            "'per_pixel_spatial_avg', 'per_pixel_clean_reg' or "
            f"'per_sector', got {fit_mode!r}")
    if regress_type not in ('ols', 'irls'):
        raise ValueError(
            f"regress_type must be 'ols' or 'irls', got {regress_type!r}")
    if clean_reg_level not in ('frame', 'sector'):
        raise ValueError(
            "clean_reg_level must be 'frame' or 'sector', "
            f"got {clean_reg_level!r}")

    # 'per_pixel_spatial_avg' is 'per_pixel' with the control channel
    # replaced by a Gaussian neighbourhood average of s1 (a normalised
    # convolution, mode='reflect') everywhere it enters — the per-pixel
    # coupling pattern in STEP 5 and the trace subtracted in STEP 6.
    per_pixel_mode = fit_mode in ('per_pixel', 'per_pixel_spatial_avg')
    spatial_avg = (fit_mode == 'per_pixel_spatial_avg')
    # 'per_pixel_clean_reg' shares its REGRESSOR with 'global' /
    # 'per_sector' (so it is not a per_pixel_mode: no per-voxel s1
    # normalisation, no per-pixel cov accumulation) but fits β per pixel
    # against it in the extra pass added after STEP 5.
    clean_reg_mode = (fit_mode == 'per_pixel_clean_reg')
    clean_reg_sector = clean_reg_mode and clean_reg_level == 'sector'
    if spatial_avg:
        from scipy.ndimage import gaussian_filter
        # sigma acts on the two spatial axes only, never across time, so a
        # per-batch filter equals filtering the whole stack.
        _sig_xyz = (0.0, float(spatial_avg_sigma_px),
                    float(spatial_avg_sigma_px))

    T, X, Y = s1.shape
    s1, s2, batch_size = _prepare_inputs(
        s1, s2, batch_size, T, X, Y, verbose)
    n_batches = (T + batch_size - 1) // batch_size

    # ==================================================================
    # STEP 0 — Setup: output buffers and the shared sector grid
    # ==================================================================
    # Configure the output: full (T, X, Y) stack, or (aggregates_only)
    # just its per-batch whole-frame and per-sector means. The aggregate
    # sector grid mirrors QCMixin._compute_sectors: trim the FOV to dims
    # divisible by n_sec, then carve into uniform n_sec x n_sec blocks.
    return_aggregates = aggregates_only is not None
    if return_aggregates and output_path is not None:
        raise ValueError(
            "aggregates_only and output_path are mutually exclusive")

    if return_aggregates:
        agg = SimpleNamespace()
        agg.n_sec = int(aggregates_only.get('n_sectors', 8))
        agg.x_trim = (X // agg.n_sec) * agg.n_sec
        agg.y_trim = (Y // agg.n_sec) * agg.n_sec
        agg.block_x = agg.x_trim // agg.n_sec
        agg.block_y = agg.y_trim // agg.n_sec
        agg.frame_mean = np.zeros(T, dtype=np.float32)
        agg.sector_mean = np.zeros((agg.n_sec * agg.n_sec, T),
                                   dtype=np.float32)

    if verbose:
        _disk = " -> disk" if output_path else ""
        _agg = f' [aggregates n_sec={agg.n_sec}]' if return_aggregates else ''
        print(f'\tcorrecting signal (full_regress{_disk}){_agg}...')

    # Output buffer: full RAM stack, memmapped TIFF, or none (aggregates).
    if return_aggregates:
        corrected = None
    elif output_path is not None:
        if verbose:
            print(f'\t\twriting output to: {output_path}')
        corrected = tifffile.memmap(
            output_path, shape=(T, X, Y), dtype=dtype, bigtiff=True)
    else:
        corrected = np.zeros((T, X, Y), dtype=dtype)

    # coefficients[:, :, 0] = per-pixel intercept diagnostic (center_s2);
    # coefficients[:, :, 1] = beta (scalar for 'global', (X, Y) otherwise).
    coefficients = np.zeros((X, Y, 2), dtype=np.float32)

    # Build the F0 / baseline sector grid — this is the SAME grid used by
    # STEP 4's per-sector fit, STEP 5's per_sector regressor, and STEP 3's
    # per-sector s2 normalisation. For f0_level='sector' one slow airPLS
    # baseline is fit per sector so spatially varying drift is removed
    # region-by-region; 'frame'/'pixel' keep a single whole-frame baseline.
    # f0.block_of_pixel maps every pixel (row-major) to its sector index.
    n_sec_f0 = (int(aggregates_only.get('n_sectors', f0_n_sectors))
                if return_aggregates else int(f0_n_sectors))
    use_sector_baseline = (f0_level == 'sector' and n_sec_f0 >= 1)
    # 'per_pixel_clean_reg' only needs the sector grid when its regressor
    # is the sector mean; at clean_reg_level='frame' it is as
    # sector-independent as 'global'.
    _needs_sectors = (fit_mode not in ('global', 'per_pixel_clean_reg')
                      or clean_reg_sector)
    if _needs_sectors and n_sec_f0 < 1:
        raise ValueError(
            f"fit_mode={fit_mode!r} requires f0_n_sectors >= 1 "
            f"(got {n_sec_f0})")
    # per_sector and per_pixel(_spatial_avg) all fit a denoised β on each
    # sector's mean trace: per_sector applies it directly; the per-pixel
    # modes use it to calibrate the magnitude of their per-pixel coupling
    # pattern. So all three need the per-sector mean traces and grid.
    # 'per_pixel_clean_reg' fits its β per pixel instead, so it needs the
    # sector TRACES (as its regressor, when clean_reg_level='sector') but
    # never the per-sector β fits.
    fit_sector_betas = (fit_mode not in ('global', 'per_pixel_clean_reg')
                        and n_sec_f0 >= 1)
    need_sector_traces = (use_sector_baseline or fit_sector_betas
                          or clean_reg_sector)
    if need_sector_traces:
        f0 = SimpleNamespace()
        f0.n_sec = n_sec_f0
        f0.n_blocks = n_sec_f0 * n_sec_f0
        f0.block_x = max(1, X // n_sec_f0)
        f0.block_y = max(1, Y // n_sec_f0)
        f0.x_trim = f0.block_x * n_sec_f0
        f0.y_trim = f0.block_y * n_sec_f0
        _row_block = np.minimum(np.arange(X) // f0.block_x, n_sec_f0 - 1)
        _col_block = np.minimum(np.arange(Y) // f0.block_y, n_sec_f0 - 1)
        f0.block_of_pixel = (_row_block[:, None] * n_sec_f0
                             + _col_block[None, :]).astype(np.intp)
        sector_trace_s1 = np.zeros((f0.n_blocks, T), dtype=np.float64)
        sector_trace_s2 = np.zeros((f0.n_blocks, T), dtype=np.float64)

    # ==================================================================
    # STEP 1 — Pass 1: stream the stack once, accumulate spatial-mean
    # traces and per-pixel session moments
    # ==================================================================
    # One streaming pass over every frame collects everything the later
    # steps need, so pass 2 (STEP 6) never has to revisit old batches.
    #   1a. whole-frame mean trace, both channels -> feeds STEP 2
    #   1b. sector-mean trace, both channels       -> feeds STEP 4, and
    #                                                  (for 'global' /
    #                                                  'per_sector') is
    #                                                  itself the raw
    #                                                  material for the
    #                                                  STEP 5 regressor
    #   1c. per-pixel session mean/var(/cov)        -> feeds STEP 3's s2
    #                                                  normalisation
    #                                                  ('per_pixel' also
    #                                                  uses the covariance
    #                                                  for its STEP 5
    #                                                  coupling pattern)
    # For 'per_pixel_spatial_avg' the per-pixel s1 moments (mean, var, and
    # cross-moment with s2) are accumulated from the Gaussian-averaged
    # control s̃1 rather than raw s1, so STEP 5a's Pearson r and STEP 3c's
    # s1 normalisation both refer to the averaged control automatically.
    need_cov = per_pixel_mode
    if verbose:
        print(f'\t\tpass 1/2: spatial means & per-pixel moments '
              f'({n_batches} batches)...')
    mean_trace_s1 = np.zeros(T, dtype=np.float64)
    mean_trace_s2 = np.zeros(T, dtype=np.float64)
    pix_sum_s1 = np.zeros((X, Y), dtype=np.float64)
    pix_sum_s2 = np.zeros((X, Y), dtype=np.float64)
    pix_sum_s1sq = np.zeros((X, Y), dtype=np.float64)
    pix_sum_s2sq = np.zeros((X, Y), dtype=np.float64)
    pix_sum_s1s2 = np.zeros((X, Y), dtype=np.float64) if need_cov else None

    for bi, t_start, t_end, s1_b, s2_b in _batched_f32(s1, s2, T, batch_size):
        if verbose:
            print(f'\t\t\tbatch {bi+1}/{n_batches} '
                  f'({(bi+1)/n_batches*100:.1f}%)...', end='\r')
        # 1a. whole-frame mean trace: average ACROSS ALL PIXELS, per frame.
        mean_trace_s1[t_start:t_end] = s1_b.mean(axis=(1, 2), dtype=np.float64)
        mean_trace_s2[t_start:t_end] = s2_b.mean(axis=(1, 2), dtype=np.float64)
        if need_sector_traces:
            # 1b. sector-mean trace: average ACROSS THE PIXELS OF EACH
            # SECTOR, per frame — one (n_blocks,) vector per frame.
            sector_trace_s1[:, t_start:t_end] = _sector_means(
                s1_b, f0.n_sec, f0.block_x, f0.block_y, f0.x_trim, f0.y_trim)
            sector_trace_s2[:, t_start:t_end] = _sector_means(
                s2_b, f0.n_sec, f0.block_x, f0.block_y, f0.x_trim, f0.y_trim)
        # 1c. per-pixel session moments (mean, var, and cross-moment for
        # 'per_pixel') — running sums, finalised once the loop ends. For
        # 'per_pixel_spatial_avg' the s1 moments come from the Gaussian
        # neighbourhood average s̃1 (a normalised convolution — with no
        # mask, mode='reflect' already sums the kernel to 1 everywhere, so
        # a plain gaussian_filter of the data is the average); the whole-
        # frame / sector traces above still use raw s1 (baselines + β fit).
        s1_px = (gaussian_filter(s1_b, sigma=_sig_xyz, mode='reflect')
                 if spatial_avg else s1_b)
        pix_sum_s1 += s1_px.sum(axis=0, dtype=np.float64)
        pix_sum_s2 += s2_b.sum(axis=0, dtype=np.float64)
        pix_sum_s1sq += np.einsum('ijk,ijk->jk', s1_px, s1_px, dtype=np.float64)
        pix_sum_s2sq += np.einsum('ijk,ijk->jk', s2_b, s2_b, dtype=np.float64)
        if need_cov:
            pix_sum_s1s2 += np.einsum(
                'ijk,ijk->jk', s1_px, s2_b, dtype=np.float64)
    if verbose:
        print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')

    # 1c (finalise). Per-pixel session moments (float64 accumulation,
    # float32 storage — numerically stable for large T), grouped in `px`.
    px = SimpleNamespace()
    px.mean_s1 = (pix_sum_s1 / T).astype(np.float32)
    px.mean_s2 = (pix_sum_s2 / T).astype(np.float32)
    px.var_s1 = np.maximum(
        pix_sum_s1sq / T - px.mean_s1.astype(np.float64) ** 2,
        1e-10).astype(np.float32)
    px.var_s2 = np.maximum(
        pix_sum_s2sq / T - px.mean_s2.astype(np.float64) ** 2,
        1e-10).astype(np.float32)
    if need_cov:
        px.cov_s1s2 = (
            pix_sum_s1s2 / T
            - px.mean_s1.astype(np.float64)
            * px.mean_s2.astype(np.float64)).astype(np.float32)
    del pix_sum_s1, pix_sum_s2, pix_sum_s1sq, pix_sum_s2sq, pix_sum_s1s2

    # ==================================================================
    # STEP 2 — Fit the coupling slope: GLOBAL (whole-frame) fit
    # ==================================================================
    # This whole block is FIT-ONLY: nothing computed here is subtracted
    # directly from the raw data — it only produces the dimensionless
    # slope beta_global, and (as a side effect) the whole-frame airPLS
    # baseline reused wherever f0_level != 'sector'. See STEP 5 for what
    # actually gets subtracted.
    #   i.   smooth    — moving-average lowpass, denoising for the fit only
    #   ii.  debase    — airPLS baseline removal, per channel
    #   iii. trim      — drop the first trim_initial warm-up frames
    #                    (fit only; STEP 6 still corrects every frame)
    #   iv.  normalise — median-subtract, MAD/std-divide
    #   v.   fit       — OLS / robust slope of s2 on s1 (the centring
    #                    stands in for an intercept)
    # i. smooth (fit denoising only).
    trace_s1_smooth = _moving_average(mean_trace_s1, smooth_window)
    trace_s2_smooth = _moving_average(mean_trace_s2, smooth_window)
    if verbose:
        print('\t\trunning airPLS baselines on s̄1(t), s̄2(t)...')
    # ii. debase: airPLS baseline of the smoothed trace, per channel.
    baseline_s1 = _airpls(trace_s1_smooth, lam=airpls_lam,
                          porder=airpls_porder, max_iter=airpls_max_iter)
    baseline_s2 = _airpls(trace_s2_smooth, lam=airpls_lam,
                          porder=airpls_porder, max_iter=airpls_max_iter)

    # iii. trim: warm-up frames dropped from the slope fit only; the full
    # T-frame output is still written in STEP 6.
    n_warmup = int(max(0, min(trim_initial, T - 2)))
    trace_s1_debased = trace_s1_smooth[n_warmup:] - baseline_s1[n_warmup:]
    trace_s2_debased = trace_s2_smooth[n_warmup:] - baseline_s2[n_warmup:]
    # iv. normalise: median-centre, then divide by a robust (MAD, default)
    # or classical (std) scale.
    trace_s1_norm, med_s1, std_s1 = _centre_scale(trace_s1_debased,
                                                  beta_scale)
    trace_s2_norm, med_s2, std_s2 = _centre_scale(trace_s2_debased,
                                                  beta_scale)

    # v. fit: coupling slope of normalised s̄2 on normalised s̄1 —
    # non-negative OLS (beta_loss='linear') or a robust M-estimator
    # slope otherwise (the median centring stands in for an intercept).
    beta_global = _fit_norm_slope(
        trace_s1_norm, trace_s2_norm, nn_slope=nn_slope,
        loss=beta_loss, f_scale=beta_f_scale, regress_type=regress_type)
    if verbose:
        _beta_ols_ref = _fit_norm_slope(
            trace_s1_norm, trace_s2_norm, nn_slope=nn_slope, loss='linear')
        print(f'\t\t\tβ₁(norm) = {beta_global:.6g} '
              f'(OLS ref {_beta_ols_ref:.6g}, loss={beta_loss}, '
              f'solver={regress_type}, '
              f'nn={"on" if nn_slope else "off"})')
        print(f'\t\t\tσ̄1={std_s1:.4g}, σ̄2={std_s2:.4g}, '
              f'm̄1={med_s1:.4g}, m̄2={med_s2:.4g}')

    # ==================================================================
    # STEP 3 — Normalise the signal channel s2 (independent of beta)
    # ==================================================================
    # This builds the OTHER operand of the STEP 6 subtraction: per-voxel
    # centring m_xy and scale sigma_xy for s2 (and, for fit_mode=
    # 'per_pixel', for s1 too), pooled to whatever granularity f0_level
    # asks for ('pixel' / 'sector' / 'frame'). It runs before the sector
    # fits because 3a's per-sector airPLS baselines are shared with
    # STEP 4's per-sector fit. Everything pass 2 (STEP 6) needs is
    # precomputed in this section, so the streaming loop there only does
    # arithmetic.
    mean_baseline_s1 = np.float32(np.mean(baseline_s1))
    mean_baseline_s2 = np.float32(np.mean(baseline_s2))

    # ------------------------------------------------------------------
    # 3a. Per-sector airPLS baselines — shared with STEP 4's per-sector
    #     fit and STEP 5's 'per_sector' regressor
    # ------------------------------------------------------------------
    # From the pass-1 (STEP 1b) sector mean traces, fit in parallel over
    # sectors: smooth (denoising) then airPLS baseline, per sector, per
    # channel. (The global slope in STEP 2 used the whole-frame baseline
    # instead.)
    if need_sector_traces:
        if verbose:
            print(f'\t\trunning per-sector airPLS baselines '
                  f'({f0.n_blocks} sectors × 2)...')
        _smooth_s1 = [_moving_average(sector_trace_s1[_s], smooth_window)
                      for _s in range(f0.n_blocks)]
        _smooth_s2 = [_moving_average(sector_trace_s2[_s], smooth_window)
                      for _s in range(f0.n_blocks)]
        sector_baseline_s1 = np.asarray(Parallel(n_jobs=-1, prefer='threads')(
            delayed(_airpls)(_x, lam=airpls_lam, porder=airpls_porder,
                             max_iter=airpls_max_iter) for _x in _smooth_s1),
            dtype=np.float32)
        sector_baseline_s2 = np.asarray(Parallel(n_jobs=-1, prefer='threads')(
            delayed(_airpls)(_x, lam=airpls_lam, porder=airpls_porder,
                             max_iter=airpls_max_iter) for _x in _smooth_s2),
            dtype=np.float32)
        del sector_trace_s2
    # ------------------------------------------------------------------
    # 3b. Per-frame baseline actually subtracted from s2 in STEP 6: per-
    #     sector (broadcast to pixels) if f0_level='sector', else a
    #     single whole-frame scalar for every pixel.
    # ------------------------------------------------------------------
    if use_sector_baseline:
        # Per-sector baseline time-means replace the scalar baseline mean
        # in the centring constant so the two stay consistent.
        mean_baseline_s1_xy = sector_baseline_s1.mean(
            axis=1)[f0.block_of_pixel].astype(np.float32)
        mean_baseline_s2_xy = sector_baseline_s2.mean(
            axis=1)[f0.block_of_pixel].astype(np.float32)
    else:
        mean_baseline_s1_xy = float(mean_baseline_s1)
        mean_baseline_s2_xy = float(mean_baseline_s2)

    # ------------------------------------------------------------------
    # 3c. Centring (m_xy) and scale (sigma_xy) for the dF/F normalisation
    #     — this is s2_norm's normalisation, used by every fit_mode
    # ------------------------------------------------------------------
    # Pooled to the requested F0 granularity:
    #   center = (pooled) session-mean of (s − baseline)
    #   scale  = (pooled) std of s  (Var(s−baseline) ≈ Var(s), slow base)
    pooled_mean_s1, pooled_var_s1 = _pool_moments(
        px.mean_s1, px.var_s1, f0_level, n_sec_f0)
    pooled_mean_s2, pooled_var_s2 = _pool_moments(
        px.mean_s2, px.var_s2, f0_level, n_sec_f0)
    if verbose:
        _bl = 'per-sector' if use_sector_baseline else 'whole-frame'
        print(f'\t\tF0 granularity: {f0_level}'
              f'{f" (n_sec={n_sec_f0})" if f0_level == "sector" else ""}'
              f'; airPLS baseline: {_bl}')

    # Per-voxel normalisation constants, grouped in `norm` for pass 2.
    norm = SimpleNamespace()
    norm.center_s1 = (pooled_mean_s1 - mean_baseline_s1_xy).astype(np.float32)
    norm.center_s2 = (pooled_mean_s2 - mean_baseline_s2_xy).astype(np.float32)
    scale_s1_xy = np.sqrt(pooled_var_s1).astype(np.float32)
    scale_s2_xy = np.sqrt(pooled_var_s2).astype(np.float32)
    # Robust floor so dark pixels (sigma ≪ median) don't amplify noise.
    scale_s1_med = float(np.median(scale_s1_xy))
    scale_s2_med = float(np.median(scale_s2_xy))
    scale_s1_floor = np.float32(max(scale_s1_med * 0.01, eps))
    scale_s2_floor = np.float32(max(scale_s2_med * 0.01, eps))
    scale_s1_xy = np.maximum(scale_s1_xy, scale_s1_floor)
    scale_s2_xy = np.maximum(scale_s2_xy, scale_s2_floor)
    # Pre-invert so pass 2 multiplies instead of dividing per voxel.
    norm.inv_scale_s1 = (np.float32(1.0) / scale_s1_xy).astype(np.float32)
    norm.inv_scale_s2 = (np.float32(1.0) / scale_s2_xy).astype(np.float32)

    # ==================================================================
    # STEP 4 — Fit the coupling slope: PER-SECTOR fits
    # ==================================================================
    # Needed by fit_mode='per_sector' (its beta applied directly) and
    # 'per_pixel' (used to calibrate the per-pixel coupling *pattern* up
    # to a global-comparable magnitude — a raw per-pixel correlation is
    # attenuated far below the true coupling by shot noise, since a
    # single pixel cannot spatially average). Same five sub-steps as
    # STEP 2 (smooth -> debase -> normalise -> fit), just independently
    # per sector, reusing 3a's per-sector airPLS baselines.
    if fit_sector_betas:
        # i-v, per sector (already smoothed in 3a): debase -> normalise
        # -> fit. beta_sector[s] = OLS/robust slope, i.e. the Pearson r
        # of that sector's smoothed, baseline-removed trace.
        beta_sector = np.empty(f0.n_blocks, dtype=np.float64)
        for _s in range(f0.n_blocks):
            _d1 = (_smooth_s1[_s][n_warmup:]
                   - sector_baseline_s1[_s][n_warmup:])
            _d2 = (_smooth_s2[_s][n_warmup:]
                   - sector_baseline_s2[_s][n_warmup:])
            _n1, _, _ = _centre_scale(_d1, beta_scale)
            _n2, _, _ = _centre_scale(_d2, beta_scale)
            beta_sector[_s] = _fit_norm_slope(
                _n1, _n2, nn_slope=nn_slope,
                loss=beta_loss, f_scale=beta_f_scale,
                regress_type=regress_type)

    # ==================================================================
    # STEP 5 — Build the regressor that is ACTUALLY SUBTRACTED (per
    # fit_mode) — raw (unsmoothed), not the STEP 2/4 fit trace
    # ==================================================================
    # The moving-average smoothing in STEPs 2 and 4 was for denoising the
    # fit only. Subtracting that SMOOTHED trace from the raw per-frame
    # data would blur out the fast-varying part of a genuinely coherent
    # artefact and leave it in the residual — so every branch below
    # rebuilds its regressor from the RAW pass-1 trace: debase it by the
    # matching STEP 2/3a baseline, then re-normalise by ITS OWN median/
    # scale (not the smoothed trace's, which has a different, smaller
    # scale).
    #   'per_pixel'  (5a): no precomputed regressor — STEP 6 normalises
    #                      each pixel's own raw control trace directly.
    #   'per_pixel_spatial_avg' (5a): same, but STEP 6 normalises each
    #                      pixel's Gaussian-averaged control trace.
    #   'per_sector' (5b): each sector's own raw mean control trace,
    #                      broadcast to every pixel of that sector.
    #   'global'     (5c): the whole-frame raw mean control trace,
    #                      identical for every pixel.
    if per_pixel_mode:
        # 5a. Per-pixel coupling pattern: the per-pixel Pearson r. For
        # 'per_pixel' it is the raw per-pixel r (noise-attenuated, hence
        # too small on its own); for 'per_pixel_spatial_avg' px.var_s1 /
        # px.cov_s1s2 were accumulated from the Gaussian-averaged control
        # s̃1 (STEP 1c), so r is already less attenuated. Either way the
        # per-sector calibration below rescales it to the sector β.
        raw_scale_s1 = np.sqrt(px.var_s1).astype(np.float32)
        raw_scale_s2 = np.sqrt(px.var_s2).astype(np.float32)
        denom_pearson = np.maximum(raw_scale_s1 * raw_scale_s2,
                                   np.float32(eps))
        r_pix = (px.cov_s1s2 / denom_pearson).astype(np.float64)
        if nn_slope:
            r_pix = np.maximum(r_pix, 0.0)
        r_pix = np.clip(r_pix, -1.0, 1.0)
        # Scale each sector's mean r up to that sector's denoised β. This
        # restores the global-comparable magnitude (undoing the per-pixel
        # noise attenuation) while keeping the per-pixel spatial pattern.
        # Sectors whose mean r is too small to calibrate reliably fall
        # back to the sector β.
        _flat = f0.block_of_pixel.ravel()
        _cnt = np.bincount(_flat, minlength=f0.n_blocks)
        _sum_r = np.bincount(_flat, weights=r_pix.ravel(),
                             minlength=f0.n_blocks)
        _mean_r = _sum_r / np.maximum(_cnt, 1)
        _reliable = _mean_r > 1e-2
        _gain = np.ones(f0.n_blocks, dtype=np.float64)
        _gain[_reliable] = beta_sector[_reliable] / _mean_r[_reliable]
        _beta_pix = r_pix * _gain[f0.block_of_pixel]
        # Unreliable (near-zero coupling) sectors: use the sector β as-is.
        _unrel = ~_reliable[f0.block_of_pixel]
        _beta_pix[_unrel] = beta_sector[f0.block_of_pixel][_unrel]
        if nn_slope:
            _beta_pix = np.maximum(_beta_pix, 0.0)
        _beta_pix = np.minimum(_beta_pix, 1.0)
        norm.beta = _beta_pix.astype(np.float32)
        # Per-pixel β couples a per-voxel control trace, so pass 2
        # normalises s1 per voxel rather than using a precomputed trace.
        # For 'per_pixel_spatial_avg' pass 2 first Gaussian-averages the
        # s1 batch (the `spatial_avg` branch in STEP 6); center_s1 /
        # inv_scale_s1 already describe s̃1 (STEP 3c pooled the averaged
        # moments), so the same normalisation then applies.
        norm.reg_trace = None
        coefficients[:, :, 0] = norm.center_s2
        coefficients[:, :, 1] = norm.beta
        del raw_scale_s1, raw_scale_s2, denom_pearson, r_pix, px.cov_s1s2
    elif fit_mode == 'per_sector' or clean_reg_sector:
        # 5b. One β per sector, broadcast to every pixel of its block.
        # (For 'per_pixel_clean_reg' only the REGRESSOR is built here —
        # its β is fit per pixel against it in STEP 5d below.)
        if fit_mode == 'per_sector':
            norm.beta = beta_sector.astype(np.float32)[f0.block_of_pixel]
        # Regressor: each sector's own RAW (unsmoothed) mean control trace,
        # debased and normalised by its own median/scale. Unsmoothed so no
        # temporal blur is subtracted from the raw per-frame signal (the
        # moving average is for denoising the β *fit* only); beta_sector
        # transfers from the smoothed fit to this raw trace the same way
        # the per_pixel branch already transfers a sector β onto raw
        # per-pixel data.
        _debased_sector_raw = sector_trace_s1 - sector_baseline_s1
        norm.reg_trace = np.stack(
            [_centre_scale(_debased_sector_raw[_s], beta_scale)[0]
             for _s in range(f0.n_blocks)]).astype(np.float32)
        coefficients[:, :, 0] = norm.center_s2
        if fit_mode == 'per_sector':
            coefficients[:, :, 1] = norm.beta
        del sector_trace_s1, _debased_sector_raw
    else:  # 'global', or 'per_pixel_clean_reg' at clean_reg_level='frame'
        # 5c. One global β, identical for every pixel. (Again, for
        # 'per_pixel_clean_reg' only the regressor is built here.)
        if not clean_reg_mode:
            norm.beta = np.float32(beta_global)
        # Regressor: the whole-frame RAW (unsmoothed) mean control trace,
        # debased and normalised by its own median/scale — see the
        # per_sector comment above for why it must be unsmoothed.
        norm.reg_trace = _centre_scale(
            mean_trace_s1 - baseline_s1, beta_scale)[0].astype(np.float32)
        coefficients[:, :, 0] = norm.center_s2
        if not clean_reg_mode:
            coefficients[:, :, 1] = float(norm.beta)
    del px.var_s1, px.var_s2

    # Smoothed sector traces are only needed for the per-sector β fit; the
    # per-sector baselines persist only when pass 2 subtracts them. Raw
    # sector_trace_s1 is only kept past this point by fit_mode='per_sector'
    # (deleted above once its reg_trace is built); the per-pixel modes
    # never use it.
    if need_sector_traces:
        del _smooth_s1, _smooth_s2
        if not use_sector_baseline:
            del sector_baseline_s1, sector_baseline_s2
        if per_pixel_mode:
            del sector_trace_s1

    baseline_s1_f32 = baseline_s1.astype(np.float32)
    baseline_s2_f32 = baseline_s2.astype(np.float32)

    # ==================================================================
    # STEP 5d — 'per_pixel_clean_reg': fit beta per pixel against the
    # STEP 5 regressor (one EXTRA streaming pass over s2 only)
    # ==================================================================
    # Every other fit_mode derives beta from quantities pass 1 already
    # accumulated. This one cannot: the regressor is only defined once
    # the airPLS baseline exists (STEP 2), which is after pass 1 has
    # finished streaming, so the per-pixel cross-moment
    # Σ_t s2_norm(t,x,y)·r_norm(t,x,y) needs its own pass. Only s2 is
    # read (the regressor is a precomputed 1-D / per-block trace), and
    # only this fit_mode pays the cost.
    #
    # In normalised space the minimum-residual coefficient is
    #     beta(x,y) = Σ_t s2_norm·r_norm / Σ_t r_norm²
    # the ordinary least-squares slope, which minimises the residual
    # variance at that pixel. Because the regressor is a spatial mean it
    # carries almost no shot noise, so this is NOT attenuated the way
    # fit_mode='per_pixel' is (see that branch's per-sector
    # recalibration): extra variance from real signal and shot noise
    # lands in the pixel's own scale, never in the regressor.
    #
    # NOTE the slope and the correlation are the same number only when
    # s2_norm has unit variance per pixel, i.e. f0_level='pixel'. At the
    # default f0_level='frame' the scale is pooled over the whole FOV, so
    # a bright pixel's slope legitimately exceeds 1 and a dim pixel's is
    # small regardless of how tightly it tracks the artefact. The SLOPE is
    # what must be subtracted, but the gate below has to be applied to the
    # CORRELATION — otherwise it would zero dim pixels and clip bright
    # ones on the strength of their brightness rather than their coupling.
    # So the pass accumulates Σ s2_norm² as well, and the two quantities
    # are kept separate.
    if clean_reg_mode:
        if verbose:
            print(f'\t\tpass 1b/3: per-pixel β vs clean regressor '
                  f'(level={clean_reg_level}, {n_batches} batches)...')
        _num = np.zeros((X, Y), dtype=np.float64)
        _sum_yy = np.zeros((X, Y), dtype=np.float64)
        _reg = norm.reg_trace
        # Σ_t r² : scalar for a whole-frame regressor, per-block for a
        # sector one. Both are known in closed form from reg_trace.
        if _reg.ndim == 1:
            _den = np.float64(np.dot(_reg.astype(np.float64),
                                     _reg.astype(np.float64)))
            _tau = _autocorr_time(_reg)
        else:
            _den_blocks = np.einsum('bt,bt->b', _reg.astype(np.float64),
                                    _reg.astype(np.float64))
            _den = _den_blocks[f0.block_of_pixel]
            _tau = float(np.median([_autocorr_time(_reg[_s])
                                    for _s in range(_reg.shape[0])]))
        for bi, t_start, t_end, s2_b in _batched_f32_one(
                s2, T, batch_size):
            if verbose:
                print(f'\t\t\tbatch {bi+1}/{n_batches} '
                      f'({(bi+1)/n_batches*100:.1f}%)...', end='\r')
            # Normalise s2 exactly as STEP 6 will (6a + 6b), so beta is
            # fit against the same operand it will be subtracted from.
            if use_sector_baseline:
                s2_b -= sector_baseline_s2[:, t_start:t_end].T[
                    :, f0.block_of_pixel]
            else:
                s2_b -= baseline_s2_f32[t_start:t_end][:, None, None]
            s2_b -= norm.center_s2
            s2_b *= norm.inv_scale_s2
            _sum_yy += np.einsum('txy,txy->xy', s2_b, s2_b,
                                 dtype=np.float64)
            if _reg.ndim == 1:
                _num += np.einsum('txy,t->xy', s2_b,
                                  _reg[t_start:t_end], dtype=np.float64)
            else:
                _num += np.einsum(
                    'txy,txy->xy', s2_b,
                    _reg[:, t_start:t_end].T[:, f0.block_of_pixel],
                    dtype=np.float64)
        if verbose:
            print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')

        _beta_pix = _num / np.maximum(_den, 1e-10)
        # Pearson r of the same fit, used only for the gate (see the note
        # above on why the slope itself must not be gated or capped).
        _rho_pix = _num / np.maximum(np.sqrt(_sum_yy * _den), 1e-10)
        # Coupling-significance gate, mirroring correct_two_stage_regress's
        # min_block_corr — but on N_eff = T/tau rather than raw T, since a
        # haemodynamic regressor is heavily autocorrelated (_autocorr_time).
        _n_eff = max(T / max(_tau, 1.0), 1.0)
        _gate = max(float(min_pixel_corr), 3.0 / np.sqrt(_n_eff))
        _weak = np.abs(_rho_pix) < _gate
        _n_gated = int(_weak.sum())
        _beta_pix[_weak] = 0.0
        if nn_slope:
            _beta_pix = np.maximum(_beta_pix, 0.0)
        norm.beta = _beta_pix.astype(np.float32)
        coefficients[:, :, 1] = norm.beta
        if verbose:
            print(f'\t\t\tτ={_tau:.1f} frames, N_eff={_n_eff:.0f}, '
                  f'gate |r|<{_gate:.4f} -> {_n_gated}/{X*Y} px zeroed; '
                  f'|r| median {float(np.median(np.abs(_rho_pix))):.3f}')
        del _num, _sum_yy, _reg

    if verbose:
        if fit_mode != 'global':
            _b = np.asarray(norm.beta)
            _tag = f'{fit_mode} (n_sec={max(1, n_sec_f0)})'
            print(f'\t\tfit_mode={_tag}: β median '
                  f'{float(np.median(_b)):.4f}, range '
                  f'[{float(_b.min()):.4f}, {float(_b.max()):.4f}] '
                  f'(global β ref {beta_global:.4f})')
        print(f'\t\tσ̂1 median {scale_s1_med:.4g} '
              f'(floor {float(scale_s1_floor):.4g}), '
              f'σ̂2 median {scale_s2_med:.4g} '
              f'(floor {float(scale_s2_floor):.4g})')

    # ==================================================================
    # STEP 6 — Pass 2: stream the stack again, normalise s2, subtract
    # beta * regressor, write / aggregate
    # ==================================================================
    # Per voxel and frame:
    #     zdFF(t,x,y) = s2_norm(t,x,y) − beta(x,y) · r(t,x,y)
    # where s2_norm is STEP 3's normalisation and r is STEP 5's regressor
    # (``reg_trace``: (T,) for 'global', (n_blocks, T) for 'per_sector', or
    # None for the per-pixel modes, which instead normalise a per-voxel
    # control trace here — the pixel's own s1 for 'per_pixel', its Gaussian
    # neighbourhood average s̃1 for 'per_pixel_spatial_avg').
    # Everything below runs in-place on the float32 batch copies.
    center_s1 = norm.center_s1
    center_s2 = norm.center_s2
    inv_scale_s1 = norm.inv_scale_s1
    inv_scale_s2 = norm.inv_scale_s2
    beta = norm.beta
    reg_trace = norm.reg_trace

    if verbose:
        _what = 'aggregating' if return_aggregates else 'writing'
        print(f'\t\tpass 2/2: {_what} zdFF ({n_batches} batches)...')
    for bi, t_start, t_end, s1_b, s2_b in _batched_f32(s1, s2, T, batch_size):
        if verbose:
            print(f'\t\t\tbatch {bi+1}/{n_batches} '
                  f'({(bi+1)/n_batches*100:.1f}%)...', end='\r')
        # 6a. per-frame baseline: per-sector (broadcast to pixels) or a
        # single whole-frame scalar per frame.
        if use_sector_baseline:
            baseline_s1_b = \
                sector_baseline_s1[:, t_start:t_end].T[:, f0.block_of_pixel]
            baseline_s2_b = \
                sector_baseline_s2[:, t_start:t_end].T[:, f0.block_of_pixel]
        else:
            baseline_s1_b = baseline_s1_f32[t_start:t_end][:, None, None]
            baseline_s2_b = baseline_s2_f32[t_start:t_end][:, None, None]
        # 6b. normalise the signal channel: s_norm = ((s−base)−center)·inv.
        s2_b -= baseline_s2_b
        s2_b -= center_s2
        s2_b *= inv_scale_s2
        # 6c. zdFF = s2_norm − beta·r: broadcast beta and r from their own
        # granularity (frame / sector / pixel — see STEP 5) out to every
        # pixel, then subtract.
        if reg_trace is None:
            # per-pixel modes: r is a per-voxel control trace, normalised
            # here directly (never precomputed — already per-voxel, so
            # nothing to broadcast). 'per_pixel' uses the pixel's own s1;
            # 'per_pixel_spatial_avg' first replaces the batch by its
            # Gaussian neighbourhood average s̃1 (mode='reflect' normalised
            # convolution). center_s1 / inv_scale_s1 already describe
            # whichever of s1 / s̃1 this mode fed into the pass-1 moments,
            # so the same normalisation applies to both.
            if spatial_avg:
                s1_b = gaussian_filter(s1_b, sigma=_sig_xyz, mode='reflect')
            s1_b -= baseline_s1_b
            s1_b -= center_s1
            s1_b *= inv_scale_s1
            s1_b *= beta
            s2_b -= s1_b
        elif reg_trace.ndim == 1:
            # 'global': one normalised frame-mean trace for every pixel.
            s2_b -= beta * reg_trace[t_start:t_end][:, None, None]
        else:
            # 'per_sector': each pixel takes its own sector's mean trace.
            s2_b -= beta * reg_trace[:, t_start:t_end].T[:, f0.block_of_pixel]
        # 6d. clamp into float16 range before cast / aggregation.
        np.clip(s2_b, -32000.0, 32000.0, out=s2_b)
        if return_aggregates:
            agg.frame_mean[t_start:t_end] = s2_b.mean(
                axis=(1, 2), dtype=np.float32)
            agg.sector_mean[:, t_start:t_end] = _sector_means(
                s2_b, agg.n_sec, agg.block_x, agg.block_y,
                agg.x_trim, agg.y_trim)
        else:
            corrected[t_start:t_end] = _clip_to_dtype(s2_b, dtype)
    if verbose:
        print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')

    if return_aggregates:
        return ({'frame_f': agg.frame_mean, 'sector_f': agg.sector_mean,
                 'n_sectors': agg.n_sec, 'is_aggregates': True}, coefficients)
    if output_path is not None:
        corrected.flush()
        if verbose:
            print('\t\tflushed output to disk')
    return corrected, coefficients


def xval_control_residual(s1, s2, n_folds=2, reg_level='frame', n_sectors=8,
                          beta_global=None, verbose=True):
    """Cross-validated residual coupling to the control channel.

    A correction-quality metric for REAL recordings, where the simulation
    metrics (``recovery`` / ``leakage``, which need the true signal and
    the true artefact) do not exist. It answers the one question that is
    answerable without ground truth: **after correction, how much of the
    control channel's time course is still present in the sensor?**

    The correct answer is zero. A *negative* residual means the
    correction over-subtracted — it removed more than was there and
    inverted the artefact — which is exactly the failure mode of fitting
    beta on a spatial mean and applying it per pixel (see
    ``correct_full_regress``'s ``fit_mode='per_pixel_clean_reg'``).

    Why the scoring must be held out
    --------------------------------
    An OLS residual is orthogonal to its own regressor **by
    construction**, so an in-sample score of a per-pixel fit is
    identically zero no matter how badly the model generalises, and
    therefore says nothing. Beta is fit on one set of frames and scored
    on the others, which makes the number meaningful for per-pixel and
    fixed-beta correctors alike.

    Folds are interleaved (``t % n_folds``), not contiguous blocks, so
    slow drift and bleaching are shared evenly between fit and score
    rather than being confounded with the split.

    Parameters
    ----------
    s1, s2 : (T, X, Y) ndarray or memmap
        Control and signal channels, raw units.
    n_folds : int
        Number of cross-validation folds. Default 2 (split-half).
    reg_level : {'frame', 'sector'}
        Spatial scale of the control regressor, matching
        ``correct_full_regress``'s ``clean_reg_level``.
    n_sectors : int
        Blocks per axis when ``reg_level='sector'``. Default 8.
    beta_global : float or None
        If given, ALSO score a fixed-beta corrector using this
        coefficient, for direct comparison against the per-pixel fit
        (pass the beta that ``fit_mode='global'`` reported). None skips
        it.
    verbose : bool
        Print a summary.

    Returns
    -------
    out : SimpleNamespace
        ``.resid_pix`` (X, Y) held-out residual correlation with the
        regressor, averaged over folds; ``.resid_mean`` its spatial mean;
        ``.beta`` (X, Y) beta averaged over folds; ``.beta_stability``
        the across-fold Pearson r of the beta maps (1 fold pair only when
        ``n_folds=2``); ``.resid_pix_fixed`` / ``.resid_mean_fixed`` the
        same residual for the fixed-``beta_global`` corrector, or None;
        ``.n_eff`` the autocorrelation-corrected effective sample size.
    """
    if s1.shape != s2.shape:
        raise ValueError(f"s1 and s2 must have same shape, got "
                         f"{s1.shape} and {s2.shape}")
    if reg_level not in ('frame', 'sector'):
        raise ValueError(
            f"reg_level must be 'frame' or 'sector', got {reg_level!r}")
    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    T, X, Y = s1.shape
    _s1 = np.asarray(s1, dtype=np.float32)
    _s2 = np.asarray(s2, dtype=np.float32)

    # Build the clean regressor stack: (T,) for 'frame', or (T, X, Y)
    # broadcast from per-block means for 'sector'.
    if reg_level == 'frame':
        reg = _s1.mean(axis=(1, 2), dtype=np.float64)
        reg_xy = None
    else:
        _bx, _by = max(1, X // n_sectors), max(1, Y // n_sectors)
        _rb = np.minimum(np.arange(X) // _bx, n_sectors - 1)
        _cb = np.minimum(np.arange(Y) // _by, n_sectors - 1)
        _bop = (_rb[:, None] * n_sectors + _cb[None, :]).astype(np.intp)
        _sec = _sector_means(_s1, n_sectors, _bx, _by,
                             _bx * n_sectors, _by * n_sectors)
        reg = _sec.astype(np.float64)
        reg_xy = _bop

    def _z(arr, axis=0):
        _m = np.nanmean(arr, axis=axis, keepdims=True)
        _s = np.nanstd(arr, axis=axis, keepdims=True)
        return (arr - _m) / (_s + 1e-9)

    folds = [np.arange(T)[np.arange(T) % n_folds == k] for k in range(n_folds)]
    beta_folds, resid_folds, resid_fixed_folds = [], [], []
    for k in range(n_folds):
        fit_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        score_idx = folds[k]
        # Regressor and normalised signal on each index set.
        if reg_level == 'frame':
            r_fit = _z(reg[fit_idx])
            r_sc = _z(reg[score_idx])
            r_fit_xy = r_fit[:, None, None]
            r_sc_xy = r_sc[:, None, None]
        else:
            r_fit = _z(reg[:, fit_idx], axis=1)
            r_sc = _z(reg[:, score_idx], axis=1)
            r_fit_xy = r_fit.T[:, reg_xy]
            r_sc_xy = r_sc.T[:, reg_xy]
        y_fit = _z(_s2[fit_idx])
        y_sc = _z(_s2[score_idx])
        # Per-pixel OLS coefficient on the fit fold.
        _num = np.einsum('txy,txy->xy', y_fit,
                         np.broadcast_to(r_fit_xy, y_fit.shape),
                         dtype=np.float64)
        _den = np.einsum('txy,txy->xy',
                         np.broadcast_to(r_fit_xy, y_fit.shape),
                         np.broadcast_to(r_fit_xy, y_fit.shape),
                         dtype=np.float64)
        beta = _num / np.maximum(_den, 1e-10)
        beta_folds.append(beta)
        # Score on the held-out fold.
        resid = y_sc - beta[None] * r_sc_xy
        resid_folds.append(_corr_map_1d(resid, r_sc_xy))
        if beta_global is not None:
            resid_fixed_folds.append(
                _corr_map_1d(y_sc - np.float32(beta_global) * r_sc_xy,
                             r_sc_xy))

    out = SimpleNamespace()
    out.beta = np.mean(beta_folds, axis=0).astype(np.float32)
    out.resid_pix = np.mean(resid_folds, axis=0).astype(np.float32)
    out.resid_mean = float(np.nanmean(out.resid_pix))
    if beta_global is not None:
        out.resid_pix_fixed = np.mean(resid_fixed_folds, axis=0).astype(
            np.float32)
        out.resid_mean_fixed = float(np.nanmean(out.resid_pix_fixed))
    else:
        out.resid_pix_fixed = None
        out.resid_mean_fixed = None
    _pairs = [np.corrcoef(beta_folds[i].ravel(), beta_folds[j].ravel())[0, 1]
              for i in range(n_folds) for j in range(i + 1, n_folds)]
    out.beta_stability = float(np.nanmean(_pairs)) if _pairs else np.nan
    _ref = reg if reg_level == 'frame' else reg.mean(axis=0)
    out.n_eff = float(T / _autocorr_time(_ref))

    if verbose:
        print(f'\txval_control_residual (n_folds={n_folds}, '
              f'reg_level={reg_level}):')
        print(f'\t\theld-out resid corr w/ control: '
              f'{out.resid_mean:+.4f}  (0 = fully removed, '
              f'<0 = over-subtracted)')
        if out.resid_mean_fixed is not None:
            print(f'\t\t  same for fixed β={beta_global:.4f}: '
                  f'{out.resid_mean_fixed:+.4f}')
        print(f'\t\tβ median {float(np.median(out.beta)):.4f}, '
              f'across-fold stability r={out.beta_stability:.4f}, '
              f'N_eff={out.n_eff:.0f}')
    return out


def _corr_map_1d(stack, reg_xy):
    """Per-pixel Pearson r between a (T, X, Y) stack and a regressor.

    ``reg_xy`` broadcasts against ``stack`` — a (T, 1, 1) whole-frame
    trace or a full (T, X, Y) per-block one.
    """
    _r = np.broadcast_to(reg_xy, stack.shape)
    _sc = stack - np.nanmean(stack, axis=0, keepdims=True)
    _rc = _r - np.nanmean(_r, axis=0, keepdims=True)
    _num = np.einsum('txy,txy->xy', _sc, _rc, dtype=np.float64)
    _den = np.sqrt(
        np.einsum('txy,txy->xy', _sc, _sc, dtype=np.float64)
        * np.einsum('txy,txy->xy', _rc, _rc, dtype=np.float64))
    return _num / np.maximum(_den, 1e-12)


def correct_two_stage_regress(
        s1, s2, batch_size=1000, dtype=np.float16, verbose=True,
        output_path=None, eps=1e-6,
        smooth_window=10,
        airpls_lam=1e5, airpls_porder=2, airpls_max_iter=50,
        trim_initial=0, nn_slope=False,
        aggregates_only=None,
        sector_levels=(2, 4, 8, 16),
        f0_level='sector',
        orthogonalise=True,
        min_block_var=1e-2, min_block_corr=0.15,
        beta_loss='huber', beta_f_scale=1.0, beta_scale='mad'):
    """Two-stage hierarchical full-regression correction.

    Decomposes the control channel into spatial scales and regresses the
    signal against a sequence of mutually-orthogonal, high-SNR spatially-
    averaged regressors. Unlike ``correct_full_regress``'s per-pixel /
    per-sector modes — which subtract β·s1_norm(t,x,y), injecting the
    single-pixel control trace's own shot noise — every regressor here is
    a spatial average (whole-frame or sector mean), so the correction
    carries no single-pixel control noise.

    Stage 1 — whole frame:
        g(t) = whole-frame mean control (averaged over ~all pixels, very
        clean). Fit one global scalar β₁ and remove β₁·g(t) everywhere.

    Stage 2 — multi-scale cascade (``sector_levels``, coarse → fine):
        At each level l and block b, take the sector-mean control trace,
        orthogonalise it (Gram-Schmidt) against g and every coarser
        ancestor-block residual to get r_{l,b}(t), fit a per-block slope
        β_{l,b} on r_{l,b}, and remove β_{l,b}·r_{l,b}(t). Because every
        regressor is orthogonal to all coarser ones, the stages do not
        fight (sequential fit = joint fit).

    Final per voxel::

        zdFF(t,x,y) = s2_norm(t,x,y) − β₁·g(t)
                      − Σ_l β_{l,block_l(x,y)} · r_{l,block_l(x,y)}(t)

    where ``s2_norm = ((s2 − baseline(t)) − center_xy)·inv_scale_xy`` (the
    same normalisation as ``correct_full_regress``), and g(t) and every
    r_{l,b}(t) are precomputed length-T 1-D traces — so Pass 2 streams
    only s2 and the per-voxel work is a handful of broadcast subtractions.

    Parameters
    ----------
    s1, s2 : (T, X, Y) ndarray or memmap
        Control / isosbestic-like reference (s1) and the real signal to
        correct (s2).
    batch_size : int
        Frames per streaming batch. Default 1000.
    dtype : np.dtype
        Output dtype. Default np.float16.
    verbose : bool
        Print progress and fitted coefficients.
    output_path : str or None
        If given, stream the corrected stack to a memmapped TIFF instead
        of a RAM buffer. Mutually exclusive with ``aggregates_only``.
    eps : float
        Floor protecting per-pixel divides.
    smooth_window : int
        Moving-average window before airPLS (1 disables).
    airpls_lam, airpls_porder, airpls_max_iter : float, int, int
        airPLS baseline parameters (see ``correct_full_regress``).
    trim_initial : int
        Leading frames dropped from the fits / projections only; the full
        T-frame output is still written.
    nn_slope : bool
        Enforce β ≥ 0 for every fitted slope.
    aggregates_only : dict or None
        If e.g. ``{'n_sectors': 8}``, skip the (T, X, Y) buffer and stream-
        compute the whole-frame + per-sector mean of zdFF (mirrors
        ``correct_full_regress``); returns those aggregates instead of the
        full stack.
    sector_levels : tuple of int
        Stage-2 nested sector grids, coarse → fine (blocks per axis). Must
        be a strictly increasing divisor chain (each divides the next) so
        block nesting is exact. The finest level doubles as the F0 sector
        grid when ``f0_level='sector'``. Empty disables Stage 2 (global-
        only, ≈ ``correct_full_regress(fit_mode='global')``).
    f0_level : {'sector', 'pixel', 'frame'}
        Granularity of the per-voxel s2 normalisation (center, scale) and
        the airPLS baseline. 'sector' uses the finest cascade grid.
    orthogonalise : bool
        Gram-Schmidt each level against all coarser levels (default True).
        If False, each level's regressor is only orthogonalised against g
        (debug / ablation).
    min_block_var : float
        Reliability gate: blocks whose orthogonal-residual variance over
        the fit window is below this get β=0 (dark / fully explained by
        coarser levels).
    min_block_corr : float
        Coupling-significance gate: a block's β is set to 0 unless its
        orthogonal control regressor correlates with that block's signal
        trace above ``max(min_block_corr, 3/sqrt(N_fit))``. Without this,
        a control sector carrying only shot noise (no shared
        haemodynamics) is normalised up to unit variance and subtracted as
        noise, degrading the output below the no-correction baseline.
        Default 0.15; set to 0 to disable (falls back to the 3/sqrt(N)
        noise floor).
    beta_loss : {'huber', 'cauchy', 'soft_l1', 'arctan', 'linear'}
        Loss for every 1-D coupling-slope fit (see ``correct_full_regress``).
    beta_f_scale : float
        Soft threshold of the robust loss. Only used when
        ``beta_loss != 'linear'``.
    beta_scale : {'mad', 'std'}
        Scale estimator for the trace normalisation preceding each fit.

    Returns
    -------
    corrected : (T, X, Y) ndarray / memmap, or dict
        Per-pixel zdFF. With ``aggregates_only`` set, instead returns
        ``{'frame_f': (T,), 'sector_f': (n_sec², T), 'n_sectors': int,
        'is_aggregates': True}``.
    diagnostics : SimpleNamespace
        ``.beta1`` (scalar global slope), ``.sector_levels`` (tuple),
        ``.beta`` (list of per-level (n_sec_l²,) slope arrays), and
        ``.center_s2`` (per-pixel intercept diagnostic).
    """
    if s1.shape != s2.shape:
        raise ValueError(
            f"s1 and s2 must have same shape, "
            f"got {s1.shape} and {s2.shape}")

    T, X, Y = s1.shape

    # Validate the cascade: strictly increasing divisor chain so each
    # finer grid nests exactly inside the coarser ones.
    levels = [int(n) for n in sector_levels]
    if any(n < 1 for n in levels):
        raise ValueError(
            f"sector_levels must be >= 1, got {sector_levels}")
    for _a, _b in zip(levels, levels[1:]):
        if _b <= _a or _b % _a != 0:
            raise ValueError(
                "sector_levels must be a strictly increasing divisor "
                f"chain (each divides the next), got {sector_levels}")
    n_levels = len(levels)
    finest = levels[-1] if levels else 0

    s1, s2, batch_size = _prepare_inputs(
        s1, s2, batch_size, T, X, Y, verbose)
    n_batches = (T + batch_size - 1) // batch_size

    # ------------------------------------------------------------------
    # Output mode & QC aggregate grid (mirrors correct_full_regress).
    # ------------------------------------------------------------------
    return_aggregates = aggregates_only is not None
    if return_aggregates and output_path is not None:
        raise ValueError(
            "aggregates_only and output_path are mutually exclusive")
    if return_aggregates:
        agg = SimpleNamespace()
        agg.n_sec = int(aggregates_only.get('n_sectors', 8))
        agg.x_trim = (X // agg.n_sec) * agg.n_sec
        agg.y_trim = (Y // agg.n_sec) * agg.n_sec
        agg.block_x = agg.x_trim // agg.n_sec
        agg.block_y = agg.y_trim // agg.n_sec
        agg.frame_mean = np.zeros(T, dtype=np.float32)
        agg.sector_mean = np.zeros((agg.n_sec * agg.n_sec, T),
                                   dtype=np.float32)

    if verbose:
        _disk = " -> disk" if output_path else ""
        _agg = f' [aggregates n_sec={agg.n_sec}]' if return_aggregates else ''
        print(f'\tcorrecting signal (two_stage, '
              f'levels={tuple(levels)}{_disk}){_agg}...')

    if return_aggregates:
        corrected = None
    elif output_path is not None:
        if verbose:
            print(f'\t\twriting output to: {output_path}')
        corrected = tifffile.memmap(
            output_path, shape=(T, X, Y), dtype=dtype, bigtiff=True)
    else:
        corrected = np.zeros((T, X, Y), dtype=dtype)

    # ------------------------------------------------------------------
    # Per-level sector grids (same convention as the F0 grid and
    # _pool_moments: block = row_block * n_sec + col_block, row from X).
    # ------------------------------------------------------------------
    grids = []
    for _ns in levels:
        _g = SimpleNamespace()
        _g.n_sec = _ns
        _g.n_blocks = _ns * _ns
        _g.block_x = max(1, X // _ns)
        _g.block_y = max(1, Y // _ns)
        _g.x_trim = _g.block_x * _ns
        _g.y_trim = _g.block_y * _ns
        _row = np.minimum(np.arange(X) // _g.block_x, _ns - 1)
        _col = np.minimum(np.arange(Y) // _g.block_y, _ns - 1)
        _g.block_of_pixel = (_row[:, None] * _ns
                             + _col[None, :]).astype(np.intp)
        _g.sector_trace_s1 = np.zeros((_g.n_blocks, T), dtype=np.float64)
        _g.sector_trace_s2 = np.zeros((_g.n_blocks, T), dtype=np.float64)
        grids.append(_g)

    # F0 grid (per-voxel normalisation): the finest cascade level, or a
    # default 8 when Stage 2 is disabled. A per-sector airPLS baseline is
    # only available when there is at least one cascade level (it reuses
    # the finest level's s2 baselines); with no levels we fall back to the
    # whole-frame baseline.
    n_sec_f0 = finest if finest >= 1 else 8
    use_sector_baseline = (f0_level == 'sector' and n_sec_f0 >= 1
                           and n_levels >= 1)

    # ------------------------------------------------------------------
    # Pass 1 — global mean traces, per-level sector means, s2 moments.
    # Only s2 per-pixel moments are needed (Pass 2 never normalises s1).
    # ------------------------------------------------------------------
    if verbose:
        print(f'\t\tpass 1/2: spatial means & per-sector traces '
              f'({n_batches} batches)...')
    mean_trace_s1 = np.zeros(T, dtype=np.float64)
    mean_trace_s2 = np.zeros(T, dtype=np.float64)
    pix_sum_s2 = np.zeros((X, Y), dtype=np.float64)
    pix_sum_s2sq = np.zeros((X, Y), dtype=np.float64)

    for bi, t_start, t_end, s1_b, s2_b in _batched_f32(s1, s2, T, batch_size):
        if verbose:
            print(f'\t\t\tbatch {bi+1}/{n_batches} '
                  f'({(bi+1)/n_batches*100:.1f}%)...', end='\r')
        mean_trace_s1[t_start:t_end] = s1_b.mean(axis=(1, 2), dtype=np.float64)
        mean_trace_s2[t_start:t_end] = s2_b.mean(axis=(1, 2), dtype=np.float64)
        n_frames_b = s1_b.shape[0]
        for _g in grids:
            _g.sector_trace_s1[:, t_start:t_end] = \
                s1_b[:, :_g.x_trim, :_g.y_trim].reshape(
                    n_frames_b, _g.n_sec, _g.block_x, _g.n_sec, _g.block_y
                ).mean(axis=(2, 4)).reshape(n_frames_b, -1).T
            _g.sector_trace_s2[:, t_start:t_end] = \
                s2_b[:, :_g.x_trim, :_g.y_trim].reshape(
                    n_frames_b, _g.n_sec, _g.block_x, _g.n_sec, _g.block_y
                ).mean(axis=(2, 4)).reshape(n_frames_b, -1).T
        pix_sum_s2 += s2_b.sum(axis=0, dtype=np.float64)
        pix_sum_s2sq += np.einsum('ijk,ijk->jk', s2_b, s2_b, dtype=np.float64)
    if verbose:
        print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')

    px = SimpleNamespace()
    px.mean_s2 = (pix_sum_s2 / T).astype(np.float32)
    px.var_s2 = np.maximum(
        pix_sum_s2sq / T - px.mean_s2.astype(np.float64) ** 2,
        1e-10).astype(np.float32)
    del pix_sum_s2, pix_sum_s2sq

    # ------------------------------------------------------------------
    # Phase A — build the 1-D regressors and fit every slope.
    # ------------------------------------------------------------------
    n_warmup = int(max(0, min(trim_initial, T - 2)))
    fit = slice(n_warmup, None)

    def _normalise_trace(raw):
        """smooth -> airPLS -> debase -> median/scale (fit-window stats).

        Returns the full length-T normalised trace; its [fit] slice is the
        median-centred, scale-normalised trace used for the slope fits.
        """
        _smooth = _moving_average(raw, smooth_window)
        _base = _airpls(_smooth, lam=airpls_lam, porder=airpls_porder,
                        max_iter=airpls_max_iter)
        _deb = _smooth - _base
        _med = float(np.median(_deb[fit]))
        _sd = _norm_scale(_deb[fit], beta_scale)
        return (_deb - _med) / _sd, _base

    # Stage-1 global regressor and slope.
    if verbose:
        print('\t\trunning airPLS baselines on s̄1(t), s̄2(t)...')
    g_full, _ = _normalise_trace(mean_trace_s1)
    s2bar_full, baseline_s2 = _normalise_trace(mean_trace_s2)
    beta1 = _fit_norm_slope(
        g_full[fit], s2bar_full[fit], nn_slope=nn_slope,
        loss=beta_loss, f_scale=beta_f_scale)
    gg = max(float(np.dot(g_full[fit], g_full[fit])), 1e-10)
    if verbose:
        print(f'\t\t\tβ₁(global, norm) = {beta1:.6g} '
              f'(loss={beta_loss}, nn={"on" if nn_slope else "off"})')

    # Stage-2 cascade: per-level per-sector airPLS, orthogonalisation, fit.
    r_levels = []        # list of (n_blocks_l, T) orthogonal regressors
    rr_levels = []       # list of (n_blocks_l,) fit-window <r, r>
    beta_levels = []     # list of (n_blocks_l,) per-block slopes
    sector_baseline_s2_fin = None
    for li, _g in enumerate(grids):
        if verbose:
            print(f'\t\tlevel {li+1}/{n_levels} (n_sec={_g.n_sec}): '
                  f'per-sector airPLS + orthogonalise '
                  f'({_g.n_blocks} blocks)...')
        # Normalise each block's control and signal sector trace. airPLS is
        # the cost; run the blocks in parallel threads.
        _norm_s1 = Parallel(n_jobs=-1, prefer='threads')(
            delayed(_normalise_trace)(_g.sector_trace_s1[_s])
            for _s in range(_g.n_blocks))
        _norm_s2 = Parallel(n_jobs=-1, prefer='threads')(
            delayed(_normalise_trace)(_g.sector_trace_s2[_s])
            for _s in range(_g.n_blocks))
        cb_full = np.asarray([_t for _t, _ in _norm_s1], dtype=np.float64)
        s2sec_full = np.asarray([_t for _t, _ in _norm_s2], dtype=np.float64)
        if li == n_levels - 1:
            # Finest level's s2 baselines double as the F0 per-sector
            # baseline for the per-voxel normalisation in Pass 2.
            sector_baseline_s2_fin = np.asarray(
                [_b for _, _b in _norm_s2], dtype=np.float32)

        r_l = np.empty((_g.n_blocks, T), dtype=np.float64)
        rr_l = np.empty(_g.n_blocks, dtype=np.float64)
        beta_l = np.zeros(_g.n_blocks, dtype=np.float64)
        for b in range(_g.n_blocks):
            cb = cb_full[b]
            r = cb.copy()
            # Orthogonalise against the clean global regressor.
            r -= (float(np.dot(cb[fit], g_full[fit])) / gg) * g_full
            # ...and against every coarser ancestor-block residual. The
            # chain {g, r_{k,anc}} is mutually orthogonal by induction, so
            # each projection uses the original cb and they simply sum.
            if orthogonalise:
                _row = b // _g.n_sec
                _col = b % _g.n_sec
                _prow = min(_row * _g.block_x + _g.block_x // 2, X - 1)
                _pcol = min(_col * _g.block_y + _g.block_y // 2, Y - 1)
                for lk in range(li):
                    _anc = int(grids[lk].block_of_pixel[_prow, _pcol])
                    _rk = r_levels[lk][_anc]
                    _rrk = rr_levels[lk][_anc]
                    r -= (float(np.dot(cb[fit], _rk[fit])) / _rrk) * _rk
            r_l[b] = r
            rr_l[b] = max(float(np.dot(r[fit], r[fit])), 1e-10)
            # Reliability gate: near-zero orthogonal residual => no fit.
            _n_fit = T - n_warmup
            if rr_l[b] / max(_n_fit, 1) < min_block_var:
                beta_l[b] = 0.0
                continue
            # Coupling-significance gate. Because each regressor is
            # normalised to unit scale, a control sector carrying only
            # shot noise (no shared haemodynamics) still reads as a
            # full-amplitude regressor; fitting and subtracting it just
            # injects noise. Only subtract where the orthogonal control
            # regressor genuinely correlates with this block's signal
            # trace (|corr| above the ~1/sqrt(N) noise floor).
            _yb = s2sec_full[b][fit]
            _yy = max(float(np.dot(_yb, _yb)), 1e-10)
            _corr = float(np.dot(r[fit], _yb)) / np.sqrt(rr_l[b] * _yy)
            _corr_gate = max(min_block_corr, 3.0 / np.sqrt(max(_n_fit, 1)))
            if abs(_corr) < _corr_gate:
                beta_l[b] = 0.0
                continue
            beta_l[b] = _fit_norm_slope(
                r[fit], _yb, nn_slope=nn_slope,
                loss=beta_loss, f_scale=beta_f_scale)
        r_levels.append(r_l)
        rr_levels.append(rr_l)
        beta_levels.append(beta_l)
        if verbose:
            _nz = int(np.count_nonzero(beta_l))
            print(f'\t\t\tβ median {float(np.median(beta_l)):.4f}, '
                  f'range [{float(beta_l.min()):.4f}, '
                  f'{float(beta_l.max()):.4f}], '
                  f'{_nz}/{_g.n_blocks} active')
        # Sector traces no longer needed once normalised.
        del _g.sector_trace_s1, _g.sector_trace_s2

    # Pre-scale each level's regressor by its per-block β (fold β in once)
    # and cast to float32 for Pass 2.
    R_levels = [(beta_levels[li][:, None] * r_levels[li]).astype(np.float32)
                for li in range(n_levels)]
    del r_levels
    g_full_f32 = g_full.astype(np.float32)

    # ------------------------------------------------------------------
    # Per-voxel s2 normalisation constants (mirrors correct_full_regress).
    # ------------------------------------------------------------------
    mean_baseline_s2 = np.float32(np.mean(baseline_s2))
    if use_sector_baseline:
        bop_fin = grids[-1].block_of_pixel
        mean_baseline_s2_xy = sector_baseline_s2_fin.mean(
            axis=1)[bop_fin].astype(np.float32)
    else:
        mean_baseline_s2_xy = float(mean_baseline_s2)
    pooled_mean_s2, pooled_var_s2 = _pool_moments(
        px.mean_s2, px.var_s2, f0_level, n_sec_f0)
    center_s2 = (pooled_mean_s2 - mean_baseline_s2_xy).astype(np.float32)
    scale_s2_xy = np.sqrt(pooled_var_s2).astype(np.float32)
    scale_s2_med = float(np.median(scale_s2_xy))
    scale_s2_floor = np.float32(max(scale_s2_med * 0.01, eps))
    scale_s2_xy = np.maximum(scale_s2_xy, scale_s2_floor)
    inv_scale_s2 = (np.float32(1.0) / scale_s2_xy).astype(np.float32)
    baseline_s2_f32 = baseline_s2.astype(np.float32)
    if verbose:
        _bl = 'per-sector' if use_sector_baseline else 'whole-frame'
        print(f'\t\tF0 granularity: {f0_level}'
              f'{f" (n_sec={n_sec_f0})" if f0_level == "sector" else ""}'
              f'; airPLS baseline: {_bl}')

    diagnostics = SimpleNamespace()
    diagnostics.beta1 = float(beta1)
    diagnostics.sector_levels = tuple(levels)
    diagnostics.beta = [b.copy() for b in beta_levels]
    diagnostics.center_s2 = center_s2

    # ------------------------------------------------------------------
    # Pass 2 — stream s2 only; subtract the precomputed 1-D regressors.
    # ------------------------------------------------------------------
    if verbose:
        _what = 'aggregating' if return_aggregates else 'writing'
        print(f'\t\tpass 2/2: {_what} zdFF ({n_batches} batches)...')
    bop_levels = [g.block_of_pixel for g in grids]
    for bi, t_start, t_end, s2_b in _batched_f32_one(s2, T, batch_size):
        if verbose:
            print(f'\t\t\tbatch {bi+1}/{n_batches} '
                  f'({(bi+1)/n_batches*100:.1f}%)...', end='\r')
        # Normalise s2: s2_norm = ((s2 − baseline) − center) · inv_scale.
        if use_sector_baseline:
            baseline_s2_b = \
                sector_baseline_s2_fin[:, t_start:t_end].T[:, bop_fin]
        else:
            baseline_s2_b = baseline_s2_f32[t_start:t_end][:, None, None]
        s2_b -= baseline_s2_b
        s2_b -= center_s2
        s2_b *= inv_scale_s2
        # Stage 1: subtract β₁·g(t) (broadcast over space).
        s2_b -= np.float32(beta1) * g_full_f32[t_start:t_end][:, None, None]
        # Stage 2: subtract each level's β-scaled residual regressor.
        for li in range(n_levels):
            s2_b -= R_levels[li][:, t_start:t_end].T[:, bop_levels[li]]
        np.clip(s2_b, -32000.0, 32000.0, out=s2_b)
        if return_aggregates:
            agg.frame_mean[t_start:t_end] = s2_b.mean(
                axis=(1, 2), dtype=np.float32)
            n_frames_b = s2_b.shape[0]
            sector_block = s2_b[:, :agg.x_trim, :agg.y_trim].reshape(
                n_frames_b, agg.n_sec, agg.block_x,
                agg.n_sec, agg.block_y).mean(axis=(2, 4))
            agg.sector_mean[:, t_start:t_end] = \
                sector_block.reshape(n_frames_b, -1).T
        else:
            corrected[t_start:t_end] = _clip_to_dtype(s2_b, dtype)
    if verbose:
        print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')

    if return_aggregates:
        return ({'frame_f': agg.frame_mean, 'sector_f': agg.sector_mean,
                 'n_sectors': agg.n_sec, 'is_aggregates': True}, diagnostics)
    if output_path is not None:
        corrected.flush()
        if verbose:
            print('\t\tflushed output to disk')
    return corrected, diagnostics


# =============================================================================
# Pixel-wise dF/F correction by a Gaussian spatial average of control pixels
# =============================================================================

def correct_pixel_spatial_subtr(
        s1, s2, batch_size=1000, dtype=np.float16, verbose=True,
        output_path=None, eps=1e-6,
        remove_baseline=False, smooth_window=10,
        airpls_lam=1e5, airpls_porder=2, airpls_max_iter=50,
        trim_initial=0,
        pixel_gate='gmm', gmm_log=True, gmm_smooth_px='auto',
        gmm_min_sep=1.5, gate_pct=50.0,
        segmentation_type='gmm', soma_min_darkness=0.10,
        soma_diam_px=(4.0, 20.0), soma_bg_sigma=20.0,
        ring_width_px=3, neuropil_floor_pct=None,
        sigma_px=4.0, den_min=0.25,
        f0_mode='global', n_frames_f0=100,
        trial_onsets=None, trial_baseline_frames=30, trial_window=None,
        stim_step_dff=False, stim_on_frames=None, stim_off_frames=None,
        stim_step_edge='both', stim_step_n_edge=3, stim_step_n_gap=1,
        batch_size_cap=2000, return_components=False,
        aggregate_sectors=None):
    """Per-pixel dF/F correction against a Gaussian spatial average of s1.

    Unlike the ``correct_full_regress`` family, no coupling slope β is
    fitted: the control's dF/F is subtracted with unit gain. This is the
    principled form for haemodynamics — absorption is multiplicative, so
    it produces the *same fractional* change in both channels.

    Pipeline:
        1. (Optional, ``remove_baseline``) whole-frame moving-average
           lowpass + airPLS baseline per channel; only the *fluctuating*
           part b(t) − mean_t(b) is subtracted, so the DC level survives
           and F0 stays meaningful.
        2. Per-pixel expression gate: a 2-component Gaussian mixture on
           the (log, spatially smoothed) session mean image; only the
           higher-intensity component is kept.
        3. dF/F per high-SNR s2 pixel, F0 from ``n_frames_f0`` frames
           (or, with ``f0_mode='per_trial'``, from the frames
           immediately before each trial).
        4. dF/F of a Gaussian spatial average of the surrounding
           *high-SNR s1* pixels (a normalised convolution — only
           unmasked control pixels contribute).
        5. ``dff_corr = dff_sig − dff_ctrl``.

    Pixels that fail the gate are NaN, so downstream reductions must be
    nan-aware (``np.nanmean``).

    Parameters
    ----------
    s1, s2 : (T, X, Y) ndarray or memmap
        Control / isosbestic-like reference (s1, e.g. green) and the
        real signal to correct (s2, e.g. red).
    batch_size : int
        Frames per streaming batch. Default 1000.
    dtype : np.dtype
        Output dtype. Must be a float type — invalid pixels are marked
        with NaN, which integer dtypes cannot represent. Default
        np.float16 (dF/F is O(0.01–1); float16 resolves ~5e-4 relative).
    verbose : bool
        Print progress, SNR summary and mask occupancy.
    output_path : str or None
        If given, stream the corrected stack to a memmapped TIFF instead
        of a RAM buffer. Mutually exclusive with ``return_components``.
    eps : float
        Floor protecting the normalised-convolution divide.
    remove_baseline : bool
        Fit and subtract a whole-frame airPLS baseline per channel
        before the dF/F (step 1 above). Default False: subtracting the
        control's dF/F already removes common-mode drift, since a
        multiplicative artefact (bleaching, absorption) produces the
        same fractional change in both channels and cancels in
        ``dff_sig − dff_ctrl``. Running airPLS on top is then at best
        redundant and at worst harmful — it can eat real slow signal,
        and it removes drift additively (see Notes). It matters least
        with ``f0_mode='per_trial'``, where each trial's own F0 already
        absorbs slow drift. Turn it on when the two channels drift
        *differently* (e.g. unequal bleaching rates), which the unit-gain
        subtraction cannot cancel. ``smooth_window`` and the ``airpls_*``
        params are ignored when this is False.
    smooth_window : int
        Moving-average window before airPLS (1 disables). Only used when
        ``remove_baseline=True``.
    airpls_lam, airpls_porder, airpls_max_iter : float, int, int
        airPLS baseline: smoothness penalty, difference order, IRLS
        iteration cap. Same meaning as in ``correct_full_regress``.
        Only used when ``remove_baseline=True``.
    trim_initial : int
        Leading warm-up frames excluded from the SNR estimate and from
        the global F0 window. The full T-frame output is still written.
    pixel_gate : {'gmm', 'percentile', 'none'}
        How expressing pixels are selected from each channel's session
        mean image. 'gmm' (default) fits a 2-component Gaussian mixture
        and keeps the higher-intensity component — the threshold is
        data-adaptive, so nothing needs recalibrating per rig.
        'percentile' cuts at ``gate_pct``. 'none' keeps every pixel
        (still subject to the ``den`` and F0 terms of ``mask_out``).

        Both masks are gated the same way, but they do different jobs:
        the s1 (control) mask selects which pixels *contribute* to the
        spatial average, while the s2 (signal) mask decides where output
        exists at all.

        The gate selects the expressing *field* — a contiguous tissue
        mask, ~52% of the FOV on a real sensor channel — not individual
        expressing cells.
    gmm_log : bool
        Fit the mixture on log intensity. Default True, and it matters:
        2-photon intensity distributions are right-skewed, so a
        2-Gaussian fit on the raw scale is misspecified. Measured on a
        real sensor channel, raw separates the modes by only 1.24 σ
        (below ``gmm_min_sep``, so the gate would silently fall back to
        a percentile every time) against 2.31 σ on log.
    gmm_smooth_px : float or 'auto'
        Gaussian sigma (pixels) applied to the mean image before the
        fit. Per-pixel shot noise otherwise scatters pixels across the
        cut and speckles the mask. 'auto' (default) uses 0.5% of the
        FOV's smaller axis, rounded down. A *fixed* sigma is silently
        FOV-dependent — 2 px on a 512x512 recording blurs the same
        fraction of frame as 16 px on a 64x64 sim, which erases
        cell-sized features and destroys the very bimodality the gate
        needs. 'auto' resolves to 2 px at 512x512 (the value this gate
        was calibrated on) and 0 px at 64x64, where the frame-averaged
        mean image is already noise-free. 0 disables.
        The resolved value is echoed in ``info.gmm_smooth_px``.
    gmm_min_sep : float
        Minimum mode separation |μ_hi − μ_lo| / max(σ) for the GMM cut to
        be trusted. Below this — or if a 1-component fit has lower BIC —
        the data is not really bimodal, so the fit is discarded, a
        RuntimeWarning is raised, and ``gate_pct`` is used instead. A
        2-component fit always returns *something*; this is what stops a
        smooth continuum being split at an arbitrary point and reported
        with false confidence.
    gate_pct : float
        Percentile cut for pixel_gate='percentile', and the fallback when
        the GMM guard fires. Default 50.
    segmentation_type : {'gmm', 'cells'}
        How the *signal* (s2) mask is built. 'gmm' (default) uses the
        ``pixel_gate`` expression gate above — nothing changes. 'cells'
        is cell-aware: it detects cell bodies as negative-contrast dark
        holes (membrane-bound sensors like GRAB express in the neuropil,
        not in somata, so cell bodies read *darker* than their
        surround). It then builds ``mask_sig`` as the ``pixel_gate``
        expression gate (the same one 'gmm' uses) with the detected
        somata *excluded* — i.e. 'cells' is a strict refinement of 'gmm',
        the same expressing pixels minus the soma interiors, not "all
        non-soma pixels". This keeps dim / background pixels (whose
        near-zero F0 would otherwise blow up dF/F and drag the corrected
        whole-frame trace off zero) out of ``mask_out``. It additionally
        emits per-cell donut-ring dF/F traces (see ``info.cell_*``
        below). The *control* (s1) mask always uses the expression gate,
        regardless.

        Caveat measured on real GRAB data: the ring just outside a soma
        is ordinary neuropil (same brightness as bulk neuropil), not a
        distinct brighter membrane compartment. The donut *attributes*
        signal to a cell; it does not isolate a separate optical signal.
    soma_min_darkness : float
        Cell mode. Minimum relative dip ``(bg - F) / bg`` below the local
        background for a pixel to seed a soma. Default 0.10.
    soma_diam_px : (float, float)
        Cell mode. Keep detections whose area-equivalent diameter
        ``2·sqrt(area/pi)`` (pixels) falls in this band. Default (4, 20).
    soma_bg_sigma : float
        Cell mode. Gaussian sigma (px) for the large-scale local
        background against which the dip is measured. Default 20.
    ring_width_px : int
        Cell mode. Width (px) of the donut annulus grown outward from
        each soma for its per-cell trace. Default 3.
    neuropil_floor_pct : float or None
        Cell mode. If set, also require ``mask_sig`` pixels to sit above
        this percentile of the signal mean image — a permissive tissue
        floor. Default None (keep all non-soma pixels; ``den``/F0 terms
        of ``mask_out`` still trim true background).
    sigma_px : float
        Gaussian sigma of the spatial average, in pixels. Default 2.0
        (the centre pixel then carries only ≈ 4% of the kernel weight,
        so this really is a neighbourhood average). The centre pixel is
        included — s1 and s2 are different channels, so there is no
        self-contamination.
    den_min : float
        Minimum fraction of Gaussian weight that must come from
        high-SNR control pixels for a pixel's spatial average to be
        trusted. The normalising denominator lies in [0, 1], so this is
        directly interpretable. Default 0.25.
    f0_mode : {'global', 'per_trial'}
        'global' (default): one F0 per pixel from the ``n_frames_f0``
        frames after ``trim_initial``. 'per_trial': F0 per pixel *per
        trial*, from the ``trial_baseline_frames`` frames immediately
        before each onset; output is written only inside each trial's
        ``trial_window`` and is NaN in the inter-trial gaps (the
        ``qc.dff_trial`` convention).
    n_frames_f0 : int
        Frames used for the global F0, starting at ``trim_initial``.
    trial_onsets : (n_trials,) array-like or None
        Onset frame indices (into the T axis of ``s1``/``s2``). Required
        when f0_mode='per_trial'. On a ``TwoPRec_DualColour`` instance
        ``exp``, stim onsets are stored as Timeline-clock times in
        ``exp.beh.stim.t_start`` and the per-frame timestamps in
        ``exp.rec_t``, so convert times to frame indices with::

            trial_onsets = np.searchsorted(exp.rec_t, exp.beh.stim.t_start)

        This is correct when the full stack is corrected (the default). If
        ``correct_signal`` was given a ``t_start`` (so the stack begins at
        frame ``idx_start`` rather than 0), subtract that offset:
        ``trial_onsets - idx_start``.
    trial_baseline_frames : int
        Frames immediately before each onset used for that trial's F0.
    trial_window : (int, int) or None
        Output extent per trial as frame offsets from onset, e.g.
        ``(-30, 90)``. Required when f0_mode='per_trial' — a silently
        defaulted trial window is worse than an error.
    stim_step_dff : bool
        Subtract a per-pixel, dF/F-domain stimulus-step artefact (the
        visual-stimulus light-leak) from the *corrected* signal, over the
        stim-on frames. Default False. This is the dF/F-domain counterpart
        of the raw whole-frame ``remove_stim_step`` and is the correct tool
        for ``pixel_spatial_subtr``: because the correction subtracts
        ``dff_sig − dff_ctrl`` with unit gain, an additive light-leak (which
        takes a *different fractional* size in each channel, as F0 differs)
        is neither cancelled nor guaranteed to shrink — it is amplified, and
        often sign-flipped, in proportion to how much dimmer the control
        channel is. A single raw whole-frame scalar cannot fix this: the
        leak is spatially structured (brighter toward the monitor) and the
        mismatch lives in dF/F. Here the leak's sharp jump is estimated per
        pixel from the stim edges (``estimate_pixel_step_maps``), propagated
        through the same masked-Gaussian operator for the control channel,
        converted to dF/F with each pixel's (or trial's) F0, and the
        resulting ``step_dff_sig − step_dff_ctrl`` is subtracted over the
        stim-on window. Only the instantaneous edge jump is used, so slow
        real fluorescence kinetics are largely preserved. Requires
        ``stim_on_frames`` and ``stim_off_frames``. Works with both
        f0_mode='global' and 'per_trial'.
    stim_on_frames, stim_off_frames : array-like of int or None
        Per-trial stimulus onset / offset frame indices (into the T axis),
        required when ``stim_step_dff=True``. Use the *visual-stimulus*
        offset (where the leak ends), not reward time. Convert behaviour
        times to frames as for ``trial_onsets``, e.g.
        ``np.searchsorted(exp.rec_t, exp.beh.stim.t_start)`` for onsets and
        the same against the stimulus-off times, subtracting ``idx_start``
        if the corrected stack does not begin at frame 0.
    stim_step_edge : {'both', 'on', 'off'}
        Which stim edges feed the per-pixel step estimate. Default 'both'.
        Use 'on' when the OFF edge sits on a reward-evoked transient in a
        functional control channel. Only used when ``stim_step_dff=True``.
    stim_step_n_edge, stim_step_n_gap : int
        Frames averaged each side of an edge / skipped at the transition,
        for the per-pixel step estimate. Defaults 3 and 1. Only used when
        ``stim_step_dff=True``.
    batch_size_cap : int
        Hard ceiling on the auto-scaled batch size. ``_prepare_inputs``
        only ever *raises* batch_size, so lowering the default alone
        gives no protection against a large auto-scaled batch.
    return_components : bool
        Also return the full dff_sig / dff_ctrl stacks on ``info``
        (3x the memory). Mutually exclusive with ``output_path``.

    Returns
    -------
    corrected : (T, X, Y) ndarray / memmap
        Per-pixel dF/F, NaN where the pixel was not corrected.
    info : SimpleNamespace
        ``snr_s1``, ``snr_s2``, ``mask_ctrl``, ``mask_sig``,
        ``mask_out``, ``den``, ``f0_sig``, ``f0_ctrl``, ``f0_trial``,
        ``f0_trial_ctrl``, ``trial_of_frame``, ``baseline_s1``,
        ``baseline_s2`` (None when ``remove_baseline=False``),
        ``mean_trace_s1``, ``mean_trace_s2``, ``n_valid_sig``,
        ``n_valid_ctrl``, ``n_valid_out``, the echoed parameters, and
        (with ``return_components``) ``dff_sig`` / ``dff_ctrl``.

        With ``segmentation_type='cells'`` it also carries
        ``cell_labels`` (X, Y int32 soma instance map), ``cell_ring_labels``
        (X, Y int32 donut-annulus map), ``cell_centroids`` (K, 2),
        ``cell_traces`` (K, T corrected ring dF/F, NaN in per-trial gaps)
        and ``seg`` (detection diagnostics: ``n_cells``, ``diam_px``,
        ``coverage``, ``used_watershed``). These are None under 'gmm'.

    Notes
    -----
    With ``remove_baseline=True`` the airPLS drift is removed
    **additively** and identically at every pixel, while real drift
    (bleaching, absorption) is closer to multiplicative — so a dim pixel
    receives more drift correction, in dF/F units, than it should. The
    multiplicative form ``F_corr = s · mean_t(b)/b(t)`` would fix this
    but breaks the identity ``mean_t(F_corr) = mean_t(s)`` that lets both
    F0 and the SNR estimate be derived from a single streaming pass. The
    additive form is kept for that reason — and is one more argument for
    leaving ``remove_baseline`` off and letting the dF/F subtraction
    cancel the drift instead.

    ``info.snr_s1`` / ``snr_s2`` are returned as diagnostics but no
    longer gate anything. Under shot noise σ_n² = g·mean, so
    SNR = √(mean/g) is a monotone transform of brightness — measured
    Spearman correlation with the plain mean image is 0.97–0.99, i.e. it
    carries no information the mean image lacks. They remain the easiest
    way to read a recording's photon rate and detector gain: fitting
    σ_n² = g·mean gives g directly, and mean/g is photons per pixel per
    frame. σ_n is biased upwards by genuine dynamics
    (E[Σd²/2n] = σ_n² + mean((Δf)²)/2), so SNR is a conservative lower
    bound.
    """
    from scipy.ndimage import gaussian_filter

    if s1.shape != s2.shape:
        raise ValueError(
            f"s1 and s2 must have same shape, "
            f"got {s1.shape} and {s2.shape}")
    if np.issubdtype(np.dtype(dtype), np.integer):
        raise ValueError(
            f"dtype={dtype!r}: correct_pixel_spatial_subtr marks invalid "
            f"pixels with NaN, which integer dtypes cannot represent "
            f"(the cast would silently yield the dtype's minimum). "
            f"Use a float dtype.")
    if return_components and output_path is not None:
        raise ValueError(
            "return_components and output_path are mutually exclusive")
    if f0_mode not in ('global', 'per_trial'):
        raise ValueError(
            f"f0_mode must be 'global' or 'per_trial', got {f0_mode!r}")
    if segmentation_type not in ('gmm', 'cells'):
        raise ValueError(
            f"segmentation_type must be 'gmm' or 'cells', got "
            f"{segmentation_type!r}")

    # Cell-mode outputs; stay None under 'gmm'.
    labels = ring_labels = cell_traces = cell_centroids = seg_info = None

    T, X, Y = s1.shape
    trim_initial = int(max(0, min(trim_initial, T - 2)))

    if f0_mode == 'global' and trim_initial + n_frames_f0 > T:
        raise ValueError(
            f"trim_initial + n_frames_f0 = {trim_initial + n_frames_f0} "
            f"exceeds T = {T}")
    if f0_mode == 'per_trial':
        if trial_onsets is None or trial_window is None:
            raise ValueError(
                "f0_mode='per_trial' requires both trial_onsets and "
                "trial_window (e.g. trial_window=(-30, 90) frames "
                "relative to onset)")
    if stim_step_dff:
        if stim_on_frames is None or stim_off_frames is None:
            raise ValueError(
                "stim_step_dff=True requires stim_on_frames and "
                "stim_off_frames (per-trial stimulus on/off frame indices "
                "into the T axis)")
        if stim_step_edge not in ('both', 'on', 'off'):
            raise ValueError(
                f"stim_step_edge must be 'both', 'on' or 'off', got "
                f"{stim_step_edge!r}")

    s1, s2, batch_size = _prepare_inputs(
        s1, s2, batch_size, T, X, Y, verbose)
    # _prepare_inputs only ever raises batch_size, so cap it here.
    batch_size = int(min(batch_size, max(1, batch_size_cap)))
    n_batches = (T + batch_size - 1) // batch_size

    # Per-trial windows: F0 is taken from the frames immediately before
    # onset; the output extent is a separate, usually wider, window.
    per_trial = (f0_mode == 'per_trial')
    if per_trial:
        onsets = np.asarray(trial_onsets, dtype=np.intp).ravel()
        n_trials = onsets.size
        f0w_start = np.maximum(onsets - int(trial_baseline_frames), 0)
        f0w_end = np.minimum(onsets, T)
        outw_start = np.maximum(onsets + int(trial_window[0]), 0)
        outw_end = np.minimum(onsets + int(trial_window[1]), T)
        # Overlapping output windows would let a later trial silently
        # overwrite an earlier one in trial_of_frame.
        _order = np.argsort(outw_start)
        _os, _oe = outw_start[_order], outw_end[_order]
        if np.any(_os[1:] < _oe[:-1]):
            raise ValueError(
                "trial_window produces overlapping output windows; "
                "each frame must belong to at most one trial")
        f0_trial_s1 = np.zeros((n_trials, X, Y), dtype=np.float64)
        f0_trial_s2 = np.zeros((n_trials, X, Y), dtype=np.float64)
    else:
        n_trials = 0
        f0_start = trim_initial
        f0_end = trim_initial + int(n_frames_f0)
        pix_sum_f0_s1 = np.zeros((X, Y), dtype=np.float64)
        pix_sum_f0_s2 = np.zeros((X, Y), dtype=np.float64)

    if verbose:
        _disk = ' -> disk' if output_path else ''
        print(f'\tcorrecting signal (pixel_spatial_subtr{_disk})...')
        print(f'\t\tsigma_px={sigma_px}, pixel_gate={pixel_gate}, '
              f'f0_mode={f0_mode}'
              f'{f" (n_trials={n_trials})" if per_trial else ""}, '
              f'remove_baseline={remove_baseline}')

    # ------------------------------------------------------------------
    # Pass 1 — spatial-mean traces, per-pixel mean, noise, F0 sums
    # ------------------------------------------------------------------
    # SNR is computed on the raw s: mean_t(F_corr) == mean_t(s) exactly
    # (the subtracted term is zero-mean over t), and diff(b) is orders of
    # magnitude below single-pixel shot noise because b derives from the
    # whole-frame mean and is then airPLS-smoothed. So the baseline need
    # not be known yet and one pass suffices.
    if verbose:
        print(f'\t\tpass 1/2: mean traces, per-pixel moments & F0 '
              f'({n_batches} batches)...')
    mean_trace_s1 = np.zeros(T, dtype=np.float64)
    mean_trace_s2 = np.zeros(T, dtype=np.float64)
    pix_sum_s1 = np.zeros((X, Y), dtype=np.float64)
    pix_sum_s2 = np.zeros((X, Y), dtype=np.float64)
    sum_dsq_s1 = np.zeros((X, Y), dtype=np.float64)
    sum_dsq_s2 = np.zeros((X, Y), dtype=np.float64)
    n_diff = 0
    n_pix_frames = 0
    prev_last_s1 = prev_last_s2 = None

    for bi, t_start, t_end, s1_b, s2_b in _batched_f32(s1, s2, T, batch_size):
        if verbose:
            print(f'\t\t\tbatch {bi+1}/{n_batches} '
                  f'({(bi+1)/n_batches*100:.1f}%)...', end='\r')
        mean_trace_s1[t_start:t_end] = s1_b.mean(axis=(1, 2), dtype=np.float64)
        mean_trace_s2[t_start:t_end] = s2_b.mean(axis=(1, 2), dtype=np.float64)

        # Warm-up frames are excluded from the per-pixel mean and the
        # noise estimate (a warm-up transient would inflate sigma_n and
        # depress SNR across the whole FOV).
        _a0 = max(t_start, trim_initial)
        if _a0 < t_end:
            _o = _a0 - t_start
            pix_sum_s1 += s1_b[_o:].sum(axis=0, dtype=np.float64)
            pix_sum_s2 += s2_b[_o:].sum(axis=0, dtype=np.float64)
            n_pix_frames += t_end - _a0
            # Chunked successive differences: np.diff(s_b, axis=0) would
            # allocate a whole extra (n_b, X, Y) float32.
            _n_eff = t_end - _a0
            for c0 in range(0, _n_eff - 1, 64):
                c1 = min(c0 + 64, _n_eff - 1)
                _d1 = s1_b[_o + c0 + 1:_o + c1 + 1] - s1_b[_o + c0:_o + c1]
                sum_dsq_s1 += np.einsum('ijk,ijk->jk', _d1, _d1,
                                        dtype=np.float64)
                _d2 = s2_b[_o + c0 + 1:_o + c1 + 1] - s2_b[_o + c0:_o + c1]
                sum_dsq_s2 += np.einsum('ijk,ijk->jk', _d2, _d2,
                                        dtype=np.float64)
                n_diff += c1 - c0
        # Bridge the (t_start-1, t_start) difference across the batch seam.
        if prev_last_s1 is not None and t_start - 1 >= trim_initial:
            _d = s1_b[0] - prev_last_s1
            sum_dsq_s1 += _d.astype(np.float64) ** 2
            _d = s2_b[0] - prev_last_s2
            sum_dsq_s2 += _d.astype(np.float64) ** 2
            n_diff += 1
        # .copy() is required: a bare view pins the whole batch alive, and
        # _batched_f32 prefetches the next one while we hold it.
        prev_last_s1 = s1_b[-1].copy()
        prev_last_s2 = s2_b[-1].copy()

        # F0 sums, raw. The baseline correction is a scalar applied after
        # pass 1 once b exists, so this stays exact.
        if per_trial:
            for k in range(n_trials):
                _a = max(f0w_start[k], t_start)
                _z = min(f0w_end[k], t_end)
                if _z > _a:
                    f0_trial_s1[k] += s1_b[_a - t_start:_z - t_start].sum(
                        axis=0, dtype=np.float64)
                    f0_trial_s2[k] += s2_b[_a - t_start:_z - t_start].sum(
                        axis=0, dtype=np.float64)
        else:
            _g0 = max(t_start, f0_start)
            _g1 = min(t_end, f0_end)
            if _g1 > _g0:
                pix_sum_f0_s1 += s1_b[_g0 - t_start:_g1 - t_start].sum(
                    axis=0, dtype=np.float64)
                pix_sum_f0_s2 += s2_b[_g0 - t_start:_g1 - t_start].sum(
                    axis=0, dtype=np.float64)
    del prev_last_s1, prev_last_s2
    if verbose:
        print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')

    # ------------------------------------------------------------------
    # Between passes — baselines, SNR, masks, F0 (all cheap, no I/O)
    # ------------------------------------------------------------------
    # Optional whole-frame drift removal. Off by default: the unit-gain
    # dff_sig - dff_ctrl subtraction already cancels drift that is common
    # to both channels. When off, F_corr == s and every b-derived
    # correction term below collapses to zero.
    if remove_baseline:
        if verbose:
            print('\t\trunning airPLS baselines on s̄1(t), s̄2(t)...')
        baseline_s1 = _airpls(
            _moving_average(mean_trace_s1, smooth_window), lam=airpls_lam,
            porder=airpls_porder, max_iter=airpls_max_iter)
        baseline_s2 = _airpls(
            _moving_average(mean_trace_s2, smooth_window), lam=airpls_lam,
            porder=airpls_porder, max_iter=airpls_max_iter)
        mean_b1 = float(np.mean(baseline_s1))
        mean_b2 = float(np.mean(baseline_s2))
    else:
        if verbose:
            print('\t\tskipping airPLS baselines (remove_baseline=False)')
        baseline_s1 = baseline_s2 = None
        mean_b1 = mean_b2 = 0.0

    # SNR = mean(F) / sigma_n, sigma_n from successive frame differences
    # (Var(s[t+1] - s[t]) = 2*sigma_n^2 for temporally independent noise).
    # Diagnostic only — it no longer gates anything, because under shot
    # noise SNR = sqrt(mean/g) is just a monotone transform of the mean.
    pix_mean_s1 = (pix_sum_s1 / max(n_pix_frames, 1)).astype(np.float32)
    pix_mean_s2 = (pix_sum_s2 / max(n_pix_frames, 1)).astype(np.float32)
    sigma_n_s1 = np.sqrt(sum_dsq_s1 / (2.0 * max(n_diff, 1)))
    sigma_n_s2 = np.sqrt(sum_dsq_s2 / (2.0 * max(n_diff, 1)))
    with np.errstate(divide='ignore', invalid='ignore'):
        snr_s1 = (pix_mean_s1 / np.maximum(sigma_n_s1, eps)).astype(np.float32)
        snr_s2 = (pix_mean_s2 / np.maximum(sigma_n_s2, eps)).astype(np.float32)
    snr_s1 = np.nan_to_num(snr_s1, nan=0.0, posinf=0.0, neginf=0.0)
    snr_s2 = np.nan_to_num(snr_s2, nan=0.0, posinf=0.0, neginf=0.0)
    del pix_sum_s1, pix_sum_s2, sum_dsq_s1, sum_dsq_s2

    # Expression gate, per channel, from the session mean image.
    mask_ctrl, gate_s1 = _expression_gate(
        pix_mean_s1, method=pixel_gate, log=gmm_log,
        smooth_px=gmm_smooth_px, min_sep=gmm_min_sep, gate_pct=gate_pct,
        label='s1/control', verbose=verbose)
    if segmentation_type == 'gmm':
        mask_sig, gate_s2 = _expression_gate(
            pix_mean_s2, method=pixel_gate, log=gmm_log,
            smooth_px=gmm_smooth_px, min_sep=gmm_min_sep, gate_pct=gate_pct,
            label='s2/signal', verbose=verbose)
    else:
        # Cell-aware: detect dark-hole somata in the signal mean, then keep
        # the expression-gated neuropil with the somata carved out. Applying
        # the same GMM expression gate as 'gmm' mode (rather than keeping
        # every non-soma pixel) makes 'cells' a strict refinement of 'gmm':
        # the same strongly-expressing pixels, minus the soma interiors. Dim
        # / background pixels would otherwise dominate mask_out and, with a
        # near-zero F0, pull the corrected whole-frame trace away from 0.
        # Per-cell ring traces come after pass 2, once the corrected stack
        # exists.
        labels, seg_info = _segment_somata(
            pix_mean_s2, min_darkness=soma_min_darkness,
            diam_px=soma_diam_px, bg_sigma=soma_bg_sigma, eps=eps,
            label='s2/signal', verbose=verbose)
        if seg_info['n_cells'] == 0:
            raise ValueError(
                "segmentation_type='cells' detected no somata in the "
                f"signal mean image (soma_min_darkness={soma_min_darkness}, "
                f"soma_diam_px={soma_diam_px}); loosen these or inspect "
                "the mean image. Cell bodies must read as dark holes in "
                "brighter neuropil for this mode.")
        # Expression gate first, then punch out the soma interiors.
        mask_expr, gate_s2 = _expression_gate(
            pix_mean_s2, method=pixel_gate, log=gmm_log,
            smooth_px=gmm_smooth_px, min_sep=gmm_min_sep, gate_pct=gate_pct,
            label='s2/signal', verbose=verbose)
        mask_sig = mask_expr & ~(labels > 0)
        if neuropil_floor_pct is not None:
            _floor = np.float32(np.percentile(pix_mean_s2,
                                              float(neuropil_floor_pct)))
            mask_sig &= (pix_mean_s2 >= _floor)
    if not mask_ctrl.any():
        raise ValueError(
            f"the expression gate kept no control pixel (cut="
            f"{gate_s1['cut']:.3g}); nothing can contribute to the "
            f"spatial average")

    # Masked Gaussian average (normalised convolution). den is the
    # fraction of kernel weight coming from high-SNR control pixels;
    # scipy renormalises the truncated kernel to sum 1, so den lies in
    # [0, 1] and equals 1 across a fully-valid FOV (mode='reflect').
    mask_ctrl_f32 = mask_ctrl.astype(np.float32)
    den = gaussian_filter(
        mask_ctrl_f32, sigma=sigma_px, mode='reflect').astype(np.float32)
    inv_den = (np.float32(1.0) / np.maximum(den, np.float32(eps))
               ).astype(np.float32)

    def _masked_gauss(a):
        """Masked Gaussian average of a single (X, Y) map."""
        _num = gaussian_filter(
            np.asarray(a, dtype=np.float32) * mask_ctrl_f32,
            sigma=sigma_px, mode='reflect')
        return (_num * inv_den).astype(np.float32)

    # F0 of the control average is exact without a second pass: the
    # masked average is a fixed linear operator with time-invariant
    # weights, so mean_t(gauss_avg(s1)) == gauss_avg(mean_t(s1)).
    if per_trial:
        f0_sig = f0_ctrl = None
        inv_f0_sig = inv_f0_ctrl = None
        _n_used = (f0w_end - f0w_start).astype(np.float64)
        _bad_trial = _n_used <= 0
        _n_safe = np.maximum(_n_used, 1.0)
        f0_trial = np.empty((n_trials, X, Y), dtype=np.float32)
        f0_trial_ctrl = np.empty((n_trials, X, Y), dtype=np.float32)
        for k in range(n_trials):
            # Same clipped window for the pixel sum and the b-correction.
            if remove_baseline and not _bad_trial[k]:
                _c2 = (float(np.mean(baseline_s2[f0w_start[k]:f0w_end[k]]))
                       - mean_b2)
                _c1 = (float(np.mean(baseline_s1[f0w_start[k]:f0w_end[k]]))
                       - mean_b1)
            else:
                _c1 = _c2 = 0.0
            f0_trial[k] = (f0_trial_s2[k] / _n_safe[k] - _c2).astype(np.float32)
            _f0_s1_k = (f0_trial_s1[k] / _n_safe[k] - _c1).astype(np.float32)
            f0_trial_ctrl[k] = _masked_gauss(_f0_s1_k)
        del f0_trial_s1, f0_trial_s2
        # A trial is usable only where both F0s are strictly positive.
        # Bad trials (empty baseline window) never contribute output frames
        # (see trial_of_frame below) so they are excluded here too — one
        # bad trial's all-zero F0 row must not null every pixel globally.
        _good = ~_bad_trial
        if _good.any():
            _f0_ok = ((f0_trial[_good] > 0).all(axis=0)
                      & (f0_trial_ctrl[_good] > 0).all(axis=0))
        else:
            _f0_ok = np.zeros((X, Y), dtype=bool)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_f0_trial = (np.float32(1.0)
                            / np.where(f0_trial > 0, f0_trial, np.float32(1.0))
                            ).astype(np.float32)
            inv_f0_trial_ctrl = (
                np.float32(1.0)
                / np.where(f0_trial_ctrl > 0, f0_trial_ctrl, np.float32(1.0))
                ).astype(np.float32)
        # Map every frame to its trial (-1 = no trial). Trials whose F0
        # window was empty are dropped entirely.
        trial_of_frame = np.full(T, -1, dtype=np.int32)
        for k in range(n_trials):
            if not _bad_trial[k] and outw_end[k] > outw_start[k]:
                trial_of_frame[outw_start[k]:outw_end[k]] = k
    else:
        trial_of_frame = None
        f0_trial = f0_trial_ctrl = None
        inv_f0_trial = inv_f0_trial_ctrl = None
        if remove_baseline:
            _c2 = float(np.mean(baseline_s2[f0_start:f0_end])) - mean_b2
            _c1 = float(np.mean(baseline_s1[f0_start:f0_end])) - mean_b1
        else:
            _c1 = _c2 = 0.0
        _n_f0 = float(f0_end - f0_start)
        f0_sig = (pix_sum_f0_s2 / _n_f0 - _c2).astype(np.float32)
        _f0_s1 = (pix_sum_f0_s1 / _n_f0 - _c1).astype(np.float32)
        f0_ctrl = _masked_gauss(_f0_s1)
        del pix_sum_f0_s1, pix_sum_f0_s2
        _f0_ok = (f0_sig > 0) & (f0_ctrl > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_f0_sig = (np.float32(1.0)
                          / np.where(f0_sig > 0, f0_sig, np.float32(1.0))
                          ).astype(np.float32)
            inv_f0_ctrl = (np.float32(1.0)
                           / np.where(f0_ctrl > 0, f0_ctrl, np.float32(1.0))
                           ).astype(np.float32)

    # mask_ctrl gates which pixels *contribute* to the average; mask_sig
    # (plus enough local control weight and a usable F0) gates where
    # output exists. Additive baseline removal can push a dim pixel's F0
    # to ~0, where (F - F0)/F0 would blow past float16's range.
    mask_out = mask_sig & (den >= np.float32(den_min)) & _f0_ok
    if verbose:
        _npx = X * Y
        print(f'\t\tSNR (diagnostic): s1 median '
              f'{float(np.median(snr_s1)):.3g}, '
              f's2 median {float(np.median(snr_s2)):.3g}')
        print(f'\t\tmasks: ctrl {int(mask_ctrl.sum())}/{_npx} '
              f'({mask_ctrl.mean()*100:.1f}%), sig {int(mask_sig.sum())}/'
              f'{_npx} ({mask_sig.mean()*100:.1f}%), out '
              f'{int(mask_out.sum())}/{_npx} ({mask_out.mean()*100:.1f}%)')
        print(f'\t\tden: median {float(np.median(den)):.3f}, '
              f'{int((den < den_min).sum())} px below den_min={den_min}')
    if not mask_out.any():
        raise ValueError(
            f"mask_out is empty: no pixel passes the {pixel_gate} "
            f"expression gate (s2 cut={gate_s2['cut']:.3g}) with "
            f"den >= {den_min} and a positive F0. Inspect the mean-image "
            f"histogram and info.gate_*.")

    # Per-pixel dF/F-domain stim-step subtraction. Estimated here (masks,
    # inv_f0 and _masked_gauss all exist) and applied per stim-on frame in
    # pass 2. Unlike the raw whole-frame remove_stim_step, the leak is
    # resolved per pixel *and* in dF/F units, so it handles both the leak's
    # spatial structure and the different fractional size it takes in each
    # channel. The correction subtracts dff_sig − dff_ctrl with unit gain,
    # so the residual stim step we must remove is likewise
    # step_dff_sig − step_dff_ctrl; the control jump propagates through the
    # same masked-Gaussian operator that builds dff_ctrl (it is linear and
    # time-invariant, so the jump map filters exactly like a frame).
    step_box = None
    step_raw_sig = step_raw_ctrl = None
    n_step_edges = 0
    if stim_step_dff:
        if verbose:
            print('\t\testimating per-pixel stim step (dF/F domain)...')
        _st_s1, _st_s2, n_step_edges = estimate_pixel_step_maps(
            s1, s2, stim_on_frames, stim_off_frames, edge=stim_step_edge,
            n_edge=int(stim_step_n_edge), n_gap=int(stim_step_n_gap),
            verbose=verbose)
        step_raw_sig = _st_s2                       # (X, Y) raw jump in s2
        step_raw_ctrl = _masked_gauss(_st_s1)       # jump through dff_ctrl op
        step_box = build_stim_step_box(
            stim_on_frames, stim_off_frames, T)
        if verbose:
            print(f'\t\tstim step: {int(step_box.sum())}/{T} stim-on frames '
                  f'flagged for subtraction')
        # Global mode: the per-pixel corrected-domain step is constant in
        # time, so precompute it once. per_trial rescales by each trial's F0.
        if f0_mode == 'global':
            step_dff_global = (step_raw_sig * inv_f0_sig
                               - step_raw_ctrl * inv_f0_ctrl).astype(np.float32)
        else:
            step_dff_global = None

    # ------------------------------------------------------------------
    # Pass 2 — masked spatial average, dF/F, subtract, write
    # ------------------------------------------------------------------
    if output_path is not None:
        if verbose:
            print(f'\t\twriting output to: {output_path}')
        corrected = tifffile.memmap(
            output_path, shape=(T, X, Y), dtype=dtype, bigtiff=True)
    else:
        corrected = np.zeros((T, X, Y), dtype=dtype)

    info = SimpleNamespace()
    if return_components:
        info.dff_sig = np.zeros((T, X, Y), dtype=dtype)
        info.dff_ctrl = np.zeros((T, X, Y), dtype=dtype)

    baseline_s1_f32 = (baseline_s1.astype(np.float32)
                       if remove_baseline else None)
    baseline_s2_f32 = (baseline_s2.astype(np.float32)
                       if remove_baseline else None)
    num_buf = np.empty((batch_size, X, Y), dtype=np.float32)
    _nan = np.float32(np.nan)

    # F0-weighted (ratio-of-means) whole-frame and per-sector aggregates
    # of each component (dff_sig, dff_ctrl), accumulated per frame in
    # pass 2. A *plain* spatial mean of the per-pixel dF/F stack is a
    # mean-of-ratios; over dim pixels with a tiny F0 (e.g. cells-mode
    # neuropil rings on a photon-starved sensor, or any per_trial F0 from
    # a short noisy baseline) it is dominated by a heavy tail and inflates
    # the trace (measured: median pixel −53%, plain mean +11%, correct
    # ratio-of-means +4%). Weighting each pixel's dF/F by its *own* F0
    # recovers the population dF/F, Σ ΔF / Σ F0 — the SNR-correct
    # estimator (f0·dff telescopes to ΔF). The weight MUST be the same
    # per-trial F0 that formed the dF/F denominator (a static weight does
    # not cancel a per-trial blow-up), so it is accumulated here where
    # that F0 is in scope. Downstream (the QC) uses
    # ``wf_dff_sig − wf_dff_ctrl`` (and the per-sector analogue) as the
    # corrected whole-frame / per-sector trace instead of a plain mean of
    # the stack. Accumulate numerator Σ(dff·f0) and denominator Σ(f0)
    # separately; the ratio is formed after the loop (NaN where no valid
    # frame, e.g. per_trial inter-trial gaps).
    _mask_out_flat = mask_out.ravel()
    _n_mask_out = int(_mask_out_flat.sum())
    _wf_num_sig = np.zeros(T, dtype=np.float64)
    _wf_num_ctrl = np.zeros(T, dtype=np.float64)
    _wf_den_sig = np.zeros(T, dtype=np.float64)
    _wf_den_ctrl = np.zeros(T, dtype=np.float64)

    # Optional per-sector aggregates (same F0 weighting). Pixels are
    # assigned to an n×n grid matching the QC's _compute_sectors partition
    # (floor blocks, remainder trimmed). Built once as a boolean
    # (n_mask, n_sec²) membership matrix so each frame's weighted
    # per-sector sum is a single matmul.
    _do_sec = (aggregate_sectors is not None and int(aggregate_sectors) > 0
               and _n_mask_out > 0)
    if _do_sec:
        _ns = int(aggregate_sectors)
        _by = X // _ns
        _bx = Y // _ns
        _yy, _xx = np.mgrid[0:X, 0:Y]
        _in_grid = (_yy < _by * _ns) & (_xx < _bx * _ns)
        _row = np.where(_in_grid, np.minimum(_yy // max(_by, 1), _ns - 1), -1)
        _col = np.where(_in_grid, np.minimum(_xx // max(_bx, 1), _ns - 1), -1)
        _sid_full = np.where(_in_grid, _row * _ns + _col, -1)
        _sid_mask = _sid_full.ravel()[_mask_out_flat]        # (n_mask,)
        _onehot = np.zeros((_n_mask_out, _ns * _ns), dtype=np.float32)
        _ok_sec = _sid_mask >= 0
        _onehot[np.arange(_n_mask_out)[_ok_sec], _sid_mask[_ok_sec]] = 1.0
        _sec_num_sig = np.zeros((_ns * _ns, T), dtype=np.float64)
        _sec_num_ctrl = np.zeros((_ns * _ns, T), dtype=np.float64)
        _sec_den_sig = np.zeros((_ns * _ns, T), dtype=np.float64)
        _sec_den_ctrl = np.zeros((_ns * _ns, T), dtype=np.float64)

    def _accum_agg(_t0, _t1, _sig_b, _ctl_b, _w_s, _w_c):
        """Accumulate F0-weighted num/den for frames [_t0:_t1].

        _sig_b/_ctl_b: (nf, X, Y) dF/F for those frames. _w_s/_w_c:
        (X, Y) per-pixel F0 weights (own channel). NaN dF/F contributes 0.
        """
        _sf = _sig_b.reshape(_sig_b.shape[0], -1)[:, _mask_out_flat]
        _cf = _ctl_b.reshape(_ctl_b.shape[0], -1)[:, _mask_out_flat]
        _sf = np.where(np.isfinite(_sf), _sf, 0.0).astype(np.float64)
        _cf = np.where(np.isfinite(_cf), _cf, 0.0).astype(np.float64)
        _ws = _w_s.ravel()[_mask_out_flat]
        _wc = _w_c.ravel()[_mask_out_flat]
        _ws = np.where(np.isfinite(_ws) & (_ws > 0), _ws, 0.0).astype(np.float64)
        _wc = np.where(np.isfinite(_wc) & (_wc > 0), _wc, 0.0).astype(np.float64)
        _wf_num_sig[_t0:_t1] = _sf @ _ws
        _wf_num_ctrl[_t0:_t1] = _cf @ _wc
        _wf_den_sig[_t0:_t1] = _ws.sum()
        _wf_den_ctrl[_t0:_t1] = _wc.sum()
        if _do_sec:
            _sec_num_sig[:, _t0:_t1] = ((_sf * _ws) @ _onehot).T
            _sec_num_ctrl[:, _t0:_t1] = ((_cf * _wc) @ _onehot).T
            _sec_den_sig[:, _t0:_t1] = (_ws @ _onehot)[:, None]
            _sec_den_ctrl[:, _t0:_t1] = (_wc @ _onehot)[:, None]

    if verbose:
        print(f'\t\tpass 2/2: spatial average & dF/F ({n_batches} '
              f'batches)...')
    for bi, t_start, t_end, s1_b, s2_b in _batched_f32(s1, s2, T, batch_size):
        if verbose:
            print(f'\t\t\tbatch {bi+1}/{n_batches} '
                  f'({(bi+1)/n_batches*100:.1f}%)...', end='\r')
        n_b = s1_b.shape[0]
        # F_corr: subtract only the fluctuating part of the baseline so
        # the DC level (and hence F0) survives.
        if remove_baseline:
            s1_b -= (baseline_s1_f32[t_start:t_end] - mean_b1)[:, None, None]
            s2_b -= (baseline_s2_f32[t_start:t_end] - mean_b2)[:, None, None]

        # Masked Gaussian average of the control channel. sigma=0 on axis
        # 0 is dropped from scipy's filter loop entirely, so this never
        # mixes across time and the per-batch result is identical to
        # filtering the whole stack.
        s1_b *= mask_ctrl_f32
        gaussian_filter(s1_b, sigma=(0.0, sigma_px, sigma_px),
                        output=num_buf[:n_b], mode='reflect')
        ctrl = num_buf[:n_b]
        ctrl *= inv_den

        if per_trial:
            tof_b = trial_of_frame[t_start:t_end]
            valid_b = np.zeros(n_b, dtype=bool)
            for k in np.unique(tof_b[tof_b >= 0]):
                _idx = np.flatnonzero(tof_b == k)
                _a, _z = int(_idx[0]), int(_idx[-1]) + 1  # contiguous
                valid_b[_a:_z] = True
                _sig = s2_b[_a:_z]
                _sig -= f0_trial[k]
                _sig *= inv_f0_trial[k]
                _ctl = ctrl[_a:_z]
                _ctl -= f0_trial_ctrl[k]
                _ctl *= inv_f0_trial_ctrl[k]
                if return_components:
                    info.dff_sig[t_start + _a:t_start + _z] = \
                        _clip_to_dtype(_sig, dtype)
                    info.dff_ctrl[t_start + _a:t_start + _z] = \
                        _clip_to_dtype(_ctl, dtype)
                if _n_mask_out:
                    # Weight each channel by its own per-trial F0.
                    _accum_agg(t_start + _a, t_start + _z, _sig, _ctl,
                               f0_trial[k], f0_trial_ctrl[k])
                _sig -= _ctl
                # Per-pixel stim-step subtraction, rescaled by this trial's
                # F0 (the raw jump is trial-invariant; only F0 changes).
                if stim_step_dff:
                    _mbk = step_box[t_start + _a:t_start + _z]
                    if _mbk.any():
                        _Ak = (step_raw_sig * inv_f0_trial[k]
                               - step_raw_ctrl * inv_f0_trial_ctrl[k]
                               ).astype(np.float32)
                        _sig[_mbk] -= _Ak
            s2_b[~valid_b] = _nan
        else:
            ctrl -= f0_ctrl
            ctrl *= inv_f0_ctrl
            s2_b -= f0_sig
            s2_b *= inv_f0_sig
            if return_components:
                info.dff_sig[t_start:t_end] = _clip_to_dtype(s2_b, dtype)
                info.dff_ctrl[t_start:t_end] = _clip_to_dtype(ctrl, dtype)
            if _n_mask_out:
                # Weight each channel by its own (global) per-pixel F0.
                _accum_agg(t_start, t_end, s2_b, ctrl, f0_sig, f0_ctrl)
            s2_b -= ctrl
            # Per-pixel stim-step subtraction over stim-on frames.
            if stim_step_dff:
                _mb = step_box[t_start:t_end]
                if _mb.any():
                    s2_b[_mb] -= step_dff_global

        # NaN-safe (clip is minimum/maximum, which propagate NaN).
        np.clip(s2_b, -32000.0, 32000.0, out=s2_b)
        s2_b[:, ~mask_out] = _nan
        corrected[t_start:t_end] = _clip_to_dtype(s2_b, dtype)
    if verbose:
        print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')
    del num_buf

    # Per-cell donut-ring traces (cell mode only). Each background pixel
    # is assigned to its nearest soma (a Voronoi partition), then a ring
    # of width ring_width_px around each soma, clipped to mask_out, gives
    # that cell's ROI. Traces are the frame-wise mean of the corrected
    # stack over each ring — a masked matmul, streamed so a memmapped
    # `corrected` is read back once, sequentially.
    if segmentation_type == 'cells':
        from scipy import sparse
        from scipy import ndimage as ndi
        n_cells = int(labels.max())
        ring_labels = _donut_rings(labels, mask_out, ring_width_px)
        cell_centroids = np.asarray(
            ndi.center_of_mass(np.ones_like(labels), labels,
                               np.arange(1, n_cells + 1)), dtype=np.float64)
        cell_traces = np.full((n_cells, T), np.nan, dtype=np.float32)
        _rl = ring_labels.ravel()
        _valid = _rl > 0
        if _valid.any():
            _lab = _rl[_valid]
            _inc = sparse.csr_matrix(
                (np.ones(_lab.size, dtype=np.float32),
                 (_lab - 1, np.arange(_lab.size))),
                shape=(n_cells, _lab.size))
            _counts = np.asarray(_inc.sum(axis=1)).ravel()
            if verbose:
                print(f'\t\textracting {n_cells} per-cell ring traces '
                      f'({int(_valid.sum())} ring px)...')
            for _t0 in range(0, T, batch_size):
                _t1 = min(_t0 + batch_size, T)
                _cb = np.asarray(corrected[_t0:_t1], dtype=np.float32)
                _cb = _cb.reshape(_t1 - _t0, -1)[:, _valid]
                # A per-trial gap frame is all-NaN over mask_out (hence over
                # the ring), so its column stays NaN — correct. A kept frame
                # has finite ring pixels (ring ⊂ mask_out).
                with np.errstate(invalid='ignore'):
                    cell_traces[:, _t0:_t1] = (
                        (_inc @ _cb.T) / np.maximum(_counts[:, None], 1.0))

    info.snr_s1 = snr_s1
    info.snr_s2 = snr_s2
    # F0-weighted (ratio-of-means) component traces: whole-frame and, when
    # requested, per-sector. NaN where the weight denominator is 0 (no
    # valid frame — e.g. per_trial inter-trial gaps). The QC forms the
    # corrected trace as wf_dff_sig − wf_dff_ctrl (and the sector
    # analogue).
    with np.errstate(divide='ignore', invalid='ignore'):
        info.wf_dff_sig = np.where(
            _wf_den_sig > 0, _wf_num_sig / _wf_den_sig, np.nan
            ).astype(np.float32)
        info.wf_dff_ctrl = np.where(
            _wf_den_ctrl > 0, _wf_num_ctrl / _wf_den_ctrl, np.nan
            ).astype(np.float32)
        if _do_sec:
            info.sec_dff_sig = np.where(
                _sec_den_sig > 0, _sec_num_sig / _sec_den_sig, np.nan
                ).astype(np.float32)
            info.sec_dff_ctrl = np.where(
                _sec_den_ctrl > 0, _sec_num_ctrl / _sec_den_ctrl, np.nan
                ).astype(np.float32)
        else:
            info.sec_dff_sig = None
            info.sec_dff_ctrl = None
    info.mask_ctrl = mask_ctrl
    info.mask_sig = mask_sig
    info.mask_out = mask_out
    info.den = den
    info.f0_sig = f0_sig
    info.f0_ctrl = f0_ctrl
    info.f0_trial = f0_trial
    info.f0_trial_ctrl = f0_trial_ctrl
    info.trial_of_frame = trial_of_frame
    info.baseline_s1 = baseline_s1_f32
    info.baseline_s2 = baseline_s2_f32
    info.mean_trace_s1 = mean_trace_s1.astype(np.float32)
    info.mean_trace_s2 = mean_trace_s2.astype(np.float32)
    info.n_valid_sig = int(mask_sig.sum())
    info.n_valid_ctrl = int(mask_ctrl.sum())
    info.n_valid_out = int(mask_out.sum())
    info.f0_mode = f0_mode
    info.remove_baseline = bool(remove_baseline)
    info.sigma_px = float(sigma_px)
    info.den_min = float(den_min)
    # Expression gate: what was asked for, what was actually used, and
    # whether the bimodality guard fired (gate_method_s* == 'percentile'
    # with pixel_gate='gmm' means the fit was discarded).
    info.pixel_gate = pixel_gate
    info.gate_method_s1 = gate_s1['method']
    info.gate_method_s2 = gate_s2['method']
    info.gate_cut_s1 = gate_s1['cut']
    info.gate_cut_s2 = gate_s2['cut']
    info.gate_sep_s1 = gate_s1['sep']
    info.gate_sep_s2 = gate_s2['sep']
    info.gate_bic_s1 = (gate_s1['bic1'], gate_s1['bic2'])
    info.gate_bic_s2 = (gate_s2['bic1'], gate_s2['bic2'])
    info.gate_fallback_s1 = gate_s1['fallback']
    info.gate_fallback_s2 = gate_s2['fallback']
    info.gmm_log = bool(gmm_log)
    # the resolved sigma, not the 'auto' sentinel
    info.gmm_smooth_px = gate_s2['smooth_px']
    info.gmm_min_sep = float(gmm_min_sep)
    info.gate_pct = float(gate_pct)
    # Per-pixel dF/F stim-step subtraction diagnostics (None/0 when off).
    info.stim_step_dff = bool(stim_step_dff)
    info.stim_step_edge = stim_step_edge if stim_step_dff else None
    info.stim_step_n_edges = int(n_step_edges)
    info.stim_step_raw_sig = step_raw_sig
    info.stim_step_raw_ctrl = step_raw_ctrl
    info.stim_step_box = step_box
    # Cell-aware segmentation outputs (None under segmentation_type='gmm').
    info.segmentation_type = segmentation_type
    info.cell_labels = labels
    info.cell_ring_labels = ring_labels
    info.cell_centroids = cell_centroids
    info.cell_traces = cell_traces
    info.seg = seg_info

    if output_path is not None:
        corrected.flush()
        if verbose:
            print('\t\tflushed output to disk')
    return corrected, info


# =============================================================================
# Field-inference helper: BayesNF per-channel denoiser
# =============================================================================

# torch-MPS reimplementation of the BayesNF MAP fit
# -------------------------------------------------------
# A faithful, GPU-friendly port of bayesnf's BayesianNeuralFieldMAP (NORMAL
# observation model) for Apple Silicon. bayesnf runs on jax, whose Metal
# backends are either unstable (jax-metal deadlocks) or version-incompatible
# with the pinned jax==0.4.35 / tfp stack (applejax, jax-mps need jaxlib
# 0.9/0.10). torch's MPS backend has neither problem. The ensemble is the
# leading axis of every parameter so all members train in one set of batched
# ops. Validated against bayesnf: field agreement corr >= 0.999 and matched
# ground-truth recovery, ~2.5-2.8x faster on an M1 Pro.


def _torch_fourier_features(scaled_col, freqs, denom):
    """cos/sin Fourier features for one (rescaled) input dim.

    ``freqs`` and ``denom`` are precomputed once per model (they depend only on
    the Fourier degree, not on the data) and passed in, so the hot forward path
    allocates no per-call ``arange``/frequency tensors — this matters on MPS,
    where per-op dispatch overhead dominates for these tiny tensors.

    Parameters
    ----------
    scaled_col : torch.Tensor
        (E, N) rescaled values for one feature column.
    freqs : torch.Tensor
        (degree,) angular frequencies ``2*pi*2**k`` (bayesnf convention).
    denom : torch.Tensor
        (2*degree,) per-feature normalising denominators.

    Returns
    -------
    feats : torch.Tensor
        (E, N, 2*degree) rescaled cos/sin features (bayesnf convention).
    """
    import torch
    y = scaled_col[..., None] * freqs
    feats = torch.cat([torch.cos(y), torch.sin(y)], dim=-1)
    return feats / denom


def _make_torch_bnf_ensemble(n_ens, n_feat, width, depth, input_scales,
                             interactions, fourier_degree=5,
                             log_noise_init=0.0):
    """Build a vectorized ensemble of BayesNF-style MAP neural fields.

    Mirrors ``bayesnf.models.BayesianNeuralField1D`` plus the ``fit_map`` init
    (Dense kernels ~ TruncatedNormal(0, 1, [-2, 2]); all biases / scales zero;
    ``log_noise_scale = log(std/2)``). The ensemble dimension ``E`` leads every
    parameter so members train in parallel.

    Parameters
    ----------
    n_ens, n_feat, width, depth : int
        Ensemble size, number of input features (t, x, y -> 3), MLP width and
        number of hidden layers.
    input_scales : array-like
        Per-feature scale (time scale for the time column, 1 for standardized
        spatial columns).
    interactions : sequence of (int, int)
        Feature-column index pairs whose products are added as features.
    fourier_degree : int
        Fourier degrees per input dim. Default 5.
    log_noise_init : float or array-like
        Initial log observation-noise scale. A scalar is broadcast to all
        members; a length-``n_ens`` array sets each member independently (used
        to give the two channels different noise inits in the joint fit).
        Default 0.

    Returns
    -------
    model : torch.nn.Module
        The ensemble module (defined lazily so torch is only imported when the
        torch backend is actually used).
    """
    import torch

    class _TorchBNFEnsemble(torch.nn.Module):
        def __init__(self):
            super().__init__()
            E, F = n_ens, n_feat
            self.E, self.n_feat, self.width, self.depth = E, F, width, depth
            self.interactions = interactions
            self.fdeg = fourier_degree
            self.register_buffer(
                'input_scales',
                torch.as_tensor(input_scales, dtype=torch.float32))

            # Precompute the Fourier frequencies / denominators once; they
            # depend only on the degree, so the per-epoch forward path reuses
            # these buffers instead of rebuilding them (see
            # _torch_fourier_features).
            degrees = torch.arange(fourier_degree, dtype=torch.float32)
            self.register_buffer(
                'fourier_freqs', 2.0 * np.pi * (2.0 ** degrees))
            self.register_buffer(
                'fourier_denom', torch.cat([degrees + 1, degrees + 1]))

            n_int = len(interactions)
            feat_dim = F + F * 2 * fourier_degree + n_int
            n_groups = 1 + F + (1 if n_int > 0 else 0)

            P = torch.nn.Parameter
            _z = lambda *s: P(torch.zeros(E, *s))
            self.log_scale_adjustment = _z(F)
            self.feature_scales = _z(n_groups)
            self.layer_scales = _z(depth)
            self.logit_activation_weight = _z()
            self.output_scale = _z()
            _lni = torch.as_tensor(log_noise_init, dtype=torch.float32)
            if _lni.ndim == 0:
                _lni = _lni.expand(E).clone()
            self.log_noise_scale = P(_lni)

            dims = [feat_dim] + [width] * depth
            self.W = torch.nn.ParameterList()
            self.b = torch.nn.ParameterList()
            for din, dout in zip(dims[:-1], dims[1:]):
                w = torch.empty(E, din, dout)
                torch.nn.init.trunc_normal_(w, 0.0, 1.0, -2.0, 2.0)
                self.W.append(P(w))
                self.b.append(_z(dout))
            wo = torch.empty(E, width, 1)
            torch.nn.init.trunc_normal_(wo, 0.0, 1.0, -2.0, 2.0)
            self.Wout = P(wo)
            self.bout = _z(1)

        def _features(self, x):
            E = self.E
            xb = x[None].expand(E, -1, -1)
            scale = self.input_scales * torch.exp(self.log_scale_adjustment)
            scaled_x = xb / scale[:, None, :]

            groups = [scaled_x]
            for i in range(self.n_feat):
                groups.append(_torch_fourier_features(
                    scaled_x[..., i], self.fourier_freqs,
                    self.fourier_denom))
            if self.interactions:
                # Product feature per interaction tuple; supports any order
                # (pairwise (a, b) or higher, e.g. a 3-way (t, x, y) term that
                # lets the field localise a transient in space *and* time).
                inter = [scaled_x[..., list(idx)].prod(dim=-1)
                         for idx in self.interactions]
                groups.append(torch.stack(inter, dim=-1))

            sp = torch.nn.functional.softplus(self.feature_scales)
            groups = [g * sp[:, gi][:, None, None]
                      for gi, g in enumerate(groups)]
            return torch.cat(groups, dim=-1)

        def forward(self, x):
            h = self._features(x)
            aw = torch.sigmoid(self.logit_activation_weight)[:, None, None]

            def _act(v):
                return aw * torch.nn.functional.elu(v) \
                    + (1 - aw) * torch.tanh(v)

            for li in range(self.depth):
                h = h / np.sqrt(h.shape[-1])
                lin = torch.bmm(h, self.W[li]) \
                    + self.b[li][:, None, :]
                ls = torch.nn.functional.softplus(
                    self.layer_scales[:, li])[:, None, None]
                h = _act(ls * lin)
            h = h / np.sqrt(h.shape[-1])
            out = torch.bmm(h, self.Wout) \
                + self.bout[:, None, :]
            os_ = torch.nn.functional.softplus(
                self.output_scale)[:, None, None]
            return (os_ * out)[..., 0]

    return _TorchBNFEnsemble()


def _torch_bnf_prior_logprob(model):
    """Summed Logistic(0, 1) log-prior over all params, per ensemble member.

    Mirrors bayesnf's Logistic prior over ``log_noise_scale`` and all MLP
    weights (the unused NB shape / zero-inflation params do not affect the
    NORMAL prediction and are omitted).

    Returns
    -------
    total : torch.Tensor
        (E,) prior log-probability per ensemble member.
    """
    import torch
    _lp = lambda z: -z - 2.0 * torch.nn.functional.softplus(-z)
    E = model.E
    total = torch.zeros(E, device=model.log_noise_scale.device)
    scalars = [model.log_scale_adjustment, model.feature_scales,
               model.layer_scales, model.logit_activation_weight,
               model.output_scale, model.log_noise_scale, model.bout]
    for p in scalars + list(model.b):
        pl = p if p.dim() > 0 else p[:, None]
        total = total + _lp(pl).reshape(E, -1).sum(-1)
    for w in list(model.W) + [model.Wout]:
        total = total + _lp(w).reshape(E, -1).sum(-1)
    return total


def _torch_bnf_fit_predict_dual(ds1, ds2, seed, T, Xb, Yb, x_flat, y_flat,
                                width, depth, num_epochs, ensemble_size,
                                learning_rate, interactions, device='mps',
                                verbose=False, early_stop=True,
                                es_check_every=50, es_tol=1e-4, es_patience=5,
                                batch_rows=None, mixed_precision=False,
                                mp_dtype='float16', pred_chunk=None,
                                fourier_degree=5, noise_init=None):
    """Jointly fit torch-MPS BayesNF ensembles to both channels; predict them.

    Replaces the two sequential per-channel fits inside ``infer_field_bayesnf``
    with a single training loop. The two channels share the exact same coarse
    ``(t, x, y)`` coordinate grid, so their inputs are identical; only the
    target values (and their NaN masks) differ. The ensemble axis is therefore
    doubled to ``2*ensemble_size`` — members ``[0:ensemble_size]`` fit channel
    1, members ``[ensemble_size:]`` fit channel 2 — and both channels train in
    one set of batched ops. Because each member's loss depends only on its own
    slice of the parameters (the ensemble is the leading axis of every param),
    this is mathematically the two independent fits, just dispatched together
    (the win is MPS per-op launch overhead paid once, not twice).

    Per-channel NaN masking is applied in the loss (rows are predicted over the
    full union grid, masked rows contribute nothing), so the two channels may
    have different missing pixels.

    Parameters
    ----------
    ds1, ds2 : (T, Xb, Yb) ndarray
        The two coarse (binned) channels to fit.
    seed : int
        PRNG seed for init / optimisation.
    T, Xb, Yb : int
        Number of frames and coarse grid shape.
    x_flat, y_flat : ndarray
        Flattened coarse spatial coordinates (length ``Xb*Yb``).
    width, depth, num_epochs, ensemble_size, learning_rate : see caller.
    interactions : sequence of (int, int)
        Interaction feature column pairs.
    device : str
        torch device (``'mps'`` or ``'cpu'``). Falls back to CPU if ``'mps'``
        is requested but unavailable.
    verbose : bool
        Print periodic training loss.
    early_stop : bool
        Stop before ``num_epochs`` once the (joint) loss plateaus. Default True.
    es_check_every : int
        Epochs between convergence checks (each syncs the loss to host, so this
        is kept coarse). Default 50.
    es_tol : float
        Relative-improvement threshold; a check counts as a plateau when the
        summed loss improves by less than this fraction. Default 1e-4.
    es_patience : int
        Number of consecutive plateau checks before stopping. Default 5.
    batch_rows : int or None
        If given and smaller than the number of grid rows, each epoch fits a
        random subset of ``batch_rows`` rows (minibatch SGD) instead of the
        full grid, with the per-member log-likelihood rescaled to the full-grid
        magnitude so the prior weighting is preserved. Trades a noisier
        gradient for a smaller per-step memory footprint. ``None`` (default)
        keeps exact full-batch MAP.
    mixed_precision : bool
        If True, run the forward pass / loss under ``torch.autocast`` in
        ``mp_dtype`` (master params stay float32). NB: benchmarked *no* speedup
        on an M1 Pro / torch 2.8 (autocast fp16/bf16 ran slightly slower than
        fp32 for this matmul-bound workload); kept only for other hardware /
        larger problems. Experimental — validate against the reference before
        trusting. Default False.
    mp_dtype : {'float16', 'bfloat16'}
        Autocast dtype when ``mixed_precision`` is set. Default 'float16'.
    pred_chunk : int or None
        Chunk size (in grid rows) for the final full-grid prediction forward
        pass, to bound its peak memory. ``None`` (default) predicts in one shot.

    Returns
    -------
    field1, field2 : (T, Xb, Yb) ndarray
        Ensemble-mean inferred coarse fields for the two channels (float32).
    """
    import torch
    if device == 'mps' and not torch.backends.mps.is_available():
        if verbose:
            print('\t\t\t(MPS unavailable; falling back to CPU)')
        device = 'cpu'
    if device == 'mps':
        # Disable MPS's conservative memory-reclaim watermark: for this
        # full-batch workload it otherwise proactively evicts/recompacts
        # allocations and can stall training. Set before the first MPS
        # allocation (the allocator reads it once, at init).
        os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')
    torch.manual_seed(int(seed))

    ens = ensemble_size
    E = 2 * ens
    n_pix = Xb * Yb
    N = T * n_pix
    # Long-format coordinates: one row per (t, x, y) coarse voxel. Identical
    # for both channels (deterministic grid), so the input is shared.
    x_raw = np.column_stack([
        np.repeat(np.arange(T, dtype=np.float32), n_pix),
        np.tile(x_flat, T), np.tile(y_flat, T)]).astype(np.float32)
    val1 = ds1.reshape(N).astype(np.float32)
    val2 = ds2.reshape(N).astype(np.float32)

    # Standardize spatial cols (mean/std); time offset only (min already 0).
    # Moments use all grid rows (deterministic coords), independent of masking.
    mu = np.array([0.0, x_raw[:, 1].mean(), x_raw[:, 2].mean()], np.float32)
    sd = np.array([1.0, x_raw[:, 1].std(), x_raw[:, 2].std()], np.float32)
    sd[sd == 0] = 1.0
    x_std = (x_raw - mu) / sd
    input_scales = np.array([float(max(T - 1, 1)), 1.0, 1.0], np.float32)
    # Noise-scale init. Default (std/2) is a broad prior, but the field's
    # std is dominated by real spatial/temporal *structure*, so std/2 seeds
    # the observation noise ~an order of magnitude above the true measurement
    # noise and drives amplitude shrinkage of genuine high-SNR features
    # (sharp activity hotspots). Passing ``noise_init`` (an absolute std in
    # the data's units, e.g. the per-pixel measurement noise) seeds a tighter,
    # physically-sized noise floor so the fit is penalised for smoothing real
    # signal away.
    if noise_init is not None:
        lni1 = lni2 = float(np.log(max(float(noise_init), 1e-6)))
    else:
        lni1 = float(np.log(max(np.nanstd(val1), 1e-6) / 2.0))
        lni2 = float(np.log(max(np.nanstd(val2), 1e-6) / 2.0))
    log_noise_init = np.concatenate([
        np.full(ens, lni1, np.float32), np.full(ens, lni2, np.float32)])

    # Per-member validity mask (channel 1 for the first block, channel 2 for
    # the second). NaN targets are zeroed and masked out of the loss so the
    # field is still inferred there from the smooth model.
    m1 = ~np.isnan(val1)
    m2 = ~np.isnan(val2)
    y_stack = np.stack([np.nan_to_num(val1, nan=0.0)] * ens
                       + [np.nan_to_num(val2, nan=0.0)] * ens, axis=0)
    w_stack = np.stack([m1.astype(np.float32)] * ens
                       + [m2.astype(np.float32)] * ens, axis=0)

    model = _make_torch_bnf_ensemble(
        E, 3, width, depth, input_scales,
        [tuple(p) for p in interactions],
        fourier_degree=fourier_degree,
        log_noise_init=log_noise_init).to(device)
    xt = torch.as_tensor(x_std, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_stack, dtype=torch.float32, device=device)
    wt = torch.as_tensor(w_stack, dtype=torch.float32, device=device)
    w_sum_full = wt.sum(-1)  # (E,) valid-row count per member

    # foreach (multi-tensor) Adam batches the small per-parameter updates into
    # a few multi-tensor ops. NB: the optimiser step is a small fraction of the
    # per-epoch cost here (the forward/backward matmuls dominate), and
    # ``fused=True`` benchmarked ~10% *slower* than foreach on an M1 Pro /
    # torch 2.8 for this workload, so fused is deliberately not used.
    try:
        opt = torch.optim.Adam(
            model.parameters(), lr=learning_rate, foreach=True)
    except (RuntimeError, ValueError, TypeError):
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    use_mb = batch_rows is not None and batch_rows < N
    autocast_dtype = getattr(torch, mp_dtype) if mixed_precision else None
    autocast_dev = 'mps' if device == 'mps' else 'cpu'
    _const = 0.5 * np.log(2 * np.pi)

    def _loss():
        if use_mb:
            idx = torch.randperm(N, device=device)[:batch_rows]
            xb, yb, wb = xt[idx], yt[:, idx], wt[:, idx]
            # Rescale the minibatch log-lik up to the full-grid magnitude so
            # the (data-independent) prior keeps the same relative weight.
            w_sum_mb = wb.sum(-1).clamp(min=1.0)
            ll_scale = w_sum_full / w_sum_mb
        else:
            xb, yb, wb = xt, yt, None
            ll_scale = None
        pred = model(xb)
        scale = 0.01 + torch.exp(model.log_noise_scale)[:, None]
        per_row = (-0.5 * ((yb - pred) / scale) ** 2
                   - torch.log(scale) - _const)
        if use_mb:
            ll = (per_row * wb).sum(-1) * ll_scale
        else:
            ll = (per_row * wt).sum(-1)
        return -(ll + _torch_bnf_prior_logprob(model)).sum()

    best_loss = np.inf
    n_plateau = 0
    for ep in range(num_epochs):
        opt.zero_grad()
        if autocast_dtype is not None:
            with torch.autocast(device_type=autocast_dev,
                                dtype=autocast_dtype):
                loss = _loss()
        else:
            loss = _loss()
        loss.backward()
        opt.step()

        # Convergence check (coarse: each syncs the loss to host). Stop once
        # the summed loss has improved by < es_tol (relative) for es_patience
        # consecutive checks — MAP fits usually plateau well before num_epochs.
        is_check = early_stop and (
            ep % es_check_every == 0 or ep == num_epochs - 1)
        if is_check:
            cur = loss.item()
            rel_impr = (best_loss - cur) / max(abs(best_loss), 1e-8)
            if rel_impr < es_tol:
                n_plateau += 1
            else:
                n_plateau = 0
            if cur < best_loss:
                best_loss = cur
            if verbose:
                print(f'\t\t\tepoch {ep}/{num_epochs}: '
                      f'loss/E = {cur/E:.1f}')
            if n_plateau >= es_patience:
                if verbose:
                    print(f'\t\t\tconverged at epoch {ep} '
                          f'(loss plateaued {es_patience} checks)')
                break
        elif verbose and (ep % 200 == 0 or ep == num_epochs - 1):
            print(f'\t\t\tepoch {ep}/{num_epochs}: '
                  f'loss/E = {loss.item()/E:.1f}')

    # Final full-grid prediction, optionally chunked over rows to cap peak
    # memory (each row is independent, so chunking is numerically exact).
    with torch.no_grad():
        if pred_chunk is not None and pred_chunk < N:
            parts = []
            for c0 in range(0, N, pred_chunk):
                parts.append(model(xt[c0:c0 + pred_chunk]))
            field = torch.cat(parts, dim=1)
        else:
            field = model(xt)
        field1 = field[:ens].mean(0).cpu().numpy().astype(np.float32)
        field2 = field[ens:].mean(0).cpu().numpy().astype(np.float32)
    return field1.reshape(T, Xb, Yb), field2.reshape(T, Xb, Yb)


def infer_field_bayesnf(
        s1, s2, batch_size=1000, dtype=np.float16, verbose=True,
        output_paths=None,
        bin_factor=8,
        width=128, depth=1, num_epochs=1000, ensemble_size=16,
        learning_rate=0.005, seed=0,
        interactions=((0, 1), (0, 2), (1, 2)),
        backend='torch_mps',
        early_stop=True, es_check_every=50, es_tol=1e-4, es_patience=5,
        batch_rows=None, mixed_precision=False, mp_dtype='float16',
        fourier_degree=5, noise_init=None):
    """Infer the underlying denoised fields s1_real, s2_real from s1, s2.

    This is a *helper*, not a signal corrector: it applies no coupling,
    subtraction or β. It fits an independent BayesNF (Bayesian Neural
    Field, ``BayesianNeuralFieldMAP``) spatiotemporal field to each channel
    on a spatially-downsampled grid, then reconstructs each smooth field at
    full resolution by interpolation, and returns the two inferred fields.
    A downstream real-signal-correction method is expected to consume them.

    The intuition: the real and control fluorophores have spatially
    heterogeneous expression (soma vs. neuropil), so each observed channel
    is a noisy sample of a smooth, spatiotemporally-varying underlying
    field. BayesNF learns ``f(t, x, y)`` from a long-format table of
    observations, so fitting one field per channel denoises it while
    respecting the spatial expression structure.

    The stack is reduced to a coarse ``(T, Xb, Yb)`` grid
    (``Xb = X // bin_factor``) for fitting; predicting per-pixel per-frame
    at native resolution would be billions of forward passes, so the smooth
    coarse field is bilinearly upsampled back to ``(T, X, Y)`` instead.

    Parameters
    ----------
    s1, s2 : (T, X, Y) ndarray or memmap
        The two raw channels (e.g. control and real). Treated symmetrically
        — one field is fit per channel, independently. Pixels set to NaN are
        treated as *missing* (masked): they are excluded from the BayesNF
        fit, and the field is inferred there from the smooth spatiotemporal
        model — i.e. inference across the missing pixels.
    batch_size : int
        Frames per streaming batch (binning pass and reconstruction pass,
        and the per-chunk prediction size). Default 1000.
    dtype : np.dtype
        Output dtype of the reconstructed field stacks. Default np.float16.
    verbose : bool
        Print progress.
    output_paths : (str, str) or None
        If given, stream the two reconstructed fields (s1_real, s2_real) to
        memmapped bigtiffs at these paths instead of RAM buffers (a dual
        full-res stack is memory-heavy). Default None.
    bin_factor : int
        Spatial downsample factor for the BayesNF fit. Default 8.
    width, depth : int
        BayesNF MLP hidden width and number of layers. Default 256, 2.
    num_epochs : int
        MAP training epochs per channel. Default 5000.
    ensemble_size : int
        BayesNF MAP ensemble size; must be >= ``jax.device_count()``.
        Default 16.
    learning_rate : float
        MAP optimiser learning rate. Default 0.005.
    seed : int
        Base PRNG seed (s1 uses ``seed``, s2 uses ``seed + 1``). Default 0.
    interactions : sequence of (int, int)
        Feature-column index pairs whose interactions the field may use
        (0=t, 1=x, 2=y). Default ``((0,1),(0,2),(1,2))``.
    backend : {'bayesnf', 'torch_mps', 'torch_cpu'}
        Field-fitting backend. ``'torch_mps'`` (default) uses a faithful
        torch reimplementation of bayesnf's BayesianNeuralFieldMAP that runs
        on the Apple-Silicon GPU (validated: field agreement corr >= 0.999
        vs the reference; ~2.5-2.8x faster than ``'bayesnf'`` on an M1 Pro;
        avoids jax's unstable/incompatible Metal backends). ``'torch_cpu'``
        runs the same torch model on CPU. ``'bayesnf'`` uses the reference
        jax/tfp implementation. ``torch_mps`` falls back to CPU if MPS is
        unavailable.
    early_stop : bool
        For the torch backends, stop each channel's MAP fit early once the
        loss plateaus rather than always running ``num_epochs``. Default True.
    es_check_every : int
        Epochs between convergence checks (torch backends). Default 50.
    es_tol : float
        Relative-improvement plateau threshold (torch backends). Default 1e-4.
    es_patience : int
        Consecutive plateau checks before stopping (torch backends). Default 5.
    batch_rows : int or None
        For the torch backends, minibatch the coarse-grid rows during the fit
        (see ``_torch_bnf_fit_predict_dual``) rather than full-batch. Lowers
        per-step memory at the cost of a noisier gradient. ``None`` (default)
        keeps exact full-batch MAP.
    mixed_precision : bool
        For the torch backends, run the fit forward/loss under
        ``torch.autocast`` (master weights stay float32). Benchmarked no
        speedup on M1 Pro (this workload is matmul-bound and MPS fp16 GEMM was
        not faster); ``width``/``depth`` are the effective speed levers. Kept
        for other hardware. Experimental. Default False.
    mp_dtype : {'float16', 'bfloat16'}
        Autocast dtype when ``mixed_precision`` is set. Default 'float16'.

    Returns
    -------
    s1_real, s2_real : (T, X, Y) ndarray / memmap
        The full-resolution inferred (denoised) fields for each channel.
    diagnostics : SimpleNamespace
        ``.s1_real_ds`` / ``.s2_real_ds`` the coarse ``(T, Xb, Yb)``
        inferred fields (before upsampling), ``.bin_factor``, ``.n_bin``
        the ``(Xb, Yb)`` grid shape.
    """
    if s1.shape != s2.shape:
        raise ValueError(
            f"s1 and s2 must have same shape, "
            f"got {s1.shape} and {s2.shape}")
    if bin_factor < 1:
        raise ValueError(f"bin_factor must be >= 1, got {bin_factor}")

    T, X, Y = s1.shape
    s1, s2, batch_size = _prepare_inputs(
        s1, s2, batch_size, T, X, Y, verbose)
    n_batches = (T + batch_size - 1) // batch_size

    # Coarse (binned) grid used for the BayesNF fit. FOV is trimmed to a
    # multiple of bin_factor, then averaged into Xb x Yb blocks.
    Xb = max(1, X // bin_factor)
    Yb = max(1, Y // bin_factor)
    x_trim = Xb * bin_factor
    y_trim = Yb * bin_factor
    n_pix_b = Xb * Yb

    if backend not in ('bayesnf', 'torch_mps', 'torch_cpu'):
        raise ValueError(
            "backend must be 'bayesnf', 'torch_mps' or 'torch_cpu', "
            f"got {backend!r}")
    torch_device = {'torch_mps': 'mps', 'torch_cpu': 'cpu'}.get(backend)

    # Lazy import: bayesnf pulls in jax + tensorflow-probability, which are
    # heavy and (in this env) version-fragile; the torch backend needs neither.
    # Only pay the relevant import cost when the helper is actually called.
    if backend == 'bayesnf':
        try:
            import jax
            from bayesnf import BayesianNeuralFieldMAP
        except Exception as e:
            raise ImportError(
                "infer_field_bayesnf needs the 'bayesnf' package with a "
                "compatible jax / tensorflow-probability; import failed: "
                f"{e}") from e
        import pandas as pd
    else:
        try:
            import torch  # noqa: F401
        except Exception as e:
            raise ImportError(
                f"infer_field_bayesnf backend={backend!r} needs 'torch'; "
                f"import failed: {e}") from e
    from scipy.ndimage import zoom

    if verbose:
        print(f'\tinferring fields ({backend}, bin_factor={bin_factor} '
              f'-> {Xb}x{Yb} grid, {num_epochs} epochs)...')

    # ------------------------------------------------------------------
    # Pass 1 — spatially bin both channels to the coarse grid.
    # ------------------------------------------------------------------
    # Same block-mean reshape as the sector traces in correct_full_regress:
    # trim to x_trim/y_trim, carve into Xb x Yb blocks, average each block.
    # NaN input pixels are treated as *missing* (masked): blocks average
    # only their valid pixels (nanmean), and a fully-masked block becomes
    # NaN, which is dropped from the fit table so BayesNF infers that
    # location from the smooth field (spatial inference across the hole).
    if verbose:
        print(f'\t\tpass 1: binning to coarse grid ({n_batches} batches)...')
    s1_ds = np.zeros((T, Xb, Yb), dtype=np.float32)
    s2_ds = np.zeros((T, Xb, Yb), dtype=np.float32)
    for bi, t0, t1, s1_b, s2_b in _batched_f32(s1, s2, T, batch_size):
        if verbose:
            print(f'\t\t\tbatch {bi+1}/{n_batches} '
                  f'({(bi+1)/n_batches*100:.1f}%)...', end='\r')
        nfb = s1_b.shape[0]
        with np.errstate(invalid='ignore'):
            s1_ds[t0:t1] = np.nanmean(s1_b[:, :x_trim, :y_trim].reshape(
                nfb, Xb, bin_factor, Yb, bin_factor), axis=(2, 4))
            s2_ds[t0:t1] = np.nanmean(s2_b[:, :x_trim, :y_trim].reshape(
                nfb, Xb, bin_factor, Yb, bin_factor), axis=(2, 4))
    if verbose:
        print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')

    # Per-frame coarse spatial coordinates (x slow, y fast — matches the
    # C-order ravel of the (Xb, Yb) block grid, so val = ds[t].ravel()).
    _xg, _yg = np.meshgrid(np.arange(Xb, dtype=np.float32),
                           np.arange(Yb, dtype=np.float32), indexing='ij')
    x_flat = _xg.ravel()
    y_flat = _yg.ravel()
    interactions = [list(p) for p in interactions]

    def _fit_predict(ds, _seed):
        """Fit one reference-bayesnf MAP field to coarse stack ds; predict it.

        Returns the coarse (T, Xb, Yb) posterior-mean field. Used only for the
        ``'bayesnf'`` backend; the torch backends fit both channels jointly via
        ``_torch_bnf_fit_predict_dual``.
        """
        # Long-format fit table: one row per (t, x, y) coarse voxel. Masked
        # (NaN) voxels are dropped so BayesNF fits only observed locations
        # and infers the field across the missing ones.
        t_col = np.repeat(np.arange(T, dtype=np.float32), n_pix_b)
        df = pd.DataFrame({
            't': t_col,
            'x': np.tile(x_flat, T),
            'y': np.tile(y_flat, T),
            'val': ds.reshape(T * n_pix_b).astype(np.float32)})
        df = df.dropna(subset=['val'])
        model = BayesianNeuralFieldMAP(
            feature_cols=['t', 'x', 'y'], target_col='val',
            timetype='float', standardize=['x', 'y'],
            interactions=interactions, observation_model='NORMAL',
            width=width, depth=depth)
        model.fit(df, seed=jax.random.PRNGKey(int(_seed)),
                  ensemble_size=ensemble_size, num_epochs=num_epochs,
                  learning_rate=learning_rate)
        # Predict the coarse field, chunked over frames to bound the size
        # of the returned ensemble array.
        pred = np.empty((T, Xb, Yb), dtype=np.float32)
        for c0 in range(0, T, batch_size):
            c1 = min(c0 + batch_size, T)
            nfr = c1 - c0
            df_p = pd.DataFrame({
                't': np.repeat(np.arange(c0, c1, dtype=np.float32), n_pix_b),
                'x': np.tile(x_flat, nfr),
                'y': np.tile(y_flat, nfr)})
            means, _ = model.predict(df_p, quantiles=(0.5,))
            # means: (n_devices, ensemble/n_devices, n_rows) -> ensemble mean.
            pred[c0:c1] = np.row_stack(means).mean(axis=0).astype(
                np.float32).reshape(nfr, Xb, Yb)
        return pred

    if torch_device is not None:
        # Both channels share the coarse grid, so fit them jointly in one
        # doubled-ensemble training loop (one set of batched MPS ops instead
        # of two sequential fits). pred_chunk caps the final prediction's peak
        # memory using the same frame-batch size used elsewhere.
        if verbose:
            print(f'\t\tfitting {backend} fields for s1 + s2 (joint)...')
        s1_real_ds, s2_real_ds = _torch_bnf_fit_predict_dual(
            s1_ds, s2_ds, seed, T, Xb, Yb, x_flat, y_flat,
            width, depth, num_epochs, ensemble_size, learning_rate,
            interactions, device=torch_device, verbose=verbose,
            early_stop=early_stop, es_check_every=es_check_every,
            es_tol=es_tol, es_patience=es_patience,
            batch_rows=batch_rows, mixed_precision=mixed_precision,
            mp_dtype=mp_dtype, pred_chunk=batch_size * n_pix_b,
            fourier_degree=fourier_degree, noise_init=noise_init)
    else:
        if verbose:
            print(f'\t\tfitting {backend} field for s1...')
        s1_real_ds = _fit_predict(s1_ds, seed)
        if verbose:
            print(f'\t\tfitting {backend} field for s2...')
        s2_real_ds = _fit_predict(s2_ds, seed + 1)

    # ------------------------------------------------------------------
    # Pass 2 — bilinearly upsample each coarse field back to full res.
    # ------------------------------------------------------------------
    # zoom order=1 (bilinear); the field is smooth by construction, so
    # interpolating the coarse posterior mean is the faithful full-res
    # reconstruction (and avoids per-pixel NN forward passes).
    if output_paths is not None:
        p1, p2 = output_paths
        if verbose:
            print(f'\t\twriting fields to: {p1}, {p2}')
        s1_real = tifffile.memmap(p1, shape=(T, X, Y), dtype=dtype,
                                  bigtiff=True)
        s2_real = tifffile.memmap(p2, shape=(T, X, Y), dtype=dtype,
                                  bigtiff=True)
    else:
        s1_real = np.zeros((T, X, Y), dtype=dtype)
        s2_real = np.zeros((T, X, Y), dtype=dtype)

    zoom_xy = (1.0, X / Xb, Y / Yb)
    if verbose:
        print(f'\t\tpass 2: upsampling to {X}x{Y} '
              f'({n_batches} batches)...')
    for bi in range(n_batches):
        t0 = bi * batch_size
        t1 = min(t0 + batch_size, T)
        if verbose:
            print(f'\t\t\tbatch {bi+1}/{n_batches} '
                  f'({(bi+1)/n_batches*100:.1f}%)...', end='\r')
        s1_up = zoom(s1_real_ds[t0:t1], zoom_xy, order=1)[:, :X, :Y]
        s2_up = zoom(s2_real_ds[t0:t1], zoom_xy, order=1)[:, :X, :Y]
        s1_real[t0:t1] = _clip_to_dtype(s1_up, dtype)
        s2_real[t0:t1] = _clip_to_dtype(s2_up, dtype)
    if verbose:
        print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done      ')

    if output_paths is not None:
        s1_real.flush()
        s2_real.flush()
        if verbose:
            print('\t\tflushed fields to disk')

    diagnostics = SimpleNamespace(
        s1_real_ds=s1_real_ds, s2_real_ds=s2_real_ds,
        bin_factor=bin_factor, n_bin=(Xb, Yb))
    return s1_real, s2_real, diagnostics


def _moving_average(x, w):
    """Length-preserving moving-average lowpass (reflect/edge-padded)."""
    if w <= 1:
        return np.asarray(x, dtype=np.float64)
    _w = int(w)
    _pad = _w // 2
    _xp = np.pad(np.asarray(x, dtype=np.float64), _pad, mode='edge')
    _kernel = np.ones(_w, dtype=np.float64) / _w
    return np.convolve(_xp, _kernel, mode='same')[_pad:_pad + len(x)]


def _sector_means(frames, n_sec, block_x, block_y, x_trim, y_trim):
    """Per-sector spatial means of a (n_frames, X, Y) frame batch.

    Trims the FOV to (x_trim, y_trim), carves it into an n_sec x n_sec
    grid of (block_x, block_y) blocks (row-major, matching
    ``QCMixin._compute_sectors``) and averages each block per frame.

    Parameters
    ----------
    frames : np.ndarray, shape (n_frames, X, Y)
        Batch of frames to average.
    n_sec : int
        Blocks per axis.
    block_x, block_y : int
        Block size along each spatial axis.
    x_trim, y_trim : int
        FOV extent used (``block * n_sec``); trailing pixels are dropped.

    Returns
    -------
    means : np.ndarray, shape (n_sec**2, n_frames)
        Sector-mean trace segment, sectors in row-major order.
    """
    n_frames = frames.shape[0]
    return frames[:, :x_trim, :y_trim].reshape(
        n_frames, n_sec, block_x, n_sec, block_y
    ).mean(axis=(2, 4)).reshape(n_frames, -1).T


def _fit_norm_slope(x_norm, y_norm, nn_slope=False, loss='linear',
                    f_scale=1.0, max_iter=100, regress_type='ols',
                    irls_tol=1e-8):
    """Coupling slope β of normalised y on normalised x (no intercept).

    Fits ``y_norm ≈ β · x_norm`` for median-centred, std-normalised
    traces (centring is handled upstream by the median subtraction, so no
    intercept term is fit). With ``loss='linear'`` this returns the
    closed-form non-negative OLS slope ``cov(x, y) / var(x)`` — the value
    used throughout the photometric pipeline. A robust ``loss`` instead
    minimises a robustified residual, which down-weights outlier frames
    (motion, z-drift, saturation, dropped frames) at the cost of slightly
    higher variance under purely Gaussian noise. It does not undo the
    attenuation bias from noise in the regressor (an errors-in-variables
    effect).

    ``regress_type`` selects *how* that robust objective is solved:
    ``'ols'`` hands it to ``scipy.optimize.least_squares`` (trust-region
    reflective on the robustified residual, seeded at the OLS slope);
    ``'irls'`` runs iteratively reweighted least squares in NumPy, i.e.
    repeated closed-form weighted OLS with weights w = ρ'(u²) recomputed
    from the current residuals (see ``_irls_norm_slope``). Both target the
    same M-estimator; they differ in the meaning of ``f_scale`` (absolute
    for 'ols', a multiple of the residual MAD for 'irls').

    Parameters
    ----------
    x_norm, y_norm : array-like, shape (n,)
        Median-centred, std-normalised control (x) and signal (y) traces
        over the fit window.
    nn_slope : bool
        If True, clamp a negative fitted slope to 0 (the control can only
        add signal).
    loss : {'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'}
        'linear' (default) ⇒ exact analytic OLS slope (bit-for-bit
        unchanged from the previous code), whatever ``regress_type`` is:
        the IRLS weights of a linear loss are identically 1, so IRLS
        reduces to a single closed-form OLS step. Any other value selects
        a robust M-estimator loss.
    f_scale : float
        Soft threshold of the robust loss. For ``regress_type='ols'`` it is
        absolute, in normalised-trace units (std ≈ 1, so 1.0 ≈ down-weight
        residuals beyond ~1σ); for ``'irls'`` it multiplies the residual
        MAD re-estimated each iteration. Only used when
        ``loss != 'linear'``.
    max_iter : int
        Iteration cap for the robust solver (``max_nfev`` for 'ols',
        reweighting iterations for 'irls'). Only used when
        ``loss != 'linear'``.
    regress_type : {'ols', 'irls'}
        Solver for the robust fit. Ignored when ``loss == 'linear'``.
    irls_tol : float
        Relative convergence tolerance on β for ``regress_type='irls'``.

    Returns
    -------
    beta : float
        Fitted slope, clamped to ≥ 0 when ``nn_slope`` is set.
    """
    if regress_type not in ('ols', 'irls'):
        raise ValueError(
            f"regress_type must be 'ols' or 'irls', got {regress_type!r}")

    x_norm = np.asarray(x_norm, dtype=np.float64)
    y_norm = np.asarray(y_norm, dtype=np.float64)

    # Analytic non-negative OLS slope (cov / var with the variance floor).
    mean_x = float(np.mean(x_norm))
    mean_y = float(np.mean(y_norm))
    var_x = max(float(np.var(x_norm)), 1e-10)
    cov_xy = float(np.mean(x_norm * y_norm) - mean_x * mean_y)
    beta_ols = cov_xy / var_x

    # Robust path requires finite inputs and a finite OLS seed; otherwise
    # fall back to the analytic slope so a single degenerate sector cannot
    # raise mid-correction (matches the 'linear' path exactly).
    use_robust = (loss != 'linear' and np.isfinite(beta_ols)
                  and np.isfinite(x_norm).all()
                  and np.isfinite(y_norm).all())
    if not use_robust:
        return 0.0 if (nn_slope and beta_ols < 0) else beta_ols

    if regress_type == 'irls':
        beta = _irls_norm_slope(
            x_norm, y_norm, loss=loss, f_scale=f_scale,
            beta_init=beta_ols, max_iter=max_iter, tol=irls_tol)
        return 0.0 if (nn_slope and beta < 0) else beta

    from scipy.optimize import least_squares
    _res = least_squares(
        lambda b: b[0] * x_norm - y_norm, x0=[beta_ols],
        loss=loss, f_scale=f_scale, max_nfev=max_iter)
    beta = float(_res.x[0])
    return 0.0 if (nn_slope and beta < 0) else beta


def _irls_weights(u, loss):
    """IRLS weights w = ρ'(u²) for scipy's robust loss functions.

    ``scipy.optimize.least_squares`` defines each robust loss through
    ρ(z) applied to the scaled squared residual z = u² (u = r / f_scale),
    minimising ½·Σ f_scale²·ρ(z). The stationary condition of that
    objective is a weighted least-squares normal equation with weights
    w_i = ρ'(z_i) — which is exactly what IRLS iterates on, so these
    weights make ``_irls_norm_slope`` target the same M-estimator scipy's
    trust-region solver does.

    Parameters
    ----------
    u : ndarray, shape (n,)
        Residuals divided by the soft threshold (r / f_scale).
    loss : {'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'}
        Robust loss name, matching ``scipy.optimize.least_squares``.

    Returns
    -------
    w : ndarray, shape (n,)
        Non-negative weights, ≤ 1, one per residual.
    """
    z = u * u
    if loss == 'linear':
        return np.ones_like(z)
    elif loss == 'soft_l1':
        return 1.0 / np.sqrt(1.0 + z)
    elif loss == 'huber':
        # ρ'(z) = 1 for z ≤ 1, else 1/√z = f_scale / |r|.
        return np.where(z <= 1.0, 1.0, 1.0 / np.sqrt(np.maximum(z, 1e-300)))
    elif loss == 'cauchy':
        return 1.0 / (1.0 + z)
    elif loss == 'arctan':
        return 1.0 / (1.0 + z * z)
    raise ValueError(
        "loss must be 'linear', 'soft_l1', 'huber', 'cauchy' or "
        f"'arctan', got {loss!r}")


def _irls_norm_slope(x, y, loss='huber', f_scale=1.0, beta_init=0.0,
                     max_iter=100, tol=1e-8):
    """Iteratively reweighted least squares for the no-intercept slope β.

    Solves the same robust M-estimation problem as the
    ``scipy.optimize.least_squares`` path of ``_fit_norm_slope``, but by
    the classical IRLS scheme: starting from the OLS slope, repeat

        1. residuals r = y − β·x
        2. robust scale s = MAD(r) / 0.6745 (median |r|, centred at 0 —
           the traces are median-centred upstream and the fit has no
           intercept, so the residuals are already ~zero-centred)
        3. weights w = ρ'((r / (f_scale·s))²)   [``_irls_weights``]
        4. closed-form weighted OLS update β = Σw·x·y / Σw·x²

    until β stops moving. Each step is a couple of NumPy reductions, so
    the whole fit costs a handful of passes over a length-T trace.

    Unlike scipy's ``f_scale`` (an absolute threshold in trace units),
    the threshold here is ``f_scale`` × the residual MAD, re-estimated
    every iteration. That makes the fit scale-equivariant — the same
    fraction of frames is down-weighted whether the residuals happen to
    sit at σ = 0.3 or σ = 3 — which matters because the sector traces
    that feed this fit differ in noise level by an order of magnitude
    across a dim FOV. Since the input traces are normalised to σ ≈ 1, the
    default f_scale = 1.0 still lands close to the scipy convention.

    Parameters
    ----------
    x, y : ndarray, shape (n,)
        Median-centred, normalised control (x) and signal (y) traces.
        Assumed finite (checked by the caller).
    loss : {'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'}
        Robust loss whose IRLS weights are used.
    f_scale : float
        Soft threshold as a multiple of the residual MAD (see above).
    beta_init : float
        Starting slope, normally the analytic OLS estimate.
    max_iter : int
        Maximum reweighting iterations.
    tol : float
        Convergence tolerance: stop once |Δβ| ≤ tol·max(1, |β|).

    Returns
    -------
    beta : float
        Fitted slope. Falls back to ``beta_init`` if the residual scale or
        the weighted denominator underflows (a degenerate / near-constant
        trace), so a single bad sector cannot raise mid-correction.
    """
    beta = float(beta_init)
    xy = x * y
    xx = x * x

    for _ in range(int(max_iter)):
        resid = y - beta * x
        # Robust residual scale (MAD about 0, → σ for Gaussian residuals).
        scale = float(np.median(np.abs(resid))) / 0.6745
        if not np.isfinite(scale) or scale <= 1e-12:
            # Residuals are (near-)exactly zero: the current slope already
            # fits, and no reweighting can improve on it.
            break
        w = _irls_weights(resid / (f_scale * scale), loss)
        denom = float(np.sum(w * xx))
        if not np.isfinite(denom) or denom <= 1e-12:
            break
        beta_new = float(np.sum(w * xy)) / denom
        if not np.isfinite(beta_new):
            break
        _step = abs(beta_new - beta)
        beta = beta_new
        if _step <= tol * max(1.0, abs(beta)):
            break

    return beta


def _norm_scale(x, kind='std'):
    """Scale estimate for trace normalisation: classical 'std' or robust 'mad'.

    'std' is the plain standard deviation. 'mad' is the median absolute
    deviation scaled by 1.4826 so it equals σ for Gaussian data but is not
    inflated by outlier frames (which would otherwise shrink the normalised
    signal and attenuate the fitted coupling β). Returns 1.0 when the
    estimate underflows, matching the existing ``s if s > 1e-10 else 1.0``
    guard so 'std' is bit-for-bit unchanged from the previous inline code.

    Parameters
    ----------
    x : array-like, shape (n,)
        Trace (typically smoothed + baseline-removed) to scale.
    kind : {'std', 'mad'}
        Scale estimator. Default 'std'.

    Returns
    -------
    scale : float
        Estimated scale (≥ 1e-10; else 1.0).
    """
    x = np.asarray(x, dtype=np.float64)
    if kind == 'mad':
        scale = 1.4826 * float(np.median(np.abs(x - np.median(x))))
    elif kind == 'std':
        scale = float(np.std(x))
    else:
        raise ValueError(f"beta_scale must be 'std' or 'mad', got {kind!r}")
    return scale if scale > 1e-10 else 1.0


def _centre_scale(x, kind='std'):
    """Median-centre and scale-normalise a trace.

    The shared normalisation step of the zdFF pipeline: subtract the
    median, divide by the ``_norm_scale`` estimate.

    Parameters
    ----------
    x : array-like, shape (n,)
        Trace (typically smoothed and baseline-removed).
    kind : {'std', 'mad'}
        Scale estimator, forwarded to ``_norm_scale``.

    Returns
    -------
    x_norm : np.ndarray, shape (n,)
        (x - median) / scale.
    med : float
        Median used for the centring.
    scale : float
        Scale used for the normalisation.
    """
    med = float(np.median(x))
    scale = _norm_scale(x, kind)
    return (x - med) / scale, med, scale


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

    Reads the next batch in background threads while the caller
    processes the current one, hiding disk latency. s1 and s2 are
    read on separate threads so reads of two distinct files (the
    typical dual-channel case) overlap rather than serialise.

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

    def _read_one(arr, t0, t1):
        return arr[t0:t1].astype(np.float32)

    # Two workers so s1 and s2 reads overlap rather than serialise.
    # On NVMe / multi-file setups the wall-clock per batch nearly
    # halves; on single-file / slow buses the GIL release inside
    # astype still lets the second read start while the first is
    # mid-copy. Both reads are submitted directly to the pool (not
    # nested inside another task) so they actually run in parallel.
    with ThreadPoolExecutor(max_workers=2) as pool:
        def _prefetch(t0, t1):
            return (pool.submit(_read_one, s1, t0, t1),
                    pool.submit(_read_one, s2, t0, t1))

        cur_pair = _prefetch(0, min(batch_size, T))
        for bi in range(n_batches):
            t0 = bi * batch_size
            t1 = min(t0 + batch_size, T)
            if bi + 1 < n_batches:
                ns = (bi + 1) * batch_size
                next_pair = _prefetch(ns, min(ns + batch_size, T))
            s1_b = cur_pair[0].result()
            s2_b = cur_pair[1].result()
            if bi + 1 < n_batches:
                cur_pair = next_pair
            yield bi, t0, t1, s1_b, s2_b


def _batched_f32_one(arr, T, batch_size):
    """Yield (bi, t0, t1, arr_b) as float32 with single-array I/O prefetch.

    Single-channel analogue of ``_batched_f32`` for passes that only need
    one stack (e.g. the Pass-2 of ``correct_two_stage_regress``, which
    streams s2 alone because every regressor is a precomputed 1-D trace).
    Reads the next batch on a background thread while the caller works.

    Parameters
    ----------
    arr : np.ndarray or np.memmap
        Input stack, shape (T, X, Y).
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
    arr_b : np.ndarray, float32, shape (t1-t0, X, Y)
    """
    n_batches = (T + batch_size - 1) // batch_size

    def _read_one(t0, t1):
        return arr[t0:t1].astype(np.float32)

    with ThreadPoolExecutor(max_workers=1) as pool:
        cur = pool.submit(_read_one, 0, min(batch_size, T))
        for bi in range(n_batches):
            t0 = bi * batch_size
            t1 = min(t0 + batch_size, T)
            if bi + 1 < n_batches:
                ns = (bi + 1) * batch_size
                nxt = pool.submit(_read_one, ns, min(ns + batch_size, T))
            arr_b = cur.result()
            if bi + 1 < n_batches:
                cur = nxt
            yield bi, t0, t1, arr_b


class _DetrendedView(object):
    """Lazy (T, X, Y) view that subtracts a per-pixel linear trend on read.

    Wraps an underlying (T, X, Y) array (typically a memmap) together with
    per-pixel intercept `a` and slope `b`. Slicing returns float32 batches
    with the trend removed, matching the output of
    detrend_linearly(arr, ...) but without ever materialising the full
    detrended array on disk or in RAM.

    Supports the indexing patterns used by the correction methods:
        - view[t0:t1]            -> (bt, X, Y) float32
        - view[ti]               -> (X, Y) float32
        - view[:, i_idx, j_idx]  -> (T, chunk_len) float32  (LMS path)
    """

    def __init__(self, arr, a, b):
        if a.shape != arr.shape[1:] or b.shape != arr.shape[1:]:
            raise ValueError(
                f"a/b shape {a.shape}/{b.shape} does not match "
                f"arr spatial shape {arr.shape[1:]}")
        self.arr = arr
        self.a = a.astype(np.float32, copy=False)
        self.b = b.astype(np.float32, copy=False)
        self.shape = arr.shape
        self.dtype = np.dtype(np.float32)
        self.ndim = arr.ndim
        self.nbytes = int(np.prod(arr.shape)) * self.dtype.itemsize

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        t_key = key[0]
        sp_key = key[1:]

        base = np.asarray(self.arr[key], dtype=np.float32)

        # Spatial subset of a, b matching the spatial part of the key.
        if sp_key:
            a_sub = self.a[sp_key]
            b_sub = self.b[sp_key]
        else:
            a_sub = self.a
            b_sub = self.b

        T = self.shape[0]
        if isinstance(t_key, slice):
            t_vals = np.arange(*t_key.indices(T), dtype=np.float32)
        elif isinstance(t_key, (int, np.integer)):
            ti = int(t_key)
            if ti < 0:
                ti += T
            return base - (a_sub + b_sub * np.float32(ti))
        else:
            # array / list of indices
            t_vals = np.asarray(t_key, dtype=np.float32)

        # Broadcast t_vals (n,) over spatial dims of a_sub/b_sub.
        n_sp = a_sub.ndim
        t_shape = (t_vals.size,) + (1,) * n_sp
        trend = a_sub + b_sub * t_vals.reshape(t_shape)
        return base - trend

    def __array__(self, dtype=None):
        # Defensive: someone calls np.asarray(view). Materialises full
        # T*X*Y*4 bytes — only use if you really mean it.
        out = self[:]
        return out if dtype is None else out.astype(dtype)


class _StepRemovedView(object):
    """Lazy (T, X, Y) view subtracting a per-frame whole-frame offset on read.

    Wraps an underlying (T, X, Y) array (a memmap or another lazy view)
    together with a length-T additive offset `offset_t`. Slicing returns
    float32 batches with `offset_t[t]` subtracted uniformly across every
    pixel of frame t. Used to remove a spatially-uniform, stimulus-triggered
    step artefact (e.g. visual-stimulus light leaking onto the red PMT)
    before signal correction, without ever materialising the full corrected
    array. Mirrors _DetrendedView and supports the same indexing patterns:

        - view[t0:t1]            -> (bt, X, Y) float32
        - view[ti]               -> (X, Y) float32
        - view[:, i_idx, j_idx]  -> (T, chunk_len) float32  (LMS path)
    """

    def __init__(self, arr, offset_t):
        offset_t = np.asarray(offset_t, dtype=np.float32).ravel()
        if offset_t.shape[0] != arr.shape[0]:
            raise ValueError(
                f"offset_t length {offset_t.shape[0]} does not match "
                f"arr time dimension {arr.shape[0]}")
        self.arr = arr
        self.offset = offset_t
        self.shape = arr.shape
        self.dtype = np.dtype(np.float32)
        self.ndim = arr.ndim
        self.nbytes = int(np.prod(arr.shape)) * self.dtype.itemsize

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        t_key = key[0]

        base = np.asarray(self.arr[key], dtype=np.float32)
        T = self.shape[0]

        if isinstance(t_key, slice):
            off = self.offset[t_key]
        elif isinstance(t_key, (int, np.integer)):
            ti = int(t_key)
            if ti < 0:
                ti += T
            return base - self.offset[ti]
        else:
            off = self.offset[np.asarray(t_key)]

        # Broadcast the per-frame scalar offset over the spatial dims of base.
        off_shape = (off.size,) + (1,) * (base.ndim - 1)
        return base - off.reshape(off_shape)

    def __array__(self, dtype=None):
        out = self[:]
        return out if dtype is None else out.astype(dtype)


def estimate_stim_step_amplitude(red_trace, t, stim_on_t, stim_off_t,
                                 edge='both', n_edge=3, n_gap=1,
                                 verbose=True):
    """Estimate a whole-frame stimulus-triggered step amplitude in red.

    Models the red channel as ``red_true + A * box(t)``, where ``box(t) = 1``
    while the visual stimulus is on screen (``stim_on_t`` → ``stim_off_t``)
    and 0 otherwise, and ``A`` is a single spatially-uniform amplitude shared
    across all trials — a "static" light-leak artefact (the stimulus is the
    same brightness every trial). ``A`` is estimated from the sharp edges of
    the whole-frame mean trace, exploiting that the step is instantaneous
    while real (kinetic) fluorescence changes are not:

        ON  edge (trial i): post_on  − pre_on   (step rises at stim onset)
        OFF edge (trial i): pre_off  − post_off (step falls at stim offset)

    Each side is the mean of ``n_edge`` frames, skipping ``n_gap`` frames at
    the transition itself (the partially-illuminated frame). ``A`` is the
    mean of the requested edges across all usable trials.

    Parameters
    ----------
    red_trace : np.ndarray, shape (T,)
        Whole-frame mean red trace at the recording sampling rate.
    t : np.ndarray, shape (T,)
        Frame timestamps (same units as stim_on_t / stim_off_t).
    stim_on_t : np.ndarray, shape (n_tr,)
        Per-trial stimulus on-screen onset times.
    stim_off_t : np.ndarray, shape (n_tr,)
        Per-trial stimulus offset times (here, the reward time of each
        trial). Trials with a non-finite or out-of-range time are skipped.
    edge : {'both', 'on', 'off'}
        Which edges to average. 'both' (default) uses ON and OFF edges. For
        reward-aligned offsets with a dopamine sensor (GRAB-DA), the OFF edge
        is contaminated by the reward-evoked transient — use 'on' there.
    n_edge : int
        Frames averaged on each side of an edge. (default: 3)
    n_gap : int
        Frames skipped at the transition, to avoid the partially-illuminated
        frame. (default: 1)
    verbose : bool
        Print the estimate and the number of edges used.

    Returns
    -------
    amp : float
        Estimated step amplitude A (in raw red units). 0.0 if no usable
        edges were found.
    info : types.SimpleNamespace
        Diagnostic bundle with .on_jumps, .off_jumps (per-trial arrays,
        NaN where unusable), .n_on, .n_off (counts used), .edge.
    """
    red = np.asarray(red_trace, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()
    T = red.shape[0]
    on_t = np.asarray(stim_on_t, dtype=np.float64).ravel()
    off_t = np.asarray(stim_off_t, dtype=np.float64).ravel()
    on_idx = np.searchsorted(t, on_t)
    off_idx = np.searchsorted(t, off_t)

    def _edge_jump(idx, t_event, rising):
        if not np.isfinite(t_event):
            return np.nan
        pre_lo = idx - n_gap - n_edge
        pre_hi = idx - n_gap
        post_lo = idx + n_gap
        post_hi = idx + n_gap + n_edge
        if pre_lo < 0 or post_hi > T or pre_hi <= pre_lo or post_hi <= post_lo:
            return np.nan
        pre = red[pre_lo:pre_hi].mean()
        post = red[post_lo:post_hi].mean()
        return (post - pre) if rising else (pre - post)

    on_jumps = np.array([_edge_jump(i, te, True)
                         for i, te in zip(on_idx, on_t)])
    off_jumps = np.array([_edge_jump(i, te, False)
                          for i, te in zip(off_idx, off_t)])

    parts = []
    if edge in ('both', 'on'):
        parts.append(on_jumps)
    if edge in ('both', 'off'):
        parts.append(off_jumps)
    all_jumps = np.concatenate(parts) if parts else np.array([np.nan])
    finite = all_jumps[np.isfinite(all_jumps)]
    amp = float(finite.mean()) if finite.size else 0.0

    n_on = int(np.isfinite(on_jumps).sum())
    n_off = int(np.isfinite(off_jumps).sum())
    info = SimpleNamespace(on_jumps=on_jumps, off_jumps=off_jumps,
                           n_on=n_on, n_off=n_off, edge=edge)

    if verbose:
        _used = (f'{n_on} on' if edge == 'on'
                 else f'{n_off} off' if edge == 'off'
                 else f'{n_on} on + {n_off} off')
        print(f'\tstim-step amplitude A = {amp:.3f} '
              f'(edges: {_used}; raw red units)')

    return amp, info


def build_stim_step_offset(t, stim_on_t, stim_off_t, amp):
    """Build the length-T additive offset vector ``amp * box(t)``.

    box(t) = 1 for t in [stim_on_t[i], stim_off_t[i]) for each usable trial i,
    0 elsewhere. Trials with non-finite times or off ≤ on are skipped.

    Parameters
    ----------
    t : np.ndarray, shape (T,)
        Frame timestamps.
    stim_on_t, stim_off_t : np.ndarray, shape (n_tr,)
        Per-trial stimulus onset / offset times.
    amp : float
        Step amplitude (from estimate_stim_step_amplitude).

    Returns
    -------
    offset : np.ndarray, shape (T,), float32
        Per-frame additive offset to subtract from the red channel.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    offset = np.zeros(t.shape[0], dtype=np.float32)
    on = np.asarray(stim_on_t, dtype=np.float64).ravel()
    off = np.asarray(stim_off_t, dtype=np.float64).ravel()
    n = min(on.shape[0], off.shape[0])
    for i in range(n):
        if not (np.isfinite(on[i]) and np.isfinite(off[i])):
            continue
        if off[i] <= on[i]:
            continue
        offset[(t >= on[i]) & (t < off[i])] = np.float32(amp)
    return offset


def estimate_pixel_step_maps(s1, s2, on_frames, off_frames, edge='both',
                             n_edge=3, n_gap=1, verbose=True):
    """Per-pixel stim-step jump maps for two channels, in raw units.

    The per-pixel analogue of :func:`estimate_stim_step_amplitude`: instead
    of one whole-frame amplitude it returns an (X, Y) map of the sharp,
    instantaneous jump each pixel undergoes at the stimulus edges, averaged
    over trials. Exploiting the same sharpness argument (the light-leak step
    is instantaneous while real fluorescence kinetics are not), only the
    ``n_edge`` frames on each side of every edge are read — a few hundred
    frames total — so this is cheap even on a memmapped stack and does not
    need a full streaming pass.

        ON  edge (frame i): post_on  − pre_on   (step rises at stim onset)
        OFF edge (frame i): pre_off  − post_off (step falls at stim offset)

    Both conventions estimate ``+leak``, so ON and OFF edges average
    together. Unlike the whole-frame estimator this resolves the leak's
    *spatial* structure (e.g. a gradient brighter toward the monitor),
    which a single scalar cannot subtract.

    Parameters
    ----------
    s1, s2 : (T, X, Y) ndarray or memmap
        Control (s1) and signal (s2) channels, in raw units. Views that
        support slicing (e.g. ``_DetrendedView``) are fine — the slow
        baseline is ~constant across an edge window and cancels in the jump.
    on_frames, off_frames : array-like of int
        Per-trial stimulus onset / offset frame indices (into the T axis).
        Non-finite or out-of-range edges are skipped.
    edge : {'both', 'on', 'off'}
        Which edges to average. 'both' (default) uses ON and OFF. 'on' is
        safest when the OFF edge sits on a reward-evoked transient in a
        functional control channel.
    n_edge : int
        Frames averaged on each side of an edge. (default: 3)
    n_gap : int
        Frames skipped at the transition (the partially-illuminated frame).
        (default: 1)
    verbose : bool

    Returns
    -------
    step_s1, step_s2 : (X, Y) float32
        Per-pixel raw step maps (0 where no usable edge was found).
    n_used : int
        Number of edges averaged into the maps.
    """
    T, X, Y = s1.shape
    on = np.asarray(on_frames, dtype=np.float64).ravel()
    off = np.asarray(off_frames, dtype=np.float64).ravel()

    edges = []  # (frame_idx, rising)
    if edge in ('both', 'on'):
        edges += [(f, True) for f in on]
    if edge in ('both', 'off'):
        edges += [(f, False) for f in off]

    acc1 = np.zeros((X, Y), dtype=np.float64)
    acc2 = np.zeros((X, Y), dtype=np.float64)
    n_used = 0
    for f, rising in edges:
        if not np.isfinite(f):
            continue
        idx = int(round(f))
        pre_lo, pre_hi = idx - n_gap - n_edge, idx - n_gap
        post_lo, post_hi = idx + n_gap, idx + n_gap + n_edge
        if pre_lo < 0 or post_hi > T or pre_hi <= pre_lo or post_hi <= post_lo:
            continue
        pre1 = np.asarray(s1[pre_lo:pre_hi], np.float32).mean(axis=0)
        post1 = np.asarray(s1[post_lo:post_hi], np.float32).mean(axis=0)
        pre2 = np.asarray(s2[pre_lo:pre_hi], np.float32).mean(axis=0)
        post2 = np.asarray(s2[post_lo:post_hi], np.float32).mean(axis=0)
        if rising:
            acc1 += post1 - pre1
            acc2 += post2 - pre2
        else:
            acc1 += pre1 - post1
            acc2 += pre2 - post2
        n_used += 1

    if n_used == 0:
        if verbose:
            print('\t\tpixel step: no usable stim edges — step maps are 0')
        return (np.zeros((X, Y), np.float32),
                np.zeros((X, Y), np.float32), 0)

    step_s1 = (acc1 / n_used).astype(np.float32)
    step_s2 = (acc2 / n_used).astype(np.float32)
    if verbose:
        print(f'\t\tpixel step: {n_used} edges; median raw jump '
              f's1={np.median(step_s1):.3f}, s2={np.median(step_s2):.3f}')
    return step_s1, step_s2, n_used


def build_stim_step_box(on_frames, off_frames, T):
    """Boolean length-T mask, True on frames inside a stimulus-on window.

    Frame-index analogue of :func:`build_stim_step_offset`'s support:
    ``box[f] = True`` for f in [on_frames[i], off_frames[i]) per usable
    trial i. Non-finite edges or off ≤ on are skipped.
    """
    on = np.asarray(on_frames, dtype=np.float64).ravel()
    off = np.asarray(off_frames, dtype=np.float64).ravel()
    box = np.zeros(int(T), dtype=bool)
    n = min(on.shape[0], off.shape[0])
    for i in range(n):
        if not (np.isfinite(on[i]) and np.isfinite(off[i])):
            continue
        a, b = int(round(on[i])), int(round(off[i]))
        a, b = max(a, 0), min(b, int(T))
        if b > a:
            box[a:b] = True
    return box


@functools.lru_cache(maxsize=32)
def _airpls_penalty(n, porder, lam):
    """Cached airPLS smoothness penalty H = λ · D.T @ D.

    The difference operator ``D`` of order ``porder`` and the penalty it
    induces depend only on the trace length ``n``, the difference order and
    ``lam`` — not on the trace values — so the same matrix is reused across
    every trace of a given length. Building it is comparatively expensive
    (repeated sparse construction), so results are memoised by
    ``(n, porder, lam)``. Both forms are treated as read-only by callers.

    Parameters
    ----------
    n : int
        Trace length.
    porder : int
        Order of the finite-difference operator in the penalty.
    lam : float
        Smoothness weight.

    Returns
    -------
    H : scipy.sparse.csc_matrix, shape (n, n)
        Sparse penalty, with sorted indices and a full main diagonal
        (guaranteed for ``porder >= 1``).
    diag_pos : np.ndarray, shape (n,)
        Index into ``H.data`` of each main-diagonal entry, so callers can
        add the IRLS weights straight onto the diagonal each iteration
        without rebuilding the sparse structure.
    """
    import scipy.sparse as sp

    # Difference operator D of order `porder`, built via repeated
    # left-multiplication by the first-difference operator to avoid sparse
    # row slicing (not supported on all storage formats).
    D = sp.eye(n, format='csc')
    for _ in range(porder):
        _m = D.shape[0]
        _D1 = sp.diags([-1.0, 1.0], [0, 1], shape=(_m - 1, _m),
                       format='csc')
        D = _D1 @ D
    H = (lam * (D.T @ D)).tocsc()
    H.sort_indices()

    # Locate each main-diagonal entry (row == col == j) within H.data.
    diag_pos = np.empty(n, dtype=np.intp)
    for j in range(n):
        _s, _e = H.indptr[j], H.indptr[j + 1]
        _rows = H.indices[_s:_e]
        diag_pos[j] = _s + np.searchsorted(_rows, j)
    return H, diag_pos


def _airpls(y, lam=1e8, porder=1, max_iter=50, tol=1e-3):
    """Adaptive iteratively reweighted penalised least squares baseline.

    Reference: Zhang, Chen & Liang, Analyst 2010 (airPLS). Estimates a
    smooth baseline z(t) of a 1-D trace y(t) that lies below the
    fluorescence transients. Used in the Martianova et al. (2019,
    J. Vis. Exp.) photometric pipeline; here applied to the spatial-
    mean control / signal traces before the linear regression step.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        1-D trace (e.g. the spatially-averaged control or signal time
        series at the recording's sampling rate).
    lam : float
        Smoothness penalty on the (porder)-th difference of z. Larger
        ⇒ smoother baseline. Sensible range 1e6 – 1e10 depending on
        the sampling rate / drift timescale. Default 1e8.
    porder : int
        Order of the difference operator used in the penalty. 1 ≈
        penalise first-difference (piecewise-linear baseline);
        2 ≈ penalise curvature. Default 1.
    max_iter : int
        Maximum IRLS iterations. Convergence is usually < 20.
    tol : float
        Convergence: stop when Σ|d⁻| < tol · Σ|y| where d⁻ are the
        negative residuals (points below the current baseline).

    Returns
    -------
    z : np.ndarray, shape (n,)
        Estimated baseline.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    _y = np.asarray(y, dtype=np.float64)
    n = _y.size
    if n < 3:
        return _y.copy()

    # Smoothness penalty H = λ · D.T @ D (see _airpls_penalty), plus the
    # positions of H's main diagonal within its CSC data. Both depend only
    # on (n, porder, lam) — identical for every trace of a given length —
    # so they are built once and cached rather than reconstructed on each
    # call, which otherwise dominates the runtime when airPLS is run over
    # many sector traces.
    H, _diag_pos = _airpls_penalty(n, porder, lam)

    w = np.ones(n, dtype=np.float64)
    z = _y.copy()
    _y_abs_sum = float(np.abs(_y).sum())
    if _y_abs_sum == 0:
        return z

    for it in range(1, max_iter + 1):
        # Solve (W + H) z = W y with W = diag(w). W + H shares H's sparsity
        # pattern (H already has a full main diagonal for porder >= 1), so
        # rather than building W and adding two sparse matrices each
        # iteration we copy H's data and add w straight onto the cached
        # diagonal positions — same result, far less sparse bookkeeping.
        _data = H.data.copy()
        _data[_diag_pos] += w
        _A = sp.csc_matrix((_data, H.indices, H.indptr), shape=(n, n))
        z = spla.spsolve(_A, w * _y)
        d = _y - z
        _neg_mask = d < 0
        _dn = float(np.abs(d[_neg_mask]).sum())
        if _dn < tol * _y_abs_sum:
            break
        # Reweight: points below the baseline (likely true baseline
        # samples) get exponentially increasing weight; points above
        # (transients) get zero weight.
        _scale = max(_dn, 1e-10)
        w_new = np.zeros(n, dtype=np.float64)
        w_new[_neg_mask] = np.exp(it * np.abs(d[_neg_mask]) / _scale)
        # Anchor endpoints so the baseline tracks edges.
        _edge = float(np.exp(it * np.abs(d).max() / _scale))
        w_new[0] = _edge
        w_new[-1] = _edge
        w = w_new
    return z


def detrend_linearly(arr, batch_size=500, verbose=False, output_path=None,
                     return_view=False):
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
        Ignored if return_view=True. (default: None)
    return_view : bool
        If True, skip pass 2 entirely and return a lazy _DetrendedView
        that applies the trend at read-time. Eliminates the disk write
        of the float32 detrended array (~4x the input size) and is the
        fastest option when the caller will consume the detrended array
        via slicing (e.g. all correct_signal methods). (default: False)

    Returns
    -------
    out : np.ndarray, np.memmap, or _DetrendedView
        Linearly detrended array, shape (T, X, Y), dtype float32. When
        return_view=True, returns a _DetrendedView wrapper around the
        original arr instead of a materialised array.
    """
    T, X, Y = arr.shape
    n_pixels = X * Y
    n_batches = (T + batch_size - 1) // batch_size

    # Closed-form OLS coefficients only depend on time indices, so build
    # them in float64 once for full precision, then drop to float32 for
    # per-batch work.
    t_arr_f32 = np.arange(T, dtype=np.float32)
    mean_t = (T - 1) / 2.0
    sum_t2 = T * (T - 1) * (2 * T - 1) / 6.0
    denom = sum_t2 - T * mean_t ** 2

    # One-worker prefetch: disk read of batch i+1 overlaps with CPU work
    # on batch i, matching the pattern used by _batched_f32.
    def _read_f32(t0, t1):
        return np.asarray(arr[t0:t1], dtype=np.float32)

    # ------------------------------------------------------------------
    # Pass 1: accumulate per-pixel sum(y) and sum(t*y).
    # Convert batch to float32 (half the memory traffic and ~2x BLAS
    # throughput vs float64); accumulate per-batch (P,) reductions into
    # float64 to preserve precision across the full T.
    # ------------------------------------------------------------------
    sum_y = np.zeros(n_pixels, dtype=np.float64)
    sum_ty = np.zeros(n_pixels, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_read_f32, 0, min(batch_size, T))
        for bi in range(n_batches):
            t0 = bi * batch_size
            t1 = min(t0 + batch_size, T)
            if bi + 1 < n_batches:
                ns = (bi + 1) * batch_size
                next_future = pool.submit(
                    _read_f32, ns, min(ns + batch_size, T))
            batch = future.result().reshape(t1 - t0, -1)
            if bi + 1 < n_batches:
                future = next_future
            # (P,) reductions in float32, promoted on accumulate.
            sum_y += batch.sum(axis=0, dtype=np.float64)
            sum_ty += (t_arr_f32[t0:t1] @ batch).astype(np.float64)
            if verbose:
                print(f'\t\t\tdetrend batch {bi + 1}/{n_batches}...',
                      end='\r')

    mean_y = sum_y / T
    b = ((sum_ty / T) - mean_t * mean_y) / (denom / T)
    a = mean_y - b * mean_t
    a = a.astype(np.float32).reshape(X, Y)
    b = b.astype(np.float32).reshape(X, Y)
    del sum_y, sum_ty, mean_y

    if return_view:
        if verbose:
            print('\t\t\tdetrend done (lazy view, no pass 2)             ')
        return _DetrendedView(arr, a, b)

    # ------------------------------------------------------------------
    # Pass 2: subtract per-pixel trend, write float32 output. Prefetch
    # the next batch while we subtract+write the current one — hides
    # disk read latency behind the (write-bound) output stage.
    # ------------------------------------------------------------------
    if output_path is not None:
        out = np.memmap(output_path, dtype=np.float32, mode='w+',
                        shape=(T, X, Y))
    else:
        out = np.empty((T, X, Y), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_read_f32, 0, min(batch_size, T))
        for bi in range(n_batches):
            t0 = bi * batch_size
            t1 = min(t0 + batch_size, T)
            if bi + 1 < n_batches:
                ns = (bi + 1) * batch_size
                next_future = pool.submit(
                    _read_f32, ns, min(ns + batch_size, T))
            batch = future.result()
            if bi + 1 < n_batches:
                future = next_future
            t_b = t_arr_f32[t0:t1]
            out[t0:t1] = batch - (
                a + b * t_b[:, np.newaxis, np.newaxis])
            if verbose:
                print(f'\t\t\tdetrend (apply) batch '
                      f'{bi + 1}/{n_batches}...', end='\r')
    if output_path is not None:
        out.flush()

    if verbose:
        print(f'\t\t\tdetrend done.                                ')
    return out


# =============================================================================
# Correction methods
# =============================================================================

def correct_full_regress_1d(
        s1, s2,
        smooth_window=10,
        airpls_lam=1e5, airpls_porder=2, airpls_max_iter=50,
        trim_initial=0,
        nn_slope=True,
        beta_loss='huber', beta_f_scale=1.0, beta_scale='mad',
        verbose=True,
        return_steps=False):
    """1-D variant of ``correct_full_regress``.

    Operates on two length-T 1-D traces (s1 = control / regressor,
    s2 = signal) and returns the corrected trace

        zdFF(t) = s2_norm(t) − β · s1_norm(t)

    in z-score units, following steps 1–5 of the Martianova et al.
    (2019) photometric pipeline. Use case: hemisphere-control 2-photon
    runs where one hemisphere expresses GRAB-5HT (s2) and the other
    a non-binding mutant (s1); the mut trace is treated as the
    isosbestic-like reference.

    Equivalent to taking the spatial-mean of the pixel-wise pipeline
    and skipping Phase B (per-voxel normalisation + per-voxel
    subtraction). The β is fit on the trim-warmup window but applied
    to the full T frames.

    Parameters
    ----------
    s1, s2 : array-like, shape (T,)
        Control and signal traces at the imaging sampling rate.
    smooth_window : int
        Moving-average window applied to both traces before airPLS.
        Default 10 (paper). Set to 1 to disable.
    airpls_lam : float
        airPLS smoothness penalty. Default 1e5.
    airpls_porder : int
        airPLS difference order. Default 2 (curvature-penalising).
    airpls_max_iter : int
        Max IRLS iterations for airPLS. Default 50.
    trim_initial : int
        Drop this many leading frames from the regression fit window.
        The full-length output still spans all T frames. Default 0.
    nn_slope : bool
        If True (default), enforce β ≥ 0.
    beta_loss : {'huber', 'cauchy', 'soft_l1', 'arctan', 'linear'}
        Loss for the coupling-slope fit. 'huber' (default) is a robust
        M-estimator fit via ``scipy.optimize.least_squares``; 'linear'
        recovers the closed-form non-negative OLS slope. Kept consistent
        with the pixel-wise ``correct_full_regress`` so the QC dF/F
        reconstruction matches.
    beta_f_scale : float
        Soft threshold of the robust loss, in normalised-trace units.
        Only used when ``beta_loss != 'linear'``. Default 1.
    beta_scale : {'mad', 'std'}
        Scale estimator for the median/scale normalisation. 'mad' (default,
        median absolute deviation × 1.4826) resists outlier frames; 'std'
        is the classical standard deviation. This scale also sets the
        z-unit of the returned corrected trace.
    verbose : bool
        Print the fitted β, σ, m diagnostics.
    return_steps : bool
        If True, additionally return a dict of every intermediate
        trace from the pipeline (raw, smoothed, airPLS baselines,
        baseline-removed, normalised, β-scaled control, corrected) plus
        the scalar fit constants. Used by the QC figure that visualises
        each correction step. Default False.

    Returns
    -------
    corrected : np.ndarray, shape (T,), dtype float32
        z-scored, β-subtracted signal trace.
    beta : float
        Fitted slope on the normalised fit traces.
    steps : dict
        Only returned when ``return_steps=True``. Keys: ``s1``, ``s2``
        (raw inputs), ``s1_sm``, ``s2_sm`` (lowpass), ``b1``, ``b2``
        (airPLS baselines), ``s1_full``, ``s2_full`` (baseline-removed),
        ``s1_norm``, ``s2_norm`` (median/std-normalised, full length),
        ``s1_norm_scaled`` (β·s1_norm — the regressed control),
        ``corrected``, and the scalars ``beta``, ``m1``, ``m2``,
        ``std1``, ``std2``. All traces are length T.
    """
    s1 = np.asarray(s1, dtype=np.float64).ravel()
    s2 = np.asarray(s2, dtype=np.float64).ravel()
    if s1.shape != s2.shape:
        raise ValueError(
            f's1 and s2 must have same length, got {s1.shape} '
            f'and {s2.shape}')
    T = s1.size
    if T < 4:
        raise ValueError(
            f'traces too short for 1D Martianova fit (T={T})')

    # 1. Moving-average lowpass
    # ----------
    s1_sm = _moving_average(s1, smooth_window)
    s2_sm = _moving_average(s2, smooth_window)

    # 2. airPLS baselines
    # ----------
    if verbose:
        print('\t\t1d-martianova: running airPLS baselines on s1(t), '
              's2(t)...')
    b1 = _airpls(s1_sm, lam=airpls_lam, porder=airpls_porder,
                 max_iter=airpls_max_iter)
    b2 = _airpls(s2_sm, lam=airpls_lam, porder=airpls_porder,
                 max_iter=airpls_max_iter)

    # 3. Trim warm-up — restricts only the regression fit window;
    # the corrected output still spans all T frames.
    # ----------
    _t0 = int(max(0, min(trim_initial, T - 2)))
    s1_fit = s1_sm[_t0:] - b1[_t0:]
    s2_fit = s2_sm[_t0:] - b2[_t0:]

    # 4. Per-channel median-subtract + std-divide normalisation
    # ----------
    s1_norm_fit, m1, std1 = _centre_scale(s1_fit, beta_scale)
    s2_norm_fit, m2, std2 = _centre_scale(s2_fit, beta_scale)

    # 5. Coupling slope on normalised fit traces: non-negative OLS
    # (beta_loss='linear') or a robust M-estimator slope otherwise.
    # ----------
    beta = _fit_norm_slope(s1_norm_fit, s2_norm_fit, nn_slope=nn_slope,
                           loss=beta_loss, f_scale=beta_f_scale)

    if verbose:
        _beta_ols_ref = _fit_norm_slope(
            s1_norm_fit, s2_norm_fit, nn_slope=nn_slope, loss='linear')
        print(f'\t\t\tβ = {beta:.6g} (OLS ref {_beta_ols_ref:.6g}, '
              f'loss={beta_loss}, nn={"on" if nn_slope else "off"})')
        print(f'\t\t\tσ1 = {std1:.4g}, σ2 = {std2:.4g}, '
              f'm1 = {m1:.4g}, m2 = {m2:.4g}')

    # Apply over the full T-frame trace: normalise using fit-window
    # stats, then subtract β·s1_norm. Matches Phase B of the pixel-
    # wise version reduced to a single 1-D pixel.
    # ----------
    s1_full = s1_sm - b1
    s2_full = s2_sm - b2
    s1_full_norm = (s1_full - m1) / std1
    s2_full_norm = (s2_full - m2) / std2
    corrected = s2_full_norm - beta * s1_full_norm

    if return_steps:
        steps = {
            's1': s1, 's2': s2,
            's1_sm': s1_sm, 's2_sm': s2_sm,
            'b1': b1, 'b2': b2,
            's1_full': s1_full, 's2_full': s2_full,
            's1_norm': s1_full_norm, 's2_norm': s2_full_norm,
            's1_norm_scaled': beta * s1_full_norm,
            'corrected': corrected,
            'beta': float(beta),
            'm1': m1, 'm2': m2, 'std1': std1, 'std2': std2}
        return corrected.astype(np.float32), float(beta), steps

    return corrected.astype(np.float32), float(beta)


def _pool_moments(pix_mean, pix_var, level, n_sec):
    """Pool per-pixel session mean & variance to a coarser granularity.

    Sets the granularity of the dF/F normalisation (centring m and scale
    σ) used by ``correct_full_regress``: ``'pixel'`` keeps per-pixel
    stats, ``'sector'`` pools within n_sec×n_sec blocks, ``'frame'``
    pools over the whole FOV. Variance is pooled by the law of total
    variance (within-pixel temporal variance + between-pixel variance of
    the means), so the result is the true mean / variance of the
    region's pixel-time samples and reduces exactly to the per-pixel
    values when ``level='pixel'``.

    Parameters
    ----------
    pix_mean, pix_var : np.ndarray, shape (X, Y)
        Per-pixel session mean and temporal variance.
    level : str
        'pixel', 'sector', or 'frame'.
    n_sec : int
        Blocks per axis for the 'sector' level (row-major, matching
        ``QCMixin._compute_sectors``). Ignored for the other levels.

    Returns
    -------
    mean, var : np.ndarray, shape (X, Y)
        Pooled mean and variance broadcast back to the pixel grid.
    """
    pix_mean = np.asarray(pix_mean, dtype=np.float64)
    pix_var = np.asarray(pix_var, dtype=np.float64)
    X, Y = pix_mean.shape
    if level == 'pixel':
        return pix_mean.copy(), pix_var.copy()
    if level == 'frame':
        _m = float(pix_mean.mean())
        _v = float(pix_var.mean() + pix_mean.var())
        return np.full((X, Y), _m), np.full((X, Y), _v)
    if level != 'sector':
        raise ValueError(
            f"f0_level must be 'pixel', 'sector' or 'frame', "
            f"got {level!r}")
    _ns = max(1, int(n_sec))
    _by = max(1, X // _ns)
    _bx = max(1, Y // _ns)
    _rows = np.minimum(np.arange(X) // _by, _ns - 1)
    _cols = np.minimum(np.arange(Y) // _bx, _ns - 1)
    _blk = (_rows[:, None] * _ns + _cols[None, :]).ravel()
    _nb = _ns * _ns
    _cnt = np.maximum(np.bincount(_blk, minlength=_nb).astype(np.float64),
                      1.0)
    _sm = np.bincount(_blk, weights=pix_mean.ravel(), minlength=_nb) / _cnt
    _smsq = np.bincount(_blk, weights=pix_mean.ravel() ** 2,
                        minlength=_nb) / _cnt
    _sv = np.bincount(_blk, weights=pix_var.ravel(), minlength=_nb) / _cnt
    _blk_mean = _sm
    _blk_var = _sv + (_smsq - _sm ** 2)
    return _blk_mean[_blk].reshape(X, Y), _blk_var[_blk].reshape(X, Y)


#: Gaussian smoothing applied to the mean image before the expression-gate
#: fit, as a fraction of the FOV's smaller axis. 0.5% resolves to 2 px on a
#: 512x512 recording (the value this gate was calibrated on) and to 0 px on
#: a 64x64 sim, whose frame-averaged mean image is already noise-free and
#: whose cell-sized features a 2 px blur would erase.
_GATE_SMOOTH_FOV_FRAC = 0.005


def _gate_smooth_px(shape, smooth_px='auto'):
    """Resolve ``smooth_px='auto'`` to a pixel sigma for this FOV.

    Smoothing before the GMM exists to stop per-pixel noise scattering
    pixels across the cut, so the right sigma tracks the detector's noise
    correlation length — which scales with the FOV in pixels, not with the
    image's semantic content. A fixed sigma is silently FOV-dependent: 2 px
    on 512x512 blurs the same fraction of frame as 16 px on 64x64.
    """
    if smooth_px != 'auto':
        return float(smooth_px)
    return float(int(_GATE_SMOOTH_FOV_FRAC * min(shape[0], shape[1])))


def _expression_gate(mean_img, method='gmm', log=True, smooth_px='auto',
                     min_sep=1.5, gate_pct=50.0, label='', verbose=True):
    """Select pixels that express a sensor, from the session mean image.

    Fits a 2-component Gaussian mixture to the per-pixel intensities and
    keeps the higher-intensity component — a data-adaptive replacement
    for a hand-set brightness/SNR threshold.

    Selects the expressing *field* (a contiguous tissue mask), not
    individual expressing cells.

    Parameters
    ----------
    mean_img : (X, Y) ndarray
        Per-pixel session mean fluorescence.
    method : {'gmm', 'percentile', 'none'}
        'gmm' (default) fits the mixture; 'percentile' cuts at
        ``gate_pct`` of the (transformed) intensities; 'none' keeps
        every pixel.
    log : bool
        Fit on log intensity. Default True, and it matters: 2-photon
        intensity distributions are right-skewed, so a 2-Gaussian fit on
        the raw scale is misspecified. Measured on a real sensor channel,
        raw separates the two modes by only 1.24 σ (failing ``min_sep``)
        against 2.31 σ on log.
    smooth_px : float or 'auto'
        Gaussian sigma (pixels) applied to the mean image before the fit.
        Per-pixel shot noise otherwise scatters pixels across the cut and
        speckles the mask. 'auto' (default) uses 0.5% of the FOV's smaller
        axis, rounded down — 2 px at 512x512, 0 px at 64x64. 0 disables.
    min_sep : float
        Minimum mode separation |μ_hi − μ_lo| / max(σ) for the GMM cut to
        be trusted. Below this (or if a 1-component fit has lower BIC)
        the split is not real and ``gate_pct`` is used instead.
    gate_pct : float
        Percentile cut for method='percentile' and for the GMM fallback.
    label : str
        Channel name for messages.
    verbose : bool
        Print the fitted cut and separation.

    Returns
    -------
    mask : (X, Y) bool
        True where the pixel passes the gate.
    info : dict
        ``cut`` (on the raw intensity scale), ``sep``, ``bic1``, ``bic2``,
        ``fallback`` (bool — did the guard fire), ``method``, and
        ``smooth_px`` (the *resolved* sigma, after 'auto').
    """
    if method not in ('gmm', 'percentile', 'none'):
        raise ValueError(
            f"pixel_gate must be 'gmm', 'percentile' or 'none', "
            f"got {method!r}")

    _a = np.asarray(mean_img, dtype=np.float32)
    if method == 'none':
        return (np.ones(_a.shape, dtype=bool),
                {'cut': -np.inf, 'sep': np.nan, 'bic1': np.nan,
                 'bic2': np.nan, 'fallback': False, 'method': 'none'})

    from scipy.ndimage import gaussian_filter

    smooth_px = _gate_smooth_px(_a.shape, smooth_px)
    if smooth_px:
        _a = gaussian_filter(_a, smooth_px, mode='reflect')
    # Fit and threshold on the SAME transformed scale — exponentiating
    # the cut back and comparing against the unsmoothed raw image would
    # decouple the mask from the fit.
    _v = np.log(np.maximum(_a, 1.0)) if log else _a
    _flat = _v.ravel()

    _sep = np.nan
    _bic1 = _bic2 = np.nan
    _fallback = False
    if method == 'gmm':
        from sklearn.mixture import GaussianMixture
        # Subsample the fit for large FOVs; the cut is a 1-D quantity and
        # 50k samples pin it down far more precisely than the mode
        # separation itself is meaningful.
        _fit = _flat
        if _fit.size > 50000:
            _rng = np.random.default_rng(0)
            _fit = _rng.choice(_fit, 50000, replace=False)
        _fit = _fit[:, None]
        # k=1 is the guard reference. k=3 is deliberately not fitted: BIC
        # preferring k=3 does not invalidate the k=2 low/high split, it
        # only means there are >2 tissue compartments.
        _bic1 = float(GaussianMixture(
            1, random_state=0, n_init=3).fit(_fit).bic(_fit))
        _g = GaussianMixture(2, random_state=0, n_init=3).fit(_fit)
        _bic2 = float(_g.bic(_fit))
        _mu = np.sort(_g.means_.ravel())
        _sd = np.sqrt(_g.covariances_.ravel())
        _sep = float((_mu[1] - _mu[0]) / max(float(_sd.max()), 1e-12))
        if (_bic2 < _bic1) and (_sep >= float(min_sep)):
            _grid = np.linspace(_flat.min(), _flat.max(), 500)[:, None]
            _hi = int(np.argmax(_g.means_.ravel()))
            _p_hi = _g.predict_proba(_grid)[:, _hi]
            # Take the first grid point at which the high component wins,
            # NOT the point closest to p=0.5. Well-separated modes
            # saturate predict_proba to exactly 0/1 at every grid point
            # (no point is near 0.5), so argmin(|p - 0.5|) ties across the
            # whole grid and returns index 0 — a cut at the data minimum
            # that keeps every pixel. The failure grows *more* likely the
            # cleaner the split is.
            _above = np.flatnonzero(_p_hi >= 0.5)
            if _above.size:
                _i = int(_above[0])
                if _i == 0:
                    _cut = float(_grid[0][0])
                else:
                    # Linearly interpolate the crossing between the
                    # bracketing grid points, so the cut does not carry a
                    # half-grid-step bias. Under saturation (p: 0 -> 1)
                    # this lands mid-gap, which is where any cut is
                    # equivalent anyway.
                    _g0, _g1 = float(_grid[_i - 1][0]), float(_grid[_i][0])
                    _p0, _p1 = float(_p_hi[_i - 1]), float(_p_hi[_i])
                    _cut = (_g0 + (0.5 - _p0) * (_g1 - _g0) / (_p1 - _p0)
                            if _p1 > _p0 else _g1)
            else:
                _fallback = True
                _cut = float(np.percentile(_flat, gate_pct))
                warnings.warn(
                    f"expression gate{f' [{label}]' if label else ''}: the "
                    f"fitted high component never reaches p>=0.5 on the "
                    f"intensity grid; falling back to the {gate_pct:g}th-"
                    f"percentile cut.", RuntimeWarning, stacklevel=2)
        else:
            _fallback = True
            _cut = float(np.percentile(_flat, gate_pct))
            warnings.warn(
                f"expression gate{f' [{label}]' if label else ''}: no "
                f"reliable bimodal split (separation {_sep:.2f} < "
                f"{min_sep}, BIC k=1 {_bic1:.0f} vs k=2 {_bic2:.0f}); "
                f"falling back to the {gate_pct:g}th-percentile cut. "
                f"Inspect the mean-image histogram and set pixel_gate="
                f"'percentile' with an explicit gate_pct if this is "
                f"expected.", RuntimeWarning, stacklevel=2)
    else:
        _cut = float(np.percentile(_flat, gate_pct))

    mask = _v >= _cut
    cut_raw = float(np.exp(_cut)) if log else _cut
    if verbose:
        _tag = 'percentile' if (method == 'percentile' or _fallback) \
            else 'gmm'
        print(f'\t\tgate{f" [{label}]" if label else ""} ({_tag}): '
              f'cut={cut_raw:.1f}, keeps {mask.mean()*100:.1f}%'
              + (f', sep={_sep:.2f}σ' if np.isfinite(_sep) else ''))
    return mask, {'cut': cut_raw, 'sep': _sep, 'bic1': _bic1,
                  'bic2': _bic2, 'fallback': _fallback,
                  'method': 'percentile' if _fallback else method,
                  'smooth_px': smooth_px}


def _segment_somata(mean_img, min_darkness=0.10, diam_px=(4.0, 20.0),
                    bg_sigma=20.0, eps=1e-6, label='', verbose=True):
    """Detect cell bodies as negative-contrast dark holes in a mean image.

    Membrane-bound sensors (e.g. GRAB) express in the neuropil but not
    inside somata, so cell bodies read *darker* than the surrounding
    neuropil. This segments them by their local intensity dip against a
    large-scale background, splitting touching cells by watershed
    (skimage if importable, else connected components on the seed mask —
    the holes are sparse, ~13% coverage, so components alone recover most).

    Parameters
    ----------
    mean_img : (X, Y) ndarray
        Session mean image of the signal channel.
    min_darkness : float
        Minimum relative dip ``(bg - F) / bg`` below the local background
        for a pixel to seed a soma. Default 0.10.
    diam_px : (float, float)
        Keep detections whose area-equivalent diameter
        ``2·sqrt(area/pi)`` (pixels) falls in this band. Default (4, 20).
    bg_sigma : float
        Gaussian sigma (px) for the large-scale local background against
        which the dip is measured. Default 20.
    eps : float
        Divide floor for the darkness ratio.
    label : str
        Tag for verbose output.
    verbose : bool
        Print a one-line detection summary.

    Returns
    -------
    labels : (X, Y) int32
        Instance map, 0 = neuropil/background, k = the k-th soma.
    info : dict
        ``n_cells``, ``diam_px`` (per-kept-cell array), ``coverage``
        (soma area fraction), ``used_watershed`` (bool).
    """
    from scipy import ndimage as ndi

    _a = ndi.gaussian_filter(mean_img.astype(np.float32), 1.0)
    _bg = ndi.gaussian_filter(_a, float(bg_sigma))
    _dark = (_bg - _a) / (_bg + eps)
    _seed = _dark > float(min_darkness)
    _seed = ndi.binary_opening(_seed, iterations=1)
    _seed = ndi.binary_fill_holes(_seed)

    _used_ws = False
    _labels = None
    if _seed.any():
        try:
            from skimage.feature import peak_local_max
            from skimage.segmentation import watershed
            _dist = ndi.distance_transform_edt(_seed)
            _min_d = max(1, int(round(0.5 * float(diam_px[0]))))
            _peaks = peak_local_max(
                _dist, min_distance=_min_d, labels=_seed)
            if len(_peaks):
                _mk = np.zeros(_seed.shape, dtype=np.int32)
                _mk[tuple(_peaks.T)] = np.arange(1, len(_peaks) + 1)
                _labels = watershed(-_dist, _mk, mask=_seed).astype(np.int32)
                _used_ws = True
        except ImportError:
            _labels = None
        if _labels is None:
            _labels, _ = ndi.label(_seed)
            _labels = _labels.astype(np.int32)
    else:
        _labels = np.zeros(_seed.shape, dtype=np.int32)

    # Size filter by area-equivalent diameter; relabel survivors 1..K.
    _n = int(_labels.max())
    if _n:
        _area = np.bincount(_labels.ravel(), minlength=_n + 1)[1:]
        _diam = 2.0 * np.sqrt(np.maximum(_area, 0) / np.pi)
        _keep = (_diam >= float(diam_px[0])) & (_diam <= float(diam_px[1]))
        _remap = np.zeros(_n + 1, dtype=np.int32)
        _remap[1:][_keep] = np.arange(1, int(_keep.sum()) + 1)
        _labels = _remap[_labels]
        _kept_diam = _diam[_keep]
    else:
        _kept_diam = np.zeros(0, dtype=np.float64)

    _k = int(_labels.max())
    _cov = float((_labels > 0).mean())
    if verbose:
        _med = float(np.median(_kept_diam)) if _k else float('nan')
        _how = 'watershed' if _used_ws else 'connected-comp'
        print(f'\t\tsomata{f" [{label}]" if label else ""}: {_k} cells, '
              f'median diam {_med:.1f}px, coverage {_cov*100:.1f}% ({_how})')
    return _labels, {'n_cells': _k, 'diam_px': _kept_diam,
                     'coverage': _cov, 'used_watershed': _used_ws}


def _donut_rings(labels, mask_out, ring_width_px):
    """Grow a donut annulus of the given width outward from each soma.

    Every non-soma pixel is assigned to its nearest soma (a Voronoi
    partition by Euclidean distance), then kept if it lies within
    ``ring_width_px`` of that soma and inside ``mask_out``. This gives one
    contiguous ring ROI per cell, with no ring pixel shared between two
    cells.

    Parameters
    ----------
    labels : (X, Y) int32
        Soma instance map from :func:`_segment_somata`.
    mask_out : (X, Y) bool
        Pixels where corrected output exists (rings are clipped to this).
    ring_width_px : int
        Annulus width in pixels.

    Returns
    -------
    ring_labels : (X, Y) int32
        Per-pixel cell assignment of the donut rings (0 = not a ring).
    """
    from scipy import ndimage as ndi

    soma = labels > 0
    dist, inds = ndi.distance_transform_edt(~soma, return_indices=True)
    nearest = labels[inds[0], inds[1]]
    ring = (~soma) & (dist <= float(ring_width_px)) & mask_out
    return np.where(ring, nearest, 0).astype(np.int32)


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
        output_path=None,
        fit_backend='auto',
        fit_temporal_stride=4):
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
    fit_backend : str
        Backend for the PCA fit step (pass 2). One of:
        - 'auto'   (default): use 'torch' if pytorch is importable, else
                   fall back to 'sklearn'.
        - 'torch': use torch.pca_lowrank (randomised SVD on CPU). ~20-30x
                   faster than IncrementalPCA at production scale, but
                   requires the (T_fit, n_total_fit) fit matrix to fit in
                   RAM. Use fit_temporal_stride > 1 to subsample frames
                   for the fit if RAM is tight.
        - 'sklearn': sklearn.IncrementalPCA (streaming; slower but constant
                   memory).
    fit_temporal_stride : int
        Temporal subsampling stride for the fit matrix when using the
        torch backend. Only every Nth frame contributes to the PCA fit;
        the basis is then applied to every frame in pass 3.

        The fit matrix size in RAM is roughly
            (single-channel TIFF bytes) / fit_temporal_stride
        (input is int16, fit matrix is float32 but uses 1/spatial_subsample
        of the pixels and stacks both channels). For a ~20 GB single-
        channel recording on a 16 GB M1 Pro, stride 4 gives a ~5 GB fit
        matrix — leaves enough headroom for Python, intermediate arrays,
        and pca_lowrank's internal working set. Drop to 3 if you have
        more RAM free; raise to 6+ if you see swap.

        Default 4.

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

    # Resolve fit backend
    _torch = None
    if fit_backend in ('auto', 'torch'):
        try:
            import torch as _torch
        except ImportError:
            if fit_backend == 'torch':
                raise ImportError(
                    "fit_backend='torch' requires pytorch; install it or "
                    "pass fit_backend='sklearn'")
            _torch = None
    _use_torch = _torch is not None
    if fit_backend == 'sklearn':
        _use_torch = False
    if fit_temporal_stride < 1:
        raise ValueError(
            f"fit_temporal_stride must be >= 1, got {fit_temporal_stride}")

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
    # Pass 2: fit PCA (torch.pca_lowrank or IncrementalPCA)
    # ------------------------------------------------------------------
    combined_buf = np.empty((batch_size, n_fit_pixels * 2), dtype=np.float32)
    if _use_torch:
        # Randomised SVD on the centered fit matrix held in RAM. Much
        # faster than sklearn's IncrementalPCA (~20-30x at production
        # scale). Optional temporal subsampling keeps RAM in check on
        # long recordings.
        T_fit = (T + fit_temporal_stride - 1) // fit_temporal_stride
        if verbose:
            _ram_gb = T_fit * n_total_fit * 4 / 1e9
            print(f'\t\tpass 2/5: fitting PCA [torch.pca_lowrank] '
                  f'({n_components} components, T_fit={T_fit}, '
                  f'~{_ram_gb:.2f} GB)...')
        fit_mat = np.empty((T_fit, n_total_fit), dtype=np.float32)
        write_idx = 0
        for batch_idx in range(n_batches):
            t0 = batch_idx * batch_size
            t1 = min(t0 + batch_size, T)
            bt = t1 - t0
            if verbose:
                pct = (batch_idx + 1) / n_batches * 100
                print(f'\t\t\tbatch {batch_idx+1}/{n_batches} '
                      f'({pct:.1f}%)...', end='\r')
            s1_b = np.asarray(s1[t0:t1], dtype=np.float32).reshape(bt, -1)
            s2_b = np.asarray(s2[t0:t1], dtype=np.float32).reshape(bt, -1)
            combined_buf[:bt, :n_fit_pixels] = (
                s1_b[:, fit_pixel_indices] - s1_mean)
            combined_buf[:bt, n_fit_pixels:] = (
                s2_b[:, fit_pixel_indices] - s2_mean)
            # Select strided frames within this batch that map to fit rows
            _local = np.arange(bt)
            _global = t0 + _local
            _sel = _local[_global % fit_temporal_stride == 0]
            if _sel.size == 0:
                continue
            _n_sel = _sel.size
            fit_mat[write_idx:write_idx + _n_sel] = combined_buf[_sel]
            write_idx += _n_sel
        fit_mat = fit_mat[:write_idx]
        if verbose:
            print(f'\t\t\tbatch {n_batches}/{n_batches} (100.0%)...done    ')
            print(f'\t\t\trunning torch.pca_lowrank '
                  f'(q={n_components + 10}, niter=4)...')
        # Data is already mean-centered, so disable internal centering.
        _q = min(n_components + 10, min(fit_mat.shape))
        _A = _torch.from_numpy(fit_mat)
        _U, _S, _V = _torch.pca_lowrank(
            _A, q=_q, center=False, niter=4)
        components = _V[:, :n_components].T.contiguous().numpy()
        _S2 = (_S ** 2).numpy().astype(np.float64)
        explained_variance = (_S2[:n_components] / _S2.sum()).astype(
            np.float32)
        del fit_mat, _A, _U, _S, _V, _S2
    else:
        if verbose:
            print(f'\t\tpass 2/5: fitting PCA [IncrementalPCA] '
                  f'({n_components} components, {n_batches} batches)...')
        ipca = IncrementalPCA(n_components=n_components)
        batches_fit = 0
        for batch_idx in range(n_batches):
            t0 = batch_idx * batch_size
            t1 = min(t0 + batch_size, T)
            bt = t1 - t0
            if bt < n_components:
                break
            if verbose:
                pct = (batch_idx + 1) / n_batches * 100
                print(f'\t\t\tbatch {batch_idx+1}/{n_batches} '
                      f'({pct:.1f}%)...', end='\r')
            s1_b = np.asarray(s1[t0:t1], dtype=np.float32).reshape(bt, -1)
            s2_b = np.asarray(s2[t0:t1], dtype=np.float32).reshape(bt, -1)
            combined_buf[:bt, :n_fit_pixels] = (
                s1_b[:, fit_pixel_indices] - s1_mean)
            combined_buf[:bt, n_fit_pixels:] = (
                s2_b[:, fit_pixel_indices] - s2_mean)
            ipca.partial_fit(combined_buf[:bt])
            batches_fit += 1
        if verbose:
            print(f'\t\t\t{batches_fit}/{n_batches} batches fitted '
                  f'(100.0%)...done')
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
