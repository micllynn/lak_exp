"""Synthetic dual-colour imaging simulator for signal-correction testing.

Generates a ground-truth two-photon-like dataset with known components so the
artefact-correction algorithms in :mod:`signal_correction` can be scored
against the true neural signal they are meant to recover:

    a : spatially varying real signal (sparse calcium-like transients)
    b : spatially varying haemodynamics (smooth, multi-scale, slow)
    c : 'signal' channel        = base + a + phi_s(b) + noise
    d : 'static control' channel = base +     phi_c(b) + noise

``phi_s`` and ``phi_c`` are mild, channel-specific saturating
nonlinearities, so the control channel is *not* a perfectly linear
predictor of the signal channel's haemodynamics — the regime real
dual-colour correction must cope with. The haemodynamic field also has a
spatially varying coupling gain in each channel.

The :class:`DualColourSim` class builds and stores all components; the
module-level :func:`run_regress` runs each correction method on the
simulated channels and reports how well each recovers ``a``.

Example
-------
>>> from lak_exp.sims import run_regress
>>> res = run_regress()        # builds a default sim, scores every method
"""

import os, re, inspect, hashlib, warnings
import concurrent.futures as cf
from types import SimpleNamespace

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from .utils import nanmean, fmt_kwarg_val

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from . import signal_correction as sc
except ImportError:
    from lak_exp import signal_correction as sc


__all__ = ['DualColourSim', 'run_regress', 'sweep_regress',
           'print_summary_fig', 'save_sim_videos',
           'demo_infer_field_bayesnf']


# =============================================================================
# Synthetic dataset
# =============================================================================

class DualColourSim(object):
    """Generate a synthetic dual-colour recording with known ground truth.

    Builds a real signal field ``a``, a haemodynamic field ``b``, two
    pixel-level fluorophore expression maps, and the two observed channels
    (signal ``c`` and static control ``d``). All stacks have shape
    ``(n_t, n_x, n_y)``; the expression maps are ``(n_x, n_y)``.

    Brightness model (per pixel)::

        F0_s(x,y) = f0_sig  * expr_sig(x,y)     resting signal fluorophore
        F0_c(x,y) = f0_ctrl * expr_ctrl(x,y)    resting control fluorophore
        c(t,x,y) = bg_sig  + F0_s·(1 + sig_strength·a) ·(1 + haemo_strength·phi_s(b)) + noise
        d(t,x,y) = bg_ctrl + F0_c                ·(1 + haemo_strength·phi_c(b)) + noise

    Haemodynamics enter as a multiplicative modulation of the emitted
    light, so the haemo fluctuation a pixel actually shows scales with how
    much fluorophore it expresses. A signal-only pixel (``expr_ctrl≈0``)
    therefore has **no haemodynamic readout in the control channel** —
    exactly the case where per-pixel regression has nothing local to
    regress against and spatial averaging must borrow from neighbours.

    Expression layout (see ``_make_expression``) mirrors a real dual-colour
    FOV: small **cell bodies** (somata) shared by both reporters, embedded
    in a uniformly bright **neuropil**. The **sensor** (e.g. GRAB) expresses
    in the neuropil only — never in cell bodies — while the **static**
    reporter always fills the cell bodies plus an ``overlap`` fraction of
    the surrounding neuropil.

    Sign convention: ``b`` is the *brightness* effect of the haemodynamics
    rather than blood volume, so it modulates both channels **positively**
    (``b`` up -> both channels up) and ``corr(channel, b)`` is positive.
    Physically the absorbing quantity is ``-b``; folding that minus into
    the definition of ``b`` is an exact relabelling of the model, since
    ``phi`` is an odd function and so ``+haemo_strength·phi(b)`` reproduces
    the absorbing model evaluated at ``-b``. Correction difficulty is
    unchanged: every score in :func:`run_regress` is a ``|corr|``, and the
    regressors fit a free sign anyway.

    Parameters
    ----------
    n_t : int
        Number of time frames.
    n_x, n_y : int
        Spatial dimensions (pixels).
    n_haemo_modes : int
        Number of spatial haemodynamic modes (mode 0 is near-global; the
        rest are localised), giving spatially varying haemodynamics.
    n_sources : int
        Number of neural sources (Gaussian footprints with calcium-like
        transients) making up the real signal.
    haemo_tau : float
        Temporal smoothing of the haemodynamic traces, in frames. This
        sets the haemo timescale, which must sit in the band the
        correction pipeline targets — *slower* than ``smooth_window`` and
        *faster* than the airPLS baseline (``airpls_lam``). Haemo much
        slower than this is absorbed by the per-sector baseline instead of
        the regression, so it would not exercise the regressors. Default 6.
    haemo_strength : float
        Fractional brightness modulation depth of the haemodynamics (e.g.
        0.15 = ±15% brightness swings on the resting fluorescence).
    sig_strength : float
        Fractional ΔF/F amplitude of a calcium transient at a source
        centre (on top of the resting fluorescence).
    f0_sig, f0_ctrl : float
        Resting (activity-independent) brightness of a fully expressing
        pixel for the signal / control fluorophore.
    bg_sig, bg_ctrl : float
        Background offset (autofluorescence / dark level) of each channel.
    noise_sig, noise_ctrl : float
        Per-pixel Gaussian read-noise std (brightness units).
    nonlin_sig, nonlin_ctrl : float
        Saturating-nonlinearity strength for each channel's haemodynamic
        transfer (0 = linear; larger = more compressive). Different values
        make the control a slightly nonlinear predictor of the signal
        channel's haemodynamics.
    overlap : float
        Neuropil overlap, in ``[0, 1]``: the fraction of the neuropil
        (the area *around* cell bodies) where the **static** fluorophore is
        *also* expressed, on top of the cell bodies it always fills. 0 →
        static only in cell bodies (maximally distinct from the sensor);
        1 → static fills the whole neuropil too (co-expressed with the
        sensor everywhere). The sensor never enters cell bodies regardless.
    frac_sig : float
        Sensor neuropil coverage, in ``(0, 1]``: the fraction of the FOV
        that is neuropil (and so expresses the sensor). The remaining
        ``(1 - frac_sig)`` is small **cell bodies** (somata) — shared by
        both reporters — which the sensor is excluded from and the static
        always fills. Higher = fewer / sparser cells (default 0.9 = ~10%
        cell-body coverage).
    frac_ctrl : float
        Retained for API / sweep compatibility; the static reporter's
        spatial support is now set by the shared cell bodies plus
        ``overlap`` (the neuropil fraction), so this is currently unused.
    cell_size : float or None
        Characteristic radius (px) of the cell bodies. Defaults to
        ``max(n_x, n_y) / 40`` (small nucleus-sized somata).
    seed : int or None
        RNG seed. Defaults to ``None`` (a fresh random draw each run); pass
        an int for reproducibility.
    """

    def __init__(self, n_t=2500, n_x=64, n_y=64,
                 n_haemo_modes=3, n_sources=18,
                 haemo_tau=6.0, haemo_strength=0.25, sig_strength=0.6,
                 f0_sig=100.0, f0_ctrl=100.0,
                 bg_sig=20.0, bg_ctrl=20.0,
                 noise_sig=3.0, noise_ctrl=5.0,
                 nonlin_sig=0, nonlin_ctrl=0,
                 overlap=0.3, frac_sig=0.9, frac_ctrl=0.9,
                 cell_size=None, seed=None):
        self.n_t = int(n_t)
        self.n_x = int(n_x)
        self.n_y = int(n_y)
        self.n_haemo_modes = int(n_haemo_modes)
        self.n_sources = int(n_sources)
        self.haemo_tau = float(haemo_tau)
        self.haemo_strength = float(haemo_strength)
        self.sig_strength = float(sig_strength)
        self.f0_sig = float(f0_sig)
        self.f0_ctrl = float(f0_ctrl)
        self.bg_sig = float(bg_sig)
        self.bg_ctrl = float(bg_ctrl)
        self.noise_sig = float(noise_sig)
        self.noise_ctrl = float(noise_ctrl)
        self.nonlin_sig = float(nonlin_sig)
        self.nonlin_ctrl = float(nonlin_ctrl)
        self.overlap = float(np.clip(overlap, 0.0, 1.0))
        self.frac_sig = float(np.clip(frac_sig, 1e-3, 1.0))
        self.frac_ctrl = float(np.clip(frac_ctrl, 1e-3, 1.0))
        self.cell_size = (float(cell_size) if cell_size is not None
                          else max(self.n_x, self.n_y) / 80.0)
        self.rng = np.random.default_rng(seed)

        self.t = np.arange(self.n_t)
        self.generate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self):
        """Build every component and assemble the two observed channels."""
        self._make_haemo()
        self._make_expression()
        self._make_signal()
        self._assemble_channels()
        return self

    def summary(self):
        """Return a SimpleNamespace of dataset / expression statistics."""
        _expr_floor = 0.05
        _sig_px = self.expr_sig > _expr_floor
        _ctrl_px = self.expr_ctrl > _expr_floor
        _both = _sig_px & _ctrl_px
        # Measured neuropil overlap: fraction of neuropil (non-cell) pixels
        # that also carry static expression — the realised value of the
        # `overlap` parameter.
        _npil = self.cells < 0.5
        _meas_overlap = float(
            (_ctrl_px & _npil).sum() / max(_npil.sum(), 1))
        return SimpleNamespace(
            ctrl_noise=self.noise_ctrl,
            sig_noise=self.noise_sig,
            cell_frac=float((self.cells > 0.5).mean()),
            frac_sig_expressing=float(_sig_px.mean()),
            frac_ctrl_expressing=float(_ctrl_px.mean()),
            measured_overlap=_meas_overlap,
            frac_signal_pixels_with_control=float(
                _both.sum() / max(_sig_px.sum(), 1)),
            n_signal_pixels=int(self.sig_mask.sum()))

    # ------------------------------------------------------------------
    # Component builders
    # ------------------------------------------------------------------
    def _smooth_spatial(self, smooth_px):
        """A smooth zero-mean random spatial field, shape (n_x, n_y)."""
        _f = self.rng.standard_normal((self.n_x, self.n_y))
        _f = gaussian_filter(_f, sigma=smooth_px, mode='reflect')
        _f -= _f.mean()
        _sd = _f.std()
        return _f / _sd if _sd > 1e-9 else _f

    def _smooth_temporal(self, smooth_t):
        """A smooth zero-mean random temporal trace, length n_t."""
        _x = self.rng.standard_normal(self.n_t)
        _x = gaussian_filter1d(_x, sigma=smooth_t, mode='reflect')
        _x -= _x.mean()
        _sd = _x.std()
        return _x / _sd if _sd > 1e-9 else _x

    def _make_haemo(self):
        """Spatially varying, slow haemodynamic field b(t, x, y).

        Sum of ``n_haemo_modes`` spatiotemporal modes: mode 0 is broad
        (near-global, the dominant shared component Stage 1 should remove);
        the remaining modes are progressively more localised (the
        spatially precise components the sector cascade should remove).
        """
        modes_t = np.empty((self.n_haemo_modes, self.n_t))
        modes_s = np.empty((self.n_haemo_modes, self.n_x, self.n_y))
        # Mode 0: broad spatial map, dominant weight. All modes share the
        # same temporal band (``haemo_tau``) so the haemodynamics fall where
        # the correction's regression operates; finer modes vary in space.
        weights = np.empty(self.n_haemo_modes)
        for k in range(self.n_haemo_modes):
            # Coarser modes are smoother in space and carry more power.
            _smooth_px = max(self.n_x, self.n_y) / (2.0 + 3.0 * k)
            modes_s[k] = self._smooth_spatial(_smooth_px)
            modes_t[k] = self._smooth_temporal(self.haemo_tau)
            weights[k] = 1.0 / (1.0 + k)
        # Mode 0 spatial map shifted positive so it reads as a broad,
        # mostly-coherent global drive.
        modes_s[0] = np.abs(modes_s[0]) + 1.0

        latent = np.einsum('kt,kxy->txy', modes_t * weights[:, None], modes_s)
        # Unit-std latent field; the observed brightness modulation depth is
        # set per channel by haemo_strength · expression in _assemble_channels.
        latent /= (latent.std() + 1e-9)

        self.haemo = latent.astype(np.float32)      # b (ground-truth haemo)
        self._haemo_modes_t = modes_t
        self._haemo_modes_s = modes_s

    def _make_cells(self):
        """Binary small-cell-body (somata) mask covering ~ (1 - frac_sig).

        Cell bodies are the top ``(1 - frac_sig)`` quantile of a fine-scale
        spatial field (``cell_size`` smoothing) — small, roughly round
        somata shared by both reporters: the sensor is excluded from them
        and the static always fills them. Returns a 0/1 map (n_x, n_y).
        """
        _cell_frac = float(np.clip(1.0 - self.frac_sig, 0.0, 0.95))
        if _cell_frac <= 1e-6:
            return np.zeros((self.n_x, self.n_y), dtype=np.float32)
        _spots = self._smooth_spatial(self.cell_size)
        _thr = np.quantile(_spots, 1.0 - _cell_frac)
        return (_spots >= _thr).astype(np.float32)

    def _make_expression(self):
        """Pixel-level signal / control fluorophore expression maps.

        Biophysical layout (see the example dual-colour FOV):

        - A set of small **cell bodies** (``self.cells``) shared by both
          reporters, covering ``(1 - frac_sig)`` of the FOV.
        - **Sensor** (``expr_sig``, e.g. GRAB): uniform brightness over the
          *neuropil* only — i.e. everywhere *except* the cell bodies. Never
          expressed in somata.
        - **Static** (``expr_ctrl``): uniform brightness over the cell
          bodies (always) *plus* the ``overlap`` fraction of the
          surrounding neuropil, as small patches of local fluorophore
          bleed-through scattered evenly across the whole FOV (not one
          contiguous region). At ``overlap=0`` the static is confined to
          cell bodies; at ``overlap=1`` it fills the whole neuropil too.

        Stored as ``expr_sig`` and ``expr_ctrl`` (n_x, n_y), with the
        shared ``cells`` mask kept for inspection.
        """
        self.cells = self._make_cells()
        _npil = 1.0 - self.cells                       # neuropil (around cells)

        # Sensor: neuropil only, never in cell bodies.
        self.expr_sig = _npil.astype(np.float32)

        # Static: cell bodies always, plus `overlap` fraction of neuropil.
        _ov = self.overlap
        if _ov >= 1.0 - 1e-6:
            _static_npil = _npil
        elif _ov <= 1e-6:
            _static_npil = np.zeros_like(_npil)
        else:
            # Pick an `_ov` fraction of the neuropil via a fine-scale
            # selector field (correlation length ``cell_size``), so the
            # bleed-through patches are small and scattered evenly across
            # the whole FOV rather than one contiguous region concentrated
            # on one side. Soft-threshold (sigmoid) around the cutoff so
            # expression fades gradually at each patch edge instead of
            # stepping sharply from full to zero.
            _sel = self._smooth_spatial(self.cell_size)
            _npil_vals = _sel[self.cells < 0.5]
            _thr = np.quantile(_npil_vals, 1.0 - _ov)
            _bw = float(np.std(_npil_vals)) * 0.15 + 1e-9
            _static_npil = 1.0 / (1.0 + np.exp(-(_sel - _thr) / _bw))
            _static_npil *= (self.cells < 0.5)
        _static_support = np.clip(np.maximum(self.cells, _static_npil), 0.0, 1.0)
        self.expr_ctrl = _static_support.astype(np.float32)

    def _calcium_trace(self, n_events, tau, amp):
        """Sparse calcium-like trace: exponential transients at onsets."""
        tr = np.zeros(self.n_t)
        _kern = amp * np.exp(-np.arange(0, int(6 * tau)) / tau)
        onsets = self.rng.integers(0, self.n_t, size=n_events)
        for o in onsets:
            _end = min(o + _kern.size, self.n_t)
            tr[o:_end] += _kern[:_end - o]
        return tr

    def _gauss_blob(self, cx, cy, sigma):
        """Normalised Gaussian spatial footprint, shape (n_x, n_y)."""
        _xv, _yv = np.meshgrid(np.arange(self.n_x), np.arange(self.n_y),
                               indexing='ij')
        return np.exp(-((_xv - cx) ** 2 + (_yv - cy) ** 2)
                      / (2.0 * sigma ** 2))

    def _make_signal(self):
        """Spatially varying real signal a(t, x, y) from neural sources.

        ``a`` is the unitless ΔF/F activity field (peak ~1 per transient);
        it is scaled by ``sig_strength`` and the local signal-fluorophore
        expression when the signal channel is assembled. Sources are placed
        preferentially where the signal reporter is actually expressed.
        """
        blobs = np.empty((self.n_sources, self.n_x, self.n_y))
        traces = np.empty((self.n_sources, self.n_t))
        _rate = self.n_t / 250.0          # ~ events per source
        # Sample source centres from the signal-expression map so activity
        # lands where the reporter exists (with a little slack everywhere).
        _p = self.expr_sig.ravel() + 1e-3
        _p = _p / _p.sum()
        _centres = self.rng.choice(self.n_x * self.n_y,
                                   size=self.n_sources, p=_p)
        for j in range(self.n_sources):
            _cx, _cy = np.unravel_index(_centres[j], (self.n_x, self.n_y))
            _sigma = self.rng.uniform(1.5, 3.5)
            blobs[j] = self._gauss_blob(float(_cx), float(_cy), _sigma)
            _n_ev = max(1, int(self.rng.poisson(_rate)))
            _tau = self.rng.uniform(6.0, 12.0)
            traces[j] = self._calcium_trace(_n_ev, _tau, 1.0)

        a = np.einsum('jt,jxy->txy', traces, blobs)
        self.signal = a.astype(np.float32)          # a (ΔF/F activity field)
        # Pixels with both a source footprint and meaningful signal-reporter
        # expression — the only places the neural signal is observable.
        self.sig_mask = (blobs.max(axis=0) > 0.2) & (self.expr_sig > 0.2)
        self._sig_blobs = blobs
        self._sig_traces = traces

    @staticmethod
    def _saturate(v, strength):
        """Mild compressive saturation; ``strength=0`` is the identity.

        Uses ``s·tanh(v/s)`` with ``s = std(v)/strength`` so larger
        ``strength`` compresses more. Monotonic, so it preserves the sign
        of the coupling but bends its magnitude — a controlled departure
        from the linearity the regression assumes.
        """
        if strength <= 0:
            return v
        _s = (np.std(v) + 1e-9) / strength
        return _s * np.tanh(v / _s)

    def _assemble_channels(self):
        """Combine components into the signal (c) and control (d) channels.

        Implements the multiplicative-absorption brightness model: the
        haemodynamic and signal fluctuations a pixel shows scale with its
        resting fluorophore brightness (i.e. its expression), so the
        control channel only reports haemodynamics where the control
        fluorophore is present.
        """
        # Resting fluorophore brightness maps (expression-scaled).
        f0_s = (self.f0_sig * self.expr_sig)[None]      # (1, X, Y)
        f0_c = (self.f0_ctrl * self.expr_ctrl)[None]

        # Channel-specific nonlinear haemodynamic modulation (zero-mean
        # fractional modulation of the emitted light).
        #
        # Sign convention: b is the *brightness* effect of the
        # haemodynamics, not blood volume, so it enters both channels
        # positively (b up -> both channels up), keeping the diagnostic
        # maps positive. _saturate is odd, so this is exactly the old
        # absorbing model (-haemo_strength * _saturate(b)) evaluated at -b.
        mod_s = self.haemo_strength * self._saturate(self.haemo, self.nonlin_sig)
        mod_c = self.haemo_strength * self._saturate(self.haemo, self.nonlin_ctrl)

        noise_s = self.noise_sig * self.rng.standard_normal(
            (self.n_t, self.n_x, self.n_y))
        noise_c = self.noise_ctrl * self.rng.standard_normal(
            (self.n_t, self.n_x, self.n_y))

        # c = bg + F0_s·(1 + dff·a)·(1 + mod_s) + noise
        # d = bg + F0_c·(1 + mod_c) + noise
        self.sig_chan = (
            self.bg_sig
            + f0_s * (1.0 + self.sig_strength * self.signal) * (1.0 + mod_s)
            + noise_s).astype(np.float32)
        self.ctrl_chan = (
            self.bg_ctrl + f0_c * (1.0 + mod_c) + noise_c).astype(
            np.float32)

        # Ground-truth fields for scoring field inference.
        # ``*_clean`` are the noise-free channels. ``*_mod`` are the
        # expression-*independent* time-varying modulations each channel
        # tracks — ``(1 + dff·a)·(1 + mod_s)`` for the sensor and
        # ``(1 + mod_c)`` for the control — defined at *every* pixel,
        # including the low-expression somata the sensor cannot report. A
        # channel's observed field is ``bg + F0·expr·(*_mod)``, so where
        # ``expr → 0`` the modulation is unobservable ('missing'); scoring
        # the inferred field's per-pixel dynamics against ``*_mod`` there
        # measures how well it infers across those missing pixels.
        self.sig_clean = (
            self.bg_sig
            + f0_s * (1.0 + self.sig_strength * self.signal)
            * (1.0 + mod_s)).astype(np.float32)
        self.ctrl_clean = (
            self.bg_ctrl + f0_c * (1.0 + mod_c)).astype(np.float32)
        self.sig_mod = (
            (1.0 + self.sig_strength * self.signal)
            * (1.0 + mod_s)).astype(np.float32)
        self.ctrl_mod = (1.0 + mod_c).astype(np.float32)


# =============================================================================
# Recovery assessment
# =============================================================================

def _pix_corr(u, v):
    """Per-pixel Pearson correlation along time; shapes (T, X, Y) -> (X, Y).

    NaN in, NaN out: a corrector that gates on pixel quality (e.g.
    ``correct_pixel_spatial_subtr``) marks rejected pixels NaN, and a
    rejected pixel's correlation is genuinely undefined. Callers must
    therefore reduce over the result nan-aware — see :func:`_assess`.
    """
    u = u - u.mean(axis=0)
    v = v - v.mean(axis=0)
    num = (u * v).sum(axis=0)
    den = np.sqrt((u ** 2).sum(axis=0) * (v ** 2).sum(axis=0)) + 1e-12
    return num / den


def _corr1d(u, v):
    """Pearson r over the pairwise-complete entries of two 1-D traces.

    ``np.corrcoef`` propagates NaN, so a single gated frame/pixel would
    turn an otherwise-good correlation into NaN.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    m = np.isfinite(u) & np.isfinite(v)
    if m.sum() < 2:
        return np.nan
    _u = u[m] - u[m].mean()
    _v = v[m] - v[m].mean()
    _d = np.sqrt((_u ** 2).sum() * (_v ** 2).sum())
    return float((_u * _v).sum() / _d) if _d > 0 else np.nan


def _sector_traces(stack, n_sec):
    """Block-average a (T, X, Y) stack into per-sector traces.

    Trims to the largest sub-region evenly divisible by ``n_sec`` on each
    axis (mirrors the sector grid used elsewhere for per-sector fits, e.g.
    :func:`signal_correction.correct_full_regress`'s ``f0_n_sectors``), then
    nan-aware block-means so a corrector's gated (NaN) pixels only blank
    the sectors they actually fall in.

    Parameters
    ----------
    stack : (T, X, Y) ndarray
    n_sec : int
        Sectors per axis (grid is ``n_sec`` x ``n_sec``).

    Returns
    -------
    traces : (T, n_sec * n_sec) ndarray
        One time course per sector, in row-major sector order.
    """
    stack = np.asarray(stack, dtype=np.float64)
    n_t, n_x, n_y = stack.shape
    n_sec = int(n_sec)
    x_trim = (n_x // n_sec) * n_sec
    y_trim = (n_y // n_sec) * n_sec
    block_x, block_y = x_trim // n_sec, y_trim // n_sec
    _s = stack[:, :x_trim, :y_trim].reshape(
        n_t, n_sec, block_x, n_sec, block_y)
    # An all-NaN sector (every pixel gated out) is an expected
    # "Mean of empty slice"; the NaN propagates and is masked off by the
    # per-sector finite check downstream.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        return np.nanmean(_s, axis=(2, 4)).reshape(n_t, n_sec * n_sec)


def _sector_amp_map(stack, n_sec):
    """Per-sector response amplitude: temporal std of each sector's trace.

    One scalar per sector, so the result is a coarse spatial map of *how
    much* response each region carries. Baseline-invariant (std discards
    the DC level), which is what makes it comparable between the raw
    channel — which sits on a large ``bg + F0·expr`` pedestal — and a
    corrected stack, whose baseline has been removed.

    Parameters
    ----------
    stack : (T, X, Y) ndarray
    n_sec : int
        Sectors per axis (grid is ``n_sec`` x ``n_sec``).

    Returns
    -------
    amp : (n_sec * n_sec,) ndarray
        Response amplitude per sector, in row-major sector order.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Degrees of freedom <= 0')
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        return np.nanstd(_sector_traces(stack, n_sec), axis=0)


def _sector_amp_corr(corrected, sim, n_sec=8):
    """Spatial fidelity: correlation of per-sector response-amplitude maps.

    Reduces both the true signal ``a`` and the observed (corrected or raw)
    channel to one response amplitude per sector on an ``n_sec`` x
    ``n_sec`` grid (see :func:`_sector_amp_map`), then correlates the two
    maps *across sectors*. It answers "does the recovered activity have
    the right relative magnitude in the right places" — i.e. how faithful
    the recovered **spatial pattern** is, independent of overall scale.

    Two design points, both load-bearing:

    - **Across sectors, not within.** Correlating each sector's time
      course and averaging (the obvious alternative) normalises every
      sector independently, so it cannot see relative amplitude across
      space: scrambling the spatial amplitude pattern by three orders of
      magnitude leaves that score at exactly 1.0. This metric drops to
      ~0.55 on the same input.
    - **Centred (Pearson), not raw cosine.** Amplitude maps are
      non-negative, so an uncentred cosine between two of them is bounded
      well above zero however badly the patterns disagree — it scores
      ~0.6 on pure noise. Centring removes that floor.

    Parameters
    ----------
    corrected : (T, X, Y) ndarray
        Corrected (or raw) signal channel.
    sim : DualColourSim
        Holds the ground-truth ``.signal`` (a).
    n_sec : int
        Sectors per axis (grid is ``n_sec`` x ``n_sec``, default 8x8).

    Returns
    -------
    r : float
        Pearson r across sectors between the two amplitude maps: 1 =
        spatial pattern perfectly faithful, 0 = unrelated, negative =
        anticorrelated. NaN if fewer than 2 sectors are finite in both.
    """
    return _corr1d(_sector_amp_map(sim.signal, n_sec),
                   _sector_amp_map(corrected, n_sec))


def _assess(corrected, sim, n_sector=8):
    """Score one corrected stack against the ground-truth signal/haemo.

    Parameters
    ----------
    corrected : (T, X, Y) ndarray
        Corrected signal channel (normalised units; scale-invariant
        correlations are used so units do not matter).
    sim : DualColourSim
        The simulation holding the ground-truth ``signal`` and ``haemo``.
    n_sector : int
        Sectors per axis for :func:`_sector_amp_corr` (grid is
        ``n_sector`` x ``n_sector``, default 8x8).

    Returns
    -------
    metrics : SimpleNamespace
        ``.recovery`` mean corr(corrected, true signal) over signal pixels;
        ``.leakage`` mean |corr(corrected, true haemo)| over signal pixels
        (where haemodynamics actually contaminate the observable signal);
        ``.leakage_signed`` the same average WITHOUT the absolute value;
        ``.recovery_all`` mean recovery over every pixel;
        ``.spatial_amp_corr`` correlation across sectors between the
        per-sector response-amplitude maps of ``corrected`` and the true
        signal (see :func:`_sector_amp_corr`).

    Notes
    -----
    Read ``leakage`` and ``leakage_signed`` together. Taking the absolute
    value first means a corrector whose residual artefact is unbiased but
    noisy (per-pixel scatter about zero) scores WORSE on ``leakage`` than
    one that systematically over-subtracts, because |·| of zero-mean
    scatter is positive while the biased corrector's residual is a
    consistent offset. ``leakage_signed`` separates the two: ~0 means the
    artefact is genuinely gone, negative means it was over-subtracted and
    is now inverted in the output, positive means under-subtracted.
    """
    corrected = np.asarray(corrected, dtype=np.float32)
    rec_map = _pix_corr(corrected, sim.signal)
    leak_map_signed = _pix_corr(corrected, sim.haemo)
    leak_map = np.abs(leak_map_signed)
    # nan-aware: a corrector may gate out pixels (NaN), and every metric
    # here reduces *across* pixels, so one gated pixel inside sig_mask
    # would otherwise turn the whole score NaN. ``n_scored`` reports how
    # many signal pixels actually survived, so a method that scores well
    # on a handful of pixels is not silently compared against one scored
    # on all of them.
    _fin_sig = np.isfinite(rec_map) & sim.sig_mask
    return SimpleNamespace(
        recovery=float(nanmean(rec_map[sim.sig_mask])),
        leakage=float(nanmean(leak_map[sim.sig_mask])),
        leakage_signed=float(nanmean(leak_map_signed[sim.sig_mask])),
        recovery_all=float(nanmean(rec_map)),
        spatial_amp_corr=_sector_amp_corr(corrected, sim, n_sec=n_sector),
        n_scored=int(_fin_sig.sum()),
        n_sig_px=int(sim.sig_mask.sum()))


def _call_corr_func(func, s1, s2, sector_levels, verbose, kwargs=None):
    """Call a :mod:`signal_correction` correction function, returning the
    corrected stack.

    Injects the shared preprocessing defaults (smoothing / airPLS / robust
    loss, and the sector grid) for whichever of them ``func`` accepts, so
    different correction functions stay directly comparable. ``kwargs``
    overrides any of these defaults for this call.

    Parameters
    ----------
    func : callable
        A ``signal_correction.correct_*`` function, e.g.
        :func:`~signal_correction.correct_full_regress`. Called as
        ``func(s1, s2, **defaults_and_kwargs)`` and expected to return
        ``(corrected, extra)``.
    s1, s2 : (T, X, Y) ndarray
        Control (regressor) and signal channels.
    sector_levels : tuple of int
        Shared sector grid; injected as ``sector_levels`` and/or
        ``f0_n_sectors`` for functions that accept them.
    verbose : bool
        Forwarded as ``verbose`` if accepted.
    kwargs : dict or None
        Per-call overrides (e.g. ``{'fit_mode': 'per_pixel'}``).

    Returns
    -------
    corrected : (T, X, Y) ndarray
    """
    kwargs = dict(kwargs or {})
    _params = inspect.signature(func).parameters
    _defaults = dict(dtype=np.float32, verbose=verbose, smooth_window=10,
                     beta_loss='huber', beta_scale='mad', nn_slope=False,
                     f0_level='sector')
    if 'sector_levels' in _params:
        _defaults['sector_levels'] = sector_levels
    if 'f0_n_sectors' in _params:
        _defaults['f0_n_sectors'] = sector_levels[-1]
    _defaults = {k: v for k, v in _defaults.items() if k in _params}
    _defaults.update(kwargs)
    out, _ = func(s1, s2, **_defaults)
    return out


def _plot_run_regress_spatial(sim, results, corrected_stacks, methods,
                              save_dir, stem, show, row_labels=None):
    """Per-method spatial summary figure, in the style of
    :func:`demo_infer_field_bayesnf`'s ``_spatial`` figure.

    One row per method (``'raw'`` plus each entry in ``methods``), with
    columns grouped into two pairs, each under its own super-title and
    separated by a dotted divider: **single frame** — ground-truth signal
    ``a`` and the (un)corrected channel, both at the frame of peak
    true-signal variance; **recording summary** — the recovery map
    ``corr(corrected, true a)`` and the leakage map ``|corr(corrected,
    true b)|``, both full-timecourse correlations over every frame. The
    true-signal footprint is outlined in cyan throughout.

    Parameters
    ----------
    sim : DualColourSim
        Simulation holding the ground-truth ``.signal`` / ``.haemo`` /
        ``.sig_mask``.
    results : dict
        ``method -> metrics`` from :func:`_assess` (as built by
        :func:`run_regress`), used for the recovery/leakage numbers in the
        panel titles.
    corrected_stacks : dict
        ``method -> (T, X, Y) ndarray``, one entry per row (including
        ``'raw'``).
    methods : list of str
        Non-``'raw'`` method labels, in display order; also the keys used
        to index ``results`` / ``corrected_stacks``.
    save_dir : str
        Output directory (assumed already created / expanded).
    stem : str
        Filename stem for the saved figure.
    show : bool
        If True, leave the figure open (``plt.show``); else close it.
    row_labels : list of str or None
        Non-``'raw'`` row y-axis labels (see :func:`_row_display_label`),
        parallel to ``methods`` but used only for display — ``methods``
        itself still indexes ``results`` / ``corrected_stacks``. Defaults
        to ``methods`` when None.

    Returns
    -------
    path : str
        The saved figure path.
    """
    row_names = ['raw'] + list(methods)
    row_disp = ['raw'] + list(row_labels if row_labels is not None else methods)
    t_star = int(np.argmax(sim.signal.reshape(sim.n_t, -1).var(axis=1)))
    _mlo = float(np.percentile(sim.signal[t_star], 1))
    _mhi = float(np.percentile(sim.signal[t_star], 99))
    _cm = plt.cm.magma

    fig, axes = plt.subplots(len(row_names), 4,
                             figsize=(14, 3.1 * len(row_names)),
                             constrained_layout=True, squeeze=False)
    # Reserve a top margin for the suptitle *and* the group super-titles
    # added below — constrained_layout only auto-reserves space for the
    # suptitle, so without this the group titles collide with it.
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.93))
    for _row, (_name, _disp) in enumerate(zip(row_names, row_disp)):
        _out = np.asarray(corrected_stacks[_name], dtype=np.float32)
        _m = results[_name]
        _rec_map = _pix_corr(_out, sim.signal)
        _leak_map = np.abs(_pix_corr(_out, sim.haemo))
        # nanpercentile: np.percentile propagates NaN, which would make
        # the colour limits NaN and blank the panel for a gated stack.
        _lo = float(np.nanpercentile(_out[t_star], 1))
        _hi = float(np.nanpercentile(_out[t_star], 99))
        _panels = [
            ('ground truth', sim.signal[t_star], _cm, _mlo, _mhi,
             'viridis'),
            ('measured', _out[t_star], _cm, _lo, _hi, None),
            (f'signal recovery: {_m.recovery:.2f}', _rec_map, 'viridis',
             0.0, 1.0, None),
            (f'haemodynamic leakage: {_m.leakage:.2f}', _leak_map,
             'viridis', 0.0, 1.0, None)]
        for _col, (_ttl, _img, _cc, _vlo, _vhi, _unused) in enumerate(
                _panels):
            _ax = axes[_row, _col]
            _im = _ax.imshow(_img, cmap=_cc, vmin=_vlo, vmax=_vhi)
            _ax.contour(sim.sig_mask.astype(float), levels=[0.5],
                       colors='cyan', linewidths=0.6)
            _ax.set_title(_ttl, fontsize=9)
            _ax.set_xticks([])
            _ax.set_yticks([])
            fig.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
        axes[_row, 0].set_ylabel(_disp, fontsize=10)
    fig.suptitle(f'run_regress method comparison — frame t={t_star}  '
                f'(cyan = true-signal footprint)', fontsize=12, y=0.995)

    # Group the single-frame (ground truth/measured) and recording-summary
    # (recovery/leakage) columns with shared super-titles and a dotted
    # divider, both sitting in the top margin reserved above (between the
    # suptitle and the axes) so neither collides with the other.
    fig.canvas.draw()
    _pos = [axes[0, c].get_position() for c in range(4)]
    _y_group = 0.94
    fig.text((_pos[0].x0 + _pos[1].x1) / 2, _y_group, 'single frame',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
    fig.text((_pos[2].x0 + _pos[3].x1) / 2, _y_group, 'recording summary',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
    _x_div = (_pos[1].x1 + _pos[2].x0) / 2
    fig.add_artist(mpl.lines.Line2D(
        [_x_div, _x_div], [0.0, _y_group], transform=fig.transFigure,
        linestyle=':', color='gray', linewidth=1.2))

    path = os.path.join(save_dir, f'{stem}_spatial.pdf')
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def _z(a):
    """Z-score a 1-D array (small epsilon guards against a flat trace).

    nan-aware, so a trace from a gated (NaN) pixel still plots its finite
    part instead of vanishing entirely.
    """
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).any():
        return a
    return (a - nanmean(a)) / (np.nanstd(a) + 1e-9)


def _plot_run_regress_traces(sim, corrected_stacks, methods, save_dir, stem,
                             show):
    """Per-method temporal summary figure, in the style of
    :func:`demo_infer_field_bayesnf`'s ``_traces`` figure.

    One row per method (``'raw'`` plus each entry in ``methods``), with two
    columns: (a) the whole-frame average (ground truth ``a`` vs. corrected,
    z-scored), and (b) the per-pixel trace at the highest-variance
    true-signal pixel (ground truth vs. corrected, z-scored).

    Parameters
    ----------
    sim : DualColourSim
        Simulation holding the ground-truth ``.signal`` and ``.sig_mask``.
    corrected_stacks : dict
        ``method -> (T, X, Y) ndarray``, one entry per row (including
        ``'raw'``).
    methods : list of str
        Non-``'raw'`` method names, in display order.
    save_dir : str
        Output directory (assumed already created / expanded).
    stem : str
        Filename stem for the saved figure.
    show : bool
        If True, leave the figure open (``plt.show``); else close it.

    Returns
    -------
    path : str
        The saved figure path.
    """
    row_names = ['raw'] + list(methods)
    _mask = sim.sig_mask
    _px = tuple(int(v) for v in np.unravel_index(
        int(np.argmax(np.where(_mask, sim.signal.var(axis=0), -np.inf))),
        (sim.n_x, sim.n_y)))

    truth_avg = sim.signal.mean(axis=(1, 2))
    truth_px = sim.signal[:, _px[0], _px[1]]

    fig, axes = plt.subplots(len(row_names), 2,
                             figsize=(13, 2.6 * len(row_names)),
                             constrained_layout=True, squeeze=False)
    for _row, _name in enumerate(row_names):
        _out = np.asarray(corrected_stacks[_name], dtype=np.float32)
        _out_avg = nanmean(_out, axis=(1, 2))
        _out_px = _out[:, _px[0], _px[1]]

        _r_avg = _corr1d(truth_avg, _out_avg)
        _ax = axes[_row, 0]
        _ax.plot(_z(truth_avg), color='k', lw=1.6, label='ground truth neuromod.')
        _ax.plot(_z(_out_avg), color='crimson', lw=1.2,
                 label='measured neuromod.')
        _ax.set_title(f'{_name}: whole-frame average  (r={_r_avg:.3f})',
                     fontsize=10)
        _ax.set_xlabel('frame')
        _ax.set_ylabel('z-scored')
        if _row == 0:
            _ax.legend(fontsize=8, frameon=False)

        _r_px = _corr1d(truth_px, _out_px)
        _ax = axes[_row, 1]
        _ax.plot(_z(truth_px), color='k', lw=1.6, label='ground truth neuromod.')
        _ax.plot(_z(_out_px), color='crimson', lw=1.2,
                 label='measured neuromod.')
        _ax.set_title(f'{_name}: pixel {_px}  (r={_r_px:.3f})', fontsize=10)
        _ax.set_xlabel('frame')
        _ax.set_ylabel('z-scored')
    fig.suptitle('run_regress temporal recovery — whole-frame average vs. '
                'per-pixel trace', fontsize=12)

    path = os.path.join(save_dir, f'{stem}_traces.pdf')
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def _plot_run_regress_channels(sim, save_dir, stem, show):
    """Diagnostic figure for a :func:`run_regress` run: what the *recorded*
    sensor looks like before any correction — the problem to be solved.

    Three panels:

    Left: whole-frame average over time, z-scored — the ground-truth signal
    ``a`` against the recorded sensor and the recorded control. The
    ground-truth/sensor pair is computed exactly as in the ``'raw'`` row of
    :func:`_plot_run_regress_traces` (same whole-frame average, same
    z-scoring), so the two panels are directly comparable; the control is
    overlaid as the extra reference this figure adds. Plotting only
    sensor-vs-control here was misleading — both are recorded channels
    dominated by the same haemodynamics, so they overlap almost perfectly
    and the grey trace reads as a ground truth it is not.

    Middle / right: per-pixel temporal correlation of the recorded sensor
    against each ground-truth field — ``corr(sensor, a)`` (the signal we
    want) and ``corr(sensor, b)`` (the haemodynamics we don't). These are
    the raw-sensor maps only; per-method corrected maps live in the
    spatial figure. They quantify the contamination the correction has to
    remove: with default parameters the haemodynamic modulation is ~8x
    larger than the ``a``-driven one, so the ``b`` map is strongly signed
    (near ±1) while the ``a`` map is weak — that imbalance is the point.

    Both maps are signed correlations, so they use a diverging colormap on
    symmetric ±1 limits, zero on the neutral midpoint. The mean over the
    signal footprint printed in each title is, by construction, the same
    number :func:`_assess` reports as the raw ``recovery`` / ``leakage``.

    Parameters
    ----------
    sim : DualColourSim
        Simulation providing ``.signal``, ``.haemo``, ``.sig_chan``,
        ``.ctrl_chan`` and ``.sig_mask``.
    save_dir : str
        Output directory (assumed already created / expanded).
    stem : str
        Filename stem for the saved figure.
    show : bool
        If True, leave the figure open (``plt.show``); else close it.

    Returns
    -------
    path : str
        The saved figure path.
    """
    _sensor = np.asarray(sim.sig_chan, dtype=np.float32)

    fig, (ax_l, ax_a, ax_b) = plt.subplots(
        1, 3, figsize=(14, 4.6), constrained_layout=True,
        gridspec_kw={'width_ratios': [1.55, 1, 1]})

    # Whole-frame average, to match the 'raw' row of
    # _plot_run_regress_traces panel-for-panel.
    _truth_avg = sim.signal.mean(axis=(1, 2))
    _sen_avg = nanmean(_sensor, axis=(1, 2))
    _ctrl_avg = nanmean(np.asarray(sim.ctrl_chan, dtype=np.float32),
                        axis=(1, 2))
    ax_l.plot(_z(_truth_avg), color='k', lw=1.6, label='ground truth neuromod.')
    ax_l.plot(_z(_sen_avg), color='crimson', lw=1.2, label='measured neuromod.')
    # Dashed and drawn last: the control tracks the sensor at r~0.99, so a
    # solid line underneath would be entirely hidden by it.
    ax_l.plot(_z(_ctrl_avg), color='slategray', lw=1.0, ls='--',
              label='control (raw)')
    ax_l.set_title('recorded whole-frame fluorescence\n'
                   f'r(sensor, a) = {_corr1d(_truth_avg, _sen_avg):.3f}  |  '
                   f'r(sensor, control) = '
                   f'{_corr1d(_ctrl_avg, _sen_avg):.3f}', fontsize=10)
    ax_l.set_xlabel('frame')
    ax_l.set_ylabel('z-scored')
    ax_l.legend(fontsize=8, frameon=False)

    # Signed correlations -> diverging map on symmetric ±1 limits, so 0
    # sits on the neutral midpoint and the sign is readable.
    _maps = [(ax_a, _pix_corr(_sensor, sim.signal),
              'corr(sensor, ground-truth signal a)'),
             (ax_b, _pix_corr(_sensor, sim.haemo),
              'corr(sensor, haemodynamics b)')]
    for _ax, _m, _ttl in _maps:
        _im = _ax.imshow(_m, cmap='RdBu_r', vmin=-1.0, vmax=1.0)
        _ax.contour(sim.sig_mask.astype(float), levels=[0.5],
                    colors='k', linewidths=0.6)
        _ax.set_title(f'{_ttl}\nmean over signal px = '
                      f'{nanmean(_m[sim.sig_mask]):+.2f}', fontsize=9)
        _ax.set_xticks([])
        _ax.set_yticks([])
        fig.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)

    fig.suptitle('recorded sensor: whole-frame trace and spatial '
                 'ground-truth correlations', fontsize=12)

    path = os.path.join(save_dir, f'{stem}_channels.pdf')
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def _active_window(sim, n_frames):
    """Frame slice of length ``n_frames`` centred on the most active period.

    The simulated signal ``a`` is sparse/event-driven, so a window starting
    at frame 0 can easily land on a quiet baseline stretch and look empty.
    Centring on the frame of peak whole-field variance instead guarantees
    the preview window actually contains activity, clamped so it stays in
    bounds.

    Parameters
    ----------
    sim : DualColourSim
        Simulation providing ``.signal`` and ``.n_t``.
    n_frames : int
        Desired window length (clamped to ``sim.n_t``).

    Returns
    -------
    sl : slice
        Frame slice to apply to any ``(T, X, Y)`` array from this sim.
    """
    n_frames = min(int(n_frames), sim.n_t)
    t_star = int(np.argmax(sim.signal.reshape(sim.n_t, -1).var(axis=1)))
    start = max(0, min(sim.n_t - n_frames, t_star - n_frames // 2))
    return slice(start, start + n_frames)


def _save_run_regress_videos(sim, corrected_stacks, methods, save_dir, stem,
                             n_frames):
    """Save ImageJ TIFF 'videos' of a :func:`run_regress` run, over a
    window of ``n_frames`` centred on the most active period (see
    :func:`_active_window`) so files stay a manageable size without landing
    on a quiet, empty-looking stretch.

    Writes:

    - ``<stem>_video_ground_truth_a_b.tif`` — the true neuromodulatory
      signal ``a`` and the true haemodynamic signal ``b`` side by side
      (each on its own scale, as in ``_signals_a_b.tif``).
    - ``<stem>_video_recorded_sensor_control.tif`` — the raw, uncorrected
      sensor and control channels side by side on a shared scale (the
      sensor carries the expression pattern, haemodynamic contamination,
      and noise all together, as actually recorded).
    - ``<stem>_video_corrected_<method>.tif`` — one per entry in
      ``methods``, the corrected signal channel (single panel).

    Parameters
    ----------
    sim : DualColourSim
        Simulation providing ``.signal``, ``.sig_chan``, ``.ctrl_chan``.
    corrected_stacks : dict
        ``method -> (T, X, Y) ndarray`` (``'raw'`` entry is ignored — the
        recorded sensor is written separately, unconditionally).
    methods : list of str
        Method names to write a corrected video for.
    save_dir : str
        Output directory (assumed already created / expanded).
    stem : str
        Filename stem.
    n_frames : int
        Number of frames to include in each video.

    Returns
    -------
    saved : list of str
        Paths actually written (empty if no TIFF writer is available).
    """
    sl = _active_window(sim, n_frames)
    saved = []

    _p_a = os.path.join(save_dir, f'{stem}_video_ground_truth_a_b.tif')
    if _write_two_panel_tiff(_p_a, sim.signal[sl], sim.haemo[sl],
                             ('ground_truth_a', 'ground_truth_b_haemo'),
                             shared_norm=False):
        saved.append(_p_a)

    _p_rec = os.path.join(save_dir,
                          f'{stem}_video_recorded_sensor_control.tif')
    if _write_two_panel_tiff(
            _p_rec, sim.sig_chan[sl], sim.ctrl_chan[sl],
            ('recorded_sensor', 'recorded_control'), shared_norm=True):
        saved.append(_p_rec)

    for name in methods:
        _p_c = os.path.join(save_dir, f'{stem}_video_corrected_{name}.tif')
        if _write_single_tiff(_p_c, corrected_stacks[name][sl],
                              f'corrected_{name}'):
            saved.append(_p_c)
    return saved


# =============================================================================
# Top-level driver
# =============================================================================

def run_regress(sim=None, methods=None, methods_kwargs=None,
                sector_levels=(2, 4, 8),
                n_sector=8, verbose=False, keep_corrected=False,
                plot=True, save_dir='~/Desktop',
                stem='run_regress_summary', show=True, save_videos=True,
                video_n_frames=100):
    """Run every correction method on a simulation and score recovery.

    Builds (or accepts) a :class:`DualColourSim`, corrects the signal
    channel against the static control with each method, and reports how
    well each recovers the ground-truth real signal ``a`` while suppressing
    the haemodynamics ``b``.

    Parameters
    ----------
    sim : DualColourSim or None
        Simulation to use. If None, a default one is built.
    methods : list of callable or None
        One :mod:`signal_correction` correction function per method, e.g.
        ``[correct_full_regress, correct_full_regress, correct_two_stage_regress]``
        (the same function may repeat with different ``methods_kwargs`` —
        e.g. several ``correct_full_regress`` calls at different
        ``fit_mode``). Each is paired positionally with the matching entry
        of ``methods_kwargs``. Defaults to ``correct_full_regress`` at
        ``fit_mode`` ``'global'`` / ``'per_sector'`` / ``'per_pixel'``, plus
        ``correct_two_stage_regress``.
    methods_kwargs : list of dict or None
        Per-method keyword overrides for the matching ``methods`` entry
        (see :func:`_call_corr_func`), e.g. ``[{'fit_mode': 'global'},
        {'fit_mode': 'per_sector'}, {'fit_mode': 'per_pixel'}, {}]``. Must
        be the same length as ``methods``. Defaults to that same triple
        (plus ``{}`` for the ``two_stage`` entry) when ``methods`` is also
        left at its default; otherwise defaults to an empty dict per
        entry.
    sector_levels : tuple of int
        Cascade levels for ``two_stage``; its finest level is also used as
        the sector grid for the ``full_regress`` modes (fair comparison).
    n_sector : int
        Sectors per axis for the ``spatial_amp_corr`` metric (grid is
        ``n_sector`` x ``n_sector``, default 8x8; see :func:`_sector_amp_corr`
        and :func:`_assess`). Independent of ``sector_levels``.
    verbose : bool
        Forward verbose output from the correction functions.
    keep_corrected : bool
        If True, also return each method's corrected stack (memory heavy).
    plot : bool
        If True (default), save per-method summary figures (in the style of
        :func:`demo_infer_field_bayesnf`'s spatial/traces figures) via
        :func:`_plot_run_regress_spatial`, :func:`_plot_run_regress_traces`,
        and :func:`_plot_run_regress_channels`.
    save_dir : str
        Output directory for the summary figures (``~`` expanded, created if
        missing). Only used when ``plot`` is True.
    stem : str
        Filename stem for the saved outputs. Every figure and video of a
        run is named ``<stem>_<method_tag>_<kwarg_tag>_...``, where
        ``method_tag`` names the ``methods`` compared (see
        :func:`_run_regress_method_tag`, e.g. ``pixel_spatial_subtr``) and
        ``kwarg_tag`` encodes ``methods_kwargs`` (see :func:`_methods_kwarg_tag`)
        — so runs over different methods or kwargs sit side by side
        instead of overwriting each other, and each file says what
        produced it. Applied to every output of the run, even ones whose
        content does not itself vary with method (the ground-truth and
        recorded-channel videos, the channels figure). Only used when
        ``plot`` or ``save_videos`` is True.
    show : bool
        If True, leave the summary figures open (``plt.show``); else close
        them. Only used when ``plot`` is True.
    save_videos : bool
        If True, also save ImageJ TIFF 'videos' (ground-truth ``a``,
        recorded sensor/control, and each method's corrected response) via
        :func:`_save_run_regress_videos`. Off by default (extra disk I/O).
    video_n_frames : int
        Number of frames to include in each saved video, centred on the
        most active period (see :func:`_active_window`) rather than always
        starting at frame 0. Default 100, so files stay a manageable size
        regardless of ``sim.n_t``. Only used when ``save_videos`` is True.

    Returns
    -------
    out : SimpleNamespace
        ``.sim`` the simulation; ``.results`` dict ``label -> metrics``
        (plus a ``'raw'`` baseline = uncorrected signal channel), keyed by
        ``.method_labels`` (the ``methods`` / ``methods_kwargs`` column
        labels, see :func:`_method_labels`); ``.corrected``
        dict when ``keep_corrected`` is set; ``.plot_paths`` dict with
        ``'spatial'`` / ``'traces'`` / ``'channels'`` saved figure paths
        (PDF — vector, so they stay sharp in a figure/manuscript) when
        ``plot`` is set; ``.video_paths`` list of saved TIFF paths when
        ``save_videos`` is set; ``.method_tag`` / ``.kwarg_tag`` the tags
        embedded in those filenames when either is set.
    """
    if sim is None:
        sim = DualColourSim()
    _default_methods = methods is None
    if methods is None:
        methods = [sc.correct_full_regress] * 4 + [sc.correct_two_stage_regress]
    if methods_kwargs is None:
        methods_kwargs = ([{'fit_mode': 'global'}, {'fit_mode': 'per_sector'},
                          {'fit_mode': 'per_pixel'},
                          {'fit_mode': 'per_pixel_clean_reg'}, {}]
                         if _default_methods else [{} for _ in methods])
    if len(methods_kwargs) != len(methods):
        raise ValueError('methods and methods_kwargs must have the same '
                         f'length (got {len(methods)} and '
                         f'{len(methods_kwargs)})')
    labels = _method_labels(methods, methods_kwargs)
    row_labels = [_row_display_label(func, lbl)
                 for func, lbl in zip(methods, labels)]

    s1 = sim.ctrl_chan      # static control (regressor)
    s2 = sim.sig_chan       # signal channel (to be corrected)

    results = {}
    corrected_stacks = {}

    # Baseline: how well does the *uncorrected* signal channel track a,
    # and how much haemodynamics does it still carry.
    results['raw'] = _assess(s2, sim, n_sector=n_sector)
    corrected_stacks['raw'] = s2

    for func, kw, lbl in zip(methods, methods_kwargs, labels):
        if verbose:
            print(f'\n=== {lbl} ===')
        out = _call_corr_func(func, s1, s2, sector_levels, verbose, kw)
        results[lbl] = _assess(out, sim, n_sector=n_sector)
        corrected_stacks[lbl] = out

    _print_table(sim, results, labels, sector_levels)

    res = SimpleNamespace(sim=sim, results=results, method_labels=labels)
    if plot or save_videos:
        save_dir = os.path.expanduser(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        # Every output of this run records which method(s) and kwargs
        # produced it, so runs over different methods/kwargs sit side by
        # side instead of overwriting each other. Folded into the stem
        # rather than appended per-file so all of a run's outputs sort
        # together; applied uniformly (even to method-independent outputs
        # like the ground-truth video) so every file names the run that
        # made it.
        res.method_tag = _run_regress_method_tag(labels)
        res.kwarg_tag = _methods_kwarg_tag(methods_kwargs)
        stem = f'{stem}_{res.method_tag}_{res.kwarg_tag}'
    if plot:
        res.plot_paths = {
            'spatial': _plot_run_regress_spatial(
                sim, results, corrected_stacks, labels, save_dir, stem,
                show, row_labels=row_labels),
            'traces': _plot_run_regress_traces(
                sim, corrected_stacks, labels, save_dir, stem, show),
            'channels': _plot_run_regress_channels(
                sim, save_dir, stem, show)}
    if save_videos:
        res.video_paths = _save_run_regress_videos(
            sim, corrected_stacks, labels, save_dir, stem, video_n_frames)
    if keep_corrected:
        res.corrected = corrected_stacks
    return res


# =============================================================================
# Parameter-space summary figure
# =============================================================================

# Metrics shown in the summary figure: (attribute, panel title, colormap,
# higher-is-better, vmin, vmax). ``recovery``/``leakage`` use the seaborn
# 'rocket' palette on a fixed [0, 1] scale (``leakage`` reversed so 'good'
# is always the bright/warm end); ``spatial_amp_corr`` is a signed
# correlation, so it gets a diverging blue/red palette centred at 0.
_FIG_METRICS = (
    ('recovery', 'signal recovered\ncorr(corrected, true a)',
     'rocket', True, 0.0, 1.0),
    ('leakage', 'haemodynamics left\n|corr(corrected, true b)|',
     'rocket_r', False, 0.0, 1.0),
    # Signed: 0 is the target, so a diverging palette centred at 0 —
    # negative (over-subtracted, artefact inverted) and positive
    # (under-subtracted) read as opposite failures rather than being
    # collapsed together by the absolute value above.
    ('leakage_signed', 'haemodynamics left, signed\ncorr(corrected, true b)',
     'RdBu_r', False, -1.0, 1.0),
    ('spatial_amp_corr', 'spatial pattern fidelity\nr(sector amplitude, true a)',
     'RdBu_r', True, -1.0, 1.0),
)


def _resolve_cmap(name):
    """Return a matplotlib Colormap for a (possibly seaborn) palette name."""
    if sns is not None:
        try:
            return sns.color_palette(name, as_cmap=True)
        except (ValueError, KeyError):
            pass
    try:
        return plt.get_cmap(name)
    except ValueError:
        # seaborn palettes unavailable -> perceptual fallback.
        return plt.get_cmap('magma_r' if name.endswith('_r') else 'magma')


# Short tokens for sim parameters, used to build reproducible filenames.
_PARAM_ABBR = {
    'n_t': 'nt', 'n_x': 'nx', 'n_y': 'ny', 'n_haemo_modes': 'nhm',
    'n_sources': 'nsrc', 'haemo_tau': 'htau', 'haemo_strength': 'hstr',
    'sig_strength': 'sstr', 'f0_sig': 'f0s', 'f0_ctrl': 'f0c', 'bg_sig': 'bgs',
    'bg_ctrl': 'bgc', 'noise_sig': 'nzs', 'noise_ctrl': 'nzc',
    'nonlin_sig': 'nls', 'nonlin_ctrl': 'nlc', 'overlap': 'ov',
    'frac_sig': 'fsig', 'frac_ctrl': 'fctrl',
    'seed': 'sd', 'frac_sensor': 'fsens',
}


def _fmt_num(v):
    """Compact, filename-safe formatting of a scalar parameter value."""
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, (int, float)):
        return f'{v:g}'
    return str(v)


def _sim_kwargs_for_param(param, val):
    """Map a sweepable param name/value to :class:`DualColourSim` kwargs.

    ``'frac_sensor'`` is a pseudo-parameter that ties the sensor's and
    control's expressing fraction together (``frac_sig = frac_ctrl``);
    every other name is forwarded as-is as a single kwarg.

    Parameters
    ----------
    param : str
        Sweep parameter name (e.g. ``'overlap'``, ``'frac_sensor'``, or any
        other :class:`DualColourSim` keyword argument).
    val : float
        Value to assign.

    Returns
    -------
    kwargs : dict
        One or more ``{kwarg: value}`` entries to pass to
        :class:`DualColourSim`.
    """
    if param == 'frac_sensor':
        return {'frac_sig': val, 'frac_ctrl': val}
    return {param: val}


def _param_abbr(param):
    """Filename-safe abbreviation token for a sweep parameter name."""
    return _PARAM_ABBR.get(param, param)


def _sim_param_tag(sim_kwargs, param1=None, param1_vals=None, param2=None,
                   param2_vals=None):
    """Build a compact tag of every sim parameter (default or overridden).

    The :class:`DualColourSim` constructor defaults are read by
    introspection and overlaid with ``sim_kwargs`` so the tag reflects the
    exact dataset. When sweeping, ``param1`` / ``param2`` (and any kwargs
    they map to, e.g. ``frac_sensor`` -> ``frac_sig`` + ``frac_ctrl``) are
    shown as ranges (``<abbr><lo>to<hi>``) rather than single values.

    Parameters
    ----------
    sim_kwargs : dict
        Overrides passed to :class:`DualColourSim`.
    param1, param2 : str or None
        Names of the swept parameters (see :func:`_sim_kwargs_for_param`).
    param1_vals, param2_vals : iterable or None
        Swept axis values, aligned with ``param1`` / ``param2``.

    Returns
    -------
    tag : str
        ``_``-joined ``<abbr><value>`` tokens, filename-safe.
    """
    sig = inspect.signature(DualColourSim.__init__)
    params = {name: (p.default if p.default is not inspect.Parameter.empty
                     else None)
              for name, p in sig.parameters.items() if name != 'self'}
    params.update(sim_kwargs)

    # Map each swept pseudo/real param to the kwarg name(s) it drives, and
    # the range token that should replace their scalar value in the tag.
    _swept = {}
    for param, vals in ((param1, param1_vals), (param2, param2_vals)):
        if param is None or vals is None:
            continue
        ab = _param_abbr(param)
        _tok = f'{ab}{_fmt_num(min(vals))}to{_fmt_num(max(vals))}'
        for kwarg in _sim_kwargs_for_param(param, None):
            _swept[kwarg] = _tok

    parts = []
    _emitted = set()
    for name, val in params.items():
        if name in _swept:
            if _swept[name] not in _emitted:
                parts.append(_swept[name])
                _emitted.add(_swept[name])
        else:
            ab = _PARAM_ABBR.get(name, name)
            parts.append(f'{ab}{_fmt_num(val)}')
    return '_'.join(parts)


def _normalise_u16(arr, vmin, vmax):
    """Scale a float array to uint16 over ``[vmin, vmax]`` (clipped).

    NaN-safe in both arguments and array, because uint16 has no NaN to
    fall back on and every failure here is silent — a blank video, not an
    error:

    - Non-finite ``vmin``/``vmax`` (e.g. from ``np.percentile`` over a
      gated stack) would make *every* pixel NaN. Note ``nan <= 0`` is
      False, so a bare ``if rng <= 0`` guard does **not** catch it. We
      fall back to the finite min/max of the data.
    - NaN pixels (gated out by a corrector) map to 0: black reads as
      "no data", and float->uint16 of NaN is otherwise undefined.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        _finite = arr[np.isfinite(arr)]
        vmin = float(_finite.min()) if _finite.size else 0.0
        vmax = float(_finite.max()) if _finite.size else 1.0
    rng = vmax - vmin
    if not np.isfinite(rng) or rng <= 0:
        rng = 1.0
    out = (arr - vmin) / rng
    np.clip(out, 0.0, 1.0, out=out)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return (out * 65535.0).astype(np.uint16)


def _write_two_panel_tiff(path, left, right, labels, shared_norm=False,
                          gap=4):
    """Write a side-by-side (left | right) uint16 TIFF stack 'video'.

    Each frame places ``left`` and ``right`` ``(X, Y)`` images next to each
    other (columns), separated by a bright gap, and the stack runs over
    time -> shape ``(T, X, 2*Y + gap)``. Saved as an ImageJ-compatible
    multipage TIFF.

    Parameters
    ----------
    path : str
        Output ``.tif`` path.
    left, right : (T, X, Y) ndarray
        The two fields to show side by side.
    labels : (str, str)
        Names of the two panels (recorded in the TIFF description).
    shared_norm : bool
        If True, normalise both panels with one shared range (use when
        they share physical units, e.g. recorded sensor vs control);
        otherwise normalise each panel independently.
    gap : int
        Width (px) of the separator column between panels.

    Returns
    -------
    path : str or None
        The written path, or None if no TIFF writer is available.
    """
    try:
        import tifffile
    except ImportError:
        print('  [skip video] tifffile not available')
        return None

    # Robust 0.5–99.5 percentile stretch so a single noise/transient spike
    # does not compress the dynamic range (purely for viewability).
    # nan-aware: np.percentile propagates NaN, and a gated stack would
    # then normalise to a uniformly blank video.
    def _range(arr):
        return (float(np.nanpercentile(arr, 0.5)),
                float(np.nanpercentile(arr, 99.5)))

    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if shared_norm:
        _ll, _lh = _range(left)
        _rl, _rh = _range(right)
        _lo, _hi = min(_ll, _rl), max(_lh, _rh)
        l16 = _normalise_u16(left, _lo, _hi)
        r16 = _normalise_u16(right, _lo, _hi)
    else:
        l16 = _normalise_u16(left, *_range(left))
        r16 = _normalise_u16(right, *_range(right))

    n_t, n_x, _ = l16.shape
    sep = np.full((n_t, n_x, gap), 65535, dtype=np.uint16)
    stack = np.concatenate([l16, sep, r16], axis=2)
    tifffile.imwrite(path, stack, imagej=True,
                     metadata={'axes': 'TYX',
                               'Labels': f'{labels[0]} | {labels[1]}'})
    return path


def _write_single_tiff(path, arr, label):
    """Write a single-panel uint16 TIFF stack 'video'.

    Parameters
    ----------
    path : str
        Output ``.tif`` path.
    arr : (T, X, Y) ndarray
        Field to write.
    label : str
        Name recorded in the TIFF description.

    Returns
    -------
    path : str or None
        The written path, or None if no TIFF writer is available.
    """
    try:
        import tifffile
    except ImportError:
        print('  [skip video] tifffile not available')
        return None

    arr = np.asarray(arr, dtype=np.float32)
    # nanpercentile: a corrector may gate pixels out (NaN), and
    # np.percentile propagates NaN — that would blank the whole video.
    lo = float(np.nanpercentile(arr, 0.5))
    hi = float(np.nanpercentile(arr, 99.5))
    u16 = _normalise_u16(arr, lo, hi)
    tifffile.imwrite(path, u16, imagej=True,
                     metadata={'axes': 'TYX', 'Labels': label})
    return path


def _methods_kwarg_tag(methods_kwargs):
    """Filename tag recording the correction kwargs a run was made with.

    Every figure and video of a :func:`run_regress` run carries it, so a
    directory of results says which parameterisation produced which file
    rather than each run silently overwriting the last. Mirrors
    ``load_exp_twop._full_regress_kwarg_tag``, but encodes only the
    caller-supplied overrides — the shared defaults are identical across
    runs, so spending filename length on them buys nothing here.

    Parameters
    ----------
    methods_kwargs : list of dict
        The ``methods_kwargs`` passed to :func:`run_regress`, one dict per
        entry in ``methods``. ``fit_mode`` is excluded — it is already
        reflected in the method labels (see :func:`_method_labels`), so
        encoding it again here would be redundant.

    Returns
    -------
    tag : str
        Filename-safe ``name-value`` tokens joined by ``_``, sorted by
        name within each method and given order across methods so the
        same kwargs always give the same tag. ``'default'`` when no
        overrides were given.
    """
    _extra = [{k: v for k, v in kw.items() if k != 'fit_mode'}
             for kw in methods_kwargs]
    if not any(_extra):
        return 'default'
    tag = '_'.join(f'{k}-{fmt_kwarg_val(v)}'
                   for kw in _extra for k, v in sorted(kw.items()))
    # Keep tuples/dicts readable rather than mangled: (2,4,8) -> 2-4-8.
    tag = tag.replace(',', '-')
    tag = re.sub(r'[^A-Za-z0-9._-]', '', tag)
    # Filename components are capped at 255 bytes; truncate long tags but
    # keep them unique, so two different kwarg sets never collide.
    if len(tag) > 80:
        tag = f'{tag[:72]}-{hashlib.sha1(tag.encode()).hexdigest()[:7]}'
    return tag


def _run_regress_method_tag(methods):
    """Filename tag naming which correction method(s) a run evaluated.

    Every figure and video of a :func:`run_regress` run carries it —
    including outputs whose content does not itself depend on method
    (e.g. the ``ground_truth_a`` / ``recorded_sensor_control`` videos,
    and the ``channels`` figure) — so every file from a run says what was
    being compared, and runs over different method sets never collide.
    Mirrors :func:`_methods_kwarg_tag`.

    Parameters
    ----------
    methods : list of str
        The ``methods`` passed to :func:`run_regress`, e.g.
        ``['pixel_spatial_subtr']`` or ``['global', 'two_stage']``.

    Returns
    -------
    tag : str
        Method names joined by ``-``, in the given order (not sorted —
        that order is the caller's comparison order and is worth
        preserving). ``'none'`` if ``methods`` is empty.
    """
    if not methods:
        return 'none'
    tag = '-'.join(methods)
    tag = re.sub(r'[^A-Za-z0-9._-]', '', tag)
    if len(tag) > 60:
        tag = f'{tag[:52]}-{hashlib.sha1(tag.encode()).hexdigest()[:7]}'
    return tag


def _method_labels(methods, methods_kwargs):
    """Column / filename labels for a ``(methods, methods_kwargs)`` spec.

    Uses each entry's ``fit_mode`` kwarg when present (the usual way
    :func:`~signal_correction.correct_full_regress` variants are told
    apart), else a shortened function name (``correct_full_regress`` ->
    ``'full'``). Collisions (e.g. the same function + kwargs given twice)
    are disambiguated with a numeric suffix.
    """
    labels = []
    for i, (func, kw) in enumerate(zip(methods, methods_kwargs)):
        if 'fit_mode' in kw:
            labels.append(str(kw['fit_mode']))
        else:
            _short = func.__name__.replace('correct_', '').replace(
                '_regress', '')
            labels.append(_short or f'method{i}')
    _seen = {}
    out = []
    for lbl in labels:
        n = _seen.get(lbl, 0)
        _seen[lbl] = n + 1
        out.append(lbl if n == 0 else f'{lbl}_{n}')
    return out


def _row_display_label(func, label):
    """Human-readable row label for a :func:`run_regress` method.

    :func:`~signal_correction.correct_full_regress` covers several
    ``fit_mode`` submethods behind one function, so its rows are prefixed
    ``'lin. regress.: <submethod>'`` to name the shared regression
    machinery explicitly; every other method keeps its plain ``label``
    (see :func:`_method_labels`).
    """
    if func is sc.correct_full_regress:
        return f'lin. regress.: {label}'
    return label


def save_sim_videos(sim, save_dir='.', stem='dual_colour_sim', gap=4,
                    methods=(), methods_kwargs=(), method_labels=None,
                    sector_levels=(2, 4, 8)):
    """Save TIFF-stack 'videos' of the ground-truth and recorded fields.

    Always writes two ImageJ-compatible multipage TIFFs:

    - ``<stem>_signals_a_b.tif`` — the real signal ``a`` (neuromodulator)
      beside the haemodynamics ``b``, each normalised independently (they
      are different physical quantities).
    - ``<stem>_sensor_control.tif`` — the recorded signal sensor beside the
      static control sensor, sharing one intensity scale (same units).

    Plus one ``<stem>_truth_vs_corrected_<label>.tif`` per entry in
    ``methods`` — the ground-truth neuromodulatory signal ``a`` (left)
    beside the corrected recovered signal (right), each normalised
    independently, showing how well that correction method reconstructs the
    true signal.

    Parameters
    ----------
    sim : DualColourSim
        A generated simulation (provides ``.signal``, ``.haemo``,
        ``.sig_chan``, ``.ctrl_chan``).
    save_dir : str
        Directory to write the TIFFs into (created if missing).
    stem : str
        Filename stem (parameters are typically already encoded here).
    gap : int
        Separator width (px) between the two side-by-side panels.
    methods : list of callable
        Correction functions from :mod:`signal_correction` (e.g.
        :func:`~signal_correction.correct_full_regress`); one
        truth-vs-corrected video is written per entry. Empty skips them.
    methods_kwargs : list of dict
        Per-entry keyword overrides, aligned with ``methods`` (see
        :func:`_call_corr_func`).
    method_labels : list of str or None
        Filename labels, aligned with ``methods``. Computed from
        ``methods`` / ``methods_kwargs`` via :func:`_method_labels` if None.
    sector_levels : tuple of int
        Shared sector grid for the corrections.

    Returns
    -------
    saved : list of str
        Paths actually written (empty if no TIFF writer is available).
    """
    os.makedirs(save_dir, exist_ok=True)
    if method_labels is None:
        method_labels = _method_labels(methods, methods_kwargs)
    saved = []
    _p_ab = os.path.join(save_dir, f'{stem}_signals_a_b.tif')
    _p_sc = os.path.join(save_dir, f'{stem}_sensor_control.tif')
    if _write_two_panel_tiff(_p_ab, sim.signal, sim.haemo,
                             ('a_neuromod', 'b_haemo'),
                             shared_norm=False, gap=gap):
        saved.append(_p_ab)
    if _write_two_panel_tiff(_p_sc, sim.sig_chan, sim.ctrl_chan,
                             ('sensor', 'control'),
                             shared_norm=True, gap=gap):
        saved.append(_p_sc)

    # One truth-vs-corrected video per requested method.
    for func, kw, lbl in zip(methods, methods_kwargs, method_labels):
        _corr = _call_corr_func(func, sim.ctrl_chan, sim.sig_chan,
                                sector_levels, False, kw)
        _p_tc = os.path.join(save_dir, f'{stem}_truth_vs_corrected_{lbl}.tif')
        if _write_two_panel_tiff(_p_tc, sim.signal, _corr,
                                 ('ground_truth_a', f'corrected_{lbl}'),
                                 shared_norm=False, gap=gap):
            saved.append(_p_tc)
    return saved


def _sweep_point(args):
    """Run one sweep grid point: build the sim, score every method.

    Module-level (and returning only plain floats) so it can be shipped to
    a worker process by :func:`_sweep_metrics_2d`.

    Parameters
    ----------
    args : tuple
        ``(i, j, pt_kwargs, methods, methods_kwargs, method_labels,
        sector_levels, n_sector, seed, sim_kwargs)``.

    Returns
    -------
    i, j : int
        Grid indices the result belongs to.
    vals : dict
        ``vals[label][metric_attr] -> float``, including ``'raw'``.
    """
    (i, j, pt_kwargs, methods, methods_kwargs, method_labels,
     sector_levels, n_sector, seed, sim_kwargs) = args
    sim = DualColourSim(seed=seed, **pt_kwargs, **sim_kwargs)
    s1, s2 = sim.ctrl_chan, sim.sig_chan
    results = {'raw': _assess(s2, sim, n_sector=n_sector)}
    for func, kw, lbl in zip(methods, methods_kwargs, method_labels):
        _corr = _call_corr_func(func, s1, s2, sector_levels, False, kw)
        results[lbl] = _assess(_corr, sim, n_sector=n_sector)
    metric_attrs = [m[0] for m in _FIG_METRICS]
    vals = {n: {a: float(getattr(r, a)) for a in metric_attrs}
            for n, r in results.items()}
    return i, j, vals


def _resolve_n_jobs(n_jobs, n_total):
    """Clamp ``n_jobs`` (None/-1 = all cores) to ``[1, n_total]``."""
    if n_jobs is None or n_jobs < 0:
        n_jobs = os.cpu_count() or 1
    return max(1, min(int(n_jobs), n_total))


def _sweep_metrics_2d(param1, param1_vals, param2, param2_vals, methods,
                      methods_kwargs, method_labels, sector_levels, seed,
                      sim_kwargs, n_jobs=None, n_sector=8):
    """Sweep ``param1`` x ``param2``, collecting metrics per method.

    Each axis value is mapped to one or more :class:`DualColourSim` kwargs
    via :func:`_sim_kwargs_for_param` (e.g. ``'frac_sensor'`` ties
    ``frac_sig`` and ``frac_ctrl`` to the same value at each grid point).

    Parameters
    ----------
    param1, param2 : str
        Names of the swept parameters (y-axis, x-axis).
    param1_vals, param2_vals : iterable
        Axis values, aligned with ``param1`` / ``param2``.
    methods : list of callable
        Correction functions from :mod:`signal_correction`.
    methods_kwargs : list of dict
        Per-entry keyword overrides, aligned with ``methods``.
    method_labels : list of str
        Column labels, aligned with ``methods`` (see :func:`_method_labels`).
    n_jobs : int or None
        Worker processes for the grid; None / -1 = all cores, 1 = serial
        (see :func:`_resolve_n_jobs`).
    n_sector : int
        Sectors per axis for the ``spatial_amp_corr`` metric (see
        :func:`_sector_amp_corr`), shared across every grid point.

    Returns
    -------
    out : dict
        ``out[metric_attr][label]`` is a 2-D ndarray of shape
        ``(n_param1, n_param2)`` (row = ``param1``, col = ``param2``);
        ``label`` includes ``'raw'`` (the uncorrected baseline).
    """
    names = ['raw'] + list(method_labels)
    metric_attrs = [m[0] for m in _FIG_METRICS]
    n_p1, n_p2 = len(param1_vals), len(param2_vals)
    out = {a: {n: np.full((n_p1, n_p2), np.nan) for n in names}
           for a in metric_attrs}
    n_total = n_p1 * n_p2

    # One job per grid point; each is fully independent (shared seed, so
    # results are identical whether run serially or in parallel).
    jobs = []
    for i, v1 in enumerate(param1_vals):
        for j, v2 in enumerate(param2_vals):
            _pt_kwargs = dict(_sim_kwargs_for_param(param1, v1))
            _pt_kwargs.update(_sim_kwargs_for_param(param2, v2))
            jobs.append((i, j, _pt_kwargs, methods, methods_kwargs,
                         method_labels, sector_levels, n_sector, seed,
                         sim_kwargs))

    def _store(i, j, vals):
        for a in metric_attrs:
            for n in names:
                out[a][n][i, j] = vals[n][a]

    n_jobs = _resolve_n_jobs(n_jobs, n_total)
    # Each grid point runs one sim through every correction method (several
    # seconds); print progress so the sweep is visibly advancing rather than
    # appearing to hang between figures.
    if n_jobs == 1:
        for _k, job in enumerate(jobs, start=1):
            i, j = job[0], job[1]
            print(f'\r  sim {_k}/{n_total} '
                  f'({param1}={param1_vals[i]:.2f}, '
                  f'{param2}={param2_vals[j]:.2f})...', end='', flush=True)
            _store(*_sweep_point(job))
    else:
        print(f'  running {n_total} sims on {n_jobs} processes...',
              flush=True)
        with cf.ProcessPoolExecutor(max_workers=n_jobs) as _ex:
            _futs = [_ex.submit(_sweep_point, job) for job in jobs]
            for _k, _fut in enumerate(cf.as_completed(_futs), start=1):
                _store(*_fut.result())
                print(f'\r  sim {_k}/{n_total} done...', end='', flush=True)
    print(f'\r  {n_total}/{n_total} sims done.' + ' ' * 20)
    return out


def _axis_ticks(vals, n_ticks=5):
    """Evenly *value*-spaced tick positions/labels for a categorical axis.

    ``vals`` label each column/row of the heatmap in plotted (index) order,
    and may themselves be unevenly spaced (e.g. a sweep denser at one end).
    Snapping ticks to evenly-spaced *indices* (the previous behaviour) then
    inherits that unevenness in the labels. Instead, pick ``n_ticks``
    evenly-spaced *values* between ``min(vals)`` and ``max(vals)`` (e.g.
    ``[0, 0.25, 0.5, 0.75, 1]`` for a 0-1 sweep) and interpolate each to a
    fractional index position, so ticks land at round numbers regardless of
    where the actual grid points fall.

    Parameters
    ----------
    vals : sequence of float
        Swept axis values, in plotted (index) order (ascending, descending,
        or irregular).
    n_ticks : int
        Target number of ticks; reduced to ``len(vals)`` if the axis has
        fewer grid points.

    Returns
    -------
    positions : ndarray
        Fractional index positions, for ``ax.set_xticks``/``set_yticks``.
    labels : list of str
        Each target value formatted to 2 decimals.
    """
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    n_ticks = max(1, min(n_ticks, n))
    targets = np.linspace(vals.min(), vals.max(), n_ticks)
    _order = np.argsort(vals)
    positions = np.interp(targets, vals[_order], _order.astype(float))
    return positions, [f'{v:.2f}' for v in targets]


def _render_heatmap_grid(metrics, metric_specs, method_labels, param1,
                         param1_vals, param2, param2_vals, annotate):
    """Render a (n_metric x n_method) heatmap grid into a new figure.

    Typography (fonts, title weight, tick sizes) is left to the active
    matplotlib stylesheet so the caller can wrap this in a style context.

    Returns
    -------
    fig, axes : Figure and (n_metric, n_method) ndarray of Axes.
    """
    n_rows = len(metric_specs)
    n_cols = len(method_labels)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.0 * n_cols + 0.9, 2.0 * n_rows),
                             squeeze=False)
    n_p1, n_p2 = len(param1_vals), len(param2_vals)
    # Cell-edge coordinates for pcolormesh (vector quads, unlike imshow's
    # rasterised bitmap, so the saved PDF stays sharp at any zoom); centres
    # land on the same integer coordinates imshow used, so ticks/annotation
    # placement below are unaffected.
    _x_edges = np.arange(n_p2 + 1) - 0.5
    _y_edges = np.arange(n_p1 + 1) - 0.5
    _xtick_pos, _xtick_lbl = _axis_ticks(param2_vals)
    _ytick_pos, _ytick_lbl = _axis_ticks(param1_vals)
    for ri, (attr, title, cmap_name, _, vmin, vmax) in enumerate(metric_specs):
        cmap = _resolve_cmap(cmap_name)
        im = None
        for ci, m in enumerate(method_labels):
            ax = axes[ri, ci]
            arr = metrics[attr][m]
            im = ax.pcolormesh(_x_edges, _y_edges, arr, cmap=cmap,
                               vmin=vmin, vmax=vmax, shading='flat')
            ax.set_xlim(_x_edges[0], _x_edges[-1])
            ax.set_ylim(_y_edges[0], _y_edges[-1])
            ax.set_aspect('auto')
            ax.set_xticks(_xtick_pos)
            ax.set_xticklabels(_xtick_lbl)
            ax.set_yticks(_ytick_pos)
            ax.set_yticklabels(_ytick_lbl)
            ax.tick_params(length=0)
            for _sp in ax.spines.values():
                _sp.set_visible(False)
            if annotate:
                for i in range(n_p1):
                    for j in range(n_p2):
                        _val = float(arr[i, j])
                        # Text colour from cell luminance (cmap-direction
                        # agnostic, since 'rocket' / 'rocket_r' / 'RdBu_r'
                        # differ); cmap() wants [0, 1], so normalise by
                        # this row's actual (vmin, vmax) rather than
                        # assuming the metric itself is in that range.
                        _norm = (_val - vmin) / (vmax - vmin)
                        _r, _g, _b, _ = cmap(np.clip(_norm, 0.0, 1.0))
                        _lum = 0.299 * _r + 0.587 * _g + 0.114 * _b
                        ax.text(j, i, f'{_val:.2f}', ha='center',
                                va='center',
                                color='white' if _lum < 0.5 else 'black')
            if ri == 0:
                ax.set_title(m)
            if ci == 0:
                ax.set_ylabel(f'{title}\n\n{param1}')
            if ri == n_rows - 1:
                ax.set_xlabel(param2)
        fig.colorbar(im, ax=list(axes[ri, :]), fraction=0.046, pad=0.02)
    return fig, axes


# ---------------------------
# Parameter space sweep
# ---------------------------

def print_summary_fig(param1='overlap', param2='frac_sensor',
                      param1_vals=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
                      param2_vals=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                      methods=None, methods_kwargs=None,
                      sector_levels=(2, 4, 8), n_sector=8, seed=None,
                      sim_kwargs=None, annotate=False, style='publication_ml',
                      save_dir='.', fname_base='dual_colour_correction',
                      split_pdfs=False, save_videos=True,
                      save_videos_truth_vs_corr='all', n_jobs=None,
                      show=True):
    """Heatmaps of correction quality over a 2-D sim parameter sweep.

    Sweeps ``param1`` (y-axis) against ``param2`` (x-axis) and draws, for
    each ``(methods, methods_kwargs)`` entry, a heatmap of each quality
    metric (colour = metric value). Each axis parameter is either a real
    :class:`DualColourSim` keyword argument (e.g. ``'overlap'``,
    ``'sig_strength'``) or the pseudo-parameter ``'frac_sensor'``, which ties
    **both** reporters' expressing fraction together (``frac_sig =
    frac_ctrl = frac_sensor`` at each grid point), so it reflects overall
    sensor coverage of the FOV. The grid is laid out as one **row per
    metric** and one **column per method** (whole-frame, per-sector,
    per-pixel ``correct_full_regress`` by default), so columns are directly
    comparable within a row (shared colour scale per row).

    Three metrics are shown:

    a. **signal recovered** — mean ``corr(corrected, true a)`` over the true
       signal pixels (higher = brighter = better).
    b. **haemodynamics left** — mean ``|corr(corrected, true b)|`` over the
       signal pixels, i.e. residual artefact (lower is better; the colormap
       is reversed so 'good' is still the bright end).
    c. **spatial pattern fidelity** — correlation *across sectors* between
       the per-sector response-amplitude map of ``corrected`` and that of
       the true signal ``a``, on an ``n_sector`` x ``n_sector`` grid
       (default 8x8; 1 = pattern perfectly faithful, 0 = unrelated, higher
       is better). Asks whether the recovered activity has the right
       relative magnitude in the right places; see :func:`_sector_amp_corr`
       for why it is centred and compared across (not within) sectors.
       Can go negative, which the fixed [0, 1] colour scale clips to the
       dark end.

    Parameters
    ----------
    param1, param2 : str
        Names of the two swept parameters (``param1`` -> heatmap y-axis,
        ``param2`` -> heatmap x-axis). Either a :class:`DualColourSim`
        keyword argument, or the pseudo-parameter ``'frac_sensor'`` (see
        above).
    param1_vals, param2_vals : iterable
        Values swept for ``param1`` / ``param2`` (bottom / left = lowest).
    methods : list of callable or None
        One :mod:`signal_correction` correction function per heatmap
        column, e.g. ``[correct_full_regress, correct_full_regress,
        correct_full_regress]``; each is paired positionally with the
        matching entry of ``methods_kwargs``. Defaults to
        ``correct_full_regress`` three times (whole-frame / per-sector /
        per-pixel, see ``methods_kwargs``).
    methods_kwargs : list of dict or None
        Per-column keyword overrides for the matching ``methods`` entry
        (see :func:`_call_corr_func`), e.g. ``[{'fit_mode': 'global'},
        {'fit_mode': 'per_sector'}, {'fit_mode': 'per_pixel'}]``. Must be
        the same length as ``methods``. Defaults to that same
        whole-frame / per-sector / per-pixel triple when ``methods`` is
        also left at its default; otherwise defaults to an empty dict per
        entry (only the shared preprocessing defaults are used).
    sector_levels : tuple of int
        Shared sector grid, injected as ``sector_levels`` / ``f0_n_sectors``
        for methods that accept them (see :func:`_call_corr_func`).
    n_sector : int
        Sectors per axis for the **spatial pattern fidelity** metric (grid is
        ``n_sector`` x ``n_sector``, default 8x8; see
        :func:`_sector_amp_corr`). Independent of ``sector_levels``.
    seed : int
        RNG seed, shared across all grid points (paired comparison).
    sim_kwargs : dict or None
        Extra keyword arguments forwarded to :class:`DualColourSim` (held
        fixed across the sweep, e.g. ``noise_ctrl``). Must not include
        ``param1`` / ``param2`` (or the kwargs they drive, e.g.
        ``frac_sig`` / ``frac_ctrl`` for ``'frac_sensor'``).
    annotate : bool
        Overlay each cell's numeric value. Defaults to False (colour-only
        heatmaps).
    style : str or None
        Matplotlib stylesheet applied while building/saving the figure(s).
        Defaults to ``'publication_ml'`` (saves vector PDF with editable
        fonts). Pass None to use the active rcParams.
    save_dir : str or None
        Directory the PDF(s) / videos are written to (created if missing).
        Defaults to the current working directory. Pass None to skip
        saving.
    fname_base : str
        Base filename. Every sim parameter (default or overridden) and the
        swept ranges are appended as a tag, so the saved names fully
        identify the dataset.
    split_pdfs : bool
        If True, save one PDF per metric (``<base>__<tag>_<metric>.pdf``);
        otherwise save a single combined PDF (``<base>__<tag>.pdf``).
    save_videos : bool
        If True, also save TIFF-stack videos of a representative sim (the
        un-swept dataset at default parameter values): the ground-truth
        signal/haemo, the recorded sensor/control channels, and the
        ground-truth vs corrected-recovery comparison(s) (see
        :func:`save_sim_videos`).
    save_videos_truth_vs_corr : ``'all'`` or iterable of int or None
        Which ``methods`` entries (by index) get a truth-vs-corrected
        video — one TIFF each, named from that entry's column label (see
        :func:`_method_labels`). ``'all'`` (default) does every entry in
        ``methods``; pass a subset of indices, or ``()`` / ``None`` for
        none.
    n_jobs : int or None
        Number of worker processes used for the sweep; grid points are
        independent and run in parallel. None (default) or -1 uses every
        core, capped at the number of grid points; 1 runs serially
        in-process. Results do not depend on ``n_jobs`` (each grid point
        owns its sim, and ``seed`` is shared).
    show : bool
        Call ``plt.show()`` before returning.

    Returns
    -------
    out : SimpleNamespace
        ``.figs`` list of figures (one if combined, one per metric if
        split); ``.metrics`` dict ``attr -> label ->
        (n_param1, n_param2)`` ndarray; ``.method_labels`` the column
        labels aligned with ``methods``; ``.param1`` / ``.param2`` the
        swept parameter names; ``.param1_vals`` / ``.param2_vals`` the
        axis values; ``.saved`` list of written PDF paths; ``.saved_videos``
        list of written TIFF paths.
    """
    _default_methods = methods is None
    if methods is None:
        methods = [sc.correct_full_regress] * 4
    if methods_kwargs is None:
        methods_kwargs = ([{'fit_mode': 'global'}, {'fit_mode': 'per_sector'},
                          {'fit_mode': 'per_pixel'},
                          {'fit_mode': 'per_pixel_clean_reg'}]
                         if _default_methods else [{} for _ in methods])
    if len(methods_kwargs) != len(methods):
        raise ValueError('methods and methods_kwargs must have the same '
                         f'length (got {len(methods)} and '
                         f'{len(methods_kwargs)})')
    method_labels = _method_labels(methods, methods_kwargs)

    sim_kwargs = dict(sim_kwargs or {})
    param1_vals = tuple(param1_vals)
    param2_vals = tuple(param2_vals)

    print(f'### sweeping {param1} {param1_vals} x '
          f'{param2} {param2_vals} '
          f'({len(param1_vals) * len(param2_vals)} sims) ###')
    metrics = _sweep_metrics_2d(param1, param1_vals, param2, param2_vals,
                                methods, methods_kwargs, method_labels,
                                sector_levels, seed, sim_kwargs,
                                n_jobs=n_jobs, n_sector=n_sector)
    # 'raw' (uncorrected) is always computed by _sweep_metrics_2d as a
    # baseline column, alongside every requested correction method.
    plot_labels = ['raw'] + list(method_labels)

    # Filename stem encodes every sim parameter + the swept ranges.
    _tag = _sim_param_tag(sim_kwargs, param1, param1_vals, param2,
                          param2_vals)
    stem = f'{fname_base}__{_tag}'

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    figs = []
    saved = []
    _ctx = (plt.style.context(style) if style is not None
            else plt.style.context({}))
    with _ctx:
        if split_pdfs:
            # One figure (and one PDF) per metric.
            for spec in _FIG_METRICS:
                fig, _ = _render_heatmap_grid(
                    metrics, [spec], plot_labels, param1, param1_vals,
                    param2, param2_vals, annotate)
                fig.suptitle(f'{param1} x {param2} — {spec[0]}')
                figs.append(fig)
                if save_dir is not None:
                    _p = os.path.join(save_dir, f'{stem}_{spec[0]}.pdf')
                    fig.savefig(_p)
                    saved.append(_p)
        else:
            # Single combined figure / PDF.
            fig, _ = _render_heatmap_grid(
                metrics, _FIG_METRICS, plot_labels, param1, param1_vals,
                param2, param2_vals, annotate)
            fig.suptitle(f'Dual-colour correction: {param1} x {param2}')
            figs.append(fig)
            if save_dir is not None:
                _p = os.path.join(save_dir, f'{stem}.pdf')
                fig.savefig(_p)
                saved.append(_p)

        if show:
            plt.show()

    for _p in saved:
        print(f'saved figure -> {_p}')

    # TIFF-stack videos of a representative (un-swept) sim.
    saved_videos = []
    if save_videos and save_dir is not None:
        if save_videos_truth_vs_corr is None:
            _tvc_idx = []
        elif save_videos_truth_vs_corr == 'all':
            _tvc_idx = list(range(len(methods)))
        else:
            _tvc_idx = list(save_videos_truth_vs_corr)
        _rep = DualColourSim(seed=seed, **sim_kwargs)
        saved_videos = save_sim_videos(
            _rep, save_dir=save_dir, stem=stem,
            methods=[methods[i] for i in _tvc_idx],
            methods_kwargs=[methods_kwargs[i] for i in _tvc_idx],
            method_labels=[method_labels[i] for i in _tvc_idx],
            sector_levels=sector_levels)
        for _p in saved_videos:
            print(f'saved video  -> {_p}')

    return SimpleNamespace(figs=figs, metrics=metrics,
                           method_labels=method_labels,
                           param1=param1, param2=param2,
                           param1_vals=param1_vals,
                           param2_vals=param2_vals,
                           saved=saved, saved_videos=saved_videos)


def _print_table(sim, results, methods, sector_levels):
    """Print a compact recovery / leakage comparison table."""
    s = sim.summary()
    print('=' * 60)
    print(f'DualColourSim  {sim.n_t}x{sim.n_x}x{sim.n_y}  '
          f'haemo_strength={sim.haemo_strength:.2f}  sig_strength={sim.sig_strength:.2f}  '
          f'ctrl_noise={s.ctrl_noise:.2f}')
    print(f'expr: sig={s.frac_sig_expressing:.0%} ctrl='
          f'{s.frac_ctrl_expressing:.0%}  overlap≈{s.measured_overlap:+.2f}  '
          f'signal px w/ control={s.frac_signal_pixels_with_control:.0%}')
    print(f'levels={tuple(sector_levels)}  '
          f'signal pixels={s.n_signal_pixels}')
    print('-' * 60)
    print(f'{"method":<22}{"recovery(a)":>13}{"haemo leak":>13}'
          f'{"signed":>10}{"Δ vs raw":>11}')
    print('-' * 70)
    raw_rec = results['raw'].recovery
    for name in ['raw'] + list(methods):
        m = results[name]
        _delta = '' if name == 'raw' else f'{m.recovery - raw_rec:+.3f}'
        print(f'{name:<22}{m.recovery:>13.3f}{m.leakage:>13.3f}'
              f'{m.leakage_signed:>+10.3f}{_delta:>11}')
    print('=' * 70)
    print('recovery(a) = mean corr(corrected, true signal) over source '
          'pixels (higher is better)')
    print('haemo leak  = mean |corr(corrected, true haemo)| over all '
          'pixels (lower is better)')
    print('signed      = the same WITHOUT |·|: ~0 = artefact removed, '
          '<0 = over-subtracted')
    print('              (inverted), >0 = under-subtracted. |·| alone '
          'penalises an unbiased')
    print('              but noisy residual more than a consistently '
          'biased one.')


# =============================================================================
# BayesNF field-inference demo
# =============================================================================

def demo_infer_field_bayesnf(sim=None, bin_factor=4, num_epochs=800,
                             ensemble_size=4, seed=0, expr_thresh_frac=0.1,
                             backend='torch_mps',
                             save_dir='~/Desktop',
                             stem='infer_field_bayesnf_demo',
                             show=False, verbose=True, **sim_kwargs):
    """Infer a DualColourSim's ground-truth field across *missing* pixels.

    Builds (or accepts) a genuine :class:`DualColourSim` — real
    time-varying neural signal ``a`` and haemodynamics ``b``, with the
    biophysical dual-colour expression layout: the **sensor** expresses in
    the **neuropil** only and the **static control** in the **cell bodies**
    (plus an ``overlap`` fraction of neuropil). Each channel's observed
    field is ``bg + F0·expr·mod``, where the expression-independent
    modulation ``mod`` (``sim.sig_mod`` / ``sim.ctrl_mod``) is the real
    time-varying signal it tracks — defined at *every* pixel, including the
    low-expression regions the channel cannot report.

    Only the two spatially heterogeneous observed channels are given to
    :func:`~signal_correction.infer_field_bayesnf`, with near-zero-expression
    pixels **masked out** (set to NaN, so they are excluded from the fit).
    The helper then infers each channel's smooth field everywhere, and we
    score how well the inferred per-pixel dynamics match the ground-truth
    ``mod`` separately over the **observed** (expressing) pixels
    [denoising] and the **missing** (masked) pixels [spatial inference].

    Parameters
    ----------
    sim : DualColourSim or None
        Simulation to use. If None, one is built from ``seed`` /
        ``sim_kwargs`` using :class:`DualColourSim`'s own defaults, with
        only ``n_t`` shortened (500 vs. the default 2500) so the BayesNF
        fit finishes quickly.
    bin_factor : int
        Spatial downsample factor passed to ``infer_field_bayesnf``.
    num_epochs : int
        BayesNF MAP training epochs per channel.
    ensemble_size : int
        BayesNF MAP ensemble size.
    seed : int
        RNG / PRNG seed (sim build and BayesNF fit).
    expr_thresh_frac : float
        A pixel is treated as *observed* where its channel expression
        exceeds ``expr_thresh_frac × max(expr)``; below that it is a
        near-zero / *missing* pixel (masked out of the fit). Default 0.1.
    backend : {'bayesnf', 'torch_mps', 'torch_cpu'}
        Forwarded to :func:`~signal_correction.infer_field_bayesnf`.
        ``'torch_mps'`` is markedly faster on Apple-Silicon and matches
        ``'bayesnf'`` closely (corr >= 0.999); prefer it for quick runs.
    save_dir : str
        Output directory (``~`` is expanded). Created if missing.
    stem : str
        Filename stem for the saved figures / arrays.
    show : bool
        If True, leave the figures open (``plt.show``); else close them.
    verbose : bool
        Print the metric table and saved paths.
    **sim_kwargs
        Forwarded to :class:`DualColourSim` when ``sim`` is None.

    Returns
    -------
    out : SimpleNamespace
        ``.sim``, ``.sig_inf`` / ``.ctrl_inf`` inferred fields,
        ``.mask_sig`` / ``.mask_ctrl`` observed-pixel masks,
        ``.diagnostics`` (from ``infer_field_bayesnf``), ``.metrics``
        (per-channel corr over observed vs. missing pixels), and
        ``.out_paths`` the written files.
    """
    if sim is None:
        # DualColourSim's own defaults (real FOV size, expression layout,
        # noise, etc.) unchanged — only n_t is drastically shortened so the
        # BayesNF fit finishes quickly; everything else stays as close to
        # the standard sim as possible.
        _defaults = dict(n_t=500, seed=seed)
        _defaults.update(sim_kwargs)
        sim = DualColourSim(**_defaults)

    # Observed channels and their ground-truth modulation / expression.
    # Codebase convention: s1 = control, s2 = signal.
    thr_sig = expr_thresh_frac * float(sim.expr_sig.max())
    thr_ctrl = expr_thresh_frac * float(sim.expr_ctrl.max())
    mask_sig = sim.expr_sig >= thr_sig        # (X, Y) True = observed
    mask_ctrl = sim.expr_ctrl >= thr_ctrl

    # Mask near-zero-expression pixels to NaN so they are 'missing' (dropped
    # from the BayesNF fit); the helper infers the field there.
    sig_in = sim.sig_chan.astype(np.float32).copy()
    ctrl_in = sim.ctrl_chan.astype(np.float32).copy()
    sig_in[:, ~mask_sig] = np.nan
    ctrl_in[:, ~mask_ctrl] = np.nan

    s1_real, s2_real, diag = sc.infer_field_bayesnf(
        ctrl_in, sig_in, bin_factor=bin_factor, num_epochs=num_epochs,
        ensemble_size=ensemble_size, seed=seed, dtype=np.float32,
        backend=backend, verbose=verbose)
    ctrl_inf = np.asarray(s1_real, dtype=np.float32)
    sig_inf = np.asarray(s2_real, dtype=np.float32)

    # Per-pixel corr of the inferred field's dynamics with the ground-truth
    # modulation (scale/offset-invariant, so the unknown expression scaling
    # at masked pixels does not matter). Raw = noisy observed channel.
    cmap_sig = _pix_corr(sig_inf, sim.sig_mod)
    cmap_ctrl = _pix_corr(ctrl_inf, sim.ctrl_mod)
    cmap_sig_raw = _pix_corr(sim.sig_chan, sim.sig_mod)
    cmap_ctrl_raw = _pix_corr(sim.ctrl_chan, sim.ctrl_mod)

    def _mn(cmap, m):
        return float(np.nanmean(cmap[m])) if m.any() else float('nan')

    metrics = SimpleNamespace(
        frac_missing_sig=float((~mask_sig).mean()),
        frac_missing_ctrl=float((~mask_ctrl).mean()),
        sig_raw_obs=_mn(cmap_sig_raw, mask_sig),
        sig_inf_obs=_mn(cmap_sig, mask_sig),
        sig_inf_missing=_mn(cmap_sig, ~mask_sig),
        ctrl_raw_obs=_mn(cmap_ctrl_raw, mask_ctrl),
        ctrl_inf_obs=_mn(cmap_ctrl, mask_ctrl),
        ctrl_inf_missing=_mn(cmap_ctrl, ~mask_ctrl))

    if verbose:
        print('\n' + '=' * 68)
        print('infer_field_bayesnf demo — recovery of true modulation '
              '(corr, higher=better)')
        print('-' * 68)
        print(f'{"channel":<9}{"missing%":>10}{"raw@obs":>11}'
              f'{"inf@obs":>11}{"inf@missing":>14}')
        print(f'{"signal":<9}{metrics.frac_missing_sig*100:>9.0f}%'
              f'{metrics.sig_raw_obs:>11.3f}{metrics.sig_inf_obs:>11.3f}'
              f'{metrics.sig_inf_missing:>14.3f}')
        print(f'{"control":<9}{metrics.frac_missing_ctrl*100:>9.0f}%'
              f'{metrics.ctrl_raw_obs:>11.3f}{metrics.ctrl_inf_obs:>11.3f}'
              f'{metrics.ctrl_inf_missing:>14.3f}')
        print('=' * 68)
        print('raw@obs   : noisy channel vs. truth on expressing pixels')
        print('inf@obs   : inferred field vs. truth on expressing pixels '
              '(denoising)')
        print('inf@missing: inferred field vs. truth on masked pixels '
              '(inference across holes)')

    save_dir = os.path.expanduser(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    out_paths = []

    # (name, observed channel, masked input, ground-truth mod, inferred,
    #  observed-mask, corr map)
    _channels = (
        ('signal', sim.sig_chan, sig_in, sim.sig_mod, sig_inf,
         mask_sig, cmap_sig),
        ('control', sim.ctrl_chan, ctrl_in, sim.ctrl_mod, ctrl_inf,
         mask_ctrl, cmap_ctrl))
    t_star = int(np.argmax(sim.sig_mod.reshape(sim.n_t, -1).var(axis=1)))
    _cm = plt.cm.magma.copy()
    _cm.set_bad('0.15')       # masked (NaN) pixels render dark grey

    # --- Figure 1: per channel — ground truth / recorded / masked /
    # inferred / corr(truth, inferred). Columns 1-2 are unmasked; the
    # cyan contour marks the observed-region border on the masked panels. ---
    fig1, axes = plt.subplots(2, 5, figsize=(16.5, 6.8),
                              constrained_layout=True)
    for _row, (_name, _obs, _mskin, _mod, _inf, _m, _cmap) in enumerate(
            _channels):
        _vmin = float(np.percentile(_obs[t_star], 1))
        _vmax = float(np.percentile(_obs[t_star], 99))
        _mlo = float(np.percentile(_mod[t_star], 1))
        _mhi = float(np.percentile(_mod[t_star], 99))
        # (title, image, cmap, vmin, vmax, draw mask contour)
        _panels = [
            ('ground truth (no mask)', _mod[t_star], 'magma', _mlo, _mhi,
             False),
            ('recorded (heterogeneous)', _obs[t_star], _cm, _vmin, _vmax,
             False),
            ('masked (low SNR removed)', _mskin[t_star], _cm, _vmin, _vmax,
             True),
            ('inferred (BayesNF)', _inf[t_star], _cm, _vmin, _vmax, True),
            ('corr(truth, inferred)', _cmap, 'viridis', 0.0, 1.0, True)]
        for _col, (_ttl, _img, _cc, _lo, _hi, _draw) in enumerate(_panels):
            _ax = axes[_row, _col]
            _im = _ax.imshow(_img, cmap=_cc, vmin=_lo, vmax=_hi)
            if _draw:
                _ax.contour(_m.astype(float), levels=[0.5], colors='cyan',
                            linewidths=0.6)
            _ax.set_title(f'{_name}: {_ttl}', fontsize=9)
            _ax.set_xticks([])
            _ax.set_yticks([])
            fig1.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
    fig1.suptitle(f'BayesNF field inference from heterogeneous expression '
                  f'— frame t={t_star}  (cyan = observed-region border)',
                  fontsize=12)
    _p1 = os.path.join(save_dir, f'{stem}_spatial.png')
    fig1.savefig(_p1, dpi=150)
    out_paths.append(_p1)

    # --- Figure 2: temporal recovery at an observed vs. a missing pixel ---
    def _z(a):
        a = np.asarray(a, dtype=np.float64)
        return (a - a.mean()) / (a.std() + 1e-9)

    fig2, axes = plt.subplots(2, 2, figsize=(13, 7),
                              constrained_layout=True)
    for _row, (_name, _obs, _mskin, _mod, _inf, _m, _cmap) in enumerate(
            _channels):
        _modvar = _mod.var(axis=0)
        for _col, (_where, _sel) in enumerate(
                (('observed pixel', _m), ('missing pixel', ~_m))):
            _scores = np.where(_sel, _modvar, -np.inf)
            _px = tuple(int(_v) for _v in np.unravel_index(
                int(np.argmax(_scores)), (sim.n_x, sim.n_y)))
            _ax = axes[_row, _col]
            _ax.plot(_z(_obs[:, _px[0], _px[1]]), color='0.7', lw=0.7,
                     label='observed (noisy)')
            _ax.plot(_z(_mod[:, _px[0], _px[1]]), color='k', lw=1.6,
                     label='ground-truth mod')
            _ax.plot(_z(_inf[:, _px[0], _px[1]]), color='crimson', lw=1.2,
                     label='inferred')
            _r = float(np.corrcoef(_mod[:, _px[0], _px[1]],
                                   _inf[:, _px[0], _px[1]])[0, 1])
            _ax.set_title(f'{_name}: {_where} {_px}  (r={_r:.3f})',
                          fontsize=10)
            _ax.set_xlabel('frame')
            _ax.set_ylabel('z-scored')
            if _row == 0 and _col == 0:
                _ax.legend(fontsize=8, frameon=False)
    fig2.suptitle('BayesNF temporal recovery — observed (denoised) vs. '
                  'missing (inferred) pixels', fontsize=12)
    _p2 = os.path.join(save_dir, f'{stem}_traces.png')
    fig2.savefig(_p2, dpi=150)
    out_paths.append(_p2)

    # --- Figure 3: whole-frame fluorescence — recorded vs. inferred ---
    fig3, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.2),
                                      constrained_layout=True)
    _channel_colors = {'signal': ('seagreen', 'crimson'),
                       'control': ('slategray', 'teal')}
    # Per-pixel recovery over observed (expressing) pixels — the meaningful
    # measure, since the whole-frame spatial mean's ``frame r`` washes out
    # the fields the same way it does in run_regress (see that figure).
    _px_raw = {'signal': metrics.sig_raw_obs, 'control': metrics.ctrl_raw_obs}
    _px_inf = {'signal': metrics.sig_inf_obs, 'control': metrics.ctrl_inf_obs}
    for _name, _obs, _mskin, _mod, _inf, _m, _cmap in _channels:
        _raw_c, _inf_c = _channel_colors[_name]
        ax_l.plot(_z(_obs.mean(axis=(1, 2))), color=_raw_c, lw=1.2,
                 label=f'{_name} (raw)')
        _mod_avg = _mod.mean(axis=(1, 2))
        _obs_avg = _obs.mean(axis=(1, 2))
        _r_raw = float(np.corrcoef(_mod_avg, _obs_avg)[0, 1])
        _r = float(np.corrcoef(_mod_avg, _inf.mean(axis=(1, 2)))[0, 1])
        _ls = '-' if _name == 'signal' else '--'
        ax_r.plot(_z(_mod_avg), color='k', lw=1.8, ls=_ls,
                 label=f'ground truth ({_name})')
        ax_r.plot(_z(_obs_avg), color=_raw_c, lw=1.0, ls=_ls, alpha=0.6,
                 label=f'{_name} raw (frame r={_r_raw:+.2f}, '
                       f'pixel r={_px_raw[_name]:+.2f})')
        ax_r.plot(_z(_inf.mean(axis=(1, 2))), color=_inf_c, lw=1.2,
                 ls=_ls, label=f'{_name} inferred (frame r={_r:+.2f}, '
                               f'pixel r={_px_inf[_name]:+.2f})')
    ax_l.set_title('recorded whole-frame fluorescence', fontsize=10)
    ax_l.set_xlabel('frame')
    ax_l.set_ylabel('z-scored')
    ax_l.legend(fontsize=8, frameon=False)
    ax_r.set_title('inferred whole-frame fluorescence vs. ground truth\n'
                  'pixel r = mean per-pixel recovery over observed pixels',
                  fontsize=10)
    ax_r.set_xlabel('frame')
    ax_r.set_ylabel('z-scored')
    ax_r.legend(fontsize=8, frameon=False)
    fig3.suptitle('BayesNF whole-frame fluorescence — recorded vs. '
                 'inferred', fontsize=12)
    _p3 = os.path.join(save_dir, f'{stem}_channels.png')
    fig3.savefig(_p3, dpi=150)
    out_paths.append(_p3)

    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)

    # Persist the arrays for reproducibility / downstream inspection.
    _pnpz = os.path.join(save_dir, f'{stem}_arrays.npz')
    np.savez_compressed(
        _pnpz, sig_chan=sim.sig_chan, ctrl_chan=sim.ctrl_chan,
        sig_mod=sim.sig_mod, ctrl_mod=sim.ctrl_mod,
        sig_inf=sig_inf, ctrl_inf=ctrl_inf,
        mask_sig=mask_sig, mask_ctrl=mask_ctrl)
    out_paths.append(_pnpz)

    if verbose:
        for _p in out_paths:
            print(f'saved: {_p}')

    return SimpleNamespace(
        sim=sim, sig_inf=sig_inf, ctrl_inf=ctrl_inf,
        mask_sig=mask_sig, mask_ctrl=mask_ctrl,
        diagnostics=diag, metrics=metrics, out_paths=out_paths)
