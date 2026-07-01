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

import os, inspect
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from . import signal_correction as sc
except ImportError:
    from lak_exp import signal_correction as sc


__all__ = ['DualColourSim', 'run_regress', 'sweep_regress',
           'print_summary_fig', 'save_sim_videos']


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
        c(t,x,y) = bg_sig  + F0_s·(1 + sig_dff·a) ·(1 − haemo_frac·phi_s(b)) + noise
        d(t,x,y) = bg_ctrl + F0_c                ·(1 − haemo_frac·phi_c(b)) + noise

    Haemodynamics enter as a multiplicative *absorption* on the emitted
    light, so the haemo fluctuation a pixel actually shows scales with how
    much fluorophore it expresses. A signal-only pixel (``expr_ctrl≈0``)
    therefore has **no haemodynamic readout in the control channel** —
    exactly the case where per-pixel regression has nothing local to
    regress against and spatial averaging must borrow from neighbours.

    Expression layout (see ``_make_expression``) mirrors a real dual-colour
    FOV: small **cell bodies** (somata) shared by both reporters, embedded
    in a smooth **neuropil** wash. The **sensor** (e.g. GRAB) expresses in
    the neuropil only — never in cell bodies — while the **static**
    reporter always fills the cell bodies plus an ``overlap`` fraction of
    the surrounding neuropil.

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
    haemo_frac : float
        Fractional brightness modulation depth of the haemodynamics (e.g.
        0.15 = ±15% absorption swings on the resting fluorescence).
    sig_dff : float
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
    expr_smooth : float or None
        Spatial smoothness (px) of the background brightness wash. Defaults
        to ``max(n_x, n_y) / 2`` — a broad, mostly-smooth gradient across
        the whole FOV (large = smoother / more uniform).
    cell_size : float or None
        Characteristic radius (px) of the cell bodies. Defaults to
        ``max(n_x, n_y) / 40`` (small nucleus-sized somata).
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(self, n_t=2500, n_x=64, n_y=64,
                 n_haemo_modes=3, n_sources=18,
                 haemo_tau=6.0, haemo_frac=0.25, sig_dff=0.6,
                 f0_sig=100.0, f0_ctrl=100.0,
                 bg_sig=20.0, bg_ctrl=20.0,
                 noise_sig=1.0, noise_ctrl=1.5,
                 nonlin_sig=0.5, nonlin_ctrl=0.3,
                 overlap=0.3, frac_sig=0.9, frac_ctrl=0.9,
                 expr_smooth=None, cell_size=None, seed=0):
        self.n_t = int(n_t)
        self.n_x = int(n_x)
        self.n_y = int(n_y)
        self.n_haemo_modes = int(n_haemo_modes)
        self.n_sources = int(n_sources)
        self.haemo_tau = float(haemo_tau)
        self.haemo_frac = float(haemo_frac)
        self.sig_dff = float(sig_dff)
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
        self.expr_smooth = (float(expr_smooth) if expr_smooth is not None
                            else max(self.n_x, self.n_y) / 2.0)
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
        # set per channel by haemo_frac · expression in _assemble_channels.
        latent /= (latent.std() + 1e-9)

        self.haemo = latent.astype(np.float32)      # b (ground-truth haemo)
        self._haemo_modes_t = modes_t
        self._haemo_modes_s = modes_s

    @staticmethod
    def _wash(field, floor=0.4):
        """Map a smooth field to a mostly-bright expression wash in [floor, 1].

        Rescales to [0, 1] then biases up to ``[floor, 1]`` so the whole
        FOV expresses (a smooth gradient wash, as in real neuropil), before
        'cell' holes are punched out. Returns shape (n_x, n_y).
        """
        _f = field - field.min()
        _p = float(_f.max())
        _f = _f / _p if _p > 1e-9 else _f
        return floor + (1.0 - floor) * _f

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
        - A smooth, mostly-bright brightness **wash** over the whole FOV
          (one broad gradient set by ``expr_smooth``).
        - **Sensor** (``expr_sig``, e.g. GRAB): the wash over the *neuropil*
          only — i.e. everywhere *except* the cell bodies. Never expressed
          in somata.
        - **Static** (``expr_ctrl``): the wash over the cell bodies (always)
          *plus* the ``overlap`` fraction of the surrounding neuropil. At
          ``overlap=0`` the static is confined to cell bodies; at
          ``overlap=1`` it fills the whole neuropil too.

        Stored as ``expr_sig`` and ``expr_ctrl`` (n_x, n_y), with the
        shared ``cells`` mask kept for inspection.
        """
        self.cells = self._make_cells()
        _npil = 1.0 - self.cells                       # neuropil (around cells)
        _wash = self._wash(self._smooth_spatial(self.expr_smooth))

        # Sensor: neuropil only, never in cell bodies.
        self.expr_sig = (_wash * _npil).astype(np.float32)

        # Static: cell bodies always, plus `overlap` fraction of neuropil.
        _ov = self.overlap
        if _ov >= 1.0 - 1e-6:
            _static_npil = _npil
        elif _ov <= 1e-6:
            _static_npil = np.zeros_like(_npil)
        else:
            # Pick a contiguous `_ov` fraction of the neuropil via a smooth
            # selector field thresholded over neuropil pixels only.
            _sel = self._smooth_spatial(self.expr_smooth)
            _npil_vals = _sel[self.cells < 0.5]
            _thr = np.quantile(_npil_vals, 1.0 - _ov)
            _static_npil = ((_sel >= _thr) & (self.cells < 0.5)).astype(
                np.float32)
        _static_support = np.maximum(self.cells, _static_npil)
        self.expr_ctrl = (_wash * _static_support).astype(np.float32)

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
        it is scaled by ``sig_dff`` and the local signal-fluorophore
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

        # Channel-specific nonlinear haemodynamic absorption (zero-mean
        # fractional modulation of the emitted light).
        absorb_s = -self.haemo_frac * self._saturate(self.haemo,
                                                     self.nonlin_sig)
        absorb_c = -self.haemo_frac * self._saturate(self.haemo,
                                                     self.nonlin_ctrl)

        noise_s = self.noise_sig * self.rng.standard_normal(
            (self.n_t, self.n_x, self.n_y))
        noise_c = self.noise_ctrl * self.rng.standard_normal(
            (self.n_t, self.n_x, self.n_y))

        # c = bg + F0_s·(1 + dff·a)·(1 + absorb_s) + noise
        # d = bg + F0_c·(1 + absorb_c) + noise
        self.sig_chan = (
            self.bg_sig
            + f0_s * (1.0 + self.sig_dff * self.signal) * (1.0 + absorb_s)
            + noise_s).astype(np.float32)
        self.ctrl_chan = (
            self.bg_ctrl + f0_c * (1.0 + absorb_c) + noise_c).astype(
            np.float32)


# =============================================================================
# Recovery assessment
# =============================================================================

def _pix_corr(u, v):
    """Per-pixel Pearson correlation along time; shapes (T, X, Y) -> (X, Y)."""
    u = u - u.mean(axis=0)
    v = v - v.mean(axis=0)
    num = (u * v).sum(axis=0)
    den = np.sqrt((u ** 2).sum(axis=0) * (v ** 2).sum(axis=0)) + 1e-12
    return num / den


def _auc_separation(score_map, mask):
    """AUC for how well ``score_map`` separates ``mask`` pixels from the rest.

    Rank-based (Mann–Whitney U) area-under-curve in ``[0, 1]``: the
    probability that a randomly chosen ``mask``-True pixel has a higher
    score than a randomly chosen background pixel. 0.5 = no separation,
    1.0 = the score map perfectly localises the masked region.

    Parameters
    ----------
    score_map : (X, Y) ndarray
        Per-pixel score (here, corr(corrected, true signal)).
    mask : (X, Y) bool ndarray
        True at the pixels that *should* score high (the true signal pixels).

    Returns
    -------
    auc : float
        Separation AUC, or NaN if either group is empty.
    """
    pos = score_map[mask].ravel()
    neg = score_map[~mask].ravel()
    if pos.size == 0 or neg.size == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind='mergesort')
    ranks = np.empty(allv.size, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    r_pos = ranks[:pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0)
                 / (pos.size * neg.size))


def _assess(corrected, sim):
    """Score one corrected stack against the ground-truth signal/haemo.

    Parameters
    ----------
    corrected : (T, X, Y) ndarray
        Corrected signal channel (normalised units; scale-invariant
        correlations are used so units do not matter).
    sim : DualColourSim
        The simulation holding the ground-truth ``signal`` and ``haemo``.

    Returns
    -------
    metrics : SimpleNamespace
        ``.recovery`` mean corr(corrected, true signal) over signal pixels;
        ``.leakage`` mean |corr(corrected, true haemo)| over signal pixels
        (where haemodynamics actually contaminate the observable signal);
        ``.recovery_all`` mean recovery over every pixel;
        ``.specificity`` AUC for how well the recovery map localises the
        true signal pixels against the background (0.5 = none, 1 = perfect).
    """
    corrected = np.asarray(corrected, dtype=np.float32)
    rec_map = _pix_corr(corrected, sim.signal)
    leak_map = np.abs(_pix_corr(corrected, sim.haemo))
    return SimpleNamespace(
        recovery=float(rec_map[sim.sig_mask].mean()),
        leakage=float(leak_map[sim.sig_mask].mean()),
        recovery_all=float(rec_map.mean()),
        specificity=_auc_separation(rec_map, sim.sig_mask))


def _run_method(name, s1, s2, sector_levels, verbose, corr_kwargs=None):
    """Dispatch a correction method by name, returning the corrected stack.

    All methods share the same preprocessing (smoothing / airPLS / robust
    loss) and the same sector grid so the comparison isolates how each
    method couples the channels. ``corr_kwargs`` overrides these shared
    defaults for the dispatched method.
    """
    corr_kwargs = corr_kwargs or {}
    if name == 'two_stage':
        _kwargs = dict(sector_levels=sector_levels, f0_level='sector',
                       dtype=np.float32, verbose=verbose, smooth_window=10,
                       beta_loss='huber', beta_scale='mad', nn_slope=False)
        _kwargs.update(corr_kwargs)
        out, _ = sc.correct_two_stage_regress(s1, s2, **_kwargs)
    elif name in ('global', 'per_frame', 'per_sector', 'per_pixel'):
        # 'per_frame' is an alias for the whole-frame ('global') fit.
        _fit = 'global' if name in ('global', 'per_frame') else name
        _kwargs = dict(fit_mode=_fit, f0_level='sector',
                       f0_n_sectors=sector_levels[-1], dtype=np.float32,
                       verbose=verbose, smooth_window=10, beta_loss='huber',
                       beta_scale='mad', nn_slope=False)
        _kwargs.update(corr_kwargs)
        out, _ = sc.correct_full_regress(s1, s2, **_kwargs)
    else:
        raise ValueError(f"unknown method {name!r}")
    return out


# =============================================================================
# Top-level driver
# =============================================================================

def run_regress(sim=None, methods=None, sector_levels=(2, 4, 8),
                verbose=False, keep_corrected=False, corr_kwargs=None):
    """Run every correction method on a simulation and score recovery.

    Builds (or accepts) a :class:`DualColourSim`, corrects the signal
    channel against the static control with each method, and reports how
    well each recovers the ground-truth real signal ``a`` while suppressing
    the haemodynamics ``b``.

    Parameters
    ----------
    sim : DualColourSim or None
        Simulation to use. If None, a default one is built.
    methods : list of str or None
        Subset of ``['global', 'per_sector', 'per_pixel', 'two_stage']``.
        Defaults to all four.
    sector_levels : tuple of int
        Cascade levels for ``two_stage``; its finest level is also used as
        the sector grid for the ``full_regress`` modes (fair comparison).
    verbose : bool
        Forward verbose output from the correction functions.
    keep_corrected : bool
        If True, also return each method's corrected stack (memory heavy).
    corr_kwargs : dict or None
        Extra keyword arguments forwarded to the correction method (see
        :func:`_run_method`), overriding its shared defaults.

    Returns
    -------
    out : SimpleNamespace
        ``.sim`` the simulation; ``.results`` dict ``method -> metrics``
        (plus a ``'raw'`` baseline = uncorrected signal channel); and
        ``.corrected`` dict when ``keep_corrected`` is set.
    """
    if sim is None:
        sim = DualColourSim()
    if methods is None:
        methods = ['global', 'per_sector', 'per_pixel', 'two_stage']

    s1 = sim.ctrl_chan      # static control (regressor)
    s2 = sim.sig_chan       # signal channel (to be corrected)

    results = {}
    corrected_stacks = {}

    # Baseline: how well does the *uncorrected* signal channel track a,
    # and how much haemodynamics does it still carry.
    results['raw'] = _assess(s2, sim)

    for name in methods:
        if verbose:
            print(f'\n=== {name} ===')
        out = _run_method(name, s1, s2, sector_levels, verbose, corr_kwargs)
        results[name] = _assess(out, sim)
        if keep_corrected:
            corrected_stacks[name] = out

    _print_table(sim, results, methods, sector_levels)

    res = SimpleNamespace(sim=sim, results=results)
    if keep_corrected:
        res.corrected = corrected_stacks
    return res


def sweep_regress(param='overlap', values=(0.0, 0.25, 0.5, 0.75, 1.0),
                  methods=None, sector_levels=(2, 4, 8), seed=0,
                  sim_kwargs=None):
    """Sweep one simulation parameter and tabulate each method's recovery.

    Useful axes for the two-stage / per-pixel comparison:

    - ``'overlap'`` — fluorophore co-expression. As signal and control
      territories separate (overlap → negative), signal pixels lose their
      *local* control readout, so per-pixel regression has nothing to
      regress against while the two-stage cascade borrows haemodynamics
      from control-expressing neighbours.
    - ``'noise_ctrl'`` — control read-noise. Per-pixel regression injects
      that noise pixel-by-pixel; the cascade's spatial averages stay clean.
    - ``'frac_ctrl'`` — control sparsity (lower = sparser coverage).

    Parameters
    ----------
    param : str
        Name of the :class:`DualColourSim` keyword to vary.
    values : iterable
        Values of ``param`` to test.
    methods : list of str or None
        Methods to compare (see :func:`run_regress`).
    sector_levels : tuple of int
        Cascade levels (and the shared sector grid for full_regress).
    seed : int
        RNG seed (shared across points for a paired comparison).
    sim_kwargs : dict or None
        Extra keyword arguments forwarded to :class:`DualColourSim`.

    Returns
    -------
    out : SimpleNamespace
        ``.param``, ``.values`` and ``.recovery`` (dict ``method -> list``
        of recovery values, aligned with ``values``).
    """
    if methods is None:
        methods = ['global', 'per_sector', 'per_pixel', 'two_stage']
    sim_kwargs = dict(sim_kwargs or {})
    recovery = {m: [] for m in ['raw'] + list(methods)}

    for val in values:
        sim = DualColourSim(seed=seed, **{param: val}, **sim_kwargs)
        res = run_regress(sim=sim, methods=methods,
                          sector_levels=sector_levels, verbose=False)
        for name in recovery:
            recovery[name].append(res.results[name].recovery)

    print('\n' + '=' * 60)
    print(f'RECOVERY vs {param.upper()}  '
          f'(corr with true signal; higher=better)')
    print('-' * 60)
    _hdr = ''.join(f'{v:>9.2f}' for v in values)
    print(f'{param:<14}{_hdr}')
    print('-' * 60)
    for name in ['raw'] + list(methods):
        _row = ''.join(f'{v:>9.3f}' for v in recovery[name])
        print(f'{name:<14}{_row}')
    print('=' * 60)
    return SimpleNamespace(param=param, values=tuple(values),
                           recovery=recovery)


# =============================================================================
# Parameter-space summary figure
# =============================================================================

# Metrics shown in the summary figure: (attribute, panel title, colormap,
# higher-is-better). All heatmaps use the seaborn 'rocket' palette on a
# fixed [0, 1] scale; ``leakage`` uses the reversed palette so that 'good'
# is always the bright/warm end of every row.
_FIG_METRICS = (
    ('recovery', 'signal recovered\ncorr(corrected, true a)',
     'rocket', True),
    ('leakage', 'haemodynamics left\n|corr(corrected, true b)|',
     'rocket_r', False),
    ('specificity', 'spatial specificity\nAUC(localises a)',
     'rocket', True),
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
    'n_sources': 'nsrc', 'haemo_tau': 'htau', 'haemo_frac': 'hfrac',
    'sig_dff': 'sdff', 'f0_sig': 'f0s', 'f0_ctrl': 'f0c', 'bg_sig': 'bgs',
    'bg_ctrl': 'bgc', 'noise_sig': 'nzs', 'noise_ctrl': 'nzc',
    'nonlin_sig': 'nls', 'nonlin_ctrl': 'nlc', 'overlap': 'ov',
    'frac_sig': 'fsig', 'frac_ctrl': 'fctrl', 'expr_smooth': 'esm',
    'seed': 'sd',
}


def _fmt_num(v):
    """Compact, filename-safe formatting of a scalar parameter value."""
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, (int, float)):
        return f'{v:g}'
    return str(v)


def _sim_param_tag(sim_kwargs, overlap_vals=None, frac_sensor_vals=None):
    """Build a compact tag of every sim parameter (default or overridden).

    The :class:`DualColourSim` constructor defaults are read by
    introspection and overlaid with ``sim_kwargs`` so the tag reflects the
    exact dataset. When sweeping, ``overlap`` and the tied ``frac_sig`` /
    ``frac_ctrl`` are shown as ranges (``ov<lo>to<hi>`` / ``fsens<lo>to<hi>``)
    rather than single values.

    Parameters
    ----------
    sim_kwargs : dict
        Overrides passed to :class:`DualColourSim`.
    overlap_vals, frac_sensor_vals : iterable or None
        Swept axis values; when given, replace the corresponding scalar(s)
        with a ``<lo>to<hi>`` range token.

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

    parts = []
    for name, val in params.items():
        ab = _PARAM_ABBR.get(name, name)
        if name == 'overlap' and overlap_vals is not None:
            parts.append(f'{ab}{_fmt_num(min(overlap_vals))}to'
                         f'{_fmt_num(max(overlap_vals))}')
        elif name in ('frac_sig', 'frac_ctrl') and (
                frac_sensor_vals is not None):
            # Tied together on the sweep -> a single fsens range token.
            if name == 'frac_sig':
                parts.append(f'fsens{_fmt_num(min(frac_sensor_vals))}to'
                             f'{_fmt_num(max(frac_sensor_vals))}')
        else:
            parts.append(f'{ab}{_fmt_num(val)}')
    return '_'.join(parts)


def _normalise_u16(arr, vmin, vmax):
    """Scale a float array to uint16 over ``[vmin, vmax]`` (clipped)."""
    rng = vmax - vmin
    if rng <= 0:
        rng = 1.0
    out = (np.asarray(arr, dtype=np.float32) - vmin) / rng
    np.clip(out, 0.0, 1.0, out=out)
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
    def _range(arr):
        return (float(np.percentile(arr, 0.5)),
                float(np.percentile(arr, 99.5)))

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


# Short aliases for the truth-vs-corrected correction methods.
_TVC_METHOD_ALIASES = {
    'frame': 'per_frame', 'per_frame': 'per_frame', 'global': 'per_frame',
    'sector': 'per_sector', 'per_sector': 'per_sector',
    'pixel': 'per_pixel', 'per_pixel': 'per_pixel',
    'two_stage': 'two_stage',
}


def save_sim_videos(sim, save_dir='.', stem='dual_colour_sim', gap=4,
                    tvc_methods=('frame', 'sector', 'pixel'),
                    sector_levels=(2, 4, 8), corr_kwargs=None):
    """Save TIFF-stack 'videos' of the ground-truth and recorded fields.

    Always writes two ImageJ-compatible multipage TIFFs:

    - ``<stem>_signals_a_b.tif`` — the real signal ``a`` (neuromodulator)
      beside the haemodynamics ``b``, each normalised independently (they
      are different physical quantities).
    - ``<stem>_sensor_control.tif`` — the recorded signal sensor beside the
      static control sensor, sharing one intensity scale (same units).

    Plus one ``<stem>_truth_vs_corrected_<method>.tif`` per entry in
    ``tvc_methods`` — the ground-truth neuromodulatory signal ``a`` (left)
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
    tvc_methods : iterable of str
        Correction methods for the truth-vs-corrected videos; one file is
        written per entry. Accepts the short aliases ``'frame'``,
        ``'sector'``, ``'pixel'`` (mapped to ``per_frame`` / ``per_sector``
        / ``per_pixel``) as well as ``'two_stage'``. Empty / None skips
        them.
    sector_levels : tuple of int
        Cascade / sector grid for the corrections.
    corr_kwargs : dict or None
        Extra keyword arguments forwarded to the correction method (see
        :func:`_run_method`), overriding its shared defaults.

    Returns
    -------
    saved : list of str
        Paths actually written (empty if no TIFF writer is available).
    """
    os.makedirs(save_dir, exist_ok=True)
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
    for _name in (tvc_methods or []):
        _m = _TVC_METHOD_ALIASES.get(_name, _name)
        _corr = _run_method(_m, sim.ctrl_chan, sim.sig_chan,
                            sector_levels, False, corr_kwargs)
        _p_tc = os.path.join(save_dir, f'{stem}_truth_vs_corrected_{_m}.tif')
        if _write_two_panel_tiff(_p_tc, sim.signal, _corr,
                                 ('ground_truth_a', f'corrected_{_m}'),
                                 shared_norm=False, gap=gap):
            saved.append(_p_tc)
    return saved


def _sweep_metrics_2d(overlap_vals, frac_sensor_vals, methods, sector_levels,
                      seed, sim_kwargs, corr_kwargs=None):
    """Sweep overlap x sensor sparsity, collecting metrics per method.

    The sensor-sparsity axis ties ``frac_sig`` and ``frac_ctrl`` to the
    same value at each grid point (both reporters share an expressing
    fraction), so the axis reflects overall sensor coverage of the FOV.

    Returns
    -------
    out : dict
        ``out[metric_attr][method]`` is a 2-D ndarray of shape
        ``(n_overlap, n_frac_sensor)`` (row = overlap, col = frac_sensor);
        ``method`` includes ``'raw'`` (the uncorrected baseline).
    """
    names = ['raw'] + list(methods)
    metric_attrs = [m[0] for m in _FIG_METRICS]
    n_ov, n_fs = len(overlap_vals), len(frac_sensor_vals)
    out = {a: {n: np.full((n_ov, n_fs), np.nan) for n in names}
           for a in metric_attrs}
    for i, ov in enumerate(overlap_vals):
        for j, fs in enumerate(frac_sensor_vals):
            sim = DualColourSim(seed=seed, overlap=ov,
                                frac_sig=fs, frac_ctrl=fs, **sim_kwargs)
            res = run_regress(sim=sim, methods=methods,
                              sector_levels=sector_levels, verbose=False,
                              corr_kwargs=corr_kwargs)
            for a in metric_attrs:
                for n in names:
                    out[a][n][i, j] = getattr(res.results[n], a)
    return out


def _render_heatmap_grid(metrics, metric_specs, methods, overlap_vals,
                         frac_sensor_vals, annotate):
    """Render a (n_metric x n_method) heatmap grid into a new figure.

    Typography (fonts, title weight, tick sizes) is left to the active
    matplotlib stylesheet so the caller can wrap this in a style context.

    Returns
    -------
    fig, axes : Figure and (n_metric, n_method) ndarray of Axes.
    """
    n_rows = len(metric_specs)
    n_cols = len(methods)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.0 * n_cols + 0.9, 2.0 * n_rows),
                             squeeze=False)
    n_ov, n_fs = len(overlap_vals), len(frac_sensor_vals)
    for ri, (attr, title, cmap_name, _) in enumerate(metric_specs):
        # Every metric is drawn on a fixed [0, 1] rocket scale.
        cmap = _resolve_cmap(cmap_name)
        im = None
        for ci, m in enumerate(methods):
            ax = axes[ri, ci]
            arr = metrics[attr][m]
            im = ax.imshow(arr, origin='lower', aspect='auto', cmap=cmap,
                           vmin=0.0, vmax=1.0)
            ax.set_xticks(range(n_fs))
            ax.set_xticklabels([f'{v:g}' for v in frac_sensor_vals])
            ax.set_yticks(range(n_ov))
            ax.set_yticklabels([f'{v:g}' for v in overlap_vals])
            ax.tick_params(length=0)
            for _sp in ax.spines.values():
                _sp.set_visible(False)
            if annotate:
                for i in range(n_ov):
                    for j in range(n_fs):
                        _val = float(arr[i, j])
                        # Text colour from cell luminance (cmap-direction
                        # agnostic, since 'rocket' and 'rocket_r' differ).
                        _r, _g, _b, _ = cmap(_val)
                        _lum = 0.299 * _r + 0.587 * _g + 0.114 * _b
                        ax.text(j, i, f'{_val:.2f}', ha='center',
                                va='center',
                                color='white' if _lum < 0.5 else 'black')
            if ri == 0:
                ax.set_title(m)
            if ci == 0:
                ax.set_ylabel(f'{title}\n\noverlap')
            if ri == n_rows - 1:
                ax.set_xlabel('frac_sensor')
        fig.colorbar(im, ax=list(axes[ri, :]), fraction=0.046, pad=0.02)
    return fig, axes


def print_summary_fig(overlap_vals=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
                      frac_sensor_vals=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                      methods=None, sector_levels=(2, 4, 8), seed=0,
                      sim_kwargs=None, annotate=False, style='publication_ml',
                      save_dir='.', fname_base='dual_colour_correction',
                      split_pdfs=False, save_videos=True,
                      save_videos_truth_vs_corr=('frame', 'sector', 'pixel'),
                      corr_kwargs=None, show=True):
    """Heatmaps of correction quality over overlap x sensor sparsity.

    Sweeps the fluorophore ``overlap`` (y-axis) against the sensor sparsity
    ``frac_sensor`` (x-axis) and draws, for each correction method, a
    heatmap of each quality metric (colour = metric value). The
    ``frac_sensor`` axis sets **both** reporters' expressing fraction
    together (``frac_sig = frac_ctrl = frac_sensor`` at each grid point),
    so it reflects overall sensor coverage of the FOV. The grid is laid
    out as one **row per metric** and one **column per method**
    (``per_frame``, ``per_sector``, ``per_pixel`` by default), so columns
    are directly comparable within a row (shared colour scale per row).

    Three metrics are shown:

    a. **signal recovered** — mean ``corr(corrected, true a)`` over the true
       signal pixels (higher = brighter = better).
    b. **haemodynamics left** — mean ``|corr(corrected, true b)|`` over the
       signal pixels, i.e. residual artefact (lower is better; the colormap
       is reversed so 'good' is still the bright end).
    c. **spatial specificity** — AUC for how well the per-pixel recovery map
       localises the true signal pixels against the background (0.5 = none,
       1 = perfect; higher is better).

    Parameters
    ----------
    overlap_vals : iterable
        Fluorophore overlap values (heatmap y-axis, bottom = lowest).
    frac_sensor_vals : iterable
        Sensor sparsity values applied to both reporters (heatmap x-axis,
        left = sparsest).
    methods : list of str or None
        Methods to compare (one heatmap column each). Defaults to
        ``['per_frame', 'per_sector', 'per_pixel']``; add ``'two_stage'``
        to include the cascade, or ``'raw'`` for the uncorrected baseline.
    sector_levels : tuple of int
        Cascade levels / shared sector grid (see :func:`run_regress`).
    seed : int
        RNG seed, shared across all grid points (paired comparison).
    sim_kwargs : dict or None
        Extra keyword arguments forwarded to :class:`DualColourSim` (held
        fixed across the sweep, e.g. ``noise_ctrl``). Note ``frac_sig`` /
        ``frac_ctrl`` are driven by ``frac_sensor_vals`` and should not be
        passed here.
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
        un-swept dataset at default overlap / sparsity): the ground-truth
        signal/haemo, the recorded sensor/control channels, and the
        ground-truth vs corrected-recovery comparison(s) (see
        :func:`save_sim_videos`).
    save_videos_truth_vs_corr : iterable of str
        Which correction methods get a truth-vs-corrected video — one TIFF
        each. Defaults to ``('frame', 'sector', 'pixel')`` (whole-frame,
        per-sector, per-pixel); each entry toggles that file on. Drop an
        entry to skip it, or pass ``()`` / ``None`` for none. ``'two_stage'``
        is also accepted.
    corr_kwargs : dict or None
        Extra keyword arguments forwarded to the chosen correction method
        (e.g. :func:`~signal_correction.correct_full_regress`), overriding
        its shared defaults (see :func:`_run_method`).
    show : bool
        Call ``plt.show()`` before returning.

    Returns
    -------
    out : SimpleNamespace
        ``.figs`` list of figures (one if combined, one per metric if
        split); ``.metrics`` dict ``attr -> method ->
        (n_overlap, n_frac_sensor)`` ndarray; ``.overlap_vals`` /
        ``.frac_sensor_vals`` the axis values; ``.saved`` list of written
        PDF paths; ``.saved_videos`` list of written TIFF paths.
    """
    if methods is None:
        methods = ['per_frame', 'per_sector', 'per_pixel']
    sim_kwargs = dict(sim_kwargs or {})
    overlap_vals = tuple(overlap_vals)
    frac_sensor_vals = tuple(frac_sensor_vals)

    print(f'### sweeping overlap {overlap_vals} x '
          f'frac_sensor {frac_sensor_vals} '
          f'({len(overlap_vals) * len(frac_sensor_vals)} sims) ###')
    metrics = _sweep_metrics_2d(overlap_vals, frac_sensor_vals, methods,
                                sector_levels, seed, sim_kwargs, corr_kwargs)

    # Filename stem encodes every sim parameter + the swept ranges.
    _tag = _sim_param_tag(sim_kwargs, overlap_vals, frac_sensor_vals)
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
                    metrics, [spec], methods, overlap_vals,
                    frac_sensor_vals, annotate)
                fig.suptitle(f'overlap x sensor sparsity — {spec[0]}')
                figs.append(fig)
                if save_dir is not None:
                    _p = os.path.join(save_dir, f'{stem}_{spec[0]}.pdf')
                    fig.savefig(_p)
                    saved.append(_p)
        else:
            # Single combined figure / PDF.
            fig, _ = _render_heatmap_grid(
                metrics, _FIG_METRICS, methods, overlap_vals,
                frac_sensor_vals, annotate)
            fig.suptitle('Dual-colour correction: overlap x sensor sparsity')
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
        _rep = DualColourSim(seed=seed, **sim_kwargs)
        saved_videos = save_sim_videos(
            _rep, save_dir=save_dir, stem=stem,
            tvc_methods=save_videos_truth_vs_corr,
            sector_levels=sector_levels, corr_kwargs=corr_kwargs)
        for _p in saved_videos:
            print(f'saved video  -> {_p}')

    return SimpleNamespace(figs=figs, metrics=metrics,
                           overlap_vals=overlap_vals,
                           frac_sensor_vals=frac_sensor_vals,
                           saved=saved, saved_videos=saved_videos)


def _print_table(sim, results, methods, sector_levels):
    """Print a compact recovery / leakage comparison table."""
    s = sim.summary()
    print('=' * 60)
    print(f'DualColourSim  {sim.n_t}x{sim.n_x}x{sim.n_y}  '
          f'haemo_frac={sim.haemo_frac:.2f}  sig_dff={sim.sig_dff:.2f}  '
          f'ctrl_noise={s.ctrl_noise:.2f}')
    print(f'expr: sig={s.frac_sig_expressing:.0%} ctrl='
          f'{s.frac_ctrl_expressing:.0%}  overlap≈{s.measured_overlap:+.2f}  '
          f'signal px w/ control={s.frac_signal_pixels_with_control:.0%}')
    print(f'levels={tuple(sector_levels)}  '
          f'signal pixels={s.n_signal_pixels}')
    print('-' * 60)
    print(f'{"method":<14}{"recovery(a)":>13}{"haemo leak":>13}'
          f'{"Δ vs raw":>11}')
    print('-' * 60)
    raw_rec = results['raw'].recovery
    for name in ['raw'] + list(methods):
        m = results[name]
        _delta = '' if name == 'raw' else f'{m.recovery - raw_rec:+.3f}'
        print(f'{name:<14}{m.recovery:>13.3f}{m.leakage:>13.3f}'
              f'{_delta:>11}')
    print('=' * 60)
    print('recovery(a) = mean corr(corrected, true signal) over source '
          'pixels (higher is better)')
    print('haemo leak  = mean |corr(corrected, true haemo)| over all '
          'pixels (lower is better)')


if __name__ == '__main__':
    run_regress()
