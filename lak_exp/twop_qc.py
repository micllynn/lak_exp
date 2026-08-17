"""
twop_qc.py  –  QCMixin for TwoPRec.

Quality-control figure that combines behavioural events, whole-frame
fluorescence, sector fluorescence (continuous time), and suite2p
registration metrics from ops.npy on one shared time axis.
"""

import os
import gc
import gzip
import pickle
import pathlib
import concurrent.futures
import xml.etree.ElementTree as ElementTree
from types import SimpleNamespace

import numpy as np
import scipy.stats as sp_stats
import tifffile
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .utils import calc_alpha, nanmean as _nanmean


# Default sub-kwargs for the grouped QC config dicts (add_qc / plt_qc).
# Each conceptual chunk of tuning knobs is passed as an optional dict;
# pass None to take all defaults, or a partial dict to deviate. See
# _merge_qc_kwargs for resolution and unknown-key validation.
# ----------
_SECTOR_DEFAULTS = {'stride': 4, 'chunk': 500}
_FRAME_F_DEFAULTS = {'compute': True, 'stride': 4, 'chunk': 500}
_Z_CORR_DEFAULTS = {'stride': 4, 'chunk': 200, 'jobs': 4,
                    'ref_n_frames': None, 'dual_ref': True}
_ZSTACK_DEFAULTS = {'path': None, 'channel': None}
_HEATMAP_DEFAULTS = {'vmin': -2, 'vmax': 4}

# Stim-step removal params, folded into correct_signal_kwargs. Keys are
# the real correction param names so _resolve_cs_kwargs can setdefault
# them straight into the self.correct_signal call.
# ----------
_STIM_STEP_DEFAULTS = {'remove_stim_step': False, 'stim_step_edge': 'both',
                       'stim_step_n_edge': 3, 'stim_step_n_gap': 1,
                       'stim_step_amp': None, 'stim_step_dff': False}

# Group-name → defaults, so _merge_qc_kwargs can look up by name.
_QC_KWARG_DEFAULTS = {'sector': _SECTOR_DEFAULTS,
                      'frame_f': _FRAME_F_DEFAULTS,
                      'z_corr': _Z_CORR_DEFAULTS,
                      'zstack': _ZSTACK_DEFAULTS,
                      'heatmap': _HEATMAP_DEFAULTS}


def _merge_qc_kwargs(group, user):
    """Resolve a grouped QC config dict against its defaults.

    Parameters
    ----------
    group : str
        One of 'sector', 'frame_f', 'z_corr', 'zstack', 'heatmap'.
    user : dict or None
        User-supplied sub-kwargs; None is treated as {}.

    Returns
    -------
    dict
        {**defaults, **user}. Raises ValueError if user contains a key
        not present in the group's defaults (catches typos / misplaced
        flat kwargs).
    """
    defaults = _QC_KWARG_DEFAULTS[group]
    user = user or {}
    unknown = set(user) - set(defaults)
    if unknown:
        raise ValueError(
            f"{group}_kwargs got unknown key(s) {sorted(unknown)}; "
            f"valid keys: {sorted(defaults)}")
    return {**defaults, **user}


def _flu_pair_from_channel(channel):
    """(real, static) dual-colour channel roles implied by ``channel``.

    ``channel`` is the QC's active channel: the one treated as the
    functional (real) signal. channel='grn' therefore corrects green
    using red as the static control; 'red', None or anything else keeps
    the historical red-signal / green-control assignment.

    Parameters
    ----------
    channel : str or None
        'red', 'grn' or None.

    Returns
    -------
    real, static : str
        Functional and control channel labels.
    """
    if channel == 'grn':
        return 'grn', 'red'
    return 'red', 'grn'


def _resolve_cs_kwargs(user, channel=None):
    """Resolve correct_signal_kwargs, folding in stim-step + channel roles.

    A copy of ``user`` with the stim-step keys (_STIM_STEP_DEFAULTS)
    filled in via setdefault, plus an explicit real_flu / static_flu
    pair. The pair is derived from ``channel`` (see
    _flu_pair_from_channel) so that the QC's active channel is the one
    that gets corrected; if the caller supplied only one of the two, the
    other is set to its complement. An explicit user value always wins.
    Not validated for unknown keys, since correct_signal_kwargs forwards
    arbitrary kwargs to self.correct_signal. Returns the resolved dict
    ({} inputs included) for a None input.
    """
    cs = dict(user or {})
    for _k, _v in _STIM_STEP_DEFAULTS.items():
        cs.setdefault(_k, _v)
    _real = cs.get('real_flu')
    _static = cs.get('static_flu')
    if _real is None and _static is None:
        _real, _static = _flu_pair_from_channel(channel)
    elif _real is None:
        _real = 'grn' if _static == 'red' else 'red'
    elif _static is None:
        _static = 'grn' if _real == 'red' else 'red'
    cs['real_flu'] = _real
    cs['static_flu'] = _static
    return cs


def corrsig_suffix_from_kwargs(correct_signal_kwargs=None,
                               correct_signal=True,
                               detrend_sigs=False):
    """Filename fragment encoding correct_signal, method and detrend state.

    Module-level so that readers of the saved QC outputs (e.g.
    ``batch_run.plt_summary``) can rebuild the exact fragment a given
    ``correct_signal_kwargs`` produced, without re-implementing — and
    hence drifting from — the naming rules. ``QCMixin._qc_corrsig_suffix``
    is a thin wrapper reading the resolved state off ``self.qc``.

    Parameters
    ----------
    correct_signal_kwargs : dict or None
        The (resolved or raw) correct_signal kwargs. Only 'method',
        'fit_mode', 'beta_loss', 'beta_scale', 'regress_type' and the
        stim-step keys affect the fragment. Ignored when
        correct_signal is False.
    correct_signal : bool
        Whether the correction was run at all.
    detrend_sigs : bool
        Whether the QC traces were linearly detrended.

    Returns
    -------
    suffix : str
        '_corrsig=<0|1>_method=<name>[_fit=<mode>][_step=<edge>]'
        '_detrend=<0|1>'.
        The `_fit=` segment is appended for full_regress so
        the global-β vs per-pixel-β output can be told apart, along
        with the β-fit loss / scale / solver whenever those deviate
        from their defaults (`_betaloss=`, `_betascale=`,
        `_regress=`), so runs differing only in how β was fit do not
        overwrite one another's saves. The
        `_step=<edge>` segment is appended when the stim light-leak
        step removal was applied, so corrected/uncorrected PDFs do
        not overwrite one another.
    """
    _cs = bool(correct_signal)
    _fit_str = ''
    _step_str = ''
    if _cs:
        _kw = correct_signal_kwargs or {}
        _method = _kw.get('method', 'full_regress')
        if _method == 'full_regress':
            _fit_str = f'_fit={_kw.get("fit_mode", "global")}'
            # Defaults mirror correct_full_regress (robust by default),
            # so an unspecified kwarg still tags the true behaviour.
            _bl = _kw.get('beta_loss', 'huber')
            if _bl != 'linear':
                _fit_str += f'_betaloss={_bl}'
            _bs = _kw.get('beta_scale', 'mad')
            if _bs != 'std':
                _fit_str += f'_betascale={_bs}'
            # Solver for the robust β fit; only tagged when it is not
            # the default, so 'ols' filenames stay as they were and an
            # irls run cannot overwrite the ols run's PDF.
            _rt = _kw.get('regress_type', 'ols')
            if _rt != 'ols':
                _fit_str += f'_regress={_rt}'
        # Stim-step config is folded into the stored (resolved)
        # correct_signal_kwargs (see add_qc / _resolve_cs_kwargs).
        _step_on = bool(_kw.get('remove_stim_step', False))
        if _step_on:
            _edge = _kw.get('stim_step_edge', 'both')
            _step_str = f'_step={_edge}'
        elif bool(_kw.get('stim_step_dff', False)):
            _edge = _kw.get('stim_step_edge', 'both')
            _step_str = f'_step=dff-{_edge}'
    else:
        _method = 'none'
    _dt = int(bool(detrend_sigs))
    return (f'_corrsig={int(_cs)}_method={_method}{_fit_str}{_step_str}'
            f'_detrend={_dt}')


def _qc_compute_signature(n_sectors, channel, compute_sectors,
                          sector_kwargs, frame_f_kwargs, z_corr_kwargs,
                          zstack_kwargs, split_lr, correct_signal,
                          correct_signal_kwargs, grab5ht_side,
                          detrend_sigs, save_tif):
    """Canonical signature of the result-affecting QC config.

    Used by plt_qc to decide whether add_qc must recompute. Includes
    only params that change the computed traces — chunk sizes and
    thread counts (I/O batching) are deliberately excluded so they do
    not force a recompute. The group dicts are the already-resolved
    (merged) dicts; correct_signal_kwargs is the resolved, pre-injection
    dict (no save_to_disk / replace_real / aggregates_only).
    """
    return {
        'n_sectors': n_sectors,
        'channel': channel,
        'compute_sectors': bool(compute_sectors),
        'sector_stride': sector_kwargs['stride'],
        'frame_compute': bool(frame_f_kwargs['compute']),
        'frame_stride': frame_f_kwargs['stride'],
        'z_corr_stride': z_corr_kwargs['stride'],
        'z_corr_ref_n_frames': z_corr_kwargs['ref_n_frames'],
        'z_corr_dual_ref': bool(z_corr_kwargs['dual_ref']),
        'zstack_path': zstack_kwargs['path'],
        'zstack_channel': zstack_kwargs['channel'],
        'split_lr': bool(split_lr),
        'correct_signal': bool(correct_signal),
        'correct_signal_kwargs': dict(correct_signal_kwargs or {}),
        'grab5ht_side': (grab5ht_side
                         if grab5ht_side in ('left', 'right') else None),
        'detrend_sigs': bool(detrend_sigs),
        'save_tif': bool(save_tif),
    }


def load_qc_fig(path):
    """Load a gzip-pickled QC matplotlib Figure for interactive viewing.

    Parameters
    ----------
    path : str
        Path to a .pkl.gz file produced by TwoPRec.plt_qc(save_pdf=True,
        save_pkl=True). Either the .pkl.gz path or the matching
        .pdf path is accepted; a trailing .pdf is auto-rewritten to
        .pkl.gz, and a figs_mbl/ parent is redirected to data_mbl/
        (where the .pkl.gz files are written).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The reconstituted figure. plt.show() is called before
        returning so the window opens automatically in the current
        matplotlib backend.

    Notes
    -----
    Pickle compatibility is sensitive to the matplotlib version used
    at save time; load in an environment with a comparable matplotlib.
    """
    if path.endswith('.pdf'):
        path = path[:-4] + '.pkl.gz'
        # The figure pdf lives in figs_mbl/ but its .pkl.gz companion
        # is written to data_mbl/; redirect the parent dir to match.
        _p = pathlib.Path(path)
        if _p.parent.name == 'figs_mbl':
            path = str(_p.parent.parent / 'data_mbl' / _p.name)
    with gzip.open(path, 'rb') as f:
        fig = pickle.load(f)
    plt.show()
    return fig


def registration_metrics(fname_ops, verbose=False):
    """Compute summary statistics of suite2p registration corrxy.

    Loads ops.npy directly and computes the mean, standard deviation,
    skew, and kurtosis of the corrxy trace across the whole recording.

    Parameters
    ----------
    fname_ops : str
        Path to a suite2p ops.npy file.
    verbose : bool
        If True, print the computed metrics.

    Returns
    -------
    metrics : SimpleNamespace
        Has attributes .mean, .std, .skew, .kurtosis.
    """
    ops = np.load(fname_ops, allow_pickle=True).item()
    corrxy = np.asarray(ops['corrXY'])

    mean = np.mean(corrxy)
    std = np.std(corrxy)
    skew = sp_stats.skew(corrxy)
    kurtosis = sp_stats.kurtosis(corrxy)

    if verbose:
        print(f'corrxy mean     = {mean:.4f}')
        print(f'corrxy std      = {std:.4f}')
        print(f'corrxy skew     = {skew:.4f}')
        print(f'corrxy kurtosis = {kurtosis:.4f}')

    return SimpleNamespace(mean=mean, std=std, skew=skew, kurtosis=kurtosis)


def grand_avg_response_mean(npy_files, channel='red_corr',
                            t_pre=2.0, t_post=6.0, dt=0.05,
                            plot=True, save_path=None,
                            figsize=None, sector_grand_avg=True):
    """Grand average across recordings of stim-aligned response means.

    Loads `.npy` summary dicts produced by `_plt_qc_response_mean`
    (one per recording, written when `save_npy=True`),
    interpolates each recording's whole-frame stim-aligned mean trace
    onto a common time grid, and computes the across-recording grand
    mean ± SEM. Optionally also computes a per-sector grand average
    by collapsing sectors within each recording first (sectors are
    reordered per-recording, so averaging across sector indices across
    recordings is not meaningful — we average the within-recording
    per-sector mean trace instead).

    Parameters
    ----------
    npy_files : list of str
        Paths to .npy files produced by `_plt_qc_response_mean` when
        `save_npy=True`. Each is a pickled dict.
    channel : str
        Which channel to grand-average. One of 'red', 'grn', 'ratio',
        or the corrected-channel key — 'red_corr' for a red-functional
        run, 'grn_corr' when plt_qc was run with channel='grn'.
        Default 'red_corr'.
    t_pre, t_post : float
        Window around stim onset (seconds) defining the common time
        grid. Defaults match `_plt_qc_response_mean`.
    dt : float
        Sampling step (seconds) of the common time grid.
    plot : bool
        If True, plot the grand average (mean ± SEM across recordings).
    save_path : str or None
        If not None, save the figure as a pdf to this path.
    figsize : tuple or None
        Figure size in inches. Default (5, 3.5) when only the whole-
        frame panel is shown, (9, 3.5) when sector grand avg is also
        shown.
    sector_grand_avg : bool
        If True (default), also compute and plot the per-recording
        mean across sectors, then grand-average that across recordings.

    Returns
    -------
    out : SimpleNamespace
        .t_rel        : (n_t,) common time vector (s)
        .mean         : (n_t,) whole-frame grand mean across recordings
        .sem          : (n_t,) whole-frame grand SEM across recordings
        .recordings   : (n_rec, n_t) per-recording whole-frame traces
        .sec_mean     : (n_t,) sector grand mean (None if disabled)
        .sec_sem      : (n_t,) sector grand SEM (None if disabled)
        .sec_recordings : (n_rec, n_t) per-rec sector-averaged traces
        .all_sectors  : (n_total_sectors, n_t) every sector from every
                        recording, sorted by post-stim peak (most
                        active first)
        .files        : list of files actually used
        .channel      : echoed channel name
        .n_recordings : int
        .fig          : matplotlib Figure (only if plot=True)
    """
    from scipy.interpolate import interp1d

    t_common = np.arange(-t_pre, t_post + dt / 2, dt)
    wf_traces = []
    sec_traces = []
    all_sec_traces = []
    used_files = []

    for _fp in npy_files:
        try:
            _d = np.load(_fp, allow_pickle=True).item()
        except (OSError, ValueError, pickle.UnpicklingError) as _e:
            print(f'\tcould not load {_fp}: {_e}')
            continue

        _wf = _d.get('whole_frame', {})
        _entry = _wf.get(channel)
        _trel = np.asarray(_d.get('t_rel', []), dtype=np.float64)
        if _entry is None or _trel.size == 0:
            print(f'\t{_fp}: channel {channel!r} not in whole_frame')
            continue
        _mn = _entry.get('mean')
        if _mn is None:
            continue
        _mn = np.asarray(_mn, dtype=np.float64)
        if _mn.size == 0 or not np.any(np.isfinite(_mn)):
            continue
        _f = interp1d(_trel, _mn, kind='linear',
                      bounds_error=False, fill_value=np.nan)
        wf_traces.append(_f(t_common))
        used_files.append(_fp)

        if sector_grand_avg:
            _secs = _d.get('sectors', {}).get(channel)
            _trel_sec = np.asarray(_d.get('t_rel_sec', _trel),
                                   dtype=np.float64)
            if _secs is None:
                sec_traces.append(np.full_like(t_common, np.nan))
                continue
            _secs = np.asarray(_secs, dtype=np.float64)
            with np.errstate(invalid='ignore'):
                _sec_mn = np.nanmean(_secs, axis=0)
            if (_trel_sec.size != _sec_mn.size
                    or not np.any(np.isfinite(_sec_mn))):
                sec_traces.append(np.full_like(t_common, np.nan))
                continue
            _fs = interp1d(_trel_sec, _sec_mn, kind='linear',
                           bounds_error=False, fill_value=np.nan)
            sec_traces.append(_fs(t_common))

            # Interpolate every individual sector trace (rows of _secs)
            # onto t_common; stack across recordings into one big matrix
            # so the sector-heatmap subplot can sort and display them.
            if _secs.shape[1] == _trel_sec.size:
                _fi = interp1d(_trel_sec, _secs, axis=1, kind='linear',
                               bounds_error=False, fill_value=np.nan)
                all_sec_traces.append(_fi(t_common))

    if not wf_traces:
        return SimpleNamespace(
            t_rel=t_common, mean=None, sem=None, recordings=None,
            sec_mean=None, sec_sem=None, sec_recordings=None,
            all_sectors=None, all_sectors_files=None,
            files=[], channel=channel, n_recordings=0)

    _arr = np.asarray(wf_traces, dtype=np.float64)
    with np.errstate(invalid='ignore'):
        _mn = np.nanmean(_arr, axis=0)
        _n = np.sum(np.isfinite(_arr), axis=0)
        _sd = np.nanstd(_arr, axis=0)
        _sem = _sd / np.sqrt(np.maximum(_n, 1))

    _sec_mn = _sec_sem = _sec_arr = None
    if sector_grand_avg and sec_traces:
        _sec_arr = np.asarray(sec_traces, dtype=np.float64)
        with np.errstate(invalid='ignore'):
            _sec_mn = np.nanmean(_sec_arr, axis=0)
            _sn = np.sum(np.isfinite(_sec_arr), axis=0)
            _ssd = np.nanstd(_sec_arr, axis=0)
            _sec_sem = _ssd / np.sqrt(np.maximum(_sn, 1))

    # Concatenate per-sector traces from every recording into one
    # (n_total_sectors, n_t) matrix, then sort rows by peak response in
    # the post-stim window so the most-active sectors sit at the top.
    # ----------
    _all_sec = None
    if all_sec_traces:
        _all_sec = np.concatenate(all_sec_traces, axis=0)
        _post_mask = t_common >= 0
        if np.any(_post_mask):
            with np.errstate(invalid='ignore'):
                _peak = np.nanmax(_all_sec[:, _post_mask], axis=1)
        else:
            with np.errstate(invalid='ignore'):
                _peak = np.nanmax(_all_sec, axis=1)
        _order = np.argsort(np.where(np.isfinite(_peak),
                                     _peak, -np.inf))[::-1]
        _all_sec = _all_sec[_order]

    out = SimpleNamespace(
        t_rel=t_common, mean=_mn, sem=_sem, recordings=_arr,
        sec_mean=_sec_mn, sec_sem=_sec_sem, sec_recordings=_sec_arr,
        all_sectors=_all_sec,
        files=used_files, channel=channel,
        n_recordings=int(_arr.shape[0]))

    if plot:
        _has_sec_avg = sector_grand_avg and _sec_mn is not None
        _has_sec_heat = _all_sec is not None and _all_sec.size > 0
        _ncols = 1 + int(_has_sec_avg) + int(_has_sec_heat)
        if figsize is None:
            figsize = (4.5 * _ncols, 3.5)
        fig, axes = plt.subplots(1, _ncols, figsize=figsize,
                                 sharex=False, sharey=False)
        if _ncols == 1:
            axes = [axes]
        _col = 0

        def _draw(ax, mean, sem, per_rec, title):
            for _row in per_rec:
                ax.plot(t_common, _row, color='grey', alpha=0.3,
                        linewidth=0.6)
            ax.fill_between(t_common, mean - sem, mean + sem,
                            color='k', alpha=0.25, linewidth=0)
            ax.plot(t_common, mean, color='k', linewidth=1.5)
            ax.axvline(x=0, color='k', linewidth=0.8,
                       linestyle='--', alpha=0.7)
            ax.axhline(y=0, color='k', linewidth=0.4, linestyle=':')
            ax.set_xlabel('time from stim (s)')
            ax.set_ylabel(f'{channel} response')
            ax.set_title(title, fontsize=10)
            for _side in ('top', 'right'):
                ax.spines[_side].set_visible(False)

        _draw(axes[_col], _mn, _sem, _arr,
              f'whole-frame grand avg ({channel})\n'
              f'n={_arr.shape[0]} recordings')
        _col += 1
        if _has_sec_avg:
            _draw(axes[_col], _sec_mn, _sec_sem, _sec_arr,
                  f'sector-mean grand avg ({channel})\n'
                  f'n={_sec_arr.shape[0]} recordings')
            _col += 1
        if _has_sec_heat:
            ax = axes[_col]
            _vlo = float(np.nanpercentile(_all_sec, 2))
            _vhi = float(np.nanpercentile(_all_sec, 98))
            _vmag = max(abs(_vlo), abs(_vhi))
            if not np.isfinite(_vmag) or _vmag == 0:
                _vmag = 1.0
            _im = ax.imshow(
                _all_sec, aspect='auto',
                extent=[t_common[0], t_common[-1],
                        _all_sec.shape[0], 0],
                cmap='gray', vmin=-_vmag, vmax=_vmag,
                interpolation='none')
            ax.axvline(x=0, color=sns.xkcd_rgb['bright red'],
                       linewidth=0.8, linestyle='--', alpha=0.8)
            ax.set_xlabel('time from stim (s)')
            ax.set_ylabel(f'sector\n(sorted, n={_all_sec.shape[0]})',
                          fontsize=8)
            ax.set_title(f'all sectors, all recordings ({channel})',
                         fontsize=10)
            _cax = ax.inset_axes([1.02, 0, 0.025, 1])
            fig.colorbar(_im, cax=_cax)
            _cax.tick_params(labelsize=7)
            _cax.set_ylabel(f'{channel} response', fontsize=7)
            for _side in ('top', 'right'):
                ax.spines[_side].set_visible(False)

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path)
        out.fig = fig

    return out


class QCMixin(object):
    """Quality-control plotting methods mixed into TwoPRec."""

    # ------------------------------------
    # Compute
    # ------------------------------------

    def add_registration_metrics(self, verbose=False):
        """Compute registration metrics from this recording's ops.npy.

        Infers the path to ops.npy from ``self.folder.img`` and stores
        the result on ``self.registration_metrics``.

        Parameters
        ----------
        verbose : bool
            If True, print the computed metrics.

        Returns
        -------
        metrics : SimpleNamespace
            Has attributes .mean, .std, .skew, .kurtosis.
        """
        fname_ops = os.path.join(self.folder.img, 'suite2p', 'plane0',
                                 'ops.npy')
        self.registration_metrics = registration_metrics(
            fname_ops, verbose=verbose)
        return self.registration_metrics

    def _qc_n_keep_frames(self, channel=None, t_end_pad=10.0):
        """Leading imaging-frame count to keep when self.trial_end is set.

        Mirrors ``correct_signal``'s t_end inference so QC traces cover
        the same range as the correction: the cutoff time is the reward
        time of trial ``trial_end`` plus ``t_end_pad`` seconds, and every
        frame at or before that time is kept.

        Parameters
        ----------
        channel : str or None
            Channel whose timestamps define the frame grid.
        t_end_pad : float
            Seconds added after trial_end's reward time. Default 10.0
            (matches correct_signal).

        Returns
        -------
        n_keep : int
            Number of frames to keep (full count if trial_end is None or
            the cutoff cannot be resolved).
        t_end : float or None
            The cutoff time in seconds, or None when not truncating.
        """
        _t = np.asarray(self._get_rec_t(channel))
        _n_full = int(_t.shape[0])
        _te = getattr(self, 'trial_end', None)
        if _te is None:
            return _n_full, None
        try:
            _rew = np.asarray(
                self.beh._data.get_event_var('totalRewardTimes'))
        except Exception:
            _rew = np.asarray(getattr(self.beh.rew, 't', []))
        _te = int(_te)
        if _rew.size <= _te:
            return _n_full, None
        _t_end = float(_rew[_te]) + float(t_end_pad)
        _n_keep = int(np.searchsorted(_t, _t_end, side='right'))
        _n_keep = max(1, min(_n_keep, _n_full))
        return _n_keep, _t_end

    def _qc_stim_t(self):
        """Stim onset times, truncated to trials <= self.trial_end."""
        _s = np.asarray(self.beh.stim.t_start).ravel()
        _te = getattr(self, 'trial_end', None)
        if _te is not None:
            _s = _s[:int(_te) + 1]
        return _s

    def _qc_rew_t(self):
        """Reward times, truncated to trials <= self.trial_end."""
        _r = np.asarray(getattr(self.beh.rew, 't', [])).ravel()
        _te = getattr(self, 'trial_end', None)
        if _te is not None:
            _r = _r[:int(_te) + 1]
        return _r

    def _qc_lick_t(self, lick_t):
        """Filter lick times to the kept range (<= self.qc._t_end)."""
        _lt = np.asarray(lick_t).ravel()
        _tend = getattr(getattr(self, 'qc', None), '_t_end', None)
        if _tend is not None and _lt.size:
            _lt = _lt[_lt <= _tend]
        return _lt

    # ------------------------------------
    # Trial-type (stimulus-subtype) helpers
    # ------------------------------------
    # Shared infrastructure for splitting the QC figures by trial-type
    # using the same stim/reward line nomenclature as the main QC plot:
    # stim lines are dark-grey dashes whose transparency encodes the
    # expected reward (size * prob, via calc_alpha); reward lines are
    # bright-blue dashes. The base-condition splitter (_qc_base_tr_conds)
    # drops derived sub-subtypes (labels carrying a '_', e.g. '0.5_rew')
    # and keeps only the primary stimulus/reward conditions ('0', '0.5',
    # '1', orientation values, ...).
    # ----------

    def _qc_exp_rew(self, truncate=True):
        """Per-trial expected reward (size * prob).

        Mirrors the main QC plot's ``self.beh.stim.size * self.beh.stim.prob``
        encoding used for the stim event-line transparency.

        Parameters
        ----------
        truncate : bool
            If True, truncate to trials <= self.trial_end (matching
            _qc_stim_t). Default True.

        Returns
        -------
        exp_rew : np.ndarray
            Per-trial expected reward.
        """
        _sz = np.asarray(getattr(self.beh.stim, 'size', []),
                         dtype=np.float64).ravel()
        _pr = np.asarray(getattr(self.beh.stim, 'prob', []),
                         dtype=np.float64).ravel()
        _n = min(_sz.size, _pr.size)
        _exp = _sz[:_n] * _pr[:_n]
        _te = getattr(self, 'trial_end', None)
        if truncate and _te is not None:
            _exp = _exp[:int(_te) + 1]
        return _exp

    def _qc_exp_rew_max(self):
        """Maximum expected reward across all trials (alpha denominator).

        Used as ``val_max`` for calc_alpha so the stim-line transparency
        scale is shared across the main QC plot and the per-trial-type
        panels. Falls back to 1.0 when no positive expected reward exists.
        """
        _exp = self._qc_exp_rew(truncate=False)
        _mx = float(np.max(_exp)) if _exp.size else 0.0
        return _mx if _mx > 0 else 1.0

    def _qc_stim_event_alphas(self, alpha_min=0.1):
        """Stim onset times and their per-trial transparency.

        Reproduces the main QC plot's stim event-line encoding: alpha[i]
        = calc_alpha(size_i * prob_i, val_max=max expected reward). Use
        for continuous-time stim markers (main plot, correction figure).

        Parameters
        ----------
        alpha_min : float
            Minimum alpha so low-value stims stay visible. Default 0.1.

        Returns
        -------
        stim_t : np.ndarray
            Stim onset times (== _qc_stim_t()).
        alphas : np.ndarray
            Per-stim alpha values, same length as stim_t.
        """
        _stim_t = self._qc_stim_t()
        _exp = self._qc_exp_rew(truncate=True)
        _n = min(_stim_t.size, _exp.size)
        _stim_t = _stim_t[:_n]
        _exp = _exp[:_n]
        _vmax = self._qc_exp_rew_max()
        _alphas = np.array(
            [calc_alpha(_e, val_max=_vmax, alpha_min=alpha_min)
             for _e in _exp], dtype=np.float64)
        return _stim_t, _alphas

    def _qc_base_tr_conds(self):
        """Ordered base stimulus conditions and their trial indices.

        Base conditions are the entries of ``self.beh.tr_conds`` whose
        label carries no '_' (primary stimulus / reward conditions such
        as '0', '0.5', '1' or orientation values); derived sub-subtypes
        ('0.5_rew', '0.5_prelick', ...) are excluded. Only conditions
        with at least one trial are returned. Falls back to a single
        ('all', None) pseudo-condition (all stim trials) when no
        trial-type parsing is available.

        Returns
        -------
        conds : list of (str, np.ndarray) tuples
            (label, trial-index array) per base condition.
        """
        _tr_conds = getattr(self.beh, 'tr_conds', None)
        _tr_inds = getattr(self.beh, 'tr_inds', None)
        if not _tr_conds or not isinstance(_tr_inds, dict):
            return [('all', None)]
        _out = []
        for _c in _tr_conds:
            if '_' in str(_c):
                continue
            _inds = np.asarray(_tr_inds.get(_c, []), dtype=np.int64).ravel()
            if _inds.size:
                _out.append((str(_c), _inds))
        return _out if _out else [('all', None)]

    def _qc_cond_stim_t(self, tr_inds):
        """Stim onset times for a subset of trials (truncated to trial_end).

        Parameters
        ----------
        tr_inds : array-like or None
            Trial indices; None returns all stim onsets (== _qc_stim_t()).

        Returns
        -------
        stim_t : np.ndarray
            Stim onset times for the requested trials.
        """
        _all = np.asarray(self.beh.stim.t_start, dtype=np.float64).ravel()
        _te = getattr(self, 'trial_end', None)
        _n = _all.size if _te is None else min(int(_te) + 1, _all.size)
        if tr_inds is None:
            return _all[:_n]
        _inds = np.asarray(tr_inds, dtype=np.int64).ravel()
        _inds = _inds[(_inds >= 0) & (_inds < _n)]
        return _all[_inds]

    def _qc_cond_alpha(self, tr_inds, alpha_min=0.1):
        """Stim-line transparency for a trial-type, main-QC style.

        The condition's mean expected reward is mapped through calc_alpha
        against the global max expected reward, exactly as the main QC
        plot encodes per-trial expected reward.

        Parameters
        ----------
        tr_inds : array-like or None
            Trial indices for the condition; None uses all trials.
        alpha_min : float
            Minimum alpha. Default 0.1.

        Returns
        -------
        alpha : float
        """
        _exp = self._qc_exp_rew(truncate=False)
        _vmax = self._qc_exp_rew_max()
        if tr_inds is None:
            _val = float(np.mean(_exp)) if _exp.size else 0.0
        else:
            _inds = np.asarray(tr_inds, dtype=np.int64).ravel()
            _inds = _inds[(_inds >= 0) & (_inds < _exp.size)]
            _val = float(np.mean(_exp[_inds])) if _inds.size else 0.0
        return calc_alpha(_val, val_max=_vmax, alpha_min=alpha_min)

    def _qc_cond_rew_params(self, tr_inds):
        """Reward probability and reward volume for a condition's trials.

        Reads the per-trial reward probability (``self.beh.stim.prob``,
        i.e. rewardProbabilityValues) and reward volume / magnitude
        (``self.beh.stim.size``, i.e. rewardMagnitudeValues) and returns
        their mean over the condition's trials (NaN when unavailable).

        Parameters
        ----------
        tr_inds : array-like or None
            Trial indices for the condition; None uses all trials.

        Returns
        -------
        p_rew : float
            Mean reward probability for the condition.
        vol_rew : float
            Mean reward volume / magnitude for the condition.
        """
        _pr = np.asarray(getattr(self.beh.stim, 'prob', []),
                         dtype=np.float64).ravel()
        _sz = np.asarray(getattr(self.beh.stim, 'size', []),
                         dtype=np.float64).ravel()
        if tr_inds is None:
            _p = float(np.nanmean(_pr)) if _pr.size else float('nan')
            _v = float(np.nanmean(_sz)) if _sz.size else float('nan')
            return _p, _v
        _i = np.asarray(tr_inds, dtype=np.int64).ravel()
        _ip = _i[(_i >= 0) & (_i < _pr.size)]
        _iv = _i[(_i >= 0) & (_i < _sz.size)]
        _p = float(np.nanmean(_pr[_ip])) if _ip.size else float('nan')
        _v = float(np.nanmean(_sz[_iv])) if _iv.size else float('nan')
        return _p, _v

    def _qc_cond_rew_rel(self, tr_inds):
        """Median stim->reward latency and reward fraction for a condition.

        Latency is taken from the block's per-trial ``totalRewardTimes``
        paired with stim onsets (non-rewarded trials have a non-positive
        delta and drop out). The reward fraction is the share of the
        condition's trials that actually received reward, used to gate /
        fade the reward marker for low-probability conditions.

        Parameters
        ----------
        tr_inds : array-like or None
            Trial indices for the condition; None uses all trials.

        Returns
        -------
        rew_rel : float or None
            Median latency (s), or None if unresolvable / no rewards.
        rew_frac : float
            Fraction of the condition's trials that were rewarded.
        """
        try:
            _stim = np.asarray(self.beh.stim.t_start,
                               dtype=np.float64).ravel()
            _rew = np.asarray(
                self.beh._data.get_event_var('totalRewardTimes'),
                dtype=np.float64).ravel()
        except (AttributeError, KeyError, TypeError, ValueError):
            return None, 0.0
        _te = getattr(self, 'trial_end', None)
        _n = min(_stim.size, _rew.size)
        if _te is not None:
            _n = min(_n, int(_te) + 1)
        if tr_inds is None:
            _inds = np.arange(_n)
        else:
            _inds = np.asarray(tr_inds, dtype=np.int64).ravel()
            _inds = _inds[(_inds >= 0) & (_inds < _n)]
        if _inds.size == 0:
            return None, 0.0
        _lat = _rew[_inds] - _stim[_inds]
        _lat = _lat[np.isfinite(_lat) & (_lat > 0)]
        _rel = float(np.median(_lat)) if _lat.size else None
        return _rel, _lat.size / _inds.size

    def _detect_lick_bouts(self, lick_t, max_ili=0.5, min_licks=4):
        """Detect rhythmic lick-bout onset times.

        A bout is a maximal run of consecutive licks whose successive
        inter-lick intervals are all <= max_ili (i.e. rhythmic licking),
        and that contains more than three licks (>= min_licks). The bout
        onset is the time of the first lick in the run.

        Parameters
        ----------
        lick_t : array-like
            Lick times (s). Sorted internally; non-finite values dropped.
        max_ili : float
            Maximum inter-lick interval (s) within a bout. Default 0.5.
        min_licks : int
            Minimum number of licks for a run to count as a bout.
            Default 4 (i.e. strictly more than three licks).

        Returns
        -------
        onsets : np.ndarray
            Bout-onset times (s), one per detected bout (ascending).
        """
        _lt = np.asarray(lick_t, dtype=np.float64).ravel()
        _lt = np.sort(_lt[np.isfinite(_lt)])
        if _lt.size < min_licks:
            return np.array([], dtype=np.float64)
        # Break the lick train into runs wherever an inter-lick interval
        # exceeds the rhythmic threshold; keep runs long enough to count.
        _breaks = np.flatnonzero(np.diff(_lt) > max_ili)
        _starts = np.concatenate(([0], _breaks + 1))
        _ends = np.concatenate((_breaks, [_lt.size - 1]))  # inclusive
        _onsets = [_lt[_s] for _s, _e in zip(_starts, _ends)
                   if (_e - _s + 1) >= min_licks]
        return np.asarray(_onsets, dtype=np.float64)

    def _trial_onsets_from_stim(self, t_start=None):
        """Stim onset frame indices, for the per-trial pixel_spatial_subtr
        correction.

        Converts the behaviour clock times in ``self.beh.stim.t_start`` to
        frame indices into ``self.rec_t`` (the conversion documented on
        ``signal_correction.correct_pixel_spatial_subtr``'s ``trial_onsets``
        parameter). When ``correct_signal`` is given a ``t_start`` the
        corrected stack begins at frame ``idx_start`` rather than 0, so the
        same offset is subtracted here; onsets that fall before the start
        of the (possibly trimmed) stack are dropped.

        Parameters
        ----------
        t_start : float or None
            The ``t_start`` (seconds) that will be passed to
            ``correct_signal``, or None if the full stack is corrected.

        Returns
        -------
        onsets : (n,) np.ndarray of intp
            Onset frame indices into the corrected stack.
        """
        _beh = getattr(self, 'beh', None)
        _stim = getattr(_beh, 'stim', None) if _beh is not None else None
        if _stim is None or getattr(_stim, 't_start', None) is None:
            raise ValueError(
                "cannot auto-fill trial_onsets for f0_mode='per_trial': "
                "self.beh.stim.t_start is unavailable. Load behaviour, or "
                "pass trial_onsets explicitly in correct_signal_kwargs.")
        if not hasattr(self, 'rec_t'):
            raise ValueError(
                "cannot auto-fill trial_onsets: self.rec_t is unavailable.")
        _stim_t = np.asarray(_stim.t_start, dtype=np.float64).ravel()
        _stim_t = _stim_t[np.isfinite(_stim_t)]
        _onsets = np.searchsorted(self.rec_t, _stim_t).astype(np.intp)
        if t_start is not None:
            _onsets = _onsets - int(
                np.searchsorted(self.rec_t, t_start, side='left'))
        return _onsets[_onsets >= 0]

    def _median_stim_rew_latency(self, max_pair_lat=5.0):
        """Median stim→reward latency in seconds, for sizing per-trial
        windows.

        Pairs each stim onset with its first reward within
        ``max_pair_lat`` seconds and returns the median of those latencies
        (0.0 if nothing pairs, e.g. no rewards). Mirrors the window logic
        in ``_save_trial_avg_to_disk`` so the per-trial dF/F window matches
        the trial-averaged QC output.
        """
        _beh = getattr(self, 'beh', None)
        _stim = getattr(_beh, 'stim', None) if _beh is not None else None
        _rew = getattr(_beh, 'rew', None) if _beh is not None else None
        if _stim is None or _rew is None:
            return 0.0
        _stim_t = np.asarray(getattr(_stim, 't_start', []),
                             dtype=np.float64).ravel()
        _stim_t = _stim_t[np.isfinite(_stim_t)]
        _rew_clean = []
        for _v in np.asarray(getattr(_rew, 't', []), dtype=object).ravel():
            try:
                _f = float(_v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(_f):
                _rew_clean.append(_f)
        _rew_t = np.sort(np.asarray(_rew_clean, dtype=np.float64))
        _lats = []
        if _stim_t.size and _rew_t.size:
            _idx = np.searchsorted(_rew_t, _stim_t, side='right')
            for _i, _ri in enumerate(_idx):
                if _ri < _rew_t.size:
                    _l = float(_rew_t[_ri] - _stim_t[_i])
                    if 0 < _l <= max_pair_lat:
                        _lats.append(_l)
        return float(np.median(_lats)) if _lats else 0.0

    def _trial_window_from_trialavg(self, t_pre=2.0, t_post=4.0):
        """Per-trial output window as (pre, post) frame offsets from onset.

        Matches the trial-averaged QC window
        ``[-t_pre, median(stim→rew) + t_post]`` (seconds), converted to
        frames via ``self.samp_rate`` — so the per-trial dF/F correction
        covers the same span as the trial-averaged TIFFs rather than a
        silently-picked default.

        Parameters
        ----------
        t_pre, t_post : float
            Pre-stim and post-reward padding (seconds); the ``trialavg_*``
            params of ``correct_signal``.

        Returns
        -------
        (int, int)
            ``(-n_pre, n_post)`` frame offsets, suitable for the
            correction's ``trial_window``.
        """
        _dt = 1.0 / float(self.samp_rate)
        _rew_lat = self._median_stim_rew_latency()
        _n_pre = int(round(float(t_pre) / _dt))
        _n_post = int(round((_rew_lat + float(t_post)) / _dt))
        return (-_n_pre, _n_post)

    def _fit_trial_window(self, window, onsets):
        """Shrink a ``(pre, post)`` frame window so the per-trial output
        windows cannot overlap.

        The per-trial correction requires each frame to belong to at most
        one trial and raises otherwise. Two consecutive onsets ``gap``
        frames apart stay disjoint only when ``n_pre + n_post <= gap``, so
        when the tightest inter-onset gap is smaller than the requested
        span the post window is trimmed first (preserving the pre-stim
        baseline); if the baseline alone exceeds the gap, the pre window is
        trimmed too. Returned unchanged when the trials are far enough
        apart or there are fewer than two onsets.

        Parameters
        ----------
        window : (int, int)
            ``(-n_pre, n_post)`` frame offsets from onset.
        onsets : array-like of int
            Onset frame indices.

        Returns
        -------
        (int, int)
            A window whose span fits the tightest inter-onset gap.
        """
        _pre = -int(window[0])
        _post = int(window[1])
        _on = np.sort(np.asarray(onsets, dtype=np.intp).ravel())
        if _on.size < 2:
            return (int(window[0]), int(window[1]))
        _min_gap = int(np.min(np.diff(_on)))
        if _pre + _post <= _min_gap:
            return (int(window[0]), int(window[1]))
        _post = max(0, _min_gap - _pre)
        if _pre > _min_gap:
            _pre = max(1, _min_gap)
            _post = 0
        return (-_pre, _post)

    def add_qc(self, channel=None, n_sectors=8, compute_sectors=False,
               split_lr=False,
               sector_kwargs=None,
               frame_f_kwargs=None,
               z_corr_kwargs=None,
               zstack_kwargs=None,
               correct_signal=False,
               correct_signal_kwargs=None,
               grab5ht_side=None,
               detrend_sigs=False,
               save_tif=False):
        """Compute continuous-time QC traces and load registration metrics.

        Conceptually-related tuning knobs are grouped into optional dicts
        (``*_kwargs``); pass None to take all defaults, or a partial dict
        to deviate from them. Unknown keys raise ValueError. Primary
        toggles stay as top-level scalars.

        Parameters
        ----------
        channel : str or None
            The functional channel: 'red' or 'grn' (dual-colour only),
            None uses self.rec. Also fixes the correction's real_flu /
            static_flu pair — channel='grn' corrects green using red —
            unless correct_signal_kwargs sets them explicitly.
        n_sectors : int
            Number of sectors per axis. Field of view is divided into
            n_sectors x n_sectors blocks; continuous-time mean fluorescence
            is computed for each block. Default 8.
        compute_sectors : bool
            If True, compute the (n_sectors**2, *) sector fluorescence
            matrix. Disabled by default because it requires a full pass
            over the imaging memmap (slow on spinning disks).
        split_lr : bool
            If True, divide each frame into a top half ('right') and a
            bottom half ('left') along the y-axis; whole-frame F is
            computed separately per half and stored under keys suffixed
            '_right' / '_left'. Sector traces are computed normally and
            split row-wise at plot time. Default False.
        sector_kwargs : dict or None
            Sector fluorescence extraction. Keys (defaults):
              stride (4) – sample every Nth frame for the sector matrix
                           (sector traces are slow-varying, so 2-4 is
                           visually lossless)
              chunk (500) – sampled frames per chunked read
            None ⇒ all defaults.
        frame_f_kwargs : dict or None
            Whole-frame mean F extraction. Keys (defaults):
              compute (True) – populate self.qc.frame_f. Single-channel
                           rides on the z_corr pass at zero extra I/O when
                           meanImg is available; otherwise a dedicated
                           chunked pass. Dual-colour: both channels get
                           dedicated passes.
              stride (4) – stride for the dedicated pass (ignored for the
                           single-channel piggyback, which uses
                           z_corr_kwargs['stride'])
              chunk (500) – sampled frames per chunked read
            None ⇒ all defaults.
        z_corr_kwargs : dict or None
            Z-corr (registration drift) computation. Keys (defaults):
              stride (4) – compute every Nth frame; intermediate values
                           are linearly interpolated
              chunk (200) – sampled frames per thread-pool chunk
              jobs (4) – number of parallel threads
              ref_n_frames (None) – if int, build the reference from the
                           mean of the first N frames of the active rec
                           instead of ops['meanImg'] (tracks drift vs
                           session start); also sets the window at *both*
                           ends when dual_ref=True (defaults to 500 there)
              dual_ref (True) – build early + late references (first/last
                           ref_n_frames), Pearson-correlate against each,
                           and expose reg.z_corr_early/.z_corr_late/
                           .z_corr_diff (early−late, signed drift). In this
                           mode the ops['meanImg']-based reg.z_corr stays
                           None.
            None ⇒ all defaults.
        zstack_kwargs : dict or None
            Z-position inference from a Bruker ZSeries. Keys (defaults):
              path (None) – ZSeries folder (per-cycle multi-page .ome.tif
                           + .xml sidecar). Each rec frame is
                           Pearson-correlated against every slice; the
                           per-frame peak is parabolically triangulated to
                           sub-slice resolution and converted to z (µm) via
                           the per-slice ZAxis values. Populates reg.z_um,
                           reg.z_um_slices, reg.zstack_corr_mat,
                           reg.zstack_slice_frac; replaces the z_corr panel
                           in plt_qc. None ⇒ disabled.
              channel (None) – 'Ch1' (red) or 'Ch2' (green); None picks the
                           channel matching the active self.rec.
            None ⇒ disabled.
        correct_signal : bool
            If True (dual-colour only), run self.correct_signal with
            replace_real=False so the original signal is preserved, then
            compute whole-frame F on the corrected real channel and store
            under the key '{real_flu}_corr' in self.qc.frame_f (with
            _right/_left suffixes when split_lr=True). Default False.
        correct_signal_kwargs : dict or None
            Extra kwargs forwarded to self.correct_signal (e.g. method,
            static_flu, real_flu, detrend, t_start, t_end). replace_real
            is always forced to False. The stim-step light-leak removal
            params are folded in here (defaults): remove_stim_step (False),
            stim_step_edge ('both'; use 'on' for GRAB-DA, whose OFF edge is
            contaminated by the reward transient), stim_step_n_edge (3;
            frames averaged each side of an edge), stim_step_n_gap (1;
            frames skipped at each transition), stim_step_amp (None; fixed
            amplitude instead of estimating). For the single-channel
            hemisphere-control 1-D path (split_lr=True, correct_signal=True,
            grab5ht_side set, no dual-colour rec): picks the method and
            forwards 1-D params
            (smooth_window, airpls_lam, airpls_porder, airpls_max_iter,
            trim_initial, nn_slope). Default None.

            For method='pixel_spatial_subtr' with f0_mode='per_trial',
            trial_onsets and trial_window are auto-filled when not
            supplied: trial_onsets from self.beh.stim.t_start via self.rec_t
            (see _trial_onsets_from_stim), and trial_window from the
            trial-averaged window [-trialavg_t_pre, median(stim→rew) +
            trialavg_t_post] in frames (see _trial_window_from_trialavg),
            shrunk if needed so per-trial output windows stay disjoint. Pass
            either explicitly to override.
        grab5ht_side : str or None
            Hemisphere ('left' or 'right') that carries the real GRAB
            signal in a single-channel hemisphere-control run; the
            other side is treated as the mutant / regressor. When set
            together with split_lr=True and correct_signal=True (and
            the recording is single-channel), runs the chosen
            correction method on the two 1-D hemisphere mean traces
            (s1=mut, s2=grab) and stores the corrected trace under
            ``frame_f['{channel}_corr_{grab5ht_side}']``. Defaults to
            None (no 1-D correction; dual-colour pixel-wise correction
            is unaffected).
        detrend_sigs : bool
            If True, subtract a per-trace linear fit from the raw 'red'
            and 'grn' whole-frame and per-sector traces (and their
            _right/_left halves when split_lr=True) after extraction.
            This is mathematically equivalent — by linearity of the
            mean — to running signal_correction.detrend_linearly on the
            underlying (T, X, Y) rec and then spatial-averaging, but it
            is applied to the 1-D / 2-D traces directly and is therefore
            fast. Default False.
        save_tif : bool
            If True (and correct_signal=True), the correction writes the
            trial-averaged stim-aligned TIFFs `{base}_corr_trialavg.tif`
            (corrected) and `{base}_dff_trialavg.tif` (raw real/static
            dF/F) into the recording's data_mbl/ folder
            (self.folder.data). Internally this sets ``save_to_disk`` on
            the correction call: the full corrected stack is streamed to
            a scratch `{base}_corr.tif` (in folder.img), read for the
            trial-averaging and QC aggregates, then deleted, so only the
            small trial-averaged TIFFs persist. Default False.

        Notes
        -----
        Stores self.qc (SimpleNamespace) with:
            .t              : (n_frames,) time vector
            .channel        : channel used for sector fluorescence
            .n_sectors      : int
            .split_lr       : bool, whether frame_f is split top/bottom
            .correct_signal_kwargs : resolved correction kwargs (stim-step
                              folded in), or None
            ._compute_sig   : dict signature of the result-affecting config
                              (used by plt_qc to decide whether to recompute)
            .frame_f        : dict {ch_label: (n_frames,)} whole-frame F,
                              empty if frame_f_kwargs['compute'] is False
            .sector_f       : (n_sectors**2, n_compute) per-sector F,
                              or None if compute_sectors is False
            .reg            : SimpleNamespace of registration metrics with
                              .xoff, .yoff, .shift_mag, .corrXY,
                              .badframes, .z_corr
        """
        print(f'computing QC traces (n_sectors={n_sectors})...')

        # Resolve the grouped tuning dicts against their defaults and
        # rebind the individual knobs as locals, so the body below reads
        # them by their familiar names. Stim-step params are folded into
        # the correction kwargs by _resolve_cs_kwargs, which also pins
        # real_flu / static_flu to `channel` (the active channel is the
        # functional one, so channel='grn' corrects green using red).
        # ----------
        sector_kwargs = _merge_qc_kwargs('sector', sector_kwargs)
        frame_f_kwargs = _merge_qc_kwargs('frame_f', frame_f_kwargs)
        z_corr_kwargs = _merge_qc_kwargs('z_corr', z_corr_kwargs)
        zstack_kwargs = _merge_qc_kwargs('zstack', zstack_kwargs)
        _cs_resolved = _resolve_cs_kwargs(correct_signal_kwargs, channel)
        sector_stride = sector_kwargs['stride']
        sector_chunk = sector_kwargs['chunk']
        compute_frame_f = frame_f_kwargs['compute']
        frame_stride = frame_f_kwargs['stride']
        frame_chunk = frame_f_kwargs['chunk']
        z_corr_stride = z_corr_kwargs['stride']
        z_corr_chunk = z_corr_kwargs['chunk']
        z_corr_jobs = z_corr_kwargs['jobs']
        z_corr_ref_n_frames = z_corr_kwargs['ref_n_frames']
        z_corr_dual_ref = z_corr_kwargs['dual_ref']
        use_zstack = zstack_kwargs['path']
        zstack_channel = zstack_kwargs['channel']

        self.qc = SimpleNamespace()
        self.qc.n_sectors = n_sectors
        self.qc.channel = channel
        self.qc.split_lr = split_lr
        self.qc.correct_signal = bool(correct_signal)
        # Store the RESOLVED correction kwargs (stim-step folded in) so
        # the corrsig-suffix builder and correction-steps panel can read
        # the stim-step config back. Call-specific keys (save_to_disk /
        # replace_real / aggregates_only) are injected later on a copy and
        # deliberately kept out of this stored dict and the signature.
        self.qc.correct_signal_kwargs = (
            dict(_cs_resolved) if correct_signal else None)
        # Flag correction methods whose corrected output is already a
        # native dF/F (subtractive dff_sig − dff_ctrl), so the QC
        # event-average pipeline must display it AS dF/F rather than ITI
        # z-scoring it. pixel_spatial_subtr is such a method for either
        # f0_mode: 'global' gives one continuous dF/F trace; 'per_trial'
        # gives a per-trial dF/F scattered into a continuous trace with
        # NaN in the inter-trial gaps (which additionally has no ITI
        # baseline to z-score against — the ITI is all NaN). The pipeline
        # reads this flag (via the _corr_predff branches) and passes the
        # corrected column through, scaled to dF/F (%), instead of z-scoring.
        self.qc.corr_native_dff = None
        if correct_signal and _cs_resolved is not None:
            if _cs_resolved.get('method') == 'pixel_spatial_subtr':
                self.qc.corr_native_dff = True
        self.qc.grab5ht_side = (
            grab5ht_side if grab5ht_side in ('left', 'right') else None)
        self.qc.detrend_sigs = bool(detrend_sigs)
        self.qc.save_tif = bool(save_tif)
        # Single canonical signature of the result-affecting config;
        # plt_qc compares this to decide whether to recompute.
        self.qc._compute_sig = _qc_compute_signature(
            n_sectors=n_sectors, channel=channel,
            compute_sectors=compute_sectors,
            sector_kwargs=sector_kwargs, frame_f_kwargs=frame_f_kwargs,
            z_corr_kwargs=z_corr_kwargs, zstack_kwargs=zstack_kwargs,
            split_lr=split_lr, correct_signal=correct_signal,
            correct_signal_kwargs=_cs_resolved, grab5ht_side=grab5ht_side,
            detrend_sigs=detrend_sigs, save_tif=save_tif)
        self.qc.t = self._get_rec_t(channel)
        self.qc.frame_f = {}

        rec = self._get_rec(channel)
        _has_grn = hasattr(self, 'rec_grn')
        _has_red = hasattr(self, 'rec_red')

        # Respect a manually-set trial_end: truncate the imaging time axis
        # and every recording view used below so all QC traces (frame_f,
        # sector_f, z_corr, registration, mean images) and the trial-based
        # correction use only up to the defined end of the recording, not
        # beyond it. The cutoff matches correct_signal's t_end inference,
        # so the corrected and raw traces span the same frames. Slicing is
        # zero-copy and a no-op when trial_end is None.
        # ----------
        _n_full = int(self.qc.t.shape[0])
        _n_keep, _t_end_qc = self._qc_n_keep_frames(channel)
        self.qc._n_keep = int(_n_keep)
        self.qc._t_end = _t_end_qc
        if _t_end_qc is not None and _n_keep < _n_full:
            print(f'\trespecting trial_end={self.trial_end}: using first '
                  f'{_n_keep}/{_n_full} frames (t ≤ {_t_end_qc:.1f}s).')
        self.qc.t = self.qc.t[:_n_keep]
        rec = rec[:_n_keep]
        _rec_grn = self.rec_grn[:_n_keep] if _has_grn else None
        _rec_red = self.rec_red[:_n_keep] if _has_red else None

        # Sector fluorescence (continuous time)
        # ----------
        if compute_sectors:
            if _has_grn and _has_red:
                print(f'\textracting sector fluorescence (dual-colour) '
                      f'(stride={sector_stride}, '
                      f'chunk={sector_chunk})...')
                self.qc.sector_f = {}
                for _ch_label, _ch_rec in [('grn', _rec_grn),
                                           ('red', _rec_red)]:
                    print(f'\t\t[{_ch_label}]')
                    self.qc.sector_f[_ch_label] = self._compute_sectors(
                        _ch_rec, n_sectors,
                        stride=sector_stride, chunk_size=sector_chunk)
            else:
                print(f'\textracting sector fluorescence '
                      f'(stride={sector_stride}, '
                      f'chunk={sector_chunk})...')
                self.qc.sector_f = self._compute_sectors(
                    rec, n_sectors,
                    stride=sector_stride, chunk_size=sector_chunk)
        else:
            print('\tskipping sector fluorescence '
                  '(compute_sectors=False).')
            self.qc.sector_f = None

        # Registration metrics from suite2p ops.npy
        # ----------
        self.add_registration_metrics(verbose=False)
        _ops = self._load_ops()
        self.qc.reg = self._load_reg_metrics(_ops)
        # Truncate per-frame registration metrics to the kept range so
        # they stay aligned with the (possibly truncated) qc.t / frame_f.
        # Only slice arrays that are at full per-frame length (avoids
        # touching index-style or already-strided fields).
        # ----------
        if _n_keep < _n_full:
            for _attr in ('xoff', 'yoff', 'corrXY', 'badframes',
                          'shift_mag'):
                _v = getattr(self.qc.reg, _attr, None)
                if _v is not None and np.asarray(_v).shape[:1] == (_n_full,):
                    setattr(self.qc.reg, _attr,
                            np.asarray(_v)[:_n_keep])

        # Suite2p corrXY (registration phase-correlation quality),
        # stim- and reward-aligned, saved to data_mbl/ so it can be
        # inspected without re-running plt_qc's full registration panel.
        # ----------
        if self.qc.reg.corrXY is not None:
            self._save_qc_corrxy_eventavg()
        _ref_img = None
        _yrange = None
        _xrange = None
        if isinstance(_ops, dict):
            _ref_img = _ops.get('meanImg', None)
            _yrange = _ops.get('yrange', None)
            _xrange = _ops.get('xrange', None)

        # Custom registration metric: match each frame to a z-stack
        # slice and infer absolute z-position (µm) per frame. Populates
        # reg.z_um, reg.z_um_slices, reg.zstack_corr_mat,
        # reg.zstack_slice_frac. Overrides the z_corr panel at plot time.
        # ----------
        if use_zstack is not None:
            print(f'\tmatching frames to zstack '
                  f'(stride={z_corr_stride}, chunk={z_corr_chunk}, '
                  f'jobs={z_corr_jobs})...')
            _zm = self._compute_zstack_match(
                use_zstack, rec,
                stride=z_corr_stride, chunk_size=z_corr_chunk,
                n_jobs=z_corr_jobs,
                zstack_channel=zstack_channel,
                yrange=_yrange, xrange=_xrange)
            self.qc.reg.z_um = _zm.z_um
            self.qc.reg.z_um_slices = _zm.z_um_slices
            self.qc.reg.zstack_corr_mat = _zm.corr_mat
            self.qc.reg.zstack_slice_frac = _zm.slice_frac
            self.qc.reg.zstack_folder = _zm.zstack_folder
            self.qc.reg.zstack_channel = _zm.zstack_channel

        _dual_done = False
        _ff_dual = None
        if z_corr_dual_ref:
            _n_dual = (z_corr_ref_n_frames
                       if z_corr_ref_n_frames is not None else 500)
            print(f'\tbuilding dual z_corr refs from first/last '
                  f'{_n_dual} frames...')
            _ref_early = self._compute_ref_from_first_frames(rec, _n_dual)
            _ref_late = self._compute_ref_from_last_frames(rec, _n_dual)
            print(f'\tcomputing z_corr [early ref] '
                  f'(stride={z_corr_stride}, chunk={z_corr_chunk}, '
                  f'jobs={z_corr_jobs})...')
            _zc_early, _ff_dual = self._compute_z_corr(
                rec, _ref_early,
                stride=z_corr_stride, chunk_size=z_corr_chunk,
                n_jobs=z_corr_jobs)
            print(f'\tcomputing z_corr [late ref] '
                  f'(stride={z_corr_stride}, chunk={z_corr_chunk}, '
                  f'jobs={z_corr_jobs})...')
            _zc_late, _ = self._compute_z_corr(
                rec, _ref_late,
                stride=z_corr_stride, chunk_size=z_corr_chunk,
                n_jobs=z_corr_jobs)
            self.qc.reg.z_corr_early = _zc_early
            self.qc.reg.z_corr_late = _zc_late
            self.qc.reg.z_corr_diff = (_zc_early
                                       - _zc_late).astype(np.float32)
            # Skip the single-ref pass below.
            _ref_img = None
            _yrange = None
            _xrange = None
            _dual_done = True
        elif z_corr_ref_n_frames is not None:
            print(f'\tbuilding z_corr ref from first '
                  f'{z_corr_ref_n_frames} frames...')
            _ref_img = self._compute_ref_from_first_frames(
                rec, z_corr_ref_n_frames)
            # ref now matches rec frame shape exactly; skip ops cropping.
            _yrange = None
            _xrange = None

        # Whole-frame F + z_corr
        # Single-channel: piggyback frame_f on z_corr's pass (zero I/O cost).
        # Dual-colour: each channel needs its own frame_f pass; z_corr runs
        # on the active channel (self.rec) only.
        # ----------
        if _has_grn and _has_red:
            if compute_frame_f:
                for _ch_label, _ch_rec in [('grn', _rec_grn),
                                           ('red', _rec_red)]:
                    print(f'\textracting whole-frame F [{_ch_label}]'
                          f'{" split_lr" if split_lr else ""} '
                          f'(stride={frame_stride}, '
                          f'chunk={frame_chunk})...')
                    for _sub_lbl, _ys in self._frame_split_items(
                            _ch_label, _ch_rec, split_lr):
                        self.qc.frame_f[_sub_lbl] = self._compute_frame_f(
                            _ch_rec,
                            stride=frame_stride, chunk_size=frame_chunk,
                            y_slice=_ys)
                if correct_signal:
                    # _cs_resolved already has the stim-step keys folded in
                    # (see _resolve_cs_kwargs); copy it so the call-specific
                    # keys below don't leak into the stored/signature dict.
                    _cs_kwargs = dict(_cs_resolved)
                    # save_tif drives the correction's save_to_disk: when
                    # True the trial-averaged corr/dF/F TIFFs are written
                    # (and the scratch full stack is deleted afterwards).
                    _cs_kwargs['save_to_disk'] = bool(save_tif)
                    _cs_kwargs['replace_real'] = False
                    _real_flu = _cs_kwargs.get('real_flu', 'red')
                    _corr_attr = f'rec_{_real_flu}_corr'
                    _agg_attr = f'rec_{_real_flu}_corr_aggregates'
                    # Pre-emptively drop any stale corrected stack or
                    # aggregates dict from a prior add_qc call before
                    # correct_signal allocates a new buffer (otherwise
                    # both sit in RAM).
                    for _stale in (_corr_attr, _agg_attr):
                        if hasattr(self, _stale):
                            delattr(self, _stale)
                    gc.collect()
                    # Inline-aggregates fast path: for full_regress
                    # (and only when split_lr is off so the sector
                    # partition is straightforward, AND save_to_disk is
                    # off because aggregates mode skips the full stack
                    # the disk artifact needs), tell the correction
                    # function to skip the (T, X, Y) output buffer and
                    # stream-compute whole-frame + per-sector mean
                    # traces directly. Removes a 14 GB allocation/write
                    # and two extra read passes over the corrected
                    # stack (frame_f + sector_f extraction).
                    _cs_method = _cs_kwargs.get('method', 'full_regress')
                    # pixel_spatial_subtr: ask the correction for F0-weighted
                    # (ratio-of-means) whole-frame + per-sector aggregates.
                    # The corrected stack is a per-pixel dF/F; a plain
                    # spatial mean of it is a mean-of-ratios that, over dim
                    # pixels with a tiny (per-trial) F0, is dominated by a
                    # heavy tail and inflates the trace (the cells-mode /
                    # per_trial GRABmutant plateau). The weighted aggregate
                    # is the SNR-correct population dF/F; it is used below
                    # for frame_f/sector_f instead of reducing the stack.
                    if (_cs_method == 'pixel_spatial_subtr'
                            and compute_sectors
                            and _cs_kwargs.get('aggregate_sectors') is None):
                        _cs_kwargs['aggregate_sectors'] = n_sectors
                    # Auto-fill the per-trial pixel_spatial_subtr inputs from
                    # behaviour so the caller need not convert clock time to
                    # frames or hand-pick a window. Each is filled only when
                    # not supplied, so an explicit value always wins.
                    if (_cs_method == 'pixel_spatial_subtr'
                            and _cs_kwargs.get('f0_mode') == 'per_trial'):
                        if _cs_kwargs.get('trial_onsets') is None:
                            _onsets = self._trial_onsets_from_stim(
                                _cs_kwargs.get('t_start'))
                            _cs_kwargs['trial_onsets'] = _onsets
                            print(f'\t\tauto-filled trial_onsets from '
                                  f'beh.stim.t_start: {_onsets.size} onsets')
                        if _cs_kwargs.get('trial_window') is None:
                            _win = self._trial_window_from_trialavg(
                                _cs_kwargs.get('trialavg_t_pre', 2.0),
                                _cs_kwargs.get('trialavg_t_post', 4.0))
                            _fit = self._fit_trial_window(
                                _win, _cs_kwargs.get('trial_onsets'))
                            if _fit != _win:
                                print(f'\t\t(trial_window {_win} overlaps '
                                      f'at the tightest ITI; shrunk to '
                                      f'{_fit})')
                            _cs_kwargs['trial_window'] = _fit
                            print(f'\t\tauto-filled trial_window '
                                  f'(frames rel. onset): {_fit}')
                    _agg_methods = ('full_regress',)
                    _use_agg = (
                        _cs_method in _agg_methods
                        and not split_lr
                        and compute_sectors
                        and not _cs_kwargs.get('save_to_disk', False))
                    if _use_agg and 'aggregates_only' not in _cs_kwargs:
                        _cs_kwargs['aggregates_only'] = {
                            'n_sectors': n_sectors}
                    elif (_cs_method in _agg_methods
                          and _cs_kwargs.get('save_to_disk', False)):
                        print('\t\t(note: save_tif=True forces the '
                              'legacy 3-pass + write path; pass '
                              'save_tif=False for the '
                              'aggregates fast path that skips the '
                              '14 GB write and 2 extra reads.)')
                    print(f'\trunning correct_signal '
                          f'(replace_real=False, real_flu={_real_flu}, '
                          f'method={_cs_method}'
                          f'{", aggregates_only" if _use_agg else ""})'
                          f'...')
                    self.correct_signal(**_cs_kwargs)

                    # Branch on what correct_signal produced.
                    _agg = getattr(self, _agg_attr, None)
                    _corr_rec = getattr(self, _corr_attr, None)
                    _corr_lbl = f'{_real_flu}_corr'
                    # Resolve any partial-recording range left by
                    # correct_signal (t_start/t_end/trial_end). Used to
                    # pad the corrected aggregates back to full length
                    # so downstream consumers (plt_qc, response_mean)
                    # can align them with self.qc.t.
                    _cs_range = getattr(self, '_cs_frame_range', None)
                    _cs_n_full = getattr(self, '_cs_n_frames_full', None)

                    def _pad_to_full(_arr, _stride=1):
                        # Pad a corrected-channel aggregate back to the
                        # full-recording length. _stride accounts for
                        # whether the array is at per-frame length
                        # (_stride=1, e.g. frame_f after np.interp) or
                        # strided length (_stride>1, e.g. sector_f
                        # which is sampled every _stride frames with
                        # no post-stride interpolation).
                        if _cs_range is None or _cs_n_full is None:
                            return _arr

                        _a, _b = _cs_range
                        if (_b - _a) == _cs_n_full:
                            return _arr
                        if _stride > 1:
                            _n_target = (_cs_n_full + _stride - 1) \
                                // _stride
                            _a_t = (_a + _stride - 1) // _stride
                        else:
                            _n_target = _cs_n_full
                            _a_t = _a
                        _arr = np.asarray(_arr)
                        _t_in = _arr.shape[-1]
                        if _arr.ndim == 2:
                            _out = np.full(
                                (_arr.shape[0], _n_target),
                                np.nan, dtype=np.float64)
                            _out[:, _a_t:_a_t + _t_in] = _arr
                        else:
                            _out = np.full(
                                _n_target, np.nan, dtype=np.float64)
                            _out[_a_t:_a_t + _t_in] = _arr
                        return _out

                    if _agg is not None:
                        # Aggregates fast path: drop precomputed traces
                        # straight into self.qc.frame_f / sector_f. No
                        # full-stack reads needed.
                        # frame_f comes out at full T — matches what
                        # _compute_frame_f returns after its
                        # post-stride np.interp. No conversion.
                        # sector_f comes out at full T but the legacy
                        # _compute_sectors(stride=sector_stride) only
                        # returns the strided samples (no interpolation
                        # back to T); downstream _plt_qc_response_mean
                        # builds sec_t from that strided length, so we
                        # must downsample here to match. Indices align
                        # with [0, stride, 2*stride, …].
                        _agg_frame = _agg['frame_f']
                        _agg_sec = _agg['sector_f']
                        if sector_stride > 1:
                            _agg_sec = _agg_sec[:, ::sector_stride]
                        print(f'\tusing inline aggregates [{_corr_lbl}] '
                              f'(frame_f shape={_agg_frame.shape}, '
                              f'sector_f shape={_agg_sec.shape}'
                              f'{f", stride={sector_stride}" if sector_stride > 1 else ""}'
                              f').')
                        self.qc.frame_f[_corr_lbl] = _pad_to_full(
                            _agg_frame, _stride=1)
                        if isinstance(self.qc.sector_f, dict):
                            self.qc.sector_f[_corr_lbl] = _pad_to_full(
                                _agg_sec, _stride=sector_stride)
                        delattr(self, _agg_attr)
                        gc.collect()
                    elif _corr_rec is not None:
                        # pixel_spatial_subtr: prefer the correction's
                        # F0-weighted (ratio-of-means) aggregates over a
                        # plain spatial mean of the per-pixel dF/F stack
                        # (which is a mean-of-ratios and inflates on dim,
                        # tiny-F0 pixels — the cells/per_trial plateau). The
                        # corrected trace is dff_sig − dff_ctrl, each
                        # component weighted by its own F0 inside the
                        # correction. Falls back to the plain-mean stack
                        # extraction when the aggregates are unavailable
                        # (older correction, split_lr, or non-pixel method).
                        _ci_agg = getattr(self, '_corrected_info', None)
                        _wf_s = getattr(_ci_agg, 'wf_dff_sig', None)
                        _use_wagg = (_cs_method == 'pixel_spatial_subtr'
                                     and not split_lr
                                     and _wf_s is not None)
                        if _use_wagg:
                            print(f'\tusing F0-weighted aggregates '
                                  f'[{_corr_lbl}] (ratio-of-means).')
                            _wf_corr = (np.asarray(_ci_agg.wf_dff_sig,
                                                   dtype=np.float64)
                                        - np.asarray(_ci_agg.wf_dff_ctrl,
                                                     dtype=np.float64))
                            self.qc.frame_f[_corr_lbl] = _pad_to_full(
                                _wf_corr, _stride=1)
                            _sec_s = getattr(_ci_agg, 'sec_dff_sig', None)
                            if (compute_sectors
                                    and isinstance(self.qc.sector_f, dict)
                                    and _sec_s is not None):
                                _sec_corr = (
                                    np.asarray(_ci_agg.sec_dff_sig,
                                               dtype=np.float64)
                                    - np.asarray(_ci_agg.sec_dff_ctrl,
                                                 dtype=np.float64))
                                if sector_stride > 1:
                                    _sec_corr = _sec_corr[:, ::sector_stride]
                                self.qc.sector_f[_corr_lbl] = _pad_to_full(
                                    _sec_corr, _stride=sector_stride)
                        else:
                            # Legacy plain-mean extraction from the stack.
                            print(f'\textracting whole-frame F '
                                  f'[{_corr_lbl}]'
                                  f'{" split_lr" if split_lr else ""} '
                                  f'(stride={frame_stride}, '
                                  f'chunk={frame_chunk})...')
                            for _sub_lbl, _ys in self._frame_split_items(
                                    _corr_lbl, _corr_rec, split_lr):
                                self.qc.frame_f[_sub_lbl] = _pad_to_full(
                                    self._compute_frame_f(
                                        _corr_rec,
                                        stride=frame_stride,
                                        chunk_size=frame_chunk,
                                        y_slice=_ys),
                                    _stride=1)
                            if compute_sectors and isinstance(
                                    self.qc.sector_f, dict):
                                print(f'\textracting sector fluorescence '
                                      f'[{_corr_lbl}] '
                                      f'(stride={sector_stride}, '
                                      f'chunk={sector_chunk})...')
                                self.qc.sector_f[_corr_lbl] = _pad_to_full(
                                    self._compute_sectors(
                                        _corr_rec, n_sectors,
                                        stride=sector_stride,
                                        chunk_size=sector_chunk),
                                    _stride=sector_stride)
                        del _corr_rec
                        delattr(self, _corr_attr)
                        gc.collect()
                    else:
                        print(f'\t\t{_corr_attr} not produced; '
                              f'skipping corrected whole-frame F.')

                    # Diagnostic (pixel_spatial_subtr): dump the whole-frame
                    # component traces (dff_sig, dff_ctrl) alongside the
                    # corrected trace and the real stim onsets, so a
                    # residual can be attributed to a channel offline
                    # (e.g. the blurred/masked control carrying a
                    # stim-locked step the unit-gain subtraction then
                    # injects). mean(dff_sig) − mean(dff_ctrl) ≈ red_corr.
                    _ci_diag = getattr(self, '_corrected_info', None)
                    if (_cs_method == 'pixel_spatial_subtr'
                            and _ci_diag is not None
                            and getattr(_ci_diag, 'wf_dff_sig', None)
                            is not None):
                        try:
                            _diag = {
                                'wf_dff_sig': _pad_to_full(
                                    np.asarray(_ci_diag.wf_dff_sig),
                                    _stride=1),
                                'wf_dff_ctrl': _pad_to_full(
                                    np.asarray(_ci_diag.wf_dff_ctrl),
                                    _stride=1),
                                'wf_corr': np.asarray(
                                    self.qc.frame_f.get(_corr_lbl)),
                                't': np.asarray(self.qc.t),
                                'stim_t': np.asarray(self._qc_stim_t()),
                                'corr_key': _corr_lbl,
                                'correct_signal_kwargs': dict(
                                    getattr(self.qc,
                                            'correct_signal_kwargs', {})
                                    or {}),
                            }
                            _diag_name = (
                                f'{self.path.animal}_{self.path.date}_'
                                f'{self.path.beh_folder}'
                                f'_corr_components_diag.npy')
                            _diag_path = os.path.join(
                                str(self.folder.data), _diag_name)
                            np.save(_diag_path, _diag, allow_pickle=True)
                            print(f'\tsaved component diagnostic: '
                                  f'{_diag_name}')
                        except Exception as _e:
                            print(f'\twarning: component diagnostic save '
                                  f'failed ({_e})')

                    # When save_to_disk wrote the full corrected stack
                    # to disk, it is a scratch file only — correct_signal
                    # has already streamed the trial-averaged corr/dF/F
                    # TIFFs from it, and the QC aggregates above were
                    # extracted from it. Delete it now (the memmap was
                    # released by the del/gc above) so only the small
                    # trial-averaged TIFFs persist.
                    _full_stack_path = getattr(
                        self, '_cs_full_stack_path', None)
                    if _full_stack_path is not None \
                            and os.path.exists(_full_stack_path):
                        try:
                            os.remove(_full_stack_path)
                            print(f'\tremoved scratch corrected stack '
                                  f'{os.path.basename(_full_stack_path)} '
                                  f'(trial-averaged TIFFs retained).')
                        except OSError as _e:
                            print(f'\twarning: could not remove scratch '
                                  f'corrected stack '
                                  f'{_full_stack_path} ({_e})')
                    self._cs_full_stack_path = None
            if _ref_img is not None:
                print(f'\tcomputing z_corr (stride={z_corr_stride}, '
                      f'chunk={z_corr_chunk}, jobs={z_corr_jobs})...')
                self.qc.reg.z_corr, _ = self._compute_z_corr(
                    rec, _ref_img,
                    stride=z_corr_stride,
                    chunk_size=z_corr_chunk,
                    n_jobs=z_corr_jobs,
                    yrange=_yrange,
                    xrange=_xrange)
            elif not _dual_done:
                print('\t\tno meanImg in ops.npy; z_corr will be blank.')
        else:
            _label = channel if channel is not None else 'frame'
            if _ref_img is not None:
                print(f'\tcomputing z_corr + whole-frame F '
                      f'(stride={z_corr_stride}, chunk={z_corr_chunk}, '
                      f'jobs={z_corr_jobs})...')
                _zc, _ff = self._compute_z_corr(
                    rec, _ref_img,
                    stride=z_corr_stride,
                    chunk_size=z_corr_chunk,
                    n_jobs=z_corr_jobs,
                    yrange=_yrange,
                    xrange=_xrange)
                self.qc.reg.z_corr = _zc
                if compute_frame_f:
                    if split_lr:
                        # Piggyback frame_f covers full FOV; need halves.
                        print(f'\textracting whole-frame F split_lr '
                              f'(stride={frame_stride}, '
                              f'chunk={frame_chunk})...')
                        for _sub_lbl, _ys in self._frame_split_items(
                                _label, rec, split_lr):
                            self.qc.frame_f[_sub_lbl] = \
                                self._compute_frame_f(
                                    rec,
                                    stride=frame_stride,
                                    chunk_size=frame_chunk,
                                    y_slice=_ys)
                    else:
                        self.qc.frame_f[_label] = _ff
            elif _dual_done and compute_frame_f:
                # The early-ref dual pass already produced per-frame mean F
                # for free; piggyback it instead of running another pass.
                if split_lr:
                    print(f'\textracting whole-frame F split_lr '
                          f'(stride={frame_stride}, '
                          f'chunk={frame_chunk})...')
                    for _sub_lbl, _ys in self._frame_split_items(
                            _label, rec, split_lr):
                        self.qc.frame_f[_sub_lbl] = self._compute_frame_f(
                            rec,
                            stride=frame_stride, chunk_size=frame_chunk,
                            y_slice=_ys)
                else:
                    self.qc.frame_f[_label] = _ff_dual
            elif compute_frame_f:
                print(f'\textracting whole-frame F'
                      f'{" split_lr" if split_lr else ""} '
                      f'(stride={frame_stride}, chunk={frame_chunk})...')
                for _sub_lbl, _ys in self._frame_split_items(
                        _label, rec, split_lr):
                    self.qc.frame_f[_sub_lbl] = self._compute_frame_f(
                        rec, stride=frame_stride, chunk_size=frame_chunk,
                        y_slice=_ys)
            elif not _dual_done:
                print('\t\tno meanImg in ops.npy and frame_f disabled.')

        # Single-channel hemisphere-control 1-D correction. When
        # split_lr=True, correct_signal=True, grab5ht_side is set, and
        # the recording is single-channel, apply the chosen correction
        # method on the two hemisphere mean traces (s1 = mut side,
        # s2 = grab side) and store the result under
        # `{label}_corr_{grab5ht_side}`. Defaults to full_regress
        # (1-D port of the pixel-wise pipeline).
        # ----------
        _single_ch = not (_has_grn and _has_red)
        _grab_side = self.qc.grab5ht_side
        if (split_lr and correct_signal and _single_ch
                and compute_frame_f and _grab_side is not None):
            _label = channel if channel is not None else 'frame'
            _ff_grab = self.qc.frame_f.get(f'{_label}_{_grab_side}')
            _mut_side = 'right' if _grab_side == 'left' else 'left'
            _ff_mut = self.qc.frame_f.get(f'{_label}_{_mut_side}')
            if _ff_grab is None or _ff_mut is None:
                print(f'\t\t[1d corr] missing {_label}_{_grab_side} '
                      f'or {_label}_{_mut_side}; skipping.')
            else:
                _cs_kwargs = dict(_cs_resolved)
                _method = _cs_kwargs.get('method', 'full_regress')
                print(f'\trunning 1-D signal correction '
                      f'(method={_method}, '
                      f'grab={_label}_{_grab_side}, '
                      f'mut={_label}_{_mut_side})...')
                if _method == 'full_regress':
                    from .signal_correction import \
                        correct_full_regress_1d
                    _kw_1d = {
                        k: _cs_kwargs[k] for k in (
                            'smooth_window', 'airpls_lam',
                            'airpls_porder', 'airpls_max_iter',
                            'trim_initial', 'nn_slope',
                            'beta_loss', 'beta_f_scale', 'beta_scale')
                        if k in _cs_kwargs}
                    _corr_tr, _beta = correct_full_regress_1d(
                        np.asarray(_ff_mut, dtype=np.float64),
                        np.asarray(_ff_grab, dtype=np.float64),
                        verbose=True, **_kw_1d)
                else:
                    print(f'\t\t[1d corr] method={_method!r} not '
                          f'supported for 1-D hemisphere correction; '
                          f'falling back to full_regress.')
                    from .signal_correction import \
                        correct_full_regress_1d
                    _corr_tr, _beta = correct_full_regress_1d(
                        np.asarray(_ff_mut, dtype=np.float64),
                        np.asarray(_ff_grab, dtype=np.float64),
                        verbose=True)
                _corr_key = f'{_label}_corr_{_grab_side}'
                self.qc.frame_f[_corr_key] = _corr_tr
                self.qc.grab5ht_beta = float(_beta)
                print(f'\t\tstored corrected trace at '
                      f'frame_f[{_corr_key!r}] (β = {_beta:.6g}).')

        # Optional post-processing: linear detrend of the raw red/grn
        # whole-frame and per-sector traces. Applied here so everything
        # downstream (plt_qc, _plt_qc_response_mean, etc.) transparently
        # uses the processed signals.
        # ----------
        if detrend_sigs:
            print('\tpost-processing red/grn traces: detrend...')
            self._apply_detrend_sigs()

        print('done.\n')
        return

    def _compute_ref_from_first_frames(self, rec, n_frames):
        """Mean image over the first n_frames of rec.

        Used as an alternative z_corr reference (vs. ops['meanImg']) so
        that z_corr tracks drift relative to the session start.

        Parameters
        ----------
        rec : np.memmap or np.ndarray
            Imaging stack, shape (n_frames_total, Ly, Lx).
        n_frames : int
            Number of leading frames to average.

        Returns
        -------
        ref_img : np.ndarray, float32
            Mean image, shape (Ly, Lx).
        """
        n = min(int(n_frames), rec.shape[0])
        return np.asarray(rec[:n], dtype=np.float32).mean(axis=0)

    def _compute_ref_from_last_frames(self, rec, n_frames):
        """Mean image over the last n_frames of rec.

        Companion to _compute_ref_from_first_frames; together they form
        the dual-reference z_corr used to detect signed z-drift.

        Parameters
        ----------
        rec : np.memmap or np.ndarray
            Imaging stack, shape (n_frames_total, Ly, Lx).
        n_frames : int
            Number of trailing frames to average.

        Returns
        -------
        ref_img : np.ndarray, float32
            Mean image, shape (Ly, Lx).
        """
        n = min(int(n_frames), rec.shape[0])
        return np.asarray(rec[-n:], dtype=np.float32).mean(axis=0)

    def _frame_split_items(self, label, rec, split_lr):
        """Return (sub_label, y_slice) pairs for whole-frame F passes.

        With split_lr=False, returns a single (label, None) entry. With
        split_lr=True, splits the frame y-axis into top half ('right')
        and bottom half ('left') and returns one entry per half.
        """
        if not split_lr:
            return [(label, None)]
        Ly = rec.shape[1]
        _half = Ly // 2
        return [(f'{label}_right', slice(0, _half)),
                (f'{label}_left', slice(_half, Ly))]

    def _load_ops(self):
        """Load and return the suite2p ops dict, or None if unavailable."""
        if hasattr(self, 'neur') and hasattr(self.neur, 'ops'):
            _ops = self.neur.ops
        else:
            _path = os.path.join(self.folder.img, 'suite2p', 'plane0',
                                 'ops.npy')
            if not os.path.isfile(_path):
                print(f'\tno suite2p ops.npy at {_path}; '
                      'registration metrics will be blank.')
                return None
            _ops = np.load(_path, allow_pickle=True)

        # np.load(allow_pickle=True) returns a 0-d ndarray holding a dict
        if isinstance(_ops, np.ndarray) and _ops.shape == ():
            _ops = _ops.item()
        return _ops

    def _load_reg_metrics(self, ops):
        """Populate xoff/yoff/corrXY/badframes from an ops dict.

        Parameters
        ----------
        ops : dict or None
            Suite2p ops dictionary returned by _load_ops.

        Returns
        -------
        reg : SimpleNamespace
            Registration metrics; any missing key is None.
        """
        reg = SimpleNamespace(xoff=None, yoff=None, shift_mag=None,
                              corrXY=None, badframes=None, z_corr=None,
                              z_corr_early=None, z_corr_late=None,
                              z_corr_diff=None,
                              z_um=None, z_um_slices=None,
                              zstack_corr_mat=None,
                              zstack_slice_frac=None,
                              zstack_folder=None,
                              zstack_channel=None)
        if ops is None:
            return reg

        def _get(key):
            try:
                val = ops[key]
            except (KeyError, TypeError, IndexError):
                return None
            return np.asarray(val) if val is not None else None

        reg.xoff = _get('xoff')
        reg.yoff = _get('yoff')
        reg.corrXY = _get('corrXY')
        reg.badframes = _get('badframes')
        if reg.xoff is not None and reg.yoff is not None:
            reg.shift_mag = np.sqrt(
                reg.xoff.astype(float)**2 + reg.yoff.astype(float)**2)
        return reg

    def _compute_sectors(self, rec, n_sectors, stride=1, chunk_size=500):
        """Per-sector mean fluorescence over time, in one chunked pass.

        The naive nested-slice approach reads the imaging memmap once per
        sector (n_sectors**2 full passes). This implementation reads the
        memmap exactly once in contiguous strided chunks and computes all
        sector means in RAM via vectorised reshape+mean.

        Parameters
        ----------
        rec : np.memmap or np.ndarray
            Imaging stack, shape (n_frames, Ly, Lx).
        n_sectors : int
            Sector grid size per axis.
        stride : int
            Sample every stride-th frame.
        chunk_size : int
            Number of sampled frames per chunked read.

        Returns
        -------
        sector_f : np.ndarray, float32
            Per-sector mean fluorescence, shape (n_sectors**2, n_compute)
            where n_compute = ceil(n_frames / stride).
        """
        n_frames, Ly, Lx = rec.shape[0], rec.shape[1], rec.shape[2]

        # Trim FOV so each axis divides evenly into n_sectors blocks
        Ly_t = (Ly // n_sectors) * n_sectors
        Lx_t = (Lx // n_sectors) * n_sectors
        by = Ly_t // n_sectors
        bx = Lx_t // n_sectors

        n_compute = (n_frames + stride - 1) // stride
        sector_f = np.zeros((n_sectors * n_sectors, n_compute),
                            dtype=np.float32)

        chunk_starts = list(range(0, n_compute, chunk_size))
        n_chunks = len(chunk_starts)
        for _ci, c_pos in enumerate(chunk_starts):
            n_in_chunk = min(chunk_size, n_compute - c_pos)
            f_start = c_pos * stride
            f_end = f_start + n_in_chunk * stride
            print(f'\t\tchunk {_ci + 1}/{n_chunks}...      ', end='\r')
            # Single sequential strided read; trim to divisible dims
            block = np.asarray(
                rec[f_start:f_end:stride, :Ly_t, :Lx_t],
                dtype=np.float32)
            k = block.shape[0]
            # (k, n_sec, by, n_sec, bx) → mean over within-block axes.
            # nan-aware: a NaN-masked corrected stack would otherwise
            # poison every sector it touches.
            means = _nanmean(block.reshape(k, n_sectors, by,
                                           n_sectors, bx), axis=(2, 4))
            sector_f[:, c_pos:c_pos + k] = means.reshape(k, -1).T
        print('')
        return sector_f

    def _compute_z_corr(self, rec, ref_img, stride=1, chunk_size=200,
                        n_jobs=4, yrange=None, xrange=None):
        """Per-frame zero-lag Pearson correlation + whole-frame mean.

        Reads the memmap in contiguous strided slices and dispatches
        fixed-size chunks to a thread pool. Numpy releases the GIL during
        array ops, so threads run in parallel on multi-core machines.
        Whole-frame mean fluorescence is computed for free from the same
        flattened chunks (it's already needed as the per-frame mean for
        Pearson centering).

        Parameters
        ----------
        rec : np.memmap or np.ndarray
            Registered imaging stack, shape (n_frames, Ly, Lx).
        ref_img : np.ndarray
            Reference image, shape (Ly_ref, Lx_ref). Typically
            ops['meanImg']. May be smaller than the rec frame size if
            suite2p trimmed edges during registration.
        stride : int
            Evaluate every stride-th frame; remaining frames are linearly
            interpolated.
        chunk_size : int
            Sampled frames per thread-pool task.
        n_jobs : int
            Thread-pool size.
        yrange, xrange : array-like length 2, or None
            Optional (lo, hi) bounds from ops['yrange']/['xrange']. Used
            to crop rec to the region matching ref_img when their shapes
            differ. If absent and shapes differ, fall back to a centred
            crop (with a warning).

        Returns
        -------
        z_corr : np.ndarray, float32
            Per-frame Pearson correlation with ref_img, shape (n_frames,).
        frame_f : np.ndarray, float32
            Per-frame mean fluorescence (over the cropped region),
            shape (n_frames,).
        """
        n_frames, Ly_r, Lx_r = rec.shape[0], rec.shape[1], rec.shape[2]
        Ly_ref, Lx_ref = ref_img.shape

        # Reconcile rec frame shape with ref_img shape. Either side may be
        # larger: e.g. suite2p run on the green channel may produce a
        # 512x512 meanImg while the registered red tiff is 492x492, or
        # vice-versa. yrange/xrange (if present) authoritatively describe
        # the valid analysis region within the *larger* FOV.
        # ----------
        _yr_size = (int(yrange[1]) - int(yrange[0])
                    if yrange is not None else None)
        _xr_size = (int(xrange[1]) - int(xrange[0])
                    if xrange is not None else None)

        if (Ly_r, Lx_r) == (Ly_ref, Lx_ref):
            y_slice = slice(None)
            x_slice = slice(None)
            ref_crop = ref_img
        elif (yrange is not None and xrange is not None
              and _yr_size == Ly_ref and _xr_size == Lx_ref
              and (Ly_r, Lx_r) != (Ly_ref, Lx_ref)):
            # yrange/xrange match ref dims → ref is already cropped, rec
            # is the larger raw FOV. Crop rec to match ref.
            y_slice = slice(int(yrange[0]), int(yrange[1]))
            x_slice = slice(int(xrange[0]), int(xrange[1]))
            ref_crop = ref_img
            print(f'\t\tcropping rec via ops yrange/xrange to '
                  f'{ref_img.shape}.')
        elif (yrange is not None and xrange is not None
              and _yr_size == Ly_r and _xr_size == Lx_r
              and (Ly_r, Lx_r) != (Ly_ref, Lx_ref)):
            # yrange/xrange match rec dims → rec is already cropped, ref
            # is the larger raw FOV. Crop ref to match rec.
            y_slice = slice(None)
            x_slice = slice(None)
            ref_crop = ref_img[int(yrange[0]):int(yrange[1]),
                               int(xrange[0]):int(xrange[1])]
            print(f'\t\tcropping ref_img via ops yrange/xrange to '
                  f'{ref_crop.shape}.')
        else:
            # Centre-crop both to their common (min) size.
            Ly_t = min(Ly_r, Ly_ref)
            Lx_t = min(Lx_r, Lx_ref)
            _y0_r = (Ly_r - Ly_t) // 2
            _x0_r = (Lx_r - Lx_t) // 2
            _y0_ref = (Ly_ref - Ly_t) // 2
            _x0_ref = (Lx_ref - Lx_t) // 2
            y_slice = slice(_y0_r, _y0_r + Ly_t)
            x_slice = slice(_x0_r, _x0_r + Lx_t)
            ref_crop = ref_img[_y0_ref:_y0_ref + Ly_t,
                               _x0_ref:_x0_ref + Lx_t]
            print(f'\t\twarning: rec {(Ly_r, Lx_r)} != ref {ref_img.shape}'
                  f'; centre-cropping both to {(Ly_t, Lx_t)} '
                  f'(no usable yrange/xrange in ops).')

        ref = ref_crop.flatten().astype(np.float32)
        ref_z = ref - ref.mean()
        ref_norm = float(np.linalg.norm(ref_z))

        n_compute = (n_frames + stride - 1) // stride
        corrs = np.empty(n_compute, dtype=np.float32)
        means = np.empty(n_compute, dtype=np.float32)
        chunk_positions = list(range(0, n_compute, chunk_size))

        def _process(c_pos):
            n_in_chunk = min(chunk_size, n_compute - c_pos)
            f_start = c_pos * stride
            f_end = f_start + n_in_chunk * stride
            frames = np.asarray(
                rec[f_start:f_end:stride, y_slice, x_slice],
                dtype=np.float32)
            flat = frames.reshape(n_in_chunk, -1)
            mu = flat.mean(axis=1)
            centered = flat - mu[:, None]
            norms = np.linalg.norm(centered, axis=1)
            return (centered @ ref_z / (norms * ref_norm + 1e-9), mu)

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_jobs) as pool:
            for c_pos, (corr_chunk, mean_chunk) in zip(
                    chunk_positions,
                    pool.map(_process, chunk_positions)):
                corrs[c_pos:c_pos + len(corr_chunk)] = corr_chunk
                means[c_pos:c_pos + len(mean_chunk)] = mean_chunk

        if stride > 1:
            ind_computed = np.arange(0, n_frames, stride)
            x_full = np.arange(n_frames)
            z_corr = np.interp(x_full, ind_computed,
                               corrs).astype(np.float32)
            frame_f = np.interp(x_full, ind_computed,
                                means).astype(np.float32)
            return z_corr, frame_f
        return corrs, means

    def _compute_zstack_match(self, zstack_folder, rec,
                              stride=4, chunk_size=200, n_jobs=4,
                              zstack_channel=None,
                              yrange=None, xrange=None):
        """Match each rec frame to the best slice of a Bruker z-stack.

        Pearson-correlates strided rec frames against every slice of a
        Bruker .ome.tif z-stack, then triangulates a sub-slice position
        via parabolic interpolation around the per-frame peak. The XML
        sidecar is parsed to attach a real z-position (µm) to every
        slice index, so the inferred best slice converts directly to a
        z-position (µm) per frame.

        Parameters
        ----------
        zstack_folder : str
            Path to a Bruker ZSeries folder. Must contain the per-cycle
            multi-page Ch1/Ch2 .ome.tif files and the .xml sidecar with
            per-frame `positionCurrent` ZAxis values.
        rec : np.memmap or np.ndarray
            Registered imaging stack, shape (n_frames, Ly, Lx).
        stride : int
            Evaluate every stride-th frame; remaining frames are
            linearly interpolated in the output z-position trace.
        chunk_size : int
            Sampled frames per thread-pool task.
        n_jobs : int
            Thread-pool size.
        zstack_channel : str or None
            'Ch1' (red) or 'Ch2' (green). Default None: chosen to match
            the active channel of `rec` if it can be inferred (rec_grn
            → Ch2, rec_red → Ch1); otherwise falls back to Ch2.
        yrange, xrange : array-like length 2, or None
            Optional (lo, hi) bounds from ops['yrange']/['xrange']. Used
            to crop rec/zstack to the analysis region when their shapes
            differ.

        Returns
        -------
        out : SimpleNamespace
            With attributes:
                .z_um           : (n_frames,) float32 — inferred z (µm)
                .z_um_slices    : (n_slices,) float64 — slice z (µm)
                .corr_mat       : (n_compute, n_slices) float32
                .slice_frac     : (n_compute,) float32 — parabolic-
                                  refined fractional slice index
                .zstack_folder  : str
                .zstack_channel : str
        """
        # Locate XML sidecar
        # ----------
        _xml_path = None
        for _f in sorted(os.listdir(zstack_folder)):
            if (_f.lower().endswith('.xml')
                    and not _f.upper().endswith('BACKUP.XML')):
                _xml_path = os.path.join(zstack_folder, _f)
                break
        if _xml_path is None:
            raise FileNotFoundError(
                f'No .xml sidecar in {zstack_folder}')

        # Parse per-slice ZAxis (µm). Bruker writes the per-frame
        # positionCurrent inside each <Frame>'s PVStateShard; the
        # global PVStateShard at the top of the Sequence reflects the
        # initial stage position only, so we iterate over <Frame>.
        # ----------
        _tree = ElementTree.parse(_xml_path)
        _root = _tree.getroot()
        _slice_pairs = []
        for _frame in _root.iter('Frame'):
            try:
                _fi = int(_frame.get('index'))
            except (TypeError, ValueError):
                continue
            _z = None
            for _pv in _frame.iter('PVStateValue'):
                if _pv.get('key') != 'positionCurrent':
                    continue
                for _sv in _pv.iter('SubindexedValues'):
                    if _sv.get('index') != 'ZAxis':
                        continue
                    for _sub in _sv.iter('SubindexedValue'):
                        try:
                            _z = float(_sub.get('value'))
                        except (TypeError, ValueError):
                            _z = None
                        break
                    break
                break
            if _z is not None:
                _slice_pairs.append((_fi, _z))
        if not _slice_pairs:
            raise RuntimeError(
                f'No per-frame ZAxis positions found in {_xml_path}')
        _slice_pairs.sort()
        z_um_slices = np.array([_z for _, _z in _slice_pairs],
                               dtype=np.float64)
        n_slices = z_um_slices.size

        # Pick zstack channel; default matches the active rec.
        # ----------
        if zstack_channel is None:
            _ch = 'Ch2'
            if (hasattr(self, 'rec_grn')
                    and getattr(self, 'rec', None) is self.rec_grn):
                _ch = 'Ch2'
            elif (hasattr(self, 'rec_red')
                    and getattr(self, 'rec', None) is self.rec_red):
                _ch = 'Ch1'
        else:
            _ch = zstack_channel

        # Find the matching multi-page .ome.tif. Bruker writes a single
        # multi-page file per channel per cycle; take the first match.
        # ----------
        _tif_path = None
        for _f in sorted(os.listdir(zstack_folder)):
            if (_f.lower().endswith('.tif')
                    and f'_{_ch}_' in _f):
                _tif_path = os.path.join(zstack_folder, _f)
                break
        if _tif_path is None:
            raise FileNotFoundError(
                f'No {_ch} .tif in {zstack_folder}')

        # Read all slices into memory; z-stacks are small (~20-200
        # slices) so this is cheap.
        # ----------
        _zstack = tifffile.imread(_tif_path)
        if _zstack.ndim == 2:
            _zstack = _zstack[None]
        if _zstack.shape[0] != n_slices:
            print(f'\t\twarning: zstack tif has {_zstack.shape[0]} '
                  f'slices but XML lists {n_slices}; truncating '
                  f'to min.')
            _n_use = min(_zstack.shape[0], n_slices)
            _zstack = _zstack[:_n_use]
            z_um_slices = z_um_slices[:_n_use]
            n_slices = _n_use

        # Reconcile rec / zstack frame shapes (mirrors _compute_z_corr).
        # ----------
        _Ly_r, _Lx_r = rec.shape[1], rec.shape[2]
        _Ly_z, _Lx_z = _zstack.shape[1], _zstack.shape[2]
        _yr_size = (int(yrange[1]) - int(yrange[0])
                    if yrange is not None else None)
        _xr_size = (int(xrange[1]) - int(xrange[0])
                    if xrange is not None else None)

        if (_Ly_r, _Lx_r) == (_Ly_z, _Lx_z):
            y_slice = slice(None)
            x_slice = slice(None)
            z_crop = _zstack
        elif (yrange is not None and xrange is not None
              and _yr_size == _Ly_z and _xr_size == _Lx_z
              and (_Ly_r, _Lx_r) != (_Ly_z, _Lx_z)):
            y_slice = slice(int(yrange[0]), int(yrange[1]))
            x_slice = slice(int(xrange[0]), int(xrange[1]))
            z_crop = _zstack
        elif (yrange is not None and xrange is not None
              and _yr_size == _Ly_r and _xr_size == _Lx_r
              and (_Ly_r, _Lx_r) != (_Ly_z, _Lx_z)):
            y_slice = slice(None)
            x_slice = slice(None)
            z_crop = _zstack[:, int(yrange[0]):int(yrange[1]),
                             int(xrange[0]):int(xrange[1])]
        else:
            _Ly_t = min(_Ly_r, _Ly_z)
            _Lx_t = min(_Lx_r, _Lx_z)
            _y0_r = (_Ly_r - _Ly_t) // 2
            _x0_r = (_Lx_r - _Lx_t) // 2
            _y0_z = (_Ly_z - _Ly_t) // 2
            _x0_z = (_Lx_z - _Lx_t) // 2
            y_slice = slice(_y0_r, _y0_r + _Ly_t)
            x_slice = slice(_x0_r, _x0_r + _Lx_t)
            z_crop = _zstack[:, _y0_z:_y0_z + _Ly_t,
                             _x0_z:_x0_z + _Lx_t]
            print(f'\t\twarning: rec {(_Ly_r, _Lx_r)} != zstack '
                  f'{(_Ly_z, _Lx_z)}; centre-cropping both to '
                  f'{(_Ly_t, _Lx_t)}.')

        # Zero-center + unit-normalise each slice once. Then per-frame
        # Pearson reduces to a single matrix multiply.
        # ----------
        _Z = z_crop.reshape(n_slices, -1).astype(np.float32)
        _Z -= _Z.mean(axis=1, keepdims=True)
        _zn = np.linalg.norm(_Z, axis=1)
        _zn[_zn == 0] = 1.0
        _Z = _Z / _zn[:, None]
        _Z_T = _Z.T  # (n_px, n_slices)

        n_frames = rec.shape[0]
        n_compute = (n_frames + stride - 1) // stride
        corr_mat = np.full((n_compute, n_slices), np.nan,
                           dtype=np.float32)

        chunk_starts = list(range(0, n_compute, chunk_size))
        n_chunks = len(chunk_starts)

        def _process(c_pos):
            n_in_chunk = min(chunk_size, n_compute - c_pos)
            f_start = c_pos * stride
            f_end = f_start + n_in_chunk * stride
            block = np.asarray(
                rec[f_start:f_end:stride, y_slice, x_slice],
                dtype=np.float32)
            _F = block.reshape(block.shape[0], -1)
            _F -= _F.mean(axis=1, keepdims=True)
            _fn = np.linalg.norm(_F, axis=1)
            _fn[_fn == 0] = 1.0
            _F = _F / _fn[:, None]
            return _F @ _Z_T

        if n_jobs > 1 and n_chunks > 1:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=n_jobs) as _pool:
                for _ci, (c_pos, _cs) in enumerate(zip(
                        chunk_starts,
                        _pool.map(_process, chunk_starts))):
                    print(f'\t\tchunk {_ci + 1}/{n_chunks}...      ',
                          end='\r')
                    corr_mat[c_pos:c_pos + _cs.shape[0]] = _cs
        else:
            for _ci, c_pos in enumerate(chunk_starts):
                print(f'\t\tchunk {_ci + 1}/{n_chunks}...      ',
                      end='\r')
                _cs = _process(c_pos)
                corr_mat[c_pos:c_pos + _cs.shape[0]] = _cs
        print('')

        # Parabolic refinement around the per-frame peak. For interior
        # peaks fit y = a(s − s*)^2 + c through (s−1, s, s+1) and take
        # the analytical vertex offset; boundary peaks stay integer.
        # ----------
        _idx_max = np.argmax(corr_mat, axis=1)
        slice_frac = _idx_max.astype(np.float64)
        for _i in range(n_compute):
            _imax = int(_idx_max[_i])
            if _imax <= 0 or _imax >= n_slices - 1:
                continue
            _y0 = float(corr_mat[_i, _imax - 1])
            _y1 = float(corr_mat[_i, _imax])
            _y2 = float(corr_mat[_i, _imax + 1])
            _denom = _y0 - 2.0 * _y1 + _y2
            if _denom == 0 or not np.isfinite(_denom):
                continue
            _offset = 0.5 * (_y0 - _y2) / _denom
            if not np.isfinite(_offset):
                continue
            _offset = float(np.clip(_offset, -1.0, 1.0))
            slice_frac[_i] = _imax + _offset

        # Convert fractional slice index to z (µm).
        # ----------
        _slice_axis = np.arange(n_slices, dtype=np.float64)
        z_um_compute = np.interp(
            slice_frac, _slice_axis, z_um_slices,
            left=float(z_um_slices[0]),
            right=float(z_um_slices[-1]))

        # Interpolate to the full per-frame timeline.
        # ----------
        if stride > 1:
            _x_compute = np.arange(n_compute, dtype=np.float64) * stride
            _x_all = np.arange(n_frames, dtype=np.float64)
            z_um_full = np.interp(_x_all, _x_compute, z_um_compute)
        else:
            z_um_full = z_um_compute

        return SimpleNamespace(
            z_um=z_um_full.astype(np.float32),
            z_um_slices=z_um_slices.astype(np.float64),
            corr_mat=corr_mat,
            slice_frac=slice_frac.astype(np.float32),
            zstack_folder=zstack_folder,
            zstack_channel=_ch)

    def _compute_frame_f(self, rec, stride=1, chunk_size=500, y_slice=None):
        """Whole-frame mean fluorescence over time, in chunked passes.

        Used when z_corr piggyback isn't possible (dual-colour second
        channel, or single-channel without ops['meanImg']).

        Parameters
        ----------
        rec : np.memmap or np.ndarray
            Imaging stack, shape (n_frames, Ly, Lx).
        stride : int
            Sample every stride-th frame; output is interpolated to
            n_frames so it aligns with self.qc.t.
        chunk_size : int
            Number of sampled frames per chunked read.
        y_slice : slice or None
            Optional restriction along the y-axis (used for split_lr).

        Returns
        -------
        frame_f : np.ndarray, float32
            Per-frame mean fluorescence, shape (n_frames,).
        """
        n_frames = rec.shape[0]
        n_compute = (n_frames + stride - 1) // stride
        means = np.zeros(n_compute, dtype=np.float32)

        chunk_starts = list(range(0, n_compute, chunk_size))
        n_chunks = len(chunk_starts)
        for _ci, c_pos in enumerate(chunk_starts):
            n_in_chunk = min(chunk_size, n_compute - c_pos)
            f_start = c_pos * stride
            f_end = f_start + n_in_chunk * stride
            print(f'\t\tchunk {_ci + 1}/{n_chunks}...      ', end='\r')
            if y_slice is None:
                block = np.asarray(rec[f_start:f_end:stride],
                                   dtype=np.float32)
            else:
                block = np.asarray(rec[f_start:f_end:stride, y_slice],
                                   dtype=np.float32)
            means[c_pos:c_pos + n_in_chunk] = _nanmean(block, axis=(1, 2))
        print('')

        if stride > 1:
            ind_computed = np.arange(0, n_frames, stride)
            return np.interp(
                np.arange(n_frames), ind_computed, means
            ).astype(np.float32)
        return means

    def _mean_image(self, rec, stride=4, chunk_size=500):
        """Temporal mean image of a stack, computed in chunked passes.

        Memory-safe for memory-mapped stacks: never holds more than
        `chunk_size` sampled frames in RAM at once.

        Parameters
        ----------
        rec : np.memmap or np.ndarray
            Imaging stack, shape (n_frames, Ly, Lx).
        stride : int
            Sample every stride-th frame (the temporal mean is robust to
            subsampling). Default 4.
        chunk_size : int
            Number of sampled frames per chunked read. Default 500.

        Returns
        -------
        mean_img : np.ndarray, float32
            Mean image over time, shape (Ly, Lx).
        """
        n_frames = rec.shape[0]
        n_compute = (n_frames + stride - 1) // stride
        acc = np.zeros(rec.shape[1:], dtype=np.float64)
        count = 0
        chunk_starts = list(range(0, n_compute, chunk_size))
        n_chunks = len(chunk_starts)
        for _ci, c_pos in enumerate(chunk_starts):
            n_in_chunk = min(chunk_size, n_compute - c_pos)
            f_start = c_pos * stride
            f_end = f_start + n_in_chunk * stride
            print(f'\t\tchunk {_ci + 1}/{n_chunks}...      ', end='\r')
            block = np.asarray(rec[f_start:f_end:stride], dtype=np.float32)
            acc += block.sum(axis=0)
            count += block.shape[0]
        print('')
        return (acc / max(count, 1)).astype(np.float32)

    def _interp_metric_to_t(self, metric, t):
        """Interpolate a registration metric array to the QC time axis.

        Parameters
        ----------
        metric : np.ndarray or None
            1-D metric from reg (may have a different length than t).
        t : np.ndarray
            QC time vector.

        Returns
        -------
        np.ndarray or None
            Metric resampled to len(t), or None if metric is None.
        """
        if metric is None:
            return None
        if metric.size == t.size:
            return metric.astype(np.float32)
        _tr = np.linspace(t[0], t[-1], metric.size)
        return np.interp(t, _tr, metric).astype(np.float32)

    def _compute_event_avg(self, trace, t, event_times,
                           t_pre=2.0, t_post=2.0, pct=False):
        """Event-triggered average with pre-event baseline normalisation.

        Parameters
        ----------
        trace : np.ndarray
            1-D metric trace aligned to t.
        t : np.ndarray
            Time vector, shape (n_frames,).
        event_times : array-like
            Event onset times in seconds.
        t_pre : float
            Seconds before each event to include (also the baseline window).
        t_post : float
            Seconds after each event to include.

        Returns
        -------
        t_rel : np.ndarray
            Relative time axis, shape (n_win,).
        mean_trace : np.ndarray or None
            Trial-averaged, baseline-subtracted trace.
        sem_trace : np.ndarray or None
            Standard error of the mean across trials.
        """
        dt = float(np.median(np.diff(t)))
        n_pre = int(round(t_pre / dt))
        n_post = int(round(t_post / dt))
        n_win = n_pre + n_post + 1
        t_rel = np.linspace(-t_pre, t_post, n_win)

        snippets = []
        for ev_t in np.asarray(event_times).ravel():
            ind_ev = int(np.argmin(np.abs(t - ev_t)))
            i0 = ind_ev - n_pre
            i1 = ind_ev + n_post + 1
            if i0 < 0 or i1 > len(trace):
                continue
            snippet = trace[i0:i1].copy().astype(np.float64)
            # NaN-tolerant baseline: a per-trial dF/F trace (f0_mode=
            # 'per_trial') is NaN outside its trial window, so a snippet
            # whose alignment window overruns that window carries NaN in
            # the tail. Use nanmean for the baseline and drop the snippet
            # only when the baseline itself has no finite samples.
            if n_pre > 0:
                with np.errstate(invalid='ignore'):
                    baseline = np.nanmean(snippet[:n_pre])
                if not np.isfinite(baseline):
                    continue
            else:
                baseline = 0.0
            if pct:
                snippet = (snippet - baseline) / (
                    abs(baseline) + 1e-9) * 100.0
            else:
                snippet -= baseline
            snippets.append(snippet)

        if not snippets:
            return t_rel, None, None

        # nanmean/nanstd across trials so a NaN gap in one trial is filled
        # by the trials that do cover that relative time; positions no
        # trial covers stay NaN.
        arr = np.array(snippets)
        with np.errstate(invalid='ignore'):
            _n_fin = np.sum(np.isfinite(arr), axis=0)
            _mean = np.nanmean(arr, axis=0)
            _std = np.nanstd(arr, axis=0)
        _mean[_n_fin == 0] = np.nan
        _sem = _std / np.sqrt(np.maximum(_n_fin, 1))
        _sem[_n_fin == 0] = np.nan
        return t_rel, _mean, _sem

    def _save_qc_corrxy_eventavg(self, t_pre=2.0, t_post=2.0):
        """Save suite2p corrXY, stim- and reward-aligned, to data_mbl/.

        Companion to the corrXY panel in ``_plt_qc_reg_stats``: computes
        the same event-triggered average (baseline-subtracted % change,
        via ``_compute_event_avg`` with ``pct=True``) but persists it as
        a standalone .npy so it can be inspected without re-running
        ``plt_qc``'s full registration figure.

        Parameters
        ----------
        t_pre, t_post : float
            Seconds before/after each event; matches
            ``_plt_qc_reg_stats``'s defaults.
        """
        t = self.qc.t
        _trace = self._interp_metric_to_t(self.qc.reg.corrXY, t)
        _stim_t = self._qc_stim_t()
        _rew_t = self._qc_rew_t()

        _t_rel_stim, _mean_stim, _sem_stim = self._compute_event_avg(
            _trace, t, _stim_t, t_pre=t_pre, t_post=t_post, pct=True)
        _t_rel_rew, _mean_rew, _sem_rew = self._compute_event_avg(
            _trace, t, _rew_t, t_pre=t_pre, t_post=t_post, pct=True)

        _summary = {
            'corrXY': np.asarray(_trace),
            't': np.asarray(t),
            'stim_t': np.asarray(_stim_t),
            'rew_t': np.asarray(_rew_t),
            't_pre': float(t_pre),
            't_post': float(t_post),
            'stim_aligned': {
                't_rel': _t_rel_stim, 'mean': _mean_stim, 'sem': _sem_stim},
            'rew_aligned': {
                't_rel': _t_rel_rew, 'mean': _mean_rew, 'sem': _sem_rew},
        }

        _ch_suffix = (f'_ch={self.qc.channel}'
                      if self.qc.channel is not None else '')
        _npy_name = (f'{self.path.animal}_{self.path.date}_'
                     f'{self.path.beh_folder}_qc_corrxy_eventavg'
                     f'{_ch_suffix}.npy')
        np.save(os.path.join(str(self.folder.data), _npy_name),
                _summary, allow_pickle=True)

    def _inter_trial_mask(self, t, t_post_rew=2.0):
        """Boolean mask over t for inter-trial periods.

        An inter-trial period for trial i is defined as the interval
        from `t_post_rew` seconds after the last reward of trial i-1
        to the start of stim i. The period before the first stim
        (and after the last trial's final reward) is also included.

        Parameters
        ----------
        t : np.ndarray
            QC time vector.
        t_post_rew : float
            Seconds after the last reward of a trial to wait before
            counting the inter-trial period as having begun.

        Returns
        -------
        mask : np.ndarray of bool
            True where t falls within an inter-trial period.
        """
        mask = np.zeros_like(t, dtype=bool)
        stim_t = np.asarray(self.beh.stim.t_start).ravel()
        rew_t = np.asarray(self.beh.rew.t).ravel()
        if stim_t.size == 0:
            return mask

        # Pre-first-stim period
        _pre_rews = rew_t[rew_t < stim_t[0]]
        _pre_start = (_pre_rews.max() + t_post_rew
                      if _pre_rews.size > 0 else t[0])
        mask |= (t >= _pre_start) & (t < stim_t[0])

        # Between trials
        for _i in range(len(stim_t) - 1):
            _t_curr = stim_t[_i]
            _t_next = stim_t[_i + 1]
            _rews_in = rew_t[(rew_t >= _t_curr) & (rew_t < _t_next)]
            if _rews_in.size == 0:
                continue
            _t_iti_start = _rews_in.max() + t_post_rew
            if _t_iti_start >= _t_next:
                continue
            mask |= (t >= _t_iti_start) & (t < _t_next)

        # Post-last-trial period
        _rews_after = rew_t[rew_t >= stim_t[-1]]
        if _rews_after.size > 0:
            mask |= (t >= _rews_after.max() + t_post_rew)

        return mask

    def _linear_detrend_rows(self, M):
        """Subtract a per-row linear fit (a + b*i, i over time samples).

        Mathematically equivalent — for whole-frame or per-sector mean
        traces — to running signal_correction.detrend_linearly on the
        underlying (T, X, Y) rec and then spatial-averaging, by linearity
        of the mean. Operates on already-extracted traces so it is fast
        and does not require a second pass over the imaging data.

        Parameters
        ----------
        M : np.ndarray
            1-D trace (n_t,) or 2-D matrix (n_rows, n_t).

        Returns
        -------
        out : np.ndarray, float64
            Detrended trace / matrix with the per-row linear fit removed.
        """
        _M = np.asarray(M, dtype=np.float64)
        _was_1d = _M.ndim == 1
        if _was_1d:
            _M = _M[None, :]
        n_t = _M.shape[1]
        if n_t < 2:
            return _M[0] if _was_1d else _M
        _x = np.arange(n_t, dtype=np.float64)
        _xc = _x - _x.mean()
        _denom = float(np.dot(_xc, _xc))
        if _denom == 0:
            return _M[0] if _was_1d else _M
        with np.errstate(invalid='ignore'):
            _b = (_M @ _xc) / _denom
            _a = np.nanmean(_M, axis=1) - _b * _x.mean()
        _out = _M - (_a[:, None] + _b[:, None] * _x[None, :])
        return _out[0] if _was_1d else _out

    def _apply_detrend_sigs(self):
        """Linearly detrend self.qc.frame_f / sector_f raw red/grn traces.

        Detrending is applied only to raw 'red' and 'grn' bases (plus
        their split_lr halves); corrected '*_corr' channels are skipped
        because correct_signal already detrends them internally.
        """
        # Whole-frame traces
        # ----------
        if isinstance(self.qc.frame_f, dict):
            for _key in list(self.qc.frame_f.keys()):
                _base = _key.replace('_right', '').replace('_left', '')
                if _base not in ('red', 'grn'):
                    continue
                _tr = np.asarray(self.qc.frame_f[_key])
                self.qc.frame_f[_key] = self._linear_detrend_rows(_tr)

        # Per-sector matrices
        # ----------
        if isinstance(self.qc.sector_f, dict):
            for _key in list(self.qc.sector_f.keys()):
                if _key not in ('red', 'grn'):
                    continue
                _M = np.asarray(self.qc.sector_f[_key])
                _M = self._linear_detrend_rows(_M)
                self.qc.sector_f[_key] = _M.astype(np.float32)

    def _qc_flu_pair(self):
        """(real, static) channel roles for the current QC state.

        The real (functional) channel is the one the correction targets;
        the static channel is the control regressor. Read back from the
        resolved correct_signal_kwargs stored by add_qc, falling back to
        the QC's active channel (self.qc.channel) — so a QC run with
        channel='grn' reports ('grn', 'red') and every downstream panel
        treats green as functional.

        Returns
        -------
        real, static : str
            Functional and control channel labels ('red' / 'grn').
        """
        _kw = getattr(self.qc, 'correct_signal_kwargs', None) or {}
        _real = _kw.get('real_flu')
        _static = _kw.get('static_flu')
        _real_d, _static_d = _flu_pair_from_channel(
            getattr(self.qc, 'channel', None))
        if _real is None and _static is None:
            return _real_d, _static_d
        if _real is None:
            _real = 'grn' if _static == 'red' else 'red'
        elif _static is None:
            _static = 'grn' if _real == 'red' else 'red'
        return _real, _static

    def _qc_corrsig_suffix(self):
        """Filename fragment encoding correct_signal, detrend, and dff
        state for QC saves.

        Thin wrapper over the module-level ``corrsig_suffix_from_kwargs``
        (which readers of these files use to rebuild the same fragment);
        see it for the full format description.

        Returns
        -------
        suffix : str
        """
        return corrsig_suffix_from_kwargs(
            correct_signal_kwargs=getattr(
                self.qc, 'correct_signal_kwargs', None),
            correct_signal=getattr(self.qc, 'correct_signal', False),
            detrend_sigs=getattr(self.qc, 'detrend_sigs', False))

    # ------------------------------------
    # Plot
    # ------------------------------------

    def plt_qc(self,
               channel='red',
               n_sectors=8,
               plot_sectors=True,
               split_lr=False,
               correct_signal=True,
               correct_signal_kwargs=None,
               grab5ht_side=None,
               detrend_sigs=False,
               sector_kwargs=None,
               frame_f_kwargs=None,
               z_corr_kwargs=None,
               zstack_kwargs=None,
               heatmap_kwargs=None,
               dff_after_correction=False,
               dff_response_mean_fig=True,
               figsize=None,
               save_pdf=True,
               save_pkl=True,
               save_npy=True,
               save_tif=False,
               plt_show=True):
        """Large stacked QC figure for a two-photon recording.

        Conceptually-related tuning knobs are grouped into optional dicts
        (``*_kwargs``); pass None to take all defaults, or a partial dict
        to deviate. Unknown keys raise ValueError. See add_qc for the full
        per-key documentation of sector_kwargs / frame_f_kwargs /
        z_corr_kwargs / zstack_kwargs; only deviations from add_qc's
        meaning are noted here.

        Parameters
        ----------
        channel : str or None
            The functional channel: 'red' or 'grn' (dual-colour only);
            None uses self.rec. Besides picking the sector-heatmap /
            z_corr channel, this sets which channel the correction
            treats as the real signal and which as the static control —
            channel='grn' corrects green using red (real_flu='grn',
            static_flu='red'), and every downstream panel (correction
            steps, correlation panels, sector ordering, trial
            reliability, the Approach-A dF/F reconstruction and the
            saved .npy stats) follows that assignment. Pass real_flu /
            static_flu in correct_signal_kwargs to override. Default
            'red'.
        n_sectors : int
            Sector grid size (per axis). Default 8.
        plot_sectors : bool
            If True, include the sector fluorescence heatmap row and
            ensure sector_f has been computed. Default True.
        split_lr : bool
            If True, divide each frame into top half ('right') and
            bottom half ('left'); whole-frame F is plotted as separate
            traces per half (right solid, left dashed) and the sector
            heatmap is split into two stacked panels. Default False.
        correct_signal : bool
            If True (dual-colour only), call self.correct_signal with
            replace_real=False, compute whole-frame F on the corrected
            real channel, and add a subplot below the red/green ratio
            row showing the corrected real-channel trace. Default True.
        correct_signal_kwargs : dict or None
            Extra kwargs forwarded to self.correct_signal (replace_real
            is forced False). Default None, which resolves to
            ``{'method': 'full_regress', 'remove_stim_step': True}``
            when correct_signal=True (stim-step light-leak removal ON by
            default for the QC figure). The stim-step params are folded in
            here (remove_stim_step, stim_step_edge, stim_step_n_edge,
            stim_step_n_gap, stim_step_amp — see add_qc); the step removal
            is visualised as a dedicated panel in the qc_correct_signal
            figure. For the 1-D hemisphere-control path (split_lr=True,
            correct_signal=True, grab5ht_side set, single-channel rec):
            picks the method and any 1-D parameters (smooth_window,
            airpls_lam, airpls_porder, airpls_max_iter, trim_initial,
            nn_slope). For method='pixel_spatial_subtr' with
            f0_mode='per_trial', trial_onsets and trial_window are
            auto-filled from self.beh.stim.t_start and the trial-averaged
            window (see add_qc) unless supplied.
        grab5ht_side : str or None
            Hemisphere ('left' or 'right') carrying the real GRAB
            signal in a single-channel hemisphere-control run; the
            other side is treated as the mutant / regressor. When set
            alongside split_lr=True and correct_signal=True for a
            single-channel recording, runs the chosen correction
            method on the two 1-D hemisphere mean traces
            (s1 = mut, s2 = grab) and stores the corrected trace at
            ``frame_f['{channel}_corr_{grab5ht_side}']``. Default
            None (no 1-D correction; dual-colour pixel-wise correction
            is unaffected).
        detrend_sigs : bool
            If True, linearly detrend the raw red and green whole-frame
            and per-sector traces before any plotting or analysis (see
            add_qc for details). Default False.
        sector_kwargs : dict or None
            Sector fluorescence extraction — keys stride (4), chunk (500).
            See add_qc. None ⇒ all defaults.
        frame_f_kwargs : dict or None
            Whole-frame F extraction — keys compute (True), stride (4),
            chunk (500). See add_qc. None ⇒ all defaults.
        z_corr_kwargs : dict or None
            Z-corr computation — keys stride (4), chunk (200), jobs (4),
            ref_n_frames (None), dual_ref (True). dual_ref overlays the
            early/late z_corr traces in the z_corr panel and adds an
            extra (early − late) signed-drift panel. See add_qc. None ⇒
            all defaults.
        zstack_kwargs : dict or None
            Z-position inference — keys path (None ⇒ disabled), channel
            (None). When path is set, replaces the z_corr panel with the
            inferred z-position trace. See add_qc. None ⇒ disabled.
        heatmap_kwargs : dict or None
            Sector-heatmap colour limits (display-only; not forwarded to
            add_qc). Keys (defaults): vmin (-2), vmax (4) — z-score
            colour limits. None ⇒ all defaults.
        dff_after_correction : bool
            If True (full_regress correction only), the corrected
            real-channel column of the stim-aligned response-mean figure
            is shown as per-trial dF/F0 (%) instead of ITI z-score. The
            z-scored zdFF output of full_regress carries no
            fluorescence scale, so a fluorescence-units corrected trace
            is first reconstructed (Approach A): the per-region 1-D
            Martianova pipeline is re-run on the stored raw control
            (grn) and signal (red) whole-frame / per-sector traces, and
            the z-score is mapped back to fluorescence via
            F_corr = sigma2 * zdFF + m2 + b2(t), where (sigma2, m2, b2)
            are that region's signal-channel normalisation constants.
            Each per-trial snippet is then converted to dF/F0 (%) using
            its pre-stim baseline as F0, exactly as the raw red/grn
            columns are under dff_response_mean_fig. Because the 1-D
            pipeline is fit per region, the reconstruction uses a
            per-region beta rather than the pixel-wise correction's
            shared beta. Default False.
        figsize : tuple or None
            Figure size in inches. None auto-sizes based on plot_sectors.
        save_pdf : bool
            If True (default), save every QC figure as a .pdf to
            self.folder.figs (figs_mbl/).
        save_pkl : bool
            If True (and save_pdf=True), also write a gzipped pickle of
            each matplotlib Figure to .pkl.gz in self.folder.data
            (data_mbl/) for later interactive inspection via
            load_qc_fig. Default True.
        save_npy : bool
            If True (default), write the stim-aligned response-mean
            summary as a .npy dict to self.folder.data (data_mbl/),
            consumed by the grand-average / cohort pipelines in
            batch_run. Was previously ``save_response_mean``.
        save_tif : bool
            If True, forward to add_qc / self.correct_signal so the
            correction writes the trial-averaged stim-aligned TIFFs
            `{base}_corr_trialavg.tif` (corrected) and
            `{base}_dff_trialavg.tif` (raw real/static dF/F) into the
            recording's data_mbl/ folder (self.folder.data; see
            correct_signal docstring for the window/marker details). The
            full corrected stack is streamed to a scratch
            `{base}_corr.tif` (in folder.img) only transiently — add_qc
            deletes it once the trial-averaged TIFFs and QC aggregates
            are derived, so only the small trial-averaged TIFFs persist.
            Was previously ``save_to_disk``. Default False (nothing
            written to disk; corrected stack stays in RAM for QC).
        plt_show : bool
            If True, calls plt.show(); otherwise closes the figure.
        """

        # Seed the default correction kwargs (stim-step removal ON by
        # default for the QC figure) and resolve the grouped tuning dicts
        # against their defaults. heatmap_kwargs is display-only; the rest
        # are forwarded to add_qc. Build the canonical compute-signature
        # so we recompute only when a result-affecting param changed
        # (chunk sizes / thread counts do not appear in the signature).
        # ----------
        if correct_signal and correct_signal_kwargs is None:
            correct_signal_kwargs = {'method': 'full_regress',
                                     'remove_stim_step': True}
        sector_kwargs = _merge_qc_kwargs('sector', sector_kwargs)
        frame_f_kwargs = _merge_qc_kwargs('frame_f', frame_f_kwargs)
        z_corr_kwargs = _merge_qc_kwargs('z_corr', z_corr_kwargs)
        zstack_kwargs = _merge_qc_kwargs('zstack', zstack_kwargs)
        heatmap_kwargs = _merge_qc_kwargs('heatmap', heatmap_kwargs)
        _z_vmin = heatmap_kwargs['vmin']
        _z_vmax = heatmap_kwargs['vmax']
        _zc_dual = z_corr_kwargs['dual_ref']
        _cs_resolved = _resolve_cs_kwargs(correct_signal_kwargs, channel)
        _sig = _qc_compute_signature(
            n_sectors=n_sectors, channel=channel,
            compute_sectors=plot_sectors,
            sector_kwargs=sector_kwargs, frame_f_kwargs=frame_f_kwargs,
            z_corr_kwargs=z_corr_kwargs, zstack_kwargs=zstack_kwargs,
            split_lr=split_lr, correct_signal=correct_signal,
            correct_signal_kwargs=_cs_resolved, grab5ht_side=grab5ht_side,
            detrend_sigs=detrend_sigs, save_tif=save_tif)

        _prev_sig = getattr(self.qc, '_compute_sig', None) \
            if hasattr(self, 'qc') else None
        _needs_sectors = plot_sectors and (
            not hasattr(self, 'qc') or self.qc.sector_f is None
            or _prev_sig is None
            or _prev_sig.get('sector_stride') != _sig['sector_stride']
            or _prev_sig.get('n_sectors') != _sig['n_sectors'])
        _needs_recompute = (
            _needs_sectors
            or not hasattr(self, 'qc')
            or _prev_sig != _sig)
        if _needs_recompute:
            self.add_qc(channel=channel, n_sectors=n_sectors,
                        compute_sectors=plot_sectors,
                        split_lr=split_lr,
                        sector_kwargs=sector_kwargs,
                        frame_f_kwargs=frame_f_kwargs,
                        z_corr_kwargs=z_corr_kwargs,
                        zstack_kwargs=zstack_kwargs,
                        correct_signal=correct_signal,
                        correct_signal_kwargs=correct_signal_kwargs,
                        grab5ht_side=grab5ht_side,
                        detrend_sigs=detrend_sigs,
                        save_tif=save_tif)

        # Plot-only flag: the corrected real-channel column of the
        # stim-aligned response-mean figure is rendered as per-trial
        # dF/F0 (%) rather than ITI z-score (see _plt_qc_response_mean).
        # Stored on self.qc so it survives without forcing a recompute.
        # ----------
        self.qc.dff_after_correction = bool(dff_after_correction)

        t = self.qc.t
        reg = self.qc.reg

        # Detect dual-channel state (after add_qc has run).
        _bases_present = set()
        for _l in self.qc.frame_f:
            _bases_present.add(
                _l.replace('_right', '').replace('_left', ''))
        _has_dual_ff = ('red' in _bases_present
                        and 'grn' in _bases_present)
        _dual_sectors = isinstance(self.qc.sector_f, dict)
        _corr_bases = sorted(b for b in _bases_present
                             if b.endswith('_corr'))
        _has_corr_ff = len(_corr_bases) > 0

        # ====================================================================
        # FIGURE — main stacked QC overview
        # ====================================================================
        # All panels below share one time axis: behavioural events,
        # licks, whole-frame & per-sector fluorescence, registration.

        # --- layout: auto-size figure height ---
        if figsize is None:
            _fig_h = 7.0 if plot_sectors else 4.5
            if _zc_dual:
                _fig_h += 0.4
            if split_lr:
                _fig_h += 0.4
            if _has_dual_ff:
                # second whole-frame F subplot (red + green split)
                _fig_h += 0.4
            if plot_sectors and _dual_sectors:
                _fig_h += 3.0
            if _has_corr_ff:
                _fig_h += 0.4
            figsize = (8.5, _fig_h)

        # Defer all figure rendering until after every panel is built.
        # In interactive backends (IPython/Jupyter) plt.figure() can pop
        # up a window immediately, which causes the helper figures to
        # appear one-at-a-time as earlier ones are dismissed.
        _was_interactive = plt.isinteractive()
        plt.ioff()

        fig = plt.figure(figsize=figsize)
        ax_zdiff = None
        ax_frame_diff = None
        ax_corr_sig = None
        ax_sectors = {}

        # --- layout: dynamic top-to-bottom row grid (rows added
        #     conditionally on plot_sectors / split_lr / z_corr_dual_ref /
        #     dual-channel) ---
        _sec_h = 1.5 if split_lr else 0.9
        _rows = [('events', 0.12), ('licks', 0.12)]
        if _has_dual_ff:
            # Red and green whole-frame F in separate subplots, each a
            # little shorter than the single-channel frame row.
            _rows.append(('frame_red', 0.28))
            _rows.append(('frame_grn', 0.28))
        else:
            _rows.append(('frame', 0.35))
        if split_lr:
            _rows.append(('frame_diff', 0.15))
        if _has_corr_ff:
            _rows.append(('corr_sig', 0.25))
        # Corrected per-sector channel present? (drives the optional
        # corrected-channel sector summary row below).
        _has_corr_sec = (_dual_sectors and any(
            _k.endswith('_corr') for _k in self.qc.sector_f))
        if plot_sectors:
            if _dual_sectors:
                for _ch in ('red', 'grn'):
                    if _ch in self.qc.sector_f:
                        _rows.append((f'sectors_{_ch}', _sec_h))
                if _has_corr_sec:
                    _rows.append(('sectors_corr', _sec_h))
            else:
                _rows.append(('sectors', _sec_h))
        _rows += [('shift', 0.225), ('corr', 0.18)]
        _has_zum = (getattr(self.qc, 'reg', None) is not None
                    and getattr(self.qc.reg, 'z_um', None) is not None)
        if _has_zum:
            # zstack-inferred z-position replaces the z_corr panel.
            _rows.append(('zum', 0.25))
        else:
            if not _zc_dual:
                _rows.append(('zcorr', 0.18))
            if _zc_dual:
                _rows.append(('zdiff', 0.18))

        _hr = [h for _, h in _rows]
        _idx = {name: i for i, (name, _) in enumerate(_rows)}
        spec = gridspec.GridSpec(
            nrows=len(_hr), ncols=1, figure=fig,
            height_ratios=_hr, hspace=0.25)

        ax_events = fig.add_subplot(spec[_idx['events'], 0])
        ax_licks = fig.add_subplot(spec[_idx['licks'], 0],
                                   sharex=ax_events)
        ax_frame = ax_frame_red = ax_frame_grn = None
        if 'frame' in _idx:
            ax_frame = fig.add_subplot(spec[_idx['frame'], 0],
                                       sharex=ax_events)
        if 'frame_red' in _idx:
            ax_frame_red = fig.add_subplot(spec[_idx['frame_red'], 0],
                                           sharex=ax_events)
        if 'frame_grn' in _idx:
            ax_frame_grn = fig.add_subplot(spec[_idx['frame_grn'], 0],
                                           sharex=ax_events)
        if 'frame_diff' in _idx:
            ax_frame_diff = fig.add_subplot(spec[_idx['frame_diff'], 0],
                                            sharex=ax_events)
        if 'corr_sig' in _idx:
            ax_corr_sig = fig.add_subplot(spec[_idx['corr_sig'], 0],
                                          sharex=ax_events)
        for _key in ('sectors', 'sectors_red', 'sectors_grn',
                     'sectors_corr'):
            if _key not in _idx:
                continue
            if split_lr:
                _sec_spec = spec[_idx[_key], 0].subgridspec(
                    2, 1, hspace=0.05)
                _ax_sec_r = fig.add_subplot(_sec_spec[0, 0],
                                            sharex=ax_events)
                _ax_sec_l = fig.add_subplot(_sec_spec[1, 0],
                                            sharex=ax_events)
                ax_sectors[_key] = (_ax_sec_r, _ax_sec_l)
            else:
                ax_sectors[_key] = fig.add_subplot(
                    spec[_idx[_key], 0], sharex=ax_events)
        ax_shift = fig.add_subplot(spec[_idx['shift'], 0],
                                   sharex=ax_events)
        ax_corr = fig.add_subplot(spec[_idx['corr'], 0],
                                  sharex=ax_events)
        ax_zcorr = None
        ax_zum = None
        if 'zcorr' in _idx:
            ax_zcorr = fig.add_subplot(spec[_idx['zcorr'], 0],
                                       sharex=ax_events)
        if 'zdiff' in _idx:
            ax_zdiff = fig.add_subplot(spec[_idx['zdiff'], 0],
                                       sharex=ax_events)
        if 'zum' in _idx:
            ax_zum = fig.add_subplot(spec[_idx['zum'], 0],
                                     sharex=ax_events)

        # --- subplot · events: stim + reward onsets ---
        # Stim-line transparency encodes per-trial expected reward
        # (calc_alpha); reused everywhere via _qc_stim_event_alphas so the
        # per-trial-type panels share this nomenclature exactly.
        _stim_t, _stim_alphas = self._qc_stim_event_alphas(alpha_min=0.1)
        _rew_t = self._qc_rew_t()
        if _stim_t.size > 0:
            for _i, (_t, _alpha) in enumerate(zip(_stim_t, _stim_alphas)):
                ax_events.vlines(_t, ymin=0, ymax=1,
                                 color=sns.xkcd_rgb['dark grey'],
                                 linewidth=1.5, alpha=_alpha,
                                 label='stim' if _i == 0 else None)
        if _rew_t.size > 0:
            ax_events.vlines(_rew_t, ymin=0, ymax=1,
                             color=sns.xkcd_rgb['bright blue'],
                             linewidth=1.5, alpha=0.9, label='rew')
        ax_events.set_ylim(0, 1)
        ax_events.set_yticks([])
        ax_events.set_ylabel('stim /\nrew', fontsize=8)
        ax_events.legend(loc='upper right', fontsize=7, ncol=2,
                         frameon=False)

        # --- subplot · licks: lick times as a point process ---
        _lick_t = self._qc_lick_t(getattr(self.beh.lick, 't_raw', []))
        if _lick_t.size > 0:
            ax_licks.vlines(_lick_t, ymin=0, ymax=1,
                            color='k', linewidth=0.5, alpha=0.7)
        ax_licks.set_ylim(0, 1)
        ax_licks.set_yticks([])
        ax_licks.set_ylabel('licks', fontsize=8)

        # --- subplot · whole-frame F: red & green (dual-colour split
        #     into ax_frame_red / ax_frame_grn; single-channel uses the
        #     single ax_frame) ---
        _ch_color = {'grn': sns.xkcd_rgb['forest green'],
                     'red': sns.xkcd_rgb['brick'],
                     'frame': sns.xkcd_rgb['dark grey']}
        _frame_ax_for = {'red': ax_frame_red, 'grn': ax_frame_grn}
        _ff_plotted = {}
        for _label, _trace in self.qc.frame_f.items():
            _base = _label.replace('_right', '').replace('_left', '')
            # Corrected channels live in the dedicated ax_corr_sig row.
            if _base.endswith('_corr'):
                continue
            _ax = _frame_ax_for.get(_base) or ax_frame
            if _ax is None:
                continue
            _color = _ch_color.get(_base, 'k')
            _ls = '--' if _label.endswith('_left') else '-'
            _ax.plot(t, _trace, color=_color, linewidth=0.7,
                     linestyle=_ls, label=_label)
            _ff_plotted[id(_ax)] = _ff_plotted.get(id(_ax), 0) + 1
        # Label reflects the post-processing applied in add_qc.
        if detrend_sigs:
            _frame_unit = 'F (detr.)'
        else:
            _frame_unit = 'F'
        if ax_frame is not None:
            ax_frame.set_ylabel(f'whole-frame\n{_frame_unit}', fontsize=8)
        for _ch, _ax in (('red', ax_frame_red), ('grn', ax_frame_grn)):
            if _ax is not None:
                _ax.set_ylabel(f'whole-frame\n{_ch} {_frame_unit}',
                               fontsize=8)
        # Legend only when >1 trace landed on an axis (e.g. split_lr
        # right/left, or both channels sharing the single ax_frame).
        for _ax in (ax_frame, ax_frame_red, ax_frame_grn):
            if _ax is not None and _ff_plotted.get(id(_ax), 0) > 1:
                _ax.legend(loc='upper right', fontsize=7, frameon=False)

        # --- subplot · frame L−R: per-frame left − right whole-frame F
        #     (split_lr only) ---
        if ax_frame_diff is not None:
            _bases = []
            for _label in self.qc.frame_f:
                if not _label.endswith('_left'):
                    continue
                _b = _label[:-len('_left')]
                if f'{_b}_right' in self.qc.frame_f and _b not in _bases:
                    _bases.append(_b)
            for _base in _bases:
                _l = self.qc.frame_f[f'{_base}_left']
                _r = self.qc.frame_f[f'{_base}_right']
                _color = _ch_color.get(_base, 'k')
                ax_frame_diff.plot(t, _l - _r, color=_color,
                                   linewidth=0.7, label=_base)
            ax_frame_diff.axhline(y=0, color='k', linewidth=0.4,
                                  linestyle=':')
            ax_frame_diff.set_ylabel('left − right\nF', fontsize=8)
            if len(_bases) > 1:
                ax_frame_diff.legend(loc='upper right', fontsize=7,
                                     frameon=False)

        # --- subplot · corrected F: corrected real-channel whole-frame F
        #     (dual-colour + correct_signal=True) ---
        if ax_corr_sig is not None:
            for _corr_base in _corr_bases:
                _corr_color = sns.xkcd_rgb.get(
                    'bright orange', sns.xkcd_rgb['orange'])
                if split_lr:
                    for _suffix, _ls in [('_right', '-'), ('_left', '--')]:
                        _cor = self.qc.frame_f.get(
                            f'{_corr_base}{_suffix}')
                        if _cor is not None:
                            ax_corr_sig.plot(
                                t, _cor, color=_corr_color,
                                linewidth=0.7, linestyle=_ls,
                                label=(f'{_corr_base} '
                                       f'({_suffix[1:]})'))
                else:
                    _cor = self.qc.frame_f.get(_corr_base)
                    if _cor is not None:
                        ax_corr_sig.plot(t, _cor, color=_corr_color,
                                         linewidth=0.7,
                                         label=_corr_base)
            ax_corr_sig.axhline(y=0, color='k', linewidth=0.4,
                                linestyle=':')
            if len(_corr_bases) > 1 or split_lr:
                ax_corr_sig.legend(loc='upper right', fontsize=7,
                                   ncol=2, frameon=False)
            ax_corr_sig.set_ylabel('corrected\nF', fontsize=8)

        # --- subplot · sector F heatmap: per-row z-scored sector traces.
        #     Shared colour scale (z_vmin / z_vmax) across red / grn / corr
        #     so magnitudes are directly comparable ---
        _secs_to_plot = []
        if isinstance(self.qc.sector_f, dict):
            for _ch in ('red', 'grn'):
                if _ch in self.qc.sector_f:
                    _secs_to_plot.append((f'sectors_{_ch}',
                                          self.qc.sector_f[_ch], _ch))
        elif self.qc.sector_f is not None:
            _secs_to_plot.append(('sectors', self.qc.sector_f, channel))

        for _key, _sec, _ch_for_cmap in _secs_to_plot:
            if _key not in ax_sectors:
                continue
            # Z-score each sector trace per row.
            _mean = np.mean(_sec, axis=1, keepdims=True)
            _std = np.std(_sec, axis=1, keepdims=True)
            _z = (_sec - _mean) / (_std + 1e-9)
            _sec_vmin, _sec_vmax = _z_vmin, _z_vmax
            _sec_cbar = 'z-score'
            _cmap = self._channel_img_cmap(_ch_for_cmap)
            _ax_entry = ax_sectors[_key]
            _ch_lbl = _ch_for_cmap if _ch_for_cmap is not None else ''
            if isinstance(_ax_entry, tuple):
                # split_lr: top rows of sector grid → 'right',
                # bottom rows → 'left'. Shared cmap/vmin/vmax across
                # both panels; one inset colorbar spans both.
                _n_half = n_sectors // 2
                _n_top = _n_half * n_sectors
                _ax_r, _ax_l = _ax_entry
                _im = None
                for _ax, _z_part, _lbl in [
                        (_ax_r, _z[:_n_top, :], 'right'),
                        (_ax_l, _z[_n_top:, :], 'left')]:
                    _nrows = _z_part.shape[0]
                    _im = _ax.imshow(_z_part, aspect='auto',
                                     extent=[t[0], t[-1], _nrows, 0],
                                     cmap=_cmap,
                                     vmin=_sec_vmin, vmax=_sec_vmax,
                                     interpolation='none')
                    _yl = (f'sector\n{_ch_lbl} {_lbl}\n(1..{_nrows})'
                           if _ch_lbl
                           else f'sector\n{_lbl}\n(1..{_nrows})')
                    _ax.set_ylabel(_yl, fontsize=8)
                    _ax.set_yticks([0, _nrows])
                # Inset colorbar shared by both halves. Y-coords are in
                # _ax_r's axes-fraction; subgridspec hspace=0.05 ⇒ a
                # height of ~2.05 spans _ax_r + gap + _ax_l.
                _cax = _ax_r.inset_axes([1.015, -1.05, 0.012, 2.05])
                fig.colorbar(_im, cax=_cax)
                _cax.tick_params(labelsize=7)
                _cax.set_ylabel(_sec_cbar, fontsize=7)
            else:
                _ax = _ax_entry
                _im = _ax.imshow(
                    _z, aspect='auto',
                    extent=[t[0], t[-1], n_sectors * n_sectors, 0],
                    cmap=_cmap, vmin=_sec_vmin, vmax=_sec_vmax,
                    interpolation='none')
                _yl = (f'sector\n{_ch_lbl}\n(1..{n_sectors**2})'
                       if _ch_lbl
                       else f'sector\n(1..{n_sectors**2})')
                _ax.set_ylabel(_yl, fontsize=8)
                _ax.set_yticks([0, n_sectors**2])
                _cax = _ax.inset_axes([1.015, 0, 0.012, 1])
                fig.colorbar(_im, cax=_cax)
                _cax.tick_params(labelsize=7)
                _cax.set_ylabel(_sec_cbar, fontsize=7)

        # --- subplot · corrected sector heatmap: per-sector corrected
        #     real channel, per-row z-scored on the same scale as the raw
        #     sector heatmaps (dual-colour + correct_signal=True only) ---
        if 'sectors_corr' in ax_sectors \
                and isinstance(self.qc.sector_f, dict):
            _corr_sec_key = next(
                (k for k in self.qc.sector_f if k.endswith('_corr')),
                None)
            _sec = self.qc.sector_f[_corr_sec_key].astype(np.float32)
            _corr_native_dff = getattr(
                self.qc, 'corr_native_dff', None) is not None
            if _corr_native_dff:
                # Native subtractive dF/F (e.g. pixel_spatial_subtr): show
                # the actual dF/F (%) rather than z-scoring it away. NaN
                # inter-trial gaps (f0_mode='per_trial') are preserved. Own
                # symmetric data-driven scale, since the raw red/grn
                # heatmaps stay in z-score units.
                _data = _sec * 100.0
                with np.errstate(invalid='ignore'):
                    _finite = _data[np.isfinite(_data)]
                _vmag = (float(np.percentile(np.abs(_finite), 98))
                         if _finite.size else 1.0)
                if not (np.isfinite(_vmag) and _vmag > 0):
                    _vmag = 1.0
                _vlo, _vhi = -_vmag, _vmag
                _cbar_lab = f'{_corr_sec_key}\ndF/F (%)'
            else:
                # Per-row z-score, displayed on the same z_vmin/z_vmax
                # scale as the raw red/grn sector heatmaps.
                _mean = np.mean(_sec, axis=1, keepdims=True)
                _std = np.std(_sec, axis=1, keepdims=True)
                _data = (_sec - _mean) / (_std + 1e-9)
                _vlo, _vhi = _z_vmin, _z_vmax
                _cbar_lab = f'{_corr_sec_key}\n(z)'
            _ylab_base = _corr_sec_key
            _ax_entry = ax_sectors['sectors_corr']
            if isinstance(_ax_entry, tuple):
                _n_half = n_sectors // 2
                _n_top = _n_half * n_sectors
                _ax_r, _ax_l = _ax_entry
                _im = None
                for _ax, _z_part, _lbl in [
                        (_ax_r, _data[:_n_top, :], 'right'),
                        (_ax_l, _data[_n_top:, :], 'left')]:
                    _nrows = _z_part.shape[0]
                    _im = _ax.imshow(_z_part, aspect='auto',
                                     extent=[t[0], t[-1], _nrows, 0],
                                     cmap='gray',
                                     vmin=_vlo, vmax=_vhi,
                                     interpolation='none')
                    _ax.set_ylabel(
                        f'sector\n{_ylab_base} {_lbl}\n(1..{_nrows})',
                        fontsize=8)
                    _ax.set_yticks([0, _nrows])
                _cax = _ax_r.inset_axes([1.015, -1.05, 0.012, 2.05])
                fig.colorbar(_im, cax=_cax)
                _cax.tick_params(labelsize=7)
                _cax.set_ylabel(_cbar_lab, fontsize=7)
            else:
                _ax = _ax_entry
                _im = _ax.imshow(
                    _data, aspect='auto',
                    extent=[t[0], t[-1], n_sectors * n_sectors, 0],
                    cmap='gray', vmin=_vlo, vmax=_vhi,
                    interpolation='none')
                _ax.set_ylabel(
                    f'sector\n{_ylab_base}\n(1..{n_sectors**2})',
                    fontsize=8)
                _ax.set_yticks([0, n_sectors**2])
                _cax = _ax.inset_axes([1.015, 0, 0.012, 1])
                fig.colorbar(_im, cax=_cax)
                _cax.tick_params(labelsize=7)
                _cax.set_ylabel(_cbar_lab, fontsize=7)

        # --- subplot · shift: registration shift magnitude (px) ---
        if reg.shift_mag is not None and reg.shift_mag.size == t.size:
            ax_shift.plot(t, reg.shift_mag, color='k', linewidth=0.6)
        elif reg.shift_mag is not None:
            _tr = np.linspace(t[0], t[-1], reg.shift_mag.size)
            ax_shift.plot(_tr, reg.shift_mag, color='k', linewidth=0.6)
        else:
            ax_shift.text(0.5, 0.5, 'no xoff/yoff in ops.npy',
                          ha='center', va='center',
                          transform=ax_shift.transAxes, fontsize=8,
                          color='grey')
        ax_shift.set_ylabel('shift\n(px)', fontsize=8)

        # --- subplot · corrXY: Suite2P registration phase-correlation ---
        if reg.corrXY is not None:
            _x = t if reg.corrXY.size == t.size \
                else np.linspace(t[0], t[-1], reg.corrXY.size)
            _cmax = float(np.nanmax(reg.corrXY))
            _corr_norm = (reg.corrXY / _cmax
                          if _cmax > 0 else reg.corrXY)
            ax_corr.plot(_x, _corr_norm, color='k', linewidth=0.6)
            ax_corr.set_ylabel('corrXY\n(norm)', fontsize=8)
        else:
            ax_corr.text(0.5, 0.5, 'no corrXY in ops.npy',
                         ha='center', va='center',
                         transform=ax_corr.transAxes, fontsize=8,
                         color='grey')
            ax_corr.set_ylabel('corrXY', fontsize=8)

        # --- subplot · z_corr: zero-lag Pearson vs. meanImg (or early /
        #     late refs when z_corr_dual_ref=True) ---
        if ax_zcorr is not None:
            if reg.z_corr is not None:
                ax_zcorr.plot(t, reg.z_corr,
                              color=sns.xkcd_rgb['dark teal'],
                              linewidth=0.6)
                ax_zcorr.set_ylabel('z_corr', fontsize=8)
            else:
                ax_zcorr.text(0.5, 0.5, 'no meanImg in ops.npy',
                              ha='center', va='center',
                              transform=ax_zcorr.transAxes, fontsize=8,
                              color='grey')
                ax_zcorr.set_ylabel('z_corr', fontsize=8)

        # --- subplot · z_corr drift: signed early − late drift indicator ---
        if ax_zdiff is not None:
            if reg.z_corr_diff is not None:
                _dmax = float(np.nanmax(np.abs(reg.z_corr_diff)))
                _zd_norm = (reg.z_corr_diff / _dmax
                            if _dmax > 0 else reg.z_corr_diff)
                ax_zdiff.plot(t, _zd_norm,
                              color=sns.xkcd_rgb['plum'], linewidth=0.6)
                ax_zdiff.axhline(y=0, color='k', linewidth=0.4,
                                 linestyle=':')
                ax_zdiff.set_ylim(-1.05, 1.05)
                ax_zdiff.set_yticks([-1, 0, 1])
                ax_zdiff.set_yticklabels(['late', '0', 'early'],
                                         fontsize=7)
            else:
                ax_zdiff.text(0.5, 0.5, 'no dual-ref z_corr',
                              ha='center', va='center',
                              transform=ax_zdiff.transAxes, fontsize=8,
                              color='grey')
            ax_zdiff.set_ylabel('z_corr diff\n(early − late)',
                                fontsize=8)

        # --- subplot · z-position: inferred from a Bruker z-stack match
        #     (zstack_kwargs['path']); replaces the z_corr panel ---
        if ax_zum is not None:
            z_um = reg.z_um
            _x = t if z_um.size == t.size \
                else np.linspace(t[0], t[-1], z_um.size)
            ax_zum.plot(_x, z_um,
                        color=sns.xkcd_rgb['dark teal'], linewidth=0.6)
            _slices = getattr(reg, 'z_um_slices', None)
            if _slices is not None and len(_slices) >= 2:
                _zmid = float(np.median(z_um[np.isfinite(z_um)])) \
                    if np.any(np.isfinite(z_um)) else None
                ax_zum.axhline(y=float(_slices[0]),
                               color='k', linewidth=0.4, linestyle=':',
                               alpha=0.5)
                ax_zum.axhline(y=float(_slices[-1]),
                               color='k', linewidth=0.4, linestyle=':',
                               alpha=0.5)
                if _zmid is not None:
                    ax_zum.axhline(y=_zmid,
                                   color=sns.xkcd_rgb['bright red'],
                                   linewidth=0.4, linestyle='--',
                                   alpha=0.6)
            _ch_tag = getattr(reg, 'zstack_channel', None)
            _ylab = (f'z (µm)\n[zstack {_ch_tag}]'
                     if _ch_tag is not None else 'z (µm)')
            ax_zum.set_ylabel(_ylab, fontsize=8)

        # --- finalize fig 1: spine/tick cosmetics + shared x-axis ---
        _all_ax = []
        # Top-to-bottom: events, licks, frame(s), frame_diff, corr_sig,
        # sectors (in row order), shift, corr, zcorr, zdiff.
        _ordered = [ax_events, ax_licks, ax_frame, ax_frame_red,
                    ax_frame_grn, ax_frame_diff, ax_corr_sig]
        for _key in ('sectors', 'sectors_red', 'sectors_grn',
                     'sectors_corr'):
            if _key in ax_sectors:
                _ordered.append(ax_sectors[_key])
        _ordered += [ax_shift, ax_corr, ax_zcorr, ax_zdiff, ax_zum]
        for _entry in _ordered:
            if _entry is None:
                continue
            if isinstance(_entry, tuple):
                _all_ax.extend(_entry)
            else:
                _all_ax.append(_entry)
        for _ax in _all_ax:
            for _side in ('top', 'right'):
                _ax.spines[_side].set_visible(False)
        for _ax in _all_ax[:-1]:
            plt.setp(_ax.get_xticklabels(), visible=False)
        _all_ax[-1].set_xlabel('time (s)')

        ax_events.set_xlim(t[0], t[-1])

        _title = f'{self.path.animal} {self.path.date} ' \
            f'{self.path.beh_folder} — QC'
        _cs_kw = getattr(self.qc, 'correct_signal_kwargs', None) or {}
        _cs_method = _cs_kw.get('method', None)
        if _cs_method == 'full_regress':
            _fm = _cs_kw.get('fit_mode', 'global')
            _title = _title + f'  [corr: {_cs_method}, fit={_fm}]'
        elif _cs_method is not None:
            _title = _title + f'  [corr: {_cs_method}]'
        fig.suptitle(_title, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        if save_pdf:
            _ch_suffix = f'_ch={channel}' if channel is not None else ''
            _cs_suffix = self._qc_corrsig_suffix()
            _fname = (f'{self.path.animal}_{self.path.date}_'
                      f'{self.path.beh_folder}'
                      f'_qc_n_sectors={n_sectors}'
                      f'{_ch_suffix}{_cs_suffix}.pdf')
            fig.savefig(os.path.join(str(self.folder.figs), _fname))
            if save_pkl:
                _pkl_name = _fname[:-4] + '.pkl.gz'
                _pkl_path = os.path.join(str(self.folder.data),
                                         _pkl_name)
                with gzip.open(_pkl_path, 'wb') as _pf:
                    pickle.dump(fig, _pf,
                                protocol=pickle.HIGHEST_PROTOCOL)

        _figs = [fig]

        # ====================================================================
        # FIGURE — mean red / green image overlay (dual-colour only)
        # ====================================================================
        _overlay_fig = self._plt_qc_mean_overlay(channel=channel,
                                                 save=save_pdf)
        if _overlay_fig is not None:
            _figs.append(_overlay_fig)

        # ====================================================================
        # FIGURE — full_regress correction steps (dual-colour + correct_signal)
        # ====================================================================
        _corrsig_fig = self._plt_qc_correct_signal(
            channel=channel, save=save_pdf, save_pickle=save_pkl)
        if _corrsig_fig is not None:
            _figs.append(_corrsig_fig)

        # ====================================================================
        # FIGURE — registration statistics
        # ====================================================================
        _figs.append(self._plt_qc_reg_stats(t, reg, channel=channel,
                                            save=save_pdf))

        # ====================================================================
        # FIGURE — event-triggered (stim-aligned) response means
        # ====================================================================
        _response_fig = self._plt_qc_response_mean(
            t, channel=channel, save=save_pdf,
            dff_sig=dff_response_mean_fig,
            save_npy=save_npy)
        if _response_fig is not None:
            _figs.append(_response_fig)

        # ====================================================================
        # FIGURE — event-triggered per-cell donut-ring responses
        # (pixel_spatial_subtr, segmentation_type='cells' only)
        # ====================================================================
        _cells_fig = self._plt_qc_response_mean_cells(
            t, channel=channel, save=save_pdf)
        if _cells_fig is not None:
            _figs.append(_cells_fig)

        # ====================================================================
        # FIGURE — lick-bout-onset-aligned response means
        # ====================================================================
        _lick_fig = self._plt_qc_lick_aligned(
            t, channel=channel, save=save_pdf, save_pickle=save_pkl,
            dff_sig=dff_response_mean_fig)
        if _lick_fig is not None:
            _figs.append(_lick_fig)

        if _was_interactive:
            plt.ion()
        if plt_show:
            plt.show()
        else:
            # Drop each figure's artists explicitly before closing —
            # matplotlib keeps circular references between Figure /
            # Axes / Artist that can delay GC of large heatmap arrays.
            # clf() detaches all axes; close() removes the manager.
            for _f in _figs:
                _f.clf()
                plt.close(_f)
            del _figs
            gc.collect()
        return

    def _plt_qc_mean_overlay(self, channel=None, save=True,
                             stride=4, chunk_size=500, figsize=None,
                             lo_pct=1.0, hi_pct=99.5):
        """Mean red and green images, shown individually and overlaid.

        Dual-colour only. Computes the temporal mean of each channel
        (chunked, memory-safe) and renders three panels: mean red (red
        colormap), mean green (green colormap), and an RGB overlay where
        co-localised signal appears yellow. Returns None for
        single-channel recordings.

        Parameters
        ----------
        channel : str or None
            Channel label for the figure filename suffix.
        save : bool
            If True, saves a pdf to self.folder.figs.
        stride : int
            Temporal subsampling for the mean image. Default 4.
        chunk_size : int
            Frames per chunked read. Default 500.
        figsize : tuple or None
            Figure size in inches. Defaults to (12, 4.4).
        lo_pct, hi_pct : float
            Percentiles used to set the per-channel display range.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
        """
        if not (hasattr(self, 'rec_red') and hasattr(self, 'rec_grn')):
            return None

        print('\tcomputing mean red/green images...')
        # Respect trial_end: average only the kept frames.
        _nk = int(getattr(self.qc, '_n_keep', self.rec_red.shape[0]))
        _mean_red = self._mean_image(self.rec_red[:_nk], stride=stride,
                                     chunk_size=chunk_size)
        _mean_grn = self._mean_image(self.rec_grn[:_nk], stride=stride,
                                     chunk_size=chunk_size)

        def _norm01(img):
            _lo = float(np.nanpercentile(img, lo_pct))
            _hi = float(np.nanpercentile(img, hi_pct))
            if not np.isfinite(_hi) or _hi <= _lo:
                _hi = _lo + 1e-9
            return np.clip((img - _lo) / (_hi - _lo), 0.0, 1.0)

        _red_n = _norm01(_mean_red)
        _grn_n = _norm01(_mean_grn)

        # RGB overlay: red channel -> R, green channel -> G. Pixels with
        # signal in both appear yellow.
        _overlay = np.zeros((*_mean_red.shape, 3), dtype=np.float32)
        _overlay[..., 0] = _red_n
        _overlay[..., 1] = _grn_n

        if figsize is None:
            figsize = (12, 4.4)
        # Link x/y across panels so zooming/panning one zooms all three.
        fig, axes = plt.subplots(1, 3, figsize=figsize,
                                 sharex=True, sharey=True)

        axes[0].imshow(_mean_red, cmap=self._channel_img_cmap('red'),
                       vmin=np.nanpercentile(_mean_red, lo_pct),
                       vmax=np.nanpercentile(_mean_red, hi_pct))
        axes[0].set_title('mean red', fontsize=10)

        axes[1].imshow(_mean_grn, cmap=self._channel_img_cmap('grn'),
                       vmin=np.nanpercentile(_mean_grn, lo_pct),
                       vmax=np.nanpercentile(_mean_grn, hi_pct))
        axes[1].set_title('mean green', fontsize=10)

        axes[2].imshow(_overlay)
        axes[2].set_title('overlay (red + green)', fontsize=10)

        for _ax in axes:
            _ax.set_xticks([])
            _ax.set_yticks([])
            _ax.set_aspect('equal')

        _title = (f'{self.path.animal} {self.path.date} '
                  f'{self.path.beh_folder} — mean red/green overlay')
        fig.suptitle(_title, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if save:
            _ch_suffix = f'_ch={channel}' if channel is not None else ''
            _fname = (f'{self.path.animal}_{self.path.date}_'
                      f'{self.path.beh_folder}'
                      f'_qc_mean_overlay{_ch_suffix}.pdf')
            fig.savefig(os.path.join(str(self.folder.figs), _fname))

        return fig

    # Correction methods whose intermediate steps the QC step-visualiser
    # can render, mapped to a short human-readable label for the figure
    # title. Only full_regress is currently visualisable.
    # ----------
    _CORRSIG_STEP_METHODS = {
        'full_regress': 'full_regress'}

    def _plt_qc_correct_signal(self, channel=None, save=True,
                               save_pickle=True, figsize=None):
        """Visualise every intermediate step of the signal correction
        on the whole-frame and per-sector traces.

        Dual-colour only, and only for ``method='full_regress'``.
        Reproduces the 1-D pipeline (``correct_full_regress_1d`` with
        ``return_steps=True``) on the stored whole-frame control / signal
        traces, then renders one row per processing stage followed by the
        per-sector corrected output: raw F → lowpass + airPLS →
        normalised + regressed control → per-sector zdFF. Returns None
        for single-channel recordings or other methods.

        When the visual-stimulus light-leak step removal was applied
        (``remove_stim_step=True``), two extra rows are prepended: a slim
        event-timing row at the very top (stim / reward dashed markers,
        with each stim line's transparency encoding that trial's expected
        reward — the same trial-type nomenclature as the main QC plot),
        and a raw vs step-removed row showing both channels with their
        independently-estimated per-channel amplitudes annotated.

        Parameters
        ----------
        channel : str or None
            Channel label for the figure filename suffix.
        save : bool
            If True, save a pdf to self.folder.figs.
        save_pickle : bool
            If True (and save is True), also write a gzipped pickle of
            the figure. Default True.
        figsize : tuple or None
            Figure size in inches. Defaults to a height that scales with
            the number of rows.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
        """
        # Gate: dual-colour + a visualisable correction method.
        # ----------
        if not bool(getattr(self.qc, 'correct_signal', False)):
            return None
        _kw = dict(getattr(self.qc, 'correct_signal_kwargs', None) or {})
        _method = _kw.get('method', 'full_regress')
        if _method not in self._CORRSIG_STEP_METHODS:
            return None
        _frame_f = getattr(self.qc, 'frame_f', None)
        if not isinstance(_frame_f, dict):
            return None
        _static = _kw.get('static_flu', 'grn')   # control = s1
        _real = _kw.get('real_flu', 'red')        # signal  = s2
        _s1 = _frame_f.get(_static)
        _s2 = _frame_f.get(_real)
        if _s1 is None or _s2 is None:
            return None
        _s1 = np.asarray(_s1, dtype=np.float64)
        _s2 = np.asarray(_s2, dtype=np.float64)
        # Raw whole-frame traces are full-length and NaN-free; guard in
        # case a partial recording left gaps.
        if not (np.all(np.isfinite(_s1)) and np.all(np.isfinite(_s2))):
            _ok = np.isfinite(_s1) & np.isfinite(_s2)
            if not np.any(_ok):
                return None
            _idx = np.arange(_s1.size)
            _s1 = np.interp(_idx, np.flatnonzero(_ok), _s1[_ok])
            _s2 = np.interp(_idx, np.flatnonzero(_ok), _s2[_ok])

        # Time axis.
        # ----------
        _t = np.asarray(getattr(self.qc, 't', np.arange(_s1.size)))
        if _t.size != _s1.size:
            _t = np.linspace(0, _s1.size - 1, _s1.size)

        _ch_color = {'grn': sns.xkcd_rgb['forest green'],
                     'red': sns.xkcd_rgb['brick']}
        _c1 = _ch_color.get(_static, sns.xkcd_rgb['dark grey'])
        _c2 = _ch_color.get(_real, 'k')

        print(f'\tbuilding correct_signal step figure '
              f'({_method}, control={_static}, signal={_real})...')

        # Method-specific panels (each a dict consumed by the shared
        # renderer below). Every builder appends the shared per-sector
        # panel as its final entry.
        # ----------
        # Only full_regress is visualisable (gated by _CORRSIG_STEP_METHODS).
        _panels = self._corrsig_panels_full_regress(
            _t, _s1, _s2, _kw, _static, _real, _c1, _c2)

        # Prepend the stim light-leak step diagnostic when that pre-
        # correction term was applied (removed inside correct_signal
        # before the control regression).
        # ----------
        _step_on = bool(_kw.get('remove_stim_step', False))
        if _step_on:
            _step_panels = self._corrsig_step_panels(
                _t, _s1, _s2, _static, _real, _c1, _c2)
            _panels = _step_panels + _panels

        # Render. Panels may set a 'height_ratio' (e.g. the slim event-
        # timing row), 'vlines' (event markers), and 'hide_y'.
        # ----------
        _nrows = len(_panels)
        _ratios = [_p.get('height_ratio', 1.0) for _p in _panels]
        if figsize is None:
            figsize = (11, 2.25 * sum(_ratios))
        fig, axes = plt.subplots(_nrows, 1, figsize=figsize, sharex=True,
                                 gridspec_kw={'height_ratios': _ratios})
        if _nrows == 1:
            axes = [axes]
        for _ax, _p in zip(axes, _panels):
            for _x, _y, _color, _ls, _lw, _lbl in _p['lines']:
                _ax.plot(_x, _y, color=_color, linestyle=_ls,
                         linewidth=_lw, label=_lbl)
            _has_vline_lbl = False
            for _vl in _p.get('vlines', []):
                _vx = np.atleast_1d(_vl['x'])
                if _vx.size == 0:
                    continue
                # Per-line transparency (e.g. stim lines encoding expected
                # reward, main-QC nomenclature) when an 'alphas' array is
                # supplied; otherwise a single scalar alpha for the group.
                _alphas = _vl.get('alphas')
                if _alphas is not None:
                    _alphas = np.atleast_1d(_alphas)
                    for _j, _xj in enumerate(_vx):
                        _aj = float(_alphas[_j]) if _j < _alphas.size \
                            else float(_alphas[-1])
                        _ax.vlines(_xj, ymin=_vl.get('ymin', 0),
                                   ymax=_vl.get('ymax', 1),
                                   color=_vl.get('color', 'k'),
                                   linestyle=_vl.get('ls', '--'),
                                   linewidth=_vl.get('lw', 1.0),
                                   alpha=_aj,
                                   label=_vl.get('label') if _j == 0
                                   else None,
                                   transform=_ax.get_xaxis_transform())
                else:
                    _ax.vlines(_vx, ymin=_vl.get('ymin', 0),
                               ymax=_vl.get('ymax', 1),
                               color=_vl.get('color', 'k'),
                               linestyle=_vl.get('ls', '--'),
                               linewidth=_vl.get('lw', 1.0),
                               alpha=_vl.get('alpha', 0.8),
                               label=_vl.get('label'),
                               transform=_ax.get_xaxis_transform())
                _has_vline_lbl = _has_vline_lbl or bool(_vl.get('label'))
            if _p.get('hline0'):
                _ax.axhline(0, color='0.6', linewidth=0.5,
                            linestyle=':')
            _ax.set_ylabel(_p['ylabel'], fontsize=8)
            if _p.get('ylim') is not None:
                _ax.set_ylim(_p['ylim'])
            if _p.get('hide_y'):
                _ax.set_yticks([])
            if any(_l[5] for _l in _p['lines']) or _has_vline_lbl:
                _ax.legend(loc='upper right', fontsize=7, frameon=False,
                           ncol=_p.get('legend_ncol', 1))
            _ax.spines['top'].set_visible(False)
            _ax.spines['right'].set_visible(False)
            _ax.tick_params(labelsize=7)
        axes[-1].set_xlabel(_panels[-1].get('xlabel', 'time (s)'),
                            fontsize=8)

        _title = (f'{self.path.animal} {self.path.date} '
                  f'{self.path.beh_folder} — '
                  f'{self._CORRSIG_STEP_METHODS[_method]} '
                  f'correction steps')
        fig.suptitle(_title, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if save:
            _ch_suffix = f'_ch={channel}' if channel is not None else ''
            _cs_suffix = self._qc_corrsig_suffix()
            _fname = (f'{self.path.animal}_{self.path.date}_'
                      f'{self.path.beh_folder}'
                      f'_qc_correct_signal{_ch_suffix}{_cs_suffix}.pdf')
            fig.savefig(os.path.join(str(self.folder.figs), _fname))
            if save_pickle:
                _pkl_path = os.path.join(
                    str(self.folder.data), _fname[:-4] + '.pkl.gz')
                with gzip.open(_pkl_path, 'wb') as _pf:
                    pickle.dump(fig, _pf,
                                protocol=pickle.HIGHEST_PROTOCOL)

        return fig

    def _corrsig_sector_panel(self, t, real, color, ylabel,
                              fallback_y=None, fallback_label=None):
        """Build the shared final panel: per-sector corrected traces.

        Plots every sector's corrected trace (faint) plus the sector
        mean (bold) from ``self.qc.sector_f['{real}_corr']``. If that is
        unavailable, falls back to a single whole-frame corrected trace.

        Parameters
        ----------
        t : np.ndarray
            Whole-frame time axis (used for the fallback trace).
        real : str
            Real-signal channel label ('red' / 'grn').
        color : str
            Colour for the fallback whole-frame trace.
        ylabel : str
            Y-axis label describing the corrected output for this method.
        fallback_y : np.ndarray or None
            Whole-frame corrected trace to draw if per-sector data is
            missing.
        fallback_label : str or None
            Legend label for the fallback trace.

        Returns
        -------
        panel : dict
        """
        _lines = []
        _sec_f = getattr(self.qc, 'sector_f', None)
        _sec_corr = None
        if isinstance(_sec_f, dict):
            _sec_corr = _sec_f.get(f'{real}_corr')
        if _sec_corr is not None:
            _sec_corr = np.asarray(_sec_corr, dtype=np.float64)
            _sec_t = np.linspace(t[0], t[-1], _sec_corr.shape[1])
            for _row in _sec_corr:
                _lines.append((_sec_t, _row, sns.xkcd_rgb['light grey'],
                               '-', 0.4, None))
            _lines.append((_sec_t, np.nanmean(_sec_corr, axis=0), 'k',
                           '-', 1.0, 'sector mean'))
        elif fallback_y is not None:
            _lines.append((t, fallback_y, color, '-', 0.9,
                           fallback_label))
        return {'ylabel': ylabel, 'lines': _lines, 'hline0': True}

    def _corrsig_step_panels(self, t, s1_raw, s2_raw, static, real, c1, c2):
        """Diagnostic panels for the visual-stimulus light-leak step removal.

        Two rows, sharing the figure's time axis:

        1. a slim event-timing row marking stimulus onsets and reward times
           with dashed vertical lines (stim grey, reward blue) — each stim
           line's transparency encodes that trial's expected reward
           (calc_alpha), exactly matching the main QC plot's trial-type
           nomenclature — placed at the very top of the figure;
        2. raw whole-frame F for *both* channels overlaid with their step-
           removed traces (``raw − A·box(t)``), each channel's step estimated
           independently — the step shows up as the gap that opens during
           each stim-on epoch and closes after removal, with the per-channel
           amplitude A annotated.

        Built from the QC whole-frame traces and behaviour timing
        (stim onset → reward time per trial). The amplitudes actually
        applied during correction (``self.stim_step_amp[channel]``) are used
        for the subtraction when available; otherwise they are re-estimated
        here. Returns [] when stimulus / reward timing is unavailable.

        Parameters
        ----------
        t : np.ndarray
            Time axis of the whole-frame traces (s).
        s1_raw, s2_raw : np.ndarray
            Raw whole-frame control (static) and real-channel F (the same
            traces the other corrsig panels treat as raw F).
        static, real : str
            Control / real channel labels ('red' / 'grn').
        c1, c2 : color
            Control / real channel colours.

        Returns
        -------
        panels : list of dict
            Zero or two panel dicts to prepend to the step figure.
        """
        from .signal_correction import (estimate_stim_step_amplitude,
                                         build_stim_step_offset)
        try:
            stim_on = np.asarray(self.beh.stim.t_start, dtype=float).ravel()
        except (AttributeError, KeyError, TypeError):
            return []
        # Visual-stimulus offset (== where the light-leak ends), not reward
        # time; matches _remove_stim_step so the panel shows the same step
        # that was actually removed. Falls back to reward for older Blocks.
        stim_off, _ = self._stim_off_times()
        if stim_off is None:
            return []
        _te = getattr(self, 'trial_end', None)
        if _te is not None:
            stim_on = stim_on[:int(_te) + 1]
            stim_off = stim_off[:int(_te) + 1]

        _kw = dict(getattr(self.qc, 'correct_signal_kwargs', None) or {})
        _edge = _kw.get('stim_step_edge', 'both')
        _n_edge = int(_kw.get('stim_step_n_edge', 3))
        _n_gap = int(_kw.get('stim_step_n_gap', 1))
        _amp_store = getattr(self, 'stim_step_amp', None)
        _amp_store = _amp_store if isinstance(_amp_store, dict) else {}

        def _step_removed(ch_raw, ch_name):
            # Prefer the amplitude actually applied; else re-estimate on the
            # plotted trace for an internally-consistent panel.
            _a = _amp_store.get(ch_name)
            if _a is None:
                _a, _ = estimate_stim_step_amplitude(
                    ch_raw, t, stim_on, stim_off, edge=_edge,
                    n_edge=_n_edge, n_gap=_n_gap, verbose=False)
            _off = build_stim_step_offset(t, stim_on, stim_off, float(_a))
            return np.asarray(ch_raw, dtype=np.float64) - _off, float(_a)

        _s2_rm, _a2 = _step_removed(s2_raw, real)
        _s1_rm, _a1 = _step_removed(s1_raw, static)

        _grey = sns.xkcd_rgb['dark grey']
        _blue = sns.xkcd_rgb['bright blue']

        # Stim markers carry the main-QC trial-type nomenclature: each stim
        # line's transparency encodes that trial's expected reward
        # (calc_alpha), so trial-types are legible by eye. Reward markers
        # use the delivered-reward times (bright blue), matching the main
        # QC plot's events row.
        # ----------
        _stim_t_al, _stim_alphas = self._qc_stim_event_alphas(alpha_min=0.1)
        _rew_t_al = self._qc_rew_t()

        _timing_panel = {
            'ylabel': 'stim /\nrew',
            'height_ratio': 0.35,
            'hide_y': True,
            'ylim': (0, 1),
            'legend_ncol': 2,
            'lines': [],
            'vlines': [
                {'x': _stim_t_al, 'alphas': _stim_alphas, 'color': _grey,
                 'ls': '--', 'lw': 1.0, 'label': 'stim'},
                {'x': _rew_t_al, 'color': _blue, 'ls': '--', 'lw': 1.0,
                 'alpha': 0.9, 'label': 'rew'}]}

        _rawcorr_panel = {
            'ylabel': 'stim step removal\nraw vs corrected F',
            'legend_ncol': 2,
            'lines': [
                (t, s2_raw, c2, '-', 0.7, f'raw {real}'),
                (t, _s2_rm, c2, '--', 0.9, f'{real} − step (A={_a2:.3g})'),
                (t, s1_raw, c1, '-', 0.7, f'raw {static}'),
                (t, _s1_rm, c1, '--', 0.9,
                 f'{static} − step (A={_a1:.3g})')]}

        return [_timing_panel, _rawcorr_panel]

    def _corrsig_panels_full_regress(self, t, s1, s2, kw, static, real,
                                     c1, c2):
        """Panel list for the full_regress step figure."""
        from .signal_correction import correct_full_regress_1d
        _kw_1d = {k: kw[k] for k in (
            'smooth_window', 'airpls_lam', 'airpls_porder',
            'airpls_max_iter', 'trim_initial', 'nn_slope',
            'beta_loss', 'beta_f_scale', 'beta_scale') if k in kw}
        _corr, _beta, _st = correct_full_regress_1d(
            s1, s2, verbose=False, return_steps=True, **_kw_1d)
        _panels = [
            {'ylabel': '1. raw\nwhole-frame F',
             'lines': [
                 (t, _st['s2'], c2, '-', 0.7, f'{real} (signal)'),
                 (t, _st['s1'], c1, '-', 0.7, f'{static} (control)')]},
            {'ylabel': '2. lowpass smoothed\nF (+ airPLS base.)',
             'legend_ncol': 2,
             'lines': [
                 (t, _st['s2_sm'], c2, '-', 0.8, f'{real} smoothed'),
                 (t, _st['s1_sm'], c1, '-', 0.8, f'{static} smoothed'),
                 (t, _st['b2'], c2, '--', 0.8, f'{real} airPLS'),
                 (t, _st['b1'], c1, '--', 0.8, f'{static} airPLS')]},
            {'ylabel': '3. normalised +\nregressed control (z)',
             'hline0': True,
             'lines': [
                 (t, _st['s2_norm'], c2, '-', 0.7, f'{real} norm (z)'),
                 (t, _st['s1_norm_scaled'], c1, '-', 0.7,
                  f'β·{static} norm (β={_beta:.3g})')]}]
        _panels.append(self._corrsig_sector_panel(
            t, real, c2, '4. per-sector zdFF\n(s2_norm − β·s1_norm)',
            fallback_y=_st['corrected'],
            fallback_label='whole-frame zdFF (1-D)'))
        return _panels

    def _plt_qc_reg_stats(self, t, reg, channel=None,
                          t_pre=2.0, t_post=2.0,
                          figsize=None, save=True):
        """QC summary of registration metrics: event-triggered averages
        on top, dual-colour correlation + behaviour block underneath.

        Top block (2 rows × 3 cols): stim-triggered (row 0) and
        reward-triggered (row 1) averages of shift_mag, corrXY, z_corr.
        Each trial baseline-subtracted using the pre-event window; SEM
        shown as a shaded band.

        Bottom block (3 rows × 2 cols):
            row 0 — frame-by-frame correlation of the whole-frame
                    control channel vs. the raw functional channel
                    (left) and its corrected version (right), both
                    z-scored with a unity line and Pearson r. Which
                    channel plays which role follows plt_qc's `channel`
                    (red vs grn by default; grn vs red for
                    channel='grn').
            row 1 — square (n_sectors × n_sectors) sector-by-sector
                    Pearson-r maps for the same two pairings.
            row 2 — licking rate split by base trial-type (one column per
                    stimulus condition), aligned to stim onset (4 s pre,
                    6 s post). Each column carries the main-QC stim/reward
                    line nomenclature: a dark-grey stim dash whose
                    transparency encodes that condition's expected reward,
                    and a bright-blue reward dash at its median
                    stim→reward latency (drawn only when the condition is
                    rewarded).

        Parameters
        ----------
        t : np.ndarray
            QC time vector from self.qc.t.
        reg : SimpleNamespace
            Registration metrics from self.qc.reg.
        channel : str or None
            Channel label used for the figure filename suffix.
        t_pre : float
            Seconds before each event for the event-avg block.
        t_post : float
            Seconds after each event for the event-avg block.
        figsize : tuple or None
            Figure size in inches. Auto-sized to accommodate both blocks
            if None.
        save : bool
            If True, saves a pdf to self.folder.figs.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        # Resolve event-avg metrics (z_corr falls back to early ref in
        # dual-ref mode, which is a usable % Δ trace).
        _zc_ev = reg.z_corr
        _zc_ev_title = 'z_corr (% Δ)'
        if _zc_ev is None and reg.z_corr_early is not None:
            _zc_ev = reg.z_corr_early
            _zc_ev_title = 'z_corr early ref (% Δ)'
        _ev_metrics = {
            'shift_mag': self._interp_metric_to_t(reg.shift_mag, t),
            'corrXY':    self._interp_metric_to_t(reg.corrXY, t),
            'z_corr':    self._interp_metric_to_t(_zc_ev, t),
        }
        _metric_keys = ['shift_mag', 'corrXY', 'z_corr']
        _ev_metric_titles = ['shift mag (px)', 'corrXY (% Δ)',
                             _zc_ev_title]
        _ev_metric_pct = {'shift_mag': False, 'corrXY': True,
                          'z_corr': True}

        _stim_t = self._qc_stim_t()
        _rew_t = self._qc_rew_t()
        _row_events = [_stim_t, _rew_t]
        _row_colors = [sns.xkcd_rgb['dark grey'],
                       sns.xkcd_rgb['bright blue']]
        _row_ylabels = ['stim-triggered', 'reward-triggered']

        # Resolve dual-colour channels for the correlation panels: the
        # corrected real-channel key (e.g. 'red_corr'), its raw base, and
        # the static reference channel. Both roles follow the QC's active
        # channel (channel='grn' ⇒ real='grn', static='red'); the
        # corrected key, when present, is authoritative for the real base.
        # ----------
        _corr_key = next(
            (k for k in self.qc.frame_f if k.endswith('_corr')), None)
        _real_qc, _static_qc = self._qc_flu_pair()
        _real_base = (_corr_key[:-len('_corr')]
                      if _corr_key else _real_qc)
        _static_base = ('grn' if _real_base == 'red' else 'red')
        _ch_col = {'grn': sns.xkcd_rgb['forest green'],
                   'red': sns.xkcd_rgb['brick']}
        _col_static = _ch_col.get(_static_base, sns.xkcd_rgb['dark grey'])
        _col_real = _ch_col.get(_real_base, sns.xkcd_rgb['dark grey'])
        _col_corr = sns.xkcd_rgb['purple']

        figsize = figsize or (10, 5 + 9.0)
        fig = plt.figure(figsize=figsize)
        # Two stacked sub-grids: event-avg block on top, correlation +
        # behaviour block underneath. The bottom block is 3 rows × 2 cols:
        # frame-by-frame correlation (red|red_corr vs green) on row 0,
        # square sector-by-sector correlation (red|red_corr vs green) on
        # row 1, and a full-width licking-rate panel spanning row 2.
        # ----------
        _ev_h = 5.0
        _bot_h = 9.0
        outer = gridspec.GridSpec(
            nrows=2, ncols=1, figure=fig,
            height_ratios=[_ev_h, _bot_h], hspace=0.3)
        ev_spec = outer[0, 0].subgridspec(2, 3, hspace=0.25, wspace=0.25)
        bot_spec = outer[1, 0].subgridspec(
            3, 2, hspace=0.5, wspace=0.3,
            height_ratios=[1.0, 1.0, 0.8])

        # ---- Event-avg block ----
        ev_axes = np.empty((2, 3), dtype=object)
        for r_idx in range(2):
            for c_idx in range(3):
                _sharex = ev_axes[0, c_idx] if r_idx > 0 else None
                _sharey = ev_axes[r_idx, 0] if c_idx > 0 else None
                ev_axes[r_idx, c_idx] = fig.add_subplot(
                    ev_spec[r_idx, c_idx],
                    sharex=_sharex, sharey=_sharey)

        for r_idx, (ev_t, ev_color, yl) in enumerate(
                zip(_row_events, _row_colors, _row_ylabels)):
            for c_idx, (mkey, mtitle) in enumerate(
                    zip(_metric_keys, _ev_metric_titles)):
                ax = ev_axes[r_idx, c_idx]
                _trace = _ev_metrics[mkey]

                if _trace is None or ev_t.size == 0:
                    ax.text(0.5, 0.5,
                            'no data' if _trace is None else 'no events',
                            ha='center', va='center',
                            transform=ax.transAxes,
                            fontsize=8, color='grey')
                else:
                    _t_rel, _mean, _sem = self._compute_event_avg(
                        _trace, t, ev_t, t_pre=t_pre, t_post=t_post,
                        pct=_ev_metric_pct[mkey])
                    if _mean is None:
                        ax.text(0.5, 0.5, 'no events in range',
                                ha='center', va='center',
                                transform=ax.transAxes,
                                fontsize=8, color='grey')
                    else:
                        ax.fill_between(_t_rel,
                                        _mean - _sem, _mean + _sem,
                                        color=ev_color, alpha=0.25,
                                        linewidth=0)
                        ax.plot(_t_rel, _mean,
                                color=ev_color, linewidth=1.2)

                ax.axvline(x=0, color=ev_color, linewidth=1.0,
                           linestyle='--', alpha=0.9)
                ax.axhline(y=0, color='k', linewidth=0.4, linestyle=':')

                if r_idx == 0:
                    ax.set_title(mtitle, fontsize=9)
                if c_idx == 0:
                    ax.set_ylabel(yl, fontsize=8)
                if r_idx == 1:
                    ax.set_xlabel('time from event (s)', fontsize=8)

                for _side in ('top', 'right'):
                    ax.spines[_side].set_visible(False)

        # Helpers shared by the correlation panels.
        # ----------
        def _finite_xy(_x, _y):
            if _x is None or _y is None:
                return None, None
            _x = np.asarray(_x, dtype=np.float64).ravel()
            _y = np.asarray(_y, dtype=np.float64).ravel()
            _n = min(_x.size, _y.size)
            _x, _y = _x[:_n], _y[:_n]
            _m = np.isfinite(_x) & np.isfinite(_y)
            return _x[_m], _y[_m]

        def _pearson(_x, _y):
            if _x is None or _x.size < 2:
                return np.nan
            return float(np.corrcoef(_x, _y)[0, 1])

        def _zscore(_a):
            _a = np.asarray(_a, dtype=np.float64)
            _mu = float(np.mean(_a))
            _sd = float(np.std(_a))
            if not np.isfinite(_sd) or _sd == 0:
                return _a - _mu
            return (_a - _mu) / _sd

        _sec_f = getattr(self.qc, 'sector_f', None)

        # ---- Row 0: frame-by-frame correlation, z-scored, with a unity
        #             line. red vs green (left), red_corr vs green (right).
        # ----------
        _ff_static = self.qc.frame_f.get(_static_base)
        _ff_real = self.qc.frame_f.get(_real_base)
        _ff_corr = self.qc.frame_f.get(_corr_key) if _corr_key else None

        def _frame_corr_panel(ax, y_raw, color, ylabel, title):
            _x, _y = _finite_xy(_ff_static, y_raw)
            if _x is None or _x.size < 2:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8, color='grey')
                ax.set_title(title, fontsize=9)
                return
            _r = _pearson(_x, _y)
            _xz, _yz = _zscore(_x), _zscore(_y)
            ax.scatter(_xz, _yz, color=color, s=1, alpha=0.3,
                       linewidths=0, rasterized=True)
            _all = np.concatenate([_xz, _yz])
            _lo = float(np.nanpercentile(_all, 0.5))
            _hi = float(np.nanpercentile(_all, 99.5))
            if _hi <= _lo:
                _hi = _lo + 1.0
            ax.plot([_lo, _hi], [_lo, _hi], color='k', linewidth=0.8,
                    linestyle='--', alpha=0.7, label='unity')
            ax.set_xlim(_lo, _hi)
            ax.set_ylim(_lo, _hi)
            ax.set_aspect('equal')
            ax.set_xlabel(f'whole-frame {_static_base} (z)', fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(f'{title}\n(r={_r:.2f})', fontsize=9)
            for _side in ('top', 'right'):
                ax.spines[_side].set_visible(False)

        ax_fc_red = fig.add_subplot(bot_spec[0, 0])
        _frame_corr_panel(ax_fc_red, _ff_real, _col_real,
                          f'whole-frame {_real_base} (z)',
                          f'frame corr: {_real_base} vs {_static_base}')
        ax_fc_corr = fig.add_subplot(bot_spec[0, 1])
        if _corr_key:
            _frame_corr_panel(ax_fc_corr, _ff_corr, _col_corr,
                              f'whole-frame {_corr_key} (z)',
                              f'frame corr: {_corr_key} vs {_static_base}')
        else:
            ax_fc_corr.text(0.5, 0.5, 'no corrected channel',
                            ha='center', va='center',
                            transform=ax_fc_corr.transAxes,
                            fontsize=8, color='grey')

        # ---- Row 1: square sector-by-sector correlation vs green.
        #             red vs green (left), red_corr vs green (right).
        # ----------
        _sec_static = (_sec_f.get(_static_base)
                       if isinstance(_sec_f, dict) else None)

        def _sector_corr_grid(sec_other):
            if _sec_static is None or sec_other is None:
                return None
            _sa = np.asarray(_sec_static, dtype=np.float64)
            _sb = np.asarray(sec_other, dtype=np.float64)
            _nsec = min(_sa.shape[0], _sb.shape[0])
            _nt = min(_sa.shape[1], _sb.shape[1])
            _sa, _sb = _sa[:_nsec, :_nt], _sb[:_nsec, :_nt]
            _r_sec = np.full(_nsec, np.nan)
            for _i in range(_nsec):
                _m = np.isfinite(_sa[_i]) & np.isfinite(_sb[_i])
                if _m.sum() >= 2:
                    _r_sec[_i] = np.corrcoef(_sa[_i, _m], _sb[_i, _m])[0, 1]
            _ns = int(self.qc.n_sectors)
            return (_r_sec.reshape(_ns, _ns)
                    if _r_sec.size == _ns * _ns
                    else _r_sec[np.newaxis, :])

        def _sector_corr_panel(ax, grid, title, cbar=False):
            if grid is None:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8, color='grey')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(title, fontsize=9)
                return
            # imshow's default aspect='equal' keeps a square FOV grid
            # square (vs the previous 'auto', which stretched it).
            _im = ax.imshow(grid, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=9)
            if cbar:
                _cax = ax.inset_axes([1.05, 0, 0.05, 1])
                fig.colorbar(_im, cax=_cax)
                _cax.tick_params(labelsize=7)
                _cax.set_ylabel('Pearson r', fontsize=7)

        _sec_real = (_sec_f.get(_real_base)
                     if isinstance(_sec_f, dict) else None)
        _sec_corr = (_sec_f.get(_corr_key)
                     if (isinstance(_sec_f, dict) and _corr_key) else None)
        ax_sc_red = fig.add_subplot(bot_spec[1, 0])
        _sector_corr_panel(ax_sc_red, _sector_corr_grid(_sec_real),
                           f'sector corr: {_real_base} vs {_static_base}',
                           cbar=False)
        ax_sc_corr = fig.add_subplot(bot_spec[1, 1])
        _sector_corr_panel(
            ax_sc_corr, _sector_corr_grid(_sec_corr),
            f'sector corr: {_corr_key} vs {_static_base}' if _corr_key
            else 'sector corr: (no corrected channel)',
            cbar=True)

        # ---- Row 2: licking rate split by trial-type, aligned to stim
        #             onset (one column per base stimulus condition; 4 s
        #             pre-stim, 6 s post-stim). Each column carries the
        #             main-QC stim/reward line nomenclature: a dark-grey
        #             stim dash whose transparency encodes that condition's
        #             expected reward, and a bright-blue reward dash at the
        #             condition's median stim→reward latency.
        # ----------
        _lick_t = self._qc_lick_t(getattr(self.beh.lick, 't_raw', []))
        _t_pre_l, _t_post_l, _bin = 4.0, 6.0, 0.2
        _edges = np.arange(-_t_pre_l, _t_post_l + _bin, _bin)
        _centers = 0.5 * (_edges[:-1] + _edges[1:])
        _lick_conds = self._qc_base_tr_conds()
        _n_lcond = len(_lick_conds)
        _lick_spec = bot_spec[2, :].subgridspec(1, _n_lcond, wspace=0.25)
        _lick_axes = []
        for _ci, (_clabel, _cinds) in enumerate(_lick_conds):
            _ax = fig.add_subplot(_lick_spec[0, _ci],
                                  sharey=_lick_axes[0] if _lick_axes else None)
            _lick_axes.append(_ax)
            _cstim = self._qc_cond_stim_t(_cinds)
            _rates = []
            if _lick_t.size and _cstim.size:
                for _ev in np.asarray(_cstim).ravel():
                    _counts, _ = np.histogram(_lick_t - float(_ev),
                                              bins=_edges)
                    _rates.append(_counts / _bin)
            if _rates:
                _rates = np.asarray(_rates, dtype=np.float64)
                _mean = _rates.mean(axis=0)
                _sem = _rates.std(axis=0) / np.sqrt(_rates.shape[0])
                _ax.fill_between(_centers, _mean - _sem, _mean + _sem,
                                 color=sns.xkcd_rgb['bright blue'],
                                 alpha=0.25, linewidth=0)
                _ax.plot(_centers, _mean,
                         color=sns.xkcd_rgb['bright blue'], linewidth=1.2)
            else:
                _ax.text(0.5, 0.5, 'no licks', ha='center', va='center',
                         transform=_ax.transAxes,
                         fontsize=8, color='grey')

            # Stim line: alpha encodes the condition's expected reward
            # (main-QC nomenclature). Reward line: drawn at the condition's
            # median latency, only when the condition actually delivers
            # reward.
            # ----------
            _stim_alpha = self._qc_cond_alpha(_cinds, alpha_min=0.1)
            _rew_rel, _rew_frac = self._qc_cond_rew_rel(_cinds)
            _ax.axvline(x=0, color=sns.xkcd_rgb['dark grey'],
                        linewidth=1.0, linestyle='--', alpha=_stim_alpha,
                        label='stim onset')
            if _rew_rel is not None and _rew_frac > 0:
                _ax.axvline(x=_rew_rel, color=sns.xkcd_rgb['bright blue'],
                            linewidth=1.0, linestyle='--', alpha=0.9,
                            label=f'reward (+{_rew_rel:.1f}s)')
            _ax.legend(loc='upper right', fontsize=6, frameon=False)
            _ax.set_xlim(-_t_pre_l, _t_post_l)
            _ax.set_xlabel('time from stim (s)', fontsize=8)
            _ax.set_title(f'lick rate — {_clabel}\n(n={_cstim.size})',
                          fontsize=8)
            if _ci == 0:
                _ax.set_ylabel('lick rate (Hz)', fontsize=8)
            else:
                plt.setp(_ax.get_yticklabels(), visible=False)
            for _side in ('top', 'right'):
                _ax.spines[_side].set_visible(False)

        _title = (f'{self.path.animal} {self.path.date} '
                  f'{self.path.beh_folder} — QC (registration statistics)')
        fig.suptitle(_title, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if save:
            _ch_suffix = f'_ch={channel}' if channel is not None else ''
            _cs_suffix = self._qc_corrsig_suffix()
            _fname = (f'{self.path.animal}_{self.path.date}_'
                      f'{self.path.beh_folder}'
                      f'_qc_reg_stats{_ch_suffix}{_cs_suffix}.pdf')
            fig.savefig(os.path.join(str(self.folder.figs), _fname))

        return fig

    def _save_qc_response_mean_singlech_split_lr(
            self, t, channel=None,
            t_pre=2.0, t_post=6.0, dff_sig=False):
        """Save a hemisphere-encoded response_mean .npy for a single-
        channel recording with split_lr=True.

        No figure is produced (that path is dual-colour only); the .npy
        layout mirrors the dual-colour save so downstream plt_all-style
        code can index hemispheres uniformly. With ``_label`` = the
        channel label (e.g. 'red'), the saved dict contains:

            whole_frame[_label]            mean+sem over both halves
            whole_frame[_label_left]       bottom-half y-rows
            whole_frame[_label_right]      top-half y-rows
            sectors[_label]                full (n_sectors², n_t)
            sectors[_label_right]          top-half rows of the grid
            sectors[_label_left]           bottom-half rows of the grid

        Parameters
        ----------
        t : np.ndarray
            QC time vector from self.qc.t.
        channel : str or None
            Channel label used to look up frame_f halves and to build
            the saved-summary keys (e.g. 'red'). Defaults to 'frame'.
        t_pre, t_post : float
            Window around each stim onset (seconds).
        dff_sig : bool
            If True, snippets are converted to dF/F0 (%) per trial using
            the pre-stim baseline. Otherwise traces are ITI z-scored
            first and then averaged. Default False.
        """
        _label = channel if channel is not None else 'frame'
        _ff_l = self.qc.frame_f.get(f'{_label}_left')
        _ff_r = self.qc.frame_f.get(f'{_label}_right')
        if _ff_l is None or _ff_r is None:
            return
        _ff_l = np.asarray(_ff_l, dtype=np.float64)
        _ff_r = np.asarray(_ff_r, dtype=np.float64)
        _ff_full = 0.5 * (_ff_l + _ff_r)

        _stim_t = self._qc_stim_t()

        def _zscore_iti(trace):
            # Z-score against the ITI baseline.
            _mask = self._inter_trial_mask(t)
            _base = np.asarray(trace)[_mask]
            _base = _base[np.isfinite(_base)]
            if _base.size < 2:
                return np.full_like(trace, np.nan, dtype=np.float64)
            _mu = float(np.mean(_base))
            _sd = float(np.std(_base))
            if _sd == 0 or not np.isfinite(_sd):
                return np.full_like(trace, np.nan, dtype=np.float64)
            return (np.asarray(trace, dtype=np.float64) - _mu) / _sd

        _dt = float(np.median(np.diff(t)))
        _n_pre = int(round(t_pre / _dt))
        _n_post = int(round(t_post / _dt))
        _n_win = _n_pre + _n_post + 1
        _t_rel = np.linspace(-t_pre, t_post, _n_win)

        def _wf_mean_sem(trace_in, use_dff):
            # If use_dff, each trial snippet is converted to dF/F0 (%)
            # against its pre-stim baseline before being averaged
            # across trials; otherwise trace_in is expected to be
            # pre-z-scored.
            _snips = []
            for _ev in np.asarray(_stim_t).ravel():
                _i = int(np.argmin(np.abs(t - _ev)))
                _i0 = _i - _n_pre
                _i1 = _i + _n_post + 1
                if _i0 < 0 or _i1 > len(trace_in):
                    continue
                _snip = np.asarray(trace_in[_i0:_i1], dtype=np.float64)
                if not np.all(np.isfinite(_snip)):
                    continue
                if use_dff:
                    if _n_pre <= 0:
                        continue
                    _f0 = float(np.mean(_snip[:_n_pre]))
                    if not np.isfinite(_f0) or _f0 == 0:
                        continue
                    _snip = (_snip - _f0) / _f0 * 100.0
                _snips.append(_snip)
            if not _snips:
                return None, None
            _arr = np.asarray(_snips, dtype=np.float64)
            return (_arr.mean(axis=0),
                    _arr.std(axis=0) / np.sqrt(_arr.shape[0]))

        # Whole-frame entries: full FOV (mean of halves), left, right.
        # ----------
        _wf = {}
        for _lbl, _trace in [(_label, _ff_full),
                             (f'{_label}_left', _ff_l),
                             (f'{_label}_right', _ff_r)]:
            if dff_sig:
                _mn, _sm = _wf_mean_sem(_trace, use_dff=True)
            else:
                _mn, _sm = _wf_mean_sem(_zscore_iti(_trace),
                                        use_dff=False)
            _wf[_lbl] = {'mean': _mn, 'sem': _sm}

        # Per-sector matrix. sector_f is a single ndarray (full grid,
        # not hemisphere-split); we apply the same row-major top/bottom
        # split that the QC figure uses.
        # ----------
        _sec = np.asarray(self.qc.sector_f, dtype=np.float32)
        n_compute = _sec.shape[1]
        sec_t = np.linspace(t[0], t[-1], n_compute)

        _sec_iti_mask = self._inter_trial_mask(sec_t)
        if dff_sig:
            _sec_in = _sec.astype(np.float64)
        else:
            _base = _sec[:, _sec_iti_mask]
            with np.errstate(invalid='ignore'):
                _mu = np.nanmean(_base, axis=1, keepdims=True)
                _sd = np.nanstd(_base, axis=1, keepdims=True)
            _z = np.full_like(_sec, np.nan, dtype=np.float64)
            _ok = np.isfinite(_sd[:, 0]) & (_sd[:, 0] > 0)
            if np.any(_ok):
                _z[_ok] = (_sec[_ok].astype(np.float64)
                           - _mu[_ok]) / _sd[_ok]
            _sec_in = _z

        dt_sec = float(np.median(np.diff(sec_t)))
        n_pre = int(round(t_pre / dt_sec))
        n_post = int(round(t_post / dt_sec))
        n_win = n_pre + n_post + 1
        t_rel_sec = np.linspace(-t_pre, t_post, n_win)

        def _stim_aligned_sectors(sec_mat, use_dff):
            _avg = np.full((sec_mat.shape[0], n_win), np.nan,
                           dtype=np.float32)
            _ev_arr = np.asarray(_stim_t).ravel()
            if use_dff:
                for _i in range(sec_mat.shape[0]):
                    _snips = []
                    for _ev in _ev_arr:
                        _ind = int(np.argmin(np.abs(sec_t - _ev)))
                        _i0 = _ind - n_pre
                        _i1 = _ind + n_post + 1
                        if _i0 < 0 or _i1 > sec_mat.shape[1]:
                            continue
                        _snip = np.asarray(sec_mat[_i, _i0:_i1],
                                           dtype=np.float64)
                        if not np.all(np.isfinite(_snip)):
                            continue
                        if n_pre <= 0:
                            continue
                        _f0 = float(np.mean(_snip[:n_pre]))
                        if not np.isfinite(_f0) or _f0 == 0:
                            continue
                        _snips.append((_snip - _f0) / _f0 * 100.0)
                    if _snips:
                        _avg[_i] = np.mean(_snips, axis=0)
            else:
                for _i in range(sec_mat.shape[0]):
                    _, _m, _ = self._compute_event_avg(
                        sec_mat[_i], sec_t, _stim_t,
                        t_pre=t_pre, t_post=t_post)
                    if _m is not None:
                        _avg[_i] = _m
            return _avg

        sec_avg_full = _stim_aligned_sectors(_sec_in, use_dff=dff_sig)

        _ns = int(self.qc.n_sectors)
        _n_top = (_ns // 2) * _ns
        _sec_dict = {
            _label: np.asarray(sec_avg_full),
            f'{_label}_right': np.asarray(sec_avg_full[:_n_top]),
            f'{_label}_left': np.asarray(sec_avg_full[_n_top:]),
        }

        # 1-D hemisphere-corrected trace (from add_qc, when
        # split_lr+correct_signal+grab5ht_side ran). Saved under
        # `{label}_corr_{grab_side}` to match the frame_f key, plus a
        # mirror `{label}_corr` for grand-average pipelines that key off
        # the *_corr suffix.
        # ----------
        _grab_side = getattr(self.qc, 'grab5ht_side', None)
        _corr_key_hem = (f'{_label}_corr_{_grab_side}'
                         if _grab_side in ('left', 'right') else None)
        _ff_corr_hem = (self.qc.frame_f.get(_corr_key_hem)
                        if _corr_key_hem else None)
        if _ff_corr_hem is not None:
            _ff_corr_hem = np.asarray(_ff_corr_hem, dtype=np.float64)
            _mn, _sm = _wf_mean_sem(_ff_corr_hem, use_dff=False)
            _wf[_corr_key_hem] = {'mean': _mn, 'sem': _sm}
            _wf[f'{_label}_corr'] = {'mean': _mn, 'sem': _sm}

        # Median stim → reward latency (matches the dual-colour save).
        # ----------
        _rew_t_all = np.asarray(getattr(self.beh.rew, 't', []),
                                dtype=object).ravel()
        _te_lim = getattr(self, 'trial_end', None)
        if _te_lim is not None:
            _rew_t_all = _rew_t_all[:int(_te_lim) + 1]
        _stim_t_arr = np.asarray(_stim_t).ravel()
        _lat = []
        for _i in range(min(len(_rew_t_all), len(_stim_t_arr))):
            _r = _rew_t_all[_i]
            if _r is None:
                continue
            try:
                _rf = float(_r)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(_rf):
                continue
            _lat.append(_rf - float(_stim_t_arr[_i]))
        _rew_t_rel = float(np.median(_lat)) if _lat else None

        _summary = {
            'mode': 'dff_sig' if dff_sig else 'iti_zscore',
            'split_lr': True,
            'channel': _label,
            'grab5ht_side': _grab_side,
            'corr_key': (f'{_label}_corr'
                         if _ff_corr_hem is not None else None),
            'grab5ht_beta': float(
                getattr(self.qc, 'grab5ht_beta', np.nan)),
            't_rel': _t_rel,
            't_rel_sec': t_rel_sec,
            'rew_t_rel': _rew_t_rel,
            'n_sectors': int(self.qc.n_sectors),
            'whole_frame': _wf,
            'sectors': _sec_dict,
        }

        _ch_suffix = f'_ch={channel}' if channel is not None else ''
        _cs_suffix = self._qc_corrsig_suffix()
        _dff_suffix = '_dffsig' if dff_sig else ''
        _npy_name = (f'{self.path.animal}_{self.path.date}_'
                     f'{self.path.beh_folder}'
                     f'_qc_response_mean'
                     f'{_ch_suffix}{_cs_suffix}{_dff_suffix}.npy')
        np.save(os.path.join(str(self.folder.data), _npy_name),
                _summary, allow_pickle=True)
        return

    def _martianova_flu_1d(self, s1, s2, kw_1d):
        """Reconstruct a fluorescence-units corrected trace (Approach A).

        Runs the 1-D Martianova pipeline (control ``s1``, signal ``s2``)
        with ``return_steps=True`` and maps the z-scored corrected output
        back to fluorescence units via

            F_corr(t) = σ₂ · zdFF(t) + m₂ + b₂(t)

        where σ₂, m₂ are the signal channel's normalisation constants and
        b₂(t) its airPLS baseline (all from the pipeline's step dict).
        The result carries s2's session DC level, so a per-trial pre-stim
        F0 is meaningful and the standard dF/F0 (%) snippet path applies.

        Parameters
        ----------
        s1, s2 : array-like, shape (T,)
            Control (regressor) and signal traces at the imaging rate.
        kw_1d : dict
            1-D Martianova parameters forwarded to
            ``correct_full_regress_1d`` (smooth_window, airpls_*,
            trim_initial, nn_slope).

        Returns
        -------
        f_corr : np.ndarray, shape (T,), float64
            Corrected signal in fluorescence units.
        """
        from .signal_correction import correct_full_regress_1d
        _corr, _beta, _st = correct_full_regress_1d(
            np.asarray(s1, dtype=np.float64),
            np.asarray(s2, dtype=np.float64),
            verbose=False, return_steps=True, **kw_1d)
        return (_st['std2'] * np.asarray(_st['corrected'], dtype=np.float64)
                + _st['m2'] + np.asarray(_st['b2'], dtype=np.float64))

    def _plt_qc_response_mean(self, t, channel=None,
                                t_pre=2.0, t_post=6.0,
                                figsize=None, save=True,
                                dff_sig=False,
                                save_npy=True):
        """Stim-aligned GRAB-style averages (dual-colour only).

        Columns: raw red (col 0) and raw green (col 1), each as a
        whole-frame mean ± SEM across stims above and a per-sector
        heatmap (stim-averaged) below, plus a spatial response map.
        A third column for the regression-corrected real channel is
        added only when a corrected channel is available (i.e. plt_qc
        was called with correct_signal=True). The red/grn ratio column
        has been removed.

        Each column is split by base trial-type (stimulus condition,
        e.g. '0' / '0.5' / '1'; derived sub-subtypes excluded): the
        whole-frame trace row overlays one mean ± SEM line per trial-type
        distinguished by linestyle (solid / dashed / dotted / dash-dot),
        and the sector heatmap and spatial map are each duplicated into
        one sub-row per trial-type (rows share the pooled sector
        ordering). Sector ordering, trial reliability, the correlation
        statistics and the saved .npy summary all remain pooled across
        every stim trial — only these three visuals split by trial-type.

        Returns None if the recording is not dual-colour.

        Parameters
        ----------
        t : np.ndarray
            QC time vector from self.qc.t.
        channel : str or None
            Channel label for the figure filename suffix.
        t_pre, t_post : float
            Window around each stim onset (seconds).
        figsize : tuple or None
            Figure size in inches. Defaults to (7, 6) for a single
            column, (12, 6) when the corrected-channel column is added.
        save : bool
            If True, saves a pdf to self.folder.figs.
        dff_sig : bool
            If True, the raw red, grn and red/grn columns are converted
            to dF/F0 (%) per trial, where F0 is the mean of the trace in
            the pre-stim baseline window (length t_pre seconds). Applied
            independently to whole-frame and per-sector traces. Default
            False (uses ITI z-score). The corrected column is z-scored
            unless ``qc.dff_after_correction`` is set (see plt_qc), in
            which case it is shown as per-trial dF/F0 (%) from the
            fluorescence-units Approach-A reconstruction.
        save_npy : bool
            If True, also writes a `.npy` file (to self.folder.data) with
            the summary data plotted in this figure: whole-frame mean
            and SEM and per-sector stim-aligned averages for each
            channel (red, grn, ratio, corr), plus sector ordering,
            sector reliability, and time vectors. Saved as a pickled
            dict (np.save with allow_pickle=True). Default True.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
        """
        # Dual-colour only for the figure path. For single-channel
        # recordings with split_lr=True we still save a hemisphere-
        # encoded response_mean .npy (no figure) so the bulk
        # hemisphere-control flow has something to load.
        # ----------
        _is_dual = (isinstance(self.qc.sector_f, dict)
                    and 'red' in self.qc.sector_f
                    and 'grn' in self.qc.sector_f)
        if not _is_dual:
            if (save_npy
                    and getattr(self.qc, 'split_lr', False)
                    and self.qc.sector_f is not None
                    and not isinstance(self.qc.sector_f, dict)):
                self._save_qc_response_mean_singlech_split_lr(
                    t, channel=channel,
                    t_pre=t_pre, t_post=t_post, dff_sig=dff_sig)
            return None

        # Whole-frame red & grn traces. Use the full-FOV keys when
        # available; if split_lr is on, average left+right halves to
        # reconstruct a whole-frame estimate.
        # ----------
        _ff_r = self.qc.frame_f.get('red')
        _ff_g = self.qc.frame_f.get('grn')
        if _ff_r is None or _ff_g is None:
            _r_l = self.qc.frame_f.get('red_left')
            _r_r = self.qc.frame_f.get('red_right')
            _g_l = self.qc.frame_f.get('grn_left')
            _g_r = self.qc.frame_f.get('grn_right')
            if all(x is not None for x in (_r_l, _r_r, _g_l, _g_r)):
                _ff_r = 0.5 * (_r_l + _r_r)
                _ff_g = 0.5 * (_g_l + _g_r)
            else:
                return None

        _sec_r = self.qc.sector_f['red'].astype(np.float32)
        _sec_g = self.qc.sector_f['grn'].astype(np.float32)

        # Channel roles: the functional (real) channel is the one the
        # correction targets, i.e. the QC's active channel (channel='grn'
        # ⇒ signal = grn, control = red). Everything role-dependent below
        # — the signal/control ratio, the Approach-A reconstruction, the
        # sector ordering and the reliability column — keys off these,
        # rather than assuming red is always functional.
        # ----------
        _real_ch, _static_ch = self._qc_flu_pair()
        _sig_is_red = (_real_ch != 'grn')
        _ff_sig = _ff_r if _sig_is_red else _ff_g
        _ff_ctrl = _ff_g if _sig_is_red else _ff_r
        _sec_sig = _sec_r if _sig_is_red else _sec_g
        _sec_ctrl = _sec_g if _sig_is_red else _sec_r

        with np.errstate(divide='ignore', invalid='ignore'):
            _ratio_ff = _ff_sig / (_ff_ctrl + 1e-9)
            _ratio_sec = _sec_sig / (_sec_ctrl + 1e-9)

        # Detect the corrected real channel (set up via correct_signal=
        # True in plt_qc). Both the whole-frame trace and the per-sector
        # matrix must be present for the right column to be drawn.
        # ----------
        _corr_key = None
        for _k in self.qc.frame_f:
            if _k.endswith('_corr') and _k in self.qc.sector_f:
                _corr_key = _k
                break
        _ff_corr = None
        _sec_corr = None
        if _corr_key is not None:
            _ff_corr_raw = self.qc.frame_f.get(_corr_key)
            if _ff_corr_raw is None:
                _ll = self.qc.frame_f.get(f'{_corr_key}_left')
                _rr = self.qc.frame_f.get(f'{_corr_key}_right')
                if _ll is not None and _rr is not None:
                    _ff_corr_raw = 0.5 * (_ll + _rr)
            if _ff_corr_raw is not None:
                _ff_corr = np.asarray(_ff_corr_raw, dtype=np.float64)
                _sec_corr = self.qc.sector_f[_corr_key].astype(np.float32)
            else:
                _corr_key = None

        _has_corr = _corr_key is not None
        _stim_t = self._qc_stim_t()

        # Correction method (drives the Approach-A dF/F reconstruction
        # for the corrected column; see _corr_dff below).
        # ----------
        _cs_kw = getattr(self.qc, 'correct_signal_kwargs', None) or {}
        _corr_method = _cs_kw.get('method', 'full_regress')

        # Helper: ITI z-score a 1D trace given the time vector it lives on.
        # ----------
        def _iti_mu_sd(trace, t_vec):
            """(mu, sd) of a trace over its inter-trial baseline.

            Returns (None, None) when the baseline is too short or has
            zero / non-finite spread.
            """
            _mask = self._inter_trial_mask(t_vec)
            _base = np.asarray(trace)[_mask]
            _base = _base[np.isfinite(_base)]
            if _base.size < 2:
                return None, None
            _mu = float(np.mean(_base))
            _sd = float(np.std(_base))
            if _sd == 0 or not np.isfinite(_sd):
                return None, None
            return _mu, _sd

        def _zscore_iti(trace, t_vec):
            _mu, _sd = _iti_mu_sd(trace, t_vec)
            if _mu is None:
                return np.full_like(trace, np.nan, dtype=np.float64)
            return (np.asarray(trace, dtype=np.float64) - _mu) / _sd

        # Z-score whole-frame traces against their own ITI baselines.
        # ----------
        _ratio_ff_z = _zscore_iti(_ratio_ff, t)
        # The corrected channel is already a native dF/F (subtractive
        # dff_sig − dff_ctrl, e.g. pixel_spatial_subtr — continuous for
        # f0_mode='global', or per-trial with NaN inter-trial gaps for
        # 'per_trial'), so it must NOT be ITI z-scored. Pass it through,
        # scaled to dF/F (%) so its amplitude is comparable to the z-score
        # columns on the shared heatmap colour scale.
        _corr_predff = getattr(self.qc, 'corr_native_dff', None) is not None
        if not _has_corr:
            _ff_corr_z = None
        elif _corr_predff:
            _ff_corr_z = np.asarray(_ff_corr, dtype=np.float64) * 100.0
        else:
            _ff_corr_z = _zscore_iti(_ff_corr, t)

        # Approach A (dff_after_correction): reconstruct fluorescence-units
        # corrected traces so the corrected column can be shown as per-trial
        # dF/F0 (%). The z-scored zdFF from full_regress carries no
        # fluorescence scale, so the 1-D Martianova pipeline is re-run per
        # region on the raw control and signal traces (whichever channels
        # carry those roles) and mapped back to F via
        # F_corr = σ2·zdFF + m2 + b2 (see _martianova_flu_1d).
        # Only active for full_regress (the linear_dff channel is
        # already per-trial dF/F; other methods stay z-scored).
        # ----------
        _corr_dff = (bool(getattr(self.qc, 'dff_after_correction', False))
                     and _has_corr and not _corr_predff
                     and _corr_method == 'full_regress')
        _ff_corr_flu = None
        _sec_corr_flu = None
        if _corr_dff:
            print('\tdff_after_correction: reconstructing fluorescence-'
                  'units corrected traces (Approach A)...')
            _kw_1d = {_k: _cs_kw[_k] for _k in (
                'smooth_window', 'airpls_lam', 'airpls_porder',
                'airpls_max_iter', 'trim_initial', 'nn_slope',
                'beta_loss', 'beta_f_scale', 'beta_scale')
                if _k in _cs_kw}
            try:
                _ff_corr_flu = self._martianova_flu_1d(
                    _ff_ctrl, _ff_sig, _kw_1d)
            except ValueError:
                _ff_corr_flu = None
            _sec_corr_flu = np.full_like(_sec_sig, np.nan, dtype=np.float64)
            for _si in range(_sec_sig.shape[0]):
                try:
                    _sec_corr_flu[_si] = self._martianova_flu_1d(
                        _sec_ctrl[_si], _sec_sig[_si], _kw_1d)
                except ValueError:
                    pass
            if _ff_corr_flu is None:
                # Reconstruction failed (e.g. too-short trace) — fall back
                # to the z-scored corrected channel.
                _corr_dff = False

        # Sector grid is computed at sector_stride; build matching t vec.
        # ----------
        n_sec_total, n_compute = _ratio_sec.shape
        sec_t = np.linspace(t[0], t[-1], n_compute)

        # Per-sector ITI z-score using each sector's own ITI baseline.
        # ----------
        _sec_iti_mask = self._inter_trial_mask(sec_t)

        def _zscore_sectors(sec_mat):
            _base = sec_mat[:, _sec_iti_mask]
            with np.errstate(invalid='ignore'):
                _mu = np.nanmean(_base, axis=1, keepdims=True)
                _sd = np.nanstd(_base, axis=1, keepdims=True)
            _z = np.full_like(sec_mat, np.nan, dtype=np.float64)
            _ok = np.isfinite(_sd[:, 0]) & (_sd[:, 0] > 0)
            if np.any(_ok):
                _z[_ok] = (sec_mat[_ok].astype(np.float64)
                           - _mu[_ok]) / _sd[_ok]
            return _z

        _ratio_sec_z = _zscore_sectors(_ratio_sec)
        if not _has_corr:
            _sec_corr_z = None
        elif _corr_predff:
            # Native dF/F (subtractive): the whole-frame trace passes
            # through scaled to %, so the sectors must match — no
            # re-z-scoring, same ×100 to dF/F (%).
            _sec_corr_z = np.asarray(_sec_corr, dtype=np.float64) * 100.0
        else:
            # Common whole-frame reference for the corrected channel.
            # The corrected whole-frame trace and the per-sector matrix
            # are both spatial means of the SAME per-pixel zdFF stack, so
            # they must be put on one scale to be comparable. We z-score
            # every sector against the *whole-frame* ITI mean/std (the
            # same μ_WF, σ_WF used for the corr whole-frame trace) rather
            # than each sector's own ITI std.
            #
            # Why this matters for the corrected channel specifically: the
            # Martianova correction removes the spatially-coherent noise
            # component, so the residual ITI noise is spatially
            # independent and averages down ~√N. A single sector (~1/64 of
            # the FOV) then has an ITI std ~√(n_sec_total) larger than the
            # whole frame's. Per-sector z-scoring would divide each
            # sector's coherent response by that inflated denominator,
            # compressing the heatmap toward 0 even though the underlying
            # response amplitude matches the whole-frame trace. Using the
            # common σ_WF keeps mean(sectors) ≈ whole-frame trace; the
            # genuine per-sector noise then shows honestly as scatter.
            # (Raw red/grn keep per-sector z-scoring — their ITI noise is
            # spatially coherent, so the two scales already agree.)
            _mu_wf, _sd_wf = _iti_mu_sd(_ff_corr, t)
            if _mu_wf is None:
                _sec_corr_z = np.full_like(
                    _sec_corr, np.nan, dtype=np.float64)
            else:
                _sec_corr_z = (np.asarray(_sec_corr, dtype=np.float64)
                               - _mu_wf) / _sd_wf

        # Raw red channel: ITI z-score for whole-frame trace and for
        # per-sector matrix. Used for the leftmost column.
        # ----------
        _ff_r_z = _zscore_iti(_ff_r, t)
        _sec_r_z = _zscore_sectors(_sec_r)

        # Raw green channel: ITI z-score for whole-frame trace and for
        # per-sector matrix. Used for the second column.
        # ----------
        _ff_g_z = _zscore_iti(_ff_g, t)
        _sec_g_z = _zscore_sectors(_sec_g)

        # Stim-aligned avgs on the QC time vector (whole-frame).
        # ----------
        _dt = float(np.median(np.diff(t)))
        _n_pre = int(round(t_pre / _dt))
        _n_post = int(round(t_post / _dt))
        _n_win = _n_pre + _n_post + 1
        _t_rel = np.linspace(-t_pre, t_post, _n_win)

        def _stim_snips(trace, dff_override=None, stim_t=None):
            # When `dff` (resolved from dff_sig or dff_override) is True,
            # the input is a raw trace and each per-trial snippet is
            # converted to dF/F0 (%) using the pre-stim baseline window
            # (length _n_pre samples) as F0. Otherwise the input is
            # assumed pre-z-scored. dff_override forces the dF/F path
            # (True) or the z-score path (False) regardless of dff_sig.
            # stim_t restricts the alignment to a subset of stim onsets
            # (a base trial-type); None uses every stim onset.
            _dff = dff_sig if dff_override is None else bool(dff_override)
            _evs = _stim_t if stim_t is None else stim_t
            _out = []
            for _ev in np.asarray(_evs).ravel():
                _i = int(np.argmin(np.abs(t - _ev)))
                _i0 = _i - _n_pre
                _i1 = _i + _n_post + 1
                if _i0 < 0 or _i1 > len(trace):
                    continue
                _snip = np.asarray(trace[_i0:_i1], dtype=np.float64)
                # NaN-tolerant: a per-trial dF/F trace is NaN outside its
                # window, so keep snippets with finite coverage (gaps are
                # averaged out nan-aware across trials by the caller) and
                # drop only those with no usable baseline.
                if not np.any(np.isfinite(_snip)):
                    continue
                if _dff:
                    if _n_pre <= 0:
                        continue
                    with np.errstate(invalid='ignore'):
                        _f0 = float(np.nanmean(_snip[:_n_pre]))
                    if not np.isfinite(_f0) or _f0 == 0:
                        continue
                    _snip = (_snip - _f0) / _f0 * 100.0
                _out.append(_snip)
            return _out

        # Stim-aligned per-sector avgs on the sector time vector.
        # ----------
        dt_sec = float(np.median(np.diff(sec_t)))
        n_pre = int(round(t_pre / dt_sec))
        n_post = int(round(t_post / dt_sec))
        n_win = n_pre + n_post + 1
        t_rel_sec = np.linspace(-t_pre, t_post, n_win)

        def _stim_aligned_sectors(sec_mat, dff_override=None, stim_t=None):
            # If `dff` (resolved from dff_sig or dff_override) is on,
            # expect raw sector traces in and convert each per-trial
            # snippet to dF/F0 (%) using its own pre-stim baseline
            # (length n_pre samples) before averaging across trials.
            # Otherwise expect pre-z-scored sectors. dff_override forces
            # the dF/F path (True) or z-score path (False) regardless of
            # dff_sig. stim_t restricts the alignment to a subset of stim
            # onsets (a base trial-type); None uses every stim onset.
            _dff = dff_sig if dff_override is None else bool(dff_override)
            _evs = _stim_t if stim_t is None else stim_t
            _avg = np.full((sec_mat.shape[0], n_win), np.nan,
                           dtype=np.float32)
            if _dff:
                _ev_arr = np.asarray(_evs).ravel()  # base trial-type subset
                for _i in range(sec_mat.shape[0]):
                    _snips = []
                    for _ev in _ev_arr:
                        _ind = int(np.argmin(np.abs(sec_t - _ev)))
                        _i0 = _ind - n_pre
                        _i1 = _ind + n_post + 1
                        if _i0 < 0 or _i1 > sec_mat.shape[1]:
                            continue
                        _snip = np.asarray(sec_mat[_i, _i0:_i1],
                                           dtype=np.float64)
                        if not np.all(np.isfinite(_snip)):
                            continue
                        if n_pre <= 0:
                            continue
                        _f0 = float(np.mean(_snip[:n_pre]))
                        if not np.isfinite(_f0) or _f0 == 0:
                            continue
                        _snips.append((_snip - _f0) / _f0 * 100.0)
                    if _snips:
                        _avg[_i] = np.mean(_snips, axis=0)
            else:
                for _i in range(sec_mat.shape[0]):
                    _, _m, _ = self._compute_event_avg(
                        sec_mat[_i], sec_t, _evs,
                        t_pre=t_pre, t_post=t_post)
                    if _m is not None:
                        _avg[_i] = _m
            return _avg

        # Pick the inputs fed into the snippet extractors. In dff_sig
        # mode we pass the raw post-processed traces and let the snippet
        # helpers convert each trial to dF/F0 (%); otherwise we pass the
        # ITI-z-scored traces.
        # ----------
        if dff_sig:
            _wf_red = _ff_r
            _wf_grn = _ff_g
            _wf_ratio = _ratio_ff
            _wf_corr = _ff_corr_z
            _sec_red_in = _sec_r
            _sec_grn_in = _sec_g
            _sec_ratio_in = _ratio_sec
            _sec_corr_in = _sec_corr_z
        else:
            _wf_red = _ff_r_z
            _wf_grn = _ff_g_z
            _wf_ratio = _ratio_ff_z
            _wf_corr = _ff_corr_z
            _sec_red_in = _sec_r_z
            _sec_grn_in = _sec_g_z
            _sec_ratio_in = _ratio_sec_z
            _sec_corr_in = _sec_corr_z

        # Corrected column: by default ITI z-sorted (dff_override=False).
        # Under dff_after_correction the fluorescence-units reconstruction
        # is fed instead, and each per-trial snippet is converted to dF/F0
        # (%) (dff_override=True).
        # ----------
        if _corr_dff:
            _wf_corr = _ff_corr_flu
            _sec_corr_in = _sec_corr_flu

        sector_avg = _stim_aligned_sectors(_sec_ratio_in)
        sector_avg_corr = (_stim_aligned_sectors(_sec_corr_in,
                                                 dff_override=_corr_dff)
                           if _has_corr else None)
        sector_avg_r = _stim_aligned_sectors(_sec_red_in)
        sector_avg_g = _stim_aligned_sectors(_sec_grn_in)

        # Per-sector trial reliability: mean off-diagonal Pearson R across
        # stim-aligned trials. Uses the same source as the row-ordering
        # (corrected channel if available, else the raw functional
        # channel) so the column tells a consistent story alongside the
        # heatmaps.
        # ----------
        def _sector_reliability(sec_z):
            _rel = np.full(sec_z.shape[0], np.nan, dtype=np.float64)
            _ev_arr = np.asarray(_stim_t).ravel()
            for _i in range(sec_z.shape[0]):
                _snips = []
                for _ev in _ev_arr:
                    _ind = int(np.argmin(np.abs(sec_t - _ev)))
                    _i0 = _ind - n_pre
                    _i1 = _ind + n_post + 1
                    if _i0 < 0 or _i1 > sec_z.shape[1]:
                        continue
                    _s = sec_z[_i, _i0:_i1]
                    if not np.all(np.isfinite(_s)):
                        continue
                    if np.std(_s) == 0:
                        continue
                    _snips.append(_s)
                if len(_snips) < 2:
                    continue
                with np.errstate(invalid='ignore'):
                    _cm = np.corrcoef(np.asarray(_snips))
                _iu = np.triu_indices(_cm.shape[0], k=1)
                _vals = _cm[_iu]
                _vals = _vals[np.isfinite(_vals)]
                if _vals.size:
                    _rel[_i] = float(np.mean(_vals))
            return _rel

        _rel_src = (_sec_corr_z if _has_corr
                    else (_sec_r_z if _sig_is_red else _sec_g_z))
        sector_reliability = _sector_reliability(_rel_src)

        # Median stim → reward latency from .beh (per-trial pairing).
        # Used both for the dashed blue reward marker on every axis and
        # for the integration window when ordering sectors.
        # ----------
        _rew_t_all = np.asarray(getattr(self.beh.rew, 't', []),
                                dtype=object).ravel()
        _te_lim = getattr(self, 'trial_end', None)
        if _te_lim is not None:
            _rew_t_all = _rew_t_all[:int(_te_lim) + 1]
        _stim_t_arr = np.asarray(_stim_t).ravel()
        _lat = []
        for _i in range(min(len(_rew_t_all), len(_stim_t_arr))):
            _r = _rew_t_all[_i]
            if _r is None:
                continue
            try:
                _rf = float(_r)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(_rf):
                continue
            _lat.append(_rf - float(_stim_t_arr[_i]))
        _rew_t_rel = float(np.median(_lat)) if _lat else None

        # Base trial-types (stimulus conditions) and their stim onsets;
        # used both for the ordering source below and for the per-type
        # heatmap / spatial-map splits later.
        # ----------
        _resp_conds = self._qc_base_tr_conds()
        _resp_cond_stim_t = {_lbl: self._qc_cond_stim_t(_inds)
                             for _lbl, _inds in _resp_conds}

        # Order sectors descending by peak stim → rew response. The sort
        # source is the SINGLE base trial-type whose maximum per-sector
        # peak amplitude is the largest (computed on the corrected channel
        # when available, else the raw functional channel); that one
        # ordering is then applied to every trial-type, so each heatmap
        # indexes the same sectors in the same rows.
        # ----------
        if _rew_t_rel is not None and _rew_t_rel > 0:
            _i0 = int(np.searchsorted(t_rel_sec, 0.0, side='left'))
            _i1 = int(np.searchsorted(t_rel_sec, _rew_t_rel,
                                      side='right'))
        else:
            _i0 = int(np.searchsorted(t_rel_sec, 0.0, side='left'))
            _i1 = sector_avg.shape[1]
        _i1 = max(_i1, _i0 + 1)
        with np.errstate(invalid='ignore'):
            _resp = np.nansum(sector_avg[:, _i0:_i1], axis=1)
            _resp_corr = (np.nansum(sector_avg_corr[:, _i0:_i1], axis=1)
                          if sector_avg_corr is not None else None)
            _resp_r = np.nansum(sector_avg_r[:, _i0:_i1], axis=1)
            _resp_g = np.nansum(sector_avg_g[:, _i0:_i1], axis=1)
            # Pooled per-sector peak (fallback ordering source). Prefer the
            # corrected real channel (cleanest activation); fall back to
            # the raw functional channel when correct_signal=False.
            _peak_corr = (np.nanmax(sector_avg_corr[:, _i0:_i1], axis=1)
                          if sector_avg_corr is not None else None)
            _peak_sig = np.nanmax(
                (sector_avg_r if _sig_is_red else sector_avg_g)[:, _i0:_i1],
                axis=1)

        # Ordering channel input + its per-trial-type peak: find the
        # trial-type with the single largest peak amplitude and sort by
        # that type's per-sector peak vector.
        # ----------
        if _has_corr:
            _ord_sec_in, _ord_dff, _ord_ch = _sec_corr_in, _corr_dff, _corr_key
            _peak_pool = _peak_corr
        else:
            _ord_sec_in = _sec_red_in if _sig_is_red else _sec_grn_in
            _ord_dff, _ord_ch = None, _real_ch
            _peak_pool = _peak_sig
        _best_label, _best_peak_vec, _best_peak_val = None, None, -np.inf
        for _lbl, _ in _resp_conds:
            _cst = _resp_cond_stim_t[_lbl]
            if _ord_sec_in is None or _cst.size == 0:
                continue
            _avg_c = _stim_aligned_sectors(
                _ord_sec_in, dff_override=_ord_dff, stim_t=_cst)
            with np.errstate(invalid='ignore'):
                _pk = np.nanmax(_avg_c[:, _i0:_i1], axis=1)
            if not np.any(np.isfinite(_pk)):
                continue
            _mx = float(np.nanmax(_pk))
            if _mx > _best_peak_val:
                _best_peak_val, _best_peak_vec, _best_label = _mx, _pk, _lbl
        if _best_peak_vec is not None:
            _order_src = _best_peak_vec
            _sort_label = f'{_ord_ch} @ {_best_label}'
        else:
            _order_src = _peak_pool
            _sort_label = _ord_ch
        _order = np.argsort(np.where(np.isfinite(_order_src),
                                     _order_src, -np.inf))[::-1]
        sector_avg = sector_avg[_order]
        if sector_avg_corr is not None:
            sector_avg_corr = sector_avg_corr[_order]
        sector_avg_r = sector_avg_r[_order]
        sector_avg_g = sector_avg_g[_order]
        sector_reliability = sector_reliability[_order]

        # Spatial maps of integrated response: reshape the per-sector
        # integral back onto the n_sectors × n_sectors FOV grid (row-major
        # order matches _compute_sectors). Used for the bottom row.
        # ----------
        _ns = int(self.qc.n_sectors)
        _resp_grid = (_resp.reshape(_ns, _ns)
                      if _resp.size == _ns * _ns else None)
        _resp_grid_corr = (_resp_corr.reshape(_ns, _ns)
                           if (_resp_corr is not None
                               and _resp_corr.size == _ns * _ns)
                           else None)
        _resp_grid_r = (_resp_r.reshape(_ns, _ns)
                        if _resp_r.size == _ns * _ns else None)
        _resp_grid_g = (_resp_g.reshape(_ns, _ns)
                        if _resp_g.size == _ns * _ns else None)

        # Whole-frame Pearson R: raw functional vs control fluorescence,
        # and (when available) corrected real-channel vs control.
        # Computed on the full QC time series over jointly finite samples.
        # ----------
        def _pearson_finite(a, b):
            _a = np.asarray(a, dtype=np.float64).ravel()
            _b = np.asarray(b, dtype=np.float64).ravel()
            _m = np.isfinite(_a) & np.isfinite(_b)
            if (_m.sum() < 3 or np.std(_a[_m]) == 0
                    or np.std(_b[_m]) == 0):
                return np.nan, np.nan
            _r, _p = sp_stats.pearsonr(_a[_m], _b[_m])
            return float(_r), float(_p)

        _pear_rg_r, _pear_rg_p = _pearson_finite(_ff_sig, _ff_ctrl)
        if _has_corr:
            _pear_cg_r, _pear_cg_p = _pearson_finite(_ff_corr, _ff_ctrl)
        else:
            _pear_cg_r = _pear_cg_p = np.nan

        # Sector-map Spearman: integrated-response sector vectors
        # (functional vs control, and corrected vs control). p-value from
        # a spatial-permutation null distribution that shuffles control
        # sector positions while holding the signal / corr map fixed.
        # ----------
        def _sector_corr_with_null(map_a, map_g, kind='spearman',
                                    n_perm=1000, seed=0):
            _a = np.asarray(map_a, dtype=np.float64).ravel()
            _g = np.asarray(map_g, dtype=np.float64).ravel()
            _m = np.isfinite(_a) & np.isfinite(_g)
            if _m.sum() < 4:
                return np.nan, np.nan, np.array([])
            _a_, _g_ = _a[_m], _g[_m]

            if kind == 'spearman':
                def _r(x, y):
                    with np.errstate(invalid='ignore'):
                        return sp_stats.spearmanr(x, y).correlation
            else:
                def _r(x, y):
                    if np.std(x) == 0 or np.std(y) == 0:
                        return np.nan
                    with np.errstate(invalid='ignore'):
                        return sp_stats.pearsonr(x, y)[0]

            _rho_obs = _r(_a_, _g_)
            if not np.isfinite(_rho_obs):
                return np.nan, np.nan, np.array([])
            _rho_obs = float(_rho_obs)
            _rng = np.random.default_rng(seed)
            _null = np.full(n_perm, np.nan, dtype=np.float64)
            for _i in range(n_perm):
                _gp = _rng.permutation(_g_)
                _rh = _r(_a_, _gp)
                if np.isfinite(_rh):
                    _null[_i] = _rh
            _null_finite = _null[np.isfinite(_null)]
            if _null_finite.size == 0:
                return _rho_obs, np.nan, _null_finite
            _p_val = float((np.sum(np.abs(_null_finite)
                                   >= abs(_rho_obs)) + 1)
                           / (_null_finite.size + 1))
            return _rho_obs, _p_val, _null_finite

        _spear_rg_rho = _spear_rg_p = np.nan
        _spear_cg_rho = _spear_cg_p = np.nan
        _spear_rg_null = _spear_cg_null = np.array([])
        _pear_sec_rg_r = _pear_sec_rg_p = np.nan
        _pear_sec_cg_r = _pear_sec_cg_p = np.nan
        _pear_sec_rg_null = _pear_sec_cg_null = np.array([])
        _resp_sig = _resp_r if _sig_is_red else _resp_g
        _resp_ctrl = _resp_g if _sig_is_red else _resp_r
        if _resp_grid_r is not None and _resp_grid_g is not None:
            _spear_rg_rho, _spear_rg_p, _spear_rg_null = \
                _sector_corr_with_null(_resp_sig, _resp_ctrl,
                                       kind='spearman')
            _pear_sec_rg_r, _pear_sec_rg_p, _pear_sec_rg_null = \
                _sector_corr_with_null(_resp_sig, _resp_ctrl,
                                       kind='pearson')
        if (_has_corr and _resp_grid_corr is not None
                and _resp_grid_g is not None):
            _spear_cg_rho, _spear_cg_p, _spear_cg_null = \
                _sector_corr_with_null(_resp_corr, _resp_ctrl,
                                       kind='spearman')
            _pear_sec_cg_r, _pear_sec_cg_p, _pear_sec_cg_null = \
                _sector_corr_with_null(_resp_corr, _resp_ctrl,
                                       kind='pearson')

        # Pre-format stats blurbs to drop under the functional and corr
        # columns. None passed for the control / ratio columns.
        # ----------
        def _fmt_stat(rho, p):
            if not np.isfinite(rho):
                return 'n/a'
            if not np.isfinite(p):
                _p_str = 'n/a'
            elif p < 1e-3:
                _p_str = f'{p:.1e}'
            else:
                _p_str = f'{p:.3f}'
            return f'R={rho:.3f}, p={_p_str}'

        _vs_ctrl = f'vs {_static_ch}'
        _stats_sig = (
            f'whole-frame Pearson ({_real_ch} {_vs_ctrl}):\n'
            f'  {_fmt_stat(_pear_rg_r, _pear_rg_p)}\n'
            f'sector Pearson ({_real_ch} {_vs_ctrl}):\n'
            f'  {_fmt_stat(_pear_sec_rg_r, _pear_sec_rg_p)}\n'
            f'sector Spearman ({_real_ch} {_vs_ctrl}):\n'
            f'  {_fmt_stat(_spear_rg_rho, _spear_rg_p)}'
        )
        if _has_corr:
            _stats_corr = (
                f'whole-frame Pearson ({_corr_key} {_vs_ctrl}):\n'
                f'  {_fmt_stat(_pear_cg_r, _pear_cg_p)}\n'
                f'sector Pearson ({_corr_key} {_vs_ctrl}):\n'
                f'  {_fmt_stat(_pear_sec_cg_r, _pear_sec_cg_p)}\n'
                f'sector Spearman ({_corr_key} {_vs_ctrl}):\n'
                f'  {_fmt_stat(_spear_cg_rho, _spear_cg_p)}'
            )
        else:
            _stats_corr = None

        # Per-base-trial-type stim-aligned data for the requested
        # trial-type splits: whole-frame trace overlays (one line per
        # type, distinct linestyle), duplicated sector heatmaps, and
        # duplicated spatial maps. Reliability, stats and the saved .npy
        # above all stay pooled — only these three visuals split by
        # trial-type. Every condition's sector_avg is re-indexed by the
        # shared _order (the max-peak trial-type's ordering) so rows
        # correspond across conditions, and its spatial map integrates the
        # same stim→rew window [_i0:_i1]. (_resp_conds / _resp_cond_stim_t
        # were resolved above for the ordering step.)
        # ----------
        _n_rcond = len(_resp_conds)
        _cond_linestyles = ['-', '--', ':', '-.']

        def _cond_sector_data(sec_in, dff_override=None):
            """Per-condition (ordered sector_avg, resp_grid) for a sector
            input, keyed by base-condition label."""
            _out = {}
            for _lbl, _ in _resp_conds:
                _st = _resp_cond_stim_t[_lbl]
                if sec_in is None or _st.size == 0:
                    _out[_lbl] = (None, None)
                    continue
                _avg = _stim_aligned_sectors(
                    sec_in, dff_override=dff_override, stim_t=_st)
                with np.errstate(invalid='ignore'):
                    _resp = np.nansum(_avg[:, _i0:_i1], axis=1)
                _grid = (_resp.reshape(_ns, _ns)
                         if _resp.size == _ns * _ns else None)
                _out[_lbl] = (_avg[_order], _grid)
            return _out

        _cond_sec_r = _cond_sector_data(_sec_red_in)
        _cond_sec_g = _cond_sector_data(_sec_grn_in)
        _cond_sec_corr = (
            _cond_sector_data(_sec_corr_in, dff_override=_corr_dff)
            if _has_corr else {})

        # Figure layout
        # ----------
        # Data columns: raw red (always), raw green (always), and the
        # corrected channel (only when correct_signal=True). Each column
        # gets its own spatial map. The red/grn ratio column has been
        # removed.
        _ncols = 3 if _has_corr else 2
        # The sector-heatmap (row 1) and spatial-map (row 2) blocks are
        # duplicated once per base trial-type, so both grow with _n_rcond;
        # the figure height scales to keep each sub-panel legible.
        if figsize is None:
            _w = 18.0 if _has_corr else 12.0
            figsize = (_w, 3.1 + 3.0 * _n_rcond)
        fig = plt.figure(figsize=figsize)
        # Last column is a narrow sector-reliability strip (only row 1).
        _width_ratios = [1.0] * _ncols + [0.18]
        spec = gridspec.GridSpec(
            nrows=3, ncols=_ncols + 1, figure=fig,
            height_ratios=[1.6, 1.6 * _n_rcond, 1.4 * _n_rcond],
            width_ratios=_width_ratios,
            hspace=0.3, wspace=0.6)

        def _draw_column(col, trace_z, cond_sector,
                         trace_label, heat_label, spatial_label,
                         trace_color, draw_spatial=True,
                         spatial_cmap='Reds',
                         dff_override=None,
                         stats_text=None,
                         heat_vmag=None,
                         spatial_vlo=None, spatial_vhi=None):
            ax_trace = fig.add_subplot(spec[0, col])
            _heat_sub = spec[1, col].subgridspec(_n_rcond, 1, hspace=0.12)
            _spatial_sub = (spec[2, col].subgridspec(_n_rcond, 1,
                                                     hspace=0.25)
                            if draw_spatial else None)
            _col_dff = (dff_sig if dff_override is None
                        else bool(dff_override))

            # --- A: whole-frame trace, one line per base trial-type,
            #        distinguished by linestyle (solid / dashed / ...) ---
            _any_snip = False
            for _ci, (_clabel, _) in enumerate(_resp_conds):
                _cst = _resp_cond_stim_t[_clabel]
                _snips = _stim_snips(trace_z, dff_override=dff_override,
                                     stim_t=_cst)
                if not _snips:
                    continue
                _any_snip = True
                _arr = np.array(_snips)
                # nan-aware: a per-trial dF/F snippet is NaN outside its
                # trial window, so average across trials ignoring gaps.
                with np.errstate(invalid='ignore'):
                    _n_fin = np.sum(np.isfinite(_arr), axis=0)
                    _mean = np.nanmean(_arr, axis=0)
                    _sem = (np.nanstd(_arr, axis=0)
                            / np.sqrt(np.maximum(_n_fin, 1)))
                _mean[_n_fin == 0] = np.nan
                _sem[_n_fin == 0] = np.nan
                _ls = _cond_linestyles[_ci % len(_cond_linestyles)]
                ax_trace.fill_between(_t_rel, _mean - _sem, _mean + _sem,
                                      color=trace_color, alpha=0.12,
                                      linewidth=0)
                ax_trace.plot(_t_rel, _mean, color=trace_color,
                              linewidth=1.1, linestyle=_ls,
                              label=_clabel)
            if not _any_snip:
                ax_trace.text(0.5, 0.5, 'no stim events in range',
                              ha='center', va='center',
                              transform=ax_trace.transAxes,
                              fontsize=8, color='grey')
            ax_trace.axvline(x=0, color='k', linewidth=0.8,
                             linestyle='--', alpha=0.7)
            if _rew_t_rel is not None:
                ax_trace.axvline(x=_rew_t_rel,
                                 color=sns.xkcd_rgb['bright blue'],
                                 linewidth=0.8, linestyle='--',
                                 alpha=0.8)
            ax_trace.axhline(y=0, color='k', linewidth=0.4,
                             linestyle=':')
            ax_trace.set_ylabel(trace_label, fontsize=8)
            # Trial-type linestyles are decoded once by the shared
            # figure-level legend at the top (built after all columns).
            plt.setp(ax_trace.get_xticklabels(), visible=False)
            ax_trace.set_xlim(t_rel_sec[0], t_rel_sec[-1])
            for _side in ('top', 'right'):
                ax_trace.spines[_side].set_visible(False)

            # --- B: per-trial-type sector heatmaps (one sub-row per
            #        trial-type; rows share the pooled sector ordering) ---
            _last_heat_ax = None
            for _ci, (_clabel, _) in enumerate(_resp_conds):
                _ax_h = fig.add_subplot(_heat_sub[_ci, 0], sharex=ax_trace)
                _last_heat_ax = _ax_h
                _savg = cond_sector.get(_clabel, (None, None))[0]
                if _savg is not None and np.any(np.isfinite(_savg)):
                    if heat_vmag is not None:
                        _vmag = heat_vmag
                    else:
                        _vlo = float(np.nanpercentile(_savg, 2))
                        _vhi = float(np.nanpercentile(_savg, 98))
                        _vmag = max(abs(_vlo), abs(_vhi))
                    _x_edges = np.linspace(t_rel_sec[0], t_rel_sec[-1],
                                           _savg.shape[1] + 1)
                    _y_edges = np.arange(_savg.shape[0] + 1)
                    _im = _ax_h.pcolormesh(
                        _x_edges, _y_edges, _savg,
                        cmap='gray', vmin=-_vmag, vmax=_vmag,
                        shading='flat')
                    _ax_h.set_ylim(_savg.shape[0], 0)
                    _ax_h.set_yticks([0, _savg.shape[0]])
                    if _ci == 0:
                        _cax = _ax_h.inset_axes([1.015, 0, 0.012, 1])
                        fig.colorbar(_im, cax=_cax)
                        _cax.tick_params(labelsize=7)
                        _cax.set_ylabel(
                            'dF/F0 (%)' if _col_dff else 'z-score',
                            fontsize=7)
                else:
                    _ax_h.text(0.5, 0.5, 'no stim events',
                               ha='center', va='center',
                               transform=_ax_h.transAxes,
                               fontsize=7, color='grey')
                _ax_h.axvline(x=0, color=sns.xkcd_rgb['bright red'],
                              linewidth=0.8, linestyle='--', alpha=0.8)
                if _rew_t_rel is not None:
                    _ax_h.axvline(x=_rew_t_rel,
                                  color=sns.xkcd_rgb['bright blue'],
                                  linewidth=0.8, linestyle='--',
                                  alpha=0.8)
                _ax_h.set_ylabel(
                    f'{heat_label}\n[{_clabel}]' if _ci == 0
                    else f'[{_clabel}]', fontsize=7)
                for _side in ('top', 'right'):
                    _ax_h.spines[_side].set_visible(False)
                if _ci < _n_rcond - 1:
                    plt.setp(_ax_h.get_xticklabels(), visible=False)
                else:
                    _ax_h.set_xlabel('time from stim (s)', fontsize=8)
            if _last_heat_ax is not None:
                _last_heat_ax.set_xlim(t_rel_sec[0], t_rel_sec[-1])

            # --- C: per-trial-type spatial sector maps (one sub-row per
            #        trial-type, same duplication as the heatmaps) ---
            _last_spatial_ax = None
            if _spatial_sub is not None:
                for _ci, (_clabel, _) in enumerate(_resp_conds):
                    _ax_s = fig.add_subplot(_spatial_sub[_ci, 0])
                    _last_spatial_ax = _ax_s
                    _grid = cond_sector.get(_clabel, (None, None))[1]
                    if _grid is not None and np.any(np.isfinite(_grid)):
                        if (spatial_vlo is not None
                                and spatial_vhi is not None):
                            _vlo, _vhi = spatial_vlo, spatial_vhi
                        else:
                            _vlo = float(np.nanpercentile(_grid, 2))
                            _vhi = float(np.nanpercentile(_grid, 98))
                        if _vhi <= _vlo:
                            _vhi = _vlo + 1e-9
                        _x_e = np.arange(_grid.shape[1] + 1)
                        _y_e = np.arange(_grid.shape[0] + 1)
                        _im_sp = _ax_s.pcolormesh(
                            _x_e, _y_e, _grid,
                            cmap=spatial_cmap, vmin=_vlo, vmax=_vhi,
                            shading='flat')
                        _ax_s.set_aspect('equal')
                        _ax_s.invert_yaxis()
                        _ax_s.set_xticks([])
                        _ax_s.set_yticks([])
                        if _ci == 0:
                            _cax = _ax_s.inset_axes([1.03, 0, 0.04, 1])
                            fig.colorbar(_im_sp, cax=_cax)
                            _cax.tick_params(labelsize=7)
                            _cax.set_ylabel('∫ response\n(stim→rew)',
                                            fontsize=7)
                    else:
                        _ax_s.text(0.5, 0.5, 'no response data',
                                   ha='center', va='center',
                                   transform=_ax_s.transAxes,
                                   fontsize=7, color='grey')
                        _ax_s.set_xticks([])
                        _ax_s.set_yticks([])
                    _ax_s.set_ylabel(
                        f'{spatial_label}\n[{_clabel}]' if _ci == 0
                        else f'[{_clabel}]', fontsize=7)

            # Stats blurb under the column (red and red_corr only),
            # anchored to the bottom-most map (spatial when present, else
            # the bottom heatmap).
            # ----------
            if stats_text:
                _anchor = _last_spatial_ax if _last_spatial_ax is not None \
                    else _last_heat_ax
                if _anchor is not None:
                    _anchor.text(0.0, -0.18, stats_text,
                                 transform=_anchor.transAxes,
                                 ha='left', va='top',
                                 fontsize=7, color='k',
                                 family='monospace')

            return ax_trace

        # Unit suffix for the raw red/grn axis / colour-bar labels:
        # dF/F0 (%) when dff_sig per-trial baseline normalisation is on,
        # ITI z-score otherwise.
        _unit_suffix = '(dF/F %)' if dff_sig else '(z)'

        # Shared colour scales across the red / grn / (corr) columns so the
        # sector heatmaps (z-score or dF/F0) and the ∫-response spatial maps
        # are directly comparable by eye — one common scale per metric,
        # not a per-column autoscale. The sector heatmap uses a symmetric
        # ±vmag from the joint 98th percentile; the spatial map uses the
        # joint 2nd/98th percentile range.
        # ----------
        def _shared_symmetric_vmag(mats):
            _vals = [np.asarray(_m, dtype=np.float64).ravel()
                     for _m in mats if _m is not None]
            if not _vals:
                return None
            _cat = np.concatenate(_vals)
            _cat = _cat[np.isfinite(_cat)]
            if _cat.size == 0:
                return None
            _lo = float(np.percentile(_cat, 2))
            _hi = float(np.percentile(_cat, 98))
            _vmag = max(abs(_lo), abs(_hi))
            return _vmag if (np.isfinite(_vmag) and _vmag > 0) else None

        def _shared_range(mats):
            _vals = [np.asarray(_m, dtype=np.float64).ravel()
                     for _m in mats if _m is not None]
            if not _vals:
                return None, None
            _cat = np.concatenate(_vals)
            _cat = _cat[np.isfinite(_cat)]
            if _cat.size == 0:
                return None, None
            _lo = float(np.percentile(_cat, 2))
            _hi = float(np.percentile(_cat, 98))
            if _hi <= _lo:
                _hi = _lo + 1e-9
            return _lo, _hi

        # Gather every per-trial-type sector heatmap / spatial map across
        # all channels so the shared colour scales make the duplicated
        # rows comparable both within and across columns.
        _heat_mats = []
        _resp_mats = []
        for _cd in (_cond_sec_r, _cond_sec_g, _cond_sec_corr):
            for _avg, _grid in _cd.values():
                if _avg is not None:
                    _heat_mats.append(_avg)
                if _grid is not None:
                    _resp_mats.append(_grid)
        _heat_vmag = _shared_symmetric_vmag(_heat_mats)
        _spatial_vlo, _spatial_vhi = _shared_range(_resp_mats)

        # Leftmost column: raw red fluorescence (dF/F0 if dff=True,
        # else ITI z-score). Sectors sorted with the same pooled ordering;
        # heatmaps / spatial maps duplicated per base trial-type. The
        # signal-vs-control stats blurb sits under whichever of the two
        # raw columns is the functional channel.
        _ax_trace_red = _draw_column(
            col=0,
            trace_z=_wf_red,
            cond_sector=_cond_sec_r,
            trace_label=f'whole-frame\nred {_unit_suffix}',
            heat_label=(f'sector\nred {_unit_suffix}\n'
                        f'(1..{n_sec_total})'),
            spatial_label='sector map\nred',
            trace_color=sns.xkcd_rgb['bright red'],
            draw_spatial=True,
            stats_text=_stats_sig if _sig_is_red else None,
            heat_vmag=_heat_vmag,
            spatial_vlo=_spatial_vlo, spatial_vhi=_spatial_vhi)

        # Second column: raw green fluorescence. Same ordering as red,
        # with its own spatial map (Greens cmap) in the bottom row.
        _ax_trace_grn = _draw_column(
            col=1,
            trace_z=_wf_grn,
            cond_sector=_cond_sec_g,
            trace_label=f'whole-frame\ngrn {_unit_suffix}',
            heat_label=(f'sector\ngrn {_unit_suffix}\n'
                        f'(1..{n_sec_total})'),
            spatial_label='sector map\ngrn',
            trace_color=sns.xkcd_rgb['forest green'],
            draw_spatial=True,
            spatial_cmap='Greens',
            stats_text=None if _sig_is_red else _stats_sig,
            heat_vmag=_heat_vmag,
            spatial_vlo=_spatial_vlo, spatial_vhi=_spatial_vhi)

        _ax_trace_corr = None
        if _has_corr:
            # Third column (only when regression-corrected data exist):
            # corrected real channel. Shown z-sorted (ITI z-score) by
            # default; under dff_after_correction it is the fluorescence-
            # units reconstruction rendered as per-trial dF/F0 (%)
            # (dff_override=_corr_dff).
            _corr_unit = '(dF/F %)' if (_corr_dff or _corr_predff) else '(z)'
            _ax_trace_corr = _draw_column(
                col=2,
                trace_z=_wf_corr,
                cond_sector=_cond_sec_corr,
                trace_label=(f'whole-frame\n{_corr_key} '
                             f'{_corr_unit}'),
                heat_label=(f'sector\n{_corr_key} {_corr_unit}\n'
                            f'(1..{sector_avg_corr.shape[0]})'),
                spatial_label=f'sector map\n{_corr_key}',
                trace_color=sns.xkcd_rgb.get(
                    'bright orange', sns.xkcd_rgb['orange']),
                draw_spatial=True,
                dff_override=_corr_dff,
                stats_text=_stats_corr,
                heat_vmag=_heat_vmag,
                spatial_vlo=_spatial_vlo, spatial_vhi=_spatial_vhi)

        # Red, green, and (when present) corrected whole-frame average
        # traces share one y-axis range (union of all autoscaled limits)
        # so their amplitudes are directly comparable by eye.
        # ----------
        _rg_axes = [_ax_trace_red, _ax_trace_grn]
        if _ax_trace_corr is not None:
            _rg_axes.append(_ax_trace_corr)
        _rg_lo = min(_ax.get_ylim()[0] for _ax in _rg_axes)
        _rg_hi = max(_ax.get_ylim()[1] for _ax in _rg_axes)
        for _ax in _rg_axes:
            _ax.set_ylim(_rg_lo, _rg_hi)

        # Narrow rightmost column: per-sector trial reliability as a
        # single-column heatmap (red cmap), aligned with the sector
        # rows of the heatmaps to its left.
        # ----------
        ax_rel = fig.add_subplot(spec[1, _ncols])
        _rel_col = sector_reliability.reshape(-1, 1)
        if np.any(np.isfinite(_rel_col)):
            _vlo = float(np.nanpercentile(_rel_col, 2))
            _vhi = float(np.nanpercentile(_rel_col, 98))
            if _vhi <= _vlo:
                _vhi = _vlo + 1e-9
            _x_rel = np.array([0.0, 1.0])
            _y_rel = np.arange(_rel_col.shape[0] + 1)
            _im_rel = ax_rel.pcolormesh(
                _x_rel, _y_rel, _rel_col,
                cmap='Reds', vmin=_vlo, vmax=_vhi,
                shading='flat')
            ax_rel.set_ylim(_rel_col.shape[0], 0)
            _cax = ax_rel.inset_axes([1.4, 0, 0.35, 1])
            fig.colorbar(_im_rel, cax=_cax)
            _cax.tick_params(labelsize=7)
            _cax.set_ylabel('avg trial R', fontsize=7)
        else:
            ax_rel.text(0.5, 0.5, 'n/a', ha='center', va='center',
                        transform=ax_rel.transAxes,
                        fontsize=8, color='grey')
        ax_rel.set_xticks([])
        ax_rel.set_yticks([])
        ax_rel.set_title('trial\nreliability', fontsize=8)
        for _side in ('top', 'right', 'left', 'bottom'):
            ax_rel.spines[_side].set_visible(False)

        # Shared trial-type legend, drawn once at the top: each base
        # trial-type's linestyle annotated with its reward probability and
        # reward volume (the labels '0' / '0.5' / '1' etc. carry the same
        # linestyles used in every column's trace overlay).
        # ----------
        _tt_handles, _tt_labels = [], []
        for _ci, (_clabel, _cinds) in enumerate(_resp_conds):
            _ls = _cond_linestyles[_ci % len(_cond_linestyles)]
            _p_rew, _v_rew = self._qc_cond_rew_params(_cinds)
            _p_str = 'n/a' if not np.isfinite(_p_rew) else f'{_p_rew:.2g}'
            _v_str = 'n/a' if not np.isfinite(_v_rew) else f'{_v_rew:.2g}'
            _tt_handles.append(plt.Line2D([0], [0], color='0.2',
                                          linestyle=_ls, linewidth=1.6))
            _tt_labels.append(
                f'{_clabel}:  p(rew)={_p_str}, vol_rew={_v_str}')
        fig.legend(_tt_handles, _tt_labels, loc='upper center',
                   bbox_to_anchor=(0.5, 0.95), ncol=min(_n_rcond, 4),
                   fontsize=8, frameon=False,
                   title='trial-type (linestyle)', title_fontsize=8)

        _title = (f'{self.path.animal} {self.path.date} '
                  f'{self.path.beh_folder} '
                  f'— QC (stimulus-aligned GRAB)')
        # Append correction method + fit_mode so the figure caption
        # itself records how the red_corr column was produced.
        _cs_kw = getattr(self.qc, 'correct_signal_kwargs', None) or {}
        _cs_method = _cs_kw.get('method', None)
        if _cs_method == 'full_regress':
            _fm = _cs_kw.get('fit_mode', 'global')
            _title = _title + f'  [corr: {_cs_method}, fit={_fm}]'
        elif _cs_method is not None:
            _title = _title + f'  [corr: {_cs_method}]'
        # Sort label may carry a trial-type tag ('red_corr @ 0.5'); escape
        # underscores and spaces for the mathbf caption.
        _sort_lbl_esc = _sort_label.replace('_', r'\_').replace(' ', r'\ ')
        _title = (_title + r'  $\mathbf{(sectors\ sorted\ by\ '
                  + _sort_lbl_esc + r')}$')
        fig.suptitle(_title, fontsize=10, y=0.998)
        fig.tight_layout(rect=[0, 0, 1, 0.91])

        # Filename tag so the dff-after-correction variant does not
        # overwrite the z-scored one.
        _dffcorr_suffix = '_dffcorr' if _corr_dff else ''
        if save:
            _ch_suffix = f'_ch={channel}' if channel is not None else ''
            _cs_suffix = self._qc_corrsig_suffix()
            _fname = (f'{self.path.animal}_{self.path.date}_'
                      f'{self.path.beh_folder}'
                      f'_qc_ratio_event_avg{_ch_suffix}'
                      f'{_cs_suffix}{_dffcorr_suffix}.pdf')
            fig.savefig(os.path.join(str(self.folder.figs), _fname))

        # Save the response-mean summary data as a single .npy (dict
        # pickled with allow_pickle=True). Stores whole-frame mean ± SEM
        # and per-sector stim-aligned averages for each plotted channel,
        # plus the sector ordering and time vectors used by the figure.
        # ----------
        if save_npy:
            def _wf_mean_sem(trace, dff_override=None):
                _snips = _stim_snips(trace, dff_override=dff_override)
                if not _snips:
                    return None, None
                _arr = np.asarray(_snips, dtype=np.float64)
                # nan-aware: per-trial dF/F snippets are NaN outside their
                # trial windows; average across trials ignoring the gaps.
                with np.errstate(invalid='ignore'):
                    _n_fin = np.sum(np.isfinite(_arr), axis=0)
                    _mn = np.nanmean(_arr, axis=0)
                    _sm = np.nanstd(_arr, axis=0) / np.sqrt(
                        np.maximum(_n_fin, 1))
                _mn[_n_fin == 0] = np.nan
                _sm[_n_fin == 0] = np.nan
                return _mn, _sm

            _r_mn, _r_sm = _wf_mean_sem(_wf_red)
            _g_mn, _g_sm = _wf_mean_sem(_wf_grn)
            _ratio_mn, _ratio_sm = _wf_mean_sem(_wf_ratio)
            _corr_mn = _corr_sm = None
            if _has_corr:
                # corr column is ITI z-sorted unless dff_after_correction
                # made it the per-trial dF/F0 (%) reconstruction.
                _corr_mn, _corr_sm = _wf_mean_sem(_wf_corr,
                                                  dff_override=_corr_dff)

            # Stat keys name the two channels actually compared, so a
            # green-functional run reads 'grn_vs_red'. For the default
            # red-functional run these are the historical
            # 'red_vs_grn' / '{corr}_vs_grn' keys.
            _k_sig = f'{_real_ch}_vs_{_static_ch}'
            _k_corr = f'{_corr_key}_vs_{_static_ch}' if _has_corr else None
            _stats = {
                'whole_frame_pearson': {
                    _k_sig: {'r': _pear_rg_r,
                             'p': _pear_rg_p},
                },
                'sector_pearson': {
                    _k_sig: {'r': _pear_sec_rg_r,
                             'p': _pear_sec_rg_p,
                             'null': np.asarray(_pear_sec_rg_null)},
                },
                'sector_spearman': {
                    _k_sig: {'rho': _spear_rg_rho,
                             'p': _spear_rg_p,
                             'null': np.asarray(_spear_rg_null)},
                },
                'sector_response_maps': {
                    'red': np.asarray(_resp_r),
                    'grn': np.asarray(_resp_g),
                },
            }
            if _has_corr:
                _stats['whole_frame_pearson'][_k_corr] = {
                    'r': _pear_cg_r, 'p': _pear_cg_p}
                _stats['sector_pearson'][_k_corr] = {
                    'r': _pear_sec_cg_r, 'p': _pear_sec_cg_p,
                    'null': np.asarray(_pear_sec_cg_null)}
                _stats['sector_spearman'][_k_corr] = {
                    'rho': _spear_cg_rho, 'p': _spear_cg_p,
                    'null': np.asarray(_spear_cg_null)}
                _stats['sector_response_maps'][_corr_key] = \
                    np.asarray(_resp_corr)

            _cs_kw_save = (
                getattr(self.qc, 'correct_signal_kwargs', None) or {})
            _summary = {
                'mode': 'dff_sig' if dff_sig else 'iti_zscore',
                'signal_ch': _real_ch,
                'control_ch': _static_ch,
                'dff_after_correction': bool(_corr_dff),
                'correct_signal_method': _cs_kw_save.get('method', None),
                'correct_signal_fit_mode':
                    _cs_kw_save.get('fit_mode', None),
                't_rel': _t_rel,
                't_rel_sec': t_rel_sec,
                'rew_t_rel': _rew_t_rel,
                'n_sectors': int(self.qc.n_sectors),
                'sector_order': np.asarray(_order),
                'sector_reliability': np.asarray(sector_reliability),
                'whole_frame': {
                    'red': {'mean': _r_mn, 'sem': _r_sm},
                    'grn': {'mean': _g_mn, 'sem': _g_sm},
                    'ratio': {'mean': _ratio_mn, 'sem': _ratio_sm},
                },
                'sectors': {
                    'red': np.asarray(sector_avg_r),
                    'grn': np.asarray(sector_avg_g),
                    'ratio': np.asarray(sector_avg),
                },
                'stats': _stats,
            }
            if _has_corr:
                # Save the corrected channel under its actual key name
                # (e.g., 'red_corr') in both whole-frame and per-sector
                # dicts. This is the canonical access path for downstream
                # grand averages (see grand_avg_response_mean).
                _summary['whole_frame'][_corr_key] = {
                    'mean': _corr_mn, 'sem': _corr_sm}
                _summary['sectors'][_corr_key] = np.asarray(sector_avg_corr)
                _summary['corr_key'] = _corr_key

            # Per-hemisphere encoding when split_lr=True. Frame_f holds
            # one half-FOV trace per channel (`{ch}_left`, `{ch}_right`);
            # sector_f is full-grid and is split by row (top n_top rows
            # → right, remaining rows → left, matching the figure
            # convention: bottom=left, top=right).
            # ----------
            _split_lr = bool(getattr(self.qc, 'split_lr', False))
            _summary['split_lr'] = _split_lr
            if _split_lr:
                _ns = int(self.qc.n_sectors)
                _n_top = (_ns // 2) * _ns

                for _side in ('left', 'right'):
                    _ff_r_h = self.qc.frame_f.get(f'red_{_side}')
                    _ff_g_h = self.qc.frame_f.get(f'grn_{_side}')
                    if _ff_r_h is None or _ff_g_h is None:
                        continue
                    _ff_r_h = np.asarray(_ff_r_h, dtype=np.float64)
                    _ff_g_h = np.asarray(_ff_g_h, dtype=np.float64)
                    # Ratio is signal / control, matching the full-FOV
                    # _ratio_ff above (so it inverts for a grn-functional
                    # recording).
                    _ff_sig_h = _ff_r_h if _sig_is_red else _ff_g_h
                    _ff_ctrl_h = _ff_g_h if _sig_is_red else _ff_r_h
                    with np.errstate(divide='ignore', invalid='ignore'):
                        _ratio_h = _ff_sig_h / (_ff_ctrl_h + 1e-9)
                    if dff_sig:
                        _r_in, _g_in, _rt_in = _ff_r_h, _ff_g_h, _ratio_h
                    else:
                        _r_in = _zscore_iti(_ff_r_h, t)
                        _g_in = _zscore_iti(_ff_g_h, t)
                        _rt_in = _zscore_iti(_ratio_h, t)
                    for _lbl, _trace in [(f'red_{_side}', _r_in),
                                         (f'grn_{_side}', _g_in),
                                         (f'ratio_{_side}', _rt_in)]:
                        _mn, _sm = _wf_mean_sem(_trace)
                        _summary['whole_frame'][_lbl] = {
                            'mean': _mn, 'sem': _sm}
                    if _has_corr:
                        _ff_c_h = self.qc.frame_f.get(
                            f'{_corr_key}_{_side}')
                        if _ff_c_h is not None:
                            if _corr_dff:
                                # Reconstruct the per-hemisphere
                                # fluorescence-units corrected trace from
                                # this recording's control / signal pair,
                                # so the saved mean matches the figure's
                                # dF/F0 column.
                                try:
                                    _c_in = self._martianova_flu_1d(
                                        _ff_ctrl_h, _ff_sig_h, _kw_1d)
                                except ValueError:
                                    _c_in = _zscore_iti(np.asarray(
                                        _ff_c_h, dtype=np.float64), t)
                            else:
                                _c_in = _zscore_iti(np.asarray(
                                    _ff_c_h, dtype=np.float64), t)
                            _mn, _sm = _wf_mean_sem(
                                _c_in, dff_override=_corr_dff)
                            _summary['whole_frame'][
                                f'{_corr_key}_{_side}'] = {
                                'mean': _mn, 'sem': _sm}

                # Per-sector matrices: undo the global sort to recover
                # the natural grid row-major order, then slice by row.
                # Same global ordering is preserved within each
                # hemisphere via _order's relative permutation.
                # ----------
                _inv_order = np.argsort(_order)
                _sec_orig = {
                    'red': np.asarray(sector_avg_r)[_inv_order],
                    'grn': np.asarray(sector_avg_g)[_inv_order],
                    'ratio': np.asarray(sector_avg)[_inv_order],
                }
                if _has_corr and sector_avg_corr is not None:
                    _sec_orig[_corr_key] = (
                        np.asarray(sector_avg_corr)[_inv_order])
                for _ch_key, _mat in _sec_orig.items():
                    _summary['sectors'][f'{_ch_key}_right'] = \
                        np.asarray(_mat[:_n_top])
                    _summary['sectors'][f'{_ch_key}_left'] = \
                        np.asarray(_mat[_n_top:])

            _ch_suffix = f'_ch={channel}' if channel is not None else ''
            _cs_suffix = self._qc_corrsig_suffix()
            _dff_suffix = '_dffsig' if dff_sig else ''
            _npy_name = (f'{self.path.animal}_{self.path.date}_'
                         f'{self.path.beh_folder}'
                         f'_qc_response_mean'
                         f'{_ch_suffix}{_cs_suffix}{_dff_suffix}'
                         f'{_dffcorr_suffix}.npy')
            np.save(os.path.join(str(self.folder.data), _npy_name),
                    _summary, allow_pickle=True)

        return fig

    def _plt_qc_response_mean_cells(self, t, channel=None,
                                    t_pre=2.0, t_post=6.0,
                                    figsize=None, save=True):
        """Stim-aligned per-cell donut-ring dF/F responses.

        Companion to _plt_qc_response_mean's sector heatmap, but rows are
        individual somata rather than spatial sectors: each row is a
        cell's donut-ring corrected dF/F trace (self._corrected_info.
        cell_traces, produced by signal_correction.correct_pixel_
        spatial_avg with segmentation_type='cells'). Only meaningful
        when plt_qc was called with correct_signal_kwargs={'method':
        'pixel_spatial_subtr', 'segmentation_type': 'cells', ...}; returns
        None otherwise.

        Parameters
        ----------
        t : np.ndarray
            QC time vector from self.qc.t.
        channel : str or None
            Channel label for the figure filename suffix.
        t_pre, t_post : float
            Window around each stim onset (seconds).
        figsize : tuple or None
            Figure size in inches. Defaults to (9, 8).
        save : bool
            If True, saves a pdf to self.folder.figs.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
        """
        _cs_kw = getattr(self.qc, 'correct_signal_kwargs', None) or {}
        if _cs_kw.get('method') != 'pixel_spatial_subtr':
            return None
        _info = getattr(self, '_corrected_info', None)
        if _info is None or getattr(
                _info, 'segmentation_type', None) != 'cells':
            return None
        cell_traces = getattr(_info, 'cell_traces', None)
        cell_centroids = getattr(_info, 'cell_centroids', None)
        if (cell_traces is None or cell_centroids is None
                or cell_traces.shape[0] == 0):
            return None

        n_cells, n_t = cell_traces.shape
        _frame_range = getattr(self, '_cs_frame_range', None)
        if (_frame_range is not None
                and (_frame_range[1] - _frame_range[0]) == n_t):
            t_cells = np.asarray(self.qc.t)[_frame_range[0]:_frame_range[1]]
        else:
            t_cells = np.asarray(self.qc.t)[:n_t]

        _stim_t = self._qc_stim_t()

        _dt = float(np.median(np.diff(t_cells)))
        _n_pre = int(round(t_pre / _dt))
        _n_post = int(round(t_post / _dt))
        _n_win = _n_pre + _n_post + 1
        _t_rel = np.linspace(-t_pre, t_post, _n_win)

        # Per-cell stim-aligned average (native dF/F fraction -> %).
        # ----------
        resp = np.full((n_cells, _n_win), np.nan, dtype=np.float64)
        for _i in range(n_cells):
            _, _m, _ = self._compute_event_avg(
                cell_traces[_i].astype(np.float64), t_cells, _stim_t,
                t_pre=t_pre, t_post=t_post, pct=False)
            if _m is not None:
                resp[_i] = _m
        resp *= 100.0

        # Median stim -> reward latency (reward marker + integration
        # window for sorting / the spatial map), same convention as
        # _plt_qc_response_mean.
        # ----------
        _rew_t_all = np.asarray(getattr(self.beh.rew, 't', []),
                                dtype=object).ravel()
        _te_lim = getattr(self, 'trial_end', None)
        if _te_lim is not None:
            _rew_t_all = _rew_t_all[:int(_te_lim) + 1]
        _lat = []
        for _i in range(min(len(_rew_t_all), len(_stim_t))):
            _r = _rew_t_all[_i]
            if _r is None:
                continue
            try:
                _rf = float(_r)
            except (TypeError, ValueError):
                continue
            if np.isfinite(_rf):
                _lat.append(_rf - float(_stim_t[_i]))
        _rew_t_rel = float(np.median(_lat)) if _lat else None

        _int_hi = _rew_t_rel if _rew_t_rel is not None else t_post
        _int_mask = (_t_rel >= 0.0) & (_t_rel <= _int_hi)
        with np.errstate(invalid='ignore'):
            _int_resp = (np.nanmean(resp[:, _int_mask], axis=1)
                        if np.any(_int_mask) else np.nanmean(resp, axis=1))

        _order = np.argsort(-np.nan_to_num(_int_resp, nan=-np.inf))
        resp_sorted = resp[_order]

        if figsize is None:
            figsize = (13.0, 8.0)
        fig = plt.figure(figsize=figsize)
        spec = gridspec.GridSpec(nrows=3, ncols=2, figure=fig,
                                 height_ratios=[1.4, 2.2, 2.2],
                                 width_ratios=[1.0, 1.6],
                                 hspace=0.45, wspace=0.35)

        # --- left column: mean functional-channel image with cell soma
        #     (cyan) and donut-ring (gold) ROI outlines overlaid — the
        #     same segmentation self._corrected_info.cell_labels /
        #     cell_ring_labels used to build the ring traces above ---
        ax_img = fig.add_subplot(spec[:, 0])
        _real_flu = _cs_kw.get('real_flu', 'red')
        _rec_img = getattr(self, f'rec_{_real_flu}', None)
        if _rec_img is None:
            _rec_img = getattr(self, f'rec_{_real_flu}_original', None)
        if _rec_img is not None:
            _nk = int(getattr(self.qc, '_n_keep', _rec_img.shape[0]))
            _mean_img = self._mean_image(_rec_img[:_nk], stride=4,
                                         chunk_size=500)
            _lo = float(np.nanpercentile(_mean_img, 1.0))
            _hi = float(np.nanpercentile(_mean_img, 99.5))
            if not np.isfinite(_hi) or _hi <= _lo:
                _hi = _lo + 1e-9
            ax_img.imshow(_mean_img, cmap=self._channel_img_cmap(_real_flu),
                         vmin=_lo, vmax=_hi)
            _cell_labels = getattr(_info, 'cell_labels', None)
            _ring_labels = getattr(_info, 'cell_ring_labels', None)
            if _ring_labels is not None:
                ax_img.contour(_ring_labels > 0, levels=[0.5],
                               colors='#ffd700', linewidths=0.4)
            if _cell_labels is not None:
                ax_img.contour(_cell_labels > 0, levels=[0.5],
                               colors='#00e5ff', linewidths=0.5)
            ax_img.scatter(cell_centroids[:, 1], cell_centroids[:, 0],
                          s=4, color='#ffffff', linewidths=0)
            ax_img.set_xlim(0, _mean_img.shape[1])
            ax_img.set_ylim(_mean_img.shape[0], 0)
        else:
            ax_img.text(0.5, 0.5, f'no rec_{_real_flu} available',
                       ha='center', va='center',
                       transform=ax_img.transAxes, fontsize=8, color='grey')
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_aspect('equal')
        ax_img.set_title(f'mean {_real_flu}\ncell (cyan) / donut (gold) '
                         f'ROIs (n={n_cells})', fontsize=9)

        # --- row 0: mean +/- SEM across all cells ---
        ax_trace = fig.add_subplot(spec[0, 1])
        with np.errstate(invalid='ignore'):
            _n_fin = np.sum(np.isfinite(resp), axis=0)
            _mean = np.nanmean(resp, axis=0)
            _sem = np.nanstd(resp, axis=0) / np.sqrt(np.maximum(_n_fin, 1))
        _mean[_n_fin == 0] = np.nan
        _sem[_n_fin == 0] = np.nan
        _color = sns.xkcd_rgb.get('bright orange', sns.xkcd_rgb['orange'])
        ax_trace.fill_between(_t_rel, _mean - _sem, _mean + _sem,
                              color=_color, alpha=0.2, linewidth=0)
        ax_trace.plot(_t_rel, _mean, color=_color, linewidth=1.2)
        ax_trace.axvline(x=0, color='k', linewidth=0.8,
                         linestyle='--', alpha=0.7)
        if _rew_t_rel is not None:
            ax_trace.axvline(x=_rew_t_rel, color=sns.xkcd_rgb['bright blue'],
                             linewidth=0.8, linestyle='--', alpha=0.8)
        ax_trace.axhline(y=0, color='k', linewidth=0.4, linestyle=':')
        ax_trace.set_ylabel(f'mean cell\ndF/F (%)\n(n={n_cells})',
                            fontsize=8)
        ax_trace.set_xlim(_t_rel[0], _t_rel[-1])
        for _side in ('top', 'right'):
            ax_trace.spines[_side].set_visible(False)
        plt.setp(ax_trace.get_xticklabels(), visible=False)

        # --- row 1: per-cell heatmap, sorted by integrated
        #     stim -> rew response ---
        ax_heat = fig.add_subplot(spec[1, 1], sharex=ax_trace)
        _finite = resp_sorted[np.isfinite(resp_sorted)]
        _vmag = (float(np.percentile(np.abs(_finite), 98))
                if _finite.size else 1.0)
        if not (np.isfinite(_vmag) and _vmag > 0):
            _vmag = 1.0
        _x_edges = np.linspace(_t_rel[0], _t_rel[-1],
                               resp_sorted.shape[1] + 1)
        _y_edges = np.arange(resp_sorted.shape[0] + 1)
        _im = ax_heat.pcolormesh(_x_edges, _y_edges, resp_sorted,
                                 cmap='gray', vmin=-_vmag, vmax=_vmag,
                                 shading='flat')
        ax_heat.set_ylim(resp_sorted.shape[0], 0)
        ax_heat.set_yticks([0, resp_sorted.shape[0]])
        ax_heat.axvline(x=0, color=sns.xkcd_rgb['bright red'],
                        linewidth=0.8, linestyle='--', alpha=0.8)
        if _rew_t_rel is not None:
            ax_heat.axvline(x=_rew_t_rel, color=sns.xkcd_rgb['bright blue'],
                            linewidth=0.8, linestyle='--', alpha=0.8)
        ax_heat.set_ylabel(f'cell (1..{n_cells})\nsorted by\n∫ response',
                           fontsize=8)
        ax_heat.set_xlabel('time from stim (s)', fontsize=8)
        for _side in ('top', 'right'):
            ax_heat.spines[_side].set_visible(False)
        _cax = ax_heat.inset_axes([1.015, 0, 0.012, 1])
        fig.colorbar(_im, cax=_cax)
        _cax.tick_params(labelsize=7)
        _cax.set_ylabel('dF/F (%)', fontsize=7)

        # --- row 2: spatial map of cell centroids, coloured by
        #     integrated stim -> rew response (same 'Reds' convention as
        #     the corrected-channel sector spatial map) ---
        ax_sp = fig.add_subplot(spec[2, 1])
        if np.any(np.isfinite(_int_resp)):
            _vlo = float(np.nanpercentile(_int_resp, 2))
            _vhi = float(np.nanpercentile(_int_resp, 98))
        else:
            _vlo, _vhi = -1.0, 1.0
        if _vhi <= _vlo:
            _vhi = _vlo + 1e-9
        _sc = ax_sp.scatter(cell_centroids[:, 1], cell_centroids[:, 0],
                            c=_int_resp, cmap='Reds', vmin=_vlo, vmax=_vhi,
                            s=25, edgecolors='k', linewidths=0.3)
        ax_sp.set_aspect('equal')
        ax_sp.invert_yaxis()
        ax_sp.set_xlabel('x (px)', fontsize=8)
        ax_sp.set_ylabel('y (px)', fontsize=8)
        _cax_sp = ax_sp.inset_axes([1.03, 0, 0.03, 1])
        fig.colorbar(_sc, cax=_cax_sp)
        _cax_sp.tick_params(labelsize=7)
        _cax_sp.set_ylabel('∫ response\n(stim→rew)\ndF/F (%)', fontsize=7)

        _title = (f'{self.path.animal} {self.path.date} '
                 f'{self.path.beh_folder} '
                 f'— QC (per-cell donut-ring stim-evoked)')
        fig.suptitle(_title, fontsize=10, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        if save:
            _ch_suffix = f'_ch={channel}' if channel is not None else ''
            _cs_suffix = self._qc_corrsig_suffix()
            _fname = (f'{self.path.animal}_{self.path.date}_'
                     f'{self.path.beh_folder}'
                     f'_qc_cells_event_avg{_ch_suffix}{_cs_suffix}.pdf')
            fig.savefig(os.path.join(str(self.folder.figs), _fname))

        return fig

    def _plt_qc_lick_aligned(self, t, channel=None,
                             t_pre=2.0, t_post=4.0,
                             bout_max_ili=0.5, bout_min_licks=4,
                             dff_sig=True,
                             figsize=None, save=True, save_pickle=True):
        """Lick-bout-onset-aligned GRAB averages (dual-colour only).

        Companion to ``_plt_qc_response_mean``, but aligned to the onset
        of rhythmic lick bouts instead of stimulus onset. A lick bout is
        a run of more than three consecutive licks whose inter-lick
        intervals are all <= ``bout_max_ili`` seconds (see
        ``_detect_lick_bouts``).

        Columns: raw red, raw green, and — when correct_signal produced a
        corrected real channel — corrected red. Each column shows the
        whole-frame mean ± SEM across bouts (row 0), a per-sector
        bout-averaged heatmap (row 1), and a spatial map of the
        integrated post-onset response (row 2).

        Normalisation mirrors the stim-aligned figure: with ``dff_sig``
        (default), the raw red and green columns are shown as per-trial
        dF/F0 (%) using each bout's pre-onset window as F0 — a *local*
        baseline that is immune to slow photobleaching/drift, so the
        three columns are amplitude-comparable. The corrected real channel
        is shown ITI z-scored by default (the z-scored zdFF carries no
        fluorescence scale); when ``qc.dff_after_correction`` is set it is
        reconstructed in fluorescence units (Approach A) and shown as
        per-trial dF/F0 (%) like the raw columns. With ``dff_sig=False``
        the raw red/grn columns use ITI z-score instead.

        Returns None for single-channel recordings or when no lick bouts
        are found.

        Parameters
        ----------
        t : np.ndarray
            QC time vector from self.qc.t.
        channel : str or None
            Channel label for the figure filename suffix.
        t_pre, t_post : float
            Window around each bout onset (seconds).
        bout_max_ili : float
            Maximum inter-lick interval within a bout (s). Default 0.5.
        bout_min_licks : int
            Minimum licks per bout. Default 4 (more than three licks).
        dff_sig : bool
            If True (default), raw red and green are converted to
            per-trial dF/F0 (%) against each bout's pre-onset baseline;
            the corrected channel stays ITI z-scored unless
            ``qc.dff_after_correction`` is set (see plt_qc), in which case
            it is shown as per-trial dF/F0 (%) from the fluorescence-units
            Approach-A reconstruction. If False, raw red/grn use ITI
            z-score. Default True (matches the stim-aligned figure's
            dff_response_mean_fig default).
        figsize : tuple or None
            Figure size; auto-sized by column count when None.
        save : bool
            If True, save a pdf to self.folder.figs.
        save_pickle : bool
            If True (and save), also gzip-pickle the figure beside the
            pdf for later interactive inspection via load_qc_fig.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
        """
        _is_dual = (isinstance(self.qc.sector_f, dict)
                    and 'red' in self.qc.sector_f
                    and 'grn' in self.qc.sector_f)
        if not _is_dual:
            return None

        # Lick-bout onsets (from licks already filtered to the QC range).
        # ----------
        _lick_t = self._qc_lick_t(getattr(self.beh.lick, 't_raw', []))
        _bout_t = self._detect_lick_bouts(
            _lick_t, max_ili=bout_max_ili, min_licks=bout_min_licks)
        if _bout_t.size == 0:
            print('\tno lick bouts detected; skipping lick-aligned GRAB.')
            return None
        print(f'\tdetected {_bout_t.size} lick bouts '
              f'(>{bout_min_licks - 1} licks, ILI<={bout_max_ili}s).')

        # Whole-frame red & grn (reconstruct from halves if split_lr).
        # ----------
        _ff_r = self.qc.frame_f.get('red')
        _ff_g = self.qc.frame_f.get('grn')
        if _ff_r is None or _ff_g is None:
            _r_l = self.qc.frame_f.get('red_left')
            _r_r = self.qc.frame_f.get('red_right')
            _g_l = self.qc.frame_f.get('grn_left')
            _g_r = self.qc.frame_f.get('grn_right')
            if all(x is not None for x in (_r_l, _r_r, _g_l, _g_r)):
                _ff_r = 0.5 * (_r_l + _r_r)
                _ff_g = 0.5 * (_g_l + _g_r)
            else:
                return None
        _sec_r = self.qc.sector_f['red'].astype(np.float32)
        _sec_g = self.qc.sector_f['grn'].astype(np.float32)

        # Channel roles (see _plt_qc_response_mean): the functional (real)
        # channel is the QC's active channel, so channel='grn' makes green
        # the signal and red the control regressor.
        # ----------
        _real_ch, _static_ch = self._qc_flu_pair()
        _sig_is_red = (_real_ch != 'grn')
        _ff_sig = _ff_r if _sig_is_red else _ff_g
        _ff_ctrl = _ff_g if _sig_is_red else _ff_r
        _sec_sig = _sec_r if _sig_is_red else _sec_g
        _sec_ctrl = _sec_g if _sig_is_red else _sec_r

        # Corrected real channel (optional — needs both frame & sector).
        # ----------
        _corr_key = None
        for _k in self.qc.frame_f:
            if _k.endswith('_corr') and _k in self.qc.sector_f:
                _corr_key = _k
                break
        _ff_corr = None
        _sec_corr = None
        if _corr_key is not None:
            _ff_corr_raw = self.qc.frame_f.get(_corr_key)
            if _ff_corr_raw is None:
                _ll = self.qc.frame_f.get(f'{_corr_key}_left')
                _rr = self.qc.frame_f.get(f'{_corr_key}_right')
                if _ll is not None and _rr is not None:
                    _ff_corr_raw = 0.5 * (_ll + _rr)
            if _ff_corr_raw is not None:
                _ff_corr = np.asarray(_ff_corr_raw, dtype=np.float64)
                _sec_corr = self.qc.sector_f[_corr_key].astype(np.float32)
            else:
                _corr_key = None
        _has_corr = _corr_key is not None

        # ITI z-score helpers.
        # ----------
        def _iti_mu_sd(trace, t_vec):
            _mask = self._inter_trial_mask(t_vec)
            _base = np.asarray(trace)[_mask]
            _base = _base[np.isfinite(_base)]
            if _base.size < 2:
                return None, None
            _mu = float(np.mean(_base))
            _sd = float(np.std(_base))
            if _sd == 0 or not np.isfinite(_sd):
                return None, None
            return _mu, _sd

        def _zscore_iti(trace, t_vec):
            _mu, _sd = _iti_mu_sd(trace, t_vec)
            if _mu is None:
                return np.full_like(trace, np.nan, dtype=np.float64)
            return (np.asarray(trace, dtype=np.float64) - _mu) / _sd

        n_sec_total, n_compute = _sec_r.shape
        sec_t = np.linspace(t[0], t[-1], n_compute)
        _sec_iti_mask = self._inter_trial_mask(sec_t)

        def _zscore_sectors(sec_mat):
            _base = sec_mat[:, _sec_iti_mask]
            with np.errstate(invalid='ignore'):
                _mu = np.nanmean(_base, axis=1, keepdims=True)
                _sd = np.nanstd(_base, axis=1, keepdims=True)
            _z = np.full_like(sec_mat, np.nan, dtype=np.float64)
            _ok = np.isfinite(_sd[:, 0]) & (_sd[:, 0] > 0)
            if np.any(_ok):
                _z[_ok] = (sec_mat[_ok].astype(np.float64)
                           - _mu[_ok]) / _sd[_ok]
            return _z

        _ff_r_z = _zscore_iti(_ff_r, t)
        _ff_g_z = _zscore_iti(_ff_g, t)
        _sec_r_z = _zscore_sectors(_sec_r)
        _sec_g_z = _zscore_sectors(_sec_g)
        if _has_corr:
            # A native dF/F corrected channel (subtractive, e.g.
            # pixel_spatial_subtr — continuous for f0_mode='global' or
            # per-trial with NaN inter-trial gaps for 'per_trial') is passed
            # through scaled to dF/F (%). Otherwise z-score the whole-frame
            # trace and put every sector on the common whole-frame ITI scale
            # (see _plt_qc_response_mean for why).
            _corr_predff = getattr(self.qc, 'corr_native_dff', None) is not None
            if _corr_predff:
                _ff_corr_z = np.asarray(_ff_corr, dtype=np.float64) * 100.0
                _sec_corr_z = np.asarray(_sec_corr, dtype=np.float64) * 100.0
            else:
                _ff_corr_z = _zscore_iti(_ff_corr, t)
                _mu_wf, _sd_wf = _iti_mu_sd(_ff_corr, t)
                if _mu_wf is None:
                    _sec_corr_z = np.full_like(
                        _sec_corr, np.nan, dtype=np.float64)
                else:
                    _sec_corr_z = (np.asarray(_sec_corr, dtype=np.float64)
                                   - _mu_wf) / _sd_wf
        else:
            _corr_predff = False
            _ff_corr_z = None
            _sec_corr_z = None

        # Approach A (dff_after_correction): reconstruct fluorescence-units
        # corrected traces so the corrected column can be shown as per-trial
        # dF/F0 (%) — mirrors _plt_qc_response_mean. The 1-D pipeline is
        # re-run per region on the raw control and signal traces
        # (whichever channels carry those roles) and mapped back to F via
        # F_corr = σ2·zdFF + m2 + b2.
        # Only active for full_regress (other methods stay z-scored).
        # ----------
        _cs_kw = getattr(self.qc, 'correct_signal_kwargs', None) or {}
        _corr_method = _cs_kw.get('method', 'full_regress')
        _corr_dff = (_has_corr
                     and bool(getattr(self.qc, 'dff_after_correction', False))
                     and not _corr_predff
                     and _corr_method == 'full_regress')
        _ff_corr_flu = None
        _sec_corr_flu = None
        if _corr_dff:
            print('\tdff_after_correction: reconstructing fluorescence-'
                  'units corrected traces (Approach A, lick-aligned)...')
            _kw_1d = {_k: _cs_kw[_k] for _k in (
                'smooth_window', 'airpls_lam', 'airpls_porder',
                'airpls_max_iter', 'trim_initial', 'nn_slope',
                'beta_loss', 'beta_f_scale', 'beta_scale')
                if _k in _cs_kw}
            try:
                _ff_corr_flu = self._martianova_flu_1d(
                    _ff_ctrl, _ff_sig, _kw_1d)
            except ValueError:
                _ff_corr_flu = None
            _sec_corr_flu = np.full_like(_sec_sig, np.nan, dtype=np.float64)
            for _si in range(_sec_sig.shape[0]):
                try:
                    _sec_corr_flu[_si] = self._martianova_flu_1d(
                        _sec_ctrl[_si], _sec_sig[_si], _kw_1d)
                except ValueError:
                    pass
            if _ff_corr_flu is None:
                _corr_dff = False

        # Bout-aligned whole-frame snippets.
        # ----------
        _dt = float(np.median(np.diff(t)))
        _n_pre = int(round(t_pre / _dt))
        _n_post = int(round(t_post / _dt))
        _n_win = _n_pre + _n_post + 1
        _t_rel = np.linspace(-t_pre, t_post, _n_win)

        def _bout_snips(trace, dff_override=None):
            # When `dff` (resolved from dff_sig or dff_override) is True,
            # `trace` is a raw trace and each per-trial snippet is
            # converted to dF/F0 (%) using its pre-onset baseline window
            # (length _n_pre samples). Otherwise `trace` is assumed
            # pre-z-scored. dff_override forces the dF/F path (True) or
            # z-score path (False) regardless of dff_sig.
            _dff = dff_sig if dff_override is None else bool(dff_override)
            _out = []
            for _ev in _bout_t:
                _i = int(np.argmin(np.abs(t - _ev)))
                _i0 = _i - _n_pre
                _i1 = _i + _n_post + 1
                if _i0 < 0 or _i1 > len(trace):
                    continue
                _snip = np.asarray(trace[_i0:_i1], dtype=np.float64)
                # NaN-tolerant: a per-trial dF/F trace is NaN outside its
                # window, so keep snippets with finite coverage (gaps are
                # averaged out nan-aware across trials by the caller) and
                # drop only those with no usable baseline.
                if not np.any(np.isfinite(_snip)):
                    continue
                if _dff:
                    if _n_pre <= 0:
                        continue
                    with np.errstate(invalid='ignore'):
                        _f0 = float(np.nanmean(_snip[:_n_pre]))
                    if not np.isfinite(_f0) or _f0 == 0:
                        continue
                    _snip = (_snip - _f0) / _f0 * 100.0
                _out.append(_snip)
            return _out

        # Bout-aligned per-sector averages on the sector time vector.
        # ----------
        dt_sec = float(np.median(np.diff(sec_t)))
        n_pre = int(round(t_pre / dt_sec))
        n_post = int(round(t_post / dt_sec))
        n_win = n_pre + n_post + 1
        t_rel_sec = np.linspace(-t_pre, t_post, n_win)

        def _bout_aligned_sectors(sec_mat, dff_override=None):
            # Same dff resolution as _bout_snips, per sector. dff path
            # converts each per-trial snippet to dF/F0 (%) vs its own
            # pre-onset baseline; else uses the baseline-subtracted
            # event average.
            _dff = dff_sig if dff_override is None else bool(dff_override)
            _avg = np.full((sec_mat.shape[0], n_win), np.nan,
                           dtype=np.float32)
            if _dff:
                for _i in range(sec_mat.shape[0]):
                    _snips = []
                    for _ev in _bout_t:
                        _ind = int(np.argmin(np.abs(sec_t - _ev)))
                        _i0 = _ind - n_pre
                        _i1 = _ind + n_post + 1
                        if _i0 < 0 or _i1 > sec_mat.shape[1] or n_pre <= 0:
                            continue
                        _snip = np.asarray(sec_mat[_i, _i0:_i1],
                                           dtype=np.float64)
                        if not np.all(np.isfinite(_snip)):
                            continue
                        _f0 = float(np.mean(_snip[:n_pre]))
                        if not np.isfinite(_f0) or _f0 == 0:
                            continue
                        _snips.append((_snip - _f0) / _f0 * 100.0)
                    if _snips:
                        _avg[_i] = np.mean(_snips, axis=0)
            else:
                for _i in range(sec_mat.shape[0]):
                    _, _m, _ = self._compute_event_avg(
                        sec_mat[_i], sec_t, _bout_t,
                        t_pre=t_pre, t_post=t_post)
                    if _m is not None:
                        _avg[_i] = _m
            return _avg

        # Inputs to the snippet extractors: in dff_sig mode the raw red /
        # grn traces are passed and per-trial dF/F0 is applied inside the
        # helpers; otherwise the ITI-z-scored traces are passed. The
        # corrected channel always uses its z-scored input (dff_override
        # =False at draw/compute time).
        # ----------
        if dff_sig:
            _wf_red, _wf_grn = _ff_r, _ff_g
            _sec_red_in, _sec_grn_in = _sec_r, _sec_g
        else:
            _wf_red, _wf_grn = _ff_r_z, _ff_g_z
            _sec_red_in, _sec_grn_in = _sec_r_z, _sec_g_z

        # Corrected column: ITI z-sorted by default (dff_override=False);
        # under dff_after_correction the fluorescence-units reconstruction
        # is fed instead and each per-trial snippet is converted to dF/F0
        # (%) (dff_override=True).
        # ----------
        _wf_corr_in = _ff_corr_flu if _corr_dff else _ff_corr_z
        _sec_corr_in = _sec_corr_flu if _corr_dff else _sec_corr_z

        sector_avg_r = _bout_aligned_sectors(_sec_red_in)
        sector_avg_g = _bout_aligned_sectors(_sec_grn_in)
        sector_avg_corr = (_bout_aligned_sectors(_sec_corr_in,
                                                 dff_override=_corr_dff)
                           if _has_corr else None)

        # Integrated post-onset response per sector (0 → t_post window).
        # ----------
        _i0 = int(np.searchsorted(t_rel_sec, 0.0, side='left'))
        _i1 = sector_avg_r.shape[1]
        with np.errstate(invalid='ignore'):
            _resp_r = np.nansum(sector_avg_r[:, _i0:_i1], axis=1)
            _resp_g = np.nansum(sector_avg_g[:, _i0:_i1], axis=1)
            _resp_corr = (np.nansum(sector_avg_corr[:, _i0:_i1], axis=1)
                          if sector_avg_corr is not None else None)
            _peak_corr = (np.nanmax(sector_avg_corr[:, _i0:_i1], axis=1)
                          if sector_avg_corr is not None else None)
            _peak_sig = np.nanmax(
                (sector_avg_r if _sig_is_red else sector_avg_g)[:, _i0:_i1],
                axis=1)

        # Order sectors by peak post-onset response (corrected if avail,
        # else the raw functional channel).
        # ----------
        if _peak_corr is not None:
            _order_src = _peak_corr
            _sort_label = _corr_key
        else:
            _order_src = _peak_sig
            _sort_label = _real_ch
        _order = np.argsort(np.where(np.isfinite(_order_src),
                                     _order_src, -np.inf))[::-1]
        sector_avg_r = sector_avg_r[_order]
        sector_avg_g = sector_avg_g[_order]
        if sector_avg_corr is not None:
            sector_avg_corr = sector_avg_corr[_order]

        _ns = int(self.qc.n_sectors)
        _resp_grid_r = (_resp_r.reshape(_ns, _ns)
                        if _resp_r.size == _ns * _ns else None)
        _resp_grid_g = (_resp_g.reshape(_ns, _ns)
                        if _resp_g.size == _ns * _ns else None)
        _resp_grid_corr = (_resp_corr.reshape(_ns, _ns)
                           if (_resp_corr is not None
                               and _resp_corr.size == _ns * _ns)
                           else None)

        # Shared colour scales across columns so the sector heatmaps and
        # spatial maps are comparable by eye (mirrors the stim figure).
        # ----------
        def _shared_symmetric_vmag(mats):
            _vals = [np.asarray(_m, dtype=np.float64).ravel()
                     for _m in mats if _m is not None]
            _cat = np.concatenate(_vals) if _vals else np.array([])
            _cat = _cat[np.isfinite(_cat)]
            if _cat.size == 0:
                return None
            _vmag = max(abs(float(np.percentile(_cat, 2))),
                        abs(float(np.percentile(_cat, 98))))
            return _vmag if (np.isfinite(_vmag) and _vmag > 0) else None

        def _shared_range(mats):
            _vals = [np.asarray(_m, dtype=np.float64).ravel()
                     for _m in mats if _m is not None]
            _cat = np.concatenate(_vals) if _vals else np.array([])
            _cat = _cat[np.isfinite(_cat)]
            if _cat.size == 0:
                return None, None
            _lo = float(np.percentile(_cat, 2))
            _hi = float(np.percentile(_cat, 98))
            if _hi <= _lo:
                _hi = _lo + 1e-9
            return _lo, _hi

        _heat_mats = [sector_avg_r, sector_avg_g]
        _resp_mats = [_resp_grid_r, _resp_grid_g]
        if _has_corr:
            if sector_avg_corr is not None:
                _heat_mats.append(sector_avg_corr)
            if _resp_grid_corr is not None:
                _resp_mats.append(_resp_grid_corr)
        _heat_vmag = _shared_symmetric_vmag(_heat_mats)
        _spatial_vlo, _spatial_vhi = _shared_range(_resp_mats)

        # Figure
        # ----------
        _ncols = 3 if _has_corr else 2
        if figsize is None:
            figsize = (18.0, 8) if _has_corr else (12.0, 8)
        fig = plt.figure(figsize=figsize)
        spec = gridspec.GridSpec(
            nrows=3, ncols=_ncols, figure=fig,
            height_ratios=[0.4, 1.0, 0.7],
            hspace=0.3, wspace=0.4)
        # Per-column display units: raw red/grn follow dff_sig (per-trial
        # dF/F0 %); the corrected channel is ITI z-score unless
        # dff_after_correction made it the per-trial dF/F0 (%)
        # reconstruction.
        _unit_rg = '(dF/F %)' if dff_sig else '(z)'
        _heat_unit_rg = 'dF/F0 (%)' if dff_sig else 'z-score'
        _unit_corr = '(dF/F %)' if (_corr_dff or _corr_predff) else '(z)'
        _heat_unit_corr = ('dF/F0 (%)' if (_corr_dff or _corr_predff)
                           else 'z-score')

        def _draw_column(col, wf_z, sector_z_avg, resp_grid,
                         trace_label, heat_label, spatial_label,
                         trace_color, spatial_cmap,
                         dff_override=None, heat_unit='z-score'):
            ax_trace = fig.add_subplot(spec[0, col])
            ax_heat = fig.add_subplot(spec[1, col], sharex=ax_trace)
            ax_spatial = fig.add_subplot(spec[2, col])

            _snips = _bout_snips(wf_z, dff_override=dff_override)
            if _snips:
                _arr = np.array(_snips)
                _mean = _arr.mean(axis=0)
                _sem = _arr.std(axis=0) / np.sqrt(_arr.shape[0])
                ax_trace.fill_between(_t_rel, _mean - _sem, _mean + _sem,
                                      color=trace_color, alpha=0.3,
                                      linewidth=0)
                ax_trace.plot(_t_rel, _mean, color=trace_color,
                              linewidth=1.2)
            else:
                ax_trace.text(0.5, 0.5, 'no bouts in range',
                              ha='center', va='center',
                              transform=ax_trace.transAxes,
                              fontsize=8, color='grey')
            ax_trace.axvline(x=0, color='k', linewidth=0.8,
                             linestyle='--', alpha=0.7)
            ax_trace.axhline(y=0, color='k', linewidth=0.4, linestyle=':')
            ax_trace.set_ylabel(trace_label, fontsize=8)

            if np.any(np.isfinite(sector_z_avg)):
                if _heat_vmag is not None:
                    _vmag = _heat_vmag
                else:
                    _vmag = max(abs(float(np.nanpercentile(
                                    sector_z_avg, 2))),
                                abs(float(np.nanpercentile(
                                    sector_z_avg, 98))))
                _x_edges = np.linspace(t_rel_sec[0], t_rel_sec[-1],
                                       sector_z_avg.shape[1] + 1)
                _y_edges = np.arange(sector_z_avg.shape[0] + 1)
                _im = ax_heat.pcolormesh(
                    _x_edges, _y_edges, sector_z_avg,
                    cmap='gray', vmin=-_vmag, vmax=_vmag, shading='flat')
                ax_heat.set_ylim(sector_z_avg.shape[0], 0)
                ax_heat.set_yticks([0, sector_z_avg.shape[0]])
                _cax = ax_heat.inset_axes([1.015, 0, 0.012, 1])
                fig.colorbar(_im, cax=_cax)
                _cax.tick_params(labelsize=7)
                _cax.set_ylabel(heat_unit, fontsize=7)
            else:
                ax_heat.text(0.5, 0.5, 'no bouts in range',
                             ha='center', va='center',
                             transform=ax_heat.transAxes,
                             fontsize=8, color='grey')
            ax_heat.axvline(x=0, color=sns.xkcd_rgb['bright red'],
                            linewidth=0.8, linestyle='--', alpha=0.8)
            ax_heat.set_ylabel(heat_label, fontsize=8)
            ax_heat.set_xlabel('time from lick-bout onset (s)')
            for _ax in (ax_trace, ax_heat):
                for _side in ('top', 'right'):
                    _ax.spines[_side].set_visible(False)
            plt.setp(ax_trace.get_xticklabels(), visible=False)
            ax_trace.set_xlim(t_rel_sec[0], t_rel_sec[-1])

            if resp_grid is not None and np.any(np.isfinite(resp_grid)):
                if _spatial_vlo is not None and _spatial_vhi is not None:
                    _vlo, _vhi = _spatial_vlo, _spatial_vhi
                else:
                    _vlo = float(np.nanpercentile(resp_grid, 2))
                    _vhi = float(np.nanpercentile(resp_grid, 98))
                if _vhi <= _vlo:
                    _vhi = _vlo + 1e-9
                _x_e = np.arange(resp_grid.shape[1] + 1)
                _y_e = np.arange(resp_grid.shape[0] + 1)
                _im_sp = ax_spatial.pcolormesh(
                    _x_e, _y_e, resp_grid, cmap=spatial_cmap,
                    vmin=_vlo, vmax=_vhi, shading='flat')
                ax_spatial.set_aspect('equal')
                ax_spatial.invert_yaxis()
                _cax = ax_spatial.inset_axes([1.03, 0, 0.04, 1])
                fig.colorbar(_im_sp, cax=_cax)
                _cax.tick_params(labelsize=7)
                _cax.set_ylabel('∫ response\n(0→post)', fontsize=7)
            else:
                ax_spatial.text(0.5, 0.5, 'no response data',
                                ha='center', va='center',
                                transform=ax_spatial.transAxes,
                                fontsize=8, color='grey')
            ax_spatial.set_xticks([])
            ax_spatial.set_yticks([])
            ax_spatial.set_ylabel(spatial_label, fontsize=8)
            return ax_trace

        _ax_r = _draw_column(
            0, _wf_red, sector_avg_r, _resp_grid_r,
            f'whole-frame\nred {_unit_rg}',
            f'sector\nred {_unit_rg}\n(1..{n_sec_total})',
            'sector map\nred', sns.xkcd_rgb['bright red'], 'Reds',
            dff_override=None, heat_unit=_heat_unit_rg)
        _ax_g = _draw_column(
            1, _wf_grn, sector_avg_g, _resp_grid_g,
            f'whole-frame\ngrn {_unit_rg}',
            f'sector\ngrn {_unit_rg}\n(1..{n_sec_total})',
            'sector map\ngrn', sns.xkcd_rgb['forest green'], 'Greens',
            dff_override=None, heat_unit=_heat_unit_rg)
        _ax_c = None
        if _has_corr:
            _ax_c = _draw_column(
                2, _wf_corr_in, sector_avg_corr, _resp_grid_corr,
                f'whole-frame\n{_corr_key} {_unit_corr}',
                f'sector\n{_corr_key} {_unit_corr}\n(1..{n_sec_total})',
                f'sector map\n{_corr_key}',
                sns.xkcd_rgb.get('bright orange', sns.xkcd_rgb['orange']),
                'Oranges',
                dff_override=_corr_dff, heat_unit=_heat_unit_corr)

        # Share one whole-frame y-range so amplitudes compare by eye.
        # ----------
        _axes = [_ax_r, _ax_g] + ([_ax_c] if _ax_c is not None else [])
        _lo = min(_a.get_ylim()[0] for _a in _axes)
        _hi = max(_a.get_ylim()[1] for _a in _axes)
        for _a in _axes:
            _a.set_ylim(_lo, _hi)

        _sort_lbl_esc = _sort_label.replace('_', r'\_')
        _title = (f'{self.path.animal} {self.path.date} '
                  f'{self.path.beh_folder} — QC (lick-aligned GRAB)  '
                  f'[{_bout_t.size} bouts]'
                  r'  $\mathbf{(sectors\ sorted\ by\ '
                  + _sort_lbl_esc + r')}$')
        fig.suptitle(_title, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if save:
            _ch_suffix = f'_ch={channel}' if channel is not None else ''
            _cs_suffix = self._qc_corrsig_suffix()
            _dffcorr_suffix = '_dffcorr' if _corr_dff else ''
            _fname = (f'{self.path.animal}_{self.path.date}_'
                      f'{self.path.beh_folder}'
                      f'_qc_lick_aligned{_ch_suffix}{_cs_suffix}'
                      f'{_dffcorr_suffix}.pdf')
            fig.savefig(os.path.join(str(self.folder.figs), _fname))
            if save_pickle:
                _pkl_path = os.path.join(
                    str(self.folder.data), _fname[:-4] + '.pkl.gz')
                with gzip.open(_pkl_path, 'wb') as _pf:
                    pickle.dump(fig, _pf,
                                protocol=pickle.HIGHEST_PROTOCOL)

        return fig
