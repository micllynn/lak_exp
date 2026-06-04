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
import concurrent.futures
import xml.etree.ElementTree as ElementTree
from types import SimpleNamespace

import numpy as np
import scipy.stats as sp_stats
import tifffile
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .utils import calc_alpha


def load_qc_fig(path):
    """Load a gzip-pickled QC matplotlib Figure for interactive viewing.

    Parameters
    ----------
    path : str
        Path to a .pkl.gz file produced by TwoPRec.plt_qc(save=True,
        save_pickle=True). Either the .pkl.gz path or the matching
        .pdf path is accepted; a trailing .pdf is auto-rewritten to
        .pkl.gz.

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
    (one per recording, written when `save_response_mean=True`),
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
        `save_response_mean=True`. Each is a pickled dict.
    channel : str
        Which channel to grand-average. One of 'red', 'grn', 'ratio',
        or the corrected-channel key (typically 'red_corr').
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

    def add_qc(self, n_sectors=8, channel=None,
               compute_sectors=False, sector_stride=4, sector_chunk=500,
               compute_frame_f=True, frame_stride=4, frame_chunk=500,
               z_corr_stride=4, z_corr_chunk=200, z_corr_jobs=4,
               z_corr_ref_n_frames=None,
               z_corr_dual_ref=True,
               use_zstack=None,
               zstack_channel=None,
               split_lr=False,
               correct_signal=False,
               correct_signal_kwargs=None,
               grab5ht_side=None,
               detrend_sigs=False,
               dff=False):
        """Compute continuous-time QC traces and load registration metrics.

        Parameters
        ----------
        n_sectors : int
            Number of sectors per axis. Field of view is divided into
            n_sectors x n_sectors blocks; continuous-time mean fluorescence
            is computed for each block.
        channel : str or None
            'red' or 'grn' (dual-colour only), None uses self.rec.
        compute_sectors : bool
            If True, compute the (n_sectors**2, *) sector fluorescence
            matrix. Disabled by default because it requires a full pass
            over the imaging memmap (slow on spinning disks).
        sector_stride : int
            Sample every sector_stride-th frame for the sector matrix.
            Sector traces are slow-varying, so stride 2-4 is visually
            lossless. Default 4.
        sector_chunk : int
            Number of sampled frames per chunked read.
        compute_frame_f : bool
            If True, populate self.qc.frame_f with whole-frame mean F.
            For single-channel recordings this rides along on the z_corr
            pass at zero additional I/O when meanImg is available;
            otherwise it falls back to a dedicated chunked pass. For
            dual-colour, both channels require dedicated passes.
            Default True.
        frame_stride : int
            Stride for whole-frame F when a dedicated pass is required
            (dual-colour, or single-channel without meanImg). Ignored for
            the single-channel piggyback case (uses z_corr_stride there).
            Default 4.
        frame_chunk : int
            Number of sampled frames per chunked read for frame_f.
        z_corr_stride : int
            Compute z_corr every z_corr_stride frames; intermediate values
            are linearly interpolated. Default 4.
        z_corr_chunk : int
            Number of sampled frames per thread-pool chunk.
        z_corr_jobs : int
            Number of parallel threads for z_corr computation.
        z_corr_ref_n_frames : int or None
            If int, compute the z_corr reference image as the mean of
            the first N frames of the active rec instead of using
            ops['meanImg']. Useful for tracking drift relative to the
            session start. If None (default), uses ops['meanImg'].
            Also controls the window size used at *both* ends when
            z_corr_dual_ref=True (defaults to 500 in that mode).
        z_corr_dual_ref : bool
            If True, build two reference images — the mean of the first
            z_corr_ref_n_frames and of the last z_corr_ref_n_frames —
            and compute a Pearson z_corr trace against each. Their
            difference (early − late) is a signed drift indicator that
            disagrees with corrXY when there is directional z-drift.
            In this mode the ops['meanImg']-based z_corr is skipped
            (reg.z_corr stays None); reg.z_corr_early, .z_corr_late,
            and .z_corr_diff are populated instead. Default False.
        use_zstack : str or None
            If a path is given, points to a Bruker ZSeries folder
            (containing per-cycle multi-page .ome.tif files and the
            .xml sidecar). Each rec frame is Pearson-correlated against
            every z-stack slice; the per-frame peak is parabolically
            triangulated to a sub-slice resolution and converted to a
            real z-position (µm) using the per-slice ZAxis values from
            the XML. Populates reg.z_um, reg.z_um_slices,
            reg.zstack_corr_mat, reg.zstack_slice_frac. Replaces the
            z_corr panel in plt_qc with the inferred z-position trace.
            Default None.
        zstack_channel : str or None
            'Ch1' (red) or 'Ch2' (green). Used only when use_zstack is
            given. Default None: chosen automatically to match the
            active channel of self.rec.
        split_lr : bool
            If True, divide each frame into a top half ('right') and a
            bottom half ('left') along the y-axis; whole-frame F is
            computed separately per half and stored under keys suffixed
            '_right' / '_left'. Sector traces are computed normally and
            split row-wise at plot time. Default False.
        correct_signal : bool
            If True (dual-colour only), run self.correct_signal with
            replace_real=False so the original signal is preserved, then
            compute whole-frame F on the corrected real channel and store
            under the key '{real_flu}_corr' in self.qc.frame_f (with
            _right/_left suffixes when split_lr=True). Default False.
        correct_signal_kwargs : dict or None
            Extra kwargs forwarded to self.correct_signal (e.g. method,
            static_flu, real_flu, detrend, t_start, t_end). replace_real
            is always forced to False. For the single-channel
            hemisphere-control 1-D path (split_lr=True, correct_signal=
            True, grab5ht_side set, no dual-colour rec): used to pick
            the method and forward 1-D parameters (smooth_window,
            airpls_lam, airpls_porder, airpls_max_iter, trim_initial,
            nn_slope). Default None.
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
            fast. Default True.
        dff : bool
            If True, convert the raw 'red' and 'grn' whole-frame and
            per-sector traces to dF/F0 (%) using a per-trace baseline F0
            equal to the mean of the trace's values inside the
            inter-trial intervals (across all trials). Applied after
            detrending if detrend_sigs is also True. Default False.

        Notes
        -----
        Stores self.qc (SimpleNamespace) with:
            .t              : (n_frames,) time vector
            .channel        : channel used for sector fluorescence
            .n_sectors      : int
            .sector_stride  : int stride used for sector_f
            .frame_stride   : int stride used for frame_f
            .z_corr_stride  : int stride used for z_corr
            .split_lr       : bool, whether frame_f is split top/bottom
            .frame_f        : dict {ch_label: (n_frames,)} whole-frame F,
                              empty if compute_frame_f is False
            .sector_f       : (n_sectors**2, n_compute) per-sector F,
                              or None if compute_sectors is False
            .reg            : SimpleNamespace of registration metrics with
                              .xoff, .yoff, .shift_mag, .corrXY,
                              .badframes, .z_corr
        """
        print(f'computing QC traces (n_sectors={n_sectors})...')

        self.qc = SimpleNamespace()
        self.qc.n_sectors = n_sectors
        self.qc.channel = channel
        self.qc.sector_stride = sector_stride
        self.qc.frame_stride = frame_stride
        self.qc.z_corr_stride = z_corr_stride
        self.qc.z_corr_ref_n_frames = z_corr_ref_n_frames
        self.qc.z_corr_dual_ref = z_corr_dual_ref
        self.qc.use_zstack = use_zstack
        self.qc.zstack_channel = zstack_channel
        self.qc.split_lr = split_lr
        self.qc.correct_signal = bool(correct_signal)
        self.qc.correct_signal_kwargs = (
            dict(correct_signal_kwargs)
            if correct_signal_kwargs else None)
        self.qc.grab5ht_side = (
            grab5ht_side if grab5ht_side in ('left', 'right') else None)
        self.qc.detrend_sigs = bool(detrend_sigs)
        self.qc.dff = bool(dff)
        self.qc.t = self._get_rec_t(channel)
        self.qc.frame_f = {}

        rec = self._get_rec(channel)
        _has_grn = hasattr(self, 'rec_grn')
        _has_red = hasattr(self, 'rec_red')

        # Sector fluorescence (continuous time)
        # ----------
        if compute_sectors:
            if _has_grn and _has_red:
                print(f'\textracting sector fluorescence (dual-colour) '
                      f'(stride={sector_stride}, '
                      f'chunk={sector_chunk})...')
                self.qc.sector_f = {}
                for _ch_label, _ch_rec in [('grn', self.rec_grn),
                                           ('red', self.rec_red)]:
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
                for _ch_label, _ch_rec in [('grn', self.rec_grn),
                                           ('red', self.rec_red)]:
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
                    _cs_kwargs = dict(correct_signal_kwargs or {})
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
                    # Inline-aggregates fast path: for linear_martianova
                    # (and only when split_lr is off so the sector
                    # partition is straightforward, AND save_to_disk is
                    # off because aggregates mode skips the full stack
                    # the disk artifact needs), tell the correction
                    # function to skip the (T, X, Y) output buffer and
                    # stream-compute whole-frame + per-sector mean
                    # traces directly. Removes a 14 GB allocation/write
                    # and two extra read passes over the corrected
                    # stack (frame_f + sector_f extraction).
                    _cs_method = _cs_kwargs.get('method', 'linear')
                    _use_agg = (
                        _cs_method == 'linear_martianova'
                        and not split_lr
                        and compute_sectors
                        and not _cs_kwargs.get('save_to_disk', False))
                    if _use_agg and 'aggregates_only' not in _cs_kwargs:
                        _cs_kwargs['aggregates_only'] = {
                            'n_sectors': n_sectors}
                    elif (_cs_method == 'linear_martianova'
                          and _cs_kwargs.get('save_to_disk', False)):
                        print('\t\t(note: save_to_disk=True forces the '
                              'legacy 3-pass + write path; pass '
                              'save_to_disk=False to plt_qc for the '
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
                        # Legacy path: extract aggregates from the full
                        # (T, X, Y) corrected stack via the same helpers
                        # the raw channels use.
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
        # `{label}_corr_{grab5ht_side}`. Defaults to linear_martianova
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
                _cs_kwargs = dict(correct_signal_kwargs or {})
                _method = _cs_kwargs.get('method', 'linear_martianova')
                print(f'\trunning 1-D signal correction '
                      f'(method={_method}, '
                      f'grab={_label}_{_grab_side}, '
                      f'mut={_label}_{_mut_side})...')
                if _method == 'linear_martianova':
                    from .signal_correction import \
                        correct_linear_martianova_1d
                    _kw_1d = {
                        k: _cs_kwargs[k] for k in (
                            'smooth_window', 'airpls_lam',
                            'airpls_porder', 'airpls_max_iter',
                            'trim_initial', 'nn_slope')
                        if k in _cs_kwargs}
                    _corr_tr, _beta = correct_linear_martianova_1d(
                        np.asarray(_ff_mut, dtype=np.float64),
                        np.asarray(_ff_grab, dtype=np.float64),
                        verbose=True, **_kw_1d)
                elif _method == 'linear':
                    # Simple OLS residual: corrected = grab − β·mut
                    # using OLS β with intercept on the raw traces.
                    _x = np.asarray(_ff_mut, dtype=np.float64)
                    _y = np.asarray(_ff_grab, dtype=np.float64)
                    _xm = float(np.mean(_x))
                    _ym = float(np.mean(_y))
                    _var_x = max(float(np.var(_x)), 1e-10)
                    _beta = float(
                        np.mean((_x - _xm) * (_y - _ym)) / _var_x)
                    _alpha = _ym - _beta * _xm
                    _corr_tr = (_y - (_alpha + _beta * _x)).astype(
                        np.float32)
                    print(f'\t\t\tβ = {_beta:.6g}, α = {_alpha:.6g}')
                else:
                    print(f'\t\t[1d corr] method={_method!r} not '
                          f'supported for 1-D hemisphere correction; '
                          f'falling back to linear_martianova.')
                    from .signal_correction import \
                        correct_linear_martianova_1d
                    _corr_tr, _beta = correct_linear_martianova_1d(
                        np.asarray(_ff_mut, dtype=np.float64),
                        np.asarray(_ff_grab, dtype=np.float64),
                        verbose=True)
                _corr_key = f'{_label}_corr_{_grab_side}'
                self.qc.frame_f[_corr_key] = _corr_tr
                self.qc.grab5ht_beta = float(_beta)
                print(f'\t\tstored corrected trace at '
                      f'frame_f[{_corr_key!r}] (β = {_beta:.6g}).')

        # Optional post-processing: linear detrend and/or dF/F0 on the
        # raw red/grn whole-frame and per-sector traces. Applied here so
        # everything downstream (plt_qc, _plt_qc_response_mean, etc.)
        # transparently uses the processed signals.
        # ----------
        if detrend_sigs or dff:
            _ops_str = []
            if detrend_sigs:
                _ops_str.append('detrend')
            if dff:
                _ops_str.append('dF/F0(ITI)')
            print(f'\tpost-processing red/grn traces: '
                  f'{", ".join(_ops_str)}...')
            self._apply_detrend_dff(detrend_sigs, dff)

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
            # (k, n_sec, by, n_sec, bx) → mean over within-block axes
            means = block.reshape(k, n_sectors, by,
                                  n_sectors, bx).mean(axis=(2, 4))
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
            means[c_pos:c_pos + n_in_chunk] = block.mean(axis=(1, 2))
        print('')

        if stride > 1:
            ind_computed = np.arange(0, n_frames, stride)
            return np.interp(
                np.arange(n_frames), ind_computed, means
            ).astype(np.float32)
        return means

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
            baseline = snippet[:n_pre].mean() if n_pre > 0 else 0.0
            if pct:
                snippet = (snippet - baseline) / (
                    abs(baseline) + 1e-9) * 100.0
            else:
                snippet -= baseline
            snippets.append(snippet)

        if not snippets:
            return t_rel, None, None

        arr = np.array(snippets)
        return (t_rel,
                arr.mean(axis=0),
                arr.std(axis=0) / np.sqrt(arr.shape[0]))

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

    def _to_dff_iti(self, M, itp_mask):
        """Convert F traces to dF/F0 (%) using an ITI baseline.

        F0 is computed per trace as the mean of the values inside
        `itp_mask` (an inter-trial-interval boolean over the time axis).
        Returns 100 * (F - F0) / F0.

        Parameters
        ----------
        M : np.ndarray
            1-D trace (n_t,) or 2-D matrix (n_rows, n_t).
        itp_mask : np.ndarray of bool
            Inter-trial-interval mask aligned to the time axis of M.

        Returns
        -------
        out : np.ndarray, float64
            dF/F0 in percent. NaNs where F0 is zero, non-finite, or the
            ITI base has fewer than 2 finite samples.
        """
        _M = np.asarray(M, dtype=np.float64)
        _was_1d = _M.ndim == 1
        if _was_1d:
            _M = _M[None, :]
        _mask = np.asarray(itp_mask, dtype=bool)
        if _mask.size != _M.shape[1] or _mask.sum() < 2:
            _out = np.full_like(_M, np.nan)
            return _out[0] if _was_1d else _out
        _base = _M[:, _mask]
        with np.errstate(invalid='ignore'):
            _F0 = np.nanmean(_base, axis=1)
        _out = np.full_like(_M, np.nan)
        _ok = np.isfinite(_F0) & (_F0 != 0)
        if np.any(_ok):
            _F0c = _F0[_ok, None]
            _out[_ok] = 100.0 * (_M[_ok] - _F0c) / _F0c
        return _out[0] if _was_1d else _out

    def _apply_detrend_dff(self, detrend_sigs, dff):
        """Post-process self.qc.frame_f / sector_f for detrend & dF/F.

        Detrending is applied only to raw 'red' and 'grn' bases (plus
        their split_lr halves); corrected '*_corr' channels are skipped
        because correct_signal already detrends them internally. dF/F0
        is applied to both raw and corrected channels so all
        fluorescence-like traces share the same dF/F (%) scale across
        downstream QC plots.

        Parameters
        ----------
        detrend_sigs : bool
            If True, subtract a per-trace linear fit (see
            _linear_detrend_rows). Raw red/grn only.
        dff : bool
            If True, convert each trace to dF/F0 (%) using the ITI
            baseline (see _to_dff_iti). Applied after detrending. Raw
            red/grn and *_corr channels.
        """
        if not (detrend_sigs or dff):
            return

        # Whole-frame traces
        # ----------
        _t = self.qc.t
        _itp_t = self._inter_trial_mask(_t) if dff else None
        if isinstance(self.qc.frame_f, dict):
            for _key in list(self.qc.frame_f.keys()):
                _base = _key.replace('_right', '').replace('_left', '')
                _is_raw_ch = _base in ('red', 'grn')
                _is_corr = _base.endswith('_corr')
                if not (_is_raw_ch or _is_corr):
                    continue
                _tr = np.asarray(self.qc.frame_f[_key])
                if detrend_sigs and _is_raw_ch:
                    _tr = self._linear_detrend_rows(_tr)
                if dff:
                    _tr = self._to_dff_iti(_tr, _itp_t)
                self.qc.frame_f[_key] = _tr

        # Per-sector matrices
        # ----------
        if isinstance(self.qc.sector_f, dict):
            _itp_sec = None
            if dff:
                _any = next(iter(self.qc.sector_f.values()))
                _n_compute = _any.shape[1]
                _sec_t = np.linspace(_t[0], _t[-1], _n_compute)
                _itp_sec = self._inter_trial_mask(_sec_t)
            for _key in list(self.qc.sector_f.keys()):
                _is_raw_ch = _key in ('red', 'grn')
                _is_corr = _key.endswith('_corr')
                if not (_is_raw_ch or _is_corr):
                    continue
                _M = np.asarray(self.qc.sector_f[_key])
                if detrend_sigs and _is_raw_ch:
                    _M = self._linear_detrend_rows(_M)
                if dff:
                    _M = self._to_dff_iti(_M, _itp_sec)
                self.qc.sector_f[_key] = _M.astype(np.float32)

    def _qc_corrsig_suffix(self):
        """Filename fragment encoding correct_signal, detrend, and dff
        state for QC saves.

        Returns
        -------
        suffix : str
            '_corrsig=<0|1>_method=<name>[_fit=<mode>]_detrend=<0|1>_dff=<0|1>'.
            The `_fit=` segment is appended for linear_martianova so
            the global-β vs per-pixel-β output can be told apart.
        """
        _cs = bool(getattr(self.qc, 'correct_signal', False))
        _fit_str = ''
        if _cs:
            _kw = getattr(self.qc, 'correct_signal_kwargs', None) or {}
            _method = _kw.get('method', 'linear')
            if _method == 'linear_martianova':
                _fit_str = f'_fit={_kw.get("fit_mode", "global")}'
        else:
            _method = 'none'
        _dt = int(bool(getattr(self.qc, 'detrend_sigs', False)))
        _df = int(bool(getattr(self.qc, 'dff', False)))
        return (f'_corrsig={int(_cs)}_method={_method}{_fit_str}'
                f'_detrend={_dt}_dff={_df}')

    # ------------------------------------
    # Plot
    # ------------------------------------

    def plt_qc(self,
               n_sectors=8,
               channel='red',
               plot_sectors=True,
               sector_stride=4,
               sector_chunk=500,
               compute_frame_f=True,
               frame_stride=4,
               frame_chunk=500,
               z_corr_stride=4,
               z_corr_chunk=200,
               z_corr_jobs=4,
               z_corr_ref_n_frames=None,
               z_corr_dual_ref=True,
               use_zstack=None,
               zstack_channel=None,
               split_lr=False,
               correct_signal=True,
               correct_signal_kwargs={
                   'method': 'linear_martianova'},
               grab5ht_side=None,
               save_to_disk=True,
               detrend_sigs=False,
               dff=False,
               dff_response_mean_fig=True,
               save_response_mean=True,
               figsize=None,
               z_vmin=-2,
               z_vmax=4,
               save=True,
               save_pickle=True,
               plt_show=True):
        """Large stacked QC figure for a two-photon recording.

        Parameters
        ----------
        n_sectors : int
            Sector grid size (per axis).
        channel : str or None
            'red' or 'grn' (dual-colour only) for sector heatmap; None
            uses self.rec. Default 'red'.
        plot_sectors : bool
            If True, include the sector fluorescence heatmap row and
            ensure sector_f has been computed. Default True.
        sector_stride : int
            Stride for sector fluorescence computation. Default 4.
        sector_chunk : int
            Frames per chunked read for sector fluorescence. Default 500.
        compute_frame_f : bool
            If True, compute whole-frame mean fluorescence. Default True.
        frame_stride : int
            Stride for dedicated whole-frame F passes (dual-colour, or
            single-channel without ops meanImg). Default 4.
        frame_chunk : int
            Frames per chunked read for whole-frame F. Default 500.
        z_corr_stride : int
            Stride for z_corr computation; intermediate values are
            linearly interpolated. Default 4.
        z_corr_chunk : int
            Frames per thread-pool chunk for z_corr. Default 200.
        z_corr_jobs : int
            Thread-pool size for z_corr. Default 4.
        z_corr_ref_n_frames : int or None
            If int, build the z_corr reference from the mean of the
            first N frames of the active rec instead of ops['meanImg'].
            Default None (uses ops['meanImg']). Also sets the window at
            both ends when z_corr_dual_ref=True (defaults to 500 there).
        z_corr_dual_ref : bool
            If True, compute z_corr against both an early reference
            (first N frames) and a late reference (last N frames),
            overlay the two traces in the z_corr panel, and add an
            extra panel for their difference (early − late) as a
            signed drift indicator. Default False.
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
            {'method': 'linear_martianova'} when correct_signal=True.
            For the 1-D hemisphere-control path (split_lr=True,
            correct_signal=True, grab5ht_side set, single-channel rec):
            picks the method (default 'linear_martianova') and any
            1-D parameters (smooth_window, airpls_lam, airpls_porder,
            airpls_max_iter, trim_initial, nn_slope).
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
        save_to_disk : bool
            Forwarded to self.correct_signal. When True, the
            correction streams its full corrected stack to
            `{base}_corr.tif` (memmap-backed) and additionally writes
            a trial-averaged stim-aligned TIFF to
            `{base}_corr_trialavg.tif` (see correct_signal docstring
            for the window/marker details). self.rec_{real}_corr
            still points to the disk-backed full stack so all QC
            panels function normally. Default False (corrected stack
            stays in RAM, nothing written to disk).
        detrend_sigs : bool
            If True (default), linearly detrend the raw red and green
            whole-frame and per-sector traces before any plotting or
            analysis (see add_qc for details). Mathematically equivalent
            — by linearity of the mean — to running
            signal_correction.detrend_linearly on the underlying (T, X,
            Y) rec arrays and then spatial-averaging.
        dff : bool
            If True, convert the raw red and green whole-frame and
            per-sector traces to dF/F0 (%) using a per-trace baseline
            F0 = mean of the values inside the inter-trial intervals.
            Applied after detrending if detrend_sigs is also True.
            Default False.
        figsize : tuple or None
            Figure size in inches. None auto-sizes based on plot_sectors.
        z_vmin, z_vmax : float
            Z-score colour limits for the sector heatmap.
        save : bool
            If True, saves a pdf to self.folder.figs.
        save_pickle : bool
            If True (and save=True), also write a gzipped pickle of the
            matplotlib Figure to .pkl.gz alongside the PDF for later
            interactive inspection via load_qc_fig. Default True.
        plt_show : bool
            If True, calls plt.show(); otherwise closes the figure.
        """

        # Forward save_to_disk into correct_signal_kwargs (without
        # mutating the caller's dict). When True the correction
        # writes a trial-averaged stim-aligned TIFF beside the source
        # channel file and skips populating self.rec_{real}_corr —
        # downstream QC steps that depend on the corrected stack will
        # be silently skipped (see add_qc's `if _corr_rec is not None`
        # guard). Default False preserves the in-RAM corrected stack
        # for QC and grand-average pipelines.
        # ----------
        if correct_signal:
            correct_signal_kwargs = dict(correct_signal_kwargs or {})
            correct_signal_kwargs['save_to_disk'] = bool(save_to_disk)

        _needs_sectors = plot_sectors and (
            not hasattr(self, 'qc') or self.qc.sector_f is None
            or self.qc.sector_stride != sector_stride
            or self.qc.n_sectors != n_sectors)
        _needs_recompute = (
            _needs_sectors
            or not hasattr(self, 'qc')
            or self.qc.n_sectors != n_sectors
            or self.qc.channel != channel
            or self.qc.frame_stride != frame_stride
            or self.qc.z_corr_stride != z_corr_stride
            or getattr(self.qc, 'z_corr_ref_n_frames', None)
                != z_corr_ref_n_frames
            or getattr(self.qc, 'z_corr_dual_ref', False)
                != z_corr_dual_ref
            or getattr(self.qc, 'use_zstack', None) != use_zstack
            or getattr(self.qc, 'zstack_channel', None)
                != zstack_channel
            or getattr(self.qc, 'split_lr', False) != split_lr
            or getattr(self.qc, 'correct_signal', False)
                != bool(correct_signal)
            or (getattr(self.qc, 'correct_signal_kwargs', None)
                != (dict(correct_signal_kwargs)
                    if correct_signal_kwargs else None))
            or getattr(self.qc, 'grab5ht_side', None)
                != (grab5ht_side
                    if grab5ht_side in ('left', 'right') else None)
            or getattr(self.qc, 'detrend_sigs', None)
                != bool(detrend_sigs)
            or getattr(self.qc, 'dff', None) != bool(dff))
        if _needs_recompute:
            self.add_qc(n_sectors=n_sectors, channel=channel,
                        compute_sectors=plot_sectors,
                        sector_stride=sector_stride,
                        sector_chunk=sector_chunk,
                        compute_frame_f=compute_frame_f,
                        frame_stride=frame_stride,
                        frame_chunk=frame_chunk,
                        z_corr_stride=z_corr_stride,
                        z_corr_chunk=z_corr_chunk,
                        z_corr_jobs=z_corr_jobs,
                        z_corr_ref_n_frames=z_corr_ref_n_frames,
                        z_corr_dual_ref=z_corr_dual_ref,
                        use_zstack=use_zstack,
                        zstack_channel=zstack_channel,
                        split_lr=split_lr,
                        correct_signal=correct_signal,
                        correct_signal_kwargs=correct_signal_kwargs,
                        grab5ht_side=grab5ht_side,
                        detrend_sigs=detrend_sigs,
                        dff=dff)

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

        if figsize is None:
            _fig_h = 7.0 if plot_sectors else 4.5
            if z_corr_dual_ref:
                _fig_h += 0.4
            if split_lr:
                _fig_h += 0.4
            if plot_sectors and _dual_sectors:
                _fig_h += 3.0
            if _has_dual_ff:
                _fig_h += 0.5
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
        ax_ratio = None
        ax_corr_sig = None
        ax_sectors = {}

        # Build the row layout dynamically. Order is fixed top-to-bottom;
        # rows are added conditionally based on plot_sectors / split_lr /
        # z_corr_dual_ref / dual-channel.
        _sec_h = 1.5 if split_lr else 0.9
        _rows = [('events', 0.12), ('licks', 0.12), ('frame', 0.35)]
        if split_lr:
            _rows.append(('frame_diff', 0.15))
        if _has_dual_ff:
            _rows.append(('ratio', 0.2))
        if _has_corr_ff:
            _rows.append(('corr_sig', 0.25))
        if plot_sectors:
            if _dual_sectors:
                for _ch in ('red', 'grn'):
                    if _ch in self.qc.sector_f:
                        _rows.append((f'sectors_{_ch}', _sec_h))
                if ('red' in self.qc.sector_f
                        and 'grn' in self.qc.sector_f):
                    _rows.append(('sectors_ratio', _sec_h))
            else:
                _rows.append(('sectors', _sec_h))
        _rows += [('shift', 0.225), ('corr', 0.18)]
        _has_zum = (getattr(self.qc, 'reg', None) is not None
                    and getattr(self.qc.reg, 'z_um', None) is not None)
        if _has_zum:
            # zstack-inferred z-position replaces the z_corr panel.
            _rows.append(('zum', 0.25))
        else:
            if not z_corr_dual_ref:
                _rows.append(('zcorr', 0.18))
            if z_corr_dual_ref:
                _rows.append(('zdiff', 0.18))

        _hr = [h for _, h in _rows]
        _idx = {name: i for i, (name, _) in enumerate(_rows)}
        spec = gridspec.GridSpec(
            nrows=len(_hr), ncols=1, figure=fig,
            height_ratios=_hr, hspace=0.25)

        ax_events = fig.add_subplot(spec[_idx['events'], 0])
        ax_licks = fig.add_subplot(spec[_idx['licks'], 0],
                                   sharex=ax_events)
        ax_frame = fig.add_subplot(spec[_idx['frame'], 0],
                                   sharex=ax_events)
        if 'frame_diff' in _idx:
            ax_frame_diff = fig.add_subplot(spec[_idx['frame_diff'], 0],
                                            sharex=ax_events)
        if 'ratio' in _idx:
            ax_ratio = fig.add_subplot(spec[_idx['ratio'], 0],
                                       sharex=ax_events)
        if 'corr_sig' in _idx:
            ax_corr_sig = fig.add_subplot(spec[_idx['corr_sig'], 0],
                                          sharex=ax_events)
        for _key in ('sectors', 'sectors_red', 'sectors_grn',
                     'sectors_ratio'):
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

        # Row 0: stim + reward onsets
        # ----------
        _stim_t = np.asarray(self.beh.stim.t_start)
        _rew_t = np.asarray(self.beh.rew.t)
        if _stim_t.size > 0:
            _exp_rew = self.beh.stim.size * self.beh.stim.prob
            _exp_rew_max = np.max(_exp_rew) if np.max(_exp_rew) > 0 else 1.0
            for _i, _t in enumerate(_stim_t):
                _alpha = calc_alpha(_exp_rew[_i], val_max=_exp_rew_max,
                                    alpha_min=0.1)
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

        # Row 1: licks as point process
        # ----------
        _lick_t = np.asarray(getattr(self.beh.lick, 't_raw', []))
        if _lick_t.size > 0:
            ax_licks.vlines(_lick_t, ymin=0, ymax=1,
                            color='k', linewidth=0.5, alpha=0.7)
        ax_licks.set_ylim(0, 1)
        ax_licks.set_yticks([])
        ax_licks.set_ylabel('licks', fontsize=8)

        # Row 2: whole-frame fluorescence (1 or 2 channels)
        # ----------
        _ch_color = {'grn': sns.xkcd_rgb['forest green'],
                     'red': sns.xkcd_rgb['brick'],
                     'frame': sns.xkcd_rgb['dark grey']}
        _ff_plotted = 0
        for _label, _trace in self.qc.frame_f.items():
            _base = _label.replace('_right', '').replace('_left', '')
            # Corrected channels live in the dedicated ax_corr_sig row.
            if _base.endswith('_corr'):
                continue
            _color = _ch_color.get(_base, 'k')
            _ls = '--' if _label.endswith('_left') else '-'
            ax_frame.plot(t, _trace, color=_color, linewidth=0.7,
                          linestyle=_ls, label=_label)
            _ff_plotted += 1
        # Label reflects the post-processing applied in add_qc.
        if dff:
            _frame_ylabel = 'whole-frame\ndF/F0 (%)'
        elif detrend_sigs:
            _frame_ylabel = 'whole-frame\nF (detr.)'
        else:
            _frame_ylabel = 'whole-frame\nF'
        ax_frame.set_ylabel(_frame_ylabel, fontsize=8)
        if _ff_plotted > 1:
            ax_frame.legend(loc='upper right', fontsize=7, frameon=False)

        # Optional row: per-frame left − right whole-frame F (split_lr)
        # ----------
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

        # Optional row: red / green whole-frame F ratio (dual-colour).
        # When dff=True the underlying red/grn traces are already
        # dF/F0 (%), so we plot the ratio directly; otherwise we
        # z-score the ratio against the ITI baseline as before.
        # ----------
        if ax_ratio is not None:
            _ratio_color = sns.xkcd_rgb['orange']
            _itp_mask = self._inter_trial_mask(t)

            def _zscore_itp(x):
                _base = np.asarray(x)[_itp_mask]
                _base = _base[np.isfinite(_base)]
                if _base.size < 2:
                    return np.full_like(x, np.nan, dtype=np.float64)
                _mu = float(np.mean(_base))
                _sd = float(np.std(_base))
                if _sd == 0 or not np.isfinite(_sd):
                    return np.full_like(x, np.nan, dtype=np.float64)
                return (np.asarray(x, dtype=np.float64) - _mu) / _sd

            def _ratio_transform(x):
                return np.asarray(x, dtype=np.float64) if dff \
                    else _zscore_itp(x)

            if split_lr:
                for _suffix, _ls, _lbl in [('_right', '-', 'right'),
                                           ('_left', '--', 'left')]:
                    _r = self.qc.frame_f.get(f'red{_suffix}')
                    _g = self.qc.frame_f.get(f'grn{_suffix}')
                    if _r is None or _g is None:
                        continue
                    with np.errstate(divide='ignore', invalid='ignore'):
                        _ratio = _r / (_g + 1e-9)
                    _ratio_v = _ratio_transform(_ratio)
                    ax_ratio.plot(t, _ratio_v, color=_ratio_color,
                                  linewidth=0.7, linestyle=_ls,
                                  label=_lbl)
                ax_ratio.legend(loc='upper right', fontsize=7,
                                ncol=2, frameon=False)
            else:
                _r = self.qc.frame_f.get('red')
                _g = self.qc.frame_f.get('grn')
                if _r is not None and _g is not None:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        _ratio = _r / (_g + 1e-9)
                    _ratio_v = _ratio_transform(_ratio)
                    ax_ratio.plot(t, _ratio_v, color=_ratio_color,
                                  linewidth=0.7)
            ax_ratio.axhline(y=0, color='k', linewidth=0.4,
                             linestyle=':')
            _ratio_ylabel = ('red / grn\n(dF/F ratio)' if dff
                             else 'red / grn\nF (z, ITI)')
            ax_ratio.set_ylabel(_ratio_ylabel, fontsize=8)

        # Optional row: corrected real-channel whole-frame F (dual-colour,
        # correct_signal=True).
        # ----------
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

        # Row 3: sector fluorescence heatmap (z-scored per row)
        # ----------
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
            # When dff=True the underlying per-sector traces are
            # already dF/F0 (%); display directly with a percentile-
            # based symmetric colour range. Otherwise z-score per row.
            if dff:
                _z = np.asarray(_sec, dtype=np.float32)
                _vmag = float(np.nanpercentile(np.abs(_z), 98))
                if not np.isfinite(_vmag) or _vmag == 0:
                    _vmag = 1.0
                _sec_vmin, _sec_vmax = -_vmag, _vmag
                _sec_cbar = 'dF/F0 (%)'
            else:
                _mean = np.mean(_sec, axis=1, keepdims=True)
                _std = np.std(_sec, axis=1, keepdims=True)
                _z = (_sec - _mean) / (_std + 1e-9)
                _sec_vmin, _sec_vmax = z_vmin, z_vmax
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

        # Per-sector summary panel (dual-colour only). Default: red/grn
        # ratio normalised to each sector's session mean (greyscale).
        # When correct_signal=True the panel switches to the per-sector
        # corrected real channel, z-scored per row.
        # ----------
        if 'sectors_ratio' in ax_sectors \
                and isinstance(self.qc.sector_f, dict) \
                and 'red' in self.qc.sector_f \
                and 'grn' in self.qc.sector_f:
            _corr_sec_key = next(
                (k for k in self.qc.sector_f if k.endswith('_corr')),
                None)
            if _corr_sec_key is not None:
                _sec = self.qc.sector_f[_corr_sec_key].astype(np.float32)
                if dff:
                    # _sec is already dF/F0 (%); display directly.
                    _data = _sec
                    _vmag = float(np.nanpercentile(np.abs(_data), 98))
                    if not np.isfinite(_vmag) or _vmag == 0:
                        _vmag = 1.0
                    _vlo, _vhi = -_vmag, _vmag
                    _ylab_base = _corr_sec_key
                    _cbar_lab = f'{_corr_sec_key}\n(dF/F %)'
                else:
                    _mean = np.mean(_sec, axis=1, keepdims=True)
                    _std = np.std(_sec, axis=1, keepdims=True)
                    _data = (_sec - _mean) / (_std + 1e-9)
                    _vmag = float(np.nanpercentile(np.abs(_data), 98))
                    _vlo, _vhi = -_vmag, _vmag
                    _ylab_base = _corr_sec_key
                    _cbar_lab = f'{_corr_sec_key}\n(z)'
            else:
                _sec_r = self.qc.sector_f['red'].astype(np.float32)
                _sec_g = self.qc.sector_f['grn'].astype(np.float32)
                with np.errstate(divide='ignore', invalid='ignore'):
                    _ratio = _sec_r / (_sec_g + 1e-9)
                if dff:
                    # red and grn rows are already dF/F0 (%); show the
                    # ratio directly without row-mean normalisation.
                    _data = _ratio
                    _vlo = float(np.nanpercentile(_data, 2))
                    _vhi = float(np.nanpercentile(_data, 98))
                    _ylab_base = 'red/grn'
                    _cbar_lab = 'red/grn\n(dF/F ratio)'
                else:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        _row_mean = np.mean(_ratio, axis=1, keepdims=True)
                        _data = _ratio / (_row_mean + 1e-9)
                    _vlo = float(np.nanpercentile(_data, 2))
                    _vhi = float(np.nanpercentile(_data, 98))
                    _ylab_base = 'red/grn'
                    _cbar_lab = 'red/grn (norm)'
            _ax_entry = ax_sectors['sectors_ratio']
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

        # Row 4: shift magnitude
        # ----------
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

        # Row 5: corrXY (Suite2P registration phase-correlation)
        # ----------
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

        # Row 6: z_corr (zero-lag Pearson correlation vs. meanImg, or
        # against early/late refs when z_corr_dual_ref=True)
        # ----------
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

        # Optional row: signed z_corr drift (early − late)
        # ----------
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

        # Optional row: z-position inferred from a Bruker z-stack match
        # (use_zstack). Replaces the z_corr / z_corr_diff panel.
        # ----------
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

        # Cosmetic cleanup across all axes
        # ----------
        _all_ax = []
        # Top-to-bottom: events, licks, frame, frame_diff, ratio,
        # sectors (in row order), shift, corr, zcorr, zdiff.
        _ordered = [ax_events, ax_licks, ax_frame, ax_frame_diff,
                    ax_ratio, ax_corr_sig]
        for _key in ('sectors', 'sectors_red', 'sectors_grn',
                     'sectors_ratio'):
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
        if _cs_method == 'linear_martianova':
            _fm = _cs_kw.get('fit_mode', 'global')
            _title = _title + f'  [corr: martianova, fit={_fm}]'
        elif _cs_method is not None:
            _title = _title + f'  [corr: {_cs_method}]'
        fig.suptitle(_title, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        if save:
            _ch_suffix = f'_ch={channel}' if channel is not None else ''
            _cs_suffix = self._qc_corrsig_suffix()
            _fname = (f'{self.path.animal}_{self.path.date}_'
                      f'{self.path.beh_folder}'
                      f'_qc_n_sectors={n_sectors}'
                      f'{_ch_suffix}{_cs_suffix}.pdf')
            fig.savefig(os.path.join(str(self.folder.figs), _fname))
            if save_pickle:
                _pkl_name = _fname[:-4] + '.pkl.gz'
                _pkl_path = os.path.join(str(self.folder.figs),
                                         _pkl_name)
                with gzip.open(_pkl_path, 'wb') as _pf:
                    pickle.dump(fig, _pf,
                                protocol=pickle.HIGHEST_PROTOCOL)

        _figs = [fig]

        # Additional QC figure 1: registration stats
        # ----------
        _figs.append(self._plt_qc_reg_stats(t, reg, channel=channel,
                                            save=save))

        # Additional QC figure 2: event-triggered response means
        # ----------
        _response_fig = self._plt_qc_response_mean(
            t, channel=channel, save=save,
            dff_sig=dff_response_mean_fig,
            save_npy=save_response_mean)
        if _response_fig is not None:
            _figs.append(_response_fig)

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

    def _plt_qc_reg_stats(self, t, reg, channel=None,
                          t_pre=2.0, t_post=2.0,
                          figsize=None, save=True):
        """QC summary of registration metrics: event-triggered averages
        on top, whole-frame F vs. metric scatter underneath.

        Top block (2 rows × 3 cols): stim-triggered (row 0) and
        reward-triggered (row 1) averages of shift_mag, corrXY, z_corr.
        Each trial baseline-subtracted using the pre-event window; SEM
        shown as a shaded band.

        Bottom block (n_channels rows × 3 cols): scatter of whole-frame F
        vs. shift_mag, corrXY, z_corr. One row per base channel present
        in self.qc.frame_f.

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

        _stim_t = np.asarray(self.beh.stim.t_start)
        _rew_t = np.asarray(self.beh.rew.t)
        _row_events = [_stim_t, _rew_t]
        _row_colors = [sns.xkcd_rgb['dark grey'],
                       sns.xkcd_rgb['bright blue']]
        _row_ylabels = ['stim-triggered', 'reward-triggered']

        # Resolve scatter metrics (z_corr falls back to z_corr_diff in
        # dual-ref mode for the scatter — a signed drift indicator).
        _sc_metrics = {
            'shift_mag': self._interp_metric_to_t(reg.shift_mag, t),
            'corrXY':    self._interp_metric_to_t(reg.corrXY, t),
            'z_corr':    self._interp_metric_to_t(reg.z_corr, t),
        }
        if _sc_metrics['z_corr'] is None and reg.z_corr_diff is not None:
            _sc_metrics['z_corr'] = self._interp_metric_to_t(
                reg.z_corr_diff, t)
        _sc_metric_xlabels = ['shift mag (px)', 'corrXY', 'z_corr']

        _ch_color = {'grn': sns.xkcd_rgb['forest green'],
                     'red': sns.xkcd_rgb['brick'],
                     'frame': sns.xkcd_rgb['dark grey']}
        _ch_full = {'grn': 'green', 'red': 'red', 'frame': 'frame'}

        _bases = []
        for _label in self.qc.frame_f:
            _b = (_label.replace('_right', '')
                       .replace('_left', ''))
            if _b not in _bases:
                _bases.append(_b)
        if not _bases:
            _bases = ['frame']
        _order = ['red', 'grn', 'frame']
        _bases.sort(
            key=lambda x: _order.index(x) if x in _order else 99)

        split_lr = getattr(self.qc, 'split_lr', False)
        n_sc_rows = len(_bases)

        figsize = figsize or (9, 5 + 2.5 * n_sc_rows + 0.5)
        fig = plt.figure(figsize=figsize)
        # Two stacked sub-grids so each block keeps its own axis sharing.
        _ev_h = 5.0
        _sc_h = 2.5 * n_sc_rows + 0.5
        outer = gridspec.GridSpec(
            nrows=2, ncols=1, figure=fig,
            height_ratios=[_ev_h, _sc_h], hspace=0.35)
        ev_spec = outer[0, 0].subgridspec(2, 3, hspace=0.25, wspace=0.25)
        sc_spec = outer[1, 0].subgridspec(
            n_sc_rows, 3, hspace=0.25, wspace=0.25)

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

        # ---- Scatter block ----
        sc_axes = np.empty((n_sc_rows, 3), dtype=object)
        for r_idx in range(n_sc_rows):
            for c_idx in range(3):
                _sharex = sc_axes[0, c_idx] if r_idx > 0 else None
                _sharey = sc_axes[r_idx, 0] if c_idx > 0 else None
                sc_axes[r_idx, c_idx] = fig.add_subplot(
                    sc_spec[r_idx, c_idx],
                    sharex=_sharex, sharey=_sharey)

        _dff_mode = bool(getattr(self.qc, 'dff', False))
        _yl_unit = 'dF/F0 (%)' if _dff_mode else 'F'
        for r_idx, _base in enumerate(_bases):
            _color = _ch_color.get(_base, 'k')
            _yl = (f'whole-frame {_yl_unit}\n'
                   f'({_ch_full.get(_base, _base)})')
            for c_idx, (mkey, mxlbl) in enumerate(
                    zip(_metric_keys, _sc_metric_xlabels)):
                ax = sc_axes[r_idx, c_idx]
                _metric = _sc_metrics[mkey]

                if split_lr:
                    _ff_r = self.qc.frame_f.get(f'{_base}_right')
                    _ff_l = self.qc.frame_f.get(f'{_base}_left')
                    _have = _metric is not None and (
                        _ff_r is not None or _ff_l is not None)
                    if not _have:
                        ax.text(0.5, 0.5, 'no data',
                                ha='center', va='center',
                                transform=ax.transAxes,
                                fontsize=8, color='grey')
                    else:
                        if _ff_r is not None:
                            ax.scatter(_metric, _ff_r,
                                       color=_color, s=1, alpha=0.3,
                                       linewidths=0, rasterized=True,
                                       label='right')
                        if _ff_l is not None:
                            ax.scatter(_metric, _ff_l,
                                       color='k', s=1, alpha=0.15,
                                       linewidths=0, rasterized=True,
                                       label='left')
                        if c_idx == 0:
                            ax.legend(loc='upper right', fontsize=7,
                                      frameon=False, markerscale=4)
                else:
                    _ff = self.qc.frame_f.get(_base)
                    if _ff is None or _metric is None:
                        ax.text(0.5, 0.5, 'no data',
                                ha='center', va='center',
                                transform=ax.transAxes,
                                fontsize=8, color='grey')
                    else:
                        ax.scatter(_metric, _ff,
                                   color=_color, s=1, alpha=0.3,
                                   linewidths=0, rasterized=True)

                if c_idx == 0:
                    ax.set_ylabel(_yl, fontsize=8)
                if r_idx == n_sc_rows - 1:
                    ax.set_xlabel(mxlbl, fontsize=8)

                for _side in ('top', 'right'):
                    ax.spines[_side].set_visible(False)

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

        _stim_t = np.asarray(self.beh.stim.t_start)
        _dff_mode = bool(getattr(self.qc, 'dff', False))

        def _zscore_iti(trace):
            # When dff=True the underlying trace is already dF/F0 (%);
            # pass through. Otherwise z-score against the ITI baseline.
            if _dff_mode:
                return np.asarray(trace, dtype=np.float64)
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
        np.save(os.path.join(str(self.folder.figs), _npy_name),
                _summary, allow_pickle=True)
        return

    def _plt_qc_response_mean(self, t, channel=None,
                                t_pre=2.0, t_post=6.0,
                                figsize=None, save=True,
                                dff_sig=False,
                                save_npy=True):
        """Stim-aligned GRAB-style averages (dual-colour only).

        Left column: whole-frame red/grn ratio (mean ± SEM across stims,
        z-scored on inter-trial periods) above, and a per-sector heatmap
        of the same ratio (z-scored per sector against its own ITI
        baseline, then stim-averaged) below.

        Right column (only when a corrected real channel is available,
        i.e. plt_qc was called with correct_signal=True): same layout
        and same ITI z-score procedure, but applied to the corrected
        real-channel trace directly (no red/grn division).

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
            If True, each plotted column (red, grn, red/grn, red_corr)
            is converted to dF/F0 (%) per trial, where F0 is the mean
            of the trace in the pre-stim baseline window (length t_pre
            seconds). Applied independently to whole-frame and per-
            sector traces. Default False (uses ITI z-score).
        save_response_mean : bool
            If True, also writes a `.npy` file alongside the pdf with
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

        with np.errstate(divide='ignore', invalid='ignore'):
            _ratio_ff = _ff_r / (_ff_g + 1e-9)
        _sec_r = self.qc.sector_f['red'].astype(np.float32)
        _sec_g = self.qc.sector_f['grn'].astype(np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            _ratio_sec = _sec_r / (_sec_g + 1e-9)

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
        _stim_t = np.asarray(self.beh.stim.t_start)
        _dff_mode = bool(getattr(self.qc, 'dff', False))

        # Helper: ITI z-score a 1D trace given the time vector it lives
        # on. When dff=True the underlying traces are already dF/F0 (%),
        # so we pass through and let the downstream plots display the
        # dF/F values directly.
        # ----------
        def _zscore_iti(trace, t_vec):
            if _dff_mode:
                return np.asarray(trace, dtype=np.float64)
            _mask = self._inter_trial_mask(t_vec)
            _base = np.asarray(trace)[_mask]
            _base = _base[np.isfinite(_base)]
            if _base.size < 2:
                return np.full_like(trace, np.nan, dtype=np.float64)
            _mu = float(np.mean(_base))
            _sd = float(np.std(_base))
            if _sd == 0 or not np.isfinite(_sd):
                return np.full_like(trace, np.nan, dtype=np.float64)
            return (np.asarray(trace, dtype=np.float64) - _mu) / _sd

        # Z-score whole-frame traces against their own ITI baselines.
        # ----------
        _ratio_ff_z = _zscore_iti(_ratio_ff, t)
        _ff_corr_z = (_zscore_iti(_ff_corr, t)
                      if _has_corr else None)

        # Sector grid is computed at sector_stride; build matching t vec.
        # ----------
        n_sec_total, n_compute = _ratio_sec.shape
        sec_t = np.linspace(t[0], t[-1], n_compute)

        # Per-sector ITI z-score using each sector's own ITI baseline.
        # ----------
        _sec_iti_mask = self._inter_trial_mask(sec_t)

        def _zscore_sectors(sec_mat):
            if _dff_mode:
                return np.asarray(sec_mat, dtype=np.float64)
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
        _sec_corr_z = (_zscore_sectors(_sec_corr)
                       if _has_corr else None)

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

        def _stim_snips(trace, dff_override=None):
            # When `dff` (resolved from dff_sig or dff_override) is True,
            # the input is a raw trace and each per-trial snippet is
            # converted to dF/F0 (%) using the pre-stim baseline window
            # (length _n_pre samples) as F0. Otherwise the input is
            # assumed pre-z-scored. dff_override=False forces the
            # z-score path even when dff_sig is on (used by the
            # corrected-channel column, which is always z-sorted).
            _dff = dff_sig if dff_override is None else bool(dff_override)
            # Guard against double dF/F: when qc.dff is on the input
            # traces are already dF/F0 (%), so the per-trial baseline
            # normalisation below must be skipped.
            _dff = _dff and not _dff_mode
            _out = []
            for _ev in np.asarray(_stim_t).ravel():
                _i = int(np.argmin(np.abs(t - _ev)))
                _i0 = _i - _n_pre
                _i1 = _i + _n_post + 1
                if _i0 < 0 or _i1 > len(trace):
                    continue
                _snip = np.asarray(trace[_i0:_i1], dtype=np.float64)
                if not np.all(np.isfinite(_snip)):
                    continue
                if _dff:
                    if _n_pre <= 0:
                        continue
                    _f0 = float(np.mean(_snip[:_n_pre]))
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

        def _stim_aligned_sectors(sec_mat, dff_override=None):
            # If `dff` (resolved from dff_sig or dff_override) is on,
            # expect raw sector traces in and convert each per-trial
            # snippet to dF/F0 (%) using its own pre-stim baseline
            # (length n_pre samples) before averaging across trials.
            # Otherwise expect pre-z-scored sectors. dff_override=False
            # forces the z-score path even when dff_sig is on.
            _dff = dff_sig if dff_override is None else bool(dff_override)
            # Guard against double dF/F: when qc.dff is on the input
            # sector traces are already dF/F0 (%), so the per-trial
            # baseline normalisation below must be skipped.
            _dff = _dff and not _dff_mode
            _avg = np.full((sec_mat.shape[0], n_win), np.nan,
                           dtype=np.float32)
            if _dff:
                _ev_arr = np.asarray(_stim_t).ravel()
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

        # Pick the inputs fed into the snippet extractors. In dff_sig
        # mode we pass the raw post-processed traces and let the snippet
        # helpers convert each trial to dF/F0 (%); otherwise we pass the
        # ITI-z-scored traces.
        # ----------
        if dff_sig:
            _wf_red = _ff_r
            _wf_grn = _ff_g
            _wf_ratio = _ratio_ff
            # red_corr column is always z-sorted regardless of dff_sig,
            # so it stays on the ITI-z-scored input.
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

        sector_avg = _stim_aligned_sectors(_sec_ratio_in)
        sector_avg_corr = (_stim_aligned_sectors(_sec_corr_in,
                                                 dff_override=False)
                           if _has_corr else None)
        sector_avg_r = _stim_aligned_sectors(_sec_red_in)
        sector_avg_g = _stim_aligned_sectors(_sec_grn_in)

        # Per-sector trial reliability: mean off-diagonal Pearson R across
        # stim-aligned trials. Uses the same source as the row-ordering
        # (corrected channel if available, else red/grn ratio z) so the
        # column tells a consistent story alongside the heatmaps.
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

        _rel_src = _sec_corr_z if _has_corr else _ratio_sec_z
        sector_reliability = _sector_reliability(_rel_src)

        # Median stim → reward latency from .beh (per-trial pairing).
        # Used both for the dashed blue reward marker on every axis and
        # for the integration window when ordering sectors.
        # ----------
        _rew_t_all = np.asarray(getattr(self.beh.rew, 't', []),
                                dtype=object).ravel()
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

        # Order sectors descending by integrated stim → rew response,
        # computed from the left-column heatmap (red/grn ratio z). The
        # same ordering is applied to the right column so corresponding
        # rows index the same sectors across panels.
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
            # Peak (max) activation per sector within the stim → rew
            # window — used as the ordering metric so sectors with the
            # strongest stim-evoked response sit at the top of the heat-
            # maps. Prefer the corrected real channel (cleanest activa-
            # tion signal); fall back to the raw red channel when
            # correct_signal=False.
            _peak_corr = (np.nanmax(sector_avg_corr[:, _i0:_i1], axis=1)
                          if sector_avg_corr is not None else None)
            _peak_r = np.nanmax(sector_avg_r[:, _i0:_i1], axis=1)
        if _peak_corr is not None:
            _order_src = _peak_corr
            _sort_label = _corr_key
        else:
            _order_src = _peak_r
            _sort_label = 'red'
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

        # Whole-frame Pearson R: raw red vs grn fluorescence, and
        # (when available) corrected real-channel vs grn. Computed on
        # the full QC time series over jointly finite samples.
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

        _pear_rg_r, _pear_rg_p = _pearson_finite(_ff_r, _ff_g)
        if _has_corr:
            _pear_cg_r, _pear_cg_p = _pearson_finite(_ff_corr, _ff_g)
        else:
            _pear_cg_r = _pear_cg_p = np.nan

        # Sector-map Spearman: integrated-response sector vectors
        # (red vs grn, and corrected vs grn). p-value from a spatial-
        # permutation null distribution that shuffles green sector
        # positions while holding red/corr fixed.
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
        if _resp_grid_r is not None and _resp_grid_g is not None:
            _spear_rg_rho, _spear_rg_p, _spear_rg_null = \
                _sector_corr_with_null(_resp_r, _resp_g,
                                       kind='spearman')
            _pear_sec_rg_r, _pear_sec_rg_p, _pear_sec_rg_null = \
                _sector_corr_with_null(_resp_r, _resp_g,
                                       kind='pearson')
        if (_has_corr and _resp_grid_corr is not None
                and _resp_grid_g is not None):
            _spear_cg_rho, _spear_cg_p, _spear_cg_null = \
                _sector_corr_with_null(_resp_corr, _resp_g,
                                       kind='spearman')
            _pear_sec_cg_r, _pear_sec_cg_p, _pear_sec_cg_null = \
                _sector_corr_with_null(_resp_corr, _resp_g,
                                       kind='pearson')

        # Pre-format stats blurbs to drop under the red and corr
        # columns. None passed for grn/ratio columns.
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

        _stats_red = (
            f'whole-frame Pearson (red vs grn):\n'
            f'  {_fmt_stat(_pear_rg_r, _pear_rg_p)}\n'
            f'sector Pearson (red vs grn):\n'
            f'  {_fmt_stat(_pear_sec_rg_r, _pear_sec_rg_p)}\n'
            f'sector Spearman (red vs grn):\n'
            f'  {_fmt_stat(_spear_rg_rho, _spear_rg_p)}'
        )
        if _has_corr:
            _stats_corr = (
                f'whole-frame Pearson ({_corr_key} vs grn):\n'
                f'  {_fmt_stat(_pear_cg_r, _pear_cg_p)}\n'
                f'sector Pearson ({_corr_key} vs grn):\n'
                f'  {_fmt_stat(_pear_sec_cg_r, _pear_sec_cg_p)}\n'
                f'sector Spearman ({_corr_key} vs grn):\n'
                f'  {_fmt_stat(_spear_cg_rho, _spear_cg_p)}'
            )
        else:
            _stats_corr = None

        # Figure layout
        # ----------
        # Data columns: raw red (always), raw green (always), red/grn
        # ratio (always), and the corrected channel (when correct_signal
        # =True). The raw-red and raw-green columns each get a spatial
        # map; the red/grn column does not (its spatial slot is left
        # empty); the corrected column gets one.
        _ncols = 4 if _has_corr else 3
        if figsize is None:
            figsize = (24.0, 8) if _has_corr else (18.0, 8)
        fig = plt.figure(figsize=figsize)
        # Last column is a narrow sector-reliability strip (only row 1).
        _width_ratios = [1.0] * _ncols + [0.18]
        spec = gridspec.GridSpec(
            nrows=3, ncols=_ncols + 1, figure=fig,
            height_ratios=[0.4, 1.0, 0.7],
            width_ratios=_width_ratios,
            hspace=0.3, wspace=0.6)

        def _draw_column(col, trace_z, sector_z_avg, resp_grid,
                         trace_label, heat_label, spatial_label,
                         trace_color, draw_spatial=True,
                         spatial_cmap='Reds',
                         dff_override=None,
                         stats_text=None):
            ax_trace = fig.add_subplot(spec[0, col])
            ax_heat = fig.add_subplot(spec[1, col], sharex=ax_trace)
            ax_spatial = (fig.add_subplot(spec[2, col])
                          if draw_spatial else None)

            _snips = _stim_snips(trace_z, dff_override=dff_override)
            if _snips:
                _arr = np.array(_snips)
                _mean = _arr.mean(axis=0)
                _sem = _arr.std(axis=0) / np.sqrt(_arr.shape[0])
                ax_trace.fill_between(_t_rel, _mean - _sem, _mean + _sem,
                                      color=trace_color,
                                      alpha=0.3, linewidth=0)
                ax_trace.plot(_t_rel, _mean,
                              color=trace_color, linewidth=1.2)
            else:
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

            if np.any(np.isfinite(sector_z_avg)):
                _vlo = float(np.nanpercentile(sector_z_avg, 2))
                _vhi = float(np.nanpercentile(sector_z_avg, 98))
                _vmag = max(abs(_vlo), abs(_vhi))
                _x_edges = np.linspace(t_rel_sec[0], t_rel_sec[-1],
                                       sector_z_avg.shape[1] + 1)
                _y_edges = np.arange(sector_z_avg.shape[0] + 1)
                _im = ax_heat.pcolormesh(
                    _x_edges, _y_edges, sector_z_avg,
                    cmap='gray', vmin=-_vmag, vmax=_vmag,
                    shading='flat')
                ax_heat.set_ylim(sector_z_avg.shape[0], 0)
                ax_heat.set_yticks([0, sector_z_avg.shape[0]])
                _cax = ax_heat.inset_axes([1.015, 0, 0.012, 1])
                fig.colorbar(_im, cax=_cax)
                _cax.tick_params(labelsize=7)
                _cax.set_ylabel('dF/F0 (%)' if _dff_mode else 'z-score',
                                fontsize=7)
            else:
                ax_heat.text(0.5, 0.5, 'no stim events in range',
                             ha='center', va='center',
                             transform=ax_heat.transAxes,
                             fontsize=8, color='grey')

            ax_heat.axvline(x=0, color=sns.xkcd_rgb['bright red'],
                            linewidth=0.8, linestyle='--', alpha=0.8)
            if _rew_t_rel is not None:
                ax_heat.axvline(x=_rew_t_rel,
                                color=sns.xkcd_rgb['bright blue'],
                                linewidth=0.8, linestyle='--',
                                alpha=0.8)
            ax_heat.set_ylabel(heat_label, fontsize=8)
            ax_heat.set_xlabel('time from stim (s)')

            for _ax in (ax_trace, ax_heat):
                for _side in ('top', 'right'):
                    _ax.spines[_side].set_visible(False)
            plt.setp(ax_trace.get_xticklabels(), visible=False)
            ax_trace.set_xlim(t_rel_sec[0], t_rel_sec[-1])

            # Bottom row: square spatial map of integrated stim-triggered
            # response per sector (red colormap; brightest = strongest).
            # ----------
            if ax_spatial is not None:
                if (resp_grid is not None
                        and np.any(np.isfinite(resp_grid))):
                    _vlo = float(np.nanpercentile(resp_grid, 2))
                    _vhi = float(np.nanpercentile(resp_grid, 98))
                    if _vhi <= _vlo:
                        _vhi = _vlo + 1e-9
                    _x_e = np.arange(resp_grid.shape[1] + 1)
                    _y_e = np.arange(resp_grid.shape[0] + 1)
                    _im_sp = ax_spatial.pcolormesh(
                        _x_e, _y_e, resp_grid,
                        cmap=spatial_cmap, vmin=_vlo, vmax=_vhi,
                        shading='flat')
                    ax_spatial.set_aspect('equal')
                    ax_spatial.invert_yaxis()
                    ax_spatial.set_xticks([])
                    ax_spatial.set_yticks([])
                    _cax = ax_spatial.inset_axes([1.03, 0, 0.04, 1])
                    fig.colorbar(_im_sp, cax=_cax)
                    _cax.tick_params(labelsize=7)
                    _cax.set_ylabel('∫ response\n(stim→rew)', fontsize=7)
                else:
                    ax_spatial.text(0.5, 0.5, 'no response data',
                                    ha='center', va='center',
                                    transform=ax_spatial.transAxes,
                                    fontsize=8, color='grey')
                    ax_spatial.set_xticks([])
                    ax_spatial.set_yticks([])
                ax_spatial.set_ylabel(spatial_label, fontsize=8)

            # Stats blurb under the column (red and red_corr only).
            # Anchored to the bottom-row axis when present, else the
            # heatmap axis.
            # ----------
            if stats_text:
                _anchor = ax_spatial if ax_spatial is not None \
                    else ax_heat
                _anchor.text(0.0, -0.18, stats_text,
                             transform=_anchor.transAxes,
                             ha='left', va='top',
                             fontsize=7, color='k',
                             family='monospace')

            return ax_trace

        # Unit suffix for axis / colour-bar labels: dF/F0 (%) when dff
        # is on, or when dff_sig per-trial baseline normalisation is on.
        # ITI z-score otherwise.
        _unit_suffix = ('(dF/F %)' if (_dff_mode or dff_sig)
                        else '(z)')

        # Leftmost column: raw red fluorescence (dF/F0 if dff=True,
        # else ITI z-score). Sectors sorted with the same ordering used
        # by the corrected/red-grn column. Spatial map shown.
        _ax_trace_red = _draw_column(
            col=0,
            trace_z=_wf_red,
            sector_z_avg=sector_avg_r,
            resp_grid=_resp_grid_r,
            trace_label=f'whole-frame\nred {_unit_suffix}',
            heat_label=(f'sector\nred {_unit_suffix}\n'
                        f'(1..{n_sec_total})'),
            spatial_label='sector map\nred',
            trace_color=sns.xkcd_rgb['bright red'],
            draw_spatial=True,
            stats_text=_stats_red)

        # Second column: raw green fluorescence. Same ordering as red,
        # with its own spatial map (Greens cmap) in the bottom row.
        _ax_trace_grn = _draw_column(
            col=1,
            trace_z=_wf_grn,
            sector_z_avg=sector_avg_g,
            resp_grid=_resp_grid_g,
            trace_label=f'whole-frame\ngrn {_unit_suffix}',
            heat_label=(f'sector\ngrn {_unit_suffix}\n'
                        f'(1..{n_sec_total})'),
            spatial_label='sector map\ngrn',
            trace_color=sns.xkcd_rgb['forest green'],
            draw_spatial=True,
            spatial_cmap='Greens')

        # Red and green whole-frame average traces share one y-axis range
        # (union of both autoscaled limits) so their dF/F amplitudes are
        # directly comparable by eye.
        # ----------
        _rg_lo = min(_ax_trace_red.get_ylim()[0],
                     _ax_trace_grn.get_ylim()[0])
        _rg_hi = max(_ax_trace_red.get_ylim()[1],
                     _ax_trace_grn.get_ylim()[1])
        _ax_trace_red.set_ylim(_rg_lo, _rg_hi)
        _ax_trace_grn.set_ylim(_rg_lo, _rg_hi)

        # Third column: red/grn ratio. Spatial map is intentionally
        # omitted (the raw-red column already shows a spatial map of the
        # same FOV; the corrected channel will show its own below).
        _draw_column(
            col=2,
            trace_z=_wf_ratio,
            sector_z_avg=sector_avg,
            resp_grid=None,
            trace_label=f'whole-frame\nred/grn {_unit_suffix}',
            heat_label=(f'sector\nred/grn {_unit_suffix}\n'
                        f'(1..{n_sec_total})'),
            spatial_label='',
            trace_color=sns.xkcd_rgb['orange'],
            draw_spatial=False)

        if _has_corr:
            # Corrected real-channel column is always shown z-sorted
            # (ITI z-score), even when dff_sig=True. dff_override=False
            # forces the helper to skip per-trial baseline normalisation.
            _corr_unit = '(dF/F %)' if _dff_mode else '(z)'
            _draw_column(
                col=3,
                trace_z=_wf_corr,
                sector_z_avg=sector_avg_corr,
                resp_grid=_resp_grid_corr,
                trace_label=(f'whole-frame\n{_corr_key} '
                             f'{_corr_unit}'),
                heat_label=(f'sector\n{_corr_key} {_corr_unit}\n'
                            f'(1..{sector_avg_corr.shape[0]})'),
                spatial_label=f'sector map\n{_corr_key}',
                trace_color=sns.xkcd_rgb.get(
                    'bright orange', sns.xkcd_rgb['orange']),
                draw_spatial=True,
                dff_override=False,
                stats_text=_stats_corr)

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

        _title = (f'{self.path.animal} {self.path.date} '
                  f'{self.path.beh_folder} '
                  f'— QC (stimulus-aligned GRAB)')
        # Append correction method + fit_mode so the figure caption
        # itself records how the red_corr column was produced.
        _cs_kw = getattr(self.qc, 'correct_signal_kwargs', None) or {}
        _cs_method = _cs_kw.get('method', None)
        if _cs_method == 'linear_martianova':
            _fm = _cs_kw.get('fit_mode', 'global')
            _title = _title + f'  [corr: martianova, fit={_fm}]'
        elif _cs_method is not None:
            _title = _title + f'  [corr: {_cs_method}]'
        _sort_lbl_esc = _sort_label.replace('_', r'\_')
        _title = (_title + r'  $\mathbf{(sectors\ sorted\ by\ '
                  + _sort_lbl_esc + r')}$')
        fig.suptitle(_title, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if save:
            _ch_suffix = f'_ch={channel}' if channel is not None else ''
            _cs_suffix = self._qc_corrsig_suffix()
            _fname = (f'{self.path.animal}_{self.path.date}_'
                      f'{self.path.beh_folder}'
                      f'_qc_ratio_event_avg{_ch_suffix}'
                      f'{_cs_suffix}.pdf')
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
                _mn = _arr.mean(axis=0)
                _sm = _arr.std(axis=0) / np.sqrt(_arr.shape[0])
                return _mn, _sm

            _r_mn, _r_sm = _wf_mean_sem(_wf_red)
            _g_mn, _g_sm = _wf_mean_sem(_wf_grn)
            _ratio_mn, _ratio_sm = _wf_mean_sem(_wf_ratio)
            _corr_mn = _corr_sm = None
            if _has_corr:
                # corr column is always z-sorted regardless of dff_sig.
                _corr_mn, _corr_sm = _wf_mean_sem(_wf_corr,
                                                  dff_override=False)

            _stats = {
                'whole_frame_pearson': {
                    'red_vs_grn': {'r': _pear_rg_r,
                                   'p': _pear_rg_p},
                },
                'sector_pearson': {
                    'red_vs_grn': {'r': _pear_sec_rg_r,
                                   'p': _pear_sec_rg_p,
                                   'null': np.asarray(_pear_sec_rg_null)},
                },
                'sector_spearman': {
                    'red_vs_grn': {'rho': _spear_rg_rho,
                                   'p': _spear_rg_p,
                                   'null': np.asarray(_spear_rg_null)},
                },
                'sector_response_maps': {
                    'red': np.asarray(_resp_r),
                    'grn': np.asarray(_resp_g),
                },
            }
            if _has_corr:
                _stats['whole_frame_pearson'][f'{_corr_key}_vs_grn'] = {
                    'r': _pear_cg_r, 'p': _pear_cg_p}
                _stats['sector_pearson'][f'{_corr_key}_vs_grn'] = {
                    'r': _pear_sec_cg_r, 'p': _pear_sec_cg_p,
                    'null': np.asarray(_pear_sec_cg_null)}
                _stats['sector_spearman'][f'{_corr_key}_vs_grn'] = {
                    'rho': _spear_cg_rho, 'p': _spear_cg_p,
                    'null': np.asarray(_spear_cg_null)}
                _stats['sector_response_maps'][_corr_key] = \
                    np.asarray(_resp_corr)

            _cs_kw_save = (
                getattr(self.qc, 'correct_signal_kwargs', None) or {})
            _summary = {
                'mode': 'dff_sig' if dff_sig else 'iti_zscore',
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
                    with np.errstate(divide='ignore', invalid='ignore'):
                        _ratio_h = _ff_r_h / (_ff_g_h + 1e-9)
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
                            _c_in = _zscore_iti(
                                np.asarray(_ff_c_h, dtype=np.float64), t)
                            _mn, _sm = _wf_mean_sem(_c_in,
                                                    dff_override=False)
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
                         f'{_ch_suffix}{_cs_suffix}{_dff_suffix}.npy')
            np.save(os.path.join(str(self.folder.figs), _npy_name),
                    _summary, allow_pickle=True)

        return fig
