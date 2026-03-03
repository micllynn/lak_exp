"""
twop_plots.py  –  PlotsMixin for TwoPRec.

All plotting methods (plt_*, run_and_plt_*, _plt_*, _finish_plot) extracted
from load_exp_twop.py.

All figure-save paths use self.folder.figs (an absolute pathlib.Path set by
TwoPRec._init_folders) rather than os.getcwd().
"""

import os
from types import SimpleNamespace

import numpy as np
import scipy as sp
import scipy.stats as sp_stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import sklearn
import sklearn.linear_model
import sklearn.preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

from .utils_twop import calc_dff


class PlotsMixin:
    """Plotting methods mixed into TwoPRec."""

    # ------------------------------------
    # Shared helper
    # ------------------------------------

    def _finish_plot(self, fig, ax, xlabel, ylabel,
                     savepath=None, show=True):
        """Finalize a plot with labels and optional save/show."""
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if savepath is not None:
            fig.savefig(savepath)
        if show:
            plt.show()

    # ------------------------------------
    # Frame / sector / neur plots
    # ------------------------------------

    def plt_frame(self,
                  figsize=(3.43, 2),
                  t_pre=2, t_post=5,
                  colors=None,
                  plot_type=None,
                  plt_show=True,
                  channel=None,
                  use_zscore=False):
        """
        Plots the average fluorescence across the whole fov,
        separated by trial-type.

        Parameters
        ----------
        plot_type : str or None
            None: plots 0, 0.5, 1
            'rew_norew': plots 0.5_rew, 0.5_norew
            'prelick_noprelick': plots 0.5_prelick, 0.5_noprelick
        channel : str or None
            Passed to add_frame() / _get_rec(). Ignored in the base class.
        use_zscore : bool
            Passed to add_frame(). If True, use z-score instead of df/f.
        """
        if colors is None:
            colors = sns.cubehelix_palette(
                n_colors=3, start=2, rot=0, dark=0.1, light=0.6)

        self.add_frame(t_pre=t_pre, t_post=t_post,
                       channel=channel, use_zscore=use_zscore)

        # build color and linestyle maps
        if hasattr(self.beh, 'tr_conds_outcome'):
            _cmap = sns.cubehelix_palette(
                as_cmap=True, start=2, rot=0, dark=0.1, light=0.6)
            _outcomes = np.asarray(self.beh.tr_conds_outcome, dtype=float)
            _order = np.argsort(_outcomes, kind='mergesort')
            _tile = np.linspace(0.0, 1.0, len(_order)) \
                if len(_order) > 1 else np.array([0.5])
            colors = {}
            for _rank, _idx in enumerate(_order):
                _tr_cond = str(self.beh._stimparser.parsed_param[_idx])
                colors[_tr_cond] = _cmap(_tile[_rank])
        else:
            _palette = sns.cubehelix_palette(
                n_colors=len(self.beh.tr_inds), start=2, rot=0,
                dark=0.1, light=0.6)
            colors = {k: _palette[i] for i, k in enumerate(self.beh.tr_inds)}
        linestyles = {k: 'solid' for k in self.beh.tr_inds}
        if '0.5_norew' in linestyles:
            linestyles['0.5_norew'] = 'dashed'
        if '0.5_noprelick' in linestyles:
            linestyles['0.5_noprelick'] = 'dashed'

        fig_avg = plt.figure(figsize=figsize)
        spec = gs.GridSpec(nrows=1, ncols=1,
                           figure=fig_avg)
        ax_trace = fig_avg.add_subplot(spec[0, 0])

        # determine which conditions to plot
        if plot_type is None:
            tr_conds = list(self.beh.tr_conds)
        elif plot_type == 'rew_norew':
            tr_conds = ['0.5_rew', '0.5_norew']
        elif plot_type == 'prelick_noprelick':
            tr_conds = ['0.5_prelick', '0.5_noprelick']

        # plot stim-aligned traces
        for ind, tr_cond in enumerate(tr_conds):
            tr_cond_outcome = self.beh.tr_conds_outcome[ind]
            _color = colors.get(tr_cond, 'grey')
            _ls = linestyles.get(tr_cond, 'solid')
            ax_trace.plot(self.frame.t,
                          np.mean(self.frame.dff[tr_cond], axis=0),
                          color=_color,
                          linestyle=_ls,
                          label=f'{tr_cond} ({tr_cond_outcome}uL)')
            ax_trace.fill_between(
                self.frame.t,
                np.mean(self.frame.dff[tr_cond], axis=0) -
                np.std(self.frame.dff[tr_cond], axis=0),
                np.mean(self.frame.dff[tr_cond], axis=0) +
                np.std(self.frame.dff[tr_cond], axis=0),
                facecolor=_color,
                alpha=0.2)

        # plot rew and stim onset markers
        ax_trace.axvline(x=0, color=sns.xkcd_rgb['dark grey'],
                         linewidth=1.5, alpha=0.8)
        ax_trace.axvline(x=2, color=sns.xkcd_rgb['bright blue'],
                         linewidth=1.5, alpha=0.8)

        ax_trace.legend()

        ax_trace.set_xlabel('time (s)')
        ax_trace.set_ylabel('z-score (frame)' if use_zscore
                            else 'df/f (frame)')

        _ch_suffix = f'ch={channel}' if channel is not None else \
            (f'ch={self.ch_img}' if hasattr(self, 'ch_img') else '')
        corr_suffix = ''
        if channel == 'grn' and hasattr(self, 'rec_t_grn') \
           and hasattr(self, 'corr_sig_method'):
            corr_suffix = f'_corr={self.corr_sig_method}'
        zscore_suffix = '_zscore' if use_zscore else ''
        fig_avg.savefig(os.path.join(
            str(self.folder.figs),
            f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{plot_type=}_{t_pre=}_{t_post=}_'
            + f'{_ch_suffix}{corr_suffix}'
            + f'{zscore_suffix}_mean_trial_activity.pdf'))

        if plt_show:
            plt.show()
        else:
            plt.close(fig_avg)

    def plt_sectors(self,
                    n_sectors=10,
                    t_pre=2, t_post=5,
                    plot_type=None,
                    resid_type='sector',
                    resid_tr_cond='1',
                    resid_alpha=0.5,
                    resid_smooth=True,
                    resid_smooth_windlen=5,
                    resid_smooth_polyorder=3,
                    resid_trial_nbins=30,
                    figsize=(6, 6),
                    img_ds_factor=50,
                    img_alpha=0.5,
                    plt_dff=None,
                    plt_prefix='',
                    plt_show=True,
                    channel=None,
                    use_zscore=False,
                    outlier_thresh=None,
                    auto_gain_dff=False,
                    minimal_output=False,
                    colors=None):
        """
        Divides the field of view into sectors, and plots a set of trial-types
        separately within each sector.
        """
        if plt_dff is None:
            plt_dff = {'x': {'gain': 0.6, 'offset': 0.2},
                       'y': {'gain': 10, 'offset': 0.2}}
        if colors is None:
            colors = sns.cubehelix_palette(
                n_colors=3, start=2, rot=0, dark=0.2, light=0.8)

        rec = self._get_rec(channel)
        rec_t = self._get_rec_t(channel)
        n_frames_pre, _, n_frames_tot, _ = self._aligned_frame_params(
            t_pre, t_post)

        # ---- resolve _use_zscore ----
        _use_zscore = use_zscore
        if not _use_zscore and \
           self.beh._stimrange.first < self.beh._stimrange.last:
            _t0 = self.beh.stim.t_start[self.beh._stimrange.first]
            _ind0 = np.argmin(np.abs(rec_t - (_t0 - t_pre)))
            _probe = np.mean(rec[_ind0:_ind0 + n_frames_pre], axis=(1, 2))
            _f0_probe = float(np.mean(_probe)) if _probe.size > 0 else 1.0
            if np.abs(_f0_probe) < 1.0:
                _use_zscore = True

        if _use_zscore and not auto_gain_dff:
            auto_gain_dff = True

        if outlier_thresh is None:
            outlier_thresh = 20 if _use_zscore else 1

        # ---- stim_list from plot_type ----
        _stim_list_map = {
            None: ['0', '0.5', '1'],
            'rew_norew': ['0.5_rew', '0.5_norew'],
            'prelick_noprelick': ['0.5_prelick', '0.5_noprelick'],
            'rew': ['0', '0.5_rew', '1'],
        }
        stim_list = _stim_list_map.get(plot_type, list(self.beh.tr_conds))

        # ---- ensure sector data ----
        _sector_cache_valid = (
            hasattr(self, 'sector')
            and hasattr(self.sector, 'dff')
            and hasattr(self.sector, 'params')
            and self.sector.n_sectors == n_sectors
            and self.sector.params.t_rew_pre == t_pre
            and self.sector.params.t_rew_post == t_post
            and self.sector.params.channel == channel
            and self.sector.params.use_zscore == _use_zscore
            and isinstance(self.sector.dff[0, 0], dict)
        )

        if not _sector_cache_valid:
            self.add_sectors(
                n_sectors=n_sectors, t_pre=t_pre, t_post=t_post,
                resid_type=resid_type,
                resid_corr_tr_cond=resid_tr_cond,
                channel=channel, use_zscore=_use_zscore,
                compute_resid_corr=False, compute_null=False)
        elif self.sector.params.resid_type != resid_type:
            print('\trecomputing residuals (resid_type changed)...')
            for n_x in range(n_sectors):
                for n_y in range(n_sectors):
                    for _tr_cond in self.sector.params.resid_tr_conds:
                        _dff_arr = self.sector.dff[n_x, n_y][_tr_cond]
                        if resid_type == 'sector':
                            self.sector.dff_resid[n_x, n_y][_tr_cond] = (
                                _dff_arr - np.mean(_dff_arr, axis=0))
                        elif resid_type == 'trial':
                            self.sector.dff_resid[n_x, n_y][_tr_cond] = (
                                _dff_arr - self.frame.dff[_tr_cond])
            self.sector.params.resid_type = resid_type

        # ---- colors/linestyle ----
        self.sector.colors = {'0': colors[0], '0.5': colors[1], '1': colors[2],
                              '0.5_rew': colors[1], '0.5_norew': colors[1],
                              '0.5_prelick': colors[1],
                              '0.5_noprelick': colors[1]}
        self.sector.linestyle = {'0': 'solid', '0.5': 'solid', '1': 'solid',
                                 '0.5_rew': 'solid', '0.5_norew': 'dashed',
                                 '0.5_prelick': 'solid',
                                 '0.5_noprelick': 'dashed'}

        # ---- figure creation ----
        fig_avg = plt.figure(figsize=figsize)
        spec_avg = gs.GridSpec(nrows=1, ncols=1, figure=fig_avg)
        ax_img = fig_avg.add_subplot(spec_avg[0, 0])

        fig_resid, ax_img_resid = None, None
        if not minimal_output:
            fig_resid = plt.figure(figsize=figsize)
            spec_resid = gs.GridSpec(nrows=1, ncols=1, figure=fig_resid)
            ax_img_resid = fig_resid.add_subplot(spec_resid[0, 0])
            ax_img_resid.set_title(
                f'residuals vs mean({resid_type}), p(rew)={resid_tr_cond}')

        print('\tcreating max projection image...')
        _rec_max = np.max(rec[::img_ds_factor, :, :], axis=0)
        _step = int(_rec_max.shape[0] / n_sectors)
        _rec_max[:, ::_step] = 0
        _rec_max[::_step, :] = 0
        _extent = [0, n_sectors, n_sectors, 0]
        ax_img.imshow(_rec_max, extent=_extent, alpha=img_alpha)
        if not minimal_output:
            ax_img_resid.imshow(_rec_max, extent=_extent, alpha=img_alpha)

        # ---- auto-gain ----
        if auto_gain_dff:
            _max_pos = max(
                (float(np.max(np.median(self.sector.dff[gx, gy][sc], axis=0)))
                 for gx in range(n_sectors)
                 for gy in range(n_sectors)
                 for sc in stim_list
                 if sc in self.sector.dff[gx, gy]),
                default=0.0)
            if _max_pos > 0:
                plt_dff = {**plt_dff,
                           'y': {**plt_dff['y'],
                                 'gain': (1.0 - plt_dff['y']['offset'])
                                         / _max_pos}}

        # ---- plot all sectors ----
        for n_x in range(n_sectors):
            for n_y in range(n_sectors):
                _t_sector = ((self.sector.t / self.sector.t[-1])
                             * plt_dff['x']['gain']
                             + n_x + plt_dff['x']['offset'])
                _dff_zero = n_y - plt_dff['y']['offset'] + 1
                _vkw = dict(linewidth=1, linestyle='dashed')

                for stim_cond in stim_list:
                    _dff_mean = np.median(
                        self.sector.dff[n_x, n_y][stim_cond], axis=0)
                    if np.max(np.abs(_dff_mean)) < outlier_thresh:
                        _shifted = (-_dff_mean * plt_dff['y']['gain']
                                    + _dff_zero)
                        ax_img.plot(_t_sector, _shifted,
                                    color=self.sector.colors[stim_cond],
                                    linestyle=self.sector.linestyle[stim_cond])

                if not minimal_output:
                    _resid_arr = self.sector.dff_resid[n_x, n_y][resid_tr_cond]
                    _n_tr_resid = _resid_arr.shape[0]
                    _rocket = sns.color_palette('rocket', _n_tr_resid)
                    for trial in range(_n_tr_resid):
                        _dff = _resid_arr[trial, :]
                        if resid_smooth:
                            _dff = sp.signal.savgol_filter(
                                _dff, resid_smooth_windlen,
                                resid_smooth_polyorder)
                        if np.max(np.abs(_dff)) < outlier_thresh:
                            _shifted = (-_dff * plt_dff['y']['gain']
                                        + _dff_zero)
                            ax_img_resid.plot(
                                _t_sector, _shifted,
                                color=_rocket[trial], alpha=resid_alpha,
                                linestyle=self.sector.linestyle[resid_tr_cond])

                _t_stim = n_x + plt_dff['x']['offset']
                _t_rew = ((2 / self.sector.t[-1]) * plt_dff['x']['gain']
                          + n_x + plt_dff['x']['offset'])
                ax_img.plot([_t_stim, _t_stim], [n_y + 0.2, n_y + 0.8],
                            color=sns.xkcd_rgb['dark grey'], **_vkw)
                ax_img.plot([_t_rew, _t_rew], [n_y + 0.2, n_y + 0.8],
                            color=sns.xkcd_rgb['bright blue'], **_vkw)
                if not minimal_output:
                    ax_img_resid.plot([_t_stim, _t_stim],
                                      [n_y + 0.2, n_y + 0.8],
                                      color=sns.xkcd_rgb['dark grey'], **_vkw)
                    ax_img_resid.plot([_t_rew, _t_rew],
                                      [n_y + 0.2, n_y + 0.8],
                                      color=sns.xkcd_rgb['bright blue'], **_vkw)
                    ax_img_resid.plot([_t_stim, _t_rew],
                                      [_dff_zero, _dff_zero],
                                      color='k', **_vkw)

        # ---- build filename base strings ----
        corr_suffix = (f'_corr={self.corr_sig_method}'
                       if channel == 'grn' and hasattr(self, 'rec_t_grn')
                       and hasattr(self, 'corr_sig_method') else '')
        zscore_suffix = f'_zscore={_use_zscore}'
        _figs_dir = str(self.folder.figs)
        _id = (f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}'
               f'_{plt_prefix}_{n_sectors=}_{plot_type=}_{t_pre=}_{t_post=}'
               f'_ch={channel}{corr_suffix}{zscore_suffix}')
        _id_resid = f'{_id}_{resid_tr_cond=}_{resid_type=}'

        fig_avg.savefig(os.path.join(_figs_dir, f'sectors_{_id}.pdf'))

        # ---- supplemental plots ----
        if not minimal_output:
            n_trs = self.sector.dff_resid[0, 0][resid_tr_cond].shape[0]
            n_rows = int(np.ceil(np.sqrt(n_trs)))

            # normalized sector signals
            fig_norm_sec_sig = plt.figure(figsize=(3.43, 1.5))
            spec_norm_sec_sig = gs.GridSpec(nrows=1, ncols=3,
                                            figure=fig_norm_sec_sig)
            axes_norm = [fig_norm_sec_sig.add_subplot(spec_norm_sec_sig[0, i])
                         for i in range(3)]
            for tr_type, ax_n in zip(['0', '0.5', '1'], axes_norm):
                for sec in range(self.sector.n_sectors):
                    sec_x, sec_y = np.divmod(sec, self.sector.n_sectors)
                    _dff = self.sector.dff[sec_x, sec_y][tr_type]
                    _sec_max = np.max(_dff, axis=1, keepdims=True)
                    _sec_max[_sec_max == 0] = 1
                    ax_n.plot(self.sector.t,
                              np.mean(_dff / _sec_max, axis=0),
                              color=self.sector.colors[tr_type], alpha=0.8)

            # residual cross-correlations (vectorized)
            _n_sec2 = self.sector.n_sectors ** 2
            _sigs = np.stack([
                np.mean(self.sector.dff_resid[
                    s // n_sectors, s % n_sectors][resid_tr_cond], axis=1)
                for s in range(_n_sec2)])
            self.sector.resid_corr_stat = np.corrcoef(_sigs)
            self.sector.resid_corr_pval = np.full_like(
                self.sector.resid_corr_stat, np.nan)

            fig_resid_corr = plt.figure(figsize=figsize)
            spec_resid_corr = gs.GridSpec(nrows=1, ncols=1,
                                          figure=fig_resid_corr)
            ax_resid_corr = fig_resid_corr.add_subplot(spec_resid_corr[0, 0])
            ax_resid_corr.imshow(self.sector.resid_corr_stat)

            fig_resid_corr_spatial = plt.figure(figsize=figsize)
            spec_rcs = gs.GridSpec(n_sectors, n_sectors,
                                   figure=fig_resid_corr_spatial)
            for s in range(_n_sec2):
                sx, sy = divmod(s, n_sectors)
                ax_s = fig_resid_corr_spatial.add_subplot(spec_rcs[sx, sy])
                ax_s.imshow(
                    self.sector.resid_corr_stat[s].reshape(
                        n_sectors, n_sectors),
                    vmax=1, vmin=-1, cmap='coolwarm')
                ax_s.set_xticks([])
                ax_s.set_yticks([])
            fig_resid_corr_spatial.suptitle(
                f'residual corrs (spatial), {resid_tr_cond=},'
                f' residuals vs mean({resid_type})')

            # per-trial residual distributions
            fig_resid_tr_sep = plt.figure(figsize=figsize)
            spec_tr = gs.GridSpec(n_rows, n_rows, figure=fig_resid_tr_sep)
            ax0 = None
            for tr in range(n_trs):
                _sector_dff_resid = [
                    float(np.mean(self.sector.dff_resid[
                        s // n_sectors, s % n_sectors][resid_tr_cond][tr, :]))
                    for s in range(_n_sec2)]
                _tr_x, _tr_y = divmod(tr, n_rows)
                _ax_kw = dict(sharex=ax0, sharey=ax0) if ax0 is not None else {}
                ax_tr = fig_resid_tr_sep.add_subplot(
                    spec_tr[_tr_x, _tr_y], **_ax_kw)
                if ax0 is None:
                    ax0 = ax_tr
                ax_tr.hist(_sector_dff_resid, bins=resid_trial_nbins,
                           histtype='step', density=True,
                           color=sns.xkcd_rgb['dark grey'])
                ax_tr.axvline(0, color='k', linewidth=1, linestyle='dashed')
                ax_tr.set_xlabel('dff resid.')
                ax_tr.set_ylabel('pdf')
                ax_tr.set_title(f'trial={tr}')
            fig_resid_tr_sep.suptitle(
                f'residuals vs mean({resid_type}) corrs (spatial),'
                f' {resid_tr_cond=}, residuals vs mean({resid_type})')

            fig_norm_sec_sig.savefig(
                os.path.join(_figs_dir, f'sectors_normed_{_id}.pdf'))
            fig_resid.savefig(
                os.path.join(_figs_dir, f'sectors_resid_{_id_resid}.pdf'))
            fig_resid_corr.savefig(
                os.path.join(_figs_dir,
                             f'sectors_resid_corr_{_id_resid}.pdf'))
            fig_resid_corr_spatial.savefig(
                os.path.join(_figs_dir,
                             f'sectors_resid_corr_spatial_{_id_resid}.pdf'))
            fig_resid_tr_sep.savefig(
                os.path.join(_figs_dir,
                             f'sectors_resid_trial_sep_{_id_resid}.pdf'))

        # ---- show/close ----
        if plt_show:
            plt.show()
        else:
            plt.close(fig_avg)
            if not minimal_output:
                for _fig in [fig_norm_sec_sig, fig_resid, fig_resid_corr,
                             fig_resid_corr_spatial, fig_resid_tr_sep]:
                    if _fig is not None:
                        plt.close(_fig)

        return

    def plt_neurs(self, t_pre=1, t_post=2,
                  figsize=(3.43, 3.43), zscore=False,
                  plt_equal=True,
                  sort_by='trial_type',
                  sort_tr_type='max',
                  cmap=None,
                  title_fontsize=8):
        """
        Plots trial-averaged neural activity (dff) for each cell as a heatmap.

        Parameters
        ----------
        sort_tr_type : str
            Trial condition key used to sort neurons when sort_by='trial_type'.
            'max' : uses tr_cond with highest outcome value.
            'min' : uses tr_cond with lowest outcome value.
        """
        if cmap is None:
            cmap = sns.diverging_palette(220, 20, as_cmap=True)

        # plot neuron masks
        self.plt_neur_masks(show=False)

        self.add_lickrates(t_prestim=t_pre,
                           t_poststim=t_post)

        # setup structure
        neurs = np.where(self.neur.iscell[:, 0] == 1)[0]
        n_neurs = neurs.shape[0]

        n_frames_pre = int(t_pre * self.samp_rate)
        n_frames_post = int((2 + t_post) * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post

        # resolve 'max'/'min' sort_tr_type
        if sort_tr_type in ('max', 'min'):
            _outcomes = self.beh.tr_conds_outcome
            _pick = np.argmax(_outcomes) if sort_tr_type == 'max' \
                else np.argmin(_outcomes)
            sort_tr_type = self.beh.tr_conds[_pick]

        _tr_conds_all = list(self.beh.tr_inds.keys())
        _tr_conds_plot = list(self.beh.tr_conds)

        self.neur.dff_aln = np.empty(n_neurs, dtype=dict)
        self.neur.dff_aln_mean = {}
        for _tr_cond in _tr_conds_all:
            self.neur.dff_aln_mean[_tr_cond] = np.zeros(
                (n_neurs, n_frames_tot))
        self.neur.x = np.zeros(n_neurs)
        self.neur.y = np.zeros(n_neurs)

        self.neur.t_aln = np.linspace(
            -1*t_pre, 2+t_post, n_frames_tot)

        # iterate through neurons
        for ind, neur in enumerate(neurs):
            self.neur.x[ind] = self.neur.stat[neur]['med'][0]
            self.neur.y[ind] = self.neur.stat[neur]['med'][1]
            self.neur.x[ind] -= self.ops.n_px_remove_sides
            self.neur.y[ind] -= self.ops.n_px_remove_sides

            self.neur.dff_aln[ind] = {}
            for _tr_cond in _tr_conds_all:
                self.neur.dff_aln[ind][_tr_cond] = np.zeros((
                    self.beh.tr_inds[_tr_cond].shape[0], n_frames_tot))

            _tr_counts = {k: 0 for k in _tr_conds_all}

            if zscore is True:
                _f_full = sp.stats.zscore(self.neur.f[neur, :])
            else:
                _f_full = self.neur.f[neur, :]

            for trial in range(self.beh._stimrange.first,
                               self.beh._stimrange.last):
                print(f'\t\t\tneur={int(neur)} | {trial=}', end='\r')
                _t_stim = self.beh.stim.t_start[trial]
                _t_start = _t_stim - t_pre

                _ind_t_start = np.argmin(np.abs(
                    self.neur.t - _t_start))

                _f = _f_full[_ind_t_start:_ind_t_start + n_frames_tot]
                _dff = calc_dff(_f, baseline_frames=n_frames_pre)

                for _cond in self._trial_cond_map.get(int(trial), []):
                    if _cond in self.neur.dff_aln[ind]:
                        self.neur.dff_aln[ind][_cond][
                            _tr_counts[_cond], :] = _dff[0:n_frames_tot]
                        _tr_counts[_cond] += 1

            for _tr_cond in _tr_conds_all:
                self.neur.dff_aln_mean[_tr_cond][ind, :] = \
                    np.mean(self.neur.dff_aln[ind][_tr_cond], axis=0)

        # process data for plotting
        _ind_stim_start = np.argmin(np.abs(self.neur.t_aln))
        _ind_stim_end = np.argmin(np.abs(self.neur.t_aln - 2))
        _ind_rew_end = np.argmin(np.abs(self.neur.t_aln - (2 + t_post)))

        if sort_by == 'trial_type':
            _sort_metric = np.max(
                self.neur.dff_aln_mean[sort_tr_type][
                    :, _ind_stim_start:_ind_stim_end],
                axis=1)
            sort_inds_corr = np.argsort(_sort_metric)

        elif sort_by == 'selectivity':
            _cond_max = _tr_conds_plot[-1]
            _cond_min = _tr_conds_plot[0]
            _sort_metric = np.abs(np.max(
                self.neur.dff_aln_mean[_cond_max][
                    :, _ind_stim_start:_ind_stim_end]
                - self.neur.dff_aln_mean[_cond_min][
                    :, _ind_stim_start:_ind_stim_end],
                axis=1))
            sort_inds_corr = np.argsort(_sort_metric)

        elif sort_by == 'reward':
            _cond_max = _tr_conds_plot[-1]
            _sort_metric = np.max(
                self.neur.dff_aln_mean[
                    _cond_max][:, _ind_stim_end:_ind_rew_end],
                axis=1)
            sort_inds_corr = np.argsort(_sort_metric)

        else:
            raise ValueError(
                "sort_by must be 'trial_type',"
                + f"'selectivity', or 'reward'; got {sort_by!r}"
            )

        plt_vmin = np.min(
            [self.neur.dff_aln_mean[k] for k in _tr_conds_plot])
        plt_vmax = np.max(
            [self.neur.dff_aln_mean[k] for k in _tr_conds_plot])

        if plt_equal is True:
            if abs(plt_vmin) < abs(plt_vmax):
                plt_vmin = -1 * plt_vmax
            elif abs(plt_vmin) > abs(plt_vmax):
                plt_vmax = -1 * plt_vmin

        n_conds = len(_tr_conds_plot)
        fig = plt.figure(figsize=figsize)
        spec = gs.GridSpec(nrows=1, ncols=n_conds, figure=fig)
        axes = [fig.add_subplot(spec[0, 0])]
        for _i in range(1, n_conds):
            axes.append(fig.add_subplot(spec[0, _i], sharey=axes[0]))

        _ind_zero = np.argmin(np.abs(self.neur.t_aln))
        _ind_rew = np.argmin(np.abs(self.neur.t_aln - 2))

        _outcomes_plot = (list(self.beh.tr_conds_outcome)
                          if hasattr(self.beh, 'tr_conds_outcome') else
                          [None] * len(_tr_conds_plot))
        for _ax, _tr_cond, _tr_cond_outcome in zip(
                axes, _tr_conds_plot, _outcomes_plot):
            _ax.pcolormesh(
                self.neur.dff_aln_mean[_tr_cond][sort_inds_corr],
                vmin=plt_vmin, vmax=plt_vmax,
                cmap=cmap)
            _ax.axvline(_ind_zero,
                        color=sns.xkcd_rgb['dark grey'],
                        linewidth=1,
                        linestyle='dashed')
            _ax.axvline(_ind_rew,
                        color=sns.xkcd_rgb['bright blue'],
                        linewidth=1,
                        linestyle='dashed')
            _ax.set_xticks(
                ticks=[0, _ind_zero, self.neur.t_aln.shape[-1]],
                labels=[f'{self.neur.t_aln[0]:.2f}', 0,
                        f'{self.neur.t_aln[-1]:.2f}'])
            _title = f'{self.beh._stimparser.parse_by}={_tr_cond}'
            if _tr_cond_outcome is not None:
                _title += f'\nOutcome: {_tr_cond_outcome}uL'
            _ax.set_title(_title, fontsize=title_fontsize)
        axes[0].set_ylabel('neuron')
        axes[n_conds // 2].set_xlabel('time from stim (s)')

        fig.savefig(os.path.join(
            str(self.folder.figs),
            f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{t_pre=}_{t_post=}_{zscore=}_{plt_equal=}_'
            + 'neur_trial_activity.pdf'))

        # second figure: population-averaged trace per trial condition
        if hasattr(self.beh, 'tr_conds_outcome'):
            _cmap = sns.cubehelix_palette(
                as_cmap=True, start=2, rot=0, dark=0.1, light=0.6).reversed()
            _outcomes = np.asarray(self.beh.tr_conds_outcome, dtype=float)
            _order = np.argsort(_outcomes, kind='mergesort')
            _tile = np.linspace(0.0, 1.0, len(_order)) \
                if len(_order) > 1 else np.array([0.5])
            _tr_colors = {}
            for _rank, _idx in enumerate(_order):
                _tr_cond = str(self.beh._stimparser.parsed_param[_idx])
                _tr_colors[_tr_cond] = _cmap(_tile[_rank])

            fig_pop, ax_pop = plt.subplots(1, 1, figsize=(3.43, 2))

            for _tr_cond, _tr_cond_outcome in zip(
                    _tr_conds_plot, self.beh.tr_conds_outcome):
                _color = _tr_colors[_tr_cond]
                _mean_pop = np.mean(
                    self.neur.dff_aln_mean[_tr_cond], axis=0)
                _sem_pop = (np.std(
                    self.neur.dff_aln_mean[_tr_cond], axis=0)
                    / np.sqrt(n_neurs))
                ax_pop.plot(
                    self.neur.t_aln, _mean_pop,
                    color=_color,
                    label=f'{_tr_cond} ({_tr_cond_outcome}uL)')
                ax_pop.fill_between(
                    self.neur.t_aln,
                    _mean_pop - _sem_pop,
                    _mean_pop + _sem_pop,
                    facecolor=_color,
                    alpha=0.2)

            ax_pop.axvline(x=0, color=sns.xkcd_rgb['dark grey'],
                           linewidth=1.5, alpha=0.8)
            ax_pop.axvline(x=2, color=sns.xkcd_rgb['bright blue'],
                           linewidth=1.5, alpha=0.8)
            ax_pop.legend(title=f'{self.beh._stimparser.parse_by} (outcome)')
            ax_pop.set_xlabel('time from stim (s)')
            ax_pop.set_ylabel('mean df/f (neurs)')

            fig_pop.savefig(os.path.join(
                str(self.folder.figs),
                f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
                + f'{t_pre=}_{t_post=}_{zscore=}_'
                + 'neur_popmean_activity.pdf'))

        plt.show()

        return

    # ------------------------------------
    # Psychometric plots
    # ------------------------------------

    def plt_optimism(self,
                     markersize=20,
                     violin_color=None,
                     zetatest_mask=False,
                     sig_thresh=0.01,
                     fontsize=10,
                     ylims=None,
                     plt_show=True):
        if violin_color is None:
            violin_color = sns.xkcd_rgb['moss green']
        if ylims is None:
            ylims = [-2, 2]

        if not hasattr(self, 'psychometrics'):
            self.add_psychometrics()
        if zetatest_mask is True and not hasattr(self.sector, 'zeta_pval'):
            self.add_zetatest_sectors()
            self.add_zetatest_neurs()

        ind_sig_lower = (sig_thresh *
                         self.psychometrics.grab_null.optimism[0].shape[0])
        ind_sig_upper = (self.psychometrics.grab_null.optimism[0].shape[0]
                         - (sig_thresh *
                            self.psychometrics.grab_null.optimism[0].shape[0]))

        print('plotting optimism...')
        fig_distrl = plt.figure(figsize=(6, 3))
        spec = gs.GridSpec(nrows=2, ncols=2,
                           height_ratios=[0.05, 0.95],
                           figure=fig_distrl)
        ax_distrl_grab_signif = fig_distrl.add_subplot(spec[0, 0])
        ax_distrl_grab_signif.set_yticks([])
        ax_distrl_grab = fig_distrl.add_subplot(
            spec[1, 0], sharex=ax_distrl_grab_signif)

        ax_distrl_neur_signif = fig_distrl.add_subplot(spec[0, 1])
        ax_distrl_neur_signif.set_yticks([])
        ax_distrl_neur = fig_distrl.add_subplot(
            spec[1, 1], sharex=ax_distrl_neur_signif)

        fig_distrl_reliability = plt.figure(figsize=(3, 3))
        ax_distrl_reliability = fig_distrl_reliability.add_subplot()

        fig_img = plt.figure(figsize=(6, 2))
        spec_img = gs.GridSpec(nrows=1, ncols=3, figure=fig_img)
        ax_img_grab = fig_img.add_subplot(spec_img[0, 0])
        ax_img_neur = fig_img.add_subplot(spec_img[0, 1])
        ax_img_grab_neur = fig_img.add_subplot(spec_img[0, 2])

        n_sec = self.sector.n_sectors
        rec_px = self.rec.shape[1]

        if zetatest_mask is False:
            optimism = self.psychometrics.grab.optimism
            sort_args_grab = np.argsort(optimism)
            optimism_neur = self.psychometrics.neur.optimism
            sort_args_neur = np.argsort(optimism_neur)

            if ylims is None:
                vmax = np.max(np.abs(optimism))
                vmin = -1 * vmax
            else:
                vmax = ylims[1]
                vmin = ylims[0]

            ax_img_grab.imshow(
                optimism.reshape((n_sec, n_sec)),
                vmin=vmin, vmax=vmax,
                cmap='coolwarm')
            ax_img_grab_neur.imshow(
                optimism.reshape((n_sec, n_sec)),
                vmin=vmin, vmax=vmax,
                cmap='coolwarm')
            ax_img_neur.scatter((self.neur.x*(n_sec/rec_px)-0.5),
                                -1*(self.neur.y*(n_sec/rec_px)-0.5),
                                s=40,
                                c=self.psychometrics.neur.optimism,
                                edgecolors='black', cmap='coolwarm',
                                vmin=vmin, vmax=vmax)
            ax_img_grab_neur.scatter(self.neur.x*(n_sec/rec_px)-0.5,
                                     self.neur.y*(n_sec/rec_px)-0.5,
                                     s=40,
                                     c=self.psychometrics.neur.optimism,
                                     edgecolors='black', cmap='coolwarm',
                                     vmin=vmin, vmax=vmax)

            ax_distrl_grab.scatter(np.arange(optimism.shape[0]),
                                   optimism[sort_args_grab],
                                   s=markersize,
                                   c=sns.xkcd_rgb['black'])
            ax_distrl_neur.scatter(np.arange(optimism_neur.shape[0]),
                                   optimism_neur[sort_args_neur],
                                   s=markersize,
                                   c=sns.xkcd_rgb['black'])
            ax_distrl_grab.axhline(self.psychometrics.lick.optimism,
                                   linestyle='dashed',
                                   color=sns.xkcd_rgb['azure'],
                                   linewidth=0.5)
            ax_distrl_neur.axhline(self.psychometrics.lick.optimism,
                                   linestyle='dashed',
                                   color=sns.xkcd_rgb['azure'],
                                   linewidth=0.5)

            for ind, sort_arg in enumerate(sort_args_grab):
                _sorted_null = np.sort(
                    self.psychometrics.grab_null.optimism[sort_arg])
                _rank = np.argmin(np.abs(
                    _sorted_null - optimism[sort_arg]))
                if _rank <= ind_sig_lower or _rank >= ind_sig_upper:
                    ax_distrl_grab_signif.text(
                        ind, 0, '*', ha='center',
                        va='bottom', fontsize=fontsize)

                parts = ax_distrl_grab.violinplot(
                    self.psychometrics.grab_null.optimism[sort_arg],
                    [ind], widths=1,
                    showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(violin_color)

            ax_distrl_reliability.scatter(
                self.psychometrics.grab.optimism_trsplit_a,
                self.psychometrics.grab.optimism_trsplit_b)
            _pearsonr = sp_stats.pearsonr(
                self.psychometrics.grab.optimism_trsplit_a,
                self.psychometrics.grab.optimism_trsplit_b)
            _linreg = sp_stats.linregress(
                self.psychometrics.grab.optimism_trsplit_a,
                self.psychometrics.grab.optimism_trsplit_b)
            _line_x = np.arange(ylims[0], ylims[1], 0.1)
            ax_distrl_reliability.plot(_line_x, _line_x,
                                       linestyle='dotted',
                                       color='black')
            ax_distrl_reliability.set_title(
                f'optimism reliability (p={_pearsonr.pvalue:.4f})')

            ax_distrl_reliability.set_xlim(ylims)
            ax_distrl_reliability.set_ylim(ylims)
            ax_distrl_reliability.set_xlabel('optimism (first half of data)')
            ax_distrl_reliability.set_xlabel('optimism (second half of data)')

        elif zetatest_mask is True:
            zetatest_grab_inds = np.where(self.sector.zeta_pval < 0.05)[0]
            optimism_grab = self.psychometrics.grab.optimism[
                zetatest_grab_inds]
            sort_args_grab = np.argsort(optimism_grab)

            optimism_grab_toplt = self.psychometrics.grab.optimism
            optimism_grab_toplt[np.where(self.sector.zeta_pval > 0.05)[0]] \
                = None

            zetatest_neur_inds = np.where(self.neur.zeta_pval < 0.05)[0]
            optimism_neur = self.psychometrics.neur.optimism[
                zetatest_neur_inds]
            sort_args_neur = np.argsort(optimism_neur)

            optimism_neur_toplt = self.psychometrics.neur.optimism[
                zetatest_neur_inds]
            optimism_neur_x = self.neur.x[zetatest_neur_inds]
            optimism_neur_y = self.neur.y[zetatest_neur_inds]

            if ylims is None:
                vmax = np.max(np.abs(optimism_grab))
                vmin = -1 * vmax
            else:
                vmax = ylims[1]
                vmin = ylims[0]

            ax_img_grab.imshow(
                optimism_grab_toplt.reshape((n_sec, n_sec)),
                vmin=vmin, vmax=vmax,
                cmap='coolwarm')
            ax_img_grab_neur.imshow(
                optimism_grab_toplt.reshape((n_sec, n_sec)),
                vmin=vmin, vmax=vmax,
                cmap='coolwarm')

            ax_img_neur.scatter(optimism_neur_x*(n_sec/rec_px)-0.5,
                                -1 * optimism_neur_y*(n_sec/rec_px)-0.5,
                                s=40,
                                c=optimism_neur_toplt,
                                edgecolors='black', cmap='coolwarm',
                                vmin=vmin, vmax=vmax)

            ax_img_grab_neur.scatter(optimism_neur_x*(n_sec/rec_px)-0.5,
                                     optimism_neur_y*(n_sec/rec_px)-0.5,
                                     s=40,
                                     c=optimism_neur_toplt,
                                     edgecolors='black', cmap='coolwarm',
                                     vmin=vmin, vmax=vmax)

            ax_distrl_grab.scatter(np.arange(optimism_grab.shape[0]),
                                   optimism_grab[sort_args_grab],
                                   s=markersize,
                                   c=sns.xkcd_rgb['black'])
            ax_distrl_neur.scatter(np.arange(optimism_neur.shape[0]),
                                   optimism_neur[sort_args_neur],
                                   s=markersize,
                                   c=sns.xkcd_rgb['black'])
            ax_distrl_grab.axhline(self.psychometrics.lick.optimism,
                                   linestyle='dashed',
                                   color=sns.xkcd_rgb['azure'],
                                   linewidth=0.5)
            ax_distrl_neur.axhline(self.psychometrics.lick.optimism,
                                   linestyle='dashed',
                                   color=sns.xkcd_rgb['azure'],
                                   linewidth=0.5)

            for ind, sort_arg in enumerate(sort_args_grab):
                _ind_in_full_grab = zetatest_grab_inds[sort_arg]
                _sorted_null = np.sort(
                    self.psychometrics.grab_null.optimism[_ind_in_full_grab])
                _rank = np.argmin(np.abs(
                    _sorted_null - optimism_grab[sort_arg]))
                if _rank <= ind_sig_lower or _rank >= ind_sig_upper:
                    ax_distrl_grab_signif.text(
                        ind, 0, '*', ha='center',
                        va='bottom', fontsize=fontsize)

                parts = ax_distrl_grab.violinplot(
                    self.psychometrics.grab_null.optimism[_ind_in_full_grab],
                    [ind], widths=1,
                    showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(violin_color)

            ax_distrl_reliability.scatter(
                self.psychometrics.grab.optimism_trsplit_a[zetatest_grab_inds],
                self.psychometrics.grab.optimism_trsplit_b[zetatest_grab_inds])
            _pearsonr = sp_stats.pearsonr(
                self.psychometrics.grab.optimism_trsplit_a[zetatest_grab_inds],
                self.psychometrics.grab.optimism_trsplit_b[zetatest_grab_inds])
            _line_x = np.arange(ylims[0], ylims[1], 0.1)
            ax_distrl_reliability.plot(_line_x, _line_x,
                                       linestyle='dotted',
                                       color='black')
            ax_distrl_reliability.set_title(
                f'optimism reliability (p={_pearsonr.pvalue:.4f})')

            ax_distrl_reliability.set_xlim(ylims)
            ax_distrl_reliability.set_ylim(ylims)
            ax_distrl_reliability.set_xlabel('optimism (first half of data)')
            ax_distrl_reliability.set_xlabel('optimism (second half of data)')

        if ylims is not None:
            ax_distrl_grab.set_ylim(ylims)
            ax_distrl_neur.set_ylim(ylims)
            ax_distrl_reliability.set_xlim(ylims)
            ax_distrl_reliability.set_ylim(ylims)

        ax_distrl_grab.axhline(0, linestyle='dotted',
                               color='k',
                               linewidth=1.5)
        ax_distrl_neur.axhline(0, linestyle='dotted',
                               color='k',
                               linewidth=1.5)

        _figs_dir = str(self.folder.figs)
        fig_distrl.savefig(os.path.join(
            _figs_dir,
            f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{zetatest_mask=}_{ylims=}_optimism_distrl.pdf'))
        fig_distrl_reliability.savefig(os.path.join(
            _figs_dir,
            f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{zetatest_mask=}_{ylims=}_optimism_distrl_reliability.pdf'))
        fig_img.savefig(os.path.join(
            _figs_dir,
            f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{zetatest_mask=}_{ylims=}_optimism_spatial.pdf'))

        if plt_show is True:
            plt.show()

        return

    def plt_null_dist(self,
                      figsize=(6, 3),
                      img_ds_factor=50,
                      img_alpha=0.5,
                      channel=None,
                      plt_dff=None):
        if plt_dff is None:
            plt_dff = {'x': {'gain': 0.6, 'offset': 0.2},
                       'y': {'gain': 10, 'offset': 0.2}}

        rec = self._get_rec(channel)

        fig = plt.figure(figsize=figsize)
        spec = gs.GridSpec(nrows=1, ncols=2,
                           figure=fig)

        ax_dff = fig.add_subplot(spec[0, 0])
        ax_dff.set_title('dff' + (f' ({channel})' if channel else ''))
        ax_dff_null = fig.add_subplot(spec[0, 1])
        ax_dff_null.set_title('dff null'
                              + (f' ({channel})' if channel else ''))

        n_sectors = self.sector.n_sectors

        print('\tcreating max projection image...')
        _rec_max = np.max(rec[::img_ds_factor, :, :], axis=0)
        _rec_max[:, ::int(_rec_max.shape[0]/n_sectors)] = 0
        _rec_max[::int(_rec_max.shape[0]/n_sectors), :] = 0

        ax_dff.imshow(_rec_max,
                      extent=[0, n_sectors,
                              n_sectors, 0],
                      alpha=img_alpha)
        ax_dff_null.imshow(_rec_max,
                           extent=[0, n_sectors,
                                   n_sectors, 0],
                           alpha=img_alpha)

        for n_x in range(n_sectors):
            for n_y in range(n_sectors):
                _t_sector = ((self.sector.t / self.sector.t[-1])
                             * plt_dff['x']['gain'])
                _t_sector = _t_sector + n_x + plt_dff['x']['offset']

                for stim_cond in ['0', '0.5', '1']:
                    _dff_mean = np.mean(self.sector.dff[n_x, n_y][stim_cond],
                                        axis=0)
                    _dff_shifted = (-1 * _dff_mean * plt_dff['y']['gain']) \
                        + n_y - plt_dff['y']['offset'] + 1
                    ax_dff.plot(
                        _t_sector, _dff_shifted,
                        color=self.sector.colors[stim_cond],
                        linestyle=self.sector.linestyle[stim_cond])

                    _dff_null_mean = np.mean(
                        self.sector_null.dff[n_x, n_y][stim_cond][0, :, :],
                        axis=0)
                    _dff_null_shifted = (-1 * _dff_null_mean
                                         * plt_dff['y']['gain']) \
                                         + n_y - plt_dff['y']['offset'] + 1
                    ax_dff_null.plot(
                        _t_sector, _dff_null_shifted,
                        color=self.sector.colors[stim_cond],
                        linestyle=self.sector.linestyle[stim_cond])

        _ch_str = f'_{channel}' if channel else ''
        fig.savefig(os.path.join(
            str(self.folder.figs),
            f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{_ch_str}null_dists.pdf'))

        plt.show()

        return

    def plt_all_trial_resp(self, vmax_neurs=10):
        """Plots each trial's response separately."""

        figs_grab = {}
        specs_grab = {}
        axs_grab = {}

        vmax = np.max(np.abs(self.sector.stim_resp['1']))
        for tr_type in ['0', '0.5', '1']:
            n_trials = self.beh.tr_inds[tr_type].shape[0]
            nrows = np.ceil(np.sqrt(n_trials)).astype(int)

            figs_grab[tr_type] = plt.figure(figsize=(6, 6))
            specs_grab[tr_type] = gs.GridSpec(nrows=nrows, ncols=nrows,
                                              figure=figs_grab[tr_type])
            axs_grab[tr_type] = []

            for ind_tr, tr in enumerate(range(n_trials)):
                _tr_x, _tr_y = np.divmod(ind_tr, nrows)
                if ind_tr == 0:
                    axs_grab[tr_type].append(
                        figs_grab[tr_type].add_subplot(
                            specs_grab[tr_type][_tr_x, _tr_y]))
                elif ind_tr > 0:
                    axs_grab[tr_type].append(figs_grab[tr_type].add_subplot(
                        specs_grab[tr_type][_tr_x, _tr_y],
                        sharex=axs_grab[tr_type][0],
                        sharey=axs_grab[tr_type][0]))

                axs_grab[tr_type][ind_tr].imshow(
                    self.sector.stim_resp[tr_type][ind_tr, :, :],
                    vmax=vmax, vmin=-1*vmax, cmap='coolwarm')

            figs_grab[tr_type].suptitle(f'stim response, p_rew={tr_type}')
            figs_grab[tr_type].savefig(os.path.join(
                str(self.folder.figs),
                f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
                + f'_sector_resp_by_trial_prew={tr_type}.pdf'))

        figs_neur = {}
        specs_neur = {}
        axs_neur = {}

        n_sec = self.sector.n_sectors
        rec_px = self.rec.shape[1]

        if vmax_neurs is None:
            vmax_neurs = np.max(np.abs(self.neur.stim_resp['1']))

        for tr_type in ['0', '0.5', '1']:
            n_trials = self.beh.tr_inds[tr_type].shape[0]
            nrows = np.ceil(np.sqrt(n_trials)).astype(int)

            figs_neur[tr_type] = plt.figure(figsize=(6, 6))
            specs_neur[tr_type] = gs.GridSpec(nrows=nrows, ncols=nrows,
                                              figure=figs_neur[tr_type])
            axs_neur[tr_type] = []

            for ind_tr, tr in enumerate(range(n_trials)):
                _tr_x, _tr_y = np.divmod(ind_tr, nrows)
                if ind_tr == 0:
                    axs_neur[tr_type].append(
                        figs_neur[tr_type].add_subplot(
                            specs_neur[tr_type][_tr_x, _tr_y]))
                elif ind_tr > 0:
                    axs_neur[tr_type].append(figs_neur[tr_type].add_subplot(
                        specs_neur[tr_type][_tr_x, _tr_y],
                        sharex=axs_neur[tr_type][0],
                        sharey=axs_neur[tr_type][0]))

                axs_neur[tr_type][ind_tr].scatter(
                    self.neur.x*(n_sec/rec_px)-0.5,
                    self.neur.y*(n_sec/rec_px)-0.5,
                    s=40, c=self.neur.stim_resp[tr_type][ind_tr, :],
                    vmax=vmax_neurs, vmin=-1*vmax_neurs, cmap='coolwarm',
                    edgecolors='black')

            axs_neur[tr_type][0].set_xlim([-0.5, 9.5])
            axs_neur[tr_type][0].set_ylim([-0.5, 9.5])
            axs_neur[tr_type][0].yaxis.set_inverted(True)

            figs_neur[tr_type].suptitle(f'stim response, p_rew={tr_type}')
            figs_neur[tr_type].savefig(os.path.join(
                str(self.folder.figs),
                f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
                + f'_neur_resp_by_trial_prew={tr_type}.pdf'))

        plt.show()

        return

    def run_and_plt_decoding_trtype_grab(
            self,
            t_start=0, t_end=2,
            t_bl_start=-0.5, t_bl_end=0,
            thresh_n_miss=6,
            thresh_n_neurs=80,
            fold_validation=3,
            n_resamples=20,
            hit_vs_miss=True,
            comp='all',
            summary_stat_fn=np.mean,
            sec_interval=5,
            neur_sampling_rule='random',
            n_jobs=6,
            classifier=sklearn.linear_model.SGDClassifier,
            classifier_kwargs=None,
            scaler=sklearn.preprocessing.StandardScaler,
            scaler_kwargs=None,
            classifier_scoring='balanced_accuracy'):
        """
        Attempts to decode trial type from a profile of GRAB sector responses.
        Must have run self.add_sectors() first.
        """
        if classifier_kwargs is None:
            classifier_kwargs = {'class_weight': 'balanced',
                                 'loss': 'hinge',
                                 'penalty': 'l2',
                                 'alpha': 1,
                                 'max_iter': 100000}
        if scaler_kwargs is None:
            scaler_kwargs = {}

        ind_t_start = np.argmin(np.abs(self.sector.t - t_start))
        ind_t_end = np.argmin(np.abs(self.sector.t - t_end))

        if scaler is None:
            self.clf = make_pipeline(classifier(**classifier_kwargs))
        else:
            self.clf = make_pipeline(scaler(**scaler_kwargs),
                                     classifier(**classifier_kwargs))

        n_tr_prew = {}
        n_trials_all = 0

        if comp == 'all':
            tr_keys = ['0', '0.5', '1']
            tr_labels = {'0': 0, '0.5': 1, '1': 2}
        elif comp == 'rew_norew':
            tr_keys = ['0', '0.5', '1']
            tr_labels = {'0': 0, '0.5': 1, '1': 1}

        for tr_key in tr_keys:
            n_tr_prew[tr_key] = self.sector.dff[0, 0][tr_key].shape[0]
            n_trials_all += n_tr_prew[tr_key]

        self.sector.decoding = SimpleNamespace()
        self.sector.decoding._data = np.zeros((n_trials_all,
                                               self.sector.n_sectors**2))
        self.sector.decoding._labels = np.zeros(n_trials_all)

        print('decoder on full dataset\n---------')
        _tr_count = 0
        for tr_key in tr_keys:
            for _tr in range(n_tr_prew[tr_key]):
                for sec in range(self.sector.n_sectors**2):
                    sec_x, sec_y = np.divmod(sec, self.sector.n_sectors)

                    self.sector.decoding._data[
                        _tr_count, sec] = summary_stat_fn(
                        self.sector.dff[sec_x, sec_y][tr_key][
                            _tr, ind_t_start:ind_t_end])

                self.sector.decoding._labels[
                    _tr_count] = tr_labels[tr_key]
                _tr_count += 1

        self.sector.decoding.cvobj_full = cross_validate(
            self.clf, self.sector.decoding._data,
            y=self.sector.decoding._labels,
            cv=fold_validation,
            n_jobs=n_jobs, scoring=classifier_scoring,
            return_estimator=True)

        print('decoder with changing sector numbers\n---------')
        n_sectors_decoding = np.arange(sec_interval,
                                       self.sector.n_sectors**2,
                                       sec_interval)

        self.sector.decoding.cvobj = {}
        self.sector.decoding.perf_means = SimpleNamespace()
        self.sector.decoding.perf_means.keys = n_sectors_decoding
        self.sector.decoding.perf_means.vals = np.zeros_like(
            self.sector.decoding.perf_means.keys, dtype=float)

        for ind_sec_decoder, n_sec_decoder in enumerate(n_sectors_decoding):
            print(f'\tn_sectors={n_sec_decoder}')
            for n_resample in range(n_resamples):
                self.sector.decoding._data = np.zeros((n_trials_all,
                                                       n_sec_decoder))
                self.sector.decoding._labels = np.zeros(n_trials_all)

                _sector_inds_subset = np.random.choice(
                    np.arange(self.sector.n_sectors**2),
                    n_sec_decoder,
                    replace=False)

                _tr_count = 0
                for tr_key in tr_keys:
                    for _tr in range(n_tr_prew[tr_key]):
                        for sec_ind, sec in enumerate(_sector_inds_subset):
                            sec_x, sec_y = np.divmod(
                                sec, self.sector.n_sectors)

                            self.sector.decoding._data[
                                _tr_count, sec_ind] = summary_stat_fn(
                                self.sector.dff[sec_x, sec_y][tr_key][
                                    _tr, ind_t_start:ind_t_end])

                        self.sector.decoding._labels[
                            _tr_count] = tr_labels[tr_key]
                        _tr_count += 1

                self.sector.decoding.cvobj[str(n_sec_decoder)] \
                    = cross_validate(
                        self.clf, self.sector.decoding._data,
                        y=self.sector.decoding._labels,
                        cv=fold_validation,
                        n_jobs=n_jobs, scoring=classifier_scoring,
                        return_estimator=True)

                _temp_perf = np.mean(self.sector.decoding.cvobj[
                    str(n_sec_decoder)]['test_score'])
                self.sector.decoding.perf_means.vals[ind_sec_decoder] \
                    += _temp_perf

            self.sector.decoding.perf_means.vals[ind_sec_decoder] \
                /= n_resamples

        print('decoder on whole-frame data\n---------')
        self.plt_stim_aligned_avg(t_pre=self.sector.params.t_rew_pre,
                                  t_post=self.sector.params.t_rew_post,
                                  plt_show=False)
        self.sector.decoding._data_1p = np.zeros(n_trials_all)
        self.sector.decoding._labels_1p = np.zeros(n_trials_all)

        _tr_count = 0
        for tr_key in tr_keys:
            for _tr in range(n_tr_prew[tr_key]):
                self.sector.decoding._data_1p[
                    _tr_count] = summary_stat_fn(
                    self.frame.dff[tr_key][
                        _tr, ind_t_start:ind_t_end])

                self.sector.decoding._labels_1p[
                    _tr_count] = tr_labels[tr_key]
                _tr_count += 1

        self.sector.decoding.cvobj_1p = cross_validate(
            self.clf, self.sector.decoding._data_1p.reshape(-1, 1),
            y=self.sector.decoding._labels_1p,
            cv=fold_validation,
            n_jobs=n_jobs, scoring=classifier_scoring,
            return_estimator=True)

        fig_perf = plt.figure(figsize=(3, 3))
        ax_perf = fig_perf.add_subplot()

        ax_perf.plot(self.sector.decoding.perf_means.keys,
                     self.sector.decoding.perf_means.vals,
                     color=sns.xkcd_rgb['grey'],
                     linewidth=1)
        ax_perf.axhline(np.mean(self.sector.decoding.cvobj_1p['test_score']),
                        color=sns.xkcd_rgb['orange red'],
                        alpha=0.6,
                        linestyle='dotted',
                        linewidth=0.5)
        ax_perf.set_xlabel('sectors')
        ax_perf.set_ylabel('decoding perf')
        ax_perf.set_title(f'decoding: {comp}')

        if comp == 'all':
            ax_perf.set_ylim([0.2, 1])
            ax_perf.axhline(0.3333, linestyle='dashed',
                            linewidth=0.5, color=sns.xkcd_rgb['grey'])
        if comp == 'rew_norew':
            ax_perf.set_ylim([0.4, 1])
            ax_perf.axhline(0.5, linestyle='dashed',
                            linewidth=0.5, color=sns.xkcd_rgb['grey'])

        if comp == 'rew_norew':
            fig_weightmap = plt.figure(figsize=(3, 3))
            ax_weightmap = fig_weightmap.add_subplot()

            self.sector.decoding.final_weights = np.zeros(
                (self.sector.n_sectors, self.sector.n_sectors))
            for fold in range(fold_validation):
                _coefs = self.sector.decoding.cvobj_full['estimator'][
                    fold]._final_estimator.coef_[0, :].reshape(
                    self.sector.n_sectors, self.sector.n_sectors)
                self.sector.decoding.final_weights += _coefs / fold_validation
            vmax_abs = np.max(np.abs(self.sector.decoding.final_weights))

            ax_weightmap.imshow(
                self.sector.decoding.final_weights.reshape(
                    self.sector.n_sectors, self.sector.n_sectors),
                vmax=vmax_abs, vmin=-1*vmax_abs, cmap='coolwarm')
            ax_weightmap.set_title(f'decoding weights: {comp}')
            ax_weightmap.set_xticks([])
            ax_weightmap.set_yticks([])
        elif comp == 'all':
            fig_weightmap = plt.figure(figsize=(8, 3))
            spec_weightmap = gs.GridSpec(nrows=1, ncols=3,
                                         figure=fig_weightmap)

            ax_weightmap = {}
            ax_weightmap['0'] = fig_weightmap.add_subplot(
                spec_weightmap[0, 0])
            ax_weightmap['0.5'] = fig_weightmap.add_subplot(
                spec_weightmap[0, 1])
            ax_weightmap['1'] = fig_weightmap.add_subplot(
                spec_weightmap[0, 2])

            self.sector.decoding.final_weights = {}
            for ind_tr_cond, tr_cond in enumerate(['0', '0.5', '1']):
                self.sector.decoding.final_weights[tr_cond] = np.zeros(
                    (self.sector.n_sectors, self.sector.n_sectors))
                for fold in range(fold_validation):
                    _coefs = self.sector.decoding.cvobj_full['estimator'][
                        fold]._final_estimator.coef_[ind_tr_cond, :].reshape(
                        self.sector.n_sectors, self.sector.n_sectors)
                    self.sector.decoding.final_weights[tr_cond] \
                        += _coefs / fold_validation
                vmax_abs = np.max(np.abs(
                    self.sector.decoding.final_weights[tr_cond]))

                ax_weightmap[tr_cond].imshow(
                    self.sector.decoding.final_weights[tr_cond].reshape(
                        self.sector.n_sectors, self.sector.n_sectors),
                    vmax=vmax_abs, vmin=-1*vmax_abs, cmap='coolwarm')
                ax_weightmap[tr_cond].set_title(
                    f'decoding weights: {tr_cond=}')
                ax_weightmap[tr_cond].set_xticks([])
                ax_weightmap[tr_cond].set_yticks([])

        _figs_dir = str(self.folder.figs)
        fig_perf.savefig(os.path.join(
            _figs_dir,
            'sectors_decoding_perf_'
            + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{comp=}_{t_start=}_{t_end=}_'
            + f'ch={self.ch_img}.pdf'))

        fig_weightmap.savefig(os.path.join(
            _figs_dir,
            'sectors_decoding_weightmap_'
            + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{comp=}_{t_start=}_{t_end=}_'
            + f'ch={self.ch_img}.pdf'))

        plt.show()

        return

    def plt_psychometric_stimscaling(self, figsize=(2, 2),
                                     marker_size=60,
                                     plt_xlim=None,
                                     plt_show=True):
        """Generates a psychometric curve of grab response vs stim prob."""
        if plt_xlim is None:
            plt_xlim = [-0.5, 2.5]

        try:
            self.psychometrics = SimpleNamespace()
            self.psychometrics.p_rew = ['0', '0.5', '1']
            self.psychometrics.stim_resp = {'0': [],
                                            '0.5': [],
                                            '1': []}
            self.psychometrics.rew_resp = {'0': [],
                                           '0.5': [],
                                           '1': []}

            self.psychometrics.stim_resp_raw = {'0': [],
                                                '0.5': [],
                                                '1': []}
            self.psychometrics.rew_resp_all = {'0': [],
                                               '0.5': [],
                                               '1': []}

            _ind_stim_start = np.argmin(np.abs(self.sector.t - 0))
            _ind_stim_end = np.argmin(np.abs(self.sector.t - 2))

            _ind_rew_bl = np.argmin(np.abs(self.sector.t - 1.5))
            _ind_rew_start = np.argmin(np.abs(self.sector.t - 2))
            _ind_rew_end = np.argmin(np.abs(self.sector.t - 5))

            _n_trace = 0
            for n_x in range(self.sector.n_sectors):
                for n_y in range(self.sector.n_sectors):
                    for trial_cond in ['0', '0.5', '1']:
                        _dff = np.mean(
                            self.sector.dff[n_x, n_y][trial_cond],
                            axis=0)

                        _mean_stim_resp = np.mean(
                            _dff[_ind_stim_start:_ind_stim_end])
                        _mean_rew_resp = np.mean(
                            _dff[_ind_rew_start:_ind_rew_end]) \
                            - np.mean(
                                _dff[_ind_rew_bl:_ind_rew_start])

                        self.psychometrics.stim_resp[trial_cond].append(
                           _mean_stim_resp)
                        self.psychometrics.rew_resp[trial_cond].append(
                            _mean_rew_resp)

                    _n_trace += 1

            colors = sns.cubehelix_palette(
                 n_colors=3,
                 start=2, rot=0,
                 dark=0.2, light=0.8)

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1)

            ax.plot([self.psychometrics.stim_resp['0'],
                     self.psychometrics.stim_resp['0.5'],
                     self.psychometrics.stim_resp['1']],
                    color=sns.xkcd_rgb['light grey'])

            ax.scatter(
                np.ones_like(self.psychometrics.stim_resp['0'])*0,
                self.psychometrics.stim_resp['0'],
                s=marker_size,
                facecolors='none',
                edgecolors=colors[0])
            ax.scatter(
                np.ones_like(self.psychometrics.stim_resp['0.5'])*1,
                self.psychometrics.stim_resp['0.5'],
                s=marker_size,
                facecolors='none',
                edgecolors=colors[1])
            ax.scatter(
                np.ones_like(self.psychometrics.stim_resp['1'])*2,
                self.psychometrics.stim_resp['1'],
                s=marker_size,
                facecolors='none',
                edgecolors=colors[2])

            ax.set_xticks([0, 1, 2], ['0%', '50%', '100%'])
            ax.set_ylabel('GRAB 5-HT stim resp.')
            ax.set_xlabel('rew. prob.')
            ax.set_xlim(plt_xlim)

            fig_corr = plt.figure(figsize=figsize)
            ax_corr = fig_corr.add_subplot(1, 1, 1)

            ax_corr.scatter(self.psychometrics.stim_resp['1'],
                            self.psychometrics.rew_resp['1'],
                            facecolors='none',
                            edgecolors=colors[2], s=10)
            ax_corr.set_xlabel('GRAB 5-HT stim resp. (100%)')
            ax_corr.set_ylabel('GRAB 5-HT rew resp. (100%)')

            _figs_dir = str(self.folder.figs)
            fig.savefig(os.path.join(
                _figs_dir,
                f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
                + f'ch={self.ch_img}_psychometric_curve.pdf'))
            fig_corr.savefig(os.path.join(
                _figs_dir,
                f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
                + f'ch={self.ch_img}_stim_rew_resp_corr.pdf'))

            if plt_show is True:
                plt.show()
            elif plt_show is False:
                plt.close(fig)
                plt.close(fig_corr)

        except Exception as e:
            print('failed to generate psychometric curve')
            print(e)

    def _plt_rew_aligned_spatial_sectors(self, n_sectors,
                                         figsize=(3.43, 2),
                                         dpi=300,
                                         scaling_factor_trace=2,
                                         scaling_factor_img=10,
                                         t_rew_pre=1, t_rew_post=3,
                                         img_ds_factor=50,
                                         ind_lastrew=None):
        print('plotting spatial sectors...')
        self.neur = SimpleNamespace()
        self.neur.params = SimpleNamespace()

        self.neur.params.n_sectors = n_sectors
        self.neur.params.t_rew_pre = t_rew_pre
        self.neur.params.t_rew_post = t_rew_post

        fig = plt.figure(figsize=figsize, dpi=dpi)
        spec = gs.GridSpec(nrows=2, ncols=2,
                           height_ratios=[0.8, 0.2],
                           figure=fig)
        ax_img = fig.add_subplot(spec[0, 0])
        ax_traces = fig.add_subplot(spec[0, 1])
        ax_rew = fig.add_subplot(spec[1, 1], sharex=ax_traces)

        _rec_max = np.max(self.rec[::img_ds_factor, :, :], axis=0)
        _rec_max[:, ::int(_rec_max.shape[0]/n_sectors)] = 0
        _rec_max[::int(_rec_max.shape[0]/n_sectors), :] = 0

        ax_img.imshow(_rec_max,
                      extent=[0, n_sectors,
                              n_sectors, 0])

        self.neur.dff_rewaligned = np.empty(n_sectors*n_sectors,
                                            dtype=np.ndarray)
        n_frames_pre = int(t_rew_pre * self.samp_rate)
        n_frames_post = int(t_rew_post * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post

        self.neur.t = np.linspace(
            -1*t_rew_pre, t_rew_post, n_frames_tot)

        _n_trace = 0
        for n_x in range(n_sectors):
            for n_y in range(n_sectors):
                print(f'\tsector {_n_trace}/{n_sectors**2}...', end='\r')
                _ind_x_lower = int((n_x / n_sectors) * self.rec.shape[1])
                _ind_x_upper = int(((n_x+1) / n_sectors) * self.rec.shape[1])

                _ind_y_lower = int((n_y / n_sectors) * self.rec.shape[1])
                _ind_y_upper = int(((n_y+1) / n_sectors) * self.rec.shape[1])

                _trace = np.mean(np.mean(
                    self.rec[:,
                             _ind_x_lower:_ind_x_upper,
                             _ind_y_lower:_ind_y_upper], axis=1), axis=1)

                ax_traces.plot(self.rec_t,
                               sp.stats.zscore(_trace)*scaling_factor_trace
                               + _n_trace,
                               color=sns.xkcd_rgb['ocean green'],
                               linewidth=0.5, alpha=0.8)

                self.neur.dff_rewaligned[_n_trace] = np.zeros((
                    self.beh.rew.t[self.beh._rewrange.first:
                                   self.beh._rewrange.last].shape[0],
                    n_frames_tot))

                for ind, t_rew in enumerate(
                        self.beh.rew.t[self.beh._rewrange.first:
                                       self.beh._rewrange.last]):
                    ind_rew = np.argmin(np.abs(self.rec_t-t_rew))
                    _rew_trace = _trace[ind_rew-n_frames_pre:
                                        ind_rew+n_frames_post]
                    self.neur.dff_rewaligned[_n_trace][ind, :] = calc_dff(
                        _rew_trace, baseline_frames=n_frames_pre)

                _rec_dt = np.diff(self.rec_t[0:n_frames_tot])[0]
                _rec_t_templ = np.arange(0, (n_frames_tot+5)*_rec_dt, _rec_dt)

                t_rewaligned_norm = ((_rec_t_templ[0:n_frames_tot]
                                      / _rec_t_templ[n_frames_tot])
                                     * 0.6)
                t_rewaligned_norm = t_rewaligned_norm + n_x + 0.2

                dff_rewaligned_mean = np.mean(
                    self.neur.dff_rewaligned[_n_trace], axis=0)
                dff_rewaligned_mean_shifted = (-1 * dff_rewaligned_mean
                                               * scaling_factor_img
                                               + n_y + 0.5)

                t_rewonset = ((_rec_t_templ[n_frames_pre]
                               / _rec_t_templ[n_frames_tot]) * 0.6) \
                               + n_x + 0.2
                ax_img.plot(t_rewaligned_norm,
                            dff_rewaligned_mean_shifted,
                            color=sns.xkcd_rgb['orangered'],
                            linewidth=0.5,
                            alpha=0.8)
                ax_img.plot([t_rewonset, t_rewonset], [n_y+0.2, n_y+0.8],
                            color=sns.xkcd_rgb['white'], linestyle='dashed',
                            linewidth=0.3)

                _n_trace += 1
                self._last_trace = _trace

        for ind, t_rew in enumerate(self.beh.rew.t):
            ax_rew.plot([t_rew, t_rew], [0, 1],
                        color=sns.xkcd_rgb['bright blue'])

        if self.fname_img.endswith('Ch1.tif'):
            prefix = 'grab'
        elif self.fname_img.endswith('Ch2.tif'):
            prefix = 'gcamp'

        # Save to enclosing folder (uses absolute path directly)
        fig.savefig(os.path.join(
            self.folder.enclosing,
            f'{self.path.animal}'
            + f'_{self.path.date}_{self.path.beh_folder}'
            + f'_{prefix}_sector_fig.pdf'))

        plt.show()

    def plt_rew_trace_by_iti(self, ind_sector=10, n_time_divs=5,
                             sns_palette='mako',
                             figsize=(3.43, 3.43), savefig_prefix='grab'):
        palette = sns.color_palette(sns_palette, n_time_divs)

        self.dff_binned_time = np.empty(n_time_divs, dtype=np.ndarray)
        itis = np.diff(self.beh.rew.t)
        t_bin_edges, iti_masks = self._bin_trials_by_iti(
            itis, n_time_divs)

        for ind_timediv in range(n_time_divs):
            _trials_in_timediv = iti_masks[ind_timediv]
            _mean_dff_in_timediv = np.mean(
                self.neur.dff_rewaligned[ind_sector][_trials_in_timediv, :],
                axis=0)
            self.dff_binned_time[ind_timediv] = _mean_dff_in_timediv

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        for ind_timediv in range(n_time_divs):
            label = f'{t_bin_edges[ind_timediv]:.1f}' \
                + f'-{t_bin_edges[ind_timediv+1]:.1f}'
            ax.plot(self.neur.t,
                    self.dff_binned_time[ind_timediv],
                    color=palette[ind_timediv], linewidth=0.8,
                    label=label)
        ax.axvline(x=0, color=sns.xkcd_rgb['bright blue'],
                   linestyle='dashed',
                   linewidth=0.8)
        ax.legend()

        self._finish_plot(
            fig, ax,
            xlabel='time from rew (s)',
            ylabel='df/f',
            savepath=os.path.join(
                str(self.folder.figs),
                f'{savefig_prefix}_dff_by_{n_time_divs}'
                + f'iti_sector{ind_sector}.pdf'),
            show=True)

    def plt_dffs_including_iti(self, ind_sector=10, n_time_divs=5,
                               t_rew_pre=2,
                               sns_palette='mako',
                               figsize=(3.43, 3.43), savefig_prefix='grab'):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        palette = sns.color_palette(sns_palette, n_time_divs)

        self.dff_binned_time = np.empty(n_time_divs, dtype=np.ndarray)
        itis = np.diff(self.beh.rew.t)

        t_bin_edges, iti_masks = self._bin_trials_by_iti(
            itis, n_time_divs)

        for ind_timediv in range(n_time_divs):
            _trials_in_timediv = iti_masks[ind_timediv]

            t_rew_post = t_bin_edges[ind_timediv]

            n_frames_pre = int(t_rew_pre * self.samp_rate)
            n_frames_post = int(t_rew_post * self.samp_rate)
            n_frames_tot = n_frames_pre + n_frames_post

            for ind, t_rew in enumerate(self.beh.rew.t[1:]):
                ind_rew = np.argmin(np.abs(self.rec_t-t_rew))
                _rew_trace = _trace[ind_rew-n_frames_pre:
                                    ind_rew+n_frames_post]
                self.neur.dff_rewaligned[_n_trace][ind, :] = calc_dff(
                    _rew_trace, baseline_frames=n_frames_pre)

            _mean_dff_in_timediv = np.mean(
                self.neur.dff_rewaligned[ind_sector][_trials_in_timediv, :],
                axis=0)
            self.dff_binned_time[ind_timediv] = _mean_dff_in_timediv

        for ind_timediv in range(n_time_divs):
            label = (f'{t_bin_edges[ind_timediv]:.1f}-'
                     f'{t_bin_edges[ind_timediv+1]:.1f}')
            ax.plot(self.neur.t,
                    self.dff_binned_time[ind_timediv],
                    color=palette[ind_timediv], linewidth=0.8,
                    label=label)
        ax.axvline(x=0, color=sns.xkcd_rgb['bright blue'],
                   linestyle='dashed',
                   linewidth=0.8)
        ax.legend()

        self._finish_plot(
            fig, ax,
            xlabel='time from rew (s)',
            ylabel='df/f',
            savepath=os.path.join(
                str(self.folder.figs),
                f'{savefig_prefix}_dff_by_{n_time_divs}'
                f'iti_sector{ind_sector}.pdf'),
            show=False)

        return

    def plt_dffs_single(self, sector=0,
                        sns_palette='mako',
                        figsize=(3.43, 3.43),
                        savefig_prefix='grab'):

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        for trial in range(self.neur.dff_rewaligned[sector].shape[0]):
            ax.plot(self.neur.t,
                    self.neur.dff_rewaligned[sector][trial, :],
                    color=sns.xkcd_rgb['ocean green'], linewidth=0.8)
        ax.axvline(x=0, color=sns.xkcd_rgb['bright blue'],
                   linestyle='dashed',
                   linewidth=0.8)

        self._finish_plot(
            fig, ax,
            xlabel='time from rew (s)',
            ylabel='df/f',
            savepath=os.path.join(
                str(self.folder.figs),
                f'{savefig_prefix}_dff_{sector=}.pdf'),
            show=True)

    def plt_dffs_all(self,
                     sns_palette='mako',
                     figsize=(3.43, 3.43),
                     savefig_prefix='grab'):

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        for sector in range(self.neur.dff_rewaligned.shape[0]):
            ax.plot(self.neur.t,
                    np.mean(self.neur.dff_rewaligned[sector], axis=0),
                    color=sns.xkcd_rgb['ocean green'], linewidth=0.8)
        ax.axvline(x=0, color=sns.xkcd_rgb['bright blue'],
                   linestyle='dashed',
                   linewidth=0.8)

        self._finish_plot(
            fig, ax,
            xlabel='time from rew (s)',
            ylabel='df/f',
            savepath=os.path.join(
                str(self.folder.figs),
                f'{savefig_prefix}_dff_all.pdf'),
            show=True)

    def plt_total_resp(self, figsize=(3.43, 3.43),
                       savefig_prefix='grab'):

        n_rews = self.neur.dff_rewaligned[0].shape[0]
        n_sectors = self.neur.dff_rewaligned.shape[0]

        self.resp_5ht = np.zeros(n_rews)

        for rew in range(self.beh._rewrange.first, self.beh._rewrange.last):
            _resp_5ht = 0
            for sector in range(n_sectors):
                _dff_integ = np.trapz(
                    self.neur.dff_rewaligned[sector][rew, :],
                    dx=1/self.samp_rate)

                _resp_5ht += _dff_integ
            _resp_5ht /= n_sectors

            self.resp_5ht[rew] = _resp_5ht

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(np.arange(n_rews), self.resp_5ht,
                color=sns.xkcd_rgb['ocean green'], linewidth=0.8)
        self._finish_plot(
            fig, ax,
            xlabel='reward number',
            ylabel='integral df/f',
            savepath=os.path.join(
                str(self.folder.figs),
                f'{savefig_prefix}_total_response.pdf'),
            show=True)

        return

    def plt_lick_resp(self, t_pre=3, t_post=3,
                      figsize=(3.43, 3.43),
                      savefig_prefix='grab',
                      channel=None):
        rec = self._get_rec(channel)

        n_frames_pre, n_frames_post, n_frames_tot, t_lickaligned = \
            self._aligned_frame_params(t_pre, t_post, t_event=0)

        dff_lick = np.zeros(n_frames_tot)
        count_dffs = 0
        for lick in self.beh.t_licks:
            _closest_rew = np.min(np.abs(
                self.beh.rew.t - lick))
            if _closest_rew > np.max([t_pre, t_post]):
                ind_lick = np.argmin(np.abs(
                    self.rec_t - lick))
                f = np.mean(np.mean(
                    rec[ind_lick-n_frames_pre:
                        ind_lick+n_frames_post, :, :],
                    axis=1), axis=1)
                dff = calc_dff(f, n_frames_pre)
                dff_lick += dff
                count_dffs += 1

        dff_lick /= count_dffs

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t_lickaligned, dff_lick,
                color=sns.xkcd_rgb['grey'], linewidth=0.8)
        ax.axvline(x=0, color=sns.xkcd_rgb['black'],
                   linestyle='dashed',
                   linewidth=0.8)

        _ch_str = f'_{channel}' if channel else ''
        self._finish_plot(
            fig, ax,
            xlabel='time from lick (s)',
            ylabel='dff',
            savepath=os.path.join(
                str(self.folder.figs),
                f'{savefig_prefix}{_ch_str}'
                f'_lick_resp_{t_pre=}_{t_post=}.pdf'),
            show=True)

    def plt_neur_masks(self, show=True):
        """Plot an image of all neuron masks from suite2p output."""
        if not hasattr(self, 'neur'):
            self.add_neurs()

        ops = self.neur.ops.item()
        mask_img = np.zeros((ops['Ly'], ops['Lx']))

        for neur in range(len(self.neur.stat)):
            ypix = self.neur.stat[neur]['ypix']
            xpix = self.neur.stat[neur]['xpix']
            mask_img[ypix, xpix] = 1

        self.neur.maskimg = mask_img

        fig, ax = plt.subplots(1, 1)
        ax.imshow(self.neur.maskimg, cmap='gray')
        ax.set_xlabel('x (px)')
        ax.set_ylabel('y (px)')
        ax.set_title('neuron masks')

        fig.savefig(os.path.join(
            str(self.folder.figs),
            'neur_masks_'
            + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}'
            + '.pdf'))

        if show:
            plt.show()
