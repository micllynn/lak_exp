"""
twop_analysis.py  –  AnalysisMixin for TwoPRec.

All data-computation methods (add_*, get_*) extracted from load_exp_twop.py
to keep that file focused on loading/init.

Requires self.beh, self.rec, self.rec_t, self.samp_rate, self.neur,
self._trial_cond_map, etc. to be populated by TwoPRec.__init__.
"""

from types import SimpleNamespace
import numpy as np
import scipy as sp
import zetapy

from .utils_twop import calc_dff


class AnalysisMixin:
    """Data-computation methods mixed into TwoPRec."""

    # ------------------------------------
    # Lick analysis
    # ------------------------------------

    def add_lickrates(self, t_prestim=2,
                      t_poststim=5,
                      bl_norm=False):
        """
        Adds a lickrate segregated by trial-type
        """

        # add base/antic lickrates for all trialtypes
        n_trials = self.beh.stim.t_start.shape[0]

        self.beh.lick.t = np.empty(n_trials,
                                   dtype=np.ndarray)
        self.beh.lick.antic_raw = np.empty(n_trials,
                                           dtype=np.ndarray)
        self.beh.lick.base_raw = np.empty(n_trials,
                                          dtype=np.ndarray)

        for trial in range(n_trials):
            # get the data
            _t_stim = self.beh.stim.t_start[trial]
            _t_rew = _t_stim + 2
            _lick_inds = np.logical_and(
                self.beh.lick.t_raw > (_t_stim - t_prestim),
                self.beh.lick.t_raw < (_t_rew + t_poststim))
            _licks = self.beh.lick.t_raw[_lick_inds]
            _licks -= _t_stim

            # correct licks
            from .utils import remove_lick_artefact_after_rew
            _t_rew_rel_to_stim = _t_rew - _t_stim
            _licks = remove_lick_artefact_after_rew(
                _licks, [_t_rew_rel_to_stim+0.03,
                         _t_rew_rel_to_stim+0.06])
            self.beh.lick.t[trial] = _licks

            # compute anticipatory lickrates
            _licks_antic = np.sum(np.logical_and(
                _licks > 0, _licks < 2))
            self.beh.lick.antic_raw[trial] = (_licks_antic
                                              / (_t_rew - _t_stim))
            _licks_base = np.sum(np.logical_and(
                _licks < 0, _licks > -4))
            self.beh.lick.base_raw[trial] = (_licks_base
                                             / (t_prestim - 0))

        # segregate lickrates per trialtype
        self.beh.lick.antic = {}
        self.beh.lick.base = {}
        for trial_type in self.beh.tr_conds:
            self.beh.lick.antic[trial_type] = np.zeros(
                self.beh.tr_inds[trial_type].shape[0])
            self.beh.lick.base[trial_type] = np.zeros(
                self.beh.tr_inds[trial_type].shape[0])

            for ind_trial, trial in enumerate(self.beh.tr_inds[trial_type]):
                if bl_norm is True:
                    _antic_licks = self.beh.lick.antic_raw[trial] \
                        - self.beh.lick.base_raw[trial]
                elif bl_norm is False:
                    _antic_licks = self.beh.lick.antic_raw[trial]

                self.beh.lick.antic[trial_type][ind_trial] \
                    = _antic_licks
                self.beh.lick.base[trial_type][ind_trial] \
                    = self.beh.lick.base_raw[trial]

        return

    # ------------------------------------
    # Whole-frame and sector fluorescence
    # ------------------------------------

    def add_frame(self, t_pre=2, t_post=5, channel=None, use_zscore=False):
        """
        Creates trial-averaged signal across the whole field of view,
        separated by trial-type (eg 0%, 50%, 100% rewarded trials).

        Parameters
        ----------
        channel : str or None
            Passed to _get_rec() / _get_rec_t(). Ignored in the base class;
            used by TwoPRec_DualColour to select the channel.
        use_zscore : bool
            If True, compute z-scored fluorescence instead of df/f.
        """
        _ch_str = f', ch={channel}' if channel is not None else ''
        print(f'creating trial-averaged signal (whole-frame{_ch_str})...')

        # setup
        self.add_lickrates()

        rec = self._get_rec(channel)
        rec_t = self._get_rec_t(channel)

        self.frame = SimpleNamespace()
        self.frame.params = SimpleNamespace()
        self.frame.params.t_rew_pre = t_pre
        self.frame.params.t_rew_post = t_post
        self.frame.params.channel = channel
        self.frame.params.use_zscore = use_zscore

        # setup stim-aligned traces
        n_frames_pre, n_frames_post, n_frames_tot, t_vec = \
            self._aligned_frame_params(t_pre, t_post, t_event=2)

        self.frame.dff = self._init_dff_by_cond(
            self.beh.tr_conds, n_frames_tot)
        self.frame.t = t_vec
        self.frame.tr_counts = {k: 0 for k in self.beh.tr_conds}

        _trials = np.arange(self.beh._stimrange.first,
                            self.beh._stimrange.last)
        _all_ind_t_start = np.searchsorted(
            rec_t, self.beh.stim.t_start[_trials] - t_pre)
        for tr_ind, trial in enumerate(_trials):
            print(f'\ttrial={int(trial)}', end='\r')
            _ind_t_start = _all_ind_t_start[tr_ind]
            _f = np.mean(np.mean(
                rec[_ind_t_start:_ind_t_start + n_frames_tot, :, :],
                axis=1), axis=1)
            if use_zscore:
                _baseline = _f[:n_frames_pre]
                _sigma = np.std(_baseline)
                _dff = (_f - np.mean(_baseline)) / _sigma \
                    if _sigma > 0 else np.zeros_like(_f)
            else:
                _dff = calc_dff(_f, baseline_frames=n_frames_pre)

            for _tr_cond in self._trial_cond_map.get(int(trial), []):
                if _tr_cond in self.frame.dff:
                    self.frame.dff[_tr_cond][
                        self.frame.tr_counts[_tr_cond], :] = \
                        _dff[0:n_frames_tot]
                    self.frame.tr_counts[_tr_cond] += 1
        print('')
        return

    def add_sectors(self,
                    n_sectors=10,
                    t_pre=2, t_post=5,
                    n_null=500,
                    resid_type='sector',
                    resid_corr_tr_cond='1',
                    channel=None,
                    use_zscore=False,
                    compute_null=False,
                    compute_resid_corr=True,
                    colors=None,
                    orientation_extra_tr_conds=('0.5_prelick',
                                                '0.5_noprelick'),
                    orientation_colors=None,
                    orientation_linestyle=None,
                    orientation_resid_tr_conds=('0', '0.5', '1')):
        """
        Divides the field of view into sectors and computes trial-averaged
        fluorescence traces per sector.

        Parameters
        ----------
        channel : str or None
            Passed to _get_rec() / _get_rec_t(). Ignored in the base class;
            used by TwoPRec_DualColour to select the channel.
        orientation_extra_tr_conds : tuple[str, ...]
            Additional trial-condition keys to include when
            beh_type='visual_pavlov'.
        orientation_colors : dict or None
            Optional per-condition color mapping for beh_type='visual_pavlov'.
        orientation_linestyle : dict or None
            Optional per-condition linestyle mapping for
            beh_type='visual_pavlov'.
        orientation_resid_tr_conds : tuple[str, ...] or None
            Trial conditions to use for residual computation when
            beh_type='visual_pavlov'.
        """
        from .twop_plots import PlotsMixin
        if colors is None:
            colors = PlotsMixin._channel_palette(channel, n_colors=3)

        self.add_lickrates(t_prestim=t_pre, t_poststim=t_post)

        rec = self._get_rec(channel)
        rec_t = self._get_rec_t(channel)

        # compute whole-frame avg in case needed
        if (not hasattr(self, 'frame')
                or self.frame.params.use_zscore != use_zscore
                or self.frame.params.channel != channel):
            self.add_frame(t_pre=t_pre, t_post=t_post,
                           channel=channel, use_zscore=use_zscore)

        _ch_str = f', ch={channel}' if channel is not None else ''
        print(f'creating trial-averaged signal (sectors{_ch_str})...')

        self.sector = SimpleNamespace()

        self.sector.params = SimpleNamespace()
        self.sector.n_sectors = n_sectors
        self.sector.params.t_rew_pre = t_pre
        self.sector.params.t_rew_post = t_post
        self.sector.n_sectors = n_sectors
        self.sector.n_null = n_null
        self.sector.params.channel = channel
        self.sector.params.use_zscore = use_zscore
        self.sector.params.resid_type = resid_type

        is_stimulus_orientation = (
            getattr(self, 'beh_type', None) == 'visual_pavlov')
        _tr_conds = list(self.beh.tr_conds)
        if is_stimulus_orientation:
            for _tr_cond in orientation_extra_tr_conds:
                if (_tr_cond in self.beh.tr_inds
                        and _tr_cond not in _tr_conds):
                    _tr_conds.append(_tr_cond)

        if is_stimulus_orientation and orientation_resid_tr_conds is not None:
            _resid_tr_conds = [
                _tr_cond for _tr_cond in orientation_resid_tr_conds
                if _tr_cond in _tr_conds]
            if len(_resid_tr_conds) == 0:
                _resid_tr_conds = list(_tr_conds)
        else:
            _resid_tr_conds = list(_tr_conds)

        self.sector.params.tr_conds = tuple(_tr_conds)
        self.sector.params.resid_tr_conds = tuple(_resid_tr_conds)

        self.sector.colors = {
            _tr_cond: colors[ind % len(colors)]
            for ind, _tr_cond in enumerate(_tr_conds)}
        self.sector.linestyle = {_tr_cond: 'solid' for _tr_cond in _tr_conds}

        if is_stimulus_orientation:
            _orientation_colors = orientation_colors
            if _orientation_colors is None:
                _orientation_colors = {'0': colors[0],
                                       '0.5': colors[1],
                                       '1': colors[2],
                                       '0.5_rew': colors[1],
                                       '0.5_norew': colors[1],
                                       '0.5_prelick': colors[1],
                                       '0.5_noprelick': colors[1]}
            _orientation_linestyle = orientation_linestyle
            if _orientation_linestyle is None:
                _orientation_linestyle = {'0': 'solid',
                                          '0.5': 'solid',
                                          '1': 'solid',
                                          '0.5_rew': 'solid',
                                          '0.5_norew': 'dashed',
                                          '0.5_prelick': 'solid',
                                          '0.5_noprelick': 'dashed'}
            for _tr_cond, _color in _orientation_colors.items():
                if _tr_cond in self.sector.colors:
                    self.sector.colors[_tr_cond] = _color
            for _tr_cond, _linestyle in _orientation_linestyle.items():
                if _tr_cond in self.sector.linestyle:
                    self.sector.linestyle[_tr_cond] = _linestyle

        # setup stim-aligned traces
        n_frames_pre, n_frames_post, n_frames_tot, t_vec = \
            self._aligned_frame_params(t_pre, t_post, t_event=2)

        _use_zscore = use_zscore
        if not _use_zscore and \
           self.beh._stimrange.first < self.beh._stimrange.last:
            _t0 = self.beh.stim.t_start[self.beh._stimrange.first]
            _ind0 = np.argmin(np.abs(rec_t - (_t0 - t_pre)))
            _probe = np.mean(np.mean(
                rec[_ind0:_ind0 + n_frames_pre, :, :], axis=1), axis=1)
            _f0_probe = float(np.mean(_probe)) if _probe.size > 0 else 1.0
            if np.abs(_f0_probe) < 1.0:
                print(f'\tWarning: baseline fluorescence mean '
                      f'({_f0_probe:.4f}) is near zero. '
                      'Signal appears to be a corrected residual. '
                      'Falling back to z-score. '
                      'Pass use_zscore=True explicitly '
                      'to suppress this check.')
                _use_zscore = True

        self.sector.params.use_zscore = _use_zscore

        self.sector.dff = np.empty((n_sectors, n_sectors), dtype=dict)
        self.sector.dff_resid = np.empty((n_sectors, n_sectors),
                                         dtype=dict)

        self.sector.t = t_vec

        # store edges of sectors
        self.sector.x = SimpleNamespace()
        self.sector.x.lower = np.zeros((n_sectors, n_sectors), dtype=int)
        self.sector.x.upper = np.zeros((n_sectors, n_sectors), dtype=int)
        self.sector.y = SimpleNamespace()
        self.sector.y.lower = np.zeros((n_sectors, n_sectors), dtype=int)
        self.sector.y.upper = np.zeros((n_sectors, n_sectors), dtype=int)

        # ---------------------
        print('\textracting aligned fluorescence traces...')
        _n_trace = 1
        for n_x in range(n_sectors):
            for n_y in range(n_sectors):
                print(f'\t\tsector {_n_trace}/{n_sectors**2}...      ',
                      end='\r')
                # calculate location of sector
                _ind_x_lower = int((n_x / n_sectors) * rec.shape[1])
                _ind_x_upper = int(((n_x+1) / n_sectors)
                                   * rec.shape[1])

                _ind_y_lower = int((n_y / n_sectors) * rec.shape[1])
                _ind_y_upper = int(((n_y+1) / n_sectors)
                                   * rec.shape[1])

                self.sector.x.lower[n_x, n_y] = _ind_x_lower
                self.sector.x.upper[n_x, n_y] = _ind_x_upper
                self.sector.y.lower[n_x, n_y] = _ind_y_lower
                self.sector.y.upper[n_x, n_y] = _ind_y_upper

                # setup structure
                self.sector.dff[n_x, n_y] = self._init_dff_by_cond(
                    _tr_conds, n_frames_tot)
                self.sector.dff_resid[n_x, n_y] = {
                    _cond: np.zeros_like(_arr)
                    for _cond, _arr in self.sector.dff[n_x, n_y].items()}

                # store stim-aligned traces
                self.sector.tr_counts = {k: 0 for k in _tr_conds}
                for trial in range(self.beh._stimrange.first,
                                   self.beh._stimrange.last):
                    _t_stim = self.beh.stim.t_start[trial]
                    _ind_t_start = np.searchsorted(
                        rec_t, _t_stim - t_pre)

                    # extract fluorescence using fixed stim-aligned window
                    _f = np.mean(np.mean(
                        rec[_ind_t_start:_ind_t_start + n_frames_tot,
                                 _ind_x_lower:_ind_x_upper,
                                 _ind_y_lower:_ind_y_upper], axis=1),
                                 axis=1)
                    if _use_zscore:
                        _baseline = _f[:n_frames_pre]
                        _sigma = np.std(_baseline)
                        _dff = (_f - np.mean(_baseline)) / _sigma \
                            if _sigma > 0 else np.zeros_like(_f)
                    else:
                        _dff = calc_dff(_f, baseline_frames=n_frames_pre)

                    for _tr_cond in self._trial_cond_map.get(int(trial), []):
                        if _tr_cond in self.sector.dff[n_x, n_y]:
                            self.sector.dff[n_x, n_y][_tr_cond][
                                self.sector.tr_counts[_tr_cond], :] = \
                                    _dff[0:n_frames_tot]
                            self.sector.tr_counts[_tr_cond] += 1

                # store dff residuals
                # ----------
                for _tr_cond in _resid_tr_conds:
                    for tr in range(
                            self.sector.dff[n_x, n_y][_tr_cond].shape[0]):
                        if resid_type == 'sector':
                            _dff_mean = np.mean(
                                self.sector.dff[n_x, n_y][_tr_cond], axis=0)
                        elif resid_type == 'trial':
                            _dff_mean = self.frame.dff[_tr_cond][tr, :]
                        else:
                            raise ValueError(
                                "resid_type must be 'sector' or 'trial',"
                                + f" got '{resid_type}'")

                        self.sector.dff_resid[n_x, n_y][_tr_cond][tr, :] \
                            = self.sector.dff[n_x, n_y][_tr_cond][tr, :] \
                            - _dff_mean

                # update _n_trace
                _n_trace += 1
        print('')

        # compute cross-correlations (optional, expensive O(n_sectors^4))
        if compute_resid_corr:
            if resid_corr_tr_cond not in _tr_conds:
                raise ValueError(
                    f"resid_corr_tr_cond '{resid_corr_tr_cond}'"
                    + " not found in trial conditions.")
            self.sector.resid_corr_stat = np.zeros(
                (self.sector.n_sectors**2, self.sector.n_sectors**2))
            self.sector.resid_corr_pval = np.zeros(
                (self.sector.n_sectors**2, self.sector.n_sectors**2))

            for sec_1 in range(self.sector.n_sectors**2):
                for sec_2 in range(self.sector.n_sectors**2):
                    sec_1_x, sec_1_y = np.divmod(sec_1, self.sector.n_sectors)
                    sec_2_x, sec_2_y = np.divmod(sec_2, self.sector.n_sectors)

                    _sec_1_sig = np.mean(
                        self.sector.dff_resid[sec_1_x, sec_1_y][
                            resid_corr_tr_cond],
                        axis=1)
                    _sec_2_sig = np.mean(
                        self.sector.dff_resid[sec_2_x, sec_2_y][
                            resid_corr_tr_cond],
                        axis=1)

                    _corr = sp.stats.pearsonr(_sec_1_sig, _sec_2_sig)
                    self.sector.resid_corr_stat[sec_1, sec_2] = _corr.statistic
                    self.sector.resid_corr_pval[sec_1, sec_2] = _corr.pvalue

        # add null distributions
        # -------------
        if compute_null:
            self.add_null_dists(n_null=n_null, t_pre=t_pre, t_post=t_post)

        return

    def add_lr_sides(self, t_pre=2, t_post=5,
                     channel=None, use_zscore=False):
        """Trial-aligned mean fluorescence for the top and bottom halves
        of the field of view.

        The image is split along axis 1 (image rows) at the midpoint. By the
        recording geometry, the bottom half of the image corresponds to the
        left side of the tissue, and the top half to the right side.

        Parameters
        ----------
        t_pre : float
            Seconds before stim onset to include.
        t_post : float
            Seconds after stim onset to include.
        channel : str or None
            Passed to _get_rec() / _get_rec_t().
        use_zscore : bool
            If True, use z-score instead of df/f.
        """
        _ch_str = f', ch={channel}' if channel is not None else ''
        print(f'creating trial-averaged signal (lr sides{_ch_str})...')

        self.add_lickrates()

        rec = self._get_rec(channel)
        rec_t = self._get_rec_t(channel)

        self.sides = SimpleNamespace()
        self.sides.params = SimpleNamespace()
        self.sides.params.t_rew_pre = t_pre
        self.sides.params.t_rew_post = t_post
        self.sides.params.channel = channel
        self.sides.params.use_zscore = use_zscore

        n_frames_pre, n_frames_post, n_frames_tot, t_vec = \
            self._aligned_frame_params(t_pre, t_post, t_event=2)
        self.sides.t = t_vec

        h_mid = rec.shape[1] // 2
        self.sides.h_mid = h_mid

        _halves = ('top', 'bottom')
        self.sides.dff = {
            half: self._init_dff_by_cond(self.beh.tr_conds, n_frames_tot)
            for half in _halves}
        self.sides.tr_counts = {
            half: {k: 0 for k in self.beh.tr_conds}
            for half in _halves}

        _trials = np.arange(self.beh._stimrange.first,
                            self.beh._stimrange.last)
        _all_ind_t_start = np.searchsorted(
            rec_t, self.beh.stim.t_start[_trials] - t_pre)

        for tr_ind, trial in enumerate(_trials):
            print(f'\ttrial={int(trial)}', end='\r')
            _ind_t_start = _all_ind_t_start[tr_ind]
            _chunk = rec[_ind_t_start:_ind_t_start + n_frames_tot, :, :]

            _f_per_half = {
                'top': np.mean(np.mean(_chunk[:, :h_mid, :], axis=1), axis=1),
                'bottom': np.mean(np.mean(_chunk[:, h_mid:, :], axis=1),
                                  axis=1)}

            for half, _f in _f_per_half.items():
                if use_zscore:
                    _baseline = _f[:n_frames_pre]
                    _sigma = np.std(_baseline)
                    _dff = (_f - np.mean(_baseline)) / _sigma \
                        if _sigma > 0 else np.zeros_like(_f)
                else:
                    _dff = calc_dff(_f, baseline_frames=n_frames_pre)

                for _tr_cond in self._trial_cond_map.get(int(trial), []):
                    if _tr_cond in self.sides.dff[half]:
                        self.sides.dff[half][_tr_cond][
                            self.sides.tr_counts[half][_tr_cond], :] = \
                            _dff[0:n_frames_tot]
                        self.sides.tr_counts[half][_tr_cond] += 1
        print('')
        return

    def add_neurs(self, t_pre=1, t_post=2, zscore=True):
        self.add_lickrates(t_prestim=t_pre,
                           t_poststim=t_post)

        print('adding aligned neur avg...')
        # setup structure
        # ---------------
        neurs = np.where(self.neur.iscell[:, 0] == 1)[0]
        n_neurs = neurs.shape[0]

        n_frames_pre, n_frames_post, n_frames_tot, t_vec = \
            self._aligned_frame_params(t_pre, t_post, t_event=2)

        _tr_conds = list(self.beh.tr_inds.keys())

        self.neur.dff_aln = np.empty(n_neurs, dtype=dict)
        self.neur.dff_aln_mean = {}
        for _tr_cond in _tr_conds:
            self.neur.dff_aln_mean[_tr_cond] = np.zeros(
                (n_neurs, n_frames_tot))
        self.neur.x = np.zeros(n_neurs)
        self.neur.y = np.zeros(n_neurs)

        self.neur.t_aln = t_vec

        # iterate through neurons
        # ----------------
        for ind, neur in enumerate(neurs):
            print(f'\tneur={int(neur)}', end='\r')
            # extract x/y locations
            self.neur.x[ind] = self.neur.stat[neur]['med'][0]
            self.neur.y[ind] = self.neur.stat[neur]['med'][1]
            # correct for image crop
            self.neur.x[ind] -= self.ops.n_px_remove_sides
            self.neur.y[ind] -= self.ops.n_px_remove_sides

            # setup structure
            # -------------
            self.neur.dff_aln[ind] = self._init_dff_by_cond(
                _tr_conds, n_frames_tot)

            _tr_counts = {k: 0 for k in _tr_conds}

            # extract fluorescence for each trial
            # -------------
            if zscore is True:
                _f_full = sp.stats.zscore(self.neur.f[neur, :])
            else:
                _f_full = self.neur.f[neur, :]

            for trial in range(self.beh._stimrange.first,
                               self.beh._stimrange.last):
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

            for _tr_cond in _tr_conds:
                self.neur.dff_aln_mean[_tr_cond][ind, :] = \
                    np.mean(self.neur.dff_aln[ind][_tr_cond], axis=0)
        print('')

        return

    # ------------------------------------
    # Psychometrics
    # ------------------------------------

    def add_psychometrics(self, t_pre=5, t_post=2):
        if not hasattr(self, 'sector'):
            self.add_sectors(t_pre=t_pre,
                             t_post=t_post)
        if not hasattr(self.neur, 'dff_aln'):
            self.add_neurs(t_pre=t_pre,
                           t_post=t_post)

        print('adding psychometrics...')

        # set up psychometrics attribute and variables
        # -------------
        print('\tsetting up attributes...')
        n_sec = self.sector.n_sectors
        self.psychometrics = SimpleNamespace(
            grab=SimpleNamespace(),
            grab_null=SimpleNamespace(),
            grab_frame=SimpleNamespace(),
            neur=SimpleNamespace(),
            lick=SimpleNamespace())
        self.psychometrics.p_rew = ['0', '0.5', '1']

        for item in ['grab', 'grab_frame', 'neur', 'lick']:
            setattr(getattr(self.psychometrics, item),
                    'stim_resp', {'0': [], '0.5': [], '1': []})
            setattr(getattr(self.psychometrics, item),
                    'stim_resp_raw', {
                        '0': np.empty(n_sec**2,
                                      dtype=np.ndarray),
                        '0.5': np.empty(n_sec**2,
                                        dtype=np.ndarray),
                        '1': np.empty(n_sec**2,
                                      dtype=np.ndarray)})
        self.psychometrics.grab_null.stim_resp = {
            '0': np.empty(n_sec**2, dtype=np.ndarray),
            '0.5': np.empty(n_sec**2, dtype=np.ndarray),
            '1': np.empty(n_sec**2, dtype=np.ndarray)}
        self.psychometrics.grab_null.stim_resp_raw = {
            '0': np.empty(n_sec**2, dtype=np.ndarray),
            '0.5': np.empty(n_sec**2, dtype=np.ndarray),
            '1': np.empty(n_sec**2, dtype=np.ndarray)}

        _ind_stim_start = np.argmin(np.abs(self.sector.t - 0))
        _ind_stim_end = np.argmin(np.abs(self.sector.t - 2))
        dt = self.sector.t[1] - self.sector.t[0]

        # extract licking data
        # -------------
        print('\textracting lick data...')
        for trial_cond in ['0', '0.5', '1']:
            self.psychometrics.lick.stim_resp[trial_cond] = np.mean(
                self.beh.lick.antic[trial_cond])

        # extract GRAB full-frame data
        # ---------------
        print('\textracting GRAB full-frame data...')
        for trial_cond in ['0', '0.5', '1']:
            self.psychometrics.grab_frame.stim_resp[trial_cond] = np.mean(
                sp.integrate.trapezoid(
                    self.frame.dff[trial_cond]
                    [:, _ind_stim_start:_ind_stim_end], dx=dt, axis=1), axis=0)
            self.psychometrics.grab_frame.stim_resp_raw[trial_cond] \
                = sp.integrate.trapezoid(
                    self.frame.dff[trial_cond]
                    [:, _ind_stim_start:_ind_stim_end], dx=dt, axis=1)

        # extract GRAB sector data
        # -----------
        print('\textracting GRAB sector data...')
        # setup sector.stim_resp for storing per-trial stim responses
        self.sector.stim_resp = {}
        for tr_cond in ['0', '0.5', '1']:
            self.sector.stim_resp[tr_cond] = np.zeros((
                self.beh.tr_inds[tr_cond].shape[0],
                self.sector.n_sectors, self.sector.n_sectors))

        _n_sec = 0
        for n_x in range(self.sector.n_sectors):
            for n_y in range(self.sector.n_sectors):
                print(f'\t\tsector {_n_sec}/{self.sector.n_sectors**2}...',
                      end='\r')
                for trial_cond in ['0', '0.5', '1']:

                    # regular GRAB
                    # --------
                    _dff_raw = self.sector.dff[n_x, n_y][trial_cond]
                    _stim_resp = sp.integrate.trapezoid(
                           _dff_raw[:, _ind_stim_start:_ind_stim_end], axis=1,
                           dx=dt)

                    self.psychometrics.grab.stim_resp_raw[
                        trial_cond][_n_sec] = _stim_resp
                    self.sector.stim_resp[trial_cond][:, n_x, n_y] \
                        = _stim_resp

                    self.psychometrics.grab.stim_resp[trial_cond].append(
                       np.mean(_stim_resp))

                    # GRAB null - generate null responses on-the-fly
                    # (memory-efficient: doesn't store full n_null traces)
                    # ---------
                    self.psychometrics.grab_null.stim_resp[trial_cond][
                        _n_sec] = []

                    # Get parameters for this sector
                    _params = self.sector_null.params[n_x, n_y]
                    _ampli_scaling = _params['ampli_scaling']
                    _bl_std = _params['bl_std']
                    _frame_data = self.frame.dff[trial_cond]
                    n_trials, n_frames = _frame_data.shape
                    _n_null = getattr(self.sector_null, 'n_null',
                                      self.sector.n_null)

                    # Generate null responses on-the-fly (vectorized)
                    for ind_null_sim in range(_n_null):
                        _noise = np.random.normal(
                            scale=_bl_std, size=(n_trials, n_frames))
                        _null_traces = (_frame_data * _ampli_scaling) + _noise
                        _dff = np.mean(_null_traces, axis=0)
                        _mean_stim_resp = sp.integrate.trapezoid(
                            _dff[_ind_stim_start:_ind_stim_end], dx=dt)
                        self.psychometrics.grab_null.stim_resp[
                            trial_cond][_n_sec].append(_mean_stim_resp)

                _n_sec += 1

        # extract neur (GCaMP) data
        # --------------
        print('\textracting neur data...')
        n_neurs = self.neur.dff_aln.shape[0]
        self.neur.stim_resp = {}
        for tr_cond in ['0', '0.5', '1']:
            self.neur.stim_resp[tr_cond] = np.zeros((
                self.beh.tr_inds[tr_cond].shape[0],
                n_neurs))

        for neur in range(n_neurs):
            print(f'\t\tneur {neur}/{n_neurs}...',
                  end='\r')
            for trial_cond in ['0', '0.5', '1']:
                _dff_raw = self.neur.dff_aln_mean[trial_cond][neur, :]
                _mean_stim_resp = np.mean(
                    _dff_raw[_ind_stim_start:_ind_stim_end])
                self.psychometrics.neur.stim_resp[trial_cond].append(
                    _mean_stim_resp)

                self.neur.stim_resp[trial_cond][:, neur] \
                    = sp.integrate.trapezoid(
                        self.neur.dff_aln[neur][trial_cond]
                        [:, _ind_stim_start:_ind_stim_end], dx=dt, axis=1)

        # compute psychometrics
        # -----------
        print('\tcomputing optimism...')

        # **** GRAB *****
        _raw = {
            '0-0.5': np.array(self.psychometrics.grab_frame.stim_resp['0.5'])
            - np.array(self.psychometrics.grab_frame.stim_resp['0']),
            '0.5-1': np.array(self.psychometrics.grab_frame.stim_resp['1'])
            - np.array(self.psychometrics.grab_frame.stim_resp['0.5']),
            '0-1': np.array(self.psychometrics.grab_frame.stim_resp['1'])
            - np.array(self.psychometrics.grab_frame.stim_resp['0'])}
        _optimism = ((_raw['0-0.5']) / (_raw['0-0.5'] + _raw['0.5-1'])) - 0.5
        self.psychometrics.grab_frame.optimism = _optimism

        # calculation on sector-by-sector GRAB (all trials)
        _raw = {
            '0-0.5': np.array(self.psychometrics.grab.stim_resp['0.5'])
            - np.array(self.psychometrics.grab.stim_resp['0']),
            '0.5-1': np.array(self.psychometrics.grab.stim_resp['1'])
            - np.array(self.psychometrics.grab.stim_resp['0.5']),
            '0-1': np.array(self.psychometrics.grab.stim_resp['1'])
            - np.array(self.psychometrics.grab.stim_resp['0'])}
        _optimism = ((_raw['0-0.5']) / (_raw['0-0.5'] + _raw['0.5-1'])) - 0.5
        self.psychometrics.grab.optimism = _optimism

        # calculation on sector-by-sector GRAB (trial split)
        self.psychometrics.grab.optimism_trsplit_a = []
        self.psychometrics.grab.optimism_trsplit_b = []

        for sector in range(self.sector.n_sectors**2):
            _raw_trsplit_a = {
                '0-0.5': np.mean(self.psychometrics.grab.stim_resp_raw['0.5'][
                    sector][::2])
                - np.mean(self.psychometrics.grab.stim_resp_raw['0'][
                    sector][::2]),
                '0.5-1': np.mean(self.psychometrics.grab.stim_resp_raw['1'][
                    sector][::2])
                - np.mean(self.psychometrics.grab.stim_resp_raw['0.5'][
                    sector][::2])}
            _raw_trsplit_b = {
                '0-0.5': np.mean(self.psychometrics.grab.stim_resp_raw['0.5'][
                    sector][1::2])
                - np.mean(self.psychometrics.grab.stim_resp_raw['0'][
                    sector][1::2]),
                '0.5-1': np.mean(self.psychometrics.grab.stim_resp_raw['1'][
                    sector][1::2])
                - np.mean(self.psychometrics.grab.stim_resp_raw['0.5'][
                    sector][1::2])}
            _optimism_trsplit_a = (
                (_raw_trsplit_a['0-0.5']) / (_raw_trsplit_a['0-0.5']
                                             + _raw_trsplit_a['0.5-1'])) - 0.5
            _optimism_trsplit_b = (
                (_raw_trsplit_b['0-0.5']) / (_raw_trsplit_b['0-0.5']
                                             + _raw_trsplit_b['0.5-1'])) - 0.5

            self.psychometrics.grab.optimism_trsplit_a.append(
                _optimism_trsplit_a)
            self.psychometrics.grab.optimism_trsplit_b.append(
                _optimism_trsplit_b)

        self.psychometrics.grab.optimism_trsplit_a = np.array(
            self.psychometrics.grab.optimism_trsplit_a)
        self.psychometrics.grab.optimism_trsplit_b = np.array(
            self.psychometrics.grab.optimism_trsplit_b)

        # ******** GRAB null ********
        self.psychometrics.grab_null.optimism = np.empty(
            self.sector.n_sectors**2, dtype=np.ndarray)
        for sector in range(self.sector.n_sectors**2):
            _raw = {
                '0-0.5': np.array(
                    self.psychometrics.grab_null.stim_resp['0.5'][sector])
                - np.array(
                    self.psychometrics.grab_null.stim_resp['0'][sector]),
                '0.5-1': np.array(
                    self.psychometrics.grab.stim_resp['1'][sector])
                - np.array(
                    self.psychometrics.grab.stim_resp['0.5'][sector]),
                '0-1': np.array(
                    self.psychometrics.grab.stim_resp['1'][sector])
                - np.array(
                    self.psychometrics.grab.stim_resp['0'][sector])}
            _optimism = ((_raw['0-0.5']) / (_raw['0-0.5']
                                            + _raw['0.5-1'])) - 0.5
            self.psychometrics.grab_null.optimism[sector] = _optimism

        # ******* GCaMP *******
        _raw = {
            '0-0.5': np.array(self.psychometrics.neur.stim_resp['0.5'])
            - np.array(self.psychometrics.neur.stim_resp['0']),
            '0.5-1': np.array(self.psychometrics.neur.stim_resp['1'])
            - np.array(self.psychometrics.neur.stim_resp['0.5']),
            '0-1': np.array(self.psychometrics.neur.stim_resp['1'])
            - np.array(self.psychometrics.neur.stim_resp['0'])}
        _optimism = ((_raw['0-0.5']) / (_raw['0-0.5'] + _raw['0.5-1'])) - 0.5
        self.psychometrics.neur.optimism = _optimism

        # ***** lick *****
        _raw = {
            '0-0.5': np.array(self.psychometrics.lick.stim_resp['0.5'])
            - np.array(self.psychometrics.lick.stim_resp['0']),
            '0.5-1': np.array(self.psychometrics.lick.stim_resp['1'])
            - np.array(self.psychometrics.lick.stim_resp['0.5']),
            '0-1': np.array(self.psychometrics.lick.stim_resp['1'])
            - np.array(self.psychometrics.lick.stim_resp['0'])}
        _optimism = ((_raw['0-0.5']) / (_raw['0-0.5'] + _raw['0.5-1'])) - 0.5

        self.psychometrics.lick.optimism = _optimism

        return

    def add_null_dists(self, n_null=100, auto_run_methods=True,
                       t_pre=2, t_post=5, channel=None):
        """
        Makes a 'null' distribution based on the alternate hypothesis
        of no spatial variability in task-related serotonin
        signaling, but instead spatial variability in 1) GRAB sensor
        expression; and 2) imaging-related noise.
        """
        print('creating null distributions'
              + (f' ({channel})' if channel else ''))

        print('\trunning checks')

        if not hasattr(self, 'frame'):
            if auto_run_methods is False:
                print('Must have run self.add_frame() first!')
                return
            elif auto_run_methods is True:
                self.add_frame(t_pre=t_pre,
                               t_post=t_post,
                               channel=channel)

        if not hasattr(self, 'sector'):
            if auto_run_methods is False:
                print('Must have run self.add_sectors() first!')
                return
            elif auto_run_methods is True:
                self.add_sectors(t_pre=t_pre,
                                 t_post=t_post,
                                 channel=channel)

        self.sector_null = SimpleNamespace()
        self.sector_null.t = self.sector.t
        self.sector_null.n_null = n_null

        self.sector_null.dff = np.empty(
            (self.sector.n_sectors, self.sector.n_sectors), dtype=dict)

        self.sector_null.params = np.empty(
            (self.sector.n_sectors, self.sector.n_sectors), dtype=dict)

        _ind_bl_start = 0
        _ind_bl_end = np.argmin(np.abs(self.sector.t - 0))
        ampli_frame = np.max(np.mean(self.frame.dff['1'], axis=0))
        self.sector_null.ampli_frame = ampli_frame

        print('\tcomputing null parameters for each sector')
        _n_trace = 0
        for n_x in range(self.sector.n_sectors):
            for n_y in range(self.sector.n_sectors):
                if (_n_trace + 1) % 10 == 0:
                    print(f'\t\tsector {_n_trace+1}/'
                          f'{self.sector.n_sectors**2}...',
                          end='\r')

                _ampli_sector = np.max(np.mean(
                    self.sector.dff[n_x, n_y]['1'], axis=0))
                _ampli_scaling = _ampli_sector / ampli_frame

                _bl_data = self.sector.dff[n_x, n_y]['1'][
                    :, _ind_bl_start:_ind_bl_end]
                _bl_std = np.mean(np.std(_bl_data, axis=1))

                self.sector_null.params[n_x, n_y] = {
                    'ampli_scaling': _ampli_scaling,
                    'bl_std': _bl_std
                }

                self.sector_null.dff[n_x, n_y] = {}
                for tr_cond in ['0', '0.5', '1', '0.5_rew', '0.5_norew']:
                    _frame_data = self.frame.dff[tr_cond]
                    n_trials, n_frames = _frame_data.shape
                    _noise = np.random.normal(
                        scale=_bl_std, size=(1, n_trials, n_frames))
                    self.sector_null.dff[n_x, n_y][tr_cond] = \
                        (_frame_data * _ampli_scaling) + _noise

                _n_trace += 1

        print(f'\t\tsector {self.sector.n_sectors**2}/'
              f'{self.sector.n_sectors**2}...done')
        return

    def add_zetatest_neurs(self, frametime_post=2,
                           zeta_type='2samp'):

        print('adding zetatest for neurs...')
        # setup structure and calculate on/off times
        inds_neur = np.where(self.neur.iscell[:, 0] == 1)[0]
        n_neurs = inds_neur.shape[0]

        framerate = 1 / (self.neur.t[1] - self.neur.t[0])
        frames_post = np.ceil(frametime_post * framerate)

        t_on_0 = self.beh.stim.t_start[self.beh.tr_inds['0']]
        t_onoff_0 = np.tile(t_on_0, (2, 1)).T
        t_onoff_0[:, 1] += 2

        t_on_1 = self.beh.stim.t_start[self.beh.tr_inds['1']]
        t_onoff_1 = np.tile(t_on_1, (2, 1)).T
        t_onoff_1[:, 1] += 2

        self.neur.zeta_pval = np.zeros(n_neurs)
        self.neur.mean_pval = np.zeros(n_neurs)
        for ind_neur_rel, ind_neur in enumerate(inds_neur):
            print(f'\tneur={int(ind_neur)}...  ', end='\r')
            _t = self.neur.t
            _f = self.neur.f[ind_neur, :]

            if zeta_type == '2samp':
                _zeta = zetapy.zetatstest2(_t, _f, t_onoff_0,
                                           _t, _f, t_onoff_1,
                                           dblUseMaxDur=frames_post)
            elif zeta_type == '1samp':
                _zeta = zetapy.zetatstest(_t, _f, t_onoff_1,
                                          dblUseMaxDur=frames_post)

            self.neur.zeta_pval[ind_neur_rel] = _zeta[1]['dblZetaP']
            self.neur.mean_pval[ind_neur_rel] = _zeta[1]['dblMeanP']
        print('')

        return

    def add_zetatest_sectors(self, frametime_post=2,
                             zeta_type='2samp', channel=None):

        if not hasattr(self, 'sector'):
            self.add_sectors(channel=channel)

        print('adding zetatest for sectors' +
              (f' ({channel})' if channel else '') + '...')
        rec = self._get_rec(channel)

        # setup structure and calculate on/off times
        n_sectors = self.sector.n_sectors**2

        framerate = 1 / (self.neur.t[1] - self.neur.t[0])
        frames_post = np.ceil(frametime_post * framerate)

        t_on_0 = self.beh.stim.t_start[self.beh.tr_inds['0']]
        t_onoff_0 = np.tile(t_on_0, (2, 1)).T
        t_onoff_0[:, 1] += 2

        t_on_1 = self.beh.stim.t_start[self.beh.tr_inds['1']]
        t_onoff_1 = np.tile(t_on_1, (2, 1)).T
        t_onoff_1[:, 1] += 2

        self.sector.zeta_pval = np.zeros(n_sectors)
        self.sector.mean_pval = np.zeros(n_sectors)
        _n_sec = 0
        for n_x in range(self.sector.n_sectors):
            for n_y in range(self.sector.n_sectors):
                print(f'\t\tsector {_n_sec}/{n_sectors}...      ',
                      end='\r')

                _t = self.rec_t
                _f = np.mean(np.mean(
                    rec[:,
                        self.sector.x.lower[n_x, n_y]:
                        self.sector.x.upper[n_x, n_y],
                        self.sector.y.lower[n_x, n_y]:
                        self.sector.y.upper[n_x, n_y]],
                    axis=1), axis=1)

                if zeta_type == '2samp':
                    _zeta = zetapy.zetatstest2(_t, _f, t_onoff_0,
                                               _t, _f, t_onoff_1,
                                               dblUseMaxDur=frames_post)
                elif zeta_type == '1samp':
                    _zeta = zetapy.zetatstest(_t, _f, t_onoff_1,
                                              dblUseMaxDur=frames_post)

                self.sector.zeta_pval[_n_sec] = _zeta[1]['dblZetaP']
                self.sector.mean_pval[_n_sec] = _zeta[1]['dblMeanP']
                _n_sec += 1
        print('')

        return

    def get_antic_licks_by_trialtype(self, bl_norm=False):
        _tr_counts = {'0': 0, '0.5': 0, '1': 0}
        lickrates_hz = {'0': 0, '0.5': 0, '1': 0}
        for trial in range(self.beh._stimrange.first,
                           self.beh._stimrange.last):
            if bl_norm is True:
                _antic_licks = self.beh.lick.antic_raw[trial] \
                    - self.beh.lick.base[trial]
            elif bl_norm is False:
                _antic_licks = self.beh.lick.antic_raw[trial]

            if self.beh.stim.prob[trial] == 0:
                lickrates_hz['0'] += _antic_licks
                _tr_counts['0'] += 1
            if self.beh.stim.prob[trial] == 0.5:
                lickrates_hz['0.5'] += _antic_licks
                _tr_counts['0.5'] += 1
            if self.beh.stim.prob[trial] == 1:
                lickrates_hz['1'] += _antic_licks
                _tr_counts['1'] += 1

        lickrates_hz['0'] /= _tr_counts['0']
        lickrates_hz['0.5'] /= _tr_counts['0.5']
        lickrates_hz['1'] /= _tr_counts['1']

        return lickrates_hz
