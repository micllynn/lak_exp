from types import SimpleNamespace
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import sklearn
import sklearn.linear_model
import sklearn.preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
import scipy.stats as sp_stats

import zetapy

import os
import pathlib
import tifffile

from .align_imgbeh import Aligner_ImgBeh
from .utils import find_event_onsets_autothresh, remove_lick_artefact_after_rew, \
    check_folder_exists
from .utils_twop import XMLParser, calc_dff
from .beh import BehDataSimpleLoad, StimParser
from . import signal_correction


class TwoPRec(object):
    def __init__(self,
                 enclosing_folder=None,
                 folder_beh=None,
                 folder_img=None,
                 fname_img=None,
                 dset_obj=None,
                 dset_ind=None,
                 ch_img=2,
                 trial_end=None,
                 rec_type='trig_rew',
                 parse_stims=True,
                 parse_by='stimulusOrientation',
                 trial_conditions=None,
                 n_px_remove_sides=10):
        """
        Loads a 2p recording (tiff) and associated behavioral folder.

        Two methods of loading are available.
        ---------------
        1. Load using a dataset object built for your experiment.
        (Specify dset_obj and dset_ind, and optionally ch_img.)
            - dset_obj is a DSetObj_5HTCtx object
            which references a .csv containing full ExpRef information for
            each recording.
            - dset_ind is an index within this dataset specifying
            recording number.
            - ch_img specifies which channel (1 or 2) tiff to
            load for the recording.
        2. Load by manually inputting folders related to the recording.
        (Specify an enclosing_folder, a folder_beh, a folder_img,
        and a fname_img.)
            - enclosing_folder is a string specifying the ExpRef folder
            (ie 'Data/MBLXXX/2025-XX-XX/')
            - folder_beh is a string specifying the behavior folder,
            referenced from the enclosing_folder location.
            (ie '1' or '2')
            - folder_img is a string specifying the imaging folder,
            referenced from the enclosing_folder location.
            (ie 'TwoP/2025-06-17_t-001')
            - fname_img is a string specifying the imaging name
            within folder_img
            (ie '2025-06-17_t-003_Cycle00001_Ch2.tif')

        Other parameters
        ----------------
        trial_end : None or int
            Optional parameter specifying the index of the last trial
            to analyze.
        rec_type : str | 'trig_rew' or 'paqio'
            Recording type. Can either be 'trig_rew' (Michael-style),
            where imaging is triggered to start upon the first reward
            delivered during the task, or 'paqio' (Marko/Sandra/Jess style)
            where imaging and behavior acquisitions are manually started,
            and synchronized with a simultaneously recorded .paq file
            that has frame times (imaging start) and reward echoes (beh start)
        n_px_remove_sides : int
            Number of pixels to remove on each side of the frame (necessary if
            dealing with suite2p motion corrected tiffs, as these can
            introduce artifacts with high values on the edges

        Object layout
        -------------
        .rec : np.ndarray
            Raw version of the .tiff file (memory-mapped), with dimensions
            (t, pix_x, pix_y).
        .rec_t : np.ndarray
            Vector with timestamps for the t dimension of .rec, in sec.
        .beh : SimpleNamespace | Behavior data for task
            .stim
            .rew
            .lick
        .ops : SimpleNamespace | stores all relevant options for
        """

        # setup names of folders, files, and ops
        # --------------
        self._init_folders(enclosing_folder, folder_beh, folder_img,
                           dset_obj, dset_ind)
        self.ch_img = ch_img
        self._init_ops(n_px_remove_sides, rec_type)

        # get filename of image of the appropriate channel
        list_img = os.listdir(self.folder.img)
        if fname_img is None:
            for _fname in list_img:
                if f'Ch{ch_img}.tif' in _fname and 'compiled' in _fname:
                    self.fname_img = _fname
        else:
            self.fname_img = fname_img

        # load behavioral data
        # -------------
        self._init_behavior(parse_stims=parse_stims, parse_by=parse_by)

        # load imaging data
        # ------------
        print('loading imaging...')
        for _file in os.listdir(self.folder.img):
            if _file.endswith('BACKUP.xml'):
                try:
                    xmlobj = XMLParser(os.path.join(self.folder.img, _file))
                    sampling_rate = xmlobj.get_framerate()
                    print(f'\tframerate is {sampling_rate:.2f}Hz')
                except:
                    print('could not parse framerate from BACKUP.xml')
                    pass
        self.samp_rate = sampling_rate

        print('\tloading tiff...')
        self.rec = tifffile.memmap(os.path.join(
            self.folder.img, self.fname_img))[
            :, n_px_remove_sides:-1*n_px_remove_sides,
                n_px_remove_sides:-1*n_px_remove_sides]

        # try to load suite2p output if available
        self._init_suite2p()

        # align behavior and imaging data
        # ----------------------
        self._init_timestamps(self.rec.shape[0], rec_type)

        # note the first and last stimulus/rew within recording bounds
        # ---------------
        self._init_stim_rew_range(trial_end=trial_end)

        # compute trial indices for each trtype
        # -------------------
        self.beh.tr_inds = {}
        # Standard case: visual stimuli
        if self.beh._stimparser.parse_by == 'stimulusOrientation':
            self.beh.tr_conds = ['0', '0.5', '1', '0.5_rew', '0.5_norew']

            self.beh.tr_inds['0'] = np.where(self.beh.stim.prob == 0)[0]
            self.beh.tr_inds['0.5'] = np.where(self.beh.stim.prob == 0.5)[0]
            self.beh.tr_inds['1'] = np.where(self.beh.stim.prob == 1)[0]
            self.beh.tr_inds['0.5_rew'] = np.where(np.logical_and(
                self.beh.stim.prob == 0.5, self.beh.rew.delivered == True))[0]
            self.beh.tr_inds['0.5_norew'] = np.where(np.logical_and(
                self.beh.stim.prob == 0.5, self.beh.rew.delivered == False))[0]

            # remove trinds outside of correct range
            for tr_cond in ['0', '0.5', '1', '0.5_rew', '0.5_norew']:
                self.beh.tr_inds[tr_cond] = np.delete(
                    self.beh.tr_inds[tr_cond],
                    ~np.isin(self.beh.tr_inds[tr_cond],
                             np.arange(
                                 self.beh._stimrange.first,
                                 self.beh._stimrange.last+1)))
            # add lickrates
            # ----------------
            self.add_lickrates()

            self.beh.tr_inds['0.5_prelick'] = np.where(np.logical_and(
                self.beh.stim.prob == 0.5, self.beh.lick.antic_raw > 0))[0]
            self.beh.tr_inds['0.5_noprelick'] = np.where(np.logical_and(
                self.beh.stim.prob == 0.5, self.beh.lick.antic_raw == 0))[0]
            for tr_cond in ['0.5_prelick', '0.5_noprelick']:
                self.beh.tr_inds[tr_cond] = np.delete(
                    self.beh.tr_inds[tr_cond],
                    ~np.isin(self.beh.tr_inds[tr_cond],
                             np.arange(
                                 self.beh._stimrange.first,
                                 self.beh._stimrange.last+1)))
        # Other visual stimuli
        else:
            self.beh.tr_conds = self.beh._stimparser.parsed_param.astype(str)
            self.beh.tr_conds_outcome = (self.beh._stimparser.prob
                                         * self.beh._stimparser.size)

            for tr_cond in self.beh.tr_conds:
                self.beh.tr_inds[str(tr_cond)] = np.where(
                    self.beh._stimparser._all_parsed_param == int(tr_cond))[0]

            # remove trinds outside of correct range
            for tr_cond in self.beh.tr_conds:
                self.beh.tr_inds[tr_cond] = np.delete(
                    self.beh.tr_inds[tr_cond],
                    ~np.isin(self.beh.tr_inds[tr_cond],
                             np.arange(
                                 self.beh._stimrange.first,
                                 self.beh._stimrange.last+1)))
            # add lickrates
            # ----------------
            self.add_lickrates()

        # build _trial_cond_map: maps each trial index -> list of condition keys
        self._trial_cond_map = {}
        for _cond, _inds in self.beh.tr_inds.items():
            for _idx in _inds:
                self._trial_cond_map.setdefault(int(_idx), []).append(_cond)
        return

    # ------------------------------------------------------------------
    # Shared __init__ helpers (called by both TwoPRec and
    # TwoPRec_DualColour.__init__)
    # ------------------------------------------------------------------

    def _init_folders(self, enclosing_folder, folder_beh, folder_img,
                      dset_obj, dset_ind):
        """Set up self.folder and self.path from explicit paths or a dataset
        object, and chdir to the enclosing folder."""
        self.folder = SimpleNamespace()
        self.path = SimpleNamespace()
        if dset_obj is None:
            self.folder.enclosing = enclosing_folder
            self.folder.img = folder_img
            self.folder.beh = folder_beh
            print(f'loading {self.folder.enclosing}...\n---------------')
        elif 'DSetObj' in str(type(dset_obj)):
            self.dset_obj = dset_obj
            self.folder.enclosing = self.dset_obj.get_path_expref(dset_ind)
            self.folder.img = self.dset_obj.get_path_img(dset_ind)
            self.folder.beh = self.dset_obj.get_path_beh(dset_ind)
            print(f'loading {self.dset_obj.get_path_expref(dset_ind)}...')
            print('---------------')

        self.path.raw = pathlib.Path(self.folder.enclosing)
        self.path.animal = self.path.raw.parts[-2]
        self.path.date = self.path.raw.parts[-1]
        self.path.beh_folder = self.folder.beh

        os.chdir(self.folder.enclosing)
        check_folder_exists('figs_mbl')

    def _init_ops(self, n_px_remove_sides, rec_type):
        """Initialise self.ops namespace with operational parameters."""
        self.ops = SimpleNamespace()
        self.ops.n_px_remove_sides = n_px_remove_sides
        self.ops.rec_type = rec_type

    def _init_behavior(self, parse_stims=True, parse_by='stimulusOrientation'):
        """Load behavioral data into self.beh from self.folder.beh.

        Populates self.beh.rew, self.beh.stim, and self.beh.lick.
        """
        self.beh = BehDataSimpleLoad(self.folder.beh,
                                     parse_stims=parse_stims,
                                     parse_by=parse_by)

        self.beh.rew = SimpleNamespace()
        self.beh.stim = SimpleNamespace()

        self.beh.rew.delivered = self.beh._data.get_event_var(
            'isRewardGivenValues')
        self.beh.rew.t = self.beh._data.get_event_var('totalRewardTimes')[
            np.where(self.beh.rew.delivered == 1)[0]]

        _daq_data = self.beh._timeline.get_daq_data()
        self.beh.licks = find_event_onsets_autothresh(
            _daq_data.sig['lickDetector'])
        self.beh.t_licks = _daq_data.t[self.beh.licks]

        self.beh.stim.t_start = self.beh._data.get_event_var(
            'stimulusOnTimes')
        self.beh.stim.stimlist = StimParser(self.beh, parse_by=parse_by)

        self.beh.stim.prob = self.beh.stim.stimlist._all_stimprobs
        self.beh.stim.size = self.beh.stim.stimlist._all_stimsizes

        self.beh.lick = SimpleNamespace()

        t_licksig = self.beh._daq_data.t
        lick_onset_inds = find_event_onsets_autothresh(
            self.beh._daq_data.sig['lickDetector'], n_stdevs=4)
        self.beh.lick.t_raw = t_licksig[lick_onset_inds]

    def _init_suite2p(self):
        """Try to load suite2p output from self.folder.img into self.neur."""
        print('\tloading suite2p neurs...')
        try:
            self.neur = SimpleNamespace()
            self.neur.f = np.load(
                os.path.join(self.folder.img,
                             'suite2p', 'plane0', 'F.npy'),
                allow_pickle=True)
            self.neur.ops = np.load(
                os.path.join(self.folder.img,
                             'suite2p', 'plane0', 'ops.npy'),
                allow_pickle=True)
            self.neur.stat = np.load(
                os.path.join(self.folder.img,
                             'suite2p', 'plane0', 'stat.npy'),
                allow_pickle=True)
            self.neur.iscell = np.load(
                os.path.join(self.folder.img,
                             'suite2p', 'plane0', 'iscell.npy'),
                allow_pickle=True)
        except Exception as e:
            print('\tcould not load suite2p output:')
            print(f'\t\t{e}')

    def _init_timestamps(self, n_frames, rec_type):
        """Create self.rec_t timestamps and store in self.neur.t if available.

        Parameters
        ----------
        n_frames : int
            Number of frames in the recording (used to construct rec_t).
        rec_type : str
            'trig_rew' or 'paqio' — determines alignment method.
        """
        print('\tcreating timestamps...')
        if rec_type == 'trig_rew':
            _t_start = self.beh.rew.t[0]
            _t_end = (n_frames / self.samp_rate) + _t_start
            self.rec_t = np.linspace(_t_start, _t_end, num=n_frames)
        elif rec_type == 'paqio':
            self._aligner = Aligner_ImgBeh()
            self._aligner.parse_img_rewechoes()
            self._aligner.parse_beh_rewechoes()
            self._aligner.compute_alignment()
            _t_start = 0
            _t_end = n_frames / self.samp_rate
            self.rec_t = np.linspace(_t_start, _t_end, num=n_frames)
            self.rec_t = self._aligner.correct_img_data(self.rec_t)

        if hasattr(self, 'neur'):
            self.neur.t = self.rec_t

    def _init_stim_rew_range(self, trial_end=None):
        """Compute self.beh._stimrange and self.beh._rewrange.

        Finds the first and last stimulus/reward indices that fall within
        the recording window. Optionally clips the last trial to trial_end.
        """
        # stims
        self.beh._stimrange = SimpleNamespace()
        n_stims = self.beh.stim.t_start.shape[0]

        _temp_first = 0
        _temp_last = n_stims - 1
        for ind_stim in range(n_stims):
            if self.beh.stim.t_start[ind_stim] + 4 > self.rec_t[-1]:
                _temp_last = ind_stim - 1
            if self.beh.stim.t_start[ind_stim] - 2 < self.rec_t[0]:
                _temp_first = ind_stim + 1

        self.beh._stimrange.first = _temp_first
        self.beh._stimrange.last = _temp_last

        # rews
        self.beh._rewrange = SimpleNamespace()
        n_rews = self.beh.rew.t.shape[0]

        _temp_first = 0
        _temp_last = n_rews - 1
        for ind_rew in range(n_rews):
            if self.beh.rew.t[ind_rew] + 4 > self.rec_t[-1]:
                _temp_last = ind_stim - 1
            if self.beh.rew.t[ind_rew] - 2 < self.rec_t[0]:
                _temp_first = ind_stim + 1

        self.beh._rewrange.first = _temp_first
        self.beh._rewrange.last = _temp_last

        # if trial_end is manually specified, replace these attributes
        if trial_end is not None:
            self.beh._stimrange.last = trial_end
            self.beh._rewrange.last = trial_end

    # ------------------------------------------------------------------

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
        # for trial_type in ['0', '0.5', '1', '0.5_rew', '0.5_norew']:
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

    def _get_rec(self, channel=None):
        """
        Returns the recording array. In the base class there is only a single
        channel, so ``channel`` is accepted but ignored.

        Subclasses (e.g. TwoPRec_DualColour) override this to select between
        red and green channels.

        Parameters
        ----------
        channel : str or None
            Ignored in the base class. Pass 'red' or 'grn' when using a
            subclass that supports dual-colour recordings.

        Returns
        -------
        np.ndarray
            The recording array (self.rec).
        """
        return self.rec

    def _get_rec_t(self, channel=None):
        """
        Returns the timestamp array for the recording. In the base class there
        is only a single set of timestamps, so ``channel`` is accepted but
        ignored.

        Subclasses (e.g. TwoPRec_DualColour) override this to return
        channel-specific timestamps when they differ (e.g. after signal
        correction that truncates the green channel).

        Parameters
        ----------
        channel : str or None
            Ignored in the base class.

        Returns
        -------
        np.ndarray
            The timestamp array (self.rec_t).
        """
        return self.rec_t

    def add_frame(self, t_pre=2, t_post=5, channel=None, use_zscore=False):

        """
        Creates trial-averaged signal across the whole field of view,
        separated by trial-type (eg 0%, 50%, 100% rewarded trials).

        plot_type controls the things plotted.
            None: plots 0, 0.5, 1
            rew_norew: plots 0.5_rew, 0.5_norew
            prelick_noprelick: plots 0.5_prelick, 0.5_noprelick

        Parameters
        ----------
        channel : str or None
            Passed to _get_rec() / _get_rec_t(). Ignored in the base class;
            used by TwoPRec_DualColour to select the channel.
        use_zscore : bool
            If True, compute z-scored fluorescence (baseline mean and std over
            the pre-stimulus window) instead of df/f. If False (default), df/f
            is used unless the baseline is near zero (corrected residual signal),
            in which case z-score is applied automatically with a warning.
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
        n_frames_pre = int(t_pre * self.samp_rate)
        n_frames_post = int((2 + t_post) * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post

        self.frame.dff = {}
        for _cond in self.beh.tr_inds:
            self.frame.dff[_cond] = np.zeros((
                self.beh.tr_inds[_cond].shape[0], n_frames_tot))
        self.frame.t = np.linspace(-1*t_pre, 2+t_post, n_frames_tot)
        self.frame.tr_counts = {k: 0 for k in self.beh.tr_inds}

        # Auto-detect near-zero baseline (corrected residual signal) and fall
        # back to z-score with a warning rather than dividing by ~0.
        _use_zscore = use_zscore
        if not _use_zscore and self.beh._stimrange.first < self.beh._stimrange.last:
            _t0 = self.beh.stim.t_start[self.beh._stimrange.first]
            _ind0 = np.searchsorted(rec_t, _t0 - t_pre)
            _probe = np.mean(np.mean(
                rec[_ind0:_ind0 + n_frames_pre, :, :], axis=1), axis=1)
            _f0_probe = float(np.mean(_probe)) if _probe.size > 0 else 1.0
            if np.abs(_f0_probe) < 1.0:
                print(f'\tWarning: baseline fluorescence mean ({_f0_probe:.4f}) '
                      'is near zero. Signal appears to be a corrected residual. '
                      'Falling back to z-score. Pass use_zscore=True explicitly '
                      'to suppress this check.')
                _use_zscore = True

        _trials = np.arange(self.beh._stimrange.first, self.beh._stimrange.last)
        _all_ind_t_start = np.searchsorted(
            rec_t, self.beh.stim.t_start[_trials] - t_pre)
        for _i, trial in enumerate(_trials):
            print(f'\t{trial=}', end='\r')
            _ind_t_start = _all_ind_t_start[_i]
            _f = np.mean(np.mean(
                rec[_ind_t_start:_ind_t_start + n_frames_tot, :, :],
                axis=1), axis=1)
            if _use_zscore:
                _baseline = _f[:n_frames_pre]
                _sigma = np.std(_baseline)
                _dff = (_f - np.mean(_baseline)) / _sigma \
                    if _sigma > 0 else np.zeros_like(_f)
            else:
                _dff = calc_dff(_f, baseline_frames=n_frames_pre)
            for _cond in self._trial_cond_map.get(int(trial), []):
                if _cond in self.frame.dff:
                    self.frame.dff[_cond][
                        self.frame.tr_counts[_cond], :] = _dff[0:n_frames_tot]
                    self.frame.tr_counts[_cond] += 1

    def add_sectors(self,
                    n_sectors=10,
                    t_pre=2, t_post=5,
                    n_null=500,
                    resid_type='sector',
                    resid_corr_tr_cond='1',
                    channel=None,
                    colors=sns.cubehelix_palette(
                        n_colors=3,
                        start=2, rot=0,
                        dark=0.2, light=0.8)):
        """
        Divides the field of view into sectors, and plots a set of trial-types
        separately within each sector.

        plot_type controls the type of plot:
            if plot_type is None:
                stim_list = ['0', '0.5', '1']
            elif plot_type == 'rew_norew':
                stim_list = ['0.5_rew', '0.5_norew']
            elif plot_type == 'prelick_noprelick':
                stim_list = ['0.5_prelick', '0.5_noprelick']
            elif plot_type == 'rew':
                stim_list = ['0', '0.5_rew', '1']

        Parameters
        ----------
        channel : str or None
            Passed to _get_rec() / _get_rec_t(). Ignored in the base class;
            used by TwoPRec_DualColour to select the channel.
        """
        # compute whole-frame avg in case needed
        if not hasattr(self, 'frame'):
            self.add_frame(t_pre=t_pre, t_post=t_post)

        print('creating trial-averaged signal (sectors)...')
        self.add_lickrates(t_prestim=t_pre,
                           t_poststim=t_post)

        self.sector = SimpleNamespace()

        self.sector.params = SimpleNamespace()
        self.sector.n_sectors = n_sectors
        self.sector.params.t_rew_pre = t_pre
        self.sector.params.t_rew_post = t_post
        self.sector.n_sectors = n_sectors
        self.sector.n_null = n_null

        if self.beh._stimparser.parse_by == 'stimulusOrientation':

            self.sector.colors = {'0': colors[0],
                                  '0.5': colors[1],
                                  '1': colors[2],
                                  '0.5_rew': colors[1],
                                  '0.5_norew': colors[1],
                                  '0.5_prelick': colors[1],
                                  '0.5_noprelick': colors[1]}
            self.sector.linestyle = {'0': 'solid',
                                     '0.5': 'solid',
                                     '1': 'solid',
                                     '0.5_rew': 'solid',
                                     '0.5_norew': 'dashed',
                                     '0.5_prelick': 'solid',
                                     '0.5_noprelick': 'dashed'}

            # setup stim-aligned traces
            n_frames_pre = int(t_pre * self.samp_rate)
            n_frames_post = int((2 + t_post) * self.samp_rate)
            n_frames_tot = n_frames_pre + n_frames_post

            self.sector.dff = np.empty((n_sectors, n_sectors), dtype=dict)
            self.sector.dff_resid = np.empty((n_sectors, n_sectors),
                                             dtype=dict)

            self.sector.t = np.linspace(
                -1*t_pre, 2+t_post, n_frames_tot)

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
                    print(f'\t\tsector {_n_trace}/{n_sectors**2}...      ')
                    # calculate location of sector
                    _ind_x_lower = int((n_x / n_sectors) * self.rec.shape[1])
                    _ind_x_upper = int(((n_x+1) / n_sectors)
                                       * self.rec.shape[1])

                    _ind_y_lower = int((n_y / n_sectors) * self.rec.shape[1])
                    _ind_y_upper = int(((n_y+1) / n_sectors)
                                       * self.rec.shape[1])

                    self.sector.x.lower[n_x, n_y] = _ind_x_lower
                    self.sector.x.upper[n_x, n_y] = _ind_x_upper
                    self.sector.y.lower[n_x, n_y] = _ind_y_lower
                    self.sector.y.upper[n_x, n_y] = _ind_y_upper

                    # setup structure
                    self.sector.dff[n_x, n_y] = {}
                    self.sector.dff_resid[n_x, n_y] = {}

                    for tr_cond in ['0', '0.5', '1', '0.5_rew',
                                    '0.5_norew', '0.5_prelick',
                                    '0.5_noprelick']:
                        self.sector.dff[n_x, n_y][tr_cond] = np.zeros((
                            self.beh.tr_inds[tr_cond].shape[0], n_frames_tot))
                        self.sector.dff_resid[n_x, n_y][tr_cond] \
                            = np.zeros_like(
                                self.sector.dff[n_x, n_y][tr_cond])

                    # store stim-aligned traces
                    self.sector.tr_counts = {'0': 0, '0.5': 0, '1': 0,
                                             '0.5_rew': 0, '0.5_norew': 0,
                                             '0.5_prelick': 0,
                                             '0.5_noprelick': 0}
                    for trial in range(self.beh._stimrange.first,
                                       self.beh._stimrange.last):
                        print(f'\t\t\t{trial=}', end='\r')
                        _t_stim = self.beh.stim.t_start[trial]
                        _t_start = _t_stim - t_pre
                        _t_end = self.beh._data.get_event_var(
                            'totalRewardTimes')[trial] + t_post

                        _ind_t_start = np.argmin(np.abs(
                            self.rec_t - _t_start))
                        _ind_t_end = np.argmin(np.abs(
                            self.rec_t - _t_end)) + 2   # add frames to end

                        # extract fluorescence
                        _f = np.mean(np.mean(
                            self.rec[_ind_t_start:_ind_t_end,
                                     _ind_x_lower:_ind_x_upper,
                                     _ind_y_lower:_ind_y_upper], axis=1),
                                     axis=1)
                        _dff = calc_dff(_f, baseline_frames=n_frames_pre)

                        if self.beh.stim.prob[trial] == 0:
                            self.sector.dff[n_x, n_y]['0'][
                                self.sector.tr_counts['0'], :] \
                                = _dff[0:n_frames_tot]
                            self.sector.tr_counts['0'] += 1
                        elif self.beh.stim.prob[trial] == 0.5:
                            self.sector.dff[n_x, n_y]['0.5'][
                                self.sector.tr_counts['0.5'], :] \
                                = _dff[0:n_frames_tot]
                            self.sector.tr_counts['0.5'] += 1
                        elif self.beh.stim.prob[trial] == 1:
                            self.sector.dff[n_x, n_y]['1'][
                                self.sector.tr_counts['1'], :] \
                                = _dff[0:n_frames_tot]
                            self.sector.tr_counts['1'] += 1

                        if self.beh.stim.prob[trial] == 0.5 \
                           and self.beh.rew.delivered[trial] == 1:
                            self.sector.dff[n_x, n_y]['0.5_rew'][
                                self.sector.tr_counts['0.5_rew'], :] \
                                = _dff[0:n_frames_tot]
                            self.sector.tr_counts['0.5_rew'] += 1
                        elif self.beh.stim.prob[trial] == 0.5 \
                             and self.beh.rew.delivered[trial] == 0:
                            self.sector.dff[n_x, n_y]['0.5_norew'][
                                self.sector.tr_counts['0.5_norew'], :] \
                                = _dff[0:n_frames_tot]
                            self.sector.tr_counts['0.5_norew'] += 1

                        if self.beh.stim.prob[trial] == 0.5 \
                           and self.beh.lick.antic_raw[trial] > 0:
                            self.sector.dff[n_x, n_y]['0.5_prelick'][
                                self.sector.tr_counts['0.5_prelick'], :] \
                                = _dff[0:n_frames_tot]
                            self.sector.tr_counts['0.5_prelick'] += 1
                        elif self.beh.stim.prob[trial] == 0.5 \
                             and self.beh.lick.antic_raw[trial] == 0:
                            self.sector.dff[n_x, n_y]['0.5_noprelick'][
                                self.sector.tr_counts['0.5_noprelick'], :] \
                                = _dff[0:n_frames_tot]
                            self.sector.tr_counts['0.5_noprelick'] += 1

                    # store dff residuals
                    # ----------
                    for tr_cond in ['0', '0.5', '1']:
                        for tr in range(
                                self.sector.dff[n_x, n_y][tr_cond].shape[0]):
                            if resid_type == 'sector':
                                _dff_mean = np.mean(
                                    self.sector.dff[n_x, n_y][tr_cond], axis=0)
                            elif resid_type == 'trial':
                                _dff_mean = self.frame.dff[tr_cond][tr, :]

                            self.sector.dff_resid[n_x, n_y][tr_cond][tr, :] \
                                = self.sector.dff[n_x, n_y][tr_cond][tr, :] \
                                - _dff_mean

                    # update _n_trace
                    _n_trace += 1

            print('\n')
            # compute and plot cross-correlations
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
            self.add_null_dists(n_null=n_null, t_pre=t_pre, t_post=t_post)

        # if not parsing by stimulusOrientation (special case):
        else:
            _tr_conds = self.beh.tr_conds
            _all_trs_cond = self.beh._stimparser._all_parsed_param

            # setup stim-aligned traces
            n_frames_pre = int(t_pre * self.samp_rate)
            n_frames_post = int((2 + t_post) * self.samp_rate)
            n_frames_tot = n_frames_pre + n_frames_post

            self.sector.dff = np.empty((n_sectors, n_sectors), dtype=dict)
            self.sector.dff_resid = np.empty((n_sectors, n_sectors),
                                             dtype=dict)

            self.sector.t = np.linspace(
                -1*t_pre, 2+t_post, n_frames_tot)

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
                    print(f'\t\tsector {_n_trace}/{n_sectors**2}...      ')
                    # calculate location of sector
                    _ind_x_lower = int((n_x / n_sectors) * self.rec.shape[1])
                    _ind_x_upper = int(((n_x+1) / n_sectors)
                                       * self.rec.shape[1])

                    _ind_y_lower = int((n_y / n_sectors) * self.rec.shape[1])
                    _ind_y_upper = int(((n_y+1) / n_sectors)
                                       * self.rec.shape[1])

                    self.sector.x.lower[n_x, n_y] = _ind_x_lower
                    self.sector.x.upper[n_x, n_y] = _ind_x_upper
                    self.sector.y.lower[n_x, n_y] = _ind_y_lower
                    self.sector.y.upper[n_x, n_y] = _ind_y_upper

                    # setup structure
                    self.sector.dff[n_x, n_y] = {}
                    self.sector.dff_resid[n_x, n_y] = {}

                    for _tr_cond in _tr_conds:
                        self.sector.dff[n_x, n_y][_tr_cond] = np.zeros((
                            self.beh.tr_inds[_tr_cond].shape[0], n_frames_tot))
                        self.sector.dff_resid[n_x, n_y][_tr_cond] \
                            = np.zeros_like(
                                self.sector.dff[n_x, n_y][_tr_cond])

                    # store stim-aligned traces
                    self.sector.tr_counts = {k: 0 for k in _tr_conds}
                    for trial in range(self.beh._stimrange.first,
                                       self.beh._stimrange.last):
                        print(f'\t\t\t{trial=}', end='\r')
                        _t_stim = self.beh.stim.t_start[trial]
                        _t_start = _t_stim - t_pre
                        _t_end = self.beh._data.get_event_var(
                            'totalRewardTimes')[trial] + t_post

                        _ind_t_start = np.argmin(np.abs(
                            self.rec_t - _t_start))
                        _ind_t_end = np.argmin(np.abs(
                            self.rec_t - _t_end)) + 2   # add frames to end

                        # extract fluorescence
                        _f = np.mean(np.mean(
                            self.rec[_ind_t_start:_ind_t_end,
                                     _ind_x_lower:_ind_x_upper,
                                     _ind_y_lower:_ind_y_upper], axis=1),
                                     axis=1)
                        _dff = calc_dff(_f, baseline_frames=n_frames_pre)

                        # save in self.sector.dff for the correct tr_cond
                        for _tr_cond in _tr_conds:
                            if _all_trs_cond[trial] == int(_tr_cond):
                                self.sector.dff[n_x, n_y][_tr_cond][
                                    self.sector.tr_counts[_tr_cond], :] \
                                    = _dff[0:n_frames_tot]
                                self.sector.tr_counts[_tr_cond] += 1

                    # store dff residuals
                    # ----------
                    for _tr_cond in _tr_conds:
                        for tr in range(
                                self.sector.dff[n_x, n_y][_tr_cond].shape[0]):
                            if resid_type == 'sector':
                                _dff_mean = np.mean(
                                    self.sector.dff[n_x, n_y][_tr_cond],
                                    axis=0)
                            elif resid_type == 'trial':
                                _dff_mean = self.frame.dff[_tr_cond][tr, :]

                            self.sector.dff_resid[n_x, n_y][_tr_cond][tr, :] \
                                = self.sector.dff[n_x, n_y][_tr_cond][tr, :] \
                                - _dff_mean

                    # update _n_trace
                    _n_trace += 1

            print('\n')
            # compute and plot cross-correlations
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
            self.add_null_dists(n_null=n_null, t_pre=t_pre, t_post=t_post)

        return

    def add_neurs(self, t_pre=1, t_post=2, zscore=True):
        self.add_lickrates(t_prestim=t_pre,
                           t_poststim=t_post)

        print('adding aligned neur avg...')
        # setup structure
        # ---------------
        neurs = np.where(self.neur.iscell[:, 0] == 1)[0]
        n_neurs = neurs.shape[0]

        n_frames_pre = int(t_pre * self.samp_rate)
        n_frames_post = int((2 + t_post) * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post

        _tr_conds = list(self.beh.tr_inds.keys())

        self.neur.dff_aln = np.empty(n_neurs, dtype=dict)
        self.neur.dff_aln_mean = {}
        for _tr_cond in _tr_conds:
            self.neur.dff_aln_mean[_tr_cond] = np.zeros(
                (n_neurs, n_frames_tot))
        self.neur.x = np.zeros(n_neurs)
        self.neur.y = np.zeros(n_neurs)

        self.neur.t_aln = np.linspace(
            -1*t_pre, 2+t_post, n_frames_tot)

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
            self.neur.dff_aln[ind] = {}
            for _tr_cond in _tr_conds:
                self.neur.dff_aln[ind][_tr_cond] = np.zeros((
                    self.beh.tr_inds[_tr_cond].shape[0], n_frames_tot))

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
                    # Each null sim: mean over trials of (scaled_frame + noise)
                    # then integrate over stim window
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
        # calculation on whole-frame GRAB (all trials)
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

    def plt_optimism(self,
                     markersize=20,
                     violin_color=sns.xkcd_rgb['moss green'],
                     zetatest_mask=False,
                     sig_thresh=0.01,
                     fontsize=10,
                     ylims=[-2, 2],
                     plt_show=True):

        # run some checks
        # --------
        if not hasattr(self, 'psychometrics'):
            self.add_psychometrics()
        if zetatest_mask is True and not hasattr(self.sector, 'zeta_pval'):
            self.add_zetatest_sectors()
            self.add_zetatest_neurs()

        # setup
        # ---------
        ind_sig_lower = (sig_thresh *
                         self.psychometrics.grab_null.optimism[0].shape[0])
        ind_sig_upper = (self.psychometrics.grab_null.optimism[0].shape[0]
                         - (sig_thresh *
                            self.psychometrics.grab_null.optimism[0].shape[0]))

        print('plotting optimism...')
        # plot optimism/pessimism, null distribution and significance
        # ----------------
        # setup figures
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

        # plot all
        # ---------------
        n_sec = self.sector.n_sectors
        rec_px = self.rec.shape[1]

        if zetatest_mask is False:
            optimism = self.psychometrics.grab.optimism
            sort_args_grab = np.argsort(optimism)
            optimism_neur = self.psychometrics.neur.optimism
            sort_args_neur = np.argsort(optimism_neur)

            # plot images
            # ---------
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
            # ax_img_neur.set_xlim([-0.5, 9.5])
            # ax_img_neur.set_ylim([-0.5, 9.5])

            # plot distributional rl
            # ------------------
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
                # test for significance: grab
                _sorted_null = np.sort(
                    self.psychometrics.grab_null.optimism[sort_arg])
                _rank = np.argmin(np.abs(
                    _sorted_null - optimism[sort_arg]))
                if _rank <= ind_sig_lower or _rank >= ind_sig_upper:
                    ax_distrl_grab_signif.text(
                        ind, 0, '*', ha='center',
                        va='bottom', fontsize=fontsize)

                # plot
                parts = ax_distrl_grab.violinplot(
                    self.psychometrics.grab_null.optimism[sort_arg],
                    [ind], widths=1,
                    showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(violin_color)

            # plt reliability
            # ---------------
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
            # _line_y = _linreg.slope * _line_x + _linreg.intercept
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

            # plot images
            # ---------
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

            # ax_img_neur.set_xlim([-0.5, 9.5])
            # ax_img_neur.set_ylim([-0.5, 9.5])
            # plot distributional rl
            # ------------------
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
                # test for significance: grab
                _ind_in_full_grab = zetatest_grab_inds[sort_arg]
                _sorted_null = np.sort(
                    self.psychometrics.grab_null.optimism[_ind_in_full_grab])
                _rank = np.argmin(np.abs(
                    _sorted_null - optimism_grab[sort_arg]))
                if _rank <= ind_sig_lower or _rank >= ind_sig_upper:
                    ax_distrl_grab_signif.text(
                        ind, 0, '*', ha='center',
                        va='bottom', fontsize=fontsize)

                # plot
                parts = ax_distrl_grab.violinplot(
                    self.psychometrics.grab_null.optimism[_ind_in_full_grab],
                    [ind], widths=1,
                    showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(violin_color)

            # plt reliability
            # ---------------
            ax_distrl_reliability.scatter(
                self.psychometrics.grab.optimism_trsplit_a[zetatest_grab_inds],
                self.psychometrics.grab.optimism_trsplit_b[zetatest_grab_inds])
            _pearsonr = sp_stats.pearsonr(
                self.psychometrics.grab.optimism_trsplit_a[zetatest_grab_inds],
                self.psychometrics.grab.optimism_trsplit_b[zetatest_grab_inds])
            _line_x = np.arange(ylims[0], ylims[1], 0.1)
            # _line_y = _linreg.slope * _line_x + _linreg.intercept
            ax_distrl_reliability.plot(_line_x, _line_x,
                                       linestyle='dotted',
                                       color='black')
            ax_distrl_reliability.set_title(
                f'optimism reliability (p={_pearsonr.pvalue:.4f})')

            ax_distrl_reliability.set_xlim(ylims)
            ax_distrl_reliability.set_ylim(ylims)
            ax_distrl_reliability.set_xlabel('optimism (first half of data)')
            ax_distrl_reliability.set_xlabel('optimism (second half of data)')

            # vmax = np.nanmax(np.abs(optimism_toplt))
            # vmin = -1 * vmax
            # ax_img_grab.imshow(
            #     optimism_toplt.reshape((self.sector.n_sectors,
            #                             self.sector.n_sectors)),
            #     vmin=vmin, vmax=vmax,
            #     cmap='coolwarm')

            # ax_distrl_grab.scatter(np.arange(optimism_masked.shape[0]),
            #                        optimism_masked[sort_args],
            #                        s=markersize,
            #                        c=sns.xkcd_rgb['black'])

            # for ind, sort_arg in enumerate(sort_args):
            #     _ind_in_full_grab = zetatest_mask_inds[sort_arg]

            #     # test for significance
            #     _sorted_null = np.sort(
            #         self.psychometrics.grab_null.optimism[_ind_in_full_grab])
            #     _rank = np.argmin(np.abs(
            #         _sorted_null - optimism_masked[sort_arg]))
            #     if _rank <= ind_sig_lower or _rank >= ind_sig_upper:
            #         ax_distrl_grab_signif.text(ind, 0, '*', ha='center',
            #                        va='bottom', fontsize=fontsize)

            #     # plot
            #     parts = ax_distrl_grab.violinplot(
            #         self.psychometrics.grab_null.optimism[
            #             _ind_in_full_grab], [ind], widths=1,
            #         showextrema=False)
            #     for pc in parts['bodies']:
            #         pc.set_facecolor(violin_color)

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

        # save figs
        fig_distrl.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{zetatest_mask=}_{ylims=}_optimism_distrl.pdf'))
        fig_distrl_reliability.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{zetatest_mask=}_{ylims=}_optimism_distrl_reliability.pdf'))
        fig_img.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{zetatest_mask=}_{ylims=}_optimism_spatial.pdf'))

        if plt_show is True:
            plt.show()

        return

    def add_null_dists(self, n_null=100, auto_run_methods=True,
                       t_pre=2, t_post=5):
        """
        Makes a 'null' distribution based on the alternate hypothesis
        of no spatial variability in task-related serotonin
        signaling, but instead spatial variability in 1) GRAB sensor
        expression; and 2) imaging-related noise.

        Reconstructs a null version of each sector's response as a
        modified version of the whole-frame response, scaled up/down

        - Must have run plt_stim_aligned_sectors first
        """
        print('creating null distributions')

        # check that sector and frame exist
        print('\trunning checks')

        if not hasattr(self, 'frame'):
            if auto_run_methods is False:
                print('Must have run self.add_frame() first!')
                return
            elif auto_run_methods is True:
                self.add_frame(t_pre=t_pre,
                               t_post=t_post)

        if not hasattr(self, 'sector'):
            if auto_run_methods is False:
                print('Must have run self.add_sectors() first!')
                return
            elif auto_run_methods is True:
                self.add_sectors(t_pre=t_pre,
                                 t_post=t_post)

        # construct the null distribution
        # Memory-optimized: store only parameters and ONE example trace per
        # sector (for plotting). Full null distributions are generated
        # on-the-fly when needed (e.g., in add_psychometrics).
        self.sector_null = SimpleNamespace()
        self.sector_null.t = self.sector.t
        self.sector_null.n_null = n_null

        # Store ONE example null trace per sector/condition (for plt_null_dist)
        self.sector_null.dff = np.empty(
            (self.sector.n_sectors, self.sector.n_sectors), dtype=dict)

        # Store parameters for on-the-fly generation
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

                # Compute scaling variables (stored for on-the-fly generation)
                _ampli_sector = np.max(np.mean(
                    self.sector.dff[n_x, n_y]['1'], axis=0))
                _ampli_scaling = _ampli_sector / ampli_frame

                # Vectorized baseline std computation
                _bl_data = self.sector.dff[n_x, n_y]['1'][
                    :, _ind_bl_start:_ind_bl_end]
                _bl_std = np.mean(np.std(_bl_data, axis=1))

                self.sector_null.params[n_x, n_y] = {
                    'ampli_scaling': _ampli_scaling,
                    'bl_std': _bl_std
                }

                # Generate ONE example null trace per condition (for plotting)
                self.sector_null.dff[n_x, n_y] = {}
                for tr_cond in ['0', '0.5', '1', '0.5_rew', '0.5_norew']:
                    _frame_data = self.frame.dff[tr_cond] 
                    n_trials, n_frames = _frame_data.shape
                    # Generate just ONE example (shape: 1, n_trials, n_frames)
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
                             zeta_type='2samp'):

        if not hasattr(self, 'sector'):
            self.add_sectors()

        print('adding zetatest for sectors...')
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
                    self.rec[:,
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

    def plt_null_dist(self,
                      figsize=(6, 3),
                      img_ds_factor=50,
                      img_alpha=0.5,
                      plt_dff={'x': {'gain': 0.6,
                                     'offset': 0.2},
                               'y': {'gain': 10,
                                     'offset': 0.2}}):
        fig = plt.figure(figsize=figsize)
        spec = gs.GridSpec(nrows=1, ncols=2,
                           figure=fig)

        ax_dff = fig.add_subplot(spec[0, 0])
        ax_dff.set_title('dff')
        ax_dff_null = fig.add_subplot(spec[0, 1])
        ax_dff_null.set_title('dff null')

        n_sectors = self.sector.n_sectors

        print('\tcreating max projection image...')
        _rec_max = np.max(self.rec[::img_ds_factor, :, :], axis=0)
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
                # **************************
                # plotting for this sector
                # **************************

                # make sector plot of stim-aligned traces
                # ----------
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

        fig.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + 'null_dists.pdf'))

        plt.show()

        return

    def plt_all_trial_resp(self, vmax_neurs=10):
        """
        Plots each trial's response separately
        """

        # GRAB responses
        # -------------
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

            # plot each trial
            for ind_tr, tr in enumerate(range(n_trials)):
                # setup axis
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

                # plot axis
                axs_grab[tr_type][ind_tr].imshow(
                    self.sector.stim_resp[tr_type][ind_tr, :, :],
                    vmax=vmax, vmin=-1*vmax, cmap='coolwarm')

            figs_grab[tr_type].suptitle(f'stim response, p_rew={tr_type}')
            figs_grab[tr_type].savefig(os.path.join(
                os.getcwd(), 'figs_mbl',
                f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
                + f'_sector_resp_by_trial_prew={tr_type}.pdf'))

        # neur responses
        # -------------
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

            # plot each trial
            for ind_tr, tr in enumerate(range(n_trials)):
                # setup axis
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

                # plot axis
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
                os.getcwd(), 'figs_mbl',
                f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
                + f'_neur_resp_by_trial_prew={tr_type}.pdf'))

        # show
        # -------
        plt.show()

        return

    def plt_frame(self,
                  figsize=(3.43, 2),
                  t_pre=2, t_post=5,
                  colors=sns.cubehelix_palette(
                      n_colors=3,
                      start=2, rot=0,
                      dark=0.1, light=0.6),
                  plot_type=None,
                  plt_show=True,
                  channel=None,
                  use_zscore=False):

        """
        Plots the average fluorescence across the whole fov,
        separated by trial-type (eg 0%, 50%, 100% rewarded trials).

        Delegates data extraction to self.add_frame().

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
        self.add_frame(t_pre=t_pre, t_post=t_post,
                       channel=channel, use_zscore=use_zscore)

        # build color and linestyle maps
        if hasattr(self.beh, 'tr_conds_outcome'):
            _cmap = sns.cubehelix_palette(
                as_cmap=True, start=2, rot=0, dark=0.1, light=0.6)
            _max = np.max(self.beh.tr_conds_outcome)
            colors = {str(p): _cmap(self.beh.tr_conds_outcome[i] / _max)
                      for i, p in enumerate(self.beh._stimparser.parsed_param)}
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
            stim_conds = list(self.beh.tr_conds)
        elif plot_type == 'rew_norew':
            stim_conds = ['0.5_rew', '0.5_norew']
        elif plot_type == 'prelick_noprelick':
            stim_conds = ['0.5_prelick', '0.5_noprelick']
        else:
            stim_conds = list(self.beh.tr_conds)

        # plot stim-aligned traces
        for stim_cond in stim_conds:
            if stim_cond not in self.frame.dff:
                continue
            _color = colors.get(stim_cond, 'grey')
            _ls = linestyles.get(stim_cond, 'solid')
            ax_trace.plot(self.frame.t,
                          np.mean(self.frame.dff[stim_cond], axis=0),
                          color=_color,
                          linestyle=_ls)
            ax_trace.fill_between(
                self.frame.t,
                np.mean(self.frame.dff[stim_cond], axis=0) -
                np.std(self.frame.dff[stim_cond], axis=0),
                np.mean(self.frame.dff[stim_cond], axis=0) +
                np.std(self.frame.dff[stim_cond], axis=0),
                facecolor=_color,
                alpha=0.2)

        # plot rew and stim onset markers
        ax_trace.axvline(x=0, color=sns.xkcd_rgb['dark grey'],
                         linewidth=1.5, alpha=0.8)
        ax_trace.axvline(x=2, color=sns.xkcd_rgb['bright blue'],
                         linewidth=1.5, alpha=0.8)

        ax_trace.set_xlabel('time (s)')
        ax_trace.set_ylabel('z-score' if use_zscore else 'df/f')

        _ch_suffix = f'ch={channel}' if channel is not None else \
            (f'ch={self.ch_img}' if hasattr(self, 'ch_img') else '')
        corr_suffix = ''
        if channel == 'grn' and hasattr(self, 'rec_t_grn') \
           and hasattr(self, 'corr_sig_method'):
            corr_suffix = f'_corr={self.corr_sig_method}'
        zscore_suffix = '_zscore' if use_zscore else ''
        fig_avg.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{plot_type=}_{t_pre=}_{t_post=}_'
            + f'{_ch_suffix}{corr_suffix}{zscore_suffix}_mean_trial_activity.pdf'))

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
                    plt_dff={'x': {'gain': 0.6,
                                   'offset': 0.2},
                             'y': {'gain': 10,
                                   'offset': 0.2}},
                    plt_prefix='',
                    plt_show=True,
                    colors=sns.cubehelix_palette(
                        n_colors=3,
                        start=2, rot=0,
                        dark=0.2, light=0.8)):
        """
        Divides the field of view into sectors, and plots a set of trial-types
        separately within each sector.

        plot_type controls the type of plot:
            if plot_type is None:
                stim_list = ['0', '0.5', '1']
            elif plot_type == 'rew_norew':
                stim_list = ['0.5_rew', '0.5_norew']
            elif plot_type == 'prelick_noprelick':
                stim_list = ['0.5_prelick', '0.5_noprelick']
            elif plot_type == 'rew':
                stim_list = ['0', '0.5_rew', '1']
        """
        print('creating trial-averaged signal (sectors)...')
        self.add_lickrates(t_prestim=t_pre,
                           t_poststim=t_post)

        self.sector = SimpleNamespace()

        self.sector.params = SimpleNamespace()
        self.sector.n_sectors = n_sectors
        self.sector.params.t_rew_pre = t_pre
        self.sector.params.t_rew_post = t_post
        self.sector.n_sectors = n_sectors

        self.sector.colors = {'0': colors[0],
                              '0.5': colors[1],
                              '1': colors[2],
                              '0.5_rew': colors[1],
                              '0.5_norew': colors[1],
                              '0.5_prelick': colors[1],
                              '0.5_noprelick': colors[1]}
        self.sector.linestyle = {'0': 'solid',
                                 '0.5': 'solid',
                                 '1': 'solid',
                                 '0.5_rew': 'solid',
                                 '0.5_norew': 'dashed',
                                 '0.5_prelick': 'solid',
                                 '0.5_noprelick': 'dashed'}

        fig_avg = plt.figure(figsize=figsize)
        fig_resid = plt.figure(figsize=figsize)
        spec_avg = gs.GridSpec(nrows=1, ncols=1,
                               figure=fig_avg)
        spec_resid = gs.GridSpec(nrows=1, ncols=1,
                                 figure=fig_resid)

        ax_img = fig_avg.add_subplot(spec_avg[0, 0])
        ax_img_resid = fig_resid.add_subplot(spec_resid[0, 0])

        print('\tcreating max projection image...')
        _rec_max = np.max(self.rec[::img_ds_factor, :, :], axis=0)
        _rec_max[:, ::int(_rec_max.shape[0]/n_sectors)] = 0
        _rec_max[::int(_rec_max.shape[0]/n_sectors), :] = 0

        ax_img.imshow(_rec_max,
                      extent=[0, n_sectors,
                              n_sectors, 0],
                      alpha=img_alpha)
        ax_img_resid.imshow(_rec_max,
                            extent=[0, n_sectors,
                                    n_sectors, 0],
                            alpha=img_alpha)
        ax_img_resid.set_title(f'residuals vs mean({resid_type}),'
                               + f' p(rew)={resid_tr_cond}')

        # setup stim-aligned traces
        n_frames_pre = int(t_pre * self.samp_rate)
        n_frames_post = int((2 + t_post) * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post

        self.sector.dff = np.empty((n_sectors, n_sectors), dtype=dict)
        self.sector.dff_resid = np.empty((n_sectors, n_sectors), dtype=dict)

        self.sector.t = np.linspace(
            -1*t_pre, 2+t_post, n_frames_tot)

        # store edges of sectors
        self.sector.x = SimpleNamespace()
        self.sector.x.lower = np.zeros((n_sectors, n_sectors))
        self.sector.x.upper = np.zeros((n_sectors, n_sectors))
        self.sector.y = SimpleNamespace()
        self.sector.y.lower = np.zeros((n_sectors, n_sectors))
        self.sector.y.upper = np.zeros((n_sectors, n_sectors))

        # compute whole-frame avg in case needed
        if not hasattr(self, 'frame'):
            self.plt_stim_aligned_avg(t_pre=t_pre, t_post=t_post, plt_show=False)

        # ---------------------
        print('\textracting aligned fluorescence traces...')
        _n_trace = 1
        for n_x in range(n_sectors):
            for n_y in range(n_sectors):
                print(f'\t\tsector {_n_trace}/{n_sectors**2}...      ')
                # calculate location of sector
                _ind_x_lower = int((n_x / n_sectors) * self.rec.shape[1])
                _ind_x_upper = int(((n_x+1) / n_sectors) * self.rec.shape[1])

                _ind_y_lower = int((n_y / n_sectors) * self.rec.shape[1])
                _ind_y_upper = int(((n_y+1) / n_sectors) * self.rec.shape[1])

                self.sector.x.lower[n_x, n_y] = _ind_x_lower
                self.sector.x.upper[n_x, n_y] = _ind_x_upper
                self.sector.y.lower[n_x, n_y] = _ind_y_lower
                self.sector.y.upper[n_x, n_y] = _ind_y_upper

                # setup structure
                self.sector.dff[n_x, n_y] = {}
                self.sector.dff_resid[n_x, n_y] = {}

                for tr_cond in ['0', '0.5', '1', '0.5_rew',
                                '0.5_norew', '0.5_prelick', '0.5_noprelick']:
                    self.sector.dff[n_x, n_y][tr_cond] = np.zeros((
                        self.beh.tr_inds[tr_cond].shape[0], n_frames_tot))
                    self.sector.dff_resid[n_x, n_y][tr_cond] = np.zeros_like(
                        self.sector.dff[n_x, n_y][tr_cond])

                # store stim-aligned traces
                self.sector.tr_counts = {'0': 0, '0.5': 0, '1': 0,
                                         '0.5_rew': 0, '0.5_norew': 0,
                                         '0.5_prelick': 0,
                                         '0.5_noprelick': 0}
                for trial in range(self.beh._stimrange.first,
                                   self.beh._stimrange.last):
                    print(f'\t\t\t{trial=}', end='\r')
                    _t_stim = self.beh.stim.t_start[trial]
                    _t_start = _t_stim - t_pre
                    _t_end = self.beh._data.get_event_var(
                        'totalRewardTimes')[trial] + t_post

                    _ind_t_start = np.argmin(np.abs(
                        self.rec_t - _t_start))
                    _ind_t_end = np.argmin(np.abs(
                        self.rec_t - _t_end)) + 2   # add frames to end

                    # extract fluorescence
                    _f = np.mean(np.mean(
                        self.rec[_ind_t_start:_ind_t_end,
                                 _ind_x_lower:_ind_x_upper,
                                 _ind_y_lower:_ind_y_upper], axis=1),
                                 axis=1)
                    _dff = calc_dff(_f, baseline_frames=n_frames_pre)
                    # _dff_shifted = (-1 * _dff * plt_dff['y']['gain']) \
                    #     + n_y - plt_dff['y']['offset'] + 1

                    if self.beh.stim.prob[trial] == 0:
                        self.sector.dff[n_x, n_y]['0'][
                            self.sector.tr_counts['0'], :] \
                            = _dff[0:n_frames_tot]
                        self.sector.tr_counts['0'] += 1
                    elif self.beh.stim.prob[trial] == 0.5:
                        self.sector.dff[n_x, n_y]['0.5'][
                            self.sector.tr_counts['0.5'], :] \
                            = _dff[0:n_frames_tot]
                        self.sector.tr_counts['0.5'] += 1
                    elif self.beh.stim.prob[trial] == 1:
                        self.sector.dff[n_x, n_y]['1'][
                            self.sector.tr_counts['1'], :] \
                            = _dff[0:n_frames_tot]
                        self.sector.tr_counts['1'] += 1

                    if self.beh.stim.prob[trial] == 0.5 \
                       and self.beh.rew.delivered[trial] == 1:
                        self.sector.dff[n_x, n_y]['0.5_rew'][
                            self.sector.tr_counts['0.5_rew'], :] \
                            = _dff[0:n_frames_tot]
                        self.sector.tr_counts['0.5_rew'] += 1
                    elif self.beh.stim.prob[trial] == 0.5 \
                         and self.beh.rew.delivered[trial] == 0:
                        self.sector.dff[n_x, n_y]['0.5_norew'][
                            self.sector.tr_counts['0.5_norew'], :] \
                            = _dff[0:n_frames_tot]
                        self.sector.tr_counts['0.5_norew'] += 1

                    if self.beh.stim.prob[trial] == 0.5 \
                       and self.beh.lick.antic_raw[trial] > 0:
                        self.sector.dff[n_x, n_y]['0.5_prelick'][
                            self.sector.tr_counts['0.5_prelick'], :] \
                            = _dff[0:n_frames_tot]
                        self.sector.tr_counts['0.5_prelick'] += 1
                    elif self.beh.stim.prob[trial] == 0.5 \
                         and self.beh.lick.antic_raw[trial] == 0:
                        self.sector.dff[n_x, n_y]['0.5_noprelick'][
                            self.sector.tr_counts['0.5_noprelick'], :] \
                            = _dff[0:n_frames_tot]
                        self.sector.tr_counts['0.5_noprelick'] += 1

                # store dff residuals
                # ----------
                for tr_cond in ['0', '0.5', '1']:
                    for tr in range(
                            self.sector.dff[n_x, n_y][tr_cond].shape[0]):
                        if resid_type == 'sector':
                            _dff_mean = np.mean(
                                self.sector.dff[n_x, n_y][tr_cond], axis=0)
                        elif resid_type == 'trial':
                            _dff_mean = self.frame.dff[tr_cond][tr, :]

                        self.sector.dff_resid[n_x, n_y][tr_cond][tr, :] \
                            = self.sector.dff[n_x, n_y][tr_cond][tr, :] \
                            - _dff_mean

                # **************************
                # plotting for this sector
                # **************************

                # make sector plot of stim-aligned traces
                # ----------
                _t_sector = ((self.sector.t / self.sector.t[-1])
                             * plt_dff['x']['gain'])
                _t_sector = _t_sector + n_x + plt_dff['x']['offset']

                if plot_type is None:
                    stim_list = ['0', '0.5', '1']
                elif plot_type == 'rew_norew':
                    stim_list = ['0.5_rew', '0.5_norew']
                elif plot_type == 'prelick_noprelick':
                    stim_list = ['0.5_prelick', '0.5_noprelick']
                elif plot_type == 'rew':
                    stim_list = ['0', '0.5_rew', '1']

                for stim_cond in stim_list:
                    _dff_mean = np.mean(self.sector.dff[n_x, n_y][stim_cond],
                                        axis=0)
                    _dff_shifted = (-1 * _dff_mean * plt_dff['y']['gain']) \
                        + n_y - plt_dff['y']['offset'] + 1

                    ax_img.plot(
                        _t_sector, _dff_shifted,
                        color=self.sector.colors[stim_cond],
                        linestyle=self.sector.linestyle[stim_cond])

                # make residuals plot
                # ------------------------
                _n_tr_resid = self.sector.dff_resid[n_x, n_y][
                    resid_tr_cond].shape[0]
                for trial in range(_n_tr_resid):
                    _dff = self.sector.dff_resid[n_x, n_y][resid_tr_cond][trial, :]
                    if resid_smooth is True:
                        _dff = sp.signal.savgol_filter(_dff, resid_smooth_windlen,
                                                       resid_smooth_polyorder)
                    _dff_shifted = (-1 * _dff * plt_dff['y']['gain']) \
                        + n_y - plt_dff['y']['offset'] + 1

                    ax_img_resid.plot(
                        _t_sector, _dff_shifted,
                        color=sns.color_palette('rocket', _n_tr_resid)[trial],
                        alpha=resid_alpha,
                        linestyle=self.sector.linestyle[stim_cond])

                # plot rew and stim lines
                # ------------
                _t_stim = (0 * plt_dff['x']['gain']) \
                    + n_x + plt_dff['x']['offset']
                _t_rew = ((2 / self.sector.t[-1])
                          * plt_dff['x']['gain']) \
                    + n_x + plt_dff['x']['offset']
                _dff_zero = n_y - plt_dff['y']['offset'] + 1

                for ax in [ax_img, ax_img_resid]:
                    ax.plot([_t_stim, _t_stim],
                            [n_y + 0.2, n_y + 0.8],
                            color=sns.xkcd_rgb['dark grey'],
                            linewidth=1,
                            linestyle='dashed')
                    ax.plot([_t_rew, _t_rew],
                            [n_y + 0.2, n_y + 0.8],
                            color=sns.xkcd_rgb['bright blue'],
                            linewidth=1,
                            linestyle='dashed')

                ax_img_resid.plot([_t_stim, _t_rew],
                                  [_dff_zero, _dff_zero],
                                  color='k',
                                  linewidth=1,
                                  linestyle='dashed')

                # update _n_trace
                _n_trace += 1

        # compute supplemental plots on data in self.sector.dff_resid
        # ---------------
        n_trs = self.sector.dff_resid[0, 0][resid_tr_cond].shape[0]
        n_rows = np.ceil(np.sqrt(n_trs)).astype(int)

        fig_norm_sec_sig = plt.figure(figsize=(3.43, 1.5))
        spec_norm_sec_sig = gs.GridSpec(nrows=1, ncols=3,
                                        figure=fig_norm_sec_sig)
        ax_norm_sec_sig_prew = {}
        ax_norm_sec_sig_prew['0'] = fig_norm_sec_sig.add_subplot(
            spec_norm_sec_sig[0, 0])
        ax_norm_sec_sig_prew['0.5'] = fig_norm_sec_sig.add_subplot(
            spec_norm_sec_sig[0, 1])
        ax_norm_sec_sig_prew['1'] = fig_norm_sec_sig.add_subplot(
            spec_norm_sec_sig[0, 2])

        fig_resid_corr = plt.figure(figsize=figsize)
        spec_resid_corr = gs.GridSpec(nrows=1, ncols=1,
                                      figure=fig_resid_corr)
        ax_resid_corr = fig_resid_corr.add_subplot(spec_resid_corr[0, 0])

        fig_resid_corr_spatial = plt.figure(figsize=figsize)
        spec_resid_corr_spatial = gs.GridSpec(
            nrows=self.sector.n_sectors,
            ncols=self.sector.n_sectors,
            figure=fig_resid_corr_spatial)

        fig_resid_tr_sep = plt.figure(figsize=figsize)
        spec_resid_tr_sep = gs.GridSpec(nrows=n_rows,
                                        ncols=n_rows,
                                        figure=fig_resid_tr_sep)

        # compute and plot trial-averaged, normalized signals for each sector
        for sec in range(self.sector.n_sectors):
            sec_x, sec_y = np.divmod(sec, self.sector.n_sectors)

            for tr_type in ['0', '0.5', '1']:
                _n_tr = self.sector.dff[0, 0][tr_type].shape[0]
                _sec_max = np.max(self.sector.dff[sec_x, sec_y][tr_type],
                                  axis=1)
                _sec_sig = np.array([self.sector.dff[sec_x, sec_y][tr_type][
                    tr, :] / _sec_max[tr] for tr in range(_n_tr)])
                _sec_sig_travg = np.mean(_sec_sig, axis=0)
                ax_norm_sec_sig_prew[tr_type].plot(
                    self.sector.t,
                    _sec_sig_travg,
                    color=self.sector.colors[tr_type],
                    alpha=0.8)

        # compute and plot cross-correlations
        self.sector.resid_corr_stat = np.zeros(
            (self.sector.n_sectors**2, self.sector.n_sectors**2))
        self.sector.resid_corr_pval = np.zeros(
            (self.sector.n_sectors**2, self.sector.n_sectors**2))

        for sec_1 in range(self.sector.n_sectors**2):
            for sec_2 in range(self.sector.n_sectors**2):
                sec_1_x, sec_1_y = np.divmod(sec_1, self.sector.n_sectors)
                sec_2_x, sec_2_y = np.divmod(sec_2, self.sector.n_sectors)

                _sec_1_sig = np.mean(
                    self.sector.dff_resid[sec_1_x, sec_1_y][resid_tr_cond],
                    axis=1)
                _sec_2_sig = np.mean(
                    self.sector.dff_resid[sec_2_x, sec_2_y][resid_tr_cond],
                    axis=1)

                _corr = sp.stats.pearsonr(_sec_1_sig, _sec_2_sig)
                self.sector.resid_corr_stat[sec_1, sec_2] = _corr.statistic
                self.sector.resid_corr_pval[sec_1, sec_2] = _corr.pvalue

        ax_resid_corr.imshow(self.sector.resid_corr_stat)

        ax_residcorr_spatial = []
        ax_count = 0
        for sec_1 in range(self.sector.n_sectors):
            for sec_2 in range(self.sector.n_sectors):
                ax_residcorr_spatial.append(
                    fig_resid_corr_spatial.add_subplot(
                        spec_resid_corr_spatial[sec_1, sec_2]))
                ax_residcorr_spatial[ax_count].imshow(
                    self.sector.resid_corr_stat[ax_count, :].reshape(
                        self.sector.n_sectors, self.sector.n_sectors),
                    vmax=1, vmin=-1, cmap='coolwarm')
                ax_residcorr_spatial[ax_count].set_xticks([])
                ax_residcorr_spatial[ax_count].set_yticks([])

                ax_count += 1

        fig_resid_corr_spatial.suptitle('residual corrs (spatial), '
                                        + f' {resid_tr_cond=},'
                                        + f' residuals vs mean({resid_type})')

        # compute and plot per-trial residual distributions
        ax = []
        for tr in range(n_trs):
            # compile response of all sectors for this trial
            _sector_dff_resid = []
            for sec_1 in range(self.sector.n_sectors):
                for sec_2 in range(self.sector.n_sectors):
                    _sector_dff_resid.append(np.mean(
                        self.sector.dff_resid[sec_1, sec_2][
                            resid_tr_cond][tr, :]))

            # set up axis
            _tr_x, _tr_y = np.divmod(tr, n_rows)
            if tr == 0:
                ax.append(fig_resid_tr_sep.add_subplot(
                    spec_resid_tr_sep[_tr_x, _tr_y]))
            elif tr > 0:
                ax.append(fig_resid_tr_sep.add_subplot(
                    spec_resid_tr_sep[_tr_x, _tr_y],
                          sharex=ax[0], sharey=ax[0]))

            # plot per-trial distributions
            ax[tr].hist(_sector_dff_resid, bins=resid_trial_nbins,
                        histtype='step',
                        density=True, color=sns.xkcd_rgb['dark grey'])
            ax[tr].axvline(0, color='k',
                           linewidth=1,
                           linestyle='dashed')
            ax[tr].set_xlabel('dff resid.')
            ax[tr].set_ylabel('pdf')
            ax[tr].set_title(f'trial={tr}')
        fig_resid_tr_sep.suptitle(f'residuals vs mean{resid_type} corrs (spatial), '
                                        + f' {resid_tr_cond=},'
                                        + f' residuals vs mean({resid_type})')

        # save all figs
        # -------------
        fig_avg.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            'sectors_'
            + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{plt_prefix}_'
            + f'{n_sectors=}_{plot_type=}_{t_pre=}_{t_post=}_'
            + f'ch={self.ch_img}.pdf'))

        fig_norm_sec_sig.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            'sectors_normed_'
            + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{plt_prefix}_'
            + f'{n_sectors=}_{plot_type=}_{t_pre=}_{t_post=}_'
            + f'ch={self.ch_img}.pdf'))

        fig_resid.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            'sectors_resid_'
            + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{plt_prefix}_'
            + f'{n_sectors=}_{plot_type=}_{t_pre=}_{t_post=}_'
            + f'{resid_tr_cond=}_{resid_type=}_'
            + f'ch={self.ch_img}.pdf'))

        fig_resid_corr.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            'sectors_resid_corr_'
            + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{plt_prefix}_'
            + f'{n_sectors=}_{plot_type=}_{t_pre=}_{t_post=}_'
            + f'{resid_tr_cond=}_{resid_type=}_'
            + f'ch={self.ch_img}.pdf'))

        fig_resid_corr_spatial.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            'sectors_resid_corr_spatial_'
            + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{plt_prefix}_'
            + f'{n_sectors=}_{plot_type=}_{t_pre=}_{t_post=}_'
            + f'{resid_tr_cond=}_{resid_type=}_'
            + f'ch={self.ch_img}.pdf'))

        fig_resid_tr_sep.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            'sectors_resid_trial_sep_'
            + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{plt_prefix}_'
            + f'{n_sectors=}_{plot_type=}_{t_pre=}_{t_post=}_'
            + f'{resid_tr_cond=}_{resid_type=}_'
            + f'ch={self.ch_img}.pdf'))

        # show all plots
        if plt_show is True:
            plt.show()
        elif plt_show is False:
            for _fig in [fig_avg, fig_norm_sec_sig, fig_resid,
                         fig_resid_corr, fig_resid_corr_spatial,
                         fig_resid_tr_sep]:
                plt.close(_fig)

        return

    def plt_neurs(self, t_pre=1, t_post=2,
                  figsize=(3.43, 3.43), zscore=False,
                  plt_equal=True,
                  sort_by='trial_type',
                  sort_tr_type='max',
                  cmap=sns.diverging_palette(
                      220, 20, as_cmap=True)):
        """
        Plots trial-averaged neural activity (dff) for each cell as a heatmap,
        one column per trial condition.

        Parameters
        ----------
        sort_tr_type : str
            Trial condition key used to sort neurons when sort_by='trial_type'.
            Can be any key present in self.beh.tr_conds, or one of two
            special strings:
                'max' : automatically uses the tr_cond with the highest outcome
                        value in self.beh.tr_conds_outcome.
                'min' : automatically uses the tr_cond with the lowest outcome
                        value in self.beh.tr_conds_outcome.
        """
        # plot neuron masks
        self.plt_neur_masks()

        self.add_lickrates(t_prestim=t_pre,
                           t_poststim=t_post)

        # setup structure
        # ---------------
        neurs = np.where(self.neur.iscell[:, 0] == 1)[0]
        n_neurs = neurs.shape[0]

        n_frames_pre = int(t_pre * self.samp_rate)
        n_frames_post = int((2 + t_post) * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post

        # resolve 'max'/'min' sort_tr_type to the corresponding tr_cond
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
        # ----------------
        for ind, neur in enumerate(neurs):
            # extract x/y locations
            self.neur.x[ind] = self.neur.stat[neur]['med'][0]
            self.neur.y[ind] = self.neur.stat[neur]['med'][1]
            # correct for image crop
            self.neur.x[ind] -= self.ops.n_px_remove_sides
            self.neur.y[ind] -= self.ops.n_px_remove_sides

            # setup per-neuron dff_aln structure
            # -------------
            self.neur.dff_aln[ind] = {}
            for _tr_cond in _tr_conds_all:
                self.neur.dff_aln[ind][_tr_cond] = np.zeros((
                    self.beh.tr_inds[_tr_cond].shape[0], n_frames_tot))

            _tr_counts = {k: 0 for k in _tr_conds_all}

            # precompute zscore for this neuron (avoids repeated full-array zscore)
            if zscore is True:
                _f_full = sp.stats.zscore(self.neur.f[neur, :])
            else:
                _f_full = self.neur.f[neur, :]

            # extract fluorescence for each trial
            # -------------
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
        # ------------
        _ind_stim_start = np.argmin(np.abs(self.neur.t_aln))
        _ind_stim_end = np.argmin(np.abs(self.neur.t_aln - 2))
        _ind_rew_end = np.argmin(np.abs(self.neur.t_aln - (2 + t_post)))

        if sort_by == 'trial_type':
            # sort by peak activation for the chosen trial type (stim window)
            _sort_metric = np.max(
                self.neur.dff_aln_mean[sort_tr_type][
                    :, _ind_stim_start:_ind_stim_end],
                axis=1)
            sort_inds_corr = np.argsort(_sort_metric)

        elif sort_by == 'selectivity':
            # sort by absolute difference between highest and lowest outcome
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
            # sort by peak response in the post-reward window on highest-outcome trials
            _cond_max = _tr_conds_plot[-1]
            _sort_metric = np.max(
                self.neur.dff_aln_mean[_cond_max][:, _ind_stim_end:_ind_rew_end],
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

        # plot figure - one column per primary tr_cond
        # -----------------
        n_conds = len(_tr_conds_plot)
        fig = plt.figure(figsize=figsize)
        spec = gs.GridSpec(nrows=1, ncols=n_conds, figure=fig)
        axes = [fig.add_subplot(spec[0, 0])]
        for _i in range(1, n_conds):
            axes.append(fig.add_subplot(spec[0, _i], sharey=axes[0]))

        _ind_zero = np.argmin(np.abs(self.neur.t_aln))
        _ind_rew = np.argmin(np.abs(self.neur.t_aln - 2))

        for _ax, _tr_cond in zip(axes, _tr_conds_plot):
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
            _ax.set_title(f'{self.beh._stimparser.parse_by}={_tr_cond}')
        axes[0].set_ylabel('neuron')
        axes[n_conds // 2].set_xlabel('time from stim (s)')

        fig.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{t_pre=}_{t_post=}_{zscore=}_{plt_equal=}_'
            + 'neur_trial_activity.pdf'))

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
            classifier_kwargs={'class_weight': 'balanced',
                               'loss': 'hinge',
                               'penalty': 'l2',
                               'alpha': 1,
                               'max_iter': 100000},
            scaler=sklearn.preprocessing.StandardScaler,
            scaler_kwargs={},
            classifier_scoring='balanced_accuracy'):

        """
        Attempts to decode trial type from a profile of GRAB sector responses
        for each trial.

        Must have run self.plt_stim_aligned_sectors() first.
        """

        # setup decoder
        # ---------------
        ind_t_start = np.argmin(np.abs(self.sector.t - t_start))
        ind_t_end = np.argmin(np.abs(self.sector.t - t_end))
        # ind_t_bl_start = np.argmin(np.abs(self.sector.t - t_bl_start))
        # ind_t_bl_end = np.argmin(np.abs(self.sector.t - t_bl_end))

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

        # classification: full dataset
        # ---------------
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

        # classification: changing neuron numbers
        # -------------------
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

                # run classification
                self.sector.decoding.cvobj[str(n_sec_decoder)] \
                    = cross_validate(
                        self.clf, self.sector.decoding._data,
                        y=self.sector.decoding._labels,
                        cv=fold_validation,
                        n_jobs=n_jobs, scoring=classifier_scoring,
                        return_estimator=True)

                # store mean value across fold validation
                _temp_perf = np.mean(self.sector.decoding.cvobj[
                    str(n_sec_decoder)]['test_score'])
                self.sector.decoding.perf_means.vals[ind_sec_decoder] \
                    += _temp_perf

            # after all iterations, take mean
            self.sector.decoding.perf_means.vals[ind_sec_decoder] \
                /= n_resamples

        # classification: whole-frame (1p), no variance
        # ---------------
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

        # make plots
        # --------------
        # perf plot
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

        # weightmap plot
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
                ax_weightmap[tr_cond].set_title(f'decoding weights: {tr_cond=}')
                ax_weightmap[tr_cond].set_xticks([])
                ax_weightmap[tr_cond].set_yticks([])

        fig_perf.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            'sectors_decoding_perf_'
            + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{comp=}_{t_start=}_{t_end=}_'
            + f'ch={self.ch_img}.pdf'))

        fig_weightmap.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            'sectors_decoding_weightmap_'
            + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{comp=}_{t_start=}_{t_end=}_'
            + f'ch={self.ch_img}.pdf'))

        # save the data
        plt.show()

        return

    def plt_psychometric_stimscaling(self, figsize=(2, 2),
                                     marker_size=60,
                                     plt_xlim=[-0.5, 2.5],
                                     plt_show=True):
        """
        Generates a psychometric curve of grab response vs stim
        probability, independently plotting each roi.

        must have ran plt_stim_aligned_sectors() to generate
        the stim-aligned dff traces in self.stim_plt.dff[x, y][stim_cond].
        """

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

            # plot psychometric
            # ---------------
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

            # plot correlation
            # ---------------
            fig_corr = plt.figure(figsize=figsize)
            ax_corr = fig_corr.add_subplot(1, 1, 1)

            ax_corr.scatter(self.psychometrics.stim_resp['1'],
                            self.psychometrics.rew_resp['1'],
                            facecolors='none',
                            edgecolors=colors[2], s=10)
            ax_corr.set_xlabel('GRAB 5-HT stim resp. (100%)')
            ax_corr.set_ylabel('GRAB 5-HT rew resp. (100%)')

            fig.savefig(os.path.join(
                os.getcwd(), 'figs_mbl',
                f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
                + f'ch={self.ch_img}_psychometric_curve.pdf'))
            fig_corr.savefig(os.path.join(
                os.getcwd(), 'figs_mbl',
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

        # setup reward-aligned traces
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
                # calculate location of sector
                _ind_x_lower = int((n_x / n_sectors) * self.rec.shape[1])
                _ind_x_upper = int(((n_x+1) / n_sectors) * self.rec.shape[1])

                _ind_y_lower = int((n_y / n_sectors) * self.rec.shape[1])
                _ind_y_upper = int(((n_y+1) / n_sectors) * self.rec.shape[1])

                # plot trace
                _trace = np.mean(np.mean(
                    self.rec[:,
                             _ind_x_lower:_ind_x_upper,
                             _ind_y_lower:_ind_y_upper], axis=1), axis=1)

                ax_traces.plot(self.rec_t,
                               sp.stats.zscore(_trace)*scaling_factor_trace
                               + _n_trace,
                               color=sns.xkcd_rgb['ocean green'],
                               linewidth=0.5, alpha=0.8)

                # plot reward-aligned trace
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

                # note here that we must invert dff_rewaligned_mean
                # because plt() traces are  plotted in the 'negative' direction
                # on top of imshow() images for some reason
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
        _count, t_bin_edges = np.histogram(itis, bins=n_time_divs)

        for ind_timediv in range(n_time_divs):
            _trials_in_timediv = np.logical_and(
                itis > t_bin_edges[ind_timediv],
                itis < t_bin_edges[ind_timediv+1])
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

        ax.set_xlabel('time from rew (s)')
        ax.set_ylabel('df/f')

        fig.savefig(f'{savefig_prefix}_dff_by_{n_time_divs}'
                    + f'iti_sector{ind_sector}.pdf')

        plt.show()

    def plt_dffs_including_iti(self, ind_sector=10, n_time_divs=5,
                               t_rew_pre=2,
                               sns_palette='mako',
                               figsize=(3.43, 3.43), savefig_prefix='grab'):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        palette = sns.color_palette(sns_palette, n_time_divs)

        self.dff_binned_time = np.empty(n_time_divs, dtype=np.ndarray)
        itis = np.diff(self.beh.rew.t)

        _count, t_bin_edges = np.histogram(itis, bins=n_time_divs)

        for ind_timediv in range(n_time_divs):
            _trials_in_timediv = np.logical_and(
                itis > t_bin_edges[ind_timediv],
                itis < t_bin_edges[ind_timediv+1])

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
            label=f'{t_bin_edges[ind_timediv]:.1f}-{t_bin_edges[ind_timediv+1]:.1f}'
            ax.plot(self.neur.t,
                    self.dff_binned_time[ind_timediv],
                    color=palette[ind_timediv], linewidth=0.8,
                    label=label)
        ax.axvline(x=0, color=sns.xkcd_rgb['bright blue'],
                   linestyle='dashed',
                   linewidth=0.8)
        ax.legend()

        ax.set_xlabel('time from rew (s)')
        ax.set_ylabel('df/f')

        fig.savefig(f'{savefig_prefix}_dff_by_{n_time_divs}iti_sector{ind_sector}.pdf')

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

        ax.set_xlabel('time from rew (s)')
        ax.set_ylabel('df/f')

        fig.savefig(f'{savefig_prefix=}_dff_{sector=}.pdf')

        plt.show()

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

        ax.set_xlabel('time from rew (s)')
        ax.set_ylabel('df/f')

        fig.savefig(f'{savefig_prefix}_dff_all.pdf')

        plt.show()

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
        ax.set_xlabel('reward number')
        ax.set_ylabel('integral df/f')

        fig.savefig(f'{savefig_prefix}_total_response.pdf')

        plt.show()

        return

    def plt_lick_resp(self, t_pre=3, t_post=3,
                      figsize=(3.43, 3.43),
                      savefig_prefix='grab'):
        n_frames_pre = int(t_pre * self.samp_rate)
        n_frames_post = int(t_post * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post

        t_lickaligned = np.linspace(
            -1*t_pre, t_post, n_frames_tot)

        dff_lick = np.zeros(n_frames_tot)
        count_dffs = 0
        for lick in self.beh.t_licks:
            _closest_rew = np.min(np.abs(
                self.beh.rew.t - lick))
            if _closest_rew > np.max([t_pre, t_post]):
                ind_lick = np.argmin(np.abs(
                    self.rec_t - lick))
                f = np.mean(np.mean(
                    self.rec[ind_lick-n_frames_pre:
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
        ax.set_xlabel('time from lick (s)')
        ax.set_ylabel('dff')
        ax.axvline(x=0, color=sns.xkcd_rgb['black'],
                   linestyle='dashed',
                   linewidth=0.8)

        fig.savefig(f'{savefig_prefix}_lick_resp_{t_pre=}_{t_post=}.pdf')
        plt.show()


    def plt_neur_masks(self):
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
            os.getcwd(), 'figs_mbl',
            'neur_masks_'
            + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}'
            + '.pdf'))

        plt.show()


class TwoPRec_DualColour(TwoPRec):
    """
    Dual-colour version of TwoPRec that loads two imaging files
    (red and green channels) and provides methods to analyze either channel.

    Inherits all functionality from TwoPRec but adds:
    - self.rec_red: Red channel imaging data
    - self.rec_grn: Green channel imaging data
    - channel='grn' kwarg on methods to select which channel to analyze

    Parameters
    ----------
    Same as TwoPRec, with additions:
    ch_img_red : int
        Channel number for red imaging file (default: 1)
    ch_img_grn : int
        Channel number for green imaging file (default: 2)
    fname_img_red : str, optional
        Filename for red channel tiff
    fname_img_grn : str, optional
        Filename for green channel tiff
    """

    def __init__(self,
                 enclosing_folder=None,
                 folder_beh=None,
                 folder_img=None,
                 fname_img_red=None,
                 fname_img_grn=None,
                 dset_obj=None,
                 dset_ind=None,
                 ch_img_red=1,
                 ch_img_grn=2,
                 trial_end=None,
                 rec_type='trig_rew',
                 n_px_remove_sides=10):
        """
        Loads a dual-colour 2p recording (two tiffs) and associated
        behavioral folder.

        See TwoPRec docstring for full parameter descriptions.
        This class loads two imaging channels:
        - Red channel (typically Ch1) -> self.rec_red
        - Green channel (typically Ch2) -> self.rec_grn
        """

        # setup names of folders, files, and ops
        # --------------
        self._init_folders(enclosing_folder, folder_beh, folder_img,
                           dset_obj, dset_ind)
        self.ch_img_red = ch_img_red
        self.ch_img_grn = ch_img_grn
        # Keep ch_img for backwards compatibility (defaults to green)
        self.ch_img = ch_img_grn
        self._init_ops(n_px_remove_sides, rec_type)

        # get filenames of images for both channels
        list_img = os.listdir(self.folder.img)

        # Red channel
        if fname_img_red is not None:
            self.fname_img_red = fname_img_red
        elif dset_obj is not None:
            # When loading from a dataset, always expect the standard names
            self.fname_img_red = f'compiled_Ch{ch_img_red}.tif'
        else:
            for _fname in list_img:
                if f'Ch{ch_img_red}.tif' in _fname and 'compiled' in _fname:
                    self.fname_img_red = _fname
                    break

        # Green channel
        if fname_img_grn is not None:
            self.fname_img_grn = fname_img_grn
        elif dset_obj is not None:
            self.fname_img_grn = f'compiled_Ch{ch_img_grn}.tif'
        else:
            for _fname in list_img:
                if f'Ch{ch_img_grn}.tif' in _fname and 'compiled' in _fname:
                    self.fname_img_grn = _fname
                    break

        # Keep fname_img for backwards compatibility (defaults to green)
        self.fname_img = self.fname_img_grn

        # load behavioral data (DualColour always uses parse_stims=False)
        # -------------
        self._init_behavior(parse_stims=False)

        # load imaging data - BOTH CHANNELS
        # ------------
        print('loading imaging...')
        for _file in os.listdir(self.folder.img):
            if _file.endswith('BACKUP.xml'):
                try:
                    xmlobj = XMLParser(os.path.join(self.folder.img, _file))
                    sampling_rate = xmlobj.get_framerate()
                    print(f'\tframerate is {sampling_rate:.2f}Hz')
                except:
                    print('could not parse framerate from BACKUP.xml')
                    pass
        self.samp_rate = sampling_rate

        # Load RED channel (keep raw memmap, crop lazily via _get_rec)
        # Storing the raw memmap avoids creating a non-contiguous view that
        # forces full-array RAM copies when batches are read.
        print('\tloading red channel tiff...')
        self._rec_red_raw = tifffile.memmap(os.path.join(
            self.folder.img, self.fname_img_red))

        # Load GREEN channel (keep raw memmap, crop lazily via _get_rec)
        print('\tloading green channel tiff...')
        self._rec_grn_raw = tifffile.memmap(os.path.join(
            self.folder.img, self.fname_img_grn))

        # Store crop bounds for lazy application
        self._crop_px = n_px_remove_sides

        # Create cropped views for backward compatibility (used for .shape etc)
        # These are views, not copies, until data is actually accessed
        c = n_px_remove_sides
        self.rec_red = self._rec_red_raw[:, c:-c, c:-c] if c > 0 else self._rec_red_raw
        self.rec_grn = self._rec_grn_raw[:, c:-c, c:-c] if c > 0 else self._rec_grn_raw

        # Keep self.rec for backwards compatibility (defaults to green)
        self.rec = self.rec_grn

        # try to load suite2p output if available
        self._init_suite2p()

        # align behavior and imaging data
        # ----------------------
        self._init_timestamps(self.rec_grn.shape[0], rec_type)

        # note the first and last stimulus/rew within recording bounds
        # ---------------
        self.trial_end = trial_end
        self._init_stim_rew_range(trial_end=trial_end)

        # compute trial indices for each trtype
        # -------------------
        self.beh.tr_inds = {}
        if self.beh._stimparser.parse_by == 'stimulusOrientation':
            self.beh.tr_conds = ['0', '0.5', '1', '0.5_rew', '0.5_norew']

            self.beh.tr_inds['0'] = np.where(self.beh.stim.prob == 0)[0]
            self.beh.tr_inds['0.5'] = np.where(self.beh.stim.prob == 0.5)[0]
            self.beh.tr_inds['1'] = np.where(self.beh.stim.prob == 1)[0]
            self.beh.tr_inds['0.5_rew'] = np.where(np.logical_and(
                self.beh.stim.prob == 0.5, self.beh.rew.delivered == True))[0]
            self.beh.tr_inds['0.5_norew'] = np.where(np.logical_and(
                self.beh.stim.prob == 0.5, self.beh.rew.delivered == False))[0]

            # remove trinds outside of correct range
            for tr_cond in ['0', '0.5', '1', '0.5_rew', '0.5_norew']:
                self.beh.tr_inds[tr_cond] = np.delete(
                    self.beh.tr_inds[tr_cond],
                    ~np.isin(self.beh.tr_inds[tr_cond],
                             np.arange(
                                 self.beh._stimrange.first,
                                 self.beh._stimrange.last+1)))
            # add lickrates
            # ----------------
            self.add_lickrates()

            self.beh.tr_inds['0.5_prelick'] = np.where(np.logical_and(
                self.beh.stim.prob == 0.5, self.beh.lick.antic_raw > 0))[0]
            self.beh.tr_inds['0.5_noprelick'] = np.where(np.logical_and(
                self.beh.stim.prob == 0.5, self.beh.lick.antic_raw == 0))[0]
            for tr_cond in ['0.5_prelick', '0.5_noprelick']:
                self.beh.tr_inds[tr_cond] = np.delete(
                    self.beh.tr_inds[tr_cond],
                    ~np.isin(self.beh.tr_inds[tr_cond],
                             np.arange(
                                 self.beh._stimrange.first,
                                 self.beh._stimrange.last+1)))
        else:
            self.beh.tr_conds = self.beh._stimparser.parsed_param.astype(str)
            self.beh.tr_conds_outcome = (self.beh._stimparser.prob
                                         * self.beh._stimparser.size)

            for param_val in self.beh._stimparser.parsed_param:
                self.beh.tr_inds[str(param_val)] = np.where(
                    self.beh._stimparser._all_parsed_param == param_val)[0]

            # remove trinds outside of correct range
            for tr_cond in [str(p) for p in self.beh._stimparser.parsed_param]:
                self.beh.tr_inds[tr_cond] = np.delete(
                    self.beh.tr_inds[tr_cond],
                    ~np.isin(self.beh.tr_inds[tr_cond],
                             np.arange(
                                 self.beh._stimrange.first,
                                 self.beh._stimrange.last+1)))
            # add lickrates
            # ----------------
            self.add_lickrates()

        # build _trial_cond_map: maps each trial index -> list of condition keys
        self._trial_cond_map = {}
        for _cond, _inds in self.beh.tr_inds.items():
            for _idx in _inds:
                self._trial_cond_map.setdefault(int(_idx), []).append(_cond)
        return

    def _get_rec(self, channel='grn'):
        """
        Helper method to get the appropriate recording based on channel.

        Returns the cropped view for backward compatibility.

        Parameters
        ----------
        channel : str
            'red' or 'grn' (default: 'grn')

        Returns
        -------
        np.ndarray
            The recording array for the specified channel (cropped view)
        """
        if channel == 'red':
            return self.rec_red
        elif channel == 'grn':
            return self.rec_grn
        else:
            raise ValueError(f"channel must be 'red' or 'grn', got '{channel}'")

    def _get_rec_raw(self, channel='grn'):
        """
        Helper method to get the raw (uncropped) memmap for efficient batch I/O.

        Use this for methods that read large batches to avoid non-contiguous
        memory access patterns. Apply cropping after reading each batch.

        Parameters
        ----------
        channel : str
            'red' or 'grn' (default: 'grn')

        Returns
        -------
        tuple
            (raw_memmap, crop_px) where crop_px is the number of pixels to
            remove from each edge. Apply as: data[:, c:-c, c:-c] if c > 0.
        """
        if channel == 'red':
            return self._rec_red_raw, self._crop_px
        elif channel == 'grn':
            # Use original green if available (for correct_signal re-runs)
            if hasattr(self, '_rec_grn_original_raw'):
                return self._rec_grn_original_raw, self._crop_px
            return self._rec_grn_raw, self._crop_px
        else:
            raise ValueError(f"channel must be 'red' or 'grn', got '{channel}'")

    def _get_rec_t(self, channel='grn'):
        """
        Helper method to get the appropriate timestamps based on channel.

        If correct_signal() was called with replace_grn=True and time slicing,
        the green channel may have different timestamps than the full recording.

        Parameters
        ----------
        channel : str
            'red' or 'grn' (default: 'grn')

        Returns
        -------
        np.ndarray
            The timestamp array for the specified channel
        """
        if channel == 'red':
            return self.rec_t
        elif channel == 'grn':
            # Use channel-specific timestamps if available (from correct_signal)
            if hasattr(self, 'rec_t_grn'):
                return self.rec_t_grn
            return self.rec_t
        else:
            raise ValueError(f"channel must be 'red' or 'grn', got '{channel}'")

    # add_frame and plt_frame are inherited from TwoPRec which now supports
    # channel, use_zscore, _get_rec/_get_rec_t, and _trial_cond_map.

    def add_sectors(self,
                    n_sectors=10,
                    t_pre=2, t_post=5,
                    n_null=500,
                    resid_type='sector',
                    resid_corr_tr_cond='1',
                    channel='grn',
                    use_zscore=False,
                    compute_null=False,
                    compute_resid_corr=True,
                    colors=sns.cubehelix_palette(
                        n_colors=3,
                        start=2, rot=0,
                        dark=0.2, light=0.8)):
        """
        Divides the field of view into sectors and extracts trial-type
        responses for each sector.

        Parameters
        ----------
        n_sectors : int
            Number of sectors in each dimension
        t_pre : float
            Time before stimulus (seconds)
        t_post : float
            Time after reward (seconds)
        n_null : int
            Number of null simulations
        resid_type : str
            Type of residual calculation ('sector' or 'trial')
        resid_corr_tr_cond : str
            Trial condition for residual correlation
        channel : str
            'red' or 'grn' - which channel to analyze (default: 'grn')
        use_zscore : bool
            If True, compute z-scored fluorescence using the baseline period
            (mean and std over first n_frames_pre frames) instead of df/f.
            (default: False)
        compute_null : bool
            If True, compute null distributions via add_null_dists().
            Set to False for faster batch processing. (default: True)
        compute_resid_corr : bool
            If True, compute O(n_sectors^4) residual cross-correlations.
            Set to False for faster batch processing. (default: True)
        colors : list
            Color palette for plotting
        """
        # compute whole-frame avg in case needed; re-run if unit mismatch
        if not hasattr(self, 'frame') \
                or self.frame.params.use_zscore != use_zscore \
                or self.frame.params.channel != channel:
            self.add_frame(t_pre=t_pre, t_post=t_post, channel=channel,
                           use_zscore=use_zscore)

        print(f'creating trial-averaged signal (sectors, {channel})...')
        self.add_lickrates(t_prestim=t_pre,
                           t_poststim=t_post)

        rec = self._get_rec(channel)
        rec_t = self._get_rec_t(channel)

        self.sector = SimpleNamespace()

        self.sector.params = SimpleNamespace()
        self.sector.n_sectors = n_sectors
        self.sector.params.t_rew_pre = t_pre
        self.sector.params.t_rew_post = t_post
        self.sector.params.channel = channel
        self.sector.params.use_zscore = use_zscore
        self.sector.params.resid_type = resid_type
        self.sector.n_sectors = n_sectors
        self.sector.n_null = n_null

        self.sector.colors = {'0': colors[0],
                              '0.5': colors[1],
                              '1': colors[2],
                              '0.5_rew': colors[1],
                              '0.5_norew': colors[1],
                              '0.5_prelick': colors[1],
                              '0.5_noprelick': colors[1]}
        self.sector.linestyle = {'0': 'solid',
                                 '0.5': 'solid',
                                 '1': 'solid',
                                 '0.5_rew': 'solid',
                                 '0.5_norew': 'dashed',
                                 '0.5_prelick': 'solid',
                                 '0.5_noprelick': 'dashed'}

        # setup stim-aligned traces
        n_frames_pre = int(t_pre * self.samp_rate)
        n_frames_post = int((2 + t_post) * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post

        self.sector.dff = np.empty((n_sectors, n_sectors), dtype=dict)
        self.sector.dff_resid = np.empty((n_sectors, n_sectors), dtype=dict)

        self.sector.t = np.linspace(
            -1*t_pre, 2+t_post, n_frames_tot)

        # store edges of sectors
        self.sector.x = SimpleNamespace()
        self.sector.x.lower = np.zeros((n_sectors, n_sectors), dtype=int)
        self.sector.x.upper = np.zeros((n_sectors, n_sectors), dtype=int)
        self.sector.y = SimpleNamespace()
        self.sector.y.lower = np.zeros((n_sectors, n_sectors), dtype=int)
        self.sector.y.upper = np.zeros((n_sectors, n_sectors), dtype=int)

        # Auto-detect near-zero baseline (corrected residual signal) once
        # before the sector loop to avoid per-trial dF/F division by ~0.
        _use_zscore = use_zscore
        if not _use_zscore and self.beh._stimrange.first < self.beh._stimrange.last:
            _t0 = self.beh.stim.t_start[self.beh._stimrange.first]
            _ind0 = np.argmin(np.abs(rec_t - (_t0 - t_pre)))
            _probe = np.mean(np.mean(
                rec[_ind0:_ind0 + n_frames_pre, :, :], axis=1), axis=1)
            _f0_probe = float(np.mean(_probe)) if _probe.size > 0 else 1.0
            if np.abs(_f0_probe) < 1.0:
                print(f'\tWarning: baseline fluorescence mean ({_f0_probe:.4f}) '
                      'is near zero. Signal appears to be a corrected residual. '
                      'Falling back to z-score. Pass use_zscore=True explicitly '
                      'to suppress this check.')
                _use_zscore = True

        # ---------------------
        # Precompute trial time indices once (avoid repeated O(n) searches)
        print('\tprecomputing trial indices...')
        _n_trials = self.beh._stimrange.last - self.beh._stimrange.first
        _trial_ind_start = np.empty(_n_trials, dtype=int)
        _trial_ind_end = np.empty(_n_trials, dtype=int)
        _reward_times = self.beh._data.get_event_var('totalRewardTimes')

        for _i, trial in enumerate(range(self.beh._stimrange.first,
                                         self.beh._stimrange.last)):
            _t_stim = self.beh.stim.t_start[trial]
            _t_start = _t_stim - t_pre
            _t_end = _reward_times[trial] + t_post
            _trial_ind_start[_i] = np.argmin(np.abs(rec_t - _t_start))
            _trial_ind_end[_i] = np.argmin(np.abs(rec_t - _t_end)) + 2

        # ---------------------
        # Precompute sector bounds (once, not per-trial)
        print('\tprecomputing sector bounds...')
        _sector_x_lower = np.zeros((n_sectors, n_sectors), dtype=int)
        _sector_x_upper = np.zeros((n_sectors, n_sectors), dtype=int)
        _sector_y_lower = np.zeros((n_sectors, n_sectors), dtype=int)
        _sector_y_upper = np.zeros((n_sectors, n_sectors), dtype=int)

        for n_x in range(n_sectors):
            for n_y in range(n_sectors):
                _sector_x_lower[n_x, n_y] = int((n_x / n_sectors) * rec.shape[1])
                _sector_x_upper[n_x, n_y] = int(((n_x+1) / n_sectors) * rec.shape[1])
                _sector_y_lower[n_x, n_y] = int((n_y / n_sectors) * rec.shape[1])
                _sector_y_upper[n_x, n_y] = int(((n_y+1) / n_sectors) * rec.shape[1])

        self.sector.x.lower = _sector_x_lower
        self.sector.x.upper = _sector_x_upper
        self.sector.y.lower = _sector_y_lower
        self.sector.y.upper = _sector_y_upper

        # ---------------------
        # Initialize data structures for all sectors
        for n_x in range(n_sectors):
            for n_y in range(n_sectors):
                self.sector.dff[n_x, n_y] = {}
                self.sector.dff_resid[n_x, n_y] = {}
                for tr_cond in ['0', '0.5', '1', '0.5_rew',
                                '0.5_norew', '0.5_prelick', '0.5_noprelick']:
                    self.sector.dff[n_x, n_y][tr_cond] = np.zeros((
                        self.beh.tr_inds[tr_cond].shape[0], n_frames_tot))
                    self.sector.dff_resid[n_x, n_y][tr_cond] = np.zeros_like(
                        self.sector.dff[n_x, n_y][tr_cond])

        # ---------------------
        # Extract traces: iterate over TRIALS (not sectors) to minimize I/O
        # Each trial's full frame is read ONCE, then all sectors extracted in-memory
        print('\textracting aligned fluorescence traces...')
        self.sector.tr_counts = {'0': 0, '0.5': 0, '1': 0,
                                 '0.5_rew': 0, '0.5_norew': 0,
                                 '0.5_prelick': 0, '0.5_noprelick': 0}

        for _i, trial in enumerate(range(self.beh._stimrange.first,
                                         self.beh._stimrange.last)):
            if (_i + 1) % 10 == 0 or _i == _n_trials - 1:
                print(f'\t\ttrial {_i+1}/{_n_trials}...      ', end='\r')

            _ind_t_start = _trial_ind_start[_i]
            _ind_t_end = _trial_ind_end[_i]

            # Read full frame for this trial ONCE (contiguous temporal read)
            _frame_data = np.asarray(rec[_ind_t_start:_ind_t_end, :, :])

            # Determine which conditions this trial belongs to
            _prob = self.beh.stim.prob[trial]
            _rew = self.beh.rew.delivered[trial]
            _antic = self.beh.lick.antic_raw[trial]

            # Extract all sectors from this frame (in-memory slicing, fast)
            for n_x in range(n_sectors):
                for n_y in range(n_sectors):
                    _xl, _xu = _sector_x_lower[n_x, n_y], _sector_x_upper[n_x, n_y]
                    _yl, _yu = _sector_y_lower[n_x, n_y], _sector_y_upper[n_x, n_y]

                    # Extract sector and compute mean fluorescence
                    _f = np.mean(np.mean(
                        _frame_data[:, _xl:_xu, _yl:_yu], axis=1), axis=1)

                    # Compute dff or zscore
                    if _use_zscore:
                        _baseline = _f[:n_frames_pre]
                        _sigma = np.std(_baseline)
                        _dff = (_f - np.mean(_baseline)) / _sigma \
                            if _sigma > 0 else np.zeros_like(_f)
                    else:
                        _dff = calc_dff(_f, baseline_frames=n_frames_pre)

                    _dff_slice = _dff[0:n_frames_tot]

                    # Store in appropriate condition arrays
                    if _prob == 0:
                        self.sector.dff[n_x, n_y]['0'][
                            self.sector.tr_counts['0'], :] = _dff_slice
                    elif _prob == 0.5:
                        self.sector.dff[n_x, n_y]['0.5'][
                            self.sector.tr_counts['0.5'], :] = _dff_slice
                    elif _prob == 1:
                        self.sector.dff[n_x, n_y]['1'][
                            self.sector.tr_counts['1'], :] = _dff_slice

                    if _prob == 0.5 and _rew == 1:
                        self.sector.dff[n_x, n_y]['0.5_rew'][
                            self.sector.tr_counts['0.5_rew'], :] = _dff_slice
                    elif _prob == 0.5 and _rew == 0:
                        self.sector.dff[n_x, n_y]['0.5_norew'][
                            self.sector.tr_counts['0.5_norew'], :] = _dff_slice

                    if _prob == 0.5 and _antic > 0:
                        self.sector.dff[n_x, n_y]['0.5_prelick'][
                            self.sector.tr_counts['0.5_prelick'], :] = _dff_slice
                    elif _prob == 0.5 and _antic == 0:
                        self.sector.dff[n_x, n_y]['0.5_noprelick'][
                            self.sector.tr_counts['0.5_noprelick'], :] = _dff_slice

            # Update trial counters AFTER processing all sectors for this trial
            if _prob == 0:
                self.sector.tr_counts['0'] += 1
            elif _prob == 0.5:
                self.sector.tr_counts['0.5'] += 1
                if _rew == 1:
                    self.sector.tr_counts['0.5_rew'] += 1
                else:
                    self.sector.tr_counts['0.5_norew'] += 1
                if _antic > 0:
                    self.sector.tr_counts['0.5_prelick'] += 1
                else:
                    self.sector.tr_counts['0.5_noprelick'] += 1
            elif _prob == 1:
                self.sector.tr_counts['1'] += 1

        print(f'\t\ttrial {_n_trials}/{_n_trials}...done      ')

        # ---------------------
        # Compute dff residuals (vectorized, per sector)
        print('\tcomputing dff residuals...')
        for n_x in range(n_sectors):
            for n_y in range(n_sectors):
                for tr_cond in ['0', '0.5', '1']:
                    _sector_dff = self.sector.dff[n_x, n_y][tr_cond]
                    if resid_type == 'sector':
                        # compute mean once, broadcast subtract
                        _dff_mean = np.mean(_sector_dff, axis=0, keepdims=True)
                        self.sector.dff_resid[n_x, n_y][tr_cond] = \
                            _sector_dff - _dff_mean
                    elif resid_type == 'trial':
                        # per-trial mean from frame
                        self.sector.dff_resid[n_x, n_y][tr_cond] = \
                            _sector_dff - self.frame.dff[tr_cond]
        print('\t\tdone')

        # compute cross-correlations (optional, expensive O(n_sectors^4))
        if compute_resid_corr:
            print('\tcomputing residual cross-correlations...')
            self.sector.resid_corr_stat = np.zeros(
                (self.sector.n_sectors**2, self.sector.n_sectors**2))
            self.sector.resid_corr_pval = np.zeros(
                (self.sector.n_sectors**2, self.sector.n_sectors**2))

            for sec_1 in range(self.sector.n_sectors**2):
                for sec_2 in range(self.sector.n_sectors**2):
                    sec_1_x, sec_1_y = np.divmod(sec_1, self.sector.n_sectors)
                    sec_2_x, sec_2_y = np.divmod(sec_2, self.sector.n_sectors)

                    _sec_1_sig = np.mean(
                        self.sector.dff_resid[sec_1_x, sec_1_y][resid_corr_tr_cond],
                        axis=1)
                    _sec_2_sig = np.mean(
                        self.sector.dff_resid[sec_2_x, sec_2_y][resid_corr_tr_cond],
                        axis=1)

                    _corr = sp.stats.pearsonr(_sec_1_sig, _sec_2_sig)
                    self.sector.resid_corr_stat[sec_1, sec_2] = _corr.statistic
                    self.sector.resid_corr_pval[sec_1, sec_2] = _corr.pvalue

        # add null distributions (optional, expensive)
        # -------------
        if compute_null:
            self.add_null_dists(n_null=n_null, t_pre=t_pre, t_post=t_post,
                                channel=channel)

        return

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
                    plt_dff={'x': {'gain': 0.6,
                                   'offset': 0.2},
                             'y': {'gain': 10,
                                   'offset': 0.2}},
                    plt_prefix='',
                    plt_show=True,
                    channel='grn',
                    use_zscore=False,
                    outlier_thresh=None,
                    auto_gain_dff=False,
                    minimal_output=False,
                    colors=sns.cubehelix_palette(
                        n_colors=3,
                        start=2, rot=0,
                        dark=0.2, light=0.8)):
        """
        Divides the field of view into sectors, and plots a set of trial-types
        separately within each sector.

        Dual-colour version of TwoPRec.plt_sectors: accepts a channel kwarg
        to select which imaging channel to use for fluorescence extraction
        and the max-projection background image.

        Parameters
        ----------
        channel : str
            Which channel to use: 'grn' or 'red' (default: 'grn')
        use_zscore : bool
            If True, compute z-scored fluorescence (baseline-normalised to
            SD units) instead of df/f. (default: False)
        outlier_thresh : float or None
            Sectors/trials whose peak absolute signal exceeds this value are
            not plotted (outlier suppression). If None, defaults to 10 when
            use_zscore=True, or 1 when use_zscore=False. (default: None)
        minimal_output : bool
            If True, only generate the main sector plot (fig_avg) and skip
            all supplemental figures (residuals, correlations, histograms).
            Much faster for batch processing. (default: False)

        All other parameters are identical to TwoPRec.plt_sectors.
        """
        # resolve outlier threshold
        if outlier_thresh is None:
            outlier_thresh = 20 if use_zscore else 1

        rec = self._get_rec(channel)
        rec_t = self._get_rec_t(channel)

        print(f'creating trial-averaged signal (sectors, {channel})...')
        self.add_lickrates(t_prestim=t_pre,
                           t_poststim=t_post)

        # ---- n_frames must be known before the cache check ----
        n_frames_pre = int(t_pre * self.samp_rate)
        n_frames_post = int((2 + t_post) * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post

        # Auto-detect near-zero baseline (corrected residual signal) once
        # before the sector loop to avoid per-trial dF/F division by ~0.
        _use_zscore = use_zscore
        if not _use_zscore and self.beh._stimrange.first < self.beh._stimrange.last:
            _t0 = self.beh.stim.t_start[self.beh._stimrange.first]
            _ind0 = np.argmin(np.abs(rec_t - (_t0 - t_pre)))
            _probe = np.mean(np.mean(
                rec[_ind0:_ind0 + n_frames_pre, :, :], axis=1), axis=1)
            _f0_probe = float(np.mean(_probe)) if _probe.size > 0 else 1.0
            if np.abs(_f0_probe) < 1.0:
                print(f'\tWarning: baseline fluorescence mean ({_f0_probe:.4f}) '
                      'is near zero. Signal appears to be a corrected residual. '
                      'Falling back to z-score. Pass use_zscore=True explicitly '
                      'to suppress this check.')
                _use_zscore = True

        # auto-enable gain scaling when in z-score mode (default gain=10 is
        # calibrated for dF/F ~0.1-1.0; z-scores ~1-5 overflow sector boundaries)
        if _use_zscore and not auto_gain_dff:
            auto_gain_dff = True

        # compute whole-frame avg in case needed; re-run if unit mismatch
        if not hasattr(self, 'frame') \
                or self.frame.params.use_zscore != _use_zscore \
                or self.frame.params.channel != channel:
            self.add_frame(t_pre=t_pre, t_post=t_post, channel=channel,
                           use_zscore=_use_zscore)

        # determine stim_list once (needed by auto_gain_dff and PASS 2)
        if plot_type is None:
            stim_list = ['0', '0.5', '1']
        elif plot_type == 'rew_norew':
            stim_list = ['0.5_rew', '0.5_norew']
        elif plot_type == 'prelick_noprelick':
            stim_list = ['0.5_prelick', '0.5_noprelick']
        elif plot_type == 'rew':
            stim_list = ['0', '0.5_rew', '1']

        # ---- check if sector data from add_sectors() can be reused ----
        # A cache hit skips the entire O(n_sectors^2 * n_trials) extraction
        # loop and the O(n_trials) memmap reads it requires.
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

        if _sector_cache_valid:
            print('\treusing cached sector data from add_sectors()...')
            # refresh colors/linestyle (caller may use a different palette)
            self.sector.colors = {'0': colors[0],
                                  '0.5': colors[1],
                                  '1': colors[2],
                                  '0.5_rew': colors[1],
                                  '0.5_norew': colors[1],
                                  '0.5_prelick': colors[1],
                                  '0.5_noprelick': colors[1]}
            self.sector.linestyle = {'0': 'solid',
                                     '0.5': 'solid',
                                     '1': 'solid',
                                     '0.5_rew': 'solid',
                                     '0.5_norew': 'dashed',
                                     '0.5_prelick': 'solid',
                                     '0.5_noprelick': 'dashed'}
            self.sector.params.use_zscore = _use_zscore
            # recompute residuals only if resid_type changed (pure RAM op)
            if getattr(self.sector.params, 'resid_type', None) != resid_type:
                print('\trecomputing residuals (resid_type changed)...')
                for n_x in range(n_sectors):
                    for n_y in range(n_sectors):
                        for tr_cond in ['0', '0.5', '1']:
                            for tr in range(
                                    self.sector.dff[n_x, n_y][tr_cond].shape[0]):
                                if resid_type == 'sector':
                                    _dff_mean = np.mean(
                                        self.sector.dff[n_x, n_y][tr_cond],
                                        axis=0)
                                elif resid_type == 'trial':
                                    _dff_mean = self.frame.dff[tr_cond][tr, :]
                                self.sector.dff_resid[n_x, n_y][tr_cond][tr, :] = (
                                    self.sector.dff[n_x, n_y][tr_cond][tr, :]
                                    - _dff_mean)
                self.sector.params.resid_type = resid_type

        else:
            # ---- full extraction (cache miss) ----
            self.sector = SimpleNamespace()

            self.sector.params = SimpleNamespace()
            self.sector.n_sectors = n_sectors
            self.sector.params.t_rew_pre = t_pre
            self.sector.params.t_rew_post = t_post
            self.sector.params.channel = channel
            self.sector.params.use_zscore = _use_zscore
            self.sector.params.resid_type = resid_type
            self.sector.n_sectors = n_sectors

            self.sector.colors = {'0': colors[0],
                                  '0.5': colors[1],
                                  '1': colors[2],
                                  '0.5_rew': colors[1],
                                  '0.5_norew': colors[1],
                                  '0.5_prelick': colors[1],
                                  '0.5_noprelick': colors[1]}
            self.sector.linestyle = {'0': 'solid',
                                     '0.5': 'solid',
                                     '1': 'solid',
                                     '0.5_rew': 'solid',
                                     '0.5_norew': 'dashed',
                                     '0.5_prelick': 'solid',
                                     '0.5_noprelick': 'dashed'}

            self.sector.dff = np.empty((n_sectors, n_sectors), dtype=dict)
            self.sector.dff_resid = np.empty((n_sectors, n_sectors), dtype=dict)

            self.sector.t = np.linspace(
                -1*t_pre, 2+t_post, n_frames_tot)

            # store edges of sectors
            self.sector.x = SimpleNamespace()
            self.sector.x.lower = np.zeros((n_sectors, n_sectors))
            self.sector.x.upper = np.zeros((n_sectors, n_sectors))
            self.sector.y = SimpleNamespace()
            self.sector.y.lower = np.zeros((n_sectors, n_sectors))
            self.sector.y.upper = np.zeros((n_sectors, n_sectors))

            # ---------------------
            print('\textracting aligned fluorescence traces...')
            _n_trace = 1

            # ---- PASS 1: extract dff for all sectors ----
            for n_x in range(n_sectors):
                for n_y in range(n_sectors):
                    print(f'\t\tsector {_n_trace}/{n_sectors**2}...      ',
                          end='\r')
                    # calculate location of sector
                    _ind_x_lower = int((n_x / n_sectors) * rec.shape[1])
                    _ind_x_upper = int(((n_x+1) / n_sectors) * rec.shape[1])

                    _ind_y_lower = int((n_y / n_sectors) * rec.shape[1])
                    _ind_y_upper = int(((n_y+1) / n_sectors) * rec.shape[1])

                    self.sector.x.lower[n_x, n_y] = _ind_x_lower
                    self.sector.x.upper[n_x, n_y] = _ind_x_upper
                    self.sector.y.lower[n_x, n_y] = _ind_y_lower
                    self.sector.y.upper[n_x, n_y] = _ind_y_upper

                    # setup structure
                    self.sector.dff[n_x, n_y] = {}
                    self.sector.dff_resid[n_x, n_y] = {}

                    for tr_cond in ['0', '0.5', '1', '0.5_rew',
                                    '0.5_norew', '0.5_prelick', '0.5_noprelick']:
                        self.sector.dff[n_x, n_y][tr_cond] = np.zeros((
                            self.beh.tr_inds[tr_cond].shape[0], n_frames_tot))
                        self.sector.dff_resid[n_x, n_y][tr_cond] = np.zeros_like(
                            self.sector.dff[n_x, n_y][tr_cond])

                    # store stim-aligned traces
                    self.sector.tr_counts = {'0': 0, '0.5': 0, '1': 0,
                                             '0.5_rew': 0, '0.5_norew': 0,
                                             '0.5_prelick': 0,
                                             '0.5_noprelick': 0}
                    for trial in range(self.beh._stimrange.first,
                                       self.beh._stimrange.last):
                        _t_stim = self.beh.stim.t_start[trial]
                        _t_start = _t_stim - t_pre
                        _t_end = self.beh._data.get_event_var(
                            'totalRewardTimes')[trial] + t_post

                        _ind_t_start = np.argmin(np.abs(
                            rec_t - _t_start))
                        _ind_t_end = np.argmin(np.abs(
                            rec_t - _t_end)) + 2   # add frames to end

                        # extract fluorescence
                        _f = np.mean(np.mean(
                            rec[_ind_t_start:_ind_t_end,
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

                        if self.beh.stim.prob[trial] == 0:
                            self.sector.dff[n_x, n_y]['0'][
                                self.sector.tr_counts['0'], :] \
                                = _dff[0:n_frames_tot]
                            self.sector.tr_counts['0'] += 1
                        elif self.beh.stim.prob[trial] == 0.5:
                            self.sector.dff[n_x, n_y]['0.5'][
                                self.sector.tr_counts['0.5'], :] \
                                = _dff[0:n_frames_tot]
                            self.sector.tr_counts['0.5'] += 1
                        elif self.beh.stim.prob[trial] == 1:
                            self.sector.dff[n_x, n_y]['1'][
                                self.sector.tr_counts['1'], :] \
                                = _dff[0:n_frames_tot]
                            self.sector.tr_counts['1'] += 1

                        if self.beh.stim.prob[trial] == 0.5 \
                           and self.beh.rew.delivered[trial] == 1:
                            self.sector.dff[n_x, n_y]['0.5_rew'][
                                self.sector.tr_counts['0.5_rew'], :] \
                                = _dff[0:n_frames_tot]
                            self.sector.tr_counts['0.5_rew'] += 1
                        elif self.beh.stim.prob[trial] == 0.5 \
                             and self.beh.rew.delivered[trial] == 0:
                            self.sector.dff[n_x, n_y]['0.5_norew'][
                                self.sector.tr_counts['0.5_norew'], :] \
                                = _dff[0:n_frames_tot]
                            self.sector.tr_counts['0.5_norew'] += 1

                        if self.beh.stim.prob[trial] == 0.5 \
                           and self.beh.lick.antic_raw[trial] > 0:
                            self.sector.dff[n_x, n_y]['0.5_prelick'][
                                self.sector.tr_counts['0.5_prelick'], :] \
                                = _dff[0:n_frames_tot]
                            self.sector.tr_counts['0.5_prelick'] += 1
                        elif self.beh.stim.prob[trial] == 0.5 \
                             and self.beh.lick.antic_raw[trial] == 0:
                            self.sector.dff[n_x, n_y]['0.5_noprelick'][
                                self.sector.tr_counts['0.5_noprelick'], :] \
                                = _dff[0:n_frames_tot]
                            self.sector.tr_counts['0.5_noprelick'] += 1

                    # store dff residuals
                    # ----------
                    for tr_cond in ['0', '0.5', '1']:
                        for tr in range(
                                self.sector.dff[n_x, n_y][tr_cond].shape[0]):
                            if resid_type == 'sector':
                                _dff_mean = np.mean(
                                    self.sector.dff[n_x, n_y][tr_cond], axis=0)
                            elif resid_type == 'trial':
                                _dff_mean = self.frame.dff[tr_cond][tr, :]

                            self.sector.dff_resid[n_x, n_y][tr_cond][tr, :] \
                                = self.sector.dff[n_x, n_y][tr_cond][tr, :] \
                                - _dff_mean

                    _n_trace += 1

        # ---- figure creation (always, after data is ready) ----
        fig_avg = plt.figure(figsize=figsize)
        spec_avg = gs.GridSpec(nrows=1, ncols=1,
                               figure=fig_avg)
        ax_img = fig_avg.add_subplot(spec_avg[0, 0])

        # residual figure only created when not in minimal mode
        fig_resid = None
        ax_img_resid = None
        if not minimal_output:
            fig_resid = plt.figure(figsize=figsize)
            spec_resid = gs.GridSpec(nrows=1, ncols=1,
                                     figure=fig_resid)
            ax_img_resid = fig_resid.add_subplot(spec_resid[0, 0])

        print('\tcreating max projection image...')
        _rec_max = np.max(rec[::img_ds_factor, :, :], axis=0)
        _rec_max[:, ::int(_rec_max.shape[0]/n_sectors)] = 0
        _rec_max[::int(_rec_max.shape[0]/n_sectors), :] = 0

        ax_img.imshow(_rec_max,
                      extent=[0, n_sectors,
                              n_sectors, 0],
                      alpha=img_alpha)
        if not minimal_output:
            ax_img_resid.imshow(_rec_max,
                                extent=[0, n_sectors,
                                        n_sectors, 0],
                                alpha=img_alpha)
            ax_img_resid.set_title(f'residuals vs mean({resid_type}),'
                                   + f' p(rew)={resid_tr_cond}')

        # ---- auto-gain: set y-gain from global max across all sectors ----
        if auto_gain_dff:
            _max_pos = 0.0
            for n_x in range(n_sectors):
                for n_y in range(n_sectors):
                    for stim_cond in stim_list:
                        _med = np.median(
                            self.sector.dff[n_x, n_y][stim_cond], axis=0)
                        _max_pos = max(_max_pos, float(np.max(_med)))
            if _max_pos > 0:
                _offset = plt_dff['y']['offset']
                plt_dff = dict(plt_dff)
                plt_dff['y'] = dict(plt_dff['y'])
                # positive dff is negated then plotted upward toward sector above;
                # zero-line sits at n_y + (1 - offset), so available headroom
                # before hitting the sector boundary at n_y is (1 - offset)
                plt_dff['y']['gain'] = (1.0 - _offset) / _max_pos

        # ---- PASS 2: plot all sectors ----
        for n_x in range(n_sectors):
            for n_y in range(n_sectors):

                # **************************
                # plotting for this sector
                # **************************

                # make sector plot of stim-aligned traces
                # ----------
                _t_sector = ((self.sector.t / self.sector.t[-1])
                             * plt_dff['x']['gain'])
                _t_sector = _t_sector + n_x + plt_dff['x']['offset']

                for stim_cond in stim_list:
                    _dff_mean = np.median(self.sector.dff[n_x, n_y][stim_cond],
                                          axis=0)
                    _dff_shifted = (-1 * _dff_mean * plt_dff['y']['gain']) \
                        + n_y - plt_dff['y']['offset'] + 1

                    if np.max(np.abs(_dff_mean)) < outlier_thresh:
                        ax_img.plot(
                            _t_sector, _dff_shifted,
                            color=self.sector.colors[stim_cond],
                            linestyle=self.sector.linestyle[stim_cond])

                # make residuals plot (skip in minimal mode)
                # ------------------------
                if not minimal_output:
                    _n_tr_resid = self.sector.dff_resid[n_x, n_y][
                        resid_tr_cond].shape[0]
                    for trial in range(_n_tr_resid):
                        _dff = self.sector.dff_resid[n_x, n_y][resid_tr_cond][trial, :]
                        if resid_smooth is True:
                            _dff = sp.signal.savgol_filter(_dff, resid_smooth_windlen,
                                                           resid_smooth_polyorder)
                        _dff_shifted = (-1 * _dff * plt_dff['y']['gain']) \
                            + n_y - plt_dff['y']['offset'] + 1

                        if np.max(np.abs(_dff)) < outlier_thresh:
                            ax_img_resid.plot(
                                _t_sector, _dff_shifted,
                                color=sns.color_palette('rocket', _n_tr_resid)[trial],
                                alpha=resid_alpha,
                                linestyle=self.sector.linestyle[stim_cond])

                # plot rew and stim lines
                # ------------
                _t_stim = (0 * plt_dff['x']['gain']) \
                    + n_x + plt_dff['x']['offset']
                _t_rew = ((2 / self.sector.t[-1])
                          * plt_dff['x']['gain']) \
                    + n_x + plt_dff['x']['offset']
                _dff_zero = n_y - plt_dff['y']['offset'] + 1

                # draw stim/rew lines on main plot
                ax_img.plot([_t_stim, _t_stim],
                            [n_y + 0.2, n_y + 0.8],
                            color=sns.xkcd_rgb['dark grey'],
                            linewidth=1,
                            linestyle='dashed')
                ax_img.plot([_t_rew, _t_rew],
                            [n_y + 0.2, n_y + 0.8],
                            color=sns.xkcd_rgb['bright blue'],
                            linewidth=1,
                            linestyle='dashed')

                # draw stim/rew lines on residual plot (skip in minimal mode)
                if not minimal_output:
                    ax_img_resid.plot([_t_stim, _t_stim],
                                      [n_y + 0.2, n_y + 0.8],
                                      color=sns.xkcd_rgb['dark grey'],
                                      linewidth=1,
                                      linestyle='dashed')
                    ax_img_resid.plot([_t_rew, _t_rew],
                                      [n_y + 0.2, n_y + 0.8],
                                      color=sns.xkcd_rgb['bright blue'],
                                      linewidth=1,
                                      linestyle='dashed')
                    ax_img_resid.plot([_t_stim, _t_rew],
                                      [_dff_zero, _dff_zero],
                                      color='k',
                                      linewidth=1,
                                      linestyle='dashed')

        # Add correction method suffix if correction was applied with replace_grn
        corr_suffix = ''
        if channel == 'grn' and hasattr(self, 'rec_t_grn') \
           and hasattr(self, 'corr_sig_method'):
            corr_suffix = f'_corr={self.corr_sig_method}'

        # save main figure
        fig_avg.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            'sectors_'
            + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{plt_prefix}_'
            + f'{n_sectors=}_{plot_type=}_{t_pre=}_{t_post=}_'
            + f'zscore={self.sector.params.use_zscore}_'
            + f'ch={channel}{corr_suffix}.pdf'))

        # compute and save supplemental plots (skip in minimal mode)
        # ---------------
        if not minimal_output:
            n_trs = self.sector.dff_resid[0, 0][resid_tr_cond].shape[0]
            n_rows = np.ceil(np.sqrt(n_trs)).astype(int)

            fig_norm_sec_sig = plt.figure(figsize=(3.43, 1.5))
            spec_norm_sec_sig = gs.GridSpec(nrows=1, ncols=3,
                                            figure=fig_norm_sec_sig)
            ax_norm_sec_sig_prew = {}
            ax_norm_sec_sig_prew['0'] = fig_norm_sec_sig.add_subplot(
                spec_norm_sec_sig[0, 0])
            ax_norm_sec_sig_prew['0.5'] = fig_norm_sec_sig.add_subplot(
                spec_norm_sec_sig[0, 1])
            ax_norm_sec_sig_prew['1'] = fig_norm_sec_sig.add_subplot(
                spec_norm_sec_sig[0, 2])

            fig_resid_corr = plt.figure(figsize=figsize)
            spec_resid_corr = gs.GridSpec(nrows=1, ncols=1,
                                          figure=fig_resid_corr)
            ax_resid_corr = fig_resid_corr.add_subplot(spec_resid_corr[0, 0])

            fig_resid_corr_spatial = plt.figure(figsize=figsize)
            spec_resid_corr_spatial = gs.GridSpec(
                nrows=self.sector.n_sectors,
                ncols=self.sector.n_sectors,
                figure=fig_resid_corr_spatial)

            fig_resid_tr_sep = plt.figure(figsize=figsize)
            spec_resid_tr_sep = gs.GridSpec(nrows=n_rows,
                                            ncols=n_rows,
                                            figure=fig_resid_tr_sep)

            # compute and plot trial-averaged, normalized signals for each sector
            for sec in range(self.sector.n_sectors):
                sec_x, sec_y = np.divmod(sec, self.sector.n_sectors)

                for tr_type in ['0', '0.5', '1']:
                    _n_tr = self.sector.dff[0, 0][tr_type].shape[0]
                    _sec_max = np.max(self.sector.dff[sec_x, sec_y][tr_type],
                                      axis=1)
                    _sec_sig = np.array([self.sector.dff[sec_x, sec_y][tr_type][
                        tr, :] / _sec_max[tr] for tr in range(_n_tr)])
                    _sec_sig_travg = np.mean(_sec_sig, axis=0)
                    ax_norm_sec_sig_prew[tr_type].plot(
                        self.sector.t,
                        _sec_sig_travg,
                        color=self.sector.colors[tr_type],
                        alpha=0.8)

            # compute and plot cross-correlations
            self.sector.resid_corr_stat = np.zeros(
                (self.sector.n_sectors**2, self.sector.n_sectors**2))
            self.sector.resid_corr_pval = np.zeros(
                (self.sector.n_sectors**2, self.sector.n_sectors**2))

            for sec_1 in range(self.sector.n_sectors**2):
                for sec_2 in range(self.sector.n_sectors**2):
                    sec_1_x, sec_1_y = np.divmod(sec_1, self.sector.n_sectors)
                    sec_2_x, sec_2_y = np.divmod(sec_2, self.sector.n_sectors)

                    _sec_1_sig = np.mean(
                        self.sector.dff_resid[sec_1_x, sec_1_y][resid_tr_cond],
                        axis=1)
                    _sec_2_sig = np.mean(
                        self.sector.dff_resid[sec_2_x, sec_2_y][resid_tr_cond],
                        axis=1)

                    _corr = sp.stats.pearsonr(_sec_1_sig, _sec_2_sig)
                    self.sector.resid_corr_stat[sec_1, sec_2] = _corr.statistic
                    self.sector.resid_corr_pval[sec_1, sec_2] = _corr.pvalue

            ax_resid_corr.imshow(self.sector.resid_corr_stat)

            ax_residcorr_spatial = []
            ax_count = 0
            for sec_1 in range(self.sector.n_sectors):
                for sec_2 in range(self.sector.n_sectors):
                    ax_residcorr_spatial.append(
                        fig_resid_corr_spatial.add_subplot(
                            spec_resid_corr_spatial[sec_1, sec_2]))
                    ax_residcorr_spatial[ax_count].imshow(
                        self.sector.resid_corr_stat[ax_count, :].reshape(
                            self.sector.n_sectors, self.sector.n_sectors),
                        vmax=1, vmin=-1, cmap='coolwarm')
                    ax_residcorr_spatial[ax_count].set_xticks([])
                    ax_residcorr_spatial[ax_count].set_yticks([])

                    ax_count += 1

            fig_resid_corr_spatial.suptitle('residual corrs (spatial), '
                                            + f' {resid_tr_cond=},'
                                            + f' residuals vs mean({resid_type})')

            # compute and plot per-trial residual distributions
            ax = []
            for tr in range(n_trs):
                # compile response of all sectors for this trial
                _sector_dff_resid = []
                for sec_1 in range(self.sector.n_sectors):
                    for sec_2 in range(self.sector.n_sectors):
                        _sector_dff_resid.append(np.mean(
                            self.sector.dff_resid[sec_1, sec_2][
                                resid_tr_cond][tr, :]))

                # set up axis
                _tr_x, _tr_y = np.divmod(tr, n_rows)
                if tr == 0:
                    ax.append(fig_resid_tr_sep.add_subplot(
                        spec_resid_tr_sep[_tr_x, _tr_y]))
                elif tr > 0:
                    ax.append(fig_resid_tr_sep.add_subplot(
                        spec_resid_tr_sep[_tr_x, _tr_y],
                              sharex=ax[0], sharey=ax[0]))

                # plot per-trial distributions
                ax[tr].hist(_sector_dff_resid, bins=resid_trial_nbins,
                            histtype='step',
                            density=True, color=sns.xkcd_rgb['dark grey'])
                ax[tr].axvline(0, color='k',
                               linewidth=1,
                               linestyle='dashed')
                ax[tr].set_xlabel('dff resid.')
                ax[tr].set_ylabel('pdf')
                ax[tr].set_title(f'trial={tr}')
            fig_resid_tr_sep.suptitle(f'residuals vs mean{resid_type} corrs (spatial), '
                                            + f' {resid_tr_cond=},'
                                            + f' residuals vs mean({resid_type})')

            # save supplemental figs
            fig_norm_sec_sig.savefig(os.path.join(
                os.getcwd(), 'figs_mbl',
                'sectors_normed_'
                + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
                + f'{plt_prefix}_'
                + f'{n_sectors=}_{plot_type=}_{t_pre=}_{t_post=}_'
                + f'zscore={self.sector.params.use_zscore}_'
                + f'ch={channel}{corr_suffix}.pdf'))

            fig_resid.savefig(os.path.join(
                os.getcwd(), 'figs_mbl',
                'sectors_resid_'
                + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
                + f'{plt_prefix}_'
                + f'{n_sectors=}_{plot_type=}_{t_pre=}_{t_post=}_'
                + f'{resid_tr_cond=}_{resid_type=}_'
                + f'zscore={self.sector.params.use_zscore}_'
                + f'ch={channel}{corr_suffix}.pdf'))

            fig_resid_corr.savefig(os.path.join(
                os.getcwd(), 'figs_mbl',
                'sectors_resid_corr_'
                + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
                + f'{plt_prefix}_'
                + f'{n_sectors=}_{plot_type=}_{t_pre=}_{t_post=}_'
                + f'{resid_tr_cond=}_{resid_type=}_'
                + f'ch={channel}{corr_suffix}.pdf'))

            fig_resid_corr_spatial.savefig(os.path.join(
                os.getcwd(), 'figs_mbl',
                'sectors_resid_corr_spatial_'
                + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
                + f'{plt_prefix}_'
                + f'{n_sectors=}_{plot_type=}_{t_pre=}_{t_post=}_'
                + f'{resid_tr_cond=}_{resid_type=}_'
                + f'ch={channel}{corr_suffix}.pdf'))

            fig_resid_tr_sep.savefig(os.path.join(
                os.getcwd(), 'figs_mbl',
                'sectors_resid_trial_sep_'
                + f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
                + f'{plt_prefix}_'
                + f'{n_sectors=}_{plot_type=}_{t_pre=}_{t_post=}_'
                + f'{resid_tr_cond=}_{resid_type=}_'
                + f'ch={channel}{corr_suffix}.pdf'))

        # show/close plots
        if plt_show is True:
            plt.show()
        elif plt_show is False:
            plt.close(fig_avg)
            if not minimal_output:
                for _fig in [fig_norm_sec_sig, fig_resid,
                             fig_resid_corr, fig_resid_corr_spatial,
                             fig_resid_tr_sep]:
                    plt.close(_fig)

        return

    def add_null_dists(self, n_null=100, auto_run_methods=True,
                       t_pre=2, t_post=5, channel='grn'):
        """
        Makes a 'null' distribution based on the alternate hypothesis
        of no spatial variability in task-related signaling.

        Parameters
        ----------
        n_null : int
            Number of null simulations
        auto_run_methods : bool
            Whether to auto-run add_frame and add_sectors if needed
        t_pre : float
            Time before stimulus (seconds)
        t_post : float
            Time after reward (seconds)
        channel : str
            'red' or 'grn' - which channel to analyze (default: 'grn')
        """
        print(f'creating null distributions ({channel})')

        # check that sector and frame exist
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
                # compute_null=False avoids infinite recursion
                self.add_sectors(t_pre=t_pre,
                                 t_post=t_post,
                                 channel=channel,
                                 compute_null=False)

        # construct the null distribution
        # Memory-optimized: store only parameters and ONE example trace per
        # sector (for plotting). Full null distributions are generated
        # on-the-fly when needed (e.g., in add_psychometrics).
        self.sector_null = SimpleNamespace()
        self.sector_null.t = self.sector.t
        self.sector_null.n_null = n_null

        # Store ONE example null trace per sector/condition (for plt_null_dist)
        self.sector_null.dff = np.empty(
            (self.sector.n_sectors, self.sector.n_sectors), dtype=dict)

        # Store parameters for on-the-fly generation
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

                # Compute scaling variables (stored for on-the-fly generation)
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

                # Generate ONE example null trace per condition (for plotting)
                self.sector_null.dff[n_x, n_y] = {}
                for tr_cond in ['0', '0.5', '1', '0.5_rew', '0.5_norew']:
                    _frame_data = self.frame.dff[tr_cond]  # (n_trials, n_frames)
                    n_trials, n_frames = _frame_data.shape
                    # Generate just ONE example (shape: 1, n_trials, n_frames)
                    _noise = np.random.normal(
                        scale=_bl_std, size=(1, n_trials, n_frames))
                    self.sector_null.dff[n_x, n_y][tr_cond] = \
                        (_frame_data * _ampli_scaling) + _noise

                _n_trace += 1

        print(f'\t\tsector {self.sector.n_sectors**2}/'
              f'{self.sector.n_sectors**2}...done')
        return

    def add_zetatest_sectors(self, frametime_post=2,
                             zeta_type='2samp', channel='grn'):
        """
        Runs ZETA test on sectors to identify task-responsive regions.

        Parameters
        ----------
        frametime_post : float
            Time after stimulus to include in analysis
        zeta_type : str
            '2samp' or '1samp'
        channel : str
            'red' or 'grn' - which channel to analyze (default: 'grn')
        """
        if not hasattr(self, 'sector'):
            self.add_sectors(channel=channel)

        print(f'adding zetatest for sectors ({channel})...')
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

    def plt_null_dist(self,
                      figsize=(6, 3),
                      img_ds_factor=50,
                      img_alpha=0.5,
                      channel='grn',
                      plt_dff={'x': {'gain': 0.6,
                                     'offset': 0.2},
                               'y': {'gain': 10,
                                     'offset': 0.2}}):
        """
        Plots null distribution comparison.

        Parameters
        ----------
        figsize : tuple
            Figure size
        img_ds_factor : int
            Downsampling factor for image
        img_alpha : float
            Alpha for image overlay
        channel : str
            'red' or 'grn' - which channel to use (default: 'grn')
        plt_dff : dict
            Plotting parameters
        """
        rec = self._get_rec(channel)

        fig = plt.figure(figsize=figsize)
        spec = gs.GridSpec(nrows=1, ncols=2,
                           figure=fig)

        ax_dff = fig.add_subplot(spec[0, 0])
        ax_dff.set_title(f'dff ({channel})')
        ax_dff_null = fig.add_subplot(spec[0, 1])
        ax_dff_null.set_title(f'dff null ({channel})')

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
                # **************************
                # plotting for this sector
                # **************************

                # make sector plot of stim-aligned traces
                # ----------
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

        fig.savefig(os.path.join(
            os.getcwd(), 'figs_mbl',
            f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            + f'{channel=}_null_dists.pdf'))

        plt.show()

        return

    def plt_lick_resp(self, t_pre=3, t_post=3,
                      figsize=(3.43, 3.43),
                      savefig_prefix='grab',
                      channel='grn'):
        """
        Plots lick-aligned response.

        Parameters
        ----------
        t_pre : float
            Time before lick (seconds)
        t_post : float
            Time after lick (seconds)
        figsize : tuple
            Figure size
        savefig_prefix : str
            Prefix for saved figure
        channel : str
            'red' or 'grn' - which channel to use (default: 'grn')
        """
        rec = self._get_rec(channel)

        n_frames_pre = int(t_pre * self.samp_rate)
        n_frames_post = int(t_post * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post

        t_lickaligned = np.linspace(
            -1*t_pre, t_post, n_frames_tot)

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
        ax.set_xlabel('time from lick (s)')
        ax.set_ylabel('dff')
        ax.axvline(x=0, color=sns.xkcd_rgb['black'],
                   linestyle='dashed',
                   linewidth=0.8)

        fig.savefig(f'{savefig_prefix}_{channel}_lick_resp_{t_pre=}_{t_post=}.pdf')
        plt.show()

    def correct_signal(self, method='linear', save_to_disk=False,
                       t_start=None, t_end=None, t_end_pad=10.0,
                       replace_grn=True, detrend=False,
                       **kwargs):
        """
        Correct green channel using red channel as reference.

        Uses the red channel (control signal) to remove shared noise sources
        from the green channel (target signal). Multiple correction methods
        are available depending on the noise characteristics.

        Memory-optimized: works directly with memory-mapped arrays, processing
        data in chunks to minimize RAM usage.

        Parameters
        ----------
        method : str
            Correction method to use:
            - 'linear': Pixel-wise OLS linear regression (default)
            - 'robust': Pixel-wise robust regression with Huber loss
            - 'lms': LMS adaptive filter per pixel
            - 'pca': PCA-based shared variance removal
            - 'ica': ICA-based shared component removal
            - 'nmf': NMF-based correction (for non-negative signals)
        save_to_disk : bool
            If True, write corrected signal directly to disk as a memory-
            mapped TIFF file instead of storing in RAM. The output file will
            be saved in the same directory as the green channel image with
            '_corr' suffix, preserving the original extension (e.g.,
            'image_grn.tif' -> 'image_grn_corr.tif'). This significantly
            reduces memory usage for large movies. (default: False)
        replace_grn : bool
            If True, replace self.rec_grn with the corrected signal after
            correction. The very first time this is done the original memmap
            is saved as self.rec_grn_original. On all subsequent calls
            (regardless of whether replace_grn is True or False) the source
            signal is always taken from self.rec_grn_original, so each call
            starts from the same raw data rather than the previously corrected
            signal. (default: False)
        t_start : float, optional
            Starting time in seconds. The closest frame at or after this time
            will be used. If None, starts from the first frame.
            (default: None)
        t_end : float, optional
            Ending time in seconds. The closest frame at or before this time
            will be used. If None and self.trial_end is set, t_end is
            inferred automatically as the reward time of self.trial_end plus
            t_end_pad seconds. If both are None, processes to the last frame.
            (default: None)
        t_end_pad : float, optional
            Padding in seconds added after the reward time of self.trial_end
            when t_end is inferred automatically. Ignored if t_end is
            provided explicitly. (default: 10.0)
        detrend : bool, optional
            If True, remove a per-pixel linear trend from both s1 (red) and
            s2 (green) before passing them to the correction function.
            Detrending is done via two streaming temporal passes (accumulate
            slope/intercept analytically, then subtract) so the full array is
            never loaded into RAM at once.  The detrended arrays are written
            to temporary raw memmap files in self.folder.img and deleted
            automatically after correction completes, keeping RAM usage to
            roughly one batch at a time rather than 2 × T × X × Y × 4 bytes.
            Detrending removes slow baseline drift that can otherwise dominate
            component-based methods (NMF, PCA, ICA) and impair their ability
            to isolate shared noise. For regression-based methods (linear,
            robust, lms) it ensures the regression is not distorted by
            differing drift rates between channels. (default: True)
        **kwargs : dict
            Additional arguments passed to the correction function.

            Common parameters (all methods):
                verbose : bool (default: True)
                    Print progress updates during correction
                dtype : np.dtype (default: np.int16)
                    Output data type

            For 'linear' and 'robust':
                huber_epsilon : float (default: 1.35, robust only)
                max_iter : int (default: 50, robust only)
                tol : float (default: 1e-4, robust only)

            For 'lms':
                filter_order : int (default: 10)
                mu : float (default: 0.01)
                normalized : bool (default: True)

            For 'pca', 'ica', 'nmf':
                n_components : int or 'auto'/None (default: 'auto' or None)
                s1_loading_threshold : float (default: 0.5)
                spatial_subsample : int (default: 4)
                    Subsample every N pixels for fitting (saves memory)
                batch_size : int (default: 500)
                    Time frames to process at once
                max_iter : int (default: 200)
                random_state : int or None (default: None)

        Returns
        -------
        corrected : np.ndarray or np.memmap
            Corrected green channel signal, shape (T, X, Y) or subset shape
            if t_start/t_end specified. If save_to_disk is True, returns a
            memory-mapped array pointing to the file.

        Examples
        --------
        >>> # Basic linear correction
        >>> corrected = rec.correct_signal(method='linear')

        >>> # Robust correction for data with outliers
        >>> corrected = rec.correct_signal(method='robust', huber_epsilon=1.5)

        >>> # Adaptive filter correction
        >>> corrected = rec.correct_signal(method='lms', filter_order=20, mu=0.005)

        >>> # PCA-based correction with memory optimization
        >>> corrected = rec.correct_signal(method='pca', spatial_subsample=8)

        >>> # Save corrected signal directly to disk (minimal RAM usage)
        >>> corrected = rec.correct_signal(method='linear', save_to_disk=True)

        >>> # Process only first 60 seconds (for testing)
        >>> corrected = rec.correct_signal(method='linear', t_end=60.0)

        >>> # Process from 30s to 90s
        >>> corrected = rec.correct_signal(
        ...     method='linear', t_start=30.0, t_end=90.0)
        """
        self.corr_sig_method = method

        # Infer t_end from trial_end if not explicitly supplied
        if t_end is None and getattr(self, 'trial_end', None) is not None:
            _rew_times = self.beh._data.get_event_var('totalRewardTimes')
            t_end = _rew_times[self.trial_end] + t_end_pad
            print(f'\tt_end inferred from trial_end={self.trial_end}: '
                  f'{t_end:.2f}s (reward time + {t_end_pad}s pad)')

        # Get full signals
        s1_full = self.rec_red  # Red channel = control (memmap)
        # If a previous correction has already replaced rec_grn, use the
        # preserved original so every call starts from the same raw signal.
        s2_full = getattr(self, 'rec_grn_original', self.rec_grn)
        n_frames = s1_full.shape[0]

        # Apply temporal slicing if specified (convert time to frame indices)
        if t_start is not None or t_end is not None:
            if not hasattr(self, 'rec_t'):
                raise ValueError(
                    "Cannot use t_start/t_end: timestamps (rec_t) not found. "
                    "Ensure recording was loaded with timestamp alignment."
                )

            # Find closest frame indices for the specified times
            if t_start is not None:
                # Find first frame at or after t_start
                idx_start = np.searchsorted(self.rec_t, t_start, side='left')
                idx_start = min(idx_start, n_frames - 1)
            else:
                idx_start = 0

            if t_end is not None:
                # Find last frame at or before t_end
                idx_end = np.searchsorted(self.rec_t, t_end, side='right')
                idx_end = min(idx_end, n_frames)
            else:
                idx_end = n_frames

            s1 = s1_full[idx_start:idx_end, :, :]
            s2 = s2_full[idx_start:idx_end, :, :]

            if kwargs.get('verbose', True):
                actual_t_start = self.rec_t[idx_start]
                actual_t_end = self.rec_t[idx_end - 1]
                print(f'\tprocessing frames {idx_start} to {idx_end} '
                      f'(t={actual_t_start:.2f}s to {actual_t_end:.2f}s, '
                      f'{idx_end - idx_start} frames)')
        else:
            s1 = s1_full
            s2 = s2_full

        # Linearly detrend both channels before correction if requested.
        # This removes slow baseline drift so the correction method focuses
        # on shared noise (motion, haemodynamics) rather than trend differences.
        # Detrended arrays are written to disk as raw numpy memmaps to avoid
        # holding 2 × T × X × Y × 4 bytes in RAM.
        _detrend_tmp_files = []
        if detrend:
            _verb = kwargs.get('verbose', True)
            _batch = kwargs.get('batch_size', 500)
            _uid = os.urandom(4).hex()
            _tmp_dir = self.folder.img
            _s1_tmp = os.path.join(_tmp_dir, f'_detrend_s1_tmp_{_uid}.dat')
            _s2_tmp = os.path.join(_tmp_dir, f'_detrend_s2_tmp_{_uid}.dat')
            _detrend_tmp_files = [_s1_tmp, _s2_tmp]
            if _verb:
                print('\tlinearly detrending s1 (red) -> disk...')
            s1 = signal_correction.detrend_linearly(
                s1, batch_size=_batch, verbose=_verb, output_path=_s1_tmp)
            if _verb:
                print('\tlinearly detrending s2 (green) -> disk...')
            s2 = signal_correction.detrend_linearly(
                s2, batch_size=_batch, verbose=_verb, output_path=_s2_tmp)

        # Construct output path if saving to disk
        output_path = None
        if save_to_disk:
            # Get base filename and preserve original extension
            base_name, ext = os.path.splitext(self.fname_img_grn)
            output_fname = f"{base_name}_corr{ext}"
            output_path = os.path.join(self.folder.img, output_fname)
            kwargs['output_path'] = output_path

        method_map = {
            'linear': lambda: signal_correction.correct_linear_regression(
                s1, s2, robust=False, **kwargs),
            'robust': lambda: signal_correction.correct_linear_regression(
                s1, s2, robust=True, **kwargs),
            'lms': lambda: signal_correction.correct_lms_adaptive(
                s1, s2, **kwargs),
            'pca': lambda: signal_correction.correct_pca_shared_variance(
                s1, s2, **kwargs),
            'ica': lambda: signal_correction.correct_ica_shared_components(
                s1, s2, **kwargs),
            'nmf': lambda: signal_correction.correct_nmf_shared_components(
                s1, s2, **kwargs),
        }

        if method not in method_map:
            valid_methods = list(method_map.keys())
            raise ValueError(
                f"Unknown method '{method}'. Valid methods: {valid_methods}")

        # Call the correction function
        result = method_map[method]()

        # All correction functions return corrected signal as first element
        corrected = result[0]

        # Remove temporary detrend files now that correction is complete
        for _tmp in _detrend_tmp_files:
            if os.path.exists(_tmp):
                try:
                    os.remove(_tmp)
                except OSError:
                    pass

        self.rec_grn_corr = corrected

        if replace_grn:
            # Preserve the original green channel the first time we replace it
            if not hasattr(self, 'rec_grn_original'):
                self.rec_grn_original = self.rec_grn
                # Also preserve the raw memmap for efficient batch access
                if hasattr(self, '_rec_grn_raw'):
                    self._rec_grn_original_raw = self._rec_grn_raw
            self.rec_grn = corrected
            # Store sliced timestamps if time slicing was applied
            if t_start is not None or t_end is not None:
                self.rec_t_grn = self.rec_t[idx_start:idx_end]
            else:
                # No slicing, timestamps match the full recording
                self.rec_t_grn = self.rec_t

    def save_corr_signal(self, fname=None, chunk_size=100):
        """
        Save self.rec_grn_corr to disk as a TIFF file using chunked writes
        and memory mapping to minimise RAM usage.

        Must be called after correct_signal() has been run. The output file
        is written frame-by-frame in chunks using tifffile.TiffWriter, matching
        the approach used in stitch_tiffs_memmap().

        Parameters
        ----------
        fname : str, optional
            Output filename (without path). Written to the current working
            directory. If None, the filename is constructed automatically from
            self.fname_img_grn and self.corr_sig_method:
            e.g. 'compiled_Ch2_corr_linear.tif'
        chunk_size : int, optional
            Number of frames to read from self.rec_grn_corr per iteration.
            Larger values are faster but use more RAM. (default: 100)

        Returns
        -------
        str
            Absolute path to the saved file.
        """
        if not hasattr(self, 'rec_grn_corr'):
            raise RuntimeError(
                "rec_grn_corr not found. Run correct_signal() first.")

        if fname is None:
            base_name, ext = os.path.splitext(self.fname_img_grn)
            method_tag = getattr(self, 'corr_sig_method', 'corr')
            fname = f"{base_name}_corr_{method_tag}{ext}"

        output_path = os.path.join(os.getcwd(), self.folder.img, fname)
        n_frames = self.rec_grn_corr.shape[0]

        print(f'saving corrected signal to {output_path}...')
        with tifffile.TiffWriter(output_path, bigtiff=True) as writer:
            for idx_start in range(0, n_frames, chunk_size):
                idx_end = min(idx_start + chunk_size, n_frames)
                print(f'\tframes {idx_start}-{idx_end}/{n_frames}', end='\r')
                chunk = np.array(self.rec_grn_corr[idx_start:idx_end])
                for frame in chunk:
                    writer.write(frame, compression=None, contiguous=True)
        print(f'\ndone.')

        return output_path
