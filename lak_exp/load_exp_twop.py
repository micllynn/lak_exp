from types import SimpleNamespace
import numpy as np
import os
import pathlib
import tifffile
import warnings

from .align_imgbeh import Aligner_ImgBeh
from .utils import (find_event_onsets_autothresh,
                    find_event_onsets_plateau)
from .utils_twop import XMLParser
from .beh import BehDataSimpleLoad, StimParserNew
from .exp_defs import ExpSubtypes
from . import signal_correction
from .twop_analysis import AnalysisMixin
from .twop_plots import PlotsMixin
from .twop_qc import QCMixin

# Suppress NumPy RuntimeWarnings in this module.
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module=r"^numpy(\.|$)")


class TwoPRec(AnalysisMixin, PlotsMixin, QCMixin):
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
                 beh_type='visual_pavlov',
                 parse_by=None,
                 subtypes=None,
                 trial_conditions=None,
                 n_px_remove_sides=10,
                 lick_type='noise',
                 lick_sin_v=5,
                 lick_sin_tol=0.02,
                 lick_sin_t_plateau=0.05):
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
        beh_type : str or None
            Preset that drives parse_by and subtypes automatically.
            'visual_pavlov' (default): parse_by='stimulusOrientation',
                subtypes = ExpSubtypes('visual_pavlov').get() (conditions
                '0', '0.5', '1', '0.5_rew', '0.5_norew',
                '0.5_prelick', '0.5_noprelick').
            'auditory_pavlov': parse_by='stimulusType', no subtypes.
            None: parse_by and subtypes must be provided explicitly.
        parse_by : str or None
            Override for the StimParserNew parse_by argument. When beh_type is
            set this defaults to None and is resolved from beh_type; pass
            explicitly to override the preset.
        subtypes : dict or None
            Override for the StimParserNew subtypes argument. When beh_type is
            set this defaults to None and is resolved from beh_type; pass
            explicitly to override or extend the preset.
        n_px_remove_sides : int
            Number of pixels to remove on each side of the frame (necessary if
            dealing with suite2p motion corrected tiffs, as these can
            introduce artifacts with high values on the edges
        lick_type : str
            'normal' (lick data recorded correctly) or 'noise' (grounding
            issue on Dual2P; signal closely resembles a sin wave with brief
            plateau periods at lick_sin_v that correspond to licks).
            Default 'noise' (the rig used for these recordings).
        lick_sin_v : float
            Plateau voltage for 'noise'-mode lick detection. Default 5.
        lick_sin_tol : float
            Voltage tolerance around lick_sin_v. Default 0.02.
        lick_sin_t_plateau : float
            Minimum plateau duration (s). Default 0.05.

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
        self.ch_img = ch_img
        self.trial_end = trial_end

        self._init_folders(enclosing_folder, folder_beh, folder_img,
                           dset_obj, dset_ind)
        self._init_ops(n_px_remove_sides, rec_type)

        # resolve parse_by and subtypes from beh_type preset
        # -------------
        self.beh_type = beh_type
        if beh_type == 'visual_pavlov':
            if parse_by is None:
                parse_by = 'stimulusOrientation'
            if subtypes is None:
                subtypes = ExpSubtypes(beh_type).get()
        elif beh_type == 'auditory_pavlov':
            if parse_by is None:
                parse_by = 'stimulusType'
            if subtypes is None:
                subtypes = ExpSubtypes(beh_type).get()
        elif beh_type is None:
            if parse_by is None:
                raise ValueError(
                    "beh_type=None requires an explicit parse_by argument")

        # load behavioral data
        # -------------
        self._init_behavior(parse_stims=parse_stims, parse_by=parse_by,
                            subtypes=subtypes,
                            lick_type=lick_type,
                            lick_sin_v=lick_sin_v,
                            lick_sin_tol=lick_sin_tol,
                            lick_sin_t_plateau=lick_sin_t_plateau)

        # load imaging data (single or dual-colour)
        # ------------
        self._init_recording(fname_img=fname_img, dset_obj=dset_obj,
                             n_px_remove_sides=n_px_remove_sides)

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
        self._init_trial_conditions(
            use_int_cast=getattr(self, '_trial_cond_use_int_cast', True))
        return

    # ------------------------------------------------------------------
    # Shared __init__ helpers (called by both TwoPRec and
    # TwoPRec_DualColour.__init__)
    # ------------------------------------------------------------------

    def _init_folders(self, enclosing_folder, folder_beh, folder_img,
                      dset_obj, dset_ind):
        """Set up self.folder and self.path from explicit paths or a dataset
        object. All sub-paths are resolved to absolute paths relative to the
        enclosing folder. Creates figs_mbl/ if it does not exist."""
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

        # Make all folder paths absolute, relative to enclosing
        _enc = pathlib.Path(self.folder.enclosing)
        self.folder.img = str(_enc / self.folder.img)
        self.folder.beh = str(_enc / self.folder.beh)
        self.folder.figs = _enc / 'figs_mbl'
        self.folder.figs.mkdir(exist_ok=True)

        self.path.raw = _enc
        self.path.animal = _enc.parts[-2]
        self.path.date = _enc.parts[-1]
        self.path.beh_folder = folder_beh  # keep original relative name

    def _init_ops(self, n_px_remove_sides, rec_type):
        """Initialise self.ops namespace with operational parameters."""
        self.ops = SimpleNamespace()
        self.ops.n_px_remove_sides = n_px_remove_sides
        self.ops.rec_type = rec_type

    def _init_behavior(self, parse_stims=True, parse_by='stimulusOrientation',
                       subtypes=None,
                       lick_type='noise',
                       lick_sin_v=5,
                       lick_sin_tol=0.02,
                       lick_sin_t_plateau=0.05):
        """Load behavioral data into self.beh from self.folder.beh.

        Populates self.beh.rew, self.beh.stim, and self.beh.lick.
        lick_type selects between 'normal' (autothresh) and 'noise'
        (plateau detection — Dual2P grounding noise) lick extraction;
        see VisualPavlovAnalysis in plt_beh.py for the same convention.
        """
        self.beh = BehDataSimpleLoad(self.folder.beh,
                                     parse_stims=parse_stims,
                                     parse_by=parse_by)

        self.beh.rew = SimpleNamespace()
        self.beh.stim = SimpleNamespace()

        self.beh.rew.delivered = self.beh._data.get_event_var(
            'isRewardGivenValues')
        # Rigbox occasionally writes one more entry into
        # isRewardGivenValues than there are totalRewardTimes (e.g. when
        # the session is cut before the trailing reward time is logged).
        # Mask the trailing delivered indices that exceed the available
        # totalRewardTimes so indexing is always safe.
        _t_rew_all = self.beh._data.get_event_var('totalRewardTimes')
        _delivered_inds = np.where(self.beh.rew.delivered == 1)[0]
        _n_valid = len(_t_rew_all)
        _trailing = _delivered_inds[_delivered_inds >= _n_valid]
        if len(_trailing) > 0:
            print(f'\tmasking {len(_trailing)} reward delivery indices '
                  f'beyond totalRewardTimes ({_n_valid}); likely an '
                  f'incomplete session log.')
            _delivered_inds = _delivered_inds[_delivered_inds < _n_valid]
        self.beh.rew.t = _t_rew_all[_delivered_inds]

        _daq_data = self.beh._timeline.get_daq_data()

        try:
            _lick_sig = _daq_data.sig['lickDetector']
            if lick_type == 'normal':
                _licks = find_event_onsets_autothresh(_lick_sig)
            elif lick_type == 'noise':
                _licks = find_event_onsets_plateau(
                    _lick_sig, _daq_data.t,
                    v=lick_sin_v, tol=lick_sin_tol,
                    t_thresh=lick_sin_t_plateau)
            else:
                raise ValueError(
                    f"lick_type must be 'normal' or 'noise', "
                    f"got {lick_type!r}")
            _licks = np.asarray(_licks).astype(np.int64).ravel()
            self.beh.licks = _licks
            self.beh.t_licks = _daq_data.t[_licks]
        except Exception as _e:
            print(f'\tlick extraction failed ({type(_e).__name__}: {_e}); '
                  f'skipping licks')
            self.beh.licks = np.array([], dtype=np.int64)
            self.beh.t_licks = np.array([], dtype=np.float64)

        self.beh.stim.t_start = self.beh._data.get_event_var(
            'stimulusOnTimes')
        self.beh.stim.stimlist = StimParserNew(self.beh, parse_by=parse_by,
                                               subtypes=subtypes)
        self.beh._stimparser = self.beh.stim.stimlist

        self.beh.stim.prob = self.beh.stim.stimlist._all_stimprobs
        self.beh.stim.size = self.beh.stim.stimlist._all_stimsizes

        self.beh.lick = SimpleNamespace()

        try:
            t_licksig = self.beh._daq_data.t
            if lick_type == 'normal':
                lick_onset_inds = find_event_onsets_autothresh(
                    self.beh._daq_data.sig['lickDetector'], n_stdevs=4)
            else:
                lick_onset_inds = find_event_onsets_plateau(
                    self.beh._daq_data.sig['lickDetector'],
                    self.beh._daq_data.t,
                    v=lick_sin_v, tol=lick_sin_tol,
                    t_thresh=lick_sin_t_plateau)
            lick_onset_inds = np.asarray(
                lick_onset_inds).astype(np.int64).ravel()
            self.beh.lick.t_raw = t_licksig[lick_onset_inds]
        except Exception as _e:
            print(f'\tlick.t_raw extraction failed '
                  f'({type(_e).__name__}: {_e}); skipping')
            self.beh.lick.t_raw = np.array([], dtype=np.float64)

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

    # ----------------------
    # internal methods called by user-facing class methods
    # ----------------------
    def _get_rec(self, channel=None):
        """
        Helper method to get the appropriate recording based on channel.

        Parameters
        ----------
        channel : str or None
            'red' or 'grn' (default: None -> self.rec if present).

        Returns
        -------
        np.ndarray
            The recording array for the specified channel.
        """
        if channel is None:
            return self.rec
        if channel == 'red' and hasattr(self, 'rec_red'):
            return self.rec_red
        if channel == 'grn' and hasattr(self, 'rec_grn'):
            return self.rec_grn
        if channel in ('red', 'grn'):
            return self.rec
        raise ValueError(f"channel must be 'red' or 'grn', got '{channel}'")

    def _get_rec_t(self, channel=None):
        """
        Helper method to get the appropriate timestamps based on channel.

        Parameters
        ----------
        channel : str or None
            'red' or 'grn' (default: None -> self.rec_t).

        Returns
        -------
        np.ndarray
            The timestamp array for the specified channel.
        """
        if channel is None:
            return self.rec_t
        if channel == 'red':
            return self.rec_t
        if channel == 'grn':
            if hasattr(self, 'rec_t_grn'):
                return self.rec_t_grn
            return self.rec_t
        raise ValueError(f"channel must be 'red' or 'grn', got '{channel}'")

    def _load_sampling_rate_from_backup_xml(self):
        """Parse sampling rate from BACKUP.xml in the imaging folder."""
        sampling_rate = None
        for _file in os.listdir(self.folder.img):
            if _file.endswith('BACKUP.xml'):
                try:
                    xmlobj = XMLParser(os.path.join(self.folder.img, _file))
                    sampling_rate = xmlobj.get_framerate()
                    print(f'\tframerate is {sampling_rate:.2f}Hz')
                except Exception:
                    print('could not parse framerate from BACKUP.xml')
        return sampling_rate

    def _init_recording(self, fname_img=None, dset_obj=None,
                        n_px_remove_sides=10, **kwargs):
        """Load imaging data for a single-channel recording."""
        # get filename of image of the appropriate channel
        list_img = os.listdir(self.folder.img)
        self.fname_img = self._resolve_channel_filename(
            list_img, self.ch_img, fname_img=fname_img,
            dset_obj=dset_obj, use_dataset_default=False)

        print('loading imaging...')
        self.samp_rate = self._load_sampling_rate_from_backup_xml()

        print('\tloading tiff...')
        _raw = tifffile.memmap(os.path.join(self.folder.img, self.fname_img))
        if _raw.ndim == 4:
            _raw = _raw.reshape(-1, _raw.shape[-2], _raw.shape[-1])
        self.rec = _raw[:, n_px_remove_sides:-1*n_px_remove_sides,
                           n_px_remove_sides:-1*n_px_remove_sides]

    def _resolve_channel_filename(self, list_img, ch_img,
                                  fname_img=None,
                                  dset_obj=None,
                                  use_dataset_default=False):
        """Resolve tiff filename for a channel
        with optional dataset defaults."""
        if fname_img is not None:
            return fname_img
        if use_dataset_default and dset_obj is not None:
            return f'compiled_Ch{ch_img}.tif'
        for _fname in list_img:
            if f'Ch{ch_img}.tif' in _fname and 'compiled' in _fname:
                return _fname
        raise FileNotFoundError(
            f"No compiled Ch{ch_img} tiff found in imaging folder. "
            f"Available files: {list_img}")

    def _filter_trial_inds_to_stimrange(self, tr_inds):
        """Filter trial indices to the current stim range."""
        _valid = np.arange(self.beh._stimrange.first,
                           self.beh._stimrange.last + 1)
        return np.delete(tr_inds, ~np.isin(tr_inds, _valid))

    def _init_trial_conditions(self, use_int_cast=False):
        """Initialize trial conditions, trial indices, and _trial_cond_map.

        For beh_type='visual_pavlov', all conditions are provided by the
        StimParserNew subtypes (ExpSubtypes('visual_pavlov').get()); the parsed_param
        (orientation) loop is skipped and tr_conds is the ordered subtype keys.

        For all other beh_types, the generic path builds tr_inds from
        parsed_param values and appends any subtypes on top.
        """
        self.beh.tr_inds = {}
        _sp = self.beh._stimparser

        if getattr(self, 'beh_type', None) == 'visual_pavlov':
            # All conditions are expressed as subtypes; skip the
            # parsed_param (orientation) loop.
            self.beh.tr_conds = list(_sp.subtype_tr_inds.keys())
            for _label, _inds in _sp.subtype_tr_inds.items():
                self.beh.tr_inds[_label] = \
                    self._filter_trial_inds_to_stimrange(_inds)

            # If anticipatoryLickValues was absent from the block file the
            # prelick subtypes will be empty — fall back to add_lickrates().
            if (not len(self.beh.tr_inds.get('0.5_prelick', []))
                    and not len(self.beh.tr_inds.get('0.5_noprelick', []))):
                self.add_lickrates()
                self.beh.tr_inds['0.5_prelick'] = \
                    self._filter_trial_inds_to_stimrange(np.where(
                        np.logical_and(self.beh.stim.prob == 0.5,
                                       self.beh.lick.antic_raw > 0))[0])
                self.beh.tr_inds['0.5_noprelick'] = \
                    self._filter_trial_inds_to_stimrange(np.where(
                        np.logical_and(self.beh.stim.prob == 0.5,
                                       self.beh.lick.antic_raw == 0))[0])
                for _lk in ('0.5_prelick', '0.5_noprelick'):
                    if _lk not in self.beh.tr_conds:
                        self.beh.tr_conds.append(_lk)
        else:
            # Generic path: build tr_inds from parsed_param values.
            self.beh.tr_conds = list(_sp.parsed_param.astype(str))
            self.beh.tr_conds_outcome = _sp.prob * _sp.size

            for param_val in _sp.parsed_param:
                match_val = int(param_val) if use_int_cast else param_val
                self.beh.tr_inds[str(param_val)] = np.where(
                    _sp._all_parsed_param == match_val)[0]

            for tr_cond in self.beh.tr_conds:
                self.beh.tr_inds[tr_cond] = \
                    self._filter_trial_inds_to_stimrange(
                        self.beh.tr_inds[tr_cond])

            self.add_lickrates()

            # Append any subtypes defined in StimParserNew.
            if hasattr(_sp, 'subtype_tr_inds') and _sp.subtype_tr_inds:
                for _label, _inds in _sp.subtype_tr_inds.items():
                    self.beh.tr_inds[_label] = \
                        self._filter_trial_inds_to_stimrange(_inds)

        self._trial_cond_map = {}
        for _cond, _inds in self.beh.tr_inds.items():
            for _idx in _inds:
                self._trial_cond_map.setdefault(int(_idx), []).append(_cond)

    def _aligned_frame_params(self, t_pre, t_post, t_event=2):
        """Compute frame counts and time vector for aligned traces."""
        n_frames_pre = int(t_pre * self.samp_rate)
        n_frames_post = int((t_event + t_post) * self.samp_rate)
        n_frames_tot = n_frames_pre + n_frames_post
        t_vec = np.linspace(-1*t_pre, t_event + t_post, n_frames_tot)
        return n_frames_pre, n_frames_post, n_frames_tot, t_vec

    def _init_dff_by_cond(self, tr_conds, n_frames_tot):
        """Initialize a dict of dff arrays keyed by trial condition."""
        return {
            _tr_cond: np.zeros((
                self.beh.tr_inds[_tr_cond].shape[0], n_frames_tot))
            for _tr_cond in tr_conds
        }

    def _bin_trials_by_iti(self, itis, n_time_divs):
        """Bin trials by ITI and return bin edges and masks."""
        _count, t_bin_edges = np.histogram(itis, bins=n_time_divs)
        masks = []
        for ind_timediv in range(n_time_divs):
            masks.append(np.logical_and(
                itis > t_bin_edges[ind_timediv],
                itis < t_bin_edges[ind_timediv+1]))
        return t_bin_edges, masks


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
                 beh_type='visual_pavlov',
                 parse_by=None,
                 subtypes=None,
                 n_px_remove_sides=10,
                 lick_type='noise',
                 lick_sin_v=5,
                 lick_sin_tol=0.02,
                 lick_sin_t_plateau=0.05):
        """
        Loads a dual-colour 2p recording (two tiffs) and associated
        behavioral folder.

        See TwoPRec docstring for full parameter descriptions.
        This class loads two imaging channels:
        - Red channel (typically Ch1) -> self.rec_red
        - Green channel (typically Ch2) -> self.rec_grn
        """
        self.ch_img_red = ch_img_red
        self.ch_img_grn = ch_img_grn
        self._fname_img_red = fname_img_red
        self._fname_img_grn = fname_img_grn
        self._trial_cond_use_int_cast = False

        super().__init__(enclosing_folder=enclosing_folder,
                         folder_beh=folder_beh,
                         folder_img=folder_img,
                         fname_img=fname_img_grn,
                         dset_obj=dset_obj,
                         dset_ind=dset_ind,
                         ch_img=ch_img_grn,
                         trial_end=trial_end,
                         rec_type=rec_type,
                         parse_stims=False,
                         beh_type=beh_type,
                         parse_by=parse_by,
                         subtypes=subtypes,
                         n_px_remove_sides=n_px_remove_sides,
                         lick_type=lick_type,
                         lick_sin_v=lick_sin_v,
                         lick_sin_tol=lick_sin_tol,
                         lick_sin_t_plateau=lick_sin_t_plateau)
        return

    def _init_recording(self, fname_img=None, dset_obj=None,
                        n_px_remove_sides=10, **kwargs):
        """Load imaging data for a dual-colour recording."""
        # get filenames of images for both channels
        list_img = os.listdir(self.folder.img)
        fname_img_red = getattr(self, '_fname_img_red', None)
        fname_img_grn = (fname_img if fname_img is not None
                         else getattr(self, '_fname_img_grn', None))

        # Red channel
        self.fname_img_red = self._resolve_channel_filename(
            list_img, self.ch_img_red, fname_img=fname_img_red,
            dset_obj=dset_obj, use_dataset_default=True)

        # Green channel
        self.fname_img_grn = self._resolve_channel_filename(
            list_img, self.ch_img_grn, fname_img=fname_img_grn,
            dset_obj=dset_obj, use_dataset_default=True)

        # Keep fname_img for backwards compatibility (defaults to green)
        self.fname_img = self.fname_img_grn

        print('loading imaging...')
        self.samp_rate = self._load_sampling_rate_from_backup_xml()

        # Load RED channel (keep raw memmap, crop lazily via _get_rec)
        # Storing the raw memmap avoids creating a non-contiguous view that
        # forces full-array RAM copies when batches are read.
        print('\tloading red channel tiff...')
        self._rec_red_raw = tifffile.memmap(os.path.join(
            self.folder.img, self.fname_img_red))
        if self._rec_red_raw.ndim == 4:
            self._rec_red_raw = self._rec_red_raw.reshape(
                -1, self._rec_red_raw.shape[-2], self._rec_red_raw.shape[-1])

        # Load GREEN channel (keep raw memmap, crop lazily via _get_rec)
        print('\tloading green channel tiff...')
        self._rec_grn_raw = tifffile.memmap(os.path.join(
            self.folder.img, self.fname_img_grn))
        if self._rec_grn_raw.ndim == 4:
            self._rec_grn_raw = self._rec_grn_raw.reshape(
                -1, self._rec_grn_raw.shape[-2], self._rec_grn_raw.shape[-1])

        # Store crop bounds for lazy application
        self._crop_px = n_px_remove_sides

        # Create cropped views for backward compatibility (used for .shape etc)
        # These are views, not copies, until data is actually accessed
        c = n_px_remove_sides
        self.rec_red = self._rec_red_raw[:, c:-c, c:-c] \
            if c > 0 else self._rec_red_raw
        self.rec_grn = self._rec_grn_raw[:, c:-c, c:-c] \
            if c > 0 else self._rec_grn_raw

        # Keep self.rec for backwards compatibility (defaults to green)
        self.rec = self.rec_grn

    def _get_rec_raw(self, channel='grn'):
        """
        Helper method to get the raw (uncropped) memmap
        for efficient batch I/O.

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
            if hasattr(self, '_rec_red_original_raw'):
                return self._rec_red_original_raw, self._crop_px
            return self._rec_red_raw, self._crop_px
        elif channel == 'grn':
            if hasattr(self, '_rec_grn_original_raw'):
                return self._rec_grn_original_raw, self._crop_px
            return self._rec_grn_raw, self._crop_px
        else:
            raise ValueError(
                f"channel must be 'red' or 'grn', got '{channel}'")

    # --------------
    # add_frame, plt_frame, add_sectors, plt_sectors, add_null_dists,
    # add_zetatest_sectors, plt_null_dist, and plt_lick_resp are inherited
    # from TwoPRec which now supports channel, use_zscore, _get_rec/_get_rec_t,
    # and _trial_cond_map.
    # ---------------

    def correct_signal(self, method='linear', save_to_disk=False,
                       t_start=None, t_end=None, t_end_pad=10.0,
                       replace_real=True, detrend=False,
                       static_flu='grn', real_flu='red',
                       marker_size=25,
                       trialavg_t_pre=2.0, trialavg_t_post=4.0,
                       **kwargs):
        """
        Correct the real-signal channel using the static (control) channel.

        By default uses the green channel (static_flu='grn') to remove shared
        noise from the red channel (real_flu='red'). Swap the defaults to
        correct green using red instead.

        Memory-optimized: works directly with memory-mapped arrays, processing
        data in chunks to minimize RAM usage.

        Parameters
        ----------
        method : str
            Correction method to use:
            - 'linear': Pixel-wise OLS linear regression (default).
              Output is the residual (real − fitted control).
            - 'robust': Pixel-wise robust regression with Huber loss.
            - 'linear_photom': Photometric-style pixel-wise correction
              loosely inspired by Martianova, Aronson & Proulx, Sci.
              Rep. 2021. Fits the control channel to the real channel
              by per-pixel OLS, computes dF/F = (real − F̂) / F̂ using
              the fitted control as F0, and z-scores the result per
              pixel across time. Output is float16. Lacks lowpass /
              airPLS / per-channel normalisation / non-negative slope —
              for the faithful port see 'linear_martianova' below.
            - 'linear_martianova': Faithful port of the full Martianova
              et al. (2021) photometric pipeline. Operates on the
              spatially-averaged 1-D control & signal traces: moving-
              average lowpass → airPLS baseline removal → trim warm-up
              frames → per-channel median/std normalisation → non-
              negative OLS slope → per-pixel intercept anchored to each
              pixel's session mean. Per-voxel ΔF/F uses the global
              baseline-subtracted control and is z-scored per pixel
              across time. Output is float16. See
              signal_correction.correct_linear_martianova for the full
              parameter list (smooth_window, airpls_lam, airpls_porder,
              airpls_max_iter, trim_initial, nn_slope,
              per_pixel_offset).
            - 'lms': LMS adaptive filter per pixel
            - 'pca': PCA-based shared variance removal
            - 'ica': ICA-based shared component removal
            - 'nmf': NMF-based correction (for non-negative signals)
        static_flu : str
            Channel used as the static control reference signal. Must be
            'red' or 'grn'. (default: 'grn')
        real_flu : str
            Channel containing the real signal to be corrected. Must be
            'red' or 'grn'. (default: 'red')
        save_to_disk : bool
            If True, the correction streams its full corrected stack
            into `{base_real}_corr.tif` via tifffile.memmap (no full-
            volume RAM allocation), THEN derives three small trial-
            averaged stim-aligned TIFFs by reading the relevant
            source one trial-window at a time into a (n_win, X, Y)
            float32 accumulator:

            - `{base_real}_corr_trialavg.tif` — corrected stack in
              its native units (typically z-scored ΔF/F).
            - `{base_real}_dff_trialavg.tif`  — raw real channel
              converted to dF/F (%) using each pixel's mean over the
              pre-stim baseline window.
            - `{base_static}_dff_trialavg.tif` — same for the raw
              static channel.

            All three share the trial-avg window
            [−trialavg_t_pre, median(stim→rew) + trialavg_t_post]
            relative to each stim onset and the same white marker
            squares (see `marker_size`), so they can be overlaid
            frame-for-frame in an image viewer. The full corrected
            stack file is kept: `self.rec_{real}_corr` (and `rec_
            {real}` if replace_real=True) points to it so downstream
            QC / sector extraction still works against the disk-
            backed corrected stack. (default: False)
        replace_real : bool
            If True, replace self.rec_{real_flu} with the corrected signal
            after correction. The very first time this is done the original
            memmap is saved as self.rec_{real_flu}_original. On all subsequent
            calls the source signal is always taken from the preserved
            original so each call starts from the same raw data.
            (default: True)
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
        marker_size : int
            Edge length (in pixels) of the white marker squares stamped
            onto the trial-averaged saved file when save_to_disk=True.
            Top-left square marks the stim-onset frame; top-right
            square marks the reward-onset frame (median latency). Both
            are dtype-max (white). Set 0 to disable. Ignored when
            save_to_disk=False. (default: 25)
        trialavg_t_pre : float
            Seconds before stim onset included in the trial-averaging
            window. (default: 2.0)
        trialavg_t_post : float
            Seconds after the median reward time included in the
            trial-averaging window. (default: 4.0)
        detrend : bool, optional
            If True, remove a per-pixel linear trend from both channels before
            passing them to the correction function. Detrending is done via
            two streaming temporal passes so the full array is never loaded
            into RAM at once. The detrended arrays are written to temporary
            raw memmap files in self.folder.img and deleted automatically
            after correction completes. Detrending removes slow baseline drift
            that can otherwise dominate component-based methods (NMF, PCA,
            ICA) and impair their ability to isolate shared noise.
            (default: False)
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
            Corrected real-signal channel, shape (T, X, Y) or subset shape
            if t_start/t_end specified. If save_to_disk is True, returns a
            memory-mapped array pointing to the file.

        Examples
        --------
        >>> # Basic linear correction (green controls red by default)
        >>> corrected = rec.correct_signal(method='linear')

        >>> # Correct green using red instead
        >>> corrected = rec.correct_signal(
        ...     method='linear', static_flu='red', real_flu='grn')

        >>> # Robust correction for data with outliers
        >>> corrected = rec.correct_signal(
        ...     method='robust', huber_epsilon=1.5)

        >>> # Save corrected signal directly to disk (minimal RAM usage)
        >>> corrected = rec.correct_signal(method='linear', save_to_disk=True)

        >>> # Process only first 60 seconds (for testing)
        >>> corrected = rec.correct_signal(method='linear', t_end=60.0)
        """
        if static_flu == real_flu:
            raise ValueError(
                f"static_flu and real_flu must differ, both are '{static_flu}'")
        for _ch in (static_flu, real_flu):
            if _ch not in ('red', 'grn'):
                raise ValueError(
                    f"Channel must be 'red' or 'grn', got '{_ch}'")

        self.corr_sig_method = method
        self.corr_real_flu = real_flu
        self.corr_static_flu = static_flu

        # Infer t_end from trial_end if not explicitly supplied
        if t_end is None and getattr(self, 'trial_end', None) is not None:
            _rew_times = self.beh._data.get_event_var('totalRewardTimes')
            t_end = _rew_times[self.trial_end] + t_end_pad
            print(f'\tt_end inferred from trial_end={self.trial_end}: '
                  f'{t_end:.2f}s (reward time + {t_end_pad}s pad)')

        # Build attribute name strings for the two channels
        _real_orig_attr = f'rec_{real_flu}_original'
        _static_orig_attr = f'rec_{static_flu}_original'

        # Get full signals — always start from the preserved original if one
        # exists so repeated calls always correct the raw data.
        s1_full = getattr(self, _static_orig_attr,
                          getattr(self, f'rec_{static_flu}'))
        s2_full = getattr(self, _real_orig_attr,
                          getattr(self, f'rec_{real_flu}'))
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
            idx_start = 0
            idx_end = n_frames

        # Record the frame range covered by the corrected stack so
        # downstream consumers (QC, plotting) can pad aggregates back
        # to full-recording length when t_start / t_end / trial_end
        # truncated the input.
        self._cs_frame_range = (int(idx_start), int(idx_end))
        self._cs_n_frames_full = int(n_frames)

        # Linearly detrend both channels before correction if requested.
        # This removes slow baseline drift so the correction method
        # focuses on shared noise (motion, haemodynamics) rather than
        # trend differences. Uses return_view=True so the trend is
        # subtracted lazily at read-time inside the correction method —
        # no float32 disk write and no temp files.
        if detrend:
            _verb = kwargs.get('verbose', True)
            _batch = kwargs.get('batch_size', 500)
            if _verb:
                print(f'\tlinearly detrending s1 ({static_flu})...')
            s1 = signal_correction.detrend_linearly(
                s1, batch_size=_batch, verbose=_verb, return_view=True)
            if _verb:
                print(f'\tlinearly detrending s2 ({real_flu})...')
            s2 = signal_correction.detrend_linearly(
                s2, batch_size=_batch, verbose=_verb, return_view=True)

        # Construct output paths if saving to disk.
        # - output_path:    full corrected stack TIFF (memmap-backed).
        #                   The correction streams its frames directly
        #                   into this file, which bounds RAM during
        #                   correction itself. Kept after the call so
        #                   self.rec_{real}_corr can stay valid (the
        #                   QC pipeline reads sectors from it) and so
        #                   the trial-avg pass can stream-read it.
        # - _trialavg_path: small (n_win, X, Y) trial-averaged TIFF
        #                   derived from the full stack. Written via
        #                   tifffile.memmap so the output is also
        #                   disk-backed.
        # ----------
        output_path = None
        _trialavg_path = None
        if save_to_disk:
            _fname_real = getattr(self, f'fname_img_{real_flu}')
            base_name, ext = os.path.splitext(_fname_real)
            output_fname = f"{base_name}_corr{ext}"
            output_path = os.path.join(self.folder.img, output_fname)
            _trialavg_path = os.path.join(
                self.folder.img,
                f"{base_name}_corr_trialavg{ext}")
            kwargs['output_path'] = output_path

        method_map = {
            'linear': lambda: signal_correction.correct_linear_regression(
                s1, s2, robust=False, **kwargs),
            'robust': lambda: signal_correction.correct_linear_regression(
                s1, s2, robust=True, **kwargs),
            'linear_photom':
                lambda: signal_correction.correct_linear_photometric(
                    s1, s2, **kwargs),
            'linear_martianova':
                lambda: signal_correction.correct_linear_martianova(
                    s1, s2, **kwargs),
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

        # All correction functions return corrected signal as first element.
        # When the correction was invoked with aggregates_only mode (currently
        # supported by linear_martianova), the "signal" is actually a dict of
        # whole-frame + per-sector mean traces — short-circuit and stash on
        # self.rec_{real}_corr_aggregates rather than walking the
        # save_to_disk / replace_real paths that assume a full (T, X, Y)
        # stack. The QC pipeline reads this attribute directly.
        # ----------
        corrected = result[0]
        if isinstance(corrected, dict) and corrected.get('is_aggregates'):
            setattr(self, f'rec_{real_flu}_corr_aggregates', corrected)
            return

        # save_to_disk path: write three trial-averaged TIFFs derived
        # from the same stim window — the corrected stack (native
        # units, typically z-scored ΔF/F), and dF/F (%) trial-averages
        # of the raw real and static channels. All three share the
        # same n_win / stim-frame / rew-frame layout and markers so
        # they can be overlaid frame-for-frame in an image viewer.
        # The full corrected stack file is kept so self.rec_{real}_
        # corr (set below) stays valid for downstream consumers
        # (e.g. the QC sector / whole-frame extractors).
        # ----------
        if save_to_disk and _trialavg_path is not None:
            _verb = kwargs.get('verbose', True)
            self._save_trial_avg_to_disk(
                source=corrected,
                idx_start=idx_start,
                output_path=_trialavg_path,
                marker_size=int(marker_size),
                t_pre=float(trialavg_t_pre),
                t_post=float(trialavg_t_post),
                dff=False,
                label=f'{real_flu}_corr',
                verbose=_verb)

            # dF/F (%) trial-avgs for the two raw channels. Use the
            # time-sliced views s2 (real) and s1 (static) so idx_start
            # offset matches the corrected stack.
            # ----------
            for _src, _flu in ((s2, real_flu), (s1, static_flu)):
                _base = os.path.splitext(
                    getattr(self, f'fname_img_{_flu}'))[0]
                _dff_path = os.path.join(
                    str(self.folder.img),
                    f'{_base}_dff_trialavg.tif')
                try:
                    self._save_trial_avg_to_disk(
                        source=_src,
                        idx_start=idx_start,
                        output_path=_dff_path,
                        marker_size=int(marker_size),
                        t_pre=float(trialavg_t_pre),
                        t_post=float(trialavg_t_post),
                        dff=True,
                        label=f'{_flu} dF/F',
                        verbose=_verb)
                except Exception as _e:
                    if _verb:
                        print(f'\twarning: dF/F trial-avg for '
                              f'{_flu} failed ({_e})')

        # Store corrected signal on the channel-specific attribute
        setattr(self, f'rec_{real_flu}_corr', corrected)

        if replace_real:
            # Preserve the original real channel the first time we replace it
            if not hasattr(self, _real_orig_attr):
                setattr(self, _real_orig_attr, getattr(self, f'rec_{real_flu}'))
                _raw_attr = f'_rec_{real_flu}_raw'
                if hasattr(self, _raw_attr):
                    setattr(self, f'_rec_{real_flu}_original_raw',
                            getattr(self, _raw_attr))
            setattr(self, f'rec_{real_flu}', corrected)
            # Store sliced timestamps if time slicing was applied
            if t_start is not None or t_end is not None:
                self.rec_t_grn = self.rec_t[idx_start:idx_end]
            else:
                self.rec_t_grn = self.rec_t

    def _save_trial_avg_to_disk(self, source, idx_start,
                                  output_path, marker_size=25,
                                  t_pre=2.0, t_post=4.0,
                                  dff=False, dff_eps=1e-6,
                                  label='source',
                                  verbose=True):
        """Trial-average a (T, X, Y) source stack and save as a
        memmap-backed TIFF.

        The averaging window spans [-t_pre, median(stim→rew) + t_post]
        relative to each stim onset. Trials whose window falls
        outside the source's frame range are skipped. The output
        stack has shape (n_window_frames, X, Y) and dtype matches
        `source` (or float32 when dff=True).

        Two dtype-max (white) square markers are stamped on the
        trial-averaged stack:
        - top-left at the stim-onset frame in the window
        - top-right at the reward-onset frame (median latency)

        Parameters
        ----------
        source : np.ndarray or memmap, (T, X, Y)
            Source stack. Read via slicing one trial at a time, so a
            memmap-backed source keeps RAM bounded.
        idx_start : int
            Frame offset into self.rec_t at which source[0] lives.
        output_path : str
            Destination TIFF path.
        marker_size : int
            Edge length of the white marker squares (pixels).
        t_pre, t_post : float
            See above.
        dff : bool
            If True, convert the trial-averaged stack to dF/F (%)
            using each pixel's mean over the pre-stim baseline window
            (first `_n_pre` frames). Output dtype becomes float32 in
            this mode (regardless of source dtype). Pixels with
            baseline |F0| < dff_eps are passed through unchanged
            (set to 0) to avoid division blow-up.
        dff_eps : float
            Floor for |F0| in the dF/F divide.
        label : str
            Label used in verbose log lines (e.g. 'red', 'grn',
            'corrected').
        verbose : bool
        """
        if not hasattr(source, 'shape') or source.ndim != 3:
            raise ValueError(
                f"source must be 3D (T, X, Y); got "
                f"{getattr(source, 'shape', type(source))}")
        T, X, Y = source.shape
        _dtype = source.dtype

        _t_local = np.asarray(self.rec_t[idx_start:idx_start + T],
                              dtype=np.float64)
        if _t_local.size < 2:
            raise ValueError(
                f"need at least 2 frames of rec_t for trial-avg; "
                f"got {_t_local.size}")
        _dt = float(np.median(np.diff(_t_local)))

        # Stim times that fall within the source's frame range.
        # ----------
        _stim_t = np.asarray(getattr(self.beh.stim, 't_start', []),
                             dtype=np.float64).ravel()
        _stim_t = _stim_t[np.isfinite(_stim_t)]
        _stim_in = ((_stim_t >= _t_local[0])
                    & (_stim_t <= _t_local[-1]))
        _stim_t = _stim_t[_stim_in]

        # Reward times — used only to compute median latency for the
        # window length and the reward-marker frame position. We pair
        # each stim with its FIRST reward in (stim_t, stim_t + 5s];
        # element-wise indexing would mismisalign because self.beh.
        # rew.t is filtered to delivered rewards only and so does not
        # share an index with self.beh.stim.t_start.
        # ----------
        _rew_raw = np.asarray(getattr(self.beh.rew, 't', []),
                              dtype=object).ravel()
        _rew_clean = []
        for _v in _rew_raw:
            try:
                _f = float(_v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(_f):
                _rew_clean.append(_f)
        _rew_t = np.sort(np.asarray(_rew_clean, dtype=np.float64))

        _max_pair_lat = 5.0
        _lats = []
        if _stim_t.size and _rew_t.size:
            _idx = np.searchsorted(_rew_t, _stim_t, side='right')
            for _i, _ri in enumerate(_idx):
                if _ri >= _rew_t.size:
                    continue
                _l = float(_rew_t[_ri] - _stim_t[_i])
                if 0 < _l <= _max_pair_lat:
                    _lats.append(_l)
        _rew_lat = float(np.median(_lats)) if _lats else 0.0
        if verbose and _stim_t.size:
            print(f'\t\tpaired {len(_lats)}/{_stim_t.size} stim→rew '
                  f'(within {_max_pair_lat:.1f}s)')

        _n_pre = int(round(t_pre / _dt))
        _n_post = int(round((_rew_lat + t_post) / _dt))
        _n_win = _n_pre + _n_post + 1
        _stim_frame_in_win = _n_pre
        _rew_frame_in_win = max(
            0, min(_n_pre + int(round(_rew_lat / _dt)), _n_win - 1))

        if verbose:
            _mode = 'dF/F %' if dff else 'native'
            print(f'\ttrial-averaging [{label}] ({_mode}):')
            print(f'\t\twindow: [-{t_pre:.2f}, '
                  f'{_rew_lat + t_post:.2f}]s '
                  f'(median stim→rew={_rew_lat:.2f}s)')
            print(f'\t\tframes: {_n_win} ({_n_pre} pre, {_n_post} post), '
                  f'dt={_dt:.4f}s')
            print(f'\t\tstim events in range: {_stim_t.size}')

        # Accumulator in float32 for precision; one window-sized
        # buffer (~n_win × X × Y × 4 bytes) plus one per-trial chunk
        # of the same size, transiently.
        # ----------
        _acc = np.zeros((_n_win, X, Y), dtype=np.float32)
        _count = 0
        for _ev_idx, _ev in enumerate(_stim_t):
            _i = int(np.argmin(np.abs(_t_local - _ev)))
            _i0 = _i - _n_pre
            _i1 = _i + _n_post + 1
            if _i0 < 0 or _i1 > T:
                continue
            _chunk = np.asarray(source[_i0:_i1], dtype=np.float32)
            if _chunk.shape != _acc.shape:
                continue
            _acc += _chunk
            _count += 1
            if verbose:
                print(f'\t\t\ttrial {_ev_idx + 1}/{_stim_t.size} '
                      f'(included={_count})', end='\r')

        if verbose:
            print(f'\t\tincluded {_count}/{_stim_t.size} trials       ')
        if _count == 0:
            raise ValueError(
                "no trials survived the window filter; "
                "cannot trial-average")

        _avg_f32 = _acc / float(_count)
        del _acc

        # Optional dF/F (%) transform using per-pixel mean over the
        # pre-stim baseline window. Output stays as float32 in this
        # mode for accurate dynamic range; baseline-zero pixels get
        # zeroed out (no division blow-up).
        # ----------
        if dff:
            if _n_pre < 1:
                raise ValueError(
                    f"dff=True requires at least 1 pre-stim frame; "
                    f"got _n_pre={_n_pre} (t_pre={t_pre}s, dt={_dt}s)")
            _f0 = _avg_f32[:_n_pre].mean(axis=0)
            _f0_safe = np.where(np.abs(_f0) >= dff_eps, _f0, 1.0)
            with np.errstate(invalid='ignore', divide='ignore'):
                _avg_out = ((_avg_f32 - _f0[None]) / _f0_safe[None]
                            * 100.0)
            _bad = np.abs(_f0) < dff_eps
            if np.any(_bad):
                _avg_out[:, _bad] = 0.0
            _out_dtype = np.float32
        else:
            _avg_out = _avg_f32.astype(_dtype)
            _out_dtype = _dtype
        del _avg_f32

        # Write the trial-averaged stack via tifffile.memmap so the
        # output also lives on disk without a separate RAM buffer.
        # ----------
        if verbose:
            print(f'\twriting trial-averaged TIFF [{label}] to: '
                  f'{output_path}')
        if os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        _out = tifffile.memmap(output_path, shape=(_n_win, X, Y),
                               dtype=_out_dtype, bigtiff=True)
        _out[:] = _avg_out.astype(_out_dtype, copy=False)
        _out.flush()
        del _avg_out

        # White-square markers: dtype-max top-left at stim frame,
        # dtype-max top-right at reward frame. np.iinfo / np.finfo
        # return python scalars; numpy auto-casts them to the memmap's
        # dtype on assignment. (Don't call _out_dtype(...) directly —
        # numpy 2.x dtype objects are no longer callable.)
        # ----------
        _ms = int(min(marker_size, X, Y))
        if _ms > 0:
            _dt_norm = np.dtype(_out_dtype)
            if np.issubdtype(_dt_norm, np.integer):
                _white = _dt_norm.type(np.iinfo(_dt_norm).max)
            else:
                _white = _dt_norm.type(np.finfo(_dt_norm).max)
            _out[_stim_frame_in_win, :_ms, :_ms] = _white
            _out[_rew_frame_in_win, :_ms, Y - _ms:] = _white
            _out.flush()
            if verbose:
                print(f'\tmarkers: white {_ms}x{_ms} top-left at '
                      f'frame {_stim_frame_in_win} (stim onset); '
                      f'top-right at frame {_rew_frame_in_win} '
                      f'(rew onset, lat={_rew_lat:.2f}s)')

        del _out

    def _stamp_corr_event_markers(self, corrected, idx_start=0,
                                   marker_size=8, output_path=None,
                                   verbose=True):
        """Stamp event-aligned marker pixels onto a disk-backed
        corrected stack.

        Stim onsets paint a `marker_size` square black block at the
        top-left of the corresponding frame (nearest to each stim
        time); reward deliveries paint a white block.

        When `output_path` is given, the saved TIFF is REOPENED via a
        fresh tifffile.memmap(mode='r+') handle for writing — the
        memmap returned by the correction function has already been
        flushed once and in some tifffile versions further writes
        through that stale handle don't propagate. Falling back to
        the passed-in `corrected` array (in-RAM stack) is supported
        for callers that don't go through save_to_disk.

        `idx_start` is the frame offset into self.rec_t at which the
        first frame of the stack lives (non-zero when correct_signal
        was called with t_start).
        """
        if marker_size <= 0:
            return

        # Choose write target: prefer a fresh r+ memmap on the saved
        # file when we have its path, else write through the passed-
        # in array (for save_to_disk=False callers).
        # ----------
        _own_target = False
        if output_path is not None and os.path.exists(output_path):
            try:
                target = tifffile.memmap(output_path, mode='r+')
                _own_target = True
            except Exception as _e:
                if verbose:
                    print(f'\tcould not reopen {output_path} for marker '
                          f'stamping ({_e}); falling back to existing '
                          f'memmap handle')
                target = corrected
        else:
            target = corrected

        if target.ndim != 3:
            if verbose:
                print(f'\tmarker stamping: unexpected target shape '
                      f'{target.shape}; skipping')
            if _own_target:
                del target
            return

        T, X, Y = target.shape
        _ms = int(min(marker_size, X, Y))
        if _ms <= 0:
            if _own_target:
                del target
            return

        _dt = target.dtype
        if np.issubdtype(_dt, np.integer):
            _info = np.iinfo(_dt)
            _black = _dt.type(_info.min)
            _white = _dt.type(_info.max)
        else:
            _info = np.finfo(_dt)
            _black = _dt.type(_info.min)
            _white = _dt.type(_info.max)

        _t_local = np.asarray(self.rec_t[idx_start:idx_start + T],
                              dtype=np.float64)
        if _t_local.size == 0:
            if _own_target:
                del target
            return

        def _nearest_frame_inds(events):
            _ev = np.asarray(events, dtype=np.float64).ravel()
            _ev = _ev[np.isfinite(_ev)]
            if _ev.size == 0:
                return np.array([], dtype=np.int64)
            _in = (_ev >= _t_local[0]) & (_ev <= _t_local[-1])
            _ev = _ev[_in]
            if _ev.size == 0:
                return np.array([], dtype=np.int64)
            _r = np.searchsorted(_t_local, _ev, side='left')
            _r = np.clip(_r, 1, _t_local.size - 1)
            _prev = _t_local[_r - 1]
            _curr = _t_local[_r]
            _choose_prev = (_ev - _prev) <= (_curr - _ev)
            return np.where(_choose_prev, _r - 1, _r).astype(np.int64)

        _stim_t = np.asarray(getattr(self.beh.stim, 't_start', []),
                             dtype=np.float64).ravel()

        # self.beh.rew.t may contain None entries for undelivered
        # trials — strip them before passing through.
        _rew_raw = np.asarray(getattr(self.beh.rew, 't', []),
                              dtype=object).ravel()
        _rew_clean = []
        for _v in _rew_raw:
            try:
                _f = float(_v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(_f):
                _rew_clean.append(_f)
        _rew_t = np.asarray(_rew_clean, dtype=np.float64)

        _stim_inds = _nearest_frame_inds(_stim_t)
        _rew_inds = _nearest_frame_inds(_rew_t)

        if verbose:
            print(f'\tstamping markers on saved file: '
                  f'{_stim_inds.size} stim (black={float(_black):.3g}), '
                  f'{_rew_inds.size} rew (white={float(_white):.3g}), '
                  f'block={_ms}x{_ms}, target shape={target.shape}, '
                  f'dtype={_dt}')
            if _stim_inds.size:
                print(f'\t\tstim frames (first 5): '
                      f'{_stim_inds[:5].tolist()} ... '
                      f'last: {int(_stim_inds[-1])}')
            if _rew_inds.size:
                print(f'\t\trew frames  (first 5): '
                      f'{_rew_inds[:5].tolist()} ... '
                      f'last: {int(_rew_inds[-1])}')

        for _i in _stim_inds:
            target[int(_i), :_ms, :_ms] = _black
        if hasattr(target, 'flush'):
            target.flush()
        for _i in _rew_inds:
            target[int(_i), :_ms, :_ms] = _white
        if hasattr(target, 'flush'):
            target.flush()

        # Read-back verification on the last written frame of each
        # kind. Detects silently-dropped writes (e.g. stale-handle
        # memmaps or wrong-shape targets).
        # ----------
        if verbose:
            def _check(inds, expected, label):
                if inds.size == 0:
                    return
                _i = int(inds[-1])
                _got = float(target[_i, 0, 0])
                _ok = np.isclose(_got, float(expected),
                                 rtol=0, atol=1e-3)
                _tag = 'ok' if _ok else 'MISMATCH'
                print(f'\t\tverify {label} frame {_i}: '
                      f'pix[0,0]={_got:.4g} '
                      f'(expected {float(expected):.4g}) [{_tag}]')
            _check(_stim_inds, _black, 'stim')
            _check(_rew_inds, _white, 'rew')

        if _own_target:
            # Release the reopened memmap so its handle isn't held
            # for the rest of the session. The saved file is now
            # fully written and flushed.
            del target

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
            Output filename (without path). Written to self.folder.img.
            If None, the filename is constructed automatically from
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
        _real_flu = getattr(self, 'corr_real_flu', 'grn')
        _corr_attr = f'rec_{_real_flu}_corr'
        if not hasattr(self, _corr_attr):
            raise RuntimeError(
                f"{_corr_attr} not found. Run correct_signal() first.")

        _corr = getattr(self, _corr_attr)

        if fname is None:
            _fname_real = getattr(self, f'fname_img_{_real_flu}',
                                  self.fname_img_grn)
            base_name, ext = os.path.splitext(_fname_real)
            method_tag = getattr(self, 'corr_sig_method', 'corr')
            fname = f"{base_name}_corr_{method_tag}{ext}"

        output_path = os.path.join(self.folder.img, fname)
        n_frames = _corr.shape[0]

        print(f'saving corrected signal to {output_path}...')
        with tifffile.TiffWriter(output_path, bigtiff=True) as writer:
            for idx_start in range(0, n_frames, chunk_size):
                idx_end = min(idx_start + chunk_size, n_frames)
                print(f'\tframes {idx_start}-{idx_end}/{n_frames}', end='\r')
                chunk = np.array(_corr[idx_start:idx_end])
                for frame in chunk:
                    writer.write(frame, compression=None, contiguous=True)
        print(f'\ndone.')

        return output_path

    # ------------------------------------------------------------------
    # Dual-colour plotting
    # ------------------------------------------------------------------

    def plt_frame_dualcolour(self,
                             figsize=(7, 2.5),
                             t_pre=2, t_post=5,
                             colors=None,
                             plot_type=None,
                             plt_show=True,
                             use_zscore=False):
        """Side-by-side whole-frame fluorescence for red and green channels.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height) in inches.
        t_pre : float
            Seconds before stimulus onset.
        t_post : float
            Seconds after reward delivery.
        colors : list or None
            Colour palette forwarded to plt_frame.
        plot_type : str or None
            Forwarded to plt_frame.
        plt_show : bool
            If True, call plt.show() after plotting.
        use_zscore : bool
            If True, plot z-scored traces instead of df/f.
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=figsize)
        spec = gridspec.GridSpec(nrows=1, ncols=2, figure=fig,
                                 wspace=0.35)
        ax_red = fig.add_subplot(spec[0, 0])
        ax_grn = fig.add_subplot(spec[0, 1])

        self.plt_frame(t_pre=t_pre, t_post=t_post, colors=colors,
                        plot_type=plot_type, channel='red',
                        use_zscore=use_zscore, ax=ax_red)
        self.plt_frame(t_pre=t_pre, t_post=t_post, colors=colors,
                        plot_type=plot_type, channel='grn',
                        use_zscore=use_zscore, ax=ax_grn)

        ax_red.set_title('red')
        ax_grn.set_title('green')

        _corr_suffix = ''
        if hasattr(self, 'corr_sig_method'):
            _corr_suffix = f'_corr={self.corr_sig_method}'
        _zscore_suffix = '_zscore' if use_zscore else ''
        fig.savefig(os.path.join(
            str(self.folder.figs),
            f'{self.path.animal}_{self.path.date}_{self.path.beh_folder}_'
            f'{plot_type=}_{t_pre=}_{t_post=}_dualcolour'
            f'{_corr_suffix}{_zscore_suffix}_mean_trial_activity.pdf'))

        if plt_show:
            plt.show()
        else:
            plt.close(fig)

    def plt_sectors_dualcolour(self,
                               n_sectors=10,
                               t_pre=2, t_post=5,
                               plot_type=None,
                               figsize=(12, 6),
                               compare_figsize=None,
                               img_ds_factor=50,
                               img_alpha=0.5,
                               plt_dff=None,
                               plt_prefix='',
                               plt_show=True,
                               use_zscore=False,
                               outlier_thresh=None,
                               auto_gain_dff=False,
                               auto_gain_dff_scale=0.7,
                               colors=None):
        """Side-by-side sector overlay for red and green channels.

        Parameters
        ----------
        n_sectors : int
            Number of sectors per side (total sectors = n_sectors**2).
        t_pre : float
            Seconds before stimulus onset.
        t_post : float
            Seconds after reward delivery.
        plot_type : str or None
            Forwarded to plt_sectors.
        figsize : tuple
            Figure size (width, height) in inches.
        img_ds_factor : int
            Down-sample factor for the max-projection background image.
        img_alpha : float
            Opacity of the background image.
        plt_dff : dict or None
            Gain/offset dict forwarded to plt_sectors.
        plt_prefix : str
            Prefix for the saved filename.
        plt_show : bool
            If True, call plt.show() after plotting.
        use_zscore : bool
            If True, plot z-scored traces instead of df/f.
        outlier_thresh : float or None
            Forwarded to plt_sectors.
        auto_gain_dff : bool
            If True, auto-scale the df/f gain.
        auto_gain_dff_scale : float
            Multiplier applied on top of the auto-gain calculation.
            Values < 1 compress the y-scale, > 1 expand it. Default 0.7.
        compare_figsize : tuple or None
            Figure size for the per-sector red/green comparison figure.
            Defaults to (2*n_sectors, 2*n_sectors).
        colors : list or None
            Colour palette forwarded to plt_sectors.
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import seaborn as sns
        from copy import deepcopy

        fig = plt.figure(figsize=figsize)
        spec = gridspec.GridSpec(nrows=1, ncols=2, figure=fig,
                                 wspace=0.3)
        ax_red = fig.add_subplot(spec[0, 0])
        ax_grn = fig.add_subplot(spec[0, 1])

        _shared_kw = dict(
            n_sectors=n_sectors, t_pre=t_pre, t_post=t_post,
            plot_type=plot_type, img_ds_factor=img_ds_factor,
            img_alpha=img_alpha, plt_dff=plt_dff,
            plt_prefix=plt_prefix, use_zscore=use_zscore,
            outlier_thresh=outlier_thresh,
            auto_gain_dff=auto_gain_dff,
            auto_gain_dff_scale=auto_gain_dff_scale, colors=colors,
            minimal_output=True)

        self.plt_sectors(channel='red', ax=ax_red, **_shared_kw)
        _red_dff = deepcopy(self.sector.dff)
        _red_t = np.array(self.sector.t)
        _red_use_zscore = self.sector.params.use_zscore

        self.plt_sectors(channel='grn', ax=ax_grn, **_shared_kw)
        _grn_dff = deepcopy(self.sector.dff)
        _grn_t = np.array(self.sector.t)
        _grn_use_zscore = self.sector.params.use_zscore

        ax_red.set_title('red')
        ax_grn.set_title('green')

        _corr_suffix = ''
        if hasattr(self, 'corr_sig_method'):
            _corr_suffix = f'_corr={self.corr_sig_method}'
        _zscore_suffix = f'_zscore={use_zscore}'
        fig.savefig(os.path.join(
            str(self.folder.figs),
            f'sectors_{self.path.animal}_{self.path.date}'
            f'_{self.path.beh_folder}_{plt_prefix}'
            f'_{n_sectors=}_{plot_type=}_{t_pre=}_{t_post=}'
            f'_dualcolour{_corr_suffix}{_zscore_suffix}.pdf'))

        # -----------------------------------------------------------
        # Second figure: per-sector red/green overlay, split by tr_cond.
        # Layout mirrors plt_sectors — tiled sectors on a single axes
        # over a max-projection background (green channel always), with
        # unified auto-gain so scalebars are identical across sectors.
        # Channel encoded by colour (red / green); tr_cond by linestyle.
        # -----------------------------------------------------------
        _is_visual_pavlov = (
            getattr(self, 'beh_type', None) == 'visual_pavlov')
        if _is_visual_pavlov:
            _stim_list_map = {
                None: ['0', '0.5', '1'],
                'rew_norew': ['0.5_rew', '0.5_norew'],
                'prelick_noprelick': ['0.5_prelick', '0.5_noprelick'],
                'rew': ['0', '0.5_rew', '1']}
            stim_list = _stim_list_map.get(
                plot_type, list(self.beh.tr_conds))
        else:
            stim_list = list(self.beh.tr_conds)

        # linestyle encodes trial-type
        _ls_cycle = ['solid', 'dashed', 'dotted', 'dashdot',
                     (0, (3, 1, 1, 1))]
        if _is_visual_pavlov:
            _vp_ls = {'0': 'dotted', '0.5': 'solid', '1': 'dashed',
                      '0.5_rew': 'solid', '0.5_norew': 'dashed',
                      '0.5_prelick': 'solid', '0.5_noprelick': 'dashed'}
            cond_ls = {k: _vp_ls.get(k, _ls_cycle[i % len(_ls_cycle)])
                       for i, k in enumerate(stim_list)}
        else:
            cond_ls = {k: _ls_cycle[i % len(_ls_cycle)]
                       for i, k in enumerate(stim_list)}

        # plt_dff defaults matching plt_sectors
        if plt_dff is None:
            _plt_dff_cmp = {'x': {'gain': 0.6, 'offset': 0.2},
                            'y': {'gain': 10, 'offset': 0.2}}
        else:
            _plt_dff_cmp = {'x': dict(plt_dff['x']),
                            'y': dict(plt_dff['y'])}

        # unified auto-gain across both channels / all sectors / conds
        if auto_gain_dff:
            _max_abs = 0.0
            for _dff_arr in (_red_dff, _grn_dff):
                for gx in range(n_sectors):
                    for gy in range(n_sectors):
                        _cell = _dff_arr[gx, gy]
                        if not isinstance(_cell, dict):
                            continue
                        for sc in stim_list:
                            if sc not in _cell:
                                continue
                            _arr = _cell[sc]
                            if _arr.shape[0] == 0:
                                continue
                            _v = float(np.max(np.abs(
                                np.median(_arr, axis=0))))
                            if _v > _max_abs:
                                _max_abs = _v
            if _max_abs > 0:
                _plt_dff_cmp['y']['gain'] = (
                    auto_gain_dff_scale * 0.5 / _max_abs)

        if compare_figsize is None:
            compare_figsize = figsize
        fig_cmp = plt.figure(figsize=compare_figsize)
        spec_cmp = gridspec.GridSpec(nrows=1, ncols=1, figure=fig_cmp)
        ax_cmp = fig_cmp.add_subplot(spec_cmp[0, 0])

        # background: green-channel max projection (always green)
        _rec_grn = self._get_rec('grn')
        _rec_max = np.max(_rec_grn[::img_ds_factor, :, :], axis=0)
        _step = int(_rec_max.shape[0] / n_sectors)
        _rec_max[:, ::_step] = 0
        _rec_max[::_step, :] = 0
        _extent = [0, n_sectors, n_sectors, 0]
        ax_cmp.imshow(_rec_max, extent=_extent, alpha=img_alpha,
                      cmap=self._channel_img_cmap('grn'))

        _ch_colors = {'red': '#ffe0e0', 'grn': '#e0ffe0'}
        _ch_data = {'red': (_red_dff, _red_t), 'grn': (_grn_dff, _grn_t)}
        _vkw = dict(linewidth=1, linestyle='dashed')

        for n_x in range(n_sectors):
            for n_y in range(n_sectors):
                _dff_zero = n_y + 0.5
                for ch_name, (_dff_arr, _t_arr) in _ch_data.items():
                    _t_sector = ((_t_arr / _t_arr[-1])
                                 * _plt_dff_cmp['x']['gain']
                                 + n_x + _plt_dff_cmp['x']['offset'])
                    _cell = _dff_arr[n_x, n_y]
                    if not isinstance(_cell, dict):
                        continue
                    for stim_cond in stim_list:
                        if stim_cond not in _cell:
                            continue
                        _arr = _cell[stim_cond]
                        if _arr.shape[0] == 0:
                            continue
                        _dff_mean = np.median(_arr, axis=0)
                        _shifted = (-_dff_mean
                                    * _plt_dff_cmp['y']['gain']
                                    + _dff_zero)
                        _shifted = np.clip(_shifted,
                                           n_y + 0.05,
                                           n_y + 0.95)
                        ax_cmp.plot(_t_sector, _shifted,
                                    color=_ch_colors[ch_name],
                                    linestyle=cond_ls[stim_cond],
                                    linewidth=1,
                                    alpha=0.8,
                                    label=(f'{ch_name} {stim_cond}'
                                           if (n_x == 0 and n_y == 0)
                                           else None))

                _t_stim = n_x + _plt_dff_cmp['x']['offset']
                _t_rew = ((2 / _red_t[-1])
                          * _plt_dff_cmp['x']['gain']
                          + n_x + _plt_dff_cmp['x']['offset'])
                ax_cmp.plot([_t_stim, _t_stim],
                            [n_y + 0.2, n_y + 0.8],
                            color=sns.xkcd_rgb['dark grey'], **_vkw)
                ax_cmp.plot([_t_rew, _t_rew],
                            [n_y + 0.2, n_y + 0.8],
                            color=sns.xkcd_rgb['bright blue'], **_vkw)

        ax_cmp.set_xticks([])
        ax_cmp.set_yticks([])
        ax_cmp.legend(fontsize=6, loc='upper right',
                      bbox_to_anchor=(1.25, 1.0),
                      frameon=False)

        fig_cmp.savefig(os.path.join(
            str(self.folder.figs),
            f'sectors_{self.path.animal}_{self.path.date}'
            f'_{self.path.beh_folder}_{plt_prefix}'
            f'_{n_sectors=}_{plot_type=}_{t_pre=}_{t_post=}'
            f'_dualcolour_compare'
            f'{_corr_suffix}{_zscore_suffix}.pdf'),
            bbox_inches='tight')

        # -----------------------------------------------------------
        # Third figure: (red - green) df/f difference per sector.
        # Same tiled layout as the second figure.
        # -----------------------------------------------------------
        _diff_colors = self._channel_palette(None, n_colors=3)
        if _is_visual_pavlov:
            _vp_diff_order = {'0': 0, '0.5': 1, '1': 2,
                              '0.5_rew': 1, '0.5_norew': 1,
                              '0.5_prelick': 1, '0.5_noprelick': 1}
            diff_cond_colors = {k: _diff_colors[_vp_diff_order.get(k, 1)]
                                for k in stim_list}
        else:
            diff_cond_colors = {k: _diff_colors[i % len(_diff_colors)]
                                for i, k in enumerate(stim_list)}

        # unified auto-gain for difference traces
        _diff_max_abs = 0.0
        for gx in range(n_sectors):
            for gy in range(n_sectors):
                _r = _red_dff[gx, gy]
                _g = _grn_dff[gx, gy]
                if not isinstance(_r, dict) or not isinstance(_g, dict):
                    continue
                for sc in stim_list:
                    if sc not in _r or sc not in _g:
                        continue
                    if _r[sc].shape[0] == 0 or _g[sc].shape[0] == 0:
                        continue
                    _diff = (np.median(_r[sc], axis=0)
                             - np.median(_g[sc], axis=0))
                    _v = float(np.max(np.abs(_diff)))
                    if _v > _diff_max_abs:
                        _diff_max_abs = _v
        if auto_gain_dff and _diff_max_abs > 0:
            _gain_diff = auto_gain_dff_scale * 0.5 / _diff_max_abs
        else:
            _gain_diff = _plt_dff_cmp['y']['gain']

        fig_diff = plt.figure(figsize=compare_figsize)
        spec_diff = gridspec.GridSpec(nrows=1, ncols=1, figure=fig_diff)
        ax_diff = fig_diff.add_subplot(spec_diff[0, 0])

        ax_diff.imshow(_rec_max, extent=_extent, alpha=img_alpha,
                       cmap=self._channel_img_cmap('grn'))

        for n_x in range(n_sectors):
            for n_y in range(n_sectors):
                _dff_zero = n_y + 0.5
                _t_sector = ((_red_t / _red_t[-1])
                             * _plt_dff_cmp['x']['gain']
                             + n_x + _plt_dff_cmp['x']['offset'])
                _r_cell = _red_dff[n_x, n_y]
                _g_cell = _grn_dff[n_x, n_y]
                if not isinstance(_r_cell, dict) \
                        or not isinstance(_g_cell, dict):
                    continue
                for stim_cond in stim_list:
                    if stim_cond not in _r_cell or stim_cond not in _g_cell:
                        continue
                    if (_r_cell[stim_cond].shape[0] == 0
                            or _g_cell[stim_cond].shape[0] == 0):
                        continue
                    _diff_mean = (np.median(_r_cell[stim_cond], axis=0)
                                  - np.median(_g_cell[stim_cond], axis=0))
                    _shifted = -_diff_mean * _gain_diff + _dff_zero
                    _shifted = np.clip(_shifted, n_y + 0.05, n_y + 0.95)
                    ax_diff.plot(_t_sector, _shifted,
                                 color=diff_cond_colors[stim_cond],
                                 linestyle=cond_ls[stim_cond],
                                 linewidth=1,
                                 alpha=0.8,
                                 label=(f'{stim_cond}'
                                        if (n_x == 0 and n_y == 0)
                                        else None))

                _t_stim = n_x + _plt_dff_cmp['x']['offset']
                _t_rew = ((2 / _red_t[-1])
                          * _plt_dff_cmp['x']['gain']
                          + n_x + _plt_dff_cmp['x']['offset'])
                ax_diff.plot([_t_stim, _t_stim],
                             [n_y + 0.2, n_y + 0.8],
                             color=sns.xkcd_rgb['dark grey'], **_vkw)
                ax_diff.plot([_t_rew, _t_rew],
                             [n_y + 0.2, n_y + 0.8],
                             color=sns.xkcd_rgb['bright blue'], **_vkw)

        ax_diff.set_xticks([])
        ax_diff.set_yticks([])
        ax_diff.legend(fontsize=6, loc='upper right',
                       bbox_to_anchor=(1.25, 1.0), frameon=False)

        fig_diff.savefig(os.path.join(
            str(self.folder.figs),
            f'sectors_{self.path.animal}_{self.path.date}'
            f'_{self.path.beh_folder}_{plt_prefix}'
            f'_{n_sectors=}_{plot_type=}_{t_pre=}_{t_post=}'
            f'_dualcolour_diff'
            f'{_corr_suffix}{_zscore_suffix}.pdf'),
            bbox_inches='tight')

        if plt_show:
            plt.show()
        else:
            plt.close(fig)
            plt.close(fig_cmp)
            plt.close(fig_diff)
