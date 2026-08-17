from types import SimpleNamespace
import hashlib
import inspect
import re
import numpy as np
import os
import pathlib
import tifffile
import warnings

from .align_imgbeh import Aligner_ImgBeh
from .utils import (find_event_onsets,
                    find_event_onsets_autothresh,
                    find_event_onsets_plateau,
                    fmt_kwarg_val as _fmt_kwarg_val)
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


# Longest suffix appended to the `_corr_base` stem built in
# correct_signal ('_celltraces_trialavg.npy'); the kwarg tag's length
# budget is computed against it so EVERY file derived from that stem
# fits inside a filesystem name component.
_CORR_BASE_MAX_SUFFIX = len('_celltraces_trialavg.npy')
# POSIX NAME_MAX: a single path component may not exceed 255 bytes
# (macOS/APFS and ext4 both enforce this; exceeding it raises
# OSError errno 63, 'File name too long').
_FNAME_MAX_LEN = 255


def _full_regress_kwarg_tag(kwargs, max_len=96):
    """Build a bounded filename tag recording a correct_full_regress run.

    Reflects the signature of
    ``signal_correction.correct_full_regress`` so the saved file records
    the parameterisation used. Only the kwargs that DEVIATE from the
    function's own defaults are spelled out as ``name-value`` tokens; a
    7-hex-digit hash of the *full* resolved parameterisation (every
    non-skipped parameter, default or not) is always appended. So the
    readable part stays short and says what was special about this run,
    while the hash still distinguishes any two runs that differ in any
    parameter at all — including runs made before and after a default
    changes, which the readable part alone could not tell apart.

    Spelling out every parameter (the previous behaviour) overflowed the
    255-byte filesystem limit on a name component once the stem, the
    ``_celltraces_trialavg.npy`` suffix and ~17 parameters were summed —
    ``OSError: [Errno 63] File name too long``. ``max_len`` caps the
    returned tag; the readable part is truncated (never the hash) when
    the overrides alone would exceed it.

    Omitted are only the inputs/arguments that do not affect the saved
    data:
    - `s1`, `s2`     : the positional input stacks.
    - `output_path`  : the file being named (circular).
    - `verbose`      : console output only.
    - `aggregates_only` : mutually exclusive with `output_path`, so
      always None on the disk-writing path.

    Parameters
    ----------
    kwargs : dict
        The keyword arguments forwarded to the correction function.
        Values absent here fall back to the function's own defaults.
    max_len : int
        Hard cap on the length of the returned tag, hash included.
        Default 96. ``correct_signal`` passes the exact budget left by
        the recording's filename stem.

    Returns
    -------
    tag : str
        e.g. ``fit_mode-per_pixel_regress_type-irls-3f9a1c2`` — the
        overrides, then the full-parameterisation hash. Runs taking every
        default give just the hash (``default-<hash>``).
    """
    sig = inspect.signature(signal_correction.correct_full_regress)
    skip = {'s1', 's2', 'output_path', 'verbose', 'aggregates_only'}
    parts = []
    full = []
    for name, param in sig.parameters.items():
        if name in skip:
            continue
        has_default = param.default is not inspect.Parameter.empty
        if name in kwargs:
            val = kwargs[name]
        elif has_default:
            val = param.default
        else:
            continue
        token = f'{name}-{_fmt_kwarg_val(val)}'
        full.append(token)
        # Compare the *rendered* values rather than the objects: some
        # kwargs are arrays/dicts whose == is not a bool.
        if not (has_default
                and _fmt_kwarg_val(val) == _fmt_kwarg_val(param.default)):
            parts.append(token)

    digest = hashlib.sha1('_'.join(full).encode()).hexdigest()[:7]
    tag = '_'.join(parts) if parts else 'default'
    # Keep tuples/dicts readable rather than mangled: (2,4,8) -> 2-4-8.
    tag = tag.replace(',', '-')
    tag = re.sub(r'[^A-Za-z0-9._-]', '', tag)
    # Truncate the readable part only — the hash is what guarantees two
    # different parameterisations never collide, so it always survives.
    keep = max(1, int(max_len) - len(digest) - 1)
    if len(tag) > keep:
        tag = tag[:keep].rstrip('_-.')
    return f'{tag}-{digest}'


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
        self._init_timestamps(self._n_frames_for_timestamps(), rec_type)

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
        enclosing folder. Creates figs_mbl/ and data_mbl/ if they do not
        exist."""
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
        self.folder.data = _enc / 'data_mbl'
        self.folder.data.mkdir(exist_ok=True)

        self.path.raw = _enc
        self.path.animal = _enc.parts[-2]
        self.path.date = _enc.parts[-1]
        self.path.beh_folder = folder_beh  # keep original relative name

    def _rec_file_prefix(self):
        """'{animal}_{date}_{beh}_' -- the per-recording filename stem.

        data_mbl/ and figs_mbl/ sit at the DATE level and are therefore
        shared by every behaviour session of that date. Any file named
        after the imaging stem alone ('compiled_Ch1...') would collide
        between sessions, so all such outputs carry this prefix — the
        same convention the QC figures and .npy summaries already use.

        Returns
        -------
        prefix : str
            Empty string if the path namespace is not populated, so
            callers degrade to the old unprefixed names rather than
            raising.
        """
        _p = getattr(self, 'path', None)
        _parts = [getattr(_p, _k, None)
                  for _k in ('animal', 'date', 'beh_folder')]
        if _p is None or any(_v is None for _v in _parts):
            return ''
        return '_'.join(str(_v) for _v in _parts) + '_'

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

    def _n_frames_for_timestamps(self):
        """Return the frame count used to build rec_t.

        Uses the loaded recording (self.rec) if available; otherwise falls
        back to the number of frames in the suite2p output. The latter case
        occurs when the tiffs are not loaded for a dual-colour recording
        (fname_img_red and fname_img_grn both None).

        Returns
        -------
        n_frames : int
            Number of imaging frames.
        """
        if getattr(self, 'rec', None) is not None:
            return self.rec.shape[0]
        if hasattr(self, 'neur') and getattr(self.neur, 'f', None) is not None:
            return self.neur.f.shape[1]
        raise RuntimeError(
            "Cannot determine frame count: neither imaging data nor "
            "suite2p output was loaded.")

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
            # Anchor the imaging clock on the first physical reward echo
            # (Timeline DAQ). The reward_echo TTL marks the actual reward
            # delivery; the Block file's totalRewardTimes sit ~30 ms
            # earlier on a separate software clock. Fall back to the Block
            # reward time if the echo channel is unavailable.
            _t_start = self._first_reward_echo_t()
            if _t_start is None:
                _t_start = float(self.beh.rew.t[0])
                print('\t\treward_echo channel unavailable; anchoring '
                      'rec_t on Block reward time (~30 ms less accurate)')
            self.rec_t = _t_start + np.arange(n_frames) / self.samp_rate
        elif rec_type == 'paqio':
            self._aligner = Aligner_ImgBeh()
            self._aligner.parse_img_rewechoes(folder=self.folder.beh)
            self._aligner.parse_beh_rewechoes(folder=self.folder.beh)
            self._aligner.compute_alignment()
            self.rec_t = np.arange(n_frames) / self.samp_rate
            self.rec_t = self._aligner.correct_img_data(self.rec_t)

        if hasattr(self, 'neur'):
            self.neur.t = self.rec_t

    def _first_reward_echo_t(self):
        """Return the time of the first physical reward echo, or None.

        The reward_echo TTL on the behavioural Timeline DAQ marks actual
        reward delivery and is the correct anchor for the imaging clock
        under rec_type='trig_rew'. Returns None if the reward_echo channel
        or DAQ data is unavailable so the caller can fall back to the Block
        reward time.

        Returns
        -------
        t_echo : float or None
            Timeline-clock time (s) of the first reward echo onset.
        """
        try:
            _daq = self.beh._daq_data
            _inds = find_event_onsets(_daq.sig['reward_echo'], thresh=3)
            if len(_inds) == 0:
                return None
            return float(_daq.t[_inds[0]])
        except (AttributeError, KeyError):
            return None

    def _init_stim_rew_range(self, trial_end=None):
        """Compute self.beh._stimrange and self.beh._rewrange.

        Finds the first and last stimulus/reward indices that fall within
        the recording window. When the imaging file is shorter than the
        behavioural session, trials whose window extends past the end of
        the recording (or before its start) are excluded so that only
        trials with imaging coverage are kept. Optionally clips the last
        trial to trial_end.

        An event is considered in range when its onset has at least 2 s of
        recording before it and 4 s after it (the trial window margins).

        Parameters
        ----------
        trial_end : None or int
            If given, overrides the computed last index for both ranges.
        """
        self.beh._stimrange = self._event_range(self.beh.stim.t_start)
        self.beh._rewrange = self._event_range(self.beh.rew.t)

        _n_stims = self.beh.stim.t_start.shape[0]
        _n_kept = self.beh._stimrange.last - self.beh._stimrange.first + 1
        if _n_kept < _n_stims:
            print(f'\timaging shorter than behaviour: keeping stims '
                  f'{self.beh._stimrange.first}-{self.beh._stimrange.last} '
                  f'({_n_kept}/{_n_stims} trials with imaging coverage)')

        # if trial_end is manually specified, replace these attributes
        if trial_end is not None:
            self.beh._stimrange.last = trial_end
            self.beh._rewrange.last = trial_end

    def _event_range(self, t_events, t_pre=2, t_post=4):
        """Return first/last indices of events covered by the recording.

        Parameters
        ----------
        t_events : np.ndarray
            Event onset times (s), assumed monotonically increasing.
        t_pre : float
            Required recording margin before each onset (s).
        t_post : float
            Required recording margin after each onset (s).

        Returns
        -------
        SimpleNamespace
            .first and .last indices into t_events bounding the events
            that fall within the recording. If no event is covered,
            .first is 0 and .last is -1 (an empty range).
        """
        t_events = np.asarray(t_events)
        _in_range = np.where(
            (t_events - t_pre >= self.rec_t[0])
            & (t_events + t_post <= self.rec_t[-1]))[0]

        _range = SimpleNamespace()
        if len(_in_range) > 0:
            _range.first = int(_in_range[0])
            _range.last = int(_in_range[-1])
        else:
            _range.first = 0
            _range.last = -1
        return _range

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
                 dX_red=-5.79,
                 dY_red=-4.30,
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

        Red/green laser-path misalignment correction
        ---------------------------------------------
        The red and green laser paths are slightly misaligned, so the
        red movie is rigidly shifted in x and y to register it onto the
        green channel. The shift is specified in microns and converted to
        an integer pixel shift using micronsPerPixel from the imaging
        BACKUP.xml.

        dX_red : float
            Red laser-path misalignment in x (column, + right) relative to
            green, in microns. The red movie is shifted by dX_red / um
            columns to register it onto green; negative means red sits to
            the LEFT of green. Default -5.79 (from the single-bead
            calibration 2026-06-03_z-003).
        dY_red : float
            Red laser-path misalignment in y relative to green, in
            microns, in a y-up convention. Negative means red sits BELOW
            green. Converted to a row shift via -dY_red / um (the array /
            imshow place row 0 at the top, so the y axis is inverted
            relative to the row index). Default -4.30 (from the single-bead
            calibration 2026-06-03_z-003).
        """
        self.ch_img_red = ch_img_red
        self.ch_img_grn = ch_img_grn
        self._fname_img_red = fname_img_red
        self._fname_img_grn = fname_img_grn
        self._dX_red = dX_red
        self._dY_red = dY_red
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
        """Load imaging data for a dual-colour recording.

        If both fname_img_red and fname_img_grn are None (and no dataset
        object supplies defaults), the channel tiffs are not loaded: only
        the suite2p folder in folder_img/suite2p and its summary neural
        activity are used. In that case self.rec_red, self.rec_grn and
        self.rec are set to None.
        """
        fname_img_red = getattr(self, '_fname_img_red', None)
        fname_img_grn = (fname_img if fname_img is not None
                         else getattr(self, '_fname_img_grn', None))

        # The sampling rate is always needed (to build rec_t) and is read
        # from BACKUP.xml independently of the tiffs.
        self.samp_rate = self._load_sampling_rate_from_backup_xml()

        # If neither channel filename is given (and no dataset defaults
        # apply), skip tiff loading entirely and rely on the suite2p folder
        # for neural activity.
        if (fname_img_red is None and fname_img_grn is None
                and dset_obj is None):
            print('no imaging filenames given; skipping tiff load '
                  '(suite2p only)')
            self.fname_img_red = None
            self.fname_img_grn = None
            self.fname_img = None
            self._rec_red_raw = None
            self._rec_grn_raw = None
            self._crop_px = n_px_remove_sides
            self.rec_red = None
            self.rec_grn = None
            self.rec = None
            return

        # get filenames of images for both channels
        list_img = os.listdir(self.folder.img)

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

        # Correct red/green laser-path misalignment: rigidly shift the
        # red movie onto the green channel. Replaces both raw arrays with
        # aligned overlap views (zero-copy), so every downstream consumer
        # (_get_rec_raw, cropping, correct_signal, QC) sees aligned data.
        # ----------
        self._align_red_to_grn_inplace()

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

    def _load_microns_per_pixel_from_backup_xml(self):
        """Parse microns-per-pixel from BACKUP.xml in the imaging folder.

        Returns
        -------
        microns : dict or None
            {'x': float, 'y': float} microns per pixel, or None if no
            BACKUP.xml could be parsed.
        """
        for _file in os.listdir(self.folder.img):
            if _file.endswith('BACKUP.xml'):
                try:
                    xmlobj = XMLParser(os.path.join(self.folder.img, _file))
                    return xmlobj.get_microns_per_pixel()
                except Exception:
                    print('\tcould not parse micronsPerPixel from '
                          'BACKUP.xml')
        return None

    @staticmethod
    def _red_shift_px(dX_red, dY_red, umperpx_x, umperpx_y):
        """Integer (row, col) shift describing red's displacement relative
        to green, in array-index space.

        Column follows x directly (off_col = dX_red / um): dX_red is red's
        column (x, + right) offset relative to green in microns, so a
        negative dX_red places red at a smaller column (to the left). Row
        uses a flipped sign (off_row = -dY_red / um) because the array /
        imshow place row 0 at the top with the row index increasing
        downward, so dY_red < 0 (red below green, in a y-up convention)
        maps to a LARGER red row index.

        Verified against the single-bead dual-colour z-stack calibration
        (2026-06-03_z-003): the measured red-minus-green bead offset is
        (row +173.6 px, col -234.0 px) = (+4.30 um down, -5.79 um in x) at
        0.02475 um/px, which this formula reproduces (residual ~4 px).

        Parameters
        ----------
        dX_red, dY_red : float
            Red-vs-green misalignment in microns. dX_red is the column
            (x, + right) offset; dY_red is the y-up offset (negative =
            red below green).
        umperpx_x, umperpx_y : float
            Microns per pixel for the x and y axes.

        Returns
        -------
        off_row, off_col : int
            Red's displacement (in pixels) relative to green: a red
            feature aligned with green index (r, c) is found in the raw
            red array at (r + off_row, c + off_col).
        """
        off_col = int(round(dX_red / umperpx_x))
        off_row = int(round(-dY_red / umperpx_y))
        return off_row, off_col

    @staticmethod
    def _align_red_grn(red_raw, grn_raw, off_row, off_col):
        """Return aligned (red, grn) views cropped to their common region.

        red is shifted by (off_row, off_col) onto green: the returned
        red[t, r, c] is taken from red_raw[t, r + off_row, c + off_col],
        and green is cropped to the same valid output region so both
        outputs share one shape. Both are pure views (zero copy).

        Parameters
        ----------
        red_raw, grn_raw : np.ndarray, shape (T, H, W)
            Raw red and green stacks (equal H, W).
        off_row, off_col : int
            Red displacement relative to green (see _red_shift_px).

        Returns
        -------
        red_aligned, grn_aligned : np.ndarray
            Views of shape (T, H - |off_row|, W - |off_col|).
        """
        H, W = red_raw.shape[-2], red_raw.shape[-1]
        r_lo, r_hi = max(0, -off_row), min(H, H - off_row)
        c_lo, c_hi = max(0, -off_col), min(W, W - off_col)
        red_aligned = red_raw[:,
                              r_lo + off_row:r_hi + off_row,
                              c_lo + off_col:c_hi + off_col]
        grn_aligned = grn_raw[:, r_lo:r_hi, c_lo:c_hi]
        return red_aligned, grn_aligned

    def _align_red_to_grn_inplace(self):
        """Shift self._rec_red_raw onto self._rec_grn_raw and store both
        aligned views back. Reads micronsPerPixel from BACKUP.xml to
        convert self._dX_red / self._dY_red (microns) to a pixel shift.

        No-op (identity) when the shift rounds to zero pixels or when
        micronsPerPixel is unavailable.
        """
        dX_red = getattr(self, '_dX_red', 0.0) or 0.0
        dY_red = getattr(self, '_dY_red', 0.0) or 0.0

        microns = self._load_microns_per_pixel_from_backup_xml()
        if microns is None:
            print('\tskipping red/grn alignment (no micronsPerPixel)')
            self.red_shift_px = (0, 0)
            return

        off_row, off_col = self._red_shift_px(
            dX_red, dY_red, microns['x'], microns['y'])
        self.red_shift_px = (off_row, off_col)
        self.ops.dX_red = dX_red
        self.ops.dY_red = dY_red
        self.ops.microns_per_pixel = microns

        if off_row == 0 and off_col == 0:
            print('\tred/grn alignment is 0 px; no shift applied')
            return

        print(f'\tcorrecting red/grn misalignment: '
              f'dX={dX_red:.2f}um, dY={dY_red:.2f}um -> '
              f'shift red by (row={off_row:+d}, col={off_col:+d}) px '
              f'(um/px x={microns["x"]:.3f}, y={microns["y"]:.3f})')

        self._rec_red_raw, self._rec_grn_raw = self._align_red_grn(
            self._rec_red_raw, self._rec_grn_raw, off_row, off_col)

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

    def _stim_off_times(self):
        """Per-trial visual-stimulus offset times (behaviour clock).

        The visual stimulus and its sync square turn off at
        ``stimulusOffTimes`` (task def: ``stimulusOff =
        stimulusOn.delay(stimulusDuration)``) — this is where the
        stimulus light-leak actually ends. Reward time is a *different*
        event (``rewardOnsetTime``) and coincides with stimulus offset
        only when ``stimulusDuration == rewardOnsetTime``; keying the
        step off reward instead silently halves the estimate (the OFF
        edge measures no jump) and mis-places the subtraction box
        whenever the two differ. Falls back to ``totalRewardTimes`` for
        older Blocks that never logged ``stimulusOffTimes``.

        Returns
        -------
        off_t : np.ndarray or None
            Per-trial offset times, or None if neither event is available.
        source : str or None
            'stimulusOffTimes' or 'totalRewardTimes' — which was used.
        """
        if not hasattr(self, 'beh') or not hasattr(self.beh, '_data'):
            return None, None
        for _var in ('stimulusOffTimes', 'totalRewardTimes'):
            try:
                _v = np.asarray(self.beh._data.get_event_var(_var),
                                dtype=float).ravel()
            except (AttributeError, KeyError, TypeError):
                continue
            if _v.size:
                return _v, _var
        return None, None

    def _remove_stim_step(self, sig, idx_start, idx_end, channel='red',
                          edge='both', n_edge=3, n_gap=1, amp_override=None,
                          batch_size=1000, verbose=True):
        """Wrap a channel in a view with its stim-step artefact removed.

        Estimates a single, spatially-uniform stimulus-triggered step
        amplitude A for `channel` from its whole-frame mean trace (sharp
        on/off edges, see signal_correction.estimate_stim_step_amplitude),
        then returns a lazy _StepRemovedView over `sig` that subtracts A
        while the visual stimulus is on screen (stim onset → reward time).
        Each channel is estimated independently. Memmap-safe: the whole-
        frame mean is computed in batches and the correction is applied at
        read-time. The per-channel estimate and offset vector are stashed in
        self.stim_step_amp[channel] / self.stim_step_offset[channel].

        Parameters
        ----------
        sig : (T, X, Y) array-like
            The (already time-sliced) raw channel to correct.
        idx_start, idx_end : int
            Frame range of `sig` within the full recording — used to slice
            the timestamps (self.rec_t) to match.
        channel : str
            Name of the channel being corrected ('red' or 'grn'); keys the
            stored estimate / offset and labels messages.
        edge : {'both', 'on', 'off'}
            Edges used to estimate A. 'on' is recommended for GRAB-DA, whose
            reward-evoked transient contaminates the reward-aligned OFF edge.
        n_edge, n_gap : int
            Frames averaged on each side of an edge / skipped at the
            transition.
        amp_override : float or None
            If given, use this amplitude instead of estimating from the data
            (applied to whichever channel this call corrects).
        batch_size : int
            Frames per batch for the whole-frame mean pass.
        verbose : bool

        Returns
        -------
        sig_view : signal_correction._StepRemovedView or original sig
            The step-removed view (or the unchanged input if stim timing is
            unavailable).
        """
        # Need timestamps and behaviour timing to define box(t).
        if not hasattr(self, 'rec_t') or not hasattr(self, 'beh'):
            if verbose:
                print('\tremove_stim_step: rec_t or beh missing — skipped')
            return sig

        t_slice = np.asarray(self.rec_t)[idx_start:idx_end]

        try:
            stim_on_t = np.asarray(self.beh.stim.t_start, dtype=float).ravel()
        except (AttributeError, KeyError, TypeError) as _e:
            if verbose:
                print(f'\tremove_stim_step: stim onset timing '
                      f'unavailable ({_e}) — skipped')
            return sig
        stim_off_t, _off_src = self._stim_off_times()
        if stim_off_t is None:
            if verbose:
                print('\tremove_stim_step: stim-off timing unavailable '
                      '— skipped')
            return sig
        if verbose and _off_src != 'stimulusOffTimes':
            print(f'\tremove_stim_step: stimulusOffTimes missing; '
                  f'falling back to {_off_src} for stim offset')

        # Whole-frame mean trace of the raw channel (one value/frame).
        T = sig.shape[0]
        ch_trace = np.empty(T, dtype=np.float64)
        for _t0 in range(0, T, batch_size):
            _t1 = min(_t0 + batch_size, T)
            _blk = np.asarray(sig[_t0:_t1], dtype=np.float64)
            ch_trace[_t0:_t1] = _blk.reshape(_blk.shape[0], -1).mean(axis=1)

        if amp_override is not None:
            amp = float(amp_override)
            if verbose:
                print(f'\tstim-step amplitude A[{channel}] = {amp:.3f} '
                      f'(user-supplied; raw units)')
        else:
            if verbose:
                print(f'\testimating stim step for {channel}:')
            amp, _info = signal_correction.estimate_stim_step_amplitude(
                ch_trace, t_slice, stim_on_t, stim_off_t,
                edge=edge, n_edge=n_edge, n_gap=n_gap, verbose=verbose)

        offset = signal_correction.build_stim_step_offset(
            t_slice, stim_on_t, stim_off_t, amp)

        # Stash per-channel for inspection / QC.
        if not isinstance(getattr(self, 'stim_step_amp', None), dict):
            self.stim_step_amp = {}
            self.stim_step_offset = {}
        self.stim_step_amp[channel] = amp
        self.stim_step_offset[channel] = offset
        self.stim_step_edge = edge

        if verbose:
            _n_on = int((offset != 0).sum())
            print(f'\tsubtracting stim step from {channel} over '
                  f'{_n_on}/{T} frames (stim-on epochs)')

        return signal_correction._StepRemovedView(sig, offset)

    def correct_signal(self, method='full_regress', save_to_disk=False,
                       t_start=None, t_end=None, t_end_pad=10.0,
                       replace_real=True, detrend=False,
                       static_flu='grn', real_flu='red',
                       marker_size=25,
                       trialavg_t_pre=2.0, trialavg_t_post=4.0,
                       remove_stim_step=False, stim_step_edge='both',
                       stim_step_n_edge=3, stim_step_n_gap=1,
                       stim_step_amp=None, stim_step_dff=False,
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
            - 'full_regress' (default): full-regression photometric
              correction (faithful port of the Martianova et al. 2019
              pipeline). Operates on the spatially-averaged 1-D control
              & signal traces — moving-average lowpass → airPLS baseline
              removal → trim warm-up frames → per-channel median/std
              normalisation → non-negative OLS slope — then applies a
              per-voxel normalised subtraction zdFF = s2_norm − β·s1_norm.
              Output is float16. See
              signal_correction.correct_full_regress for the full
              parameter list (smooth_window, airpls_lam, airpls_porder,
              airpls_max_iter, trim_initial, nn_slope, fit_mode,
              f0_level, f0_n_sectors).
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
            into a scratch `{base_real}_corr.tif` (beside the source
            channel in folder.img) via tifffile.memmap (no full-volume
            RAM allocation), THEN derives three small trial-averaged
            stim-aligned TIFFs — written to the recording's data_mbl/
            folder (self.folder.data) — by reading the relevant source
            one trial-window at a time into a (n_win, X, Y) float32
            accumulator.

            Every file written to data_mbl/ is prefixed
            `{animal}_{date}_{beh}_` (see `_rec_file_prefix`): that
            folder sits at the DATE level and is shared by all of a
            date's behaviour sessions, so names built from the imaging
            stem alone would collide — session 2 silently overwriting
            session 1, and readers unable to tell which session a file
            came from. Writing `{pfx}` for that prefix:

            - `{pfx}{base_real}_corr_trialavg.tif` — corrected stack in
              its native units (typically z-scored ΔF/F).
            - `{pfx}{base_real}_dff_trialavg.tif`  — raw real channel
              converted to dF/F (%) using each pixel's mean over the
              pre-stim baseline window.
            - `{pfx}{base_static}_dff_trialavg.tif` — same for the raw
              static channel.

            When the correction method is 'pixel_spatial_subtr' (either
            `segmentation_type`), the per-pixel corrected trace of
            every non-masked pixel (`self._corrected_info.mask_out`) is
            also written, as a further *pair* of files:

            - `{pfx}{base_real}_corr_pixeltraces_trial.npy` — a plain
              (n_trials, n_win, n_valid) array of the individual trial
              windows, in the corrected stack's own dtype (float16).
              It is streamed to disk one trial at a time via
              `np.lib.format.open_memmap`, so it is never resident in
              RAM regardless of how many GB it runs to.
            - `{pfx}{base_real}_corr_pixeltraces_trial_meta.npy` — a small
              pickled dict with the trial-average
              (`pixel_traces_trialavg`, (n_win, n_valid) float32), the
              `pixel_rows` / `pixel_cols` needed to scatter traces back
              onto the (X, Y) frame, `t_win` and the marker frames.

            Read both back with `utils_twop.load_pixel_trial_traces`
            (re-exported as `lak_exp.load_pixel_trial_traces`), which
            memmaps the per-trial cube rather than loading it.

            With `segmentation_type='cells'` on top of that, one more
            file — `{pfx}{base_real}_corr_celltraces_trialavg.npy` — is
            written: a pickled dict with the mean per-cell donut-ring
            dF/F trace aligned to stim onset (`cell_traces_trialavg`,
            (n_cells, n_win)), `cell_centroids`, `t_win` (seconds
            relative to stim onset), and the marker frame indices.

            All three TIFFs share the trial-avg window
            [−trialavg_t_pre, median(stim→rew) + trialavg_t_post]
            relative to each stim onset and the same white marker
            squares (see `marker_size`), so they can be overlaid
            frame-for-frame in an image viewer. During the call the
            full corrected stack file is kept as scratch:
            `self.rec_{real}_corr` (and `rec_{real}` if
            replace_real=True) points to it so downstream QC / sector
            extraction works against the disk-backed corrected stack,
            and its path is recorded on `self._cs_full_stack_path`.
            When the correction is driven through the QC pipeline,
            add_qc deletes this scratch stack after extracting the QC
            aggregates, so only the small trial-averaged TIFFs persist.
            (default: False)
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
        remove_stim_step : bool, optional
            If True, detect and remove a visual-stimulus-triggered step
            artefact from *both* channels (estimated independently per
            channel) *before* the control-channel correction. Light from
            the on-screen visual stimulus leaks onto the PMT(s), adding a
            spatially-uniform additive offset that switches on at stim onset
            and off at the visual-stimulus offset (``stimulusOffTimes``, with
            a ``totalRewardTimes`` fallback for older Blocks). Each channel's
            amplitude A is estimated from its whole-frame mean trace's sharp
            on/off edges and subtracted as a lazy read-time view (memmap-
            safe). The per-channel estimates are stored as dicts on
            self.stim_step_amp / self.stim_step_offset, keyed by 'red' /
            'grn'. NB: for method='pixel_spatial_subtr' prefer ``stim_step_dff``
            — the raw whole-frame subtraction cannot flatten the leak's
            spatial structure or its different fractional size per channel.
            (default: False)
        stim_step_edge : {'both', 'on', 'off'}, optional
            Which edges to use when estimating the step amplitude.
            'both' (default) averages the stim-onset jump and the reward-
            time drop. For a dopamine sensor (GRAB-DA), the reward-evoked
            transient contaminates the OFF edge — use 'on'. Ignored when
            stim_step_amp is provided. (default: 'both')
        stim_step_n_edge : int, optional
            Frames averaged on each side of an edge during estimation.
            (default: 3)
        stim_step_n_gap : int, optional
            Frames skipped right at each transition (the partially-
            illuminated frame). (default: 1)
        stim_step_amp : float or None, optional
            If given, use this fixed step amplitude instead of estimating it
            from the data. (default: None)
        stim_step_dff : bool, optional
            Per-pixel, dF/F-domain stim-step subtraction, for
            method='pixel_spatial_subtr' only. The correct alternative to
            ``remove_stim_step`` for that method: the unit-gain
            dff_sig − dff_ctrl subtraction amplifies an additive light-leak
            (whose fractional size differs between channels) and injects any
            real control-channel stim response, so the residual is best
            removed per pixel and in dF/F units *after* correction. The stim
            on/off frame indices are derived here from ``beh.stim.t_start``
            and the visual-stimulus offset (see ``_stim_off_times``) and
            passed to ``correct_pixel_spatial_subtr``; ``stim_step_edge`` /
            ``stim_step_n_edge`` / ``stim_step_n_gap`` control the per-pixel
            edge estimate. Mutually exclusive with ``remove_stim_step``.
            (default: False)
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

            For 'full_regress':
                smooth_window, airpls_lam, airpls_porder,
                airpls_max_iter, trim_initial, nn_slope, fit_mode,
                f0_level, f0_n_sectors (see
                signal_correction.correct_full_regress).

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

        Notes
        -----
        Also sets ``self.rec_{real_flu}_corr``. For
        ``method='pixel_spatial_subtr'`` the correction's ``info``
        SimpleNamespace (masks, den, F0, gate diagnostics, and — in cell
        mode — per-cell labels / ring traces) is stored on
        ``self._corrected_info``; it is set to None for every other method,
        so it always reflects the most recent correction.

        Examples
        --------
        >>> # Default full-regress correction (green controls red)
        >>> corrected = rec.correct_signal(method='full_regress')

        >>> # Correct green using red instead
        >>> corrected = rec.correct_signal(
        ...     method='full_regress', static_flu='red', real_flu='grn')

        >>> # Global (paper-faithful) β instead of per-pixel
        >>> corrected = rec.correct_signal(
        ...     method='full_regress', fit_mode='global')

        >>> # One β per sector (middle ground between global & per-pixel)
        >>> corrected = rec.correct_signal(
        ...     method='full_regress', fit_mode='per_sector', f0_n_sectors=8)

        >>> # Save corrected signal directly to disk (minimal RAM usage)
        >>> corrected = rec.correct_signal(
        ...     method='full_regress', save_to_disk=True)

        >>> # Process only first 60 seconds (for testing)
        >>> corrected = rec.correct_signal(method='full_regress', t_end=60.0)
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

        # Optionally remove a visual-stimulus-triggered step artefact from
        # BOTH channels before correction, estimated independently per
        # channel. Light from the on-screen visual stimulus leaks onto the
        # PMT(s), adding a spatially-uniform additive offset that switches on
        # at stim onset and off at stim offset (reward time). This is removed
        # first — as lazy read-time views — so the downstream regression sees
        # clean traces in both the control and the real channel.
        # ----------
        if remove_stim_step:
            self.stim_step_amp = {}
            self.stim_step_offset = {}
            _ss_verbose = kwargs.get('verbose', True)
            s1 = self._remove_stim_step(
                s1, idx_start=idx_start, idx_end=idx_end,
                channel=static_flu, edge=stim_step_edge,
                n_edge=int(stim_step_n_edge), n_gap=int(stim_step_n_gap),
                amp_override=stim_step_amp, verbose=_ss_verbose)
            s2 = self._remove_stim_step(
                s2, idx_start=idx_start, idx_end=idx_end,
                channel=real_flu, edge=stim_step_edge,
                n_edge=int(stim_step_n_edge), n_gap=int(stim_step_n_gap),
                amp_override=stim_step_amp, verbose=_ss_verbose)

        # Optionally enable per-pixel, dF/F-domain stim-step subtraction
        # inside pixel_spatial_subtr. Unlike remove_stim_step (raw, whole-
        # frame, applied to both channels before correction), this removes
        # the leak per pixel and in dF/F units from the *corrected* trace —
        # the correct tool for the unit-gain dff_sig - dff_ctrl subtraction,
        # which otherwise amplifies an additive leak (different fractional
        # size per channel) and injects any real control-channel signal.
        # We convert stim on/off *times* to frame indices into the corrected
        # stack (which begins at idx_start) here, so the correction needs no
        # behaviour access. Uses the visual-stimulus offset, not reward.
        # ----------
        if stim_step_dff:
            if method != 'pixel_spatial_subtr':
                raise ValueError(
                    "stim_step_dff is only supported for "
                    "method='pixel_spatial_subtr'")
            if remove_stim_step:
                raise ValueError(
                    "remove_stim_step (raw) and stim_step_dff (dF/F) both "
                    "subtract the stim step; enable only one")
            _on_t = np.asarray(self.beh.stim.t_start, dtype=float).ravel()
            _off_t, _off_src = self._stim_off_times()
            if _off_t is None:
                raise ValueError(
                    "stim_step_dff=True but stim-off timing is unavailable "
                    "(no stimulusOffTimes / totalRewardTimes)")
            _n = min(_on_t.size, _off_t.size)
            _on_t, _off_t = _on_t[:_n], _off_t[:_n]
            _te = getattr(self, 'trial_end', None)
            if _te is not None:
                _on_t = _on_t[:int(_te) + 1]
                _off_t = _off_t[:int(_te) + 1]
            _rec_t = np.asarray(self.rec_t)
            kwargs['stim_step_dff'] = True
            kwargs['stim_on_frames'] = np.searchsorted(_rec_t, _on_t) \
                - idx_start
            kwargs['stim_off_frames'] = np.searchsorted(_rec_t, _off_t) \
                - idx_start
            kwargs['stim_step_edge'] = stim_step_edge
            kwargs['stim_step_n_edge'] = int(stim_step_n_edge)
            kwargs['stim_step_n_gap'] = int(stim_step_n_gap)
            if kwargs.get('verbose', True):
                _src = ('' if _off_src == 'stimulusOffTimes'
                        else f' (stim-off from {_off_src})')
                print(f'\tstim_step_dff: per-pixel dF/F step subtraction '
                      f'over {_on_t.size} trials, edge={stim_step_edge}'
                      f'{_src}')

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
        #                   correction itself, and self.rec_{real}_corr
        #                   points to it so the QC pipeline can read
        #                   sectors and the trial-avg pass can stream
        #                   over it. It is a *scratch* file only: the
        #                   path is recorded on self._cs_full_stack_path
        #                   so add_qc can delete it once the trial-avg
        #                   TIFFs and QC aggregates have been derived
        #                   (only the small trial-averaged TIFFs persist).
        # - _trialavg_path: small (n_win, X, Y) trial-averaged TIFF
        #                   derived from the full stack. Written via
        #                   tifffile.memmap so the output is also
        #                   disk-backed.
        # ----------
        output_path = None
        _trialavg_path = None
        self._cs_full_stack_path = None
        if save_to_disk:
            _fname_real = getattr(self, f'fname_img_{real_flu}')
            base_name, ext = os.path.splitext(_fname_real)
            # Every derived file below lands in self.folder.data
            # (data_mbl/), which is shared by ALL behaviour sessions of a
            # date. The imaging stem alone ('compiled_Ch1') is identical
            # across those sessions, so naming by it let session 2
            # silently overwrite session 1's outputs, and left readers
            # unable to attribute a file to a session. Prefix them with
            # the recording id — the same {animal}_{date}_{beh}
            # convention the QC figures and .npy summaries already use.
            _rec_prefix = self._rec_file_prefix()
            if method == 'full_regress':
                # For full_regress, encode the correction kwargs in the
                # filename so the saved stack records its
                # parameterisation. The tag is capped at whatever length
                # is left over once the prefix, the recording's own stem
                # and the longest suffix any file built from _corr_base
                # carries are accounted for — otherwise a long stem plus
                # ~17 spelled-out kwargs overflows the 255-byte limit on
                # a name component (OSError errno 63).
                _tag_budget = (_FNAME_MAX_LEN - len(_rec_prefix)
                               - len(base_name)
                               - len('_corr_') - _CORR_BASE_MAX_SUFFIX)
                _kw_tag = _full_regress_kwarg_tag(
                    kwargs, max_len=min(96, _tag_budget))
                _stem_base = f"{base_name}_corr_{_kw_tag}"
            else:
                _stem_base = f"{base_name}_corr"
            # data_mbl/ is shared across sessions -> prefixed stem.
            _corr_base = f"{_rec_prefix}{_stem_base}"
            # The scratch full stack goes to folder.img, which is already
            # per-recording, so it keeps its historical unprefixed name.
            output_fname = f"{_stem_base}{ext}"
            # Scratch full stack stays beside the source channel in
            # folder.img (it is deleted by add_qc); only the small
            # trial-averaged TIFFs persist, and those go to data_mbl/.
            output_path = os.path.join(self.folder.img, output_fname)
            _trialavg_path = os.path.join(
                str(self.folder.data),
                f"{_corr_base}_trialavg{ext}")
            kwargs['output_path'] = output_path
            # Record the scratch full-stack path so add_qc can remove it
            # after deriving the trial-avg TIFFs and QC aggregates.
            self._cs_full_stack_path = output_path

        method_map = {
            'full_regress':
                lambda: signal_correction.correct_full_regress(
                    s1, s2, **kwargs),
            'two_stage':
                lambda: signal_correction.correct_two_stage_regress(
                    s1, s2, **kwargs),
            'pixel_spatial_subtr':
                lambda: signal_correction.correct_pixel_spatial_subtr(
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
        # supported by full_regress), the "signal" is actually a dict of
        # whole-frame + per-sector mean traces — short-circuit and stash on
        # self.rec_{real}_corr_aggregates rather than walking the
        # save_to_disk / replace_real paths that assume a full (T, X, Y)
        # stack. The QC pipeline reads this attribute directly.
        # ----------
        corrected = result[0]

        # Stash the pixel_spatial_subtr correction's info namespace (masks,
        # den, F0, gate diagnostics, and — in cell mode — per-cell labels /
        # ring traces) so callers and the QC pipeline can inspect it after a
        # plt_qc / add_qc (or direct) run. Only this method returns an info
        # SimpleNamespace as its second value; the others return arrays or
        # component indices there, so _corrected_info is cleared for them to
        # keep it unambiguously tied to the most recent correction.
        if method == 'pixel_spatial_subtr' and len(result) > 1:
            self._corrected_info = result[1]
        else:
            self._corrected_info = None

        if isinstance(corrected, dict) and corrected.get('is_aggregates'):
            setattr(self, f'rec_{real_flu}_corr_aggregates', corrected)
            return

        # save_to_disk path: write three trial-averaged TIFFs derived
        # from the same stim window — the corrected stack (per-pixel
        # baseline-subtracted z-scored ΔF/F, so the pre-stim period is
        # uniformly black and only responsive pixels light up after
        # stim), and dF/F (%) trial-averages of the raw real and static
        # channels. All three share the same n_win / stim-frame /
        # rew-frame layout and markers so they can be overlaid
        # frame-for-frame in an image viewer.
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
                baseline_subtract=True,
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
                    str(self.folder.data),
                    f'{_rec_prefix}{_base}_dff_trialavg.tif')
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

            # Per-pixel (non-masked, i.e. mask_out=True) corrected
            # traces, individual-trial windows AND their trial-average,
            # for whichever segmentation_type ('gmm' or 'cells') the
            # correction used — mask_out always exists on
            # self._corrected_info for pixel_spatial_subtr.
            # ----------
            if (self._corrected_info is not None
                    and getattr(self._corrected_info, 'mask_out', None)
                    is not None):
                _pix_path = os.path.join(
                    str(self.folder.data),
                    f'{_corr_base}_pixeltraces_trial.npy')
                try:
                    self._save_pixel_trial_traces_to_disk(
                        source=corrected,
                        idx_start=idx_start,
                        output_path=_pix_path,
                        t_pre=float(trialavg_t_pre),
                        t_post=float(trialavg_t_post),
                        verbose=_verb)
                except Exception as _e:
                    if _verb:
                        print(f'\twarning: per-pixel trial traces '
                              f'failed ({_e})')

            # Cell mode also emits mean per-cell donut-ring traces,
            # trial-aligned to stim onset, alongside the trial-avg TIFFs.
            # ----------
            if (kwargs.get('segmentation_type') == 'cells'
                    and self._corrected_info is not None
                    and getattr(self._corrected_info, 'cell_traces', None)
                    is not None):
                _cell_path = os.path.join(
                    str(self.folder.data),
                    f'{_corr_base}_celltraces_trialavg.npy')
                try:
                    self._save_cell_trialavg_traces_to_disk(
                        idx_start=idx_start,
                        output_path=_cell_path,
                        t_pre=float(trialavg_t_pre),
                        t_post=float(trialavg_t_post),
                        verbose=_verb)
                except Exception as _e:
                    if _verb:
                        print(f'\twarning: per-cell trial-avg traces '
                              f'failed ({_e})')

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
                                  baseline_subtract=False,
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
        baseline_subtract : bool
            If True, subtract each pixel's mean over the pre-stim
            baseline window (first `_n_pre` frames) from the whole
            trial-averaged trace — a pure SUBTRACTION, no divide. This
            is the correct baseline normalisation for an already-
            normalised input (e.g. the z-scored ΔF/F corrected channel,
            whose per-pixel baseline ≈ 0 makes a dF/F divide unstable).
            The pre-stim period becomes ≈ 0 everywhere (uniformly black)
            and only stim-responsive pixels deviate, so anatomical
            structure no longer shows through the baseline. Output dtype
            becomes float32. Mutually exclusive with `dff`. The white
            frame markers are placed at an on-scale bright value
            (response 99.9th percentile) rather than the dtype max, so
            they do not crush the display contrast.
        label : str
            Label used in verbose log lines (e.g. 'red', 'grn',
            'corrected').
        verbose : bool
        """
        if not hasattr(source, 'shape') or source.ndim != 3:
            raise ValueError(
                f"source must be 3D (T, X, Y); got "
                f"{getattr(source, 'shape', type(source))}")
        if dff and baseline_subtract:
            raise ValueError(
                "dff and baseline_subtract are mutually exclusive")
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
            _mode = ('dF/F %' if dff
                     else 'baseline-subtracted' if baseline_subtract
                     else 'native')
            print(f'\ttrial-averaging [{label}] ({_mode}):')
            print(f'\t\twindow: [-{t_pre:.2f}, '
                  f'{_rew_lat + t_post:.2f}]s '
                  f'(median stim→rew={_rew_lat:.2f}s)')
            print(f'\t\tframes: {_n_win} ({_n_pre} pre, {_n_post} post), '
                  f'dt={_dt:.4f}s')
            print(f'\t\tstim events in range: {_stim_t.size}')

        # Accumulator in float32 for precision; one window-sized
        # buffer (~n_win × X × Y × 4 bytes) plus one per-trial chunk
        # of the same size, transiently. NaN-aware: a per-trial (f0_mode=
        # 'per_trial') corrected stack is NaN outside each trial's window
        # and at non-mask_out pixels, so accumulation is done with
        # nansum plus a per-cell finite count. A (frame, pixel) cell is
        # averaged over however many trials actually covered it, and stays
        # NaN only where no trial contributed a finite value.
        # ----------
        _acc = np.zeros((_n_win, X, Y), dtype=np.float32)
        _n_fin = np.zeros((_n_win, X, Y), dtype=np.int32)
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
            _fin = np.isfinite(_chunk)
            np.add(_acc, np.where(_fin, _chunk, np.float32(0.0)), out=_acc)
            _n_fin += _fin
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

        with np.errstate(invalid='ignore', divide='ignore'):
            _avg_f32 = (_acc / _n_fin).astype(np.float32)
        _avg_f32[_n_fin == 0] = np.nan
        del _acc, _n_fin

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
            with np.errstate(invalid='ignore'):
                _f0 = np.nanmean(_avg_f32[:_n_pre], axis=0)
            _f0_safe = np.where(np.abs(_f0) >= dff_eps, _f0, 1.0)
            with np.errstate(invalid='ignore', divide='ignore'):
                _avg_out = ((_avg_f32 - _f0[None]) / _f0_safe[None]
                            * 100.0)
            _bad = np.abs(_f0) < dff_eps
            if np.any(_bad):
                _avg_out[:, _bad] = 0.0
            _out_dtype = np.float32
        elif baseline_subtract:
            # Per-pixel pre-stim baseline SUBTRACTION (no divide). The
            # baseline window becomes ≈ 0 for every pixel, so the stack
            # is uniformly black until stim and only responsive pixels
            # deviate. Correct for already-normalised inputs (z-scored
            # ΔF/F) where a dF/F divide would blow up on near-zero F0.
            if _n_pre < 1:
                raise ValueError(
                    f"baseline_subtract=True requires at least 1 "
                    f"pre-stim frame; got _n_pre={_n_pre} "
                    f"(t_pre={t_pre}s, dt={_dt}s)")
            with np.errstate(invalid='ignore'):
                _f0 = np.nanmean(_avg_f32[:_n_pre], axis=0)
            _avg_out = _avg_f32 - _f0[None]
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
            if baseline_subtract:
                # On-scale bright marker: the 99.9th percentile of the
                # (signed) response, so the marker is the brightest thing
                # on screen without saturating the auto-contrast that
                # keeps the baseline black. Fall back to the data max if
                # the percentile is non-finite or non-positive.
                _hi = float(np.nanpercentile(np.asarray(_out), 99.9))
                if not np.isfinite(_hi) or _hi <= 0:
                    _hi = float(np.nanmax(np.asarray(_out)))
                if not np.isfinite(_hi) or _hi <= 0:
                    _hi = 1.0
                _white = _dt_norm.type(_hi)
            elif np.issubdtype(_dt_norm, np.integer):
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

    def _save_pixel_trial_traces_to_disk(self, source, idx_start,
                                           output_path,
                                           t_pre=2.0, t_post=4.0,
                                           dtype=None, flush_every=10,
                                           verbose=True):
        """Save per-trial AND trial-averaged traces for valid pixels.

        Companion to ``_save_trial_avg_to_disk`` /
        ``_save_cell_trialavg_traces_to_disk``: rather than only keeping
        the across-trial average image stack, this keeps every trial's
        window individually for every non-masked pixel
        (``self._corrected_info.mask_out``), which per-pixel statistics
        across trials need and a spatial/trial average would wash out.
        Works identically for ``segmentation_type='gmm'`` and
        ``'cells'`` — ``mask_out`` is populated either way.

        Two files are written:

        - ``output_path`` — a plain (n_trials, n_win, n_valid) .npy
          array, allocated with ``np.lib.format.open_memmap`` and
          filled one trial at a time. The cube is *never* resident in
          RAM, either while building it or while saving it, so peak
          usage is bounded by one (n_win, n_valid) window plus the
          two (n_win, n_valid) trial-average accumulators.
        - ``{stem}_meta.npy`` — a small pickled dict holding the
          trial-average and everything needed to interpret the cube
          (pixel_rows / pixel_cols, t_win, marker frames, shapes).

        Read both back with
        ``utils_twop.load_pixel_trial_traces``, which memmaps the cube
        rather than loading it.

        Parameters
        ----------
        source : np.ndarray or memmap, (T, X, Y)
            Corrected per-pixel stack (e.g. the return value of
            ``correct_pixel_spatial_subtr`` / ``self.rec_{real}_corr``).
        idx_start : int
            Frame offset into self.rec_t at which source[0] lives.
        output_path : str
            Destination .npy path for the per-trial cube. The metadata
            sidecar path is derived from it.
        t_pre, t_post : float
            See ``_save_trial_avg_to_disk``.
        dtype : np.dtype or None
            Storage dtype of the per-trial cube. None (default) keeps
            `source`'s own dtype when it is a float type — the
            corrected stack is float16, so this neither loses precision
            nor doubles the file — and falls back to float32 otherwise.
        flush_every : int
            Flush the memmap every this many trials, to bound dirty
            page accumulation when writing to a slow/external volume.
        verbose : bool

        Notes
        -----
        Output size scales as ``n_trials * n_win * n_valid_pixels``,
        which can be large on disk (``n_valid`` is typically about a
        third to a half of the FOV) even though RAM stays bounded.
        """
        info = self._corrected_info
        mask_out = np.asarray(info.mask_out, dtype=bool)
        if not mask_out.any():
            raise ValueError(
                "mask_out has no valid pixels; nothing to save")
        _rows, _cols = np.nonzero(mask_out)
        n_valid = _rows.size

        if not hasattr(source, 'shape') or source.ndim != 3:
            raise ValueError(
                f"source must be 3D (T, X, Y); got "
                f"{getattr(source, 'shape', type(source))}")
        T, X, Y = source.shape
        if mask_out.shape != (X, Y):
            raise ValueError(
                f"mask_out shape {mask_out.shape} does not match "
                f"source frame shape {(X, Y)}")

        _t_local = np.asarray(self.rec_t[idx_start:idx_start + T],
                              dtype=np.float64)
        if _t_local.size < 2:
            raise ValueError(
                f"need at least 2 frames of rec_t for pixel trial "
                f"traces; got {_t_local.size}")
        _dt = float(np.median(np.diff(_t_local)))

        _stim_t = np.asarray(getattr(self.beh.stim, 't_start', []),
                             dtype=np.float64).ravel()
        _stim_t = _stim_t[np.isfinite(_stim_t)]
        _stim_in = ((_stim_t >= _t_local[0])
                    & (_stim_t <= _t_local[-1]))
        _stim_t = _stim_t[_stim_in]

        # Reward times, paired to stims, only to size the window and
        # place the reward marker frame (same logic as
        # _save_trial_avg_to_disk).
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

        _n_pre = int(round(t_pre / _dt))
        _n_post = int(round((_rew_lat + t_post) / _dt))
        _n_win = _n_pre + _n_post + 1
        _stim_frame_in_win = _n_pre
        _rew_frame_in_win = max(
            0, min(_n_pre + int(round(_rew_lat / _dt)), _n_win - 1))

        # First pass: which stim events have a fully in-range window.
        # ----------
        _win_starts = []
        for _ev in _stim_t:
            _i = int(np.argmin(np.abs(_t_local - _ev)))
            _i0 = _i - _n_pre
            _i1 = _i + _n_post + 1
            if _i0 < 0 or _i1 > T:
                continue
            _win_starts.append(_i0)
        n_trials = len(_win_starts)

        # Storage dtype: keep source's own float dtype (the corrected
        # stack is float16, so an upcast would double both RAM and file
        # size without adding precision), but never silently downcast a
        # non-float source.
        # ----------
        if dtype is None:
            _src_dt = np.dtype(getattr(source, 'dtype', np.float32))
            _dtype = _src_dt if np.issubdtype(_src_dt, np.floating) \
                else np.dtype(np.float32)
        else:
            _dtype = np.dtype(dtype)

        _meta_path = f'{os.path.splitext(output_path)[0]}_meta.npy'

        if verbose:
            _gb = n_trials * _n_win * n_valid * _dtype.itemsize / 1e9
            print(f'\tper-pixel trial traces (valid pixels only):')
            print(f'\t\twindow: [-{t_pre:.2f}, '
                  f'{_rew_lat + t_post:.2f}]s '
                  f'(median stim->rew={_rew_lat:.2f}s)')
            print(f'\t\tframes: {_n_win} ({_n_pre} pre, {_n_post} post), '
                  f'dt={_dt:.4f}s, valid pixels: {n_valid}')
            print(f'\t\tincluded {n_trials}/{_stim_t.size} trials')
            print(f'\t\tstreaming {_gb:.2f} GB as {_dtype} to: '
                  f'{output_path}')

        if n_trials == 0:
            raise ValueError(
                "no trials survived the window filter; "
                "cannot extract per-pixel trial traces")

        # The per-trial cube is written straight into a disk-backed .npy
        # (mode='w+' creates/truncates), one trial window at a time, and
        # the trial-average is accumulated NaN-aware alongside it — same
        # pattern as _save_trial_avg_to_disk, but reduced to valid
        # pixels. Nothing of size (n_trials, n_win, n_valid) is ever
        # held in RAM, which matters: at ~90k valid pixels and ~120
        # window frames the cube runs to several GB.
        # ----------
        pertrial = np.lib.format.open_memmap(
            output_path, mode='w+', dtype=_dtype,
            shape=(n_trials, _n_win, n_valid))
        _acc = np.zeros((_n_win, n_valid), dtype=np.float32)
        _n_fin = np.zeros((_n_win, n_valid), dtype=np.int32)
        try:
            for _k, _i0 in enumerate(_win_starts):
                # Fancy-index the source view directly so the temporary
                # is (n_win, n_valid), not a full (n_win, X, Y) frame
                # block that is then thrown away.
                _win = np.asarray(source[_i0:_i0 + _n_win])[:, _rows, _cols]
                pertrial[_k] = _win

                _w32 = _win.astype(np.float32, copy=False)
                _fin = np.isfinite(_w32)
                np.add(_acc, np.where(_fin, _w32, np.float32(0.0)), out=_acc)
                _n_fin += _fin

                if flush_every and (_k + 1) % int(flush_every) == 0:
                    pertrial.flush()
                if verbose:
                    print(f'\t\t\ttrial {_k + 1}/{n_trials}', end='\r')
            pertrial.flush()
        finally:
            del pertrial
        if verbose:
            print()

        with np.errstate(invalid='ignore', divide='ignore'):
            trialavg = (_acc / _n_fin).astype(np.float32)
        trialavg[_n_fin == 0] = np.nan
        del _acc, _n_fin

        _t_win = (np.arange(_n_win) - _stim_frame_in_win) * _dt

        if verbose:
            print(f'\twriting per-pixel trace metadata to: {_meta_path}')
        np.save(_meta_path, {
            'pixel_traces_trialavg': trialavg,
            'pixel_rows': _rows.astype(np.int32),
            'pixel_cols': _cols.astype(np.int32),
            'frame_shape': (int(X), int(Y)),
            't_win': _t_win,
            'stim_frame_in_win': int(_stim_frame_in_win),
            'rew_frame_in_win': int(_rew_frame_in_win),
            'n_trials': int(n_trials),
            'n_win': int(_n_win),
            'n_valid': int(n_valid),
            'pertrial_dtype': str(_dtype),
            'pertrial_file': os.path.basename(output_path),
        }, allow_pickle=True)

    def _save_cell_trialavg_traces_to_disk(self, idx_start, output_path,
                                             t_pre=2.0, t_post=4.0,
                                             verbose=True):
        """Trial-average the per-cell donut-ring dF/F traces and save.

        Companion to ``_save_trial_avg_to_disk``: instead of a (T, X, Y)
        image stack, this averages ``self._corrected_info.cell_traces``
        (K, T) — the per-cell ring traces emitted by
        ``signal_correction.correct_pixel_spatial_subtr`` under
        ``segmentation_type='cells'`` — over the same stim-aligned
        window [-t_pre, median(stim->rew) + t_post]. NaN-aware: a cell
        with no finite samples in a given window frame (e.g. outside its
        trial window under ``f0_mode='per_trial'``) stays NaN there
        rather than pulling the mean toward 0.

        Parameters
        ----------
        idx_start : int
            Frame offset into self.rec_t at which cell_traces[:, 0] lives.
        output_path : str
            Destination .npy path. Saved as a single pickled dict (via
            ``np.save(..., allow_pickle=True)``).
        t_pre, t_post : float
            See ``_save_trial_avg_to_disk``.
        verbose : bool
        """
        info = self._corrected_info
        if info is None or getattr(info, 'cell_traces', None) is None:
            raise ValueError(
                "no per-cell traces available; correct_signal must have "
                "been run with method='pixel_spatial_subtr' and "
                "segmentation_type='cells'")
        traces = np.asarray(info.cell_traces)
        n_cells, T = traces.shape

        _t_local = np.asarray(self.rec_t[idx_start:idx_start + T],
                              dtype=np.float64)
        if _t_local.size < 2:
            raise ValueError(
                f"need at least 2 frames of rec_t for cell trial-avg; "
                f"got {_t_local.size}")
        _dt = float(np.median(np.diff(_t_local)))

        _stim_t = np.asarray(getattr(self.beh.stim, 't_start', []),
                             dtype=np.float64).ravel()
        _stim_t = _stim_t[np.isfinite(_stim_t)]
        _stim_in = ((_stim_t >= _t_local[0])
                    & (_stim_t <= _t_local[-1]))
        _stim_t = _stim_t[_stim_in]

        # Reward times, paired to stims, only to size the window and
        # place the reward marker frame (same logic as
        # _save_trial_avg_to_disk).
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

        _n_pre = int(round(t_pre / _dt))
        _n_post = int(round((_rew_lat + t_post) / _dt))
        _n_win = _n_pre + _n_post + 1
        _stim_frame_in_win = _n_pre
        _rew_frame_in_win = max(
            0, min(_n_pre + int(round(_rew_lat / _dt)), _n_win - 1))

        if verbose:
            print(f'\ttrial-averaging [cell donut traces]:')
            print(f'\t\twindow: [-{t_pre:.2f}, '
                  f'{_rew_lat + t_post:.2f}]s '
                  f'(median stim->rew={_rew_lat:.2f}s)')
            print(f'\t\tframes: {_n_win} ({_n_pre} pre, {_n_post} post), '
                  f'dt={_dt:.4f}s, cells: {n_cells}')
            print(f'\t\tstim events in range: {_stim_t.size}')

        _acc = np.zeros((n_cells, _n_win), dtype=np.float32)
        _n_fin = np.zeros((n_cells, _n_win), dtype=np.int32)
        _count = 0
        for _ev in _stim_t:
            _i = int(np.argmin(np.abs(_t_local - _ev)))
            _i0 = _i - _n_pre
            _i1 = _i + _n_post + 1
            if _i0 < 0 or _i1 > T:
                continue
            _chunk = traces[:, _i0:_i1].astype(np.float32)
            _fin = np.isfinite(_chunk)
            np.add(_acc, np.where(_fin, _chunk, np.float32(0.0)), out=_acc)
            _n_fin += _fin
            _count += 1

        if verbose:
            print(f'\t\tincluded {_count}/{_stim_t.size} trials')
        if _count == 0:
            raise ValueError(
                "no trials survived the window filter; "
                "cannot trial-average cell traces")

        with np.errstate(invalid='ignore', divide='ignore'):
            _avg = (_acc / _n_fin).astype(np.float32)
        _avg[_n_fin == 0] = np.nan
        del _acc, _n_fin

        _t_win = (np.arange(_n_win) - _stim_frame_in_win) * _dt

        if verbose:
            print(f'\twriting per-cell trial-averaged traces to: '
                  f'{output_path}')
        np.save(output_path, {
            'cell_traces_trialavg': _avg,
            'cell_centroids': (np.asarray(info.cell_centroids)
                                if info.cell_centroids is not None
                                else None),
            't_win': _t_win,
            'stim_frame_in_win': int(_stim_frame_in_win),
            'rew_frame_in_win': int(_rew_frame_in_win),
            'n_trials': int(_count),
        }, allow_pickle=True)

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
