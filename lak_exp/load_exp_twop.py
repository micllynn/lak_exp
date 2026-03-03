from types import SimpleNamespace
import numpy as np
import os
import pathlib
import tifffile
import warnings

from .align_imgbeh import Aligner_ImgBeh
from .utils import find_event_onsets_autothresh
from .utils_twop import XMLParser
from .beh import BehDataSimpleLoad, StimParserNew
from .exp_defs import ExpSubtypes
from . import signal_correction
from .twop_analysis import AnalysisMixin
from .twop_plots import PlotsMixin

# Suppress NumPy RuntimeWarnings in this module.
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module=r"^numpy(\.|$)")


class TwoPRec(AnalysisMixin, PlotsMixin):
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
                            subtypes=subtypes)

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
                       subtypes=None):
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
        self.beh.stim.stimlist = StimParserNew(self.beh, parse_by=parse_by,
                                               subtypes=subtypes)
        self.beh._stimparser = self.beh.stim.stimlist

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
        self.rec = tifffile.memmap(os.path.join(
            self.folder.img, self.fname_img))[
            :, n_px_remove_sides:-1*n_px_remove_sides,
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
                 n_px_remove_sides=10):
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
                         n_px_remove_sides=n_px_remove_sides)
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

        # Load GREEN channel (keep raw memmap, crop lazily via _get_rec)
        print('\tloading green channel tiff...')
        self._rec_grn_raw = tifffile.memmap(os.path.join(
            self.folder.img, self.fname_img_grn))

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
            return self._rec_red_raw, self._crop_px
        elif channel == 'grn':
            # Use original green if available (for correct_signal re-runs)
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
        >>> corrected = rec.correct_signal(
        ...     method='robust', huber_epsilon=1.5)

        >>> # Adaptive filter correction
        >>> corrected = rec.correct_signal(
        ...     method='lms', filter_order=20, mu=0.005)

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
        # on shared noise (motion, haemodynamics)
        # rather than trend differences.
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
        if not hasattr(self, 'rec_grn_corr'):
            raise RuntimeError(
                "rec_grn_corr not found. Run correct_signal() first.")

        if fname is None:
            base_name, ext = os.path.splitext(self.fname_img_grn)
            method_tag = getattr(self, 'corr_sig_method', 'corr')
            fname = f"{base_name}_corr_{method_tag}{ext}"

        output_path = os.path.join(self.folder.img, fname)
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
