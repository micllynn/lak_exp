import numpy as np
import scipy.io as sp_io

from types import SimpleNamespace
import os
import copy
from .utils import dtype_to_list, convert_rewvalue_into_floats, \
    find_event_onsets, find_event_onsets_autothresh, \
    smooth_squarekern, bin_signal


class BlockParser(object):
    def __init__(self, fname):
        """
        Parses a *_Block.mat file of behavior data and extracts timestamps
        and other behavioral variables.

        Can list and extract:
            - block variables (object['block'][varname])
            - event variables corresponding to named task events
            (object['block']['events'][()][varname]).
            - param variables (object['block']['paramsValues'][()][varname])
            - input variables (object['block']['inputs'][()][varname])
            - output variables (object['block']['outputs'][()][varname])
        """
        self.blk = sp_io.loadmat(fname, squeeze_me=True)
        self.aligner_obj = False  # placeholder for aligner object if present

    def list_block_vars(self):
        """
        List all available block-level variables.

        Returns
        -------
        blockvars : list of str
            Names of all block-level variables.
        """
        blockvars = dtype_to_list(self.blk['block'].dtype)
        return blockvars

    def list_event_vars(self):
        """
        List all available event variables.

        Returns
        -------
        eventvars : list of str
            Names of all event variables.
        """
        eventvars = dtype_to_list(self.blk['block']['events'][()].dtype)
        return eventvars

    def list_param_vars(self):
        """
        List all available parameter variables.

        Returns
        -------
        paramvars : list of str
            Names of all parameter variables.
        """
        paramvars = dtype_to_list(self.blk['block']['paramsValues'][()].dtype)
        return paramvars

    def list_input_vars(self):
        """
        List all available input variables.

        Returns
        -------
        inputvars : list of str
            Names of all input variables.
        """
        inputvars = dtype_to_list(self.blk['block']['inputs'][()].dtype)
        return inputvars

    def list_output_vars(self):
        """
        List all available output variables.

        Returns
        -------
        outputvars : list of str
            Names of all output variables.
        """
        outputvars = dtype_to_list(self.blk['block']['outputs'][()].dtype)
        return outputvars

    def get_block_var(self, var):
        """
        Get a specific block-level variable.

        Parameters
        ----------
        var : str
            Name of the block variable to retrieve.

        Returns
        -------
        blockvar : array-like
            Deep copy of the requested block variable.
        """
        blockvar = copy.deepcopy(self.blk['block'][var])
        return blockvar

    def get_event_var(self, var):
        """
        Get a specific event variable.

        Parameters
        ----------
        var : str
            Name of the event variable to retrieve.

        Returns
        -------
        eventvar : array-like
            Deep copy of the requested event variable. If var is 'totalRewardValues',
            automatically converts to float format.
        """
        eventvar = copy.deepcopy(self.blk['block']['events'][()][var][()])

        if var == 'totalRewardValues':
            eventvar = convert_rewvalue_into_floats(eventvar)

        return eventvar

    def get_param_var(self, var):
        """
        Get a specific parameter variable.

        Parameters
        ----------
        var : str
            Name of the parameter variable to retrieve.

        Returns
        -------
        paramvars : array-like
            Deep copy of the requested parameter variable.
        """
        paramvars = copy.deepcopy(self.blk['block']['paramsValues'][()][var])
        return paramvars

    def get_input_var(self, var):
        """
        Get a specific input variable.

        Parameters
        ----------
        var : str
            Name of the input variable to retrieve.

        Returns
        -------
        inputvar : array-like
            Deep copy of the requested input variable.
        """
        inputvar = copy.deepcopy(self.blk['block']['inputs'][()][var][()])
        return inputvar

    def get_output_var(self, var):
        """
        Get a specific output variable.

        Parameters
        ----------
        var : str
            Name of the output variable to retrieve.

        Returns
        -------
        outputvar : array-like
            Deep copy of the requested output variable.
        """
        outputvar = copy.deepcopy(self.blk['block']['outputs'][()][var][()])
        return outputvar


class TimelineParser(object):
    def __init__(self, fname):
        """
        Parses a *_Timeline.mat file of behavior data and extracts timestamps
        and other behavioral variables.

        Can list and extract:
            - block variables (object['block'][varname])
            - event variables corresponding to named task events
            (object['block']['events'][()][varname]).
            - param variables (object['block']['paramsValues'][()][varname])
            - input variables (object['block']['inputs'][()][varname])
            - output variables (object['block']['outputs'][()][varname])
        """
        self.tl = sp_io.loadmat(fname, squeeze_me=True)

    def list_hw_params(self):
        """
        List and print all hardware parameters.

        Returns
        -------
        None
            Prints hardware parameter names.
        """
        hwparams = dtype_to_list(self.tl['Timeline'][()]['hw'].dtype)
        print(hwparams)

    def list_hw_inputs_raw(self):
        """
        List and print raw hardware input variables.

        Returns
        -------
        None
            Prints raw hardware input variables.
        """
        inputvars = self.tl['Timeline'][()]['hw']['inputs'][()]
        print(inputvars)

    def list_hw_input_params(self):
        """
        List and print hardware input parameter names.

        Returns
        -------
        None
            Prints hardware input parameter names.
        """
        input_params = dtype_to_list(
            self.tl['Timeline'][()]['hw']['inputs'][()].dtype)
        print(input_params)

    def get_hw_input_param(self, param):
        """
        Get a specific hardware input parameter.

        Parameters
        ----------
        param : str
            Name of the hardware input parameter to retrieve.

        Returns
        -------
        input_param : array-like
            Requested hardware input parameter.
        """
        input_param = self.tl['Timeline'][()]['hw']['inputs'][()][param]
        return input_param

    def get_daq_samplerate(self):
        """
        Get DAQ sampling rate.

        Returns
        -------
        f : float
            DAQ sampling rate in Hz.
        """
        f = self.tl['Timeline'][()]['hw']['daqSampleRate'][()]
        return f

    def get_daq_data(self):
        """
        Get DAQ data with time vector and signal names.

        Returns
        -------
        data : SimpleNamespace
            Object with attributes:
            - t : time vector
            - keys : list of signal names
            - sig : dict of signals indexed by name
            - _sig_raw : raw DAQ data matrix
        """
        data = SimpleNamespace()
        data._sig_raw = self.tl['Timeline'][()]['rawDAQData'][()]

        _f = self.get_daq_samplerate()
        _t_inter = 1/_f
        _t_num = data._sig_raw.shape[0]
        data.t = np.arange(0, _t_num*_t_inter, _t_inter)

        names = self.get_hw_input_param('name')
        data.keys = list(names)

        data.sig = {}
        for ind, name in enumerate(data.keys):
            data.sig[name] = data._sig_raw[:, ind]

        return data


class _StimParser_Old(object):
    def __init__(self, beh):
        """
        Takes a BehData object as input and compiles a list
        of all delivered stimuli, stored as .id (identifier in
        the original ExpDef and Block file), .size (in uL)
        and .prob (delivery probability).
        """

        self._all_stimtypes = beh._data.get_event_var(
            'stimulusTypeValues')

        self._all_stimprobs = beh._data.get_event_var(
            'rewardProbabilityValues')

        self._all_stimsizes = beh._data.get_event_var(
            'rewardMagnitudeValues')

        self.id, self._st_inds = np.unique(self._all_stimtypes, return_index=True)
        self.prob = np.zeros_like(self.id, dtype=float)
        self.size = np.zeros_like(self.id, dtype=float)

        for ind, stimtype in enumerate(self.id):
            # find the first instance of this stimtype
            _first_ind = self._st_inds[ind]
            _stimprob = self._all_stimprobs[_first_ind]
            _stimsize = self._all_stimsizes[_first_ind]

            # save the probability and reward size for this
            self.prob[ind] = _stimprob
            self.size[ind] = _stimsize

        return

    def print_summary(self):
        """
        Print a summary of all stimulus presentations.

        Returns
        -------
        None
            Prints stimulus information to console.
        """
        print('printing a list of stims...')
        for ind, stimtype in enumerate(self.id):
            print(f'id:{stimtype} | size:{self.size[ind]} ',
                  f'| prob:{self.prob[ind]}')

        return


class StimParser(object):
    def __init__(self, beh, parse_by='stimulusOrientation',
                 load_rew_mag_prob=True):
        """
        Parses trial-types from a Block file.

        Parameters
        ----------
        beh : Behavior class instance
            Behavior data object containing Block file data.
        parse_by : str or None, optional
            Method for parsing trial types. Options:
            - 'stimulusOrientation': Parse by orientation of visual grating
            - 'stimulusTypeValues': Parse by stimulus type (grating, square, etc.)
            - 'stimulusType': Works to parse by auditory stimuli types
            - Any other string: Generic parameter name from beh._data.get_param_var()
            - None: Combine all trials into one type, averaging fields
            Default is 'stimulusOrientation'.

        Notes
        -----
        When parse_by is a generic string, it calls beh._data.get_param_var(parse_by)
        and stores the result as self._all_parsed_param.

        When parse_by is None, all trials are combined into a single condition
        with averaged probability and size values.
        """
        self.parse_by = parse_by

        # parse parameters for all trials
        if load_rew_mag_prob is True:
            self._all_stimprobs = beh._data.get_event_var(
                'rewardProbabilityValues')
            self._all_stimsizes = beh._data.get_event_var(
                'rewardMagnitudeValues')

        if parse_by == 'stimulusOrientation':
            # parse orientation
            _stim_orien_raw = beh._data.get_param_var(
                'stimulusOrientation')
            self._all_stimoris = np.array(
                [_stim_orien_raw[i][0] for i in range(len(_stim_orien_raw))])

            # fill in stats for each trial type
            self.ori, _st_inds = np.unique(self._all_stimoris,
                                           return_index=True)
            self.parsed_param = self.ori
            self._all_parsed_param = self._all_stimoris

            self.prob = np.zeros_like(self.ori, dtype=float)
            self.size = np.zeros_like(self.ori, dtype=float)

            for ind, ori in enumerate(self.ori):
                # find the first instance of this stimtype
                _first_ind = _st_inds[ind]
                _stimprob = self._all_stimprobs[_first_ind]
                _stimsize = self._all_stimsizes[_first_ind]

                # save the probability and reward size for this
                self.prob[ind] = _stimprob
                self.size[ind] = _stimsize

        elif parse_by == 'stimulusTypeValues':
            # parse orientation
            self._all_stimtypes = beh._data.get_event_var(
                'stimulusTypeValues')

            # fill in stats for each trial type
            self.stimtype, _st_inds = np.unique(self._all_stimtypes,
                                                return_index=True)
            self.parsed_param = self.stimtype
            self._all_parsed_param = self._all_stimtypes
            self.prob = np.zeros_like(self.stimtype, dtype=float)
            self.size = np.zeros_like(self.stimtype, dtype=float)

            for ind, stimtype in enumerate(self.stimtype):
                # find the first instance of this stimtype
                _first_ind = _st_inds[ind]
                _stimprob = self._all_stimprobs[_first_ind]
                _stimsize = self._all_stimsizes[_first_ind]

                # save the probability and reward size for this
                self.prob[ind] = _stimprob
                self.size[ind] = _stimsize

        elif type(parse_by) == str:
            # Generic string parameter - get from param_var
            _param_raw = beh._data.get_param_var(parse_by)

            # Handle different formats (single values vs arrays)
            if len(_param_raw) > 0 and hasattr(_param_raw[0], '__len__'):
                # Array format (like stimulusOrientation)
                self._all_parsed_param = np.array(
                    [_param_raw[i][0] for i in range(len(_param_raw))])
            else:
                # Single value format
                self._all_parsed_param = np.array(_param_raw)

            self.parsed_param, _st_inds = np.unique(self._all_parsed_param,
                                                    return_index=True)

            self.prob = np.zeros_like(self.parsed_param, dtype=float)
            self.size = np.zeros_like(self.parsed_param, dtype=float)

            for ind, param_val in enumerate(self.parsed_param):
                # find the first instance of this parameter value
                _first_ind = _st_inds[ind]
                _stimprob = self._all_stimprobs[_first_ind]
                _stimsize = self._all_stimsizes[_first_ind]

                # save the probability and reward size for this
                self.prob[ind] = _stimprob
                self.size[ind] = _stimsize

            # Store ori and stimtype for compatibility
            self.ori = self.parsed_param
            self.stimtype = self.parsed_param

        elif parse_by is None:
            # Combine all trials into one type, averaging fields
            self.parsed_param = np.array([0])  # Single condition ID
            n_trials = len(self._all_stimprobs)
            self._all_parsed_param = np.zeros(n_trials)  # All trials same type

            # Average probability and size across all trials
            self.prob = np.array([np.mean(self._all_stimprobs)])
            self.size = np.array([np.mean(self._all_stimsizes)])

            # Set ori to average for compatibility
            self.ori = np.array([np.mean(self._all_stimoris)])
            self.stimtype = np.array([0])

        return

    def print_summary(self):
        """
        Print a summary of all stimulus presentations.

        Returns
        -------
        None
            Prints stimulus information to console.
        """
        print('printing a list of stims...')
        for ind, ori in enumerate(self.ori):
            print(f'orientation:{ori} | size:{self.size[ind]} ',
                  f'| prob:{self.prob[ind]}')

        return


class BehDataSimpleLoad(object):
    def __init__(self, folder, parse_stims=True,
                 parse_by='stimulusOrientation'):
        self.folder = folder
        print('loading behavior...')

        # search for appropriate block.mat file
        files = os.listdir(folder)
        for f in files:
            if f.startswith('20') and f.endswith('Block.mat'):
                self.fname_blk = f
            elif f.startswith('20') and f.endswith('Timeline.mat'):
                self.fname_tl = f

        print('\tloading Block and Timeline files')
        # load block data with BlockParser
        self._data = BlockParser(self.folder+'/'+self.fname_blk)
        n_trials = self._data.get_event_var('endTrialTimes').shape[0]

        # load timeline with TimelineParser
        self._timeline = TimelineParser(self.folder+'/'+self.fname_tl)
        self._daq_data = self._timeline.get_daq_data()

        # load stim-on
        # -----------
        print('\tparsing stim and choice properties...')
        self.stim = SimpleNamespace()

        self.stim.t_start = self._data.get_event_var('stimulusOnTimes')[
            0:n_trials]

        # # parse stims if available
        # # ------------
        if parse_stims is True:
            self._stimparser = StimParser(self, parse_by=parse_by)

        return

    def get_rews(self, t_latency=0.1):
        """
        Parse reward information into trial format, by comparing  actual
        delivered reward times or volumes (from Timeline) with expected reward
        times (from Block), and returning only the actual reward times that
        have an associated expected reward time.

        (This is necessary because rewards can be manually delivered, and these
        will show up in Timeline and must be filtered out; also, reward
        delivery can be probabilistic.)

        Parameters
        -------------
        t_latency : float
            Max latency between actual and expected reward (in s)

        Outputs
        -----------
        rew : SimpleNamespace
            rew.t : Delivered reward times, with len(rew.t) = n_trials
            rew.val :  Reward values, with len(rew.val) = n_trials
        """
        n_trials = self._data.get_event_var('endTrialTimes').shape[0]

        inds_rew_delivered = find_event_onsets(
            self._daq_data.sig['reward_echo'], thresh=3)

        t_rew_delivered = self._daq_data.t[inds_rew_delivered]
        t_rew_expected = self._data.get_event_var(
            'feedbackTimes')[0:n_trials]
        rew_val = self._data.get_output_var(
            'rewardValues')[0:n_trials]

        rew = SimpleNamespace()
        rew.t = np.full_like(t_rew_expected, fill_value=None)
        rew.val = np.full_like(t_rew_expected, fill_value=None)

        for ind_t_rew_exp, t_rew_exp in enumerate(t_rew_expected):
            _ind = np.where(np.logical_and(
                t_rew_delivered > t_rew_exp,
                t_rew_delivered < t_rew_exp + t_latency))[0]

            if _ind.shape[0] > 0:
                rew.t[ind_t_rew_exp] = t_rew_delivered[_ind[0]]
                rew.val[ind_t_rew_exp] = rew_val[_ind[0]]

        return rew

    def get_events(self, event_name, t_latency=0.4, thresh=0.1):
        """
        Parse event time information into trial format, by comparing  actual
        event times (from Timeline) with expected event
        times (from Block), and returning only the actual event times that
        have an associated expected event time.

        Parameters
        -----------------
        event_name : str
            Name of the event. Can be 'go_cue' or 'vis_stim_on'

        Outputs
        -----------
            t_event : Event times, with len(t_event) = n_trials
        """
        n_trials = self._data.get_event_var('endTrialTimes').shape[0]

        if event_name == 'go_cue':
            event_name_tl = 'sound_echo'
            event_name_blk = 'interactiveOnTimes'
        elif event_name == 'vis_stim_on':
            event_name_tl = 'photodiode'
            event_name_blk = 'stimulusOnTimes'

        inds_event_delivered = find_event_onsets(
            self._daq_data.sig[event_name_tl], thresh=thresh)
        t_event_delivered = self._daq_data.t[inds_event_delivered]
        t_event_expected = self._data.get_event_var(
            event_name_blk)[0:n_trials]

        t_event = np.full_like(t_event_expected, fill_value=None)

        for ind_t_event_exp, t_event_exp in enumerate(t_event_expected):
            _ind = np.where(np.logical_and(
                t_event_delivered > t_event_exp,
                t_event_delivered < t_event_exp + t_latency))[0]

            if _ind.shape[0] > 0:
                t_event[ind_t_event_exp] = t_event_delivered[_ind[0]]

        return t_event

    def add_metrics(self, kern_perf_fast=5, kern_perf_slow=21,
                    kern_rew_rate=5, kern_t_react=5, sort_by_ttype=True):
        print('\tadding behavior metrics...')
        self.metrics = SimpleNamespace()
        self.metrics._params = SimpleNamespace()
        self.metrics._params.sort_by_ttype = sort_by_ttype
        self.metrics._params.kern_perf_fast = kern_perf_fast
        self.metrics._params.kern_perf_slow = kern_perf_slow
        self.metrics._params.kern_rew_rate = kern_rew_rate
        self.metrics._params.kern_t_react = kern_t_react

        self.metrics.t_trial = self.choice.t

        # behavioral metrics (perf)
        # -----------
        print('\t\tperformance metrics')
        if sort_by_ttype is False:
            self.metrics.perf = SimpleNamespace()
            self.metrics.perf.fast = smooth_squarekern(
                self.choice.correct.astype(float),
                kern=kern_perf_fast)
            self.metrics.perf.slow = smooth_squarekern(
                self.choice.correct.astype(float),
                kern=kern_perf_slow)
        if sort_by_ttype is True:
            self.metrics.perf = SimpleNamespace()
            self.metrics.perf.opto = smooth_squarekern(
                self.choice.correct[self.stim.opto].astype(float),
                kern=kern_perf_fast)
            self.metrics.perf.no_opto = smooth_squarekern(
                self.choice.correct[~self.stim.opto].astype(float),
                kern=kern_perf_fast)

            self.metrics.perf.t_opto = self.metrics.t_trial[self.stim.opto]
            self.metrics.perf.t_no_opto = self.metrics.t_trial[~self.stim.opto]

        # behavioral metrics (rew)
        # -----------
        print('\t\treward metrics')
        _rew_rate = np.diff(np.array(self.rew.total_vol)) \
            / np.diff(self.choice.t)
        _rew_rate = np.append(_rew_rate[0], _rew_rate)
        _rew_rate_smoothed = smooth_squarekern(_rew_rate, kern_rew_rate)
        self.metrics.rew_rate = _rew_rate_smoothed

        # behavioral metrics (rxn)
        # -----------
        print('\t\treaction metrics')
        _rxn_time = self.choice.t - self.stim.t_start
        _rxn_time = _rxn_time[self.choice.correct == 1]
        _rxn_time_smoothed = smooth_squarekern(_rxn_time, kern_t_react)
        self.metrics.reaction_time = _rxn_time_smoothed

        _rxn_time_t_correct = self.choice.t[self.choice.correct == 1]
        self.metrics.reaction_time_t_correct = _rxn_time_t_correct

        # behavioral metrics (wheel turn max)
        # ------------
        print('\t\twheel turn metrics')
        n_trials = self.choice.t.shape[0]
        self.metrics.wheel_turn = SimpleNamespace()
        self.metrics.wheel_turn.max = np.zeros(n_trials)
        self.metrics.wheel_turn.t_latency = np.zeros(n_trials)

        for trial in range(self.choice.t.shape[0]):
            _wheel_sig = np.abs(self.wheel.sig[trial])
            self.metrics.wheel_turn.max[trial] = np.max(_wheel_sig)

        return

    def add_wheel_data(self, t_pre_stim=0.4, t_post_stim=1,
                       sampling_rate=100, t_varname='wheelTimes',
                       val_varname='wheelValues'):
        print('\tadding wheel data...')
        wheel_t = self._data.get_input_var(t_varname)
        wheel_pos = self._data.get_input_var(val_varname)
        n_trials = self.choice.t.shape[0]

        t_templ = np.arange(-1*t_pre_stim, t_post_stim,
                            1/sampling_rate)

        self.wheel = SimpleNamespace()
        self.wheel.t = np.empty(n_trials, dtype=np.ndarray)
        self.wheel.t_abs = np.empty(n_trials, dtype=np.ndarray)
        self.wheel.sig = np.empty(n_trials, dtype=np.ndarray)

        self.wheel_raw = SimpleNamespace()
        self.wheel_raw.t = np.empty(n_trials, dtype=np.ndarray)
        self.wheel_raw.t_abs = np.empty(n_trials, dtype=np.ndarray)
        self.wheel_raw.sig = np.empty(n_trials, dtype=np.ndarray)

        for trial in range(n_trials):
            _t_stim = self.stim.t_start[trial]
            _t_start = _t_stim - t_pre_stim
            _t_end = _t_stim + t_post_stim

            _ind_start = np.argmin(np.abs(wheel_t-_t_start))
            _ind_end = np.argmin(np.abs(wheel_t-_t_end))
            _bl_wheel_pos = np.mean(wheel_pos[_ind_start-10:_ind_start])

            _binned_data = bin_signal(
                wheel_pos[_ind_start:_ind_end]-_bl_wheel_pos,
                wheel_t[_ind_start:_ind_end] - _t_stim,
                t_bin_start=-1*t_pre_stim,
                t_bin_end=t_post_stim,
                sampling_rate=sampling_rate)

            self.wheel.sig[trial] = _binned_data.sig
            self.wheel.t[trial] = _binned_data.t
            self.wheel.t_abs[trial] = _binned_data.t + _t_stim

            self.wheel_raw.sig[trial] \
                = wheel_pos[_ind_start:_ind_end]-_bl_wheel_pos
            self.wheel_raw.t[trial] \
                = wheel_t[_ind_start:_ind_end] - _t_stim
            self.wheel_raw.t_abs[trial] = wheel_t[_ind_start:_ind_end]

        return

    def get_subset_trials(self, period='prestim', t_period=None,
                          trial_type_opto=True, correct=True,
                          filter_high_perf=None,
                          wheel_turn_thresh=None,
                          wheel_turn_thresh_dir='neg'):
        """
        Parameters
        -------------
        period : str
            Can either be 'prestim' (pre-vis. stim.), 'poststim'
            (between stim and rew) or 'postchoice'
        t_period : float
            Only for 'prestim' or 'postchoice'. Dictates the total
            time period before stim or after reward to consider,
            in seconds.
        trial_type_opto : None or bool
            Can either be None, True or False
        correct : None or bool
            Whether to filter by correct response trials (True),
            incorrect response trials (False),
            or no filter (None)
        """

        # first, impose trial filters
        # --------------
        tr = SimpleNamespace()

        tr_filt = np.ones_like(self.choice.t).astype(bool)

        if trial_type_opto is not None:
            if trial_type_opto is True:
                tr_filt = np.logical_and(
                    tr_filt, self.stim.opto)
            elif trial_type_opto is False:
                tr_filt = np.logical_and(
                    tr_filt, ~self.stim.opto)

        if correct is not None:
            if correct is True:
                tr_filt = np.logical_and(
                    tr_filt, self.choice.correct)
            elif correct is False:
                tr_filt = np.logical_and(
                    tr_filt, ~self.choice.correct)

        if filter_high_perf is not None:
            self.add_metrics(sort_by_ttype=False)
            if filter_high_perf is True:
                tr_filt = np.logical_and(
                    tr_filt, self.metrics.perf.fast > 0.5)
            elif filter_high_perf is False:
                tr_filt = np.logical_and(
                    tr_filt, self.metrics.perf.fast < 0.5)

        if wheel_turn_thresh is not None:
            self.add_metrics()
            if wheel_turn_thresh_dir == 'pos':
                tr_filt = np.logical_and(
                    tr_filt,
                    self.metrics.wheel_turn.max > wheel_turn_thresh)
            if wheel_turn_thresh_dir == 'neg':
                tr_filt = np.logical_and(
                    tr_filt,
                    self.metrics.wheel_turn.max < wheel_turn_thresh)

        tr.inds = np.where(tr_filt == True)[0]

        # second, extract times for subset of trial
        # ------------
        tr.t = SimpleNamespace()

        if period == 'prestim':
            tr.t.end = self.stim.t_start[tr.inds]
            if t_period is None:
                tr.t.start = np.append(0, tr.t.end)[0:-1]
            else:
                tr.t.start = tr.t.end - t_period

        elif period == 'poststim':
            tr.t.start = self.stim.t_start[tr.inds]
            if t_period is None:
                tr.t.end = self.choice.t[tr.inds]
            else:
                tr.t.end = tr.t.start + t_period

        elif period == 'postchoice':
            tr.t.start = self.choice.t[tr.inds]
            if t_period is None:
                _t_end = np.delete(self.stim.t_start[tr.inds], 0)
                _val_to_append = self.stim.t_start[tr.inds][-1] \
                    + np.mean(np.diff(self.stim.t_start[tr.inds]))
                _t_end = np.append(_t_end, _val_to_append)
                tr.t.end = _t_end
            else:
                tr.t.end = tr.t.start + t_period

        return tr


class BehData_ReportOpto(object):
    def __init__(self, folder):
        self.folder = folder
        print('loading behavior...')

        # search for appropriate block.mat file
        files = os.listdir(folder)
        for f in files:
            if f.startswith('20') and f.endswith('Block.mat'):
                self.fname_blk = f
            elif f.startswith('20') and f.endswith('Timeline.mat'):
                self.fname_tl = f

        print('\tloading Block and Timeline files')
        # load block data with BlockParser
        self._data = BlockParser(self.folder+'/'+self.fname_blk)
        n_trials = self._data.get_event_var('endTrialTimes').shape[0]

        # load timeline with TimelineParser
        self._timeline = TimelineParser(self.folder+'/'+self.fname_tl)
        self._daq_data = self._timeline.get_daq_data()

        # load stim and reward properties
        # -----------
        print('\tparsing stim and choice properties...')
        self.stim = SimpleNamespace()

        self.stim.t_start = self._data.get_event_var('stimulusOnTimes')[
            0:n_trials]
        self.stim.opto = self._data.get_event_var(
            'laserTTLmodeRightValues')[0][0:n_trials].astype(bool)

        # replace stim-start with opto-on values from timeline
        _ind_opto_on = find_event_onsets_autothresh(
            self._daq_data.sig['laser_echo'])
        _t_opto_on = self._daq_data.t[_ind_opto_on]

        for ind_trial, _opto in enumerate(self.stim.opto):
            if _opto == True:
                _ind_stim_t_start_tl = np.argmin(np.abs(
                    _t_opto_on-self.stim.t_start[ind_trial]))
                _stim_t_start_tl = _t_opto_on[_ind_stim_t_start_tl]
                self.stim.t_start[ind_trial] = _stim_t_start_tl

        # reward
        self.rew = self.get_rews()  # adds .rew.t and .rew.val
        self.rew.total_vol = self._data.get_event_var('totalRewardValues')
        self.rew.total_vol = self.rew.total_vol[0:len(self.rew.val)]

        # load choice properties
        # ----------
        self.choice = SimpleNamespace()
        # if 'isCorrectValues' in self._data.list_event_vars():
        #     self.choice.correct = self._data.get_event_var(
        #         'isCorrectValues')[0:n_trials].astype(bool)
        if 'choiceValues' in self._data.list_event_vars():
            self.choice.response = (np.abs(self._data.get_event_var(
                'choiceValues')[0:n_trials]) > 0.5).astype(bool)

        # analyze correct responses by comparing trial-type with response
        self.choice.correct = np.zeros_like(self.choice.response)
        for tr in range(self.choice.correct.shape[0]):
            if self.stim.opto[tr] == True:
                self.choice.correct[tr] = self.choice.response[tr]
            if self.stim.opto[tr] == False:
                self.choice.correct[tr] = ~self.choice.response[tr]

        self.choice.t = self._data.get_event_var(
            'responseMadeTimes')[0:n_trials]

        # add wheel data
        self.add_wheel_data()

        return

    def get_rews(self, t_latency=0.1):
        """
        Parse reward information into trial format, by comparing  actual
        delivered reward times or volumes (from Timeline) with expected reward
        times (from Block), and returning only the actual reward times that
        have an associated expected reward time.

        (This is necessary because rewards can be manually delivered, and these
        will show up in Timeline and must be filtered out; also, reward
        delivery can be probabilistic.)

        Parameters
        -------------
        t_latency : float
            Max latency between actual and expected reward (in s)

        Outputs
        -----------
        rew : SimpleNamespace
            rew.t : Delivered reward times, with len(rew.t) = n_trials
            rew.val :  Reward values, with len(rew.val) = n_trials
        """
        n_trials = self._data.get_event_var('endTrialTimes').shape[0]

        inds_rew_delivered = find_event_onsets(
            self._daq_data.sig['reward_echo'], thresh=3)

        t_rew_delivered = self._daq_data.t[inds_rew_delivered]
        t_rew_expected = self._data.get_event_var(
            'feedbackTimes')[0:n_trials]
        rew_val = self._data.get_output_var(
            'rewardValues')[0:n_trials]

        rew = SimpleNamespace()
        rew.t = np.full_like(t_rew_expected, fill_value=None)
        rew.val = np.full_like(t_rew_expected, fill_value=None)

        for ind_t_rew_exp, t_rew_exp in enumerate(t_rew_expected):
            _ind = np.where(np.logical_and(
                t_rew_delivered > t_rew_exp,
                t_rew_delivered < t_rew_exp + t_latency))[0]

            if _ind.shape[0] > 0:
                rew.t[ind_t_rew_exp] = t_rew_delivered[_ind[0]]
                rew.val[ind_t_rew_exp] = rew_val[_ind[0]]

        return rew

    def get_events(self, event_name, t_latency=0.4, thresh=0.1):
        """
        Parse event time information into trial format, by comparing  actual
        event times (from Timeline) with expected event
        times (from Block), and returning only the actual event times that
        have an associated expected event time.

        Parameters
        -----------------
        event_name : str
            Name of the event. Can be 'go_cue' or 'vis_stim_on'

        Outputs
        -----------
            t_event : Event times, with len(t_event) = n_trials
        """
        n_trials = self._data.get_event_var('endTrialTimes').shape[0]

        if event_name == 'go_cue':
            event_name_tl = 'sound_echo'
            event_name_blk = 'interactiveOnTimes'
        elif event_name == 'vis_stim_on':
            event_name_tl = 'photodiode'
            event_name_blk = 'stimulusOnTimes'

        inds_event_delivered = find_event_onsets(
            self._daq_data.sig[event_name_tl], thresh=thresh)
        t_event_delivered = self._daq_data.t[inds_event_delivered]
        t_event_expected = self._data.get_event_var(
            event_name_blk)[0:n_trials]

        t_event = np.full_like(t_event_expected, fill_value=None)

        for ind_t_event_exp, t_event_exp in enumerate(t_event_expected):
            _ind = np.where(np.logical_and(
                t_event_delivered > t_event_exp,
                t_event_delivered < t_event_exp + t_latency))[0]

            if _ind.shape[0] > 0:
                t_event[ind_t_event_exp] = t_event_delivered[_ind[0]]

        return t_event

    def add_metrics(self, kern_perf_fast=5, kern_perf_slow=21,
                    kern_rew_rate=5, kern_t_react=5, sort_by_ttype=True):
        print('\tadding behavior metrics...')
        self.metrics = SimpleNamespace()
        self.metrics._params = SimpleNamespace()
        self.metrics._params.sort_by_ttype = sort_by_ttype
        self.metrics._params.kern_perf_fast = kern_perf_fast
        self.metrics._params.kern_perf_slow = kern_perf_slow
        self.metrics._params.kern_rew_rate = kern_rew_rate
        self.metrics._params.kern_t_react = kern_t_react

        self.metrics.t_trial = self.choice.t

        # behavioral metrics (perf)
        # -----------
        print('\t\tperformance metrics')
        if sort_by_ttype is False:
            self.metrics.perf = SimpleNamespace()
            self.metrics.perf.fast = smooth_squarekern(
                self.choice.correct.astype(float),
                kern=kern_perf_fast)
            self.metrics.perf.slow = smooth_squarekern(
                self.choice.correct.astype(float),
                kern=kern_perf_slow)
        if sort_by_ttype is True:
            self.metrics.perf = SimpleNamespace()
            self.metrics.perf.opto = smooth_squarekern(
                self.choice.correct[self.stim.opto].astype(float),
                kern=kern_perf_fast)
            self.metrics.perf.no_opto = smooth_squarekern(
                self.choice.correct[~self.stim.opto].astype(float),
                kern=kern_perf_fast)

            self.metrics.perf.t_opto = self.metrics.t_trial[self.stim.opto]
            self.metrics.perf.t_no_opto = self.metrics.t_trial[~self.stim.opto]

        # behavioral metrics (rew)
        # -----------
        print('\t\treward metrics')
        _rew_rate = np.diff(np.array(self.rew.total_vol)) \
            / np.diff(self.choice.t)
        _rew_rate = np.append(_rew_rate[0], _rew_rate)
        _rew_rate_smoothed = smooth_squarekern(_rew_rate, kern_rew_rate)
        self.metrics.rew_rate = _rew_rate_smoothed

        # behavioral metrics (rxn)
        # -----------
        print('\t\treaction metrics')
        _rxn_time = self.choice.t - self.stim.t_start
        self.metrics.reaction_time_all = smooth_squarekern(
            _rxn_time, kern_t_react)
        _rxn_time = _rxn_time[self.choice.correct == 1]
        _rxn_time_smoothed = smooth_squarekern(_rxn_time, kern_t_react)
        self.metrics.reaction_time = _rxn_time_smoothed

        _rxn_time_t_correct = self.choice.t[self.choice.correct == 1]
        self.metrics.reaction_time_t_correct = _rxn_time_t_correct

        # behavioral metrics (wheel turn max)
        # ------------
        print('\t\twheel turn metrics')
        n_trials = self.choice.t.shape[0]
        self.metrics.wheel_turn = SimpleNamespace()
        self.metrics.wheel_turn.max = np.zeros(n_trials)
        self.metrics.wheel_turn.t_latency = np.zeros(n_trials)

        for trial in range(self.choice.t.shape[0]):
            _wheel_sig = np.abs(self.wheel.sig[trial])
            self.metrics.wheel_turn.max[trial] = np.max(_wheel_sig)

        return

    def add_wheel_data(self, t_pre_stim=0.4, t_post_stim=1,
                       sampling_rate=100, t_varname='wheelTimes',
                       val_varname='wheelValues'):
        print('\tadding wheel data...')
        wheel_t = self._data.get_input_var(t_varname)
        wheel_pos = self._data.get_input_var(val_varname)
        n_trials = self.choice.t.shape[0]

        t_templ = np.arange(-1*t_pre_stim, t_post_stim,
                            1/sampling_rate)

        self.wheel = SimpleNamespace()
        self.wheel.t = np.empty(n_trials, dtype=np.ndarray)
        self.wheel.t_abs = np.empty(n_trials, dtype=np.ndarray)
        self.wheel.sig = np.empty(n_trials, dtype=np.ndarray)

        self.wheel_raw = SimpleNamespace()
        self.wheel_raw.t = np.empty(n_trials, dtype=np.ndarray)
        self.wheel_raw.t_abs = np.empty(n_trials, dtype=np.ndarray)
        self.wheel_raw.sig = np.empty(n_trials, dtype=np.ndarray)

        for trial in range(n_trials):
            _t_stim = self.stim.t_start[trial]
            _t_start = _t_stim - t_pre_stim
            _t_end = _t_stim + t_post_stim

            _ind_start = np.argmin(np.abs(wheel_t-_t_start))
            _ind_end = np.argmin(np.abs(wheel_t-_t_end))
            _bl_wheel_pos = np.mean(wheel_pos[_ind_start-10:_ind_start])

            _binned_data = bin_signal(
                wheel_pos[_ind_start:_ind_end]-_bl_wheel_pos,
                wheel_t[_ind_start:_ind_end] - _t_stim,
                t_bin_start=-1*t_pre_stim,
                t_bin_end=t_post_stim,
                sampling_rate=sampling_rate)

            self.wheel.sig[trial] = _binned_data.sig
            self.wheel.t[trial] = _binned_data.t
            self.wheel.t_abs[trial] = _binned_data.t + _t_stim

            self.wheel_raw.sig[trial] \
                = wheel_pos[_ind_start:_ind_end]-_bl_wheel_pos
            self.wheel_raw.t[trial] \
                = wheel_t[_ind_start:_ind_end] - _t_stim
            self.wheel_raw.t_abs[trial] = wheel_t[_ind_start:_ind_end]

        return

    def get_subset_trials(self, period='prestim', t_period=None,
                          trial_type_opto=True, correct=True,
                          filter_high_perf=None,
                          wheel_turn_thresh=None,
                          wheel_turn_thresh_dir='neg'):
        """
        Parameters
        -------------
        period : str
            Can either be 'prestim' (pre-vis. stim.), 'poststim'
            (between stim and rew) or 'postchoice'
        t_period : float
            Only for 'prestim' or 'postchoice'. Dictates the total
            time period before stim or after reward to consider,
            in seconds.
        trial_type_opto : None or bool
            Can either be None, True or False
        correct : None or bool
            Whether to filter by correct response trials (True),
            incorrect response trials (False),
            or no filter (None)
        """

        # first, impose trial filters
        # --------------
        tr = SimpleNamespace()

        tr_filt = np.ones_like(self.choice.t).astype(bool)

        if trial_type_opto is not None:
            if trial_type_opto is True:
                tr_filt = np.logical_and(
                    tr_filt, self.stim.opto)
            elif trial_type_opto is False:
                tr_filt = np.logical_and(
                    tr_filt, ~self.stim.opto)

        if correct is not None:
            if correct is True:
                tr_filt = np.logical_and(
                    tr_filt, self.choice.correct)
            elif correct is False:
                tr_filt = np.logical_and(
                    tr_filt, ~self.choice.correct)

        if filter_high_perf is not None:
            self.add_metrics(sort_by_ttype=False)
            if filter_high_perf is True:
                tr_filt = np.logical_and(
                    tr_filt, self.metrics.perf.fast > 0.5)
            elif filter_high_perf is False:
                tr_filt = np.logical_and(
                    tr_filt, self.metrics.perf.fast < 0.5)

        if wheel_turn_thresh is not None:
            self.add_metrics()
            if wheel_turn_thresh_dir == 'pos':
                tr_filt = np.logical_and(
                    tr_filt,
                    self.metrics.wheel_turn.max > wheel_turn_thresh)
            if wheel_turn_thresh_dir == 'neg':
                tr_filt = np.logical_and(
                    tr_filt,
                    self.metrics.wheel_turn.max < wheel_turn_thresh)

        tr.inds = np.where(tr_filt == True)[0]

        # second, extract times for subset of trial
        # ------------
        tr.t = SimpleNamespace()

        if period == 'prestim':
            tr.t.end = self.stim.t_start[tr.inds]
            if t_period is None:
                tr.t.start = np.append(0, tr.t.end)[0:-1]
            else:
                tr.t.start = tr.t.end - t_period

        elif period == 'poststim':
            tr.t.start = self.stim.t_start[tr.inds]
            if t_period is None:
                tr.t.end = self.choice.t[tr.inds]
            else:
                tr.t.end = tr.t.start + t_period

        elif period == 'postchoice':
            tr.t.start = self.choice.t[tr.inds]
            if t_period is None:
                _t_end = np.delete(self.stim.t_start[tr.inds], 0)
                _val_to_append = self.stim.t_start[tr.inds][-1] \
                    + np.mean(np.diff(self.stim.t_start[tr.inds]))
                _t_end = np.append(_t_end, _val_to_append)
                tr.t.end = _t_end
            else:
                tr.t.end = tr.t.start + t_period

        return tr


class BehData_ValuePFC(object):
    def __init__(self, folder):
        self.folder = folder
        print('loading behavior...')

        # search for appropriate block.mat file
        files = os.listdir(folder)
        for f in files:
            if f.startswith('20') and f.endswith('Block.mat'):
                self.fname_blk = f
            elif f.startswith('20') and f.endswith('Timeline.mat'):
                self.fname_tl = f

        print('\tloading Block and Timeline files')
        # load block data with BlockParser
        self._data = BlockParser(self.folder+'/'+self.fname_blk)
        n_trials = self._data.get_event_var('endTrialTimes').shape[0]

        # load timeline with TimelineParser
        self._timeline = TimelineParser(self.folder+'/'+self.fname_tl)
        self._daq_data = self._timeline.get_daq_data()

        # determine which type of variables stored in Block.
        # (necessary because some of early ALK recordings
        # have different vars for stim value.
        self.datakeys = SimpleNamespace()
        if 'leftStimRewardValues' in self._data.list_event_vars():
            self.rec_dtype = 'new'
            self.datakeys.l_val = 'leftStimRewardValues'
            self.datakeys.r_val = 'rightStimRewardValues'

        elif 'leftStim1ValueValues' in self._data.list_event_vars():
            self.rec_dtype = 'old'
            self.datakeys.l_val_1 = 'leftStim1ValueValues'
            self.datakeys.r_val_1 = 'rightStim1ValueValues'
            self.datakeys.l_val_2 = 'leftStim2ValueValues'
            self.datakeys.r_val_2 = 'rightStim2ValueValues'

        # load stim properties
        # -----------
        print('\tparsing stim and choice properties...')
        self.stim = SimpleNamespace()

        # visual
        self.stim.vis = SimpleNamespace()
        self.stim.vis.types = ['None', 'VertGrating',
                               'HorzGrating', 'Circle', 'Cross']

        if self.rec_dtype == 'new':
            self.stim.vis.left_type = self._data.get_event_var(
                'leftStimTypeValues')[0:n_trials]
            self.stim.vis.left_val = (self._data.get_event_var(
                self.datakeys.l_val)[0][0:n_trials]
                                      + self._data.get_event_var(
                self.datakeys.l_val)[1][0:n_trials]) / 2

            _left_gamble = np.abs(
                self._data.get_event_var(
                    self.datakeys.l_val)[0]
                - self._data.get_event_var(
                    self.datakeys.l_val)[1]) > 0
            self.stim.vis.left_gamble = _left_gamble

            self.stim.vis.right_type = self._data.get_event_var(
                'rightStimTypeValues')[0:n_trials]

            self.stim.vis.right_val = (self._data.get_event_var(
                self.datakeys.r_val)[0][0:n_trials]
                                       + self._data.get_event_var(
                self.datakeys.r_val)[1][0:n_trials]) / 2

            _right_gamble = np.abs(
                self._data.get_event_var(
                    self.datakeys.r_val)[0]
                - self._data.get_event_var(
                    self.datakeys.r_val)[1]) > 0
            self.stim.vis.right_gamble = _right_gamble

        elif self.rec_dtype == 'old':
            self.stim.vis.left_val = (self._data.get_event_var(
                self.datakeys.l_val_1)[0:n_trials]
              + self._data.get_event_var(
                  self.datakeys.l_val_2)[0:n_trials]) / 2

            _left_gamble = np.abs(
                self._data.get_event_var(
                    self.datakeys.l_val_1)
                - self._data.get_event_var(
                    self.datakeys.l_val_2)) > 0
            self.stim.vis.left_gamble = _left_gamble

            self.stim.vis.right_val = (self._data.get_event_var(
                self.datakeys.r_val_1)[0:n_trials]
              + self._data.get_event_var(
                  self.datakeys.r_val_2)[0:n_trials]) / 2

            _right_gamble = np.abs(
                self._data.get_event_var(
                    self.datakeys.r_val_1)
                - self._data.get_event_var(
                    self.datakeys.r_val_2)) > 0
            self.stim.vis.right_gamble = _right_gamble

        self.stim.vis.t_start = self.get_events('vis_stim_on')

        # auditory go
        self.stim.t_aud_go = self.get_events('go_cue', t_latency=0.5)

        # reward
        self.rew = self.get_rews() # adds .rew.t and .rew.val
        self.rew.total_vol = self._data.get_event_var('totalRewardValues')
        self.rew.total_vol = self.rew.total_vol[0:len(self.rew.val)]

        # load choice properties
        # ----------
        self.choice = SimpleNamespace()
        self.choice.abs_val = self._data.get_event_var(
            'choiceValues')[0:n_trials]
        self.choice.correct = self._data.get_event_var(
            'madeBestChoiceValues')
        self.choice.t = self._data.get_event_var(
            'responseMadeTimes')[0:n_trials]
        # implement t_start estimation
        # self.choice.t_start = estimate_start_time(self.choice.t_end)

        # store general trialtype information
        # ----------
        self.tr_type = SimpleNamespace()
        self.tr_type.onestim = np.logical_or(self.stim.vis.right_val == 0,
                                             self.stim.vis.left_val == 0)
        self.tr_type.twostim = ~self.tr_type.onestim
        self.tr_type.gamble = np.logical_or(self.stim.vis.right_gamble == 1,
                                            self.stim.vis.left_gamble == 1)

        return

    def get_rews(self, t_latency=0.1):
        """
        Parse reward information into trial format, by comparing  actual
        delivered reward times or volumes (from Timeline) with expected reward
        times (from Block), and returning only the actual reward times that
        have an associated expected reward time.

        (This is necessary because rewards can be manually delivered, and these
        will show up in Timeline and must be filtered out; also, reward
        delivery can be probabilistic.)

        Parameters
        -------------
        t_latency : float
            Max latency between actual and expected reward (in s)

        Outputs
        -----------
        rew : SimpleNamespace
            rew.t : Delivered reward times, with len(rew.t) = n_trials
            rew.val :  Reward values, with len(rew.val) = n_trials
        """
        n_trials = self._data.get_event_var('endTrialTimes').shape[0]

        inds_rew_delivered = find_event_onsets(
            self._daq_data.sig['reward_echo'], thresh=3)

        if self.rec_dtype == 'new':
            t_rew_delivered = self._daq_data.t[inds_rew_delivered]
            t_rew_expected = self._data.get_event_var(
                'feedbackTimes')[0:n_trials]
            rew_val = self._data.get_output_var(
                'rewardValues')[0:n_trials]
        if self.rec_dtype == 'old':
            t_rew_delivered = self._daq_data.t[inds_rew_delivered]
            t_rew_expected = self._data.get_event_var(
                'rewardBeepAmplitudeTimes')[0:n_trials]
            rew_val = self._data.get_output_var(
                'rewardValues')[0:n_trials]

        rew = SimpleNamespace()
        rew.t = np.full_like(t_rew_expected, fill_value=None)
        rew.val = np.full_like(t_rew_expected, fill_value=None)

        for ind_t_rew_exp, t_rew_exp in enumerate(t_rew_expected):
            _ind = np.where(np.logical_and(
                t_rew_delivered > t_rew_exp,
                t_rew_delivered < t_rew_exp + t_latency))[0]

            if _ind.shape[0] > 0:
                rew.t[ind_t_rew_exp] = t_rew_delivered[_ind[0]]
                rew.val[ind_t_rew_exp] = rew_val[_ind[0]]

        return rew

    def get_events(self, event_name, t_latency=0.4, thresh=0.1):
        """
        Parse event time information into trial format, by comparing  actual
        event times (from Timeline) with expected event
        times (from Block), and returning only the actual event times that
        have an associated expected event time.

        Parameters
        -----------------
        event_name : str
            Name of the event. Can be 'go_cue' or 'vis_stim_on'

        Outputs
        -----------
            t_event : Event times, with len(t_event) = n_trials
        """
        n_trials = self._data.get_event_var('endTrialTimes').shape[0]

        if event_name == 'go_cue':
            event_name_tl = 'sound_echo'
            event_name_blk = 'interactiveOnTimes'
        elif event_name == 'vis_stim_on':
            event_name_tl = 'photodiode'
            event_name_blk = 'stimulusOnTimes'

        inds_event_delivered = find_event_onsets(
            self._daq_data.sig[event_name_tl], thresh=thresh)
        t_event_delivered = self._daq_data.t[inds_event_delivered]
        t_event_expected = self._data.get_event_var(
            event_name_blk)[0:n_trials]

        t_event = np.full_like(t_event_expected, fill_value=None)

        for ind_t_event_exp, t_event_exp in enumerate(t_event_expected):
            _ind = np.where(np.logical_and(
                t_event_delivered > t_event_exp,
                t_event_delivered < t_event_exp + t_latency))[0]

            if _ind.shape[0] > 0:
                t_event[ind_t_event_exp] = t_event_delivered[_ind[0]]

        return t_event

    def add_metrics(self, kern_perf_fast=5, kern_perf_slow=51,
                    kern_rew_rate=5, kern_t_react=5):
        print('\tadding behavior metrics...')
        self.metrics = SimpleNamespace()
        self.metrics.t_trial = self.choice.t
        # behavioral metrics (perf)
        # -----------
        print('\t\tperformance metrics')
        self.metrics.perf = SimpleNamespace()
        self.metrics.perf.fast = smooth_squarekern(
            self.choice.correct.astype(float),
            kern=kern_perf_fast)
        self.metrics.perf.slow = smooth_squarekern(
            self.choice.correct.astype(float),
            kern=kern_perf_slow)

        # behavioral metrics (rew)
        # -----------
        print('\t\treward metrics')
        _rew_rate = np.diff(np.array(self.rew.total_vol)) \
            / np.diff(self.choice.t)
        _rew_rate = np.append(_rew_rate[0], _rew_rate)
        _rew_rate_smoothed = smooth_squarekern(_rew_rate, kern_rew_rate)
        self.metrics.rew_rate = _rew_rate_smoothed

        # behavioral metrics (rxn)
        # -----------
        print('\t\treaction metrics')
        _rxn_time = self.choice.t - self.stim.vis.t_start
        _rxn_time_smoothed = smooth_squarekern(_rxn_time, kern_t_react)
        self.metrics.reaction_time = _rxn_time_smoothed

        return

    def get_subset_trials(self, period='prestim', t_period=None,
                          risk=None, onestim=None, twostim=None,
                          valdiff=None, rew_delivered=None):
        """
        Parameters
        -------------
        period : str
            Can either be 'prestim' (pre-vis. stim.), 'poststim'
            (between stim and rew) or 'postchoice'
        t_period : float
            Only for 'prestim' or 'postchoice'. Dictates the total
            time period before stim or after reward to consider,
            in seconds.
        ~~~ The following relate to trial-type filtering ~~~
        risk : None or bool
            Can either be None (no filter), True or False (filter
            by risky trials or non-risky trials)
        onestim : None or True
            Whether to filter by onestim trials (True) or not (None)
        twostim : None or True
            Whether to filter by twostim trials (True) or not (None)
        valdiff : None or bool
            Whether to filter by a certain value difference between
            the two stims (eg 1.5uL) nor not
        rew_delivered : None or bool
            Whether to filter by reward delivered trials (True),
            non-reward delivered trials (False),
            or no filter (None)
        """

        # first, impose trial filters
        # --------------
        tr = SimpleNamespace()

        tr_filt = np.ones_like(self.choice.t).astype(bool)
        if risk is not None:
            if risk is True:
                tr_filt = np.logical_and(
                    tr_filt, self.tr_type.gamble)
            if risk is False:
                tr_filt = np.logical_and(
                    tr_filt, ~self.tr_type.gamble)

        if onestim is not None:
            if onestim is True:
                tr_filt = np.logical_and(
                    tr_filt, self.tr_type.onestim)

        if twostim is not None:
            if twostim is True:
                tr_filt = np.logical_and(
                    tr_filt, self.tr_type.twostim)

        if valdiff is not None:
            bool_valdiff = np.abs(
                self.stim.vis.left_val - self.stim.vis.right_val) == valdiff
            tr_filt = np.logical_and(
                tr_filt, bool_valdiff)

        if rew_delivered is not None:
            if rew_delivered is True:
                tr_filt = np.logical_and(
                    tr_filt, ~np.isnan(self.rew.t))
            if rew_delivered is False:
                tr_filt = np.logical_and(
                    tr_filt, np.isnan(self.rew.t))

        tr.inds = np.where(tr_filt == True)[0]

        # second, extract times for subset of trial
        # ------------
        tr.t = SimpleNamespace()

        if period == 'prestim':
            tr.t.end = self.stim.vis.t_start[tr.inds]
            if t_period is None:
                tr.t.start = np.append(0, tr.t.end)[0:-1]
            else:
                tr.t.start = tr.t.end - t_period

        elif period == 'poststim':
            tr.t.start = self.stim.vis.t_start[tr.inds]
            if t_period is None:
                tr.t.end = self.choice.t[tr.inds]
            else:
                tr.t.end = tr.t.start + t_period

        elif period == 'postchoice':
            tr.t.start = self.choice.t[tr.inds]
            if t_period is None:
                _t_end = np.delete(self.stim.vis.t_start[tr.inds], 0)
                _val_to_append = self.stim.vis.t_start[tr.inds][-1] \
                    + np.mean(np.diff(self.stim.vis.t_start[tr.inds]))
                _t_end = np.append(_t_end, _val_to_append)
                tr.t.end = _t_end
            else:
                tr.t.end = tr.t.start + t_period

        return tr


# class TrTypeList(object):
#     """
#     Defines a set of trialtypes in a flexible manner, to be used during
#     loading of behaviors in TwoPRec
#     """

#     def __init__(self, beh):
#         self.beh = beh
#         self.trtypes = {}
#         return

#     def add_trtype(self, trtype_name, **kwargs):
#         self.trtypes[trtype_name] = {}

#         # store all kwargs
#         for key, val in kwargs.iteritems():
#             self.trtypes[trtype_name][key] = val


# class TrType(object):
#     def __init__(self, name, rew_prob=None, rew_size=None,
#                  rew_delivered=None,
#                  **kwargs):
#         self.name = name

#         self.tr_def = {}
#         for in 
#         self.tr

#         self.params = {}
#         for key, val in kwargs.iteritems():
#             self.params[key] = val
