import numpy as np
import scipy as sp
import os
import scipy.signal as sp_signal
import matplotlib.pyplot as plt
from types import SimpleNamespace

try:
    plt.style.use('publication_ml')
except:
    pass


# Tools to load and preprocess dataset
# -------------

def find_event_onsets(events, thresh=5):
    """
    Find event onset indices by detecting large positive changes in signal.

    Parameters
    ----------
    events : np.ndarray
        Event signal array (e.g., voltage trace).
    thresh : float, optional
        Threshold for detecting onset (difference must exceed this), by default 5.

    Returns
    -------
    event_inds : np.ndarray
        Indices where event onsets occur.
    """
    event_diff = np.diff(events)
    event_inds = np.where(event_diff > thresh)[0]
    return event_inds


def find_event_onsets_autothresh(events, n_stdevs=2):
    """
    Find event onsets using automatic thresholding based on signal statistics.

    Detects onsets by finding differences that exceed mean + n_stdevs*std.

    Parameters
    ----------
    events : np.ndarray
        Event signal array (e.g., voltage trace).
    n_stdevs : float, optional
        Number of standard deviations above mean for threshold, by default 2.

    Returns
    -------
    event_inds : np.ndarray
        Indices where event onsets occur.
    """
    event_diff = np.diff(events)
    event_diff_mean = np.mean(event_diff)
    event_diff_std = np.std(event_diff)

    event_inds = np.where(
        event_diff > (event_diff_mean + event_diff_std*n_stdevs))[0]

    return event_inds


def find_event_onsets_plateau(lick_sig, t,
                              v=5,
                              tol=0.02,
                              t_thresh=0.05):
    """
    Detect event onsets in the case where lick signal is recorded
    with lots of periodic noise (eg Dual2P).

    Works by finding periods where the signal plateaus at or close to
    5+-0.02 volts for greater than 50ms (defaults) and storing these.

    Parameters
    ----------
    lick_sig : np.ndarray
        Lick signal voltage trace.
    t : np.ndarray
        Time vector corresponding to lick_sig.
    v : float, optional
        Target plateau voltage value, by default 5.
    tol : float, optional
        Tolerance around target voltage for plateau detection, by default 0.02.
    t_thresh : float, optional
        Minimum duration (seconds) for a valid plateau, by default 0.05.

    Returns
    -------
    event_inds : np.ndarray
        Indices where plateau onsets begin.
    """
    dt = t[1] - t[0]

    stable = np.abs(lick_sig - v) <= tol

    # Find transitions
    diff = np.diff(stable.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    # Handle edge cases (signal starts or ends in plateau)
    if stable[0]:
        starts = np.r_[0, starts]
    if stable[-1]:
        ends = np.r_[ends, len(lick_sig)]

    # Compute durations and keep only long enough plateaus
    durations = (ends - starts) * dt
    event_inds = [start for start, end, dur in zip(starts, ends, durations)
                  if dur >= t_thresh]
    event_inds = np.array(event_inds)

    return np.array(event_inds)


def remove_event_duplicates(events, abs_refractory_period=0.1):
    """
    Remove duplicate events that occur within an absolute refractory period.

    Parameters
    ----------
    events : np.ndarray
        Array of event times.
    abs_refractory_period : float, optional
        Minimum time interval between valid events (seconds), by default 0.1.

    Returns
    -------
    events_fixed : np.ndarray
        Event times with duplicates removed.
    """
    events_diff = np.diff(events)
    inds_to_cut = []
    for ind, event_diff in enumerate(events_diff):
        if event_diff < abs_refractory_period:
            inds_to_cut.append(ind)
    events_fixed = np.delete(events, inds_to_cut)
    return events_fixed


def remove_lick_artefact_after_rew(t_licks,
                                   t_interval_after_rew=[0.03, 0.06]):
    """
    Remove lick artifact events that occur shortly after reward delivery.

    Takes a vector of lick times aligned to reward onset (t=0),
    and removes all lick times within a certain time-window after
    reward, in seconds (these are typically artefacts).

    Parameters
    ----------
    t_licks : np.ndarray
        Lick times aligned to reward onset (t=0).
    t_interval_after_rew : list of float, optional
        Time interval [start, end] (seconds) after reward to remove licks,
        by default [0.03, 0.06].

    Returns
    -------
    t_licks_corrected : np.ndarray
        Lick times with artifacts removed.
    """
    inds_to_delete = np.argwhere(np.logical_and(
        t_licks > t_interval_after_rew[0],
        t_licks < t_interval_after_rew[1]))
    t_licks_corrected = np.delete(t_licks, inds_to_delete)
    return t_licks_corrected


def calc_alpha(val, val_max, alpha_min=0.2):
    """
    Calculate a normalized transparency value for plotting.

    Calculates a normalized transparency value for a datapoint,
    given the max value observed and a target minimum alpha
    (to prevent alpha from going to 0 and becoming invisible).

    Parameters
    ----------
    val : float
        Current value to calculate alpha for.
    val_max : float
        Maximum value in the dataset.
    alpha_min : float, optional
        Minimum alpha value to prevent invisibility, by default 0.2.

    Returns
    -------
    alpha : float
        Normalized alpha value between alpha_min and 1.0.
    """
    offset = (alpha_min*val_max) / (1-alpha_min)
    alpha = (val+offset) / (val_max+offset)

    return alpha


def convert_spktimes_to_spktrain(spktimes, t,
                                 method='calc'):
    """
    Convert spike times to a binary spike train array.

    Parameters
    ----------
    spktimes : np.ndarray
        Array of spike times (seconds).
    t : np.ndarray
        Time vector for the spike train.
    method : str, optional
        Method to find spike indices: 'calc' (calculate from sampling rate)
        or 'argmin' (find closest time point), by default 'calc'.

    Returns
    -------
    spktrain : np.ndarray
        Binary spike train (1 at spike times, 0 elsewhere).
    """
    spktrain = np.zeros_like(t, dtype=np.int8)
    _samp_rate_hz = 1/(t[1]-t[0])

    for _t_spk in spktimes:
        if method == 'calc':
            _ind_spk = int(_t_spk * _samp_rate_hz)
        elif method == 'argmin':
            _ind_spk = np.argmin(np.abs(t-_t_spk))
        spktrain[_ind_spk] = 1

    return spktrain


def bin_signal(sig, t,
               t_bin_start=-0.2, t_bin_end=1, sampling_rate=100):
    """
    Bin a signal into uniform time bins with specified sampling rate.

    Parameters
    ----------
    sig : np.ndarray
        Signal to be binned.
    t : np.ndarray
        Time vector corresponding to sig.
    t_bin_start : float, optional
        Start time for binning (seconds), by default -0.2.
    t_bin_end : float, optional
        End time for binning (seconds), by default 1.
    sampling_rate : float, optional
        Target sampling rate (Hz) for binned signal, by default 100.

    Returns
    -------
    binned : SimpleNamespace
        Object with attributes:
        - t : time vector for binned signal
        - sig : binned signal values (averaged within each bin)
    """
    binned = SimpleNamespace()
    binned.t = np.arange(t_bin_start,
                         t_bin_end+1/1000,
                         1/sampling_rate)
    binned.sig = np.zeros_like(binned.t, dtype=float)

    for ind_bin_t, val in enumerate(binned.t[1:-1]):
        _inds_t = np.where(np.logical_and(
            t > binned.t[ind_bin_t],
            t < binned.t[ind_bin_t+1]))[0]
        if len(_inds_t) == 0:
            binned.sig[ind_bin_t] = binned.sig[ind_bin_t-1]
        elif len(_inds_t) > 0:
            binned.sig[ind_bin_t] = np.mean(sig[_inds_t])

    binned.sig[-1] = binned.sig[-2]
    return binned


def parse_samp_rate_from_params(fname):
    """
    Parameters
    ---------------
    fname: str
        path of the params.py from ephys folder.

    Returns
    -------------
    samp_rate : float
        Sampling rate of the ephys data, in Hz.
    """
    f = open(fname)
    config = f.read()
    samp_rate = float(
        config.partition('sample_rate = ')[2].partition('\n')[0])

    return samp_rate


def dtype_to_list(dtype):
    """Converts a numpy recarray dtype attribute into a parseable list.
    For use when loading block.mat files using sp.io.loadmat.
    """
    dtype_list = [dtype.descr[i][0] for i in range(len(dtype.descr))]
    return dtype_list


def convert_rewvalue_into_floats(rewvalue):
    """Takes totalRewardValue variable (from block.mat), parses into
    individual trials, and converts to floats
    """
    rewvalue_split = rewvalue.split('\N{MICRO SIGN}l')
    rewvalue_floats = [float(rewvalue_split[i])
                       for i in range(len(rewvalue_split)-1)]
    return rewvalue_floats


def find_key_partialmatch(dictionary, str_query):
    """
    Find a dictionary key that partially matches a query string.

    Parameters
    ----------
    dictionary : dict
        Dictionary to search through.
    str_query : str
        String to search for within dictionary keys.

    Returns
    -------
    key : str or None
        First key containing the query string, or None if no match found.
    """
    for key in list(dictionary.keys()):
        if key.find(str_query) > -1:
            return key
    return None


def match_region_partial(region, all_regions):
    """
    Find a partial match of a brain region name in a list of all regions.

    Parameters
    ----------
    region : str
        Brain region name to search for.
    all_regions : list of str
        List of all possible brain region names.

    Returns
    -------
    matched_region : str
        First matching region name from all_regions found within the query region,
        or 'none' if no match found.
    """
    for reg_to_test in all_regions:
        if reg_to_test in region.lower():
            return reg_to_test
    return 'none'


def smooth_squarekern(vec, kern, mode='mean'):
    """
    Smooth a signal by convolving with a square kernel.

    Works by first creating a padded vector with a given mode and
    convolving in the 'valid' mode to avoid edge effects and return a
    smoothed vector of identical length to the input.

    Parameters
    ----------
    vec : np.ndarray
        Input signal vector to smooth.
    kern : int
        Kernel size (must be odd number).
    mode : str, optional
        Padding mode for np.pad (e.g., 'mean', 'edge'), by default 'mean'.

    Returns
    -------
    vec_smooth : np.ndarray
        Smoothed signal vector (same length as input).
    """

    assert np.remainder(kern, 2) == 1, 'kern must be an odd number.'

    vec_pad = np.pad(vec, int((kern-1)/2),
                     mode=mode, stat_length=int((kern-1)/2))

    vec_smooth = sp.signal.convolve(
        vec_pad, np.ones(kern)/kern,
        mode='valid')

    return vec_smooth


def rescale_to_frac(times, beh, n_bins):
    """
    Takes a set of times and associated behavioral variables, rescales the
    times to span [0, 1] (rs.t), and returns a rescaled,
    binned vector of the behavioral variable (rs.beh) over these times.

    Parameters
    ----------
    times
        Vector of timestamps for each behavioral value in beh.
    beh : np.ndarray
        Vector of behavioral values, with the same length as times
    n_bins : int
        Number of bins to tile the behavioral task with.
    """
    t_bins = np.linspace(0, 1, n_bins)

    t_as_frac = (times-times[0]) / (times[-1]-times[0])
    beh_as_frac_binned = np.zeros_like(t_bins)

    _last_ind_with_data = 0
    for ind, bin_frac in enumerate(t_bins[0:-1]):
        _inds_in_times = np.where(np.logical_and(
            t_as_frac > t_bins[ind],
            t_as_frac < t_bins[ind+1]))[0]
        if len(_inds_in_times) > 0:
            _beh = np.mean(beh[_inds_in_times])
            beh_as_frac_binned[ind] = _beh

            _last_ind_with_data = ind
        else:
            if ind != 0:
                beh_as_frac_binned[ind] = beh_as_frac_binned[
                    _last_ind_with_data]
            else:
                pass

    # set last beh bin as second last (as these are bin edges)
    beh_as_frac_binned[-1] = beh_as_frac_binned[-2]

    rs = SimpleNamespace()
    rs.t = t_bins
    rs.beh = beh_as_frac_binned

    return rs


def smooth_spktrain(spktrain, t,
                    gauss_stdev_ms):
    """
    Smooth a spike train by convolving with a Gaussian kernel.

    Parameters
    ----------
    spktrain : np.ndarray
        Point process of a single neuron's spiking outputs (binary spike train).
    t : np.ndarray
        Time vector associated with spktrain, in units of seconds.
    gauss_stdev_ms : float
        Standard deviation of the gaussian kernel used for convolution,
        in units of milliseconds.

    Returns
    -------
    spktrain_smoothed : np.ndarray
        Smoothed spike train (firing rate estimate).
    """
    # setup parameters
    sampling_rate_hz = 1/(t[1]-t[0])
    gauss_stdev_inds = (gauss_stdev_ms/1000) * sampling_rate_hz
    gauss_npoints = int(gauss_stdev_inds*8)

    # define gauss kernel and normalize auc to 1
    gauss_kern = sp_signal.windows.gaussian(
        gauss_npoints, gauss_stdev_inds)
    gauss_kern_integral = np.trapz(gauss_kern,
                                   x=t[0:gauss_kern.shape[0]])
    gauss_kern /= gauss_kern_integral

    # convolve point process with gaussian
    spktrain_smoothed = sp_signal.convolve(
        spktrain,
        gauss_kern, mode='same')

    return spktrain_smoothed


def extract_activ_order(spk_mtx, time, thresh=2):
    """
    Takes a matrix (spk_mtx (neur, t)) of spikes over time for a population
    of neurons in a single trial (aligned to stim or some other event) and a
    time vector (time), and computes the relative onset time for each neuron
    using a threshold (thresh).

    Best if spk_mtx is z-scored smoothed firing rate over time,
    and then a threshold is applied to this.

    Returns
    -------------
    order : SimpleNamespace()
    order.t_onset
        Onset time, of activity for each neuron, calculated using [time]
        (None if no activity)
    order.rank
        Ordered rank of each neuron for that trial, as an integer.
        (None if no activity)
    """

    n_neurs = spk_mtx .shape[0]

    order = SimpleNamespace()
    order.t_onset = np.zeros(n_neurs)
    order.rank = np.zeros(n_neurs)

    # Compute onset times
    for neur in range(n_neurs):
        # try to get the latency
        try:
            _ind_latency = np.where(
                (spk_mtx[neur, :] > thresh) > 0)[0]
            # only take times > 0
            _ind_latency_timethresh = _ind_latency[_ind_latency > np.argwhere(
                time > 0)[0]][0]

            _t_latency = time[_ind_latency_timethresh]
        except IndexError:
            _ind_latency = None
            _t_latency = None

        order.t_onset[neur] = _t_latency

    _rank = np.argsort(order.t_onset)

    # delete all items that have a latency of nan (no activity that trial)
    _rank = np.delete(
        _rank,
        np.arange(-1, -1*(np.sum(np.isnan(order.t_onset)))-1, -1))
    order.rank = _rank

    return order


def pad_rank_with_nans(rank):
    """
    Takes an array of arrays denoting the activation order of each neuron for
    each trial, and convert to a single large array padded with nans (in
    case not every trial contains activation by all neurons
    """
    # find max number of neurons activated in a single trial
    n_tr = rank.shape[0]
    n_neurs_max = 0
    for tr in range(n_tr):
        _size = rank[tr].shape[0]
        if _size > n_neurs_max:
            n_neurs_max = _size

    rank_padded = np.zeros((n_tr, n_neurs_max))

    for tr in range(n_tr):
        _n_to_pad = n_neurs_max - rank[tr].shape[0]
        rank_padded[tr, :] = np.pad(rank[tr].astype(float),
                                    ((0, _n_to_pad)), 'constant',
                                    constant_values=np.nan)

    return rank_padded


def check_folder_exists(folder_name):
    """
    Check if a folder exists and create it if it doesn't.

    Parameters
    ----------
    folder_name : str
        Path to the folder to check/create.

    Returns
    -------
    None
    """
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    return
