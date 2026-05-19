import numpy as np
import scipy as sp
import scipy.stats as sp_stats
from sklearn import decomposition
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from types import SimpleNamespace
import os
import json

from .utils import parse_samp_rate_from_params, find_key_partialmatch, \
    match_region_partial, convert_spktimes_to_spktrain, \
    smooth_spktrain
from .histology import LoadHist


class SpikeTrain(object):
    """
    Container for spike train data with analysis methods.

    Parameters
    ----------
    arr : np.ndarray, optional
        Spike train array (neurons x time).
    t : np.ndarray, optional
        Time vector for spike train.
    spk : object, optional
        Alternative spike object with 'train' and 't' attributes.
    info : dict or object, optional
        Additional metadata about the spike train.

    Attributes
    ----------
    arr : np.ndarray
        Spike train array.
    t : np.ndarray
        Time vector.
    samp_rate : float
        Sampling rate in Hz.
    info : object
        Metadata.
    arr_shuff : np.ndarray
        Shuffled spike trains (created by add_shuff_data).
    """
    def __init__(self, arr=None, t=None, spk=None, info=None):
        if spk is None:
            self.arr = arr
            self.t = t
        else:
            self.arr = spk.train
            self.t = spk.t

        self.samp_rate = 1/(self.t[1]-self.t[0])
        self.info = info

    def add_shuff_data(self, n=10):
        """
        Generate shuffled versions of spike train for statistical testing.

        Parameters
        ----------
        n : int, optional
            Number of shuffled spike trains to generate, by default 10.

        Returns
        -------
        None
            Creates self.arr_shuff attribute containing shuffled data.
        """
        self.arr_shuff = np.ndarray(n, dtype=np.ndarray)
        for _n in range(n):
            self.arr_shuff[_n] = np.copy(self.arr)
            np.random.shuffle(self.arr_shuff[_n])

    def plt(self, figsize=(3.43, 2), dotsize=10):
        """
        Plot spike raster for all neurons in the spike train.

        Parameters
        ----------
        figsize : tuple of float, optional
            Figure size (width, height) in inches, by default (3.43, 2).
        dotsize : float, optional
            Size of spike markers, by default 10.

        Returns
        -------
        None
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        for neur in range(self.arr.shape[0]):
            _t_spks = self.t[self.arr[neur, :].astype(bool)]
            ax.scatter(_t_spks, np.ones_like(_t_spks)*neur,
                       color='k', s=dotsize)

        ax.set_xlabel('t (sec)')
        ax.set_ylabel('neur')

        plt.show()
        fig.clear()

        return

    def calc_sync_index_dragoi(self, on_shuff=False):
        """
        Calculate synchrony index using Dragoi method (coefficient of variation).

        Computes the ratio of standard deviation to mean of population spike counts
        across time bins. Higher values indicate more synchronous firing.

        Parameters
        ----------
        on_shuff : bool, optional
            If True, compute on shuffled data; if False, compute on real data,
            by default False.

        Returns
        -------
        sync_index : float
            Synchrony index (SD/mean of spike count across time).
        """
        if on_shuff is False:
            spk_count = np.sum(self.arr, axis=0)
        elif on_shuff is True:
            spk_count = np.sum(self.arr_shuff, axis=0)

        _sd = np.std(spk_count)
        _mean = np.mean(spk_count)
        sync_index = _sd / _mean

        return sync_index

    def calc_sync_index_packer(self, on_shuff=False, shuff_ind=0):
        """
        Calculate synchrony index using Packer method (pairwise correlations).

        Computes pairwise Pearson correlations between all neuron pairs and
        averages them to quantify population synchrony.

        Parameters
        ----------
        on_shuff : bool, optional
            If True, compute on shuffled data; if False, compute on real data,
            by default False.
        shuff_ind : int, optional
            Index of shuffled dataset to use if on_shuff is True, by default 0.

        Returns
        -------
        sync_index : SimpleNamespace
            Object with attributes:
            - xcorr_arr : masked array of pairwise correlations (upper triangle)
            - mean_xcorr : mean of all pairwise correlations
        """
        sync_index = SimpleNamespace()
        sync_index.xcorr_arr = np.ma.zeros(
            (self.arr.shape[0], self.arr.shape[0]))
        for neur1 in range(self.arr.shape[0]):
            for neur2 in range(self.arr.shape[0]):
                if neur2 > neur1:
                    if on_shuff is False:
                        sync_index.xcorr_arr[neur1, neur2] = sp.stats.pearsonr(
                            self.arr[neur1, :],
                            self.arr[neur2, :]).statistic
                    elif on_shuff is True:
                        sync_index.xcorr_arr[neur1, neur2] = sp.stats.pearsonr(
                            self.arr_shuff[shuff_ind][neur1, :],
                            self.arr_shuff[shuff_ind][neur2, :]).statistic
                else:
                    sync_index.xcorr_arr[neur1, neur2] = np.ma.masked
        sync_index.mean_xcorr = np.ma.mean(sync_index.xcorr_arr)

        return sync_index

    def calc_stfr(self):
        """
        Calculate spike-triggered firing rate.

        Returns
        -------
        None
            Not yet implemented.
        """
        return


class EphysData(object):
    def __init__(self, folder, aligner_obj=False, filt_clusts=True,
                 probe_file_name='probe_file.txt', multiprobe=False,
                 no_histology=False):
        """
        Class for loading and working with neuropixel data associated with a
        behavioral dataset. (Blake's ReportOpto dataset here)

        Must be already sorted using **kilosort 4** (kilosort 2.5 will not work
        due to slight differences in how the spike and clust arrays
        are organized.

        Parameters
        ---------------
        folder : str
            Name of the ephys folder with kilosort outputs.
        filt_clusts : bool
            If True, only keeps clusters that were identified by kilosort as
            'good' (ie excludes 'noise' and 'mua'.
        samp_rate_pointproc_hz : int
            Sampling rate for the point process spiketrain, in Hz.
            While the real sampling rate for neuropixels probe is quite high
            (~30k hz), this should be much lower.

        Attributes added
        --------------
        self.spk_clusters
        self.spk_times_raw
        self.spk_info

        self.samp_rate_hz

        self.ch_raw
            self.ch_raw.map
            self.ch_raw.pos
        self.ch
            self.ch.region
            self.ch.region_id
            self.ch.ml
            self.ch.ap
            self.ch.dv
        """
        self._no_histology = no_histology

        self.folder = folder
        _start_dir = os.getcwd()

        if multiprobe is False:
            n_folders_back_herbs_dir = 1
        if multiprobe is True:
            n_folders_back_herbs_dir = 2

        if self._no_histology is False:
            self.hist = LoadHist('.', ephys_folder=folder,
                                 probe_file_name=probe_file_name,
                                 n_folders_back_herbs_dir=n_folders_back_herbs_dir)
        os.chdir(_start_dir)

        print('loading ephys...')
        print('\tloading spike data and channel position data...')

        os.chdir(self.folder)
        # load spk data
        self.spk = SimpleNamespace()
        self.spk_raw = SimpleNamespace()
        self.spk_raw.clusters = np.load('spike_clusters.npy')
        self.spk_raw.clusters = self.spk_raw.clusters

        self.spk_raw.times = np.load('spike_times.npy')
        self.spk_raw.clust_kslabel = pd.read_csv('cluster_KSLabel.tsv', sep='\t')
        self.spk_raw.samp_rate_hz = parse_samp_rate_from_params('params.py')

        # load channel position data
        self.ch_raw = SimpleNamespace()
        self.ch_raw.map = np.load('channel_map.npy')
        self.ch_raw.pos = np.load('channel_positions.npy')

        n_ch = len(self.ch_raw.pos)

        self.ch = SimpleNamespace()
        self.ch.region = np.empty(n_ch, dtype=object)
        self.ch.region_id = np.zeros(n_ch)
        self.ch.ml = np.zeros(n_ch)
        self.ch.ap = np.zeros_like(self.ch.ml)
        self.ch.dv = np.zeros_like(self.ch.ml)
        self.ch.dv_plt = np.zeros_like(self.ch.ml)  # offset

        # setup colors for different brain regions
        self.plt = SimpleNamespace()
        self.plt.colors = {}
        self.plt.colors['primary somatosensory'] = sns.xkcd_rgb['seafoam green']
        self.plt.colors['supplemental somatosensory'] = sns.xkcd_rgb['blue green']
        self.plt.colors['field CA'] = sns.xkcd_rgb['apple green']
        self.plt.colors['caudoputamen'] = sns.xkcd_rgb['powder blue']
        self.plt.colors['striatum'] = sns.xkcd_rgb['powder blue']
        self.plt.colors['pallidum'] = sns.xkcd_rgb['faded blue']
        self.plt.colors['lateral geniculate'] = sns.xkcd_rgb['salmon']
        self.plt.colors['thalamus'] = sns.xkcd_rgb['pinkish']
        self.plt.colors['ventral posteromedial nucleus of the thalamus'] \
            = sns.xkcd_rgb['pinkish']
        self.plt.colors['posterior complex of the thalamus'] \
            = sns.xkcd_rgb['pinkish']
        self.plt.colors['midbrain'] = sns.xkcd_rgb['fuchsia']
        self.plt.colors['hypothalamus'] = sns.xkcd_rgb['scarlet']
        self.plt.colors['none'] = sns.xkcd_rgb['grey']

        # try to load information about histology-aligned ch locations
        print('\thistology alignment data...')

        if self._no_histology is False:
            for ind_ch in range(n_ch):
                self.ch.region[ind_ch] = self.hist.map_ch_region.ch_to_reg(ind_ch)

        # setup a dv_plt attribute that has dvs that are staggered
        self.ch.dv_plt = np.copy(self.ch.dv)
        self.ch.dv_plt[0:-1] += np.diff(
            self.ch.dv_plt)/2

        # filter data based on phy label of 'good'
        if filt_clusts is True:
            print('\tfiltering clusters...')
            # find key associated with clust id
            _key_id = find_key_partialmatch(self.spk_raw.clust_kslabel, 'id')

            # filter clusts based on quality
            _good_clusts_pd = self.spk_raw.clust_kslabel[
                self.spk_raw.clust_kslabel['KSLabel'] == 'good']
            self.spk_raw._good_clust_ind = np.array(_good_clusts_pd[_key_id])
            self.spk_raw._good_clust_mask = np.isin(
                self.spk_raw.clusters, self.spk_raw._good_clust_ind)

            self.spk_raw.clusters = self.spk_raw.clusters[
                self.spk_raw._good_clust_mask]
            self.spk_raw.times = self.spk_raw.times[
                self.spk_raw._good_clust_mask]

        # extract spiketimes and correct for sampling rate
        print('\textracting spiketimes...')
        self.spk_raw.times = [self.spk_raw.times[i]
                              for i in range(len(self.spk_raw.times))]
        self.spk_raw.times = np.array(self.spk_raw.times, dtype=float)
        self.spk_raw.times /= self.spk_raw.samp_rate_hz

        # align to behavior, if aligner_obj is present
        print('\taligning ephys to behavior...')
        if aligner_obj is False:
            self.aligner_obj = False
            print('\t\tno aligner_obj present.')
        elif aligner_obj is not False:
            self.aligner_obj = aligner_obj
            self.spk_raw.times = self.aligner_obj.correct_ephys_data(
                self.spk_raw.times)
            # self.spk_raw.times = self.deploy_aligner_on_ephys_data(
            #     self.spk_raw.times)
            print('\t\taligned.')

        # go back to original directory
        os.chdir(_start_dir)

        return

    def add_spktrain(self, samp_rate_spktrain_hz=1000):
        """
        Converts a list of spiketimes and cluster IDs
        to a spktrain (timebins with the desired sampling rate), then
        adds this spktrain as an attribute to the obj instance.

        Parameters
        --------------
        samp_rate_spktrain_hz : int
            Sampling rate for spktrain conversion

        Attributes added
        ---------------
        self.spk
            self.spk.t
            self.spk.train
        """
        print('\tadding spktrain...')
        _max_t = np.max(self.spk_raw.times)
        # _n_clusts = np.max(self.spk_raw.clusters)
        _clust_list = np.unique(self.spk_raw.clusters)

        self.spk.t = np.arange(0, _max_t, 1/samp_rate_spktrain_hz)
        # self.spk.train = np.empty(_n_clusts, dtype=np.ndarray)
        self.spk.train = np.empty(len(_clust_list), dtype=np.ndarray)
        self.spk.train_mtx = np.zeros((len(_clust_list), len(self.spk.t)))
        self.spk.raw_clust_id = _clust_list

        self.spk.info = SimpleNamespace()

        self.spk.info.clust_id = self.spk.raw_clust_id
        self.spk.info.ch = []
        self.spk.info.region_full = []
        self.spk.info.region = []

        self.spk.info.dv = []
        self.spk.info.ml = []
        self.spk.info.ap = []

        for ind_clust, clust_id in enumerate(_clust_list):
            print(f'\tcluster {ind_clust}...', end='\r')
            self.spk.raw_clust_id[ind_clust] = clust_id
            _inds_clust_spkraw = np.where(self.spk_raw.clusters == clust_id)

            self.spk.train[ind_clust] = convert_spktimes_to_spktrain(
                self.spk_raw.times[_inds_clust_spkraw], self.spk.t,
                method='calc')
            self.spk.train_mtx[ind_clust, :] = self.spk.train[ind_clust]

            try:
                # find key associated with clust id
                _ch = self.hist.map_clust_ch.clust_to_ch(clust_id)
                _region_id = self.ch.region[_ch]
                _region_key = match_region_partial(
                    _region_id,
                    list(self.plt.colors.keys()))

                self.spk.info.ch.append(_ch)
                self.spk.info.region_full.append(_region_id)
                self.spk.info.region.append(_region_key)

                _coords = self.hist.get_clust_coords(clust_id)
                self.spk.info.dv.append(_coords.dv)
                self.spk.info.ap.append(_coords.ap)
                self.spk.info.ml.append(_coords.ml)

            except:
                self.spk.info.ch.append(None)
                self.spk.info.region_full.append(None)
                self.spk.info.region.append(None)
                self.spk.info.dv.append(None)
        return

    def add_spktrain_kernconv(self, kern_sd=5, samp_rate_spktrain_hz=250):
        """
        Add kernel-convolved spike trains (smoothed firing rates).

        Generates smoothed firing rates by convolving spike trains with a
        Gaussian kernel and stores both raw and z-scored versions.

        Parameters
        ----------
        kern_sd : float, optional
            Standard deviation of Gaussian kernel in milliseconds, by default 5.
        samp_rate_spktrain_hz : float, optional
            Sampling rate for spike train in Hz, by default 250.

        Returns
        -------
        None
            Creates self.spk.train_kernconv attribute with firing rate estimates.
        """
        key_gauss_stdev = str(kern_sd)+'ms'

        self.add_spktrain(samp_rate_spktrain_hz=samp_rate_spktrain_hz)

        if hasattr(self.spk, 'kernconv') is False:
            self.spk.train_kernconv = SimpleNamespace()
            self.spk.train_kernconv.fr = {}
            self.spk.train_kernconv.fr_zscore = {}
            self.spk.train_kernconv.t = self.spk.t

        self.spk.train_kernconv.fr[key_gauss_stdev] = np.empty_like(
            self.spk.train, dtype=np.ndarray)
        self.spk.train_kernconv.fr_zscore[key_gauss_stdev] = np.empty_like(
            self.spk.train, dtype=np.ndarray)            

        print('\tconvolving with kernel...')
        for ind_clust, clust_id in enumerate(self.spk.raw_clust_id):
            print(f'\tcluster {ind_clust}...', end='\r')

            # convolve with gaussian
            _pointproc_gauss = smooth_spktrain(
                self.spk.train[ind_clust], self.spk.t,
                kern_sd)

            if np.max(_pointproc_gauss) != 0:
                self.spk.train_kernconv.fr[key_gauss_stdev][ind_clust] \
                    = _pointproc_gauss
                self.spk.train_kernconv.fr_zscore[key_gauss_stdev][ind_clust] \
                    = sp_stats.zscore(_pointproc_gauss)
            else:
                self.spk.train_kernconv.fr[key_gauss_stdev][ind_clust] \
                    = np.zeros_like(self.spk.train_kernconv.t, dtype=np.int8)
                self.spk.train_kernconv.fr_zscore[key_gauss_stdev][ind_clust] \
                    = np.zeros_like(self.spk.train_kernconv.t, dtype=np.int8)

        return

    def get_subset_spktrain(self, t_start=None, t_end=None, region=None,
                            get_kernconv=True, get_zscore=True,
                            kernconv_sd_key=None):
        """
        Retrieves a subset of the pointprocess array from a defined
        time window and region.

        Parameters
        ------------
        t_start : None or float
            Start time to extract spikes from. If None, uses full time
        t_end : None or float
            End time to extract spikes from. If None, uses full time.
        region : None or str
            Either None (extract all neurons) or str (extract neurons only
            from that named area, found in ephysobj.spk.info.region)
        """
        spk = SimpleNamespace()

        # parse by region
        # ----------------
        if region is None:
            _inds_neur = np.arange(len(self.spk.train))
        else:
            _inds_neur = np.where(np.array(self.spk.info.region) == region)[0]

        # parse by time
        # ----------------
        if t_start is not None:
            _ind_t_start = np.argmin(np.abs(self.spk.t-t_start))
        elif t_start is None:
            _ind_t_start = 0

        if t_end is not None:
            _ind_t_end = np.argmin(np.abs(self.spk.t-t_end))
        elif t_end is None:
            _ind_t_end = -1

        # save in spk simplenamespace
        # -----------
        if get_kernconv is False:
            spk.train = self.spk.train_mtx[_inds_neur,
                                           _ind_t_start:_ind_t_end]
        elif get_kernconv is True:
            # set sd key if not present
            if kernconv_sd_key is None:
                kernconv_sd_key = list(self.spk.train_kernconv.fr.keys())[0]

            # configure spk.train to be 2d matrix
            _templ = self.spk.train_kernconv.fr[kernconv_sd_key] \
                [_inds_neur[0]][_ind_t_start:_ind_t_end]
            spk.train = np.zeros((len(_inds_neur), _templ.shape[0]))

            # store values
            for _ind_rel_neur, _ind_abs_neur in enumerate(_inds_neur):
                if get_zscore is False:
                    spk.train[_ind_rel_neur, :] \
                        = self.spk.train_kernconv.fr[kernconv_sd_key] \
                        [_ind_abs_neur][_ind_t_start:_ind_t_end]
                elif get_zscore is True:
                    spk.train[_ind_rel_neur, :] \
                        = self.spk.train_kernconv.fr_zscore[kernconv_sd_key] \
                        [_ind_abs_neur][_ind_t_start:_ind_t_end]

        spk.t = self.spk.train_kernconv.t[_ind_t_start:_ind_t_end]

        return spk

    def calc_pca(self, bin_size=100, region=None, t_range=None):
        """
        Parameters
        ----------------
        bin_size : int
            Bin size in ms
        region : None or list
            Regions from which to extract units and calculate pca
        t_range : None or list
            Range of times from which to calculate pca in seconds
        """
        self.add_spktrain(samp_rate_spktrain_hz=1/(bin_size/1000))

        if type(region) == str:
            region = [region]

        if region is None:
            _clust_count = self.spk.raw_clust_id.shape[0]
            _clust_inds = np.arange(self.spk.train.shape[0])
        else:
            _clust_count = 0
            _clust_inds = []
            for ind_clust, raw_clust_id in enumerate(self.spk.raw_clust_id):
                try:
                    _reg = self.spk.info.region[ind_clust]

                    if _reg in region:
                        _clust_inds.append(ind_clust)
                        _clust_count += 1

                except IndexError:
                    pass
                except TypeError:
                    pass

        pointproc_matrix = np.zeros((_clust_count,
                                     self.spk.train[0].shape[0]),
                                    dtype=int)

        for ind_clust_in_matrix, ind_clust_id in enumerate(_clust_inds):
            _t_spk = self.spk.train[ind_clust_id]
            pointproc_matrix[ind_clust_in_matrix, :] = _t_spk

        # run pca
        # ------------
        pca = decomposition.TruncatedSVD(n_components=3)
        if t_range is None:
            pca.fit(pointproc_matrix.T)
            self.pointproc_pca = pca.transform(pointproc_matrix.T)
        else:
            ind_t_min = np.argmin(np.abs(self.spk.t - t_range[0]))
            ind_t_max = np.argmin(np.abs(self.spk.t - t_range[1]))

            pca.fit(pointproc_matrix[ind_t_min:ind_t_max].T)
            self.pointproc_pca = pca.transform(
                pointproc_matrix[ind_t_min:ind_t_max].T)

    def calc_synchrony_index(self, bin_size_count_spks=10,
                             bin_size_synch_index=1000,
                             region=None):
        """
        Calculate population synchrony index across time.

        Parameters
        ----------
        bin_size_count_spks : float, optional
            Bin size for counting spikes in milliseconds, by default 10.
        bin_size_synch_index : float, optional
            Bin size for computing synchrony index in milliseconds, by default 1000.
        region : str or list of str, optional
            Brain region(s) to analyze; if None, use all neurons, by default None.

        Returns
        -------
        None
            Creates self.synch_index attribute with synchrony measures.
        """

        bin_size_spkcount_sec = bin_size_count_spks/1000
        self.add_spktrain(samp_rate_spktrain_hz=int(1/bin_size_spkcount_sec))

        # setup subset of the spktrain matrix corresponding to the region
        if region is None:
            _clust_count = self.spk.raw_clust_id.shape[0]
            _clust_inds = np.arange(self.spk.train.shape[0])
        else:
            _clust_count = 0
            _clust_inds = []
            for ind_clust, raw_clust_id in enumerate(self.spk.raw_clust_id):
                try:
                    # find key associated with clust id
                    _key_id = find_key_partialmatch(
                        self.spk_raw.clust_info, 'id')

                    _clust_info = self.spk_raw.clust_info[
                        self.spk_raw.clust_info[_key_id] == raw_clust_id]
                    _ch = np.array(_clust_info['ch'])[0]

                    _region_id = self.ch.region[_ch]
                    _region_key = match_region_partial(
                        _region_id,
                        list(self.plt.colors.keys()))

                    if _region_key in region:
                        _clust_inds.append(ind_clust)
                        _clust_count += 1

                except IndexError:
                    pass
                except TypeError:
                    pass

        pointproc_matrix = np.zeros((_clust_count,
                                     self.spk.train[0].shape[0]),
                                    dtype=int)

        for ind_clust_in_matrix, ind_clust_id in enumerate(_clust_inds):
            _t_spk = self.spk.train[ind_clust_id]
            pointproc_matrix[ind_clust_in_matrix, :] = _t_spk

        # Compute synchrony index
        # ----------------
        spk_count = np.sum(pointproc_matrix, axis=0)
        synch_scaling_factor = int(bin_size_synch_index / bin_size_count_spks)
        n_synch = np.floor(spk_count.shape[0]
                           / synch_scaling_factor).astype(int)

        self.psi = SimpleNamespace()
        self.psi.index = np.zeros(n_synch-1)
        self.psi.t = np.zeros_like(self.psi.index)

        for ind_synch in range(n_synch-1):
            _ind_start = int(ind_synch*synch_scaling_factor)
            _ind_end = int((ind_synch+1)*synch_scaling_factor)
            _ind_mid = int((_ind_start+_ind_end)/2)

            _sd = np.std(spk_count[_ind_start:_ind_end])
            _mean = np.mean(spk_count[_ind_start:_ind_end])

            self.psi.index[ind_synch] = _sd / _mean
            self.psi.t[ind_synch] = self.spk.t[_ind_mid]

    def plt_raster(self, scatter_size=2, fig_save=False,
                   t_range=None, region=None):
        """
        Plot spike raster with neurons organized by brain region and depth.

        Parameters
        ----------
        scatter_size : float, optional
            Size of spike markers, by default 2.
        fig_save : bool, optional
            Whether to save the figure, by default False.
        t_range : list of float, optional
            Time range [t_min, t_max] to display, by default None (all time).
        region : str, optional
            Specific brain region to plot; if None, plot all regions, by default None.

        Returns
        -------
        None
        """
        fig = plt.figure(figsize=(6.86, 4))
        ax = fig.add_subplot(1, 1, 1)

        max_n_clusts = np.max(self.spk_raw.clusters)

        for clust_id in range(max_n_clusts):
            print(f'plotting {clust_id=}...', end='\r')
            _t_spk = self.spk_raw.times[self.spk_raw.clusters == clust_id]
            try:
                if self._no_histology is False:
                    _clust_info = self.spk_raw.clust_info[
                        self.spk_raw.clust_info['id'] == clust_id]
                    _ch = np.array(_clust_info['ch'])[0]
                    _region_id = self.ch.region[_ch]
                    _region_key = match_region_partial(
                        _region_id,
                        list(self.plt.colors.keys()))
                    _dv = self.ch.dv_plt[_ch]
                elif self._no_histology is True:
                    _region_key = 'none'
                    _dv = clust_id

                if region is None:
                    ax.scatter(_t_spk, np.ones_like(_t_spk)*_dv,
                               s=scatter_size,
                               color=self.plt.colors[_region_key])
                else:
                    if region == _region_key:
                        ax.scatter(_t_spk, np.ones_like(_t_spk)*_dv,
                                   s=scatter_size,
                                   color=self.plt.colors[_region_key])
            except IndexError:
                pass

        ax.set_xlabel('time (s)')
        ax.set_ylabel('dv (um)')

        if t_range is not None:
            ax.set_xlim(t_range[0], t_range[1])

        if fig_save is True:
            fig.savefig(f'figs/raster_{region=}_{t_range=}.pdf')
        plt.show()

        return

    def plt_raster_and_fr(self, scatter_size=2, fig_save=False,
                          t_range=None, region=None, kern_sd='5ms',
                          plt_areas_separately=True):
        """
        Plot spike raster and population firing rate together.

        Parameters
        ----------
        scatter_size : float, optional
            Size of spike markers, by default 2.
        fig_save : bool, optional
            Whether to save the figure, by default False.
        t_range : list of float, optional
            Time range [t_min, t_max] to display, by default None (all time).
        region : str, optional
            Specific brain region to plot; if None, plot all regions, by default None.
        kern_sd : str, optional
            Kernel standard deviation key for smoothed firing rate, by default '5ms'.
        plt_areas_separately : bool, optional
            Whether to plot different brain areas separately, by default True.

        Returns
        -------
        None
        """
        fig = plt.figure(figsize=(6.86, 4))
        spec = gs.GridSpec(nrows=2, ncols=1, height_ratios=[1, 0.2],
                           figure=fig)
        ax_raster = fig.add_subplot(spec[0, 0])
        ax_fr = fig.add_subplot(spec[1, 0], sharex=ax_raster)

        max_n_clusts = np.max(self.spk_raw.clusters)

        for clust_id in range(max_n_clusts):
            print(f'plotting {clust_id=}...', end='\r')
            _t_spk = self.spk_raw.times[self.spk_raw.clusters == clust_id]
            try:
                _clust_info = self.spk_raw.clust_info[
                    self.spk_raw.clust_info['id'] == clust_id]
                _ch = np.array(_clust_info['ch'])[0]

                _region_id = self.ch.region[_ch]
                _region_key = match_region_partial(
                    _region_id,
                    list(self.plt.colors.keys()))
                _dv = self.ch.dv_plt[_ch]

                if region is None:
                    ax_raster.scatter(_t_spk, np.ones_like(_t_spk)*_dv,
                                      s=scatter_size,
                                      color=self.plt.colors[_region_key])
                else:
                    if region == _region_key:
                        ax_raster.scatter(_t_spk, np.ones_like(_t_spk)*_dv,
                                          s=scatter_size,
                                          color=self.plt.colors[_region_key])
            except IndexError:
                pass


        # firing rate plot
        # ------------
        if plt_areas_separately is True:
            _list_areas = list(self.plt.colors.keys())

            # create struct to store region-specific firing rates
            self.plt.fr = {}
            self.plt.fr_neur_counter = {}
            for _area in _list_areas:
                self.plt.fr[_area] = np.zeros_like(self.spk.train_kernconv.t,
                                                   dtype='single')
                self.plt.fr_neur_counter[_area] = 0

            for clust_id in range(max_n_clusts):
                try:
                    _clust_info = self.spk_raw.clust_info[
                        self.spk_raw.clust_info['id'] == clust_id]
                    _ch = np.array(_clust_info['ch'])[0]

                    _region_id = self.ch.region[_ch]
                    _region_key = match_region_partial(
                        _region_id,
                        list(self.plt.colors.keys()))

                    self.plt.fr[_region_key] += self.spk.train_kernconv.fr[kern_sd]\
                        [clust_id].astype(float)
                    self.plt.fr_neur_counter[_region_key] += 1
                except IndexError:
                    pass

            # normalize each region firing rate by number of neurons and plot
            for _area in _list_areas:
                if region is None:
                    if self.plt.fr_neur_counter[_area] != 0:
                        self.plt.fr[_area] /= self.plt.fr_neur_counter[_area]
                    elif self.plt.fr_neur_counter[_area] == 0:
                        pass

                    ax_fr.plot(self.spk.train_kernconv.t, self.plt.fr[_area],
                               color=self.plt.colors[_area], linewidth=0.6)
                else:
                    if region == _area:
                        self.plt.fr[_area] /= self.plt.fr_neur_counter[_area]
                        ax_fr.plot(self.spk.train_kernconv.t, self.plt.fr[_area],
                                   color=self.plt.colors[_area], linewidth=0.6)

        ax_fr.set_ylabel('fr (Hz)')
        ax_fr.set_xlabel('time (s)')
        ax_raster.set_ylabel('dv (um)')

        if t_range is not None:
            ax_raster.set_xlim(t_range[0], t_range[1])

        if fig_save is True:
            fig.savefig('figs/raster_fr.pdf')
        plt.show()

        return


class EphysData_ValuePFC(object):
    def __init__(self, folder, aligner_obj=False, filt_clusts=True):
        """
        Class for loading and working with neuropixel data associated with
        the ValuePFC behavioral dataset.

        Must be already sorted using kilosort 2.5.

        Parameters
        ---------------
        folder : str
            Name of the ephys folder with kilosort outputs.
        filt_clusts : bool
            If True, only keeps clusters that were identified by kilosort as
            'good' (ie excludes 'noise' and 'mua'.
        samp_rate_pointproc_hz : int
            Sampling rate for the point process spiketrain, in Hz.
            While the real sampling rate for neuropixels probe is quite high
            (~30k hz), this should be much lower.

        Attributes added
        --------------
        self.spk_clusters
        self.spk_times_raw
        self.spk_info

        self.samp_rate_hz

        self.ch_raw
            self.ch_raw.map
            self.ch_raw.pos
        self.ch
            self.ch.region
            self.ch.region_id
            self.ch.ml
            self.ch.ap
            self.ch.dv
        """

        self.folder = folder
        _start_dir = os.getcwd()
        os.chdir(self.folder)

        print('loading ephys...')
        print('\tloading spike data and channel position data...')

        # load spk data
        self.spk = SimpleNamespace()
        self.spk_raw = SimpleNamespace()
        self.spk_raw.clusters = np.load('spike_clusters.npy')
        self.spk_raw.times = np.load('spike_times.npy')
        self.spk_raw.clust_info = pd.read_csv('cluster_info.tsv', sep='\t')
        self.spk_raw.samp_rate_hz = parse_samp_rate_from_params('params.py')

        # load channel position data
        self.ch_raw = SimpleNamespace()
        self.ch_raw.map = np.load('channel_map.npy')
        self.ch_raw.pos = np.load('channel_positions.npy')

        n_ch = len(self.ch_raw.pos)

        self.ch = SimpleNamespace()
        self.ch.region = np.empty(n_ch, dtype=object)
        self.ch.region_id = np.zeros(n_ch)
        self.ch.ml = np.zeros(n_ch)
        self.ch.ap = np.zeros_like(self.ch.ml)
        self.ch.dv = np.zeros_like(self.ch.ml)
        self.ch.dv_plt = np.zeros_like(self.ch.ml)  # offset

        # setup colors for different brain regions
        self.plt = SimpleNamespace()
        self.plt.colors = {}
        self.plt.colors['ACA'] = '#808022'  # olive
        self.plt.colors['AON'] = '#027BBA'  # blue
        self.plt.colors['ILA'] = '#F9B967'  # yellow
        self.plt.colors['MO'] = '#CDAFD4'  # lavender
        self.plt.colors['OLF'] = '#07A339'  # green
        self.plt.colors['ORB'] = '#E1101D'  # red
        self.plt.colors['PL'] = '#EE760D'  # orange
        self.plt.colors['none'] = [0.7, 0.7, 0.7]

        # try to load information about histology-aligned ch locations
        print('\tloading histology alignment data...')

        try:
            with open('Histology_alignment/channel_locations.json') as f:
                self.ch_raw.histalign_anat = json.load(f)

            for ind_ch in range(n_ch):
                _key_ch = 'channel_' + str(ind_ch)
                self.ch.region[ind_ch] \
                    = self.ch_raw.histalign_anat[_key_ch]['brain_region']
                self.ch.region_id[ind_ch] \
                    = self.ch_raw.histalign_anat[_key_ch]['brain_region_id']

                self.ch.ml[ind_ch] \
                    = self.ch_raw.histalign_anat[_key_ch]['x']
                self.ch.ap[ind_ch] \
                    = self.ch_raw.histalign_anat[_key_ch]['y']
                self.ch.dv[ind_ch] \
                    = self.ch_raw.histalign_anat[_key_ch]['z']

            # setup a dv_plt attribute that has dvs that are staggered
            self.ch.dv_plt = np.copy(self.ch.dv)
            self.ch.dv_plt[0:-1] += np.diff(
                self.ch.dv_plt)/2

        except FileNotFoundError:
            print('\tHistology alignment files are not present '
                  + 'for this recording.')

        # filter data based on phy label of 'good'
        if filt_clusts is True:
            print('\tfiltering clusters...')
            # find key associated with clust id
            _key_id = find_key_partialmatch(self.spk_raw.clust_info, 'id')

            # filter clusts based on quality
            _good_clusts_pd = self.spk_raw.clust_info[
                self.spk_raw.clust_info['group'] == 'good']
            self.spk_raw._good_clust_ind = np.array(_good_clusts_pd[_key_id])
            self.spk_raw._good_clust_mask = np.isin(
                self.spk_raw.clusters, self.spk_raw._good_clust_ind)

            self.spk_raw.clusters = self.spk_raw.clusters[
                self.spk_raw._good_clust_mask]
            self.spk_raw.times = self.spk_raw.times[
                self.spk_raw._good_clust_mask]

        # extract spiketimes and correct for sampling rate
        print('\textracting spiketimes...')
        self.spk_raw.times = [self.spk_raw.times[i][0]
                              for i in range(len(self.spk_raw.times))]
        self.spk_raw.times = np.array(self.spk_raw.times, dtype=float)
        self.spk_raw.times /= self.spk_raw.samp_rate_hz

        # align to behavior, if aligner_obj is present
        print('\taligning ephys to behavior...')
        if aligner_obj is False:
            self.aligner_obj = False
            print('\t\tno aligner_obj present.')
        elif aligner_obj is not False:
            self.aligner_obj = aligner_obj
            self.spk_raw.times = self.aligner_obj.correct_ephys_data(
                self.spk_raw.times)
            # self.spk_raw.times = self.deploy_aligner_on_ephys_data(
            #     self.spk_raw.times)
            print('\t\taligned.')

        # go back to original directory
        os.chdir(_start_dir)

        return

    def add_pointprocess(self, samp_rate_pointproc_hz=1000):
        """
        Converts a list of spiketimes and cluster IDs
        to a spktrain (timebins with the desired sampling rate), then
        adds this point process as an attribute to the obj instance.

        Parameters
        --------------
        samp_rate_pointproc_hz : int
            Sampling rate for point processes conversion

        Attributes added
        ---------------
        self.spk
            self.spk.t
            self.spk.pointproc
        """
        print('\tadding point process...')
        _max_t = np.max(self.spk_raw.times)
        # _n_clusts = np.max(self.spk_raw.clusters)
        _clust_list = np.unique(self.spk_raw.clusters)

        self.spk.t = np.arange(0, _max_t, 1/samp_rate_pointproc_hz)
        # self.spk.pointproc = np.empty(_n_clusts, dtype=np.ndarray)
        self.spk.pointproc = np.empty(len(_clust_list), dtype=np.ndarray)
        self.spk.pointproc_mtx = np.zeros((len(_clust_list), len(self.spk.t)))
        self.spk.raw_clust_id = _clust_list

        self.spk.info = SimpleNamespace()

        self.spk.info.clust_id = self.spk.raw_clust_id
        self.spk.info.ch = []
        self.spk.info.region_full = []
        self.spk.info.region = []
        self.spk.info.dv = []

        for ind_clust, clust_id in enumerate(_clust_list):
            print(f'\tcluster {ind_clust}...', end='\r')
            self.spk.raw_clust_id[ind_clust] = clust_id
            _inds_clust_spkraw = np.where(self.spk_raw.clusters == clust_id)

            self.spk.pointproc[ind_clust] = convert_spktimes_to_spktrain(
                self.spk_raw.times[_inds_clust_spkraw], self.spk.t,
                method='calc')
            self.spk.pointproc_mtx[ind_clust, :] = self.spk.pointproc[ind_clust]

            # try:
            # find key associated with clust id
            _key_id = find_key_partialmatch(
                self.spk_raw.clust_info, 'id')

            _clust_info = self.spk_raw.clust_info[
                self.spk_raw.clust_info[_key_id] == clust_id]
            _ch = np.array(_clust_info['ch'])[0]

            _region_id = self.ch.region[_ch]
            _region_key = match_region_partial(
                _region_id,
                list(self.plt.colors.keys()))
            _dv = self.ch.dv_plt[_ch]

            self.spk.info.ch.append(_ch)
            self.spk.info.region_full.append(_region_id)
            self.spk.info.region.append(_region_key)
            self.spk.info.dv.append(_dv)

            # except:
            #     self.spk.info.ch.append(None)
            #     self.spk.info.region_full.append(None)
            #     self.spk.info.region.append(None)
            #     self.spk.info.dv.append(None)
        return

    def add_kern_convol(self, kern_sd=5, samp_rate_pointproc_hz=250):
        key_gauss_stdev = str(kern_sd)+'ms'

        self.add_pointprocess(samp_rate_pointproc_hz=samp_rate_pointproc_hz)

        if hasattr(self.spk, 'kernconv') is False:
            self.spk.kernconv = SimpleNamespace()
            self.spk.kernconv.fr = {}
            self.spk.kernconv.t = self.spk.t

        self.spk.kernconv.fr[key_gauss_stdev] = np.empty_like(
            self.spk.pointproc, dtype=np.ndarray)

        print('\tconvolving with kernel...')
        for ind_clust, clust_id in enumerate(self.spk.raw_clust_id):
            print(f'\tcluster {ind_clust}...', end='\r')

            # convolve with gaussian
            _pointproc_gauss = smooth_spktrain(
                self.spk.pointproc[ind_clust], self.spk.t,
                kern_sd)

            if np.max(_pointproc_gauss) != 0:
                self.spk.kernconv.fr[key_gauss_stdev][ind_clust] \
                    = _pointproc_gauss
            else:
                self.spk.kernconv.fr[key_gauss_stdev][ind_clust] \
                    = np.zeros_like(self.spk.kernconv.t, dtype=np.int8)
        return

    def get_subset_pointproc(self, t_start=None, t_end=None, region=None):
        """
        Retrieves a subset of the pointprocess array from a defined
        time window and region.

        Parameters
        ------------
        t_start : None or float
            Start time to extract spikes from. If None, uses full time
        t_end : None or float
            End time to extract spikes from. If None, uses full time.
        region : None or str
            Either None (extract all neurons) or str (extract neurons only
            from that named area, found in ephysobj.spk.info.region)
        """
        spk = SimpleNamespace()

        # parse by region
        # ----------------
        if region is None:
            _inds_neur = np.arange(len(self.spk.pointproc))
        else:
            _inds_neur = np.where(np.array(self.spk.info.region) == region)[0]

        # parse by time
        # ----------------
        if t_start is not None:
            _ind_t_start = np.argmin(np.abs(self.spk.t-t_start))
        elif t_start is None:
            _ind_t_start = 0

        if t_end is not None:
            _ind_t_end = np.argmin(np.abs(self.spk.t-t_end))
        elif t_end is None:
            _ind_t_end = -1

        # save in spk simplenamespace
        # -----------
        spk.train = self.spk.pointproc_mtx[_inds_neur,
                                           _ind_t_start:_ind_t_end]
        spk.t = self.spk.t[_ind_t_start:_ind_t_end]

        return spk

    def calc_pca(self, bin_size=100, region=None, t_range=None):
        """
        Parameters
        ----------------
        bin_size : int
            Bin size in ms
        region : None or list
            Regions from which to extract units and calculate pca
        t_range : None or list
            Range of times from which to calculate pca in seconds
        """
        self.add_pointprocess(samp_rate_pointproc_hz=1/(bin_size/1000))

        if type(region) == str:
            region = [region]

        if region is None:
            _clust_count = self.spk.raw_clust_id.shape[0]
            _clust_inds = np.arange(self.spk.pointproc.shape[0])
        else:
            _clust_count = 0
            _clust_inds = []
            for ind_clust, raw_clust_id in enumerate(self.spk.raw_clust_id):
                try:
                    # find key associated with clust id
                    _key_id = find_key_partialmatch(
                        self.spk_raw.clust_info, 'id')

                    _clust_info = self.spk_raw.clust_info[
                        self.spk_raw.clust_info[_key_id] == raw_clust_id]
                    _ch = np.array(_clust_info['ch'])[0]

                    _region_id = self.ch.region[_ch]
                    _region_key = match_region_partial(
                        _region_id,
                        list(self.plt.colors.keys()))

                    if _region_key in region:
                        _clust_inds.append(ind_clust)
                        _clust_count += 1

                except IndexError:
                    pass
                except TypeError:
                    pass

        pointproc_matrix = np.zeros((_clust_count,
                                     self.spk.pointproc[0].shape[0]),
                                    dtype=int)

        for ind_clust_in_matrix, ind_clust_id in enumerate(_clust_inds):
            _t_spk = self.spk.pointproc[ind_clust_id]
            pointproc_matrix[ind_clust_in_matrix, :] = _t_spk

        # run pca
        # ------------
        pca = decomposition.TruncatedSVD(n_components=3)
        if t_range is None:
            pca.fit(pointproc_matrix.T)
            self.pointproc_pca = pca.transform(pointproc_matrix.T)
        else:
            ind_t_min = np.argmin(np.abs(self.spk.t - t_range[0]))
            ind_t_max = np.argmin(np.abs(self.spk.t - t_range[1]))

            pca.fit(pointproc_matrix[ind_t_min:ind_t_max].T)
            self.pointproc_pca = pca.transform(
                pointproc_matrix[ind_t_min:ind_t_max].T)

    def calc_synchrony_index(self, bin_size_count_spks=10,
                             bin_size_synch_index=1000,
                             region=None):

        bin_size_spkcount_sec = bin_size_count_spks/1000
        self.add_pointprocess(samp_rate_pointproc_hz=int(1/bin_size_spkcount_sec))

        # setup subset of the pointprocess matrix corresponding to the region
        if region is None:
            _clust_count = self.spk.raw_clust_id.shape[0]
            _clust_inds = np.arange(self.spk.pointproc.shape[0])
        else:
            _clust_count = 0
            _clust_inds = []
            for ind_clust, raw_clust_id in enumerate(self.spk.raw_clust_id):
                try:
                    # find key associated with clust id
                    _key_id = find_key_partialmatch(
                        self.spk_raw.clust_info, 'id')

                    _clust_info = self.spk_raw.clust_info[
                        self.spk_raw.clust_info[_key_id] == raw_clust_id]
                    _ch = np.array(_clust_info['ch'])[0]

                    _region_id = self.ch.region[_ch]
                    _region_key = match_region_partial(
                        _region_id,
                        list(self.plt.colors.keys()))

                    if _region_key in region:
                        _clust_inds.append(ind_clust)
                        _clust_count += 1

                except IndexError:
                    pass
                except TypeError:
                    pass

        pointproc_matrix = np.zeros((_clust_count,
                                     self.spk.pointproc[0].shape[0]),
                                    dtype=int)

        for ind_clust_in_matrix, ind_clust_id in enumerate(_clust_inds):
            _t_spk = self.spk.pointproc[ind_clust_id]
            pointproc_matrix[ind_clust_in_matrix, :] = _t_spk

        # Compute synchrony index
        # ----------------
        spk_count = np.sum(pointproc_matrix, axis=0)
        synch_scaling_factor = int(bin_size_synch_index / bin_size_count_spks)
        n_synch = np.floor(spk_count.shape[0]
                           / synch_scaling_factor).astype(int)

        self.psi = SimpleNamespace()
        self.psi.index = np.zeros(n_synch-1)
        self.psi.t = np.zeros_like(self.psi.index)

        for ind_synch in range(n_synch-1):
            _ind_start = int(ind_synch*synch_scaling_factor)
            _ind_end = int((ind_synch+1)*synch_scaling_factor)
            _ind_mid = int((_ind_start+_ind_end)/2)

            _sd = np.std(spk_count[_ind_start:_ind_end])
            _mean = np.mean(spk_count[_ind_start:_ind_end])

            self.psi.index[ind_synch] = _sd / _mean
            self.psi.t[ind_synch] = self.spk.t[_ind_mid]

    def plt_raster(self, scatter_size=2, fig_save=False,
                   t_range=None, region=None):
        """
        Plot spike raster with neurons organized by brain region and depth.

        Parameters
        ----------
        scatter_size : float, optional
            Size of spike markers, by default 2.
        fig_save : bool, optional
            Whether to save the figure, by default False.
        t_range : list of float, optional
            Time range [t_min, t_max] to display, by default None (all time).
        region : str, optional
            Specific brain region to plot; if None, plot all regions, by default None.

        Returns
        -------
        None
        """
        fig = plt.figure(figsize=(6.86, 4))
        ax = fig.add_subplot(1, 1, 1)

        max_n_clusts = np.max(self.spk_raw.clusters)

        for clust_id in range(max_n_clusts):
            print(f'plotting {clust_id=}...', end='\r')
            _t_spk = self.spk_raw.times[self.spk_raw.clusters == clust_id]
            try:
                _clust_info = self.spk_raw.clust_info[
                    self.spk_raw.clust_info['id'] == clust_id]
                _ch = np.array(_clust_info['ch'])[0]

                _region_id = self.ch.region[_ch]
                _region_key = match_region_partial(
                    _region_id,
                    list(self.plt.colors.keys()))
                _dv = self.ch.dv_plt[_ch]

                if region is None:
                    ax.scatter(_t_spk, np.ones_like(_t_spk)*_dv,
                               s=scatter_size,
                               color=self.plt.colors[_region_key])
                else:
                    if region == _region_key:
                        ax.scatter(_t_spk, np.ones_like(_t_spk)*_dv,
                                   s=scatter_size,
                                   color=self.plt.colors[_region_key])
            except IndexError:
                pass

        ax.set_xlabel('time (s)')
        ax.set_ylabel('dv (um)')

        if t_range is not None:
            ax.set_xlim(t_range[0], t_range[1])

        if fig_save is True:
            fig.savefig(f'figs/raster_{region=}_{t_range=}.pdf')
        plt.show()

        return

    def plt_raster_and_fr(self, scatter_size=2, fig_save=False,
                          t_range=None, region=None, kern_sd='5ms',
                          plt_areas_separately=True):
        """
        Plot spike raster and population firing rate together.

        Parameters
        ----------
        scatter_size : float, optional
            Size of spike markers, by default 2.
        fig_save : bool, optional
            Whether to save the figure, by default False.
        t_range : list of float, optional
            Time range [t_min, t_max] to display, by default None (all time).
        region : str, optional
            Specific brain region to plot; if None, plot all regions, by default None.
        kern_sd : str, optional
            Kernel standard deviation key for smoothed firing rate, by default '5ms'.
        plt_areas_separately : bool, optional
            Whether to plot different brain areas separately, by default True.

        Returns
        -------
        None
        """
        fig = plt.figure(figsize=(6.86, 4))
        spec = gs.GridSpec(nrows=2, ncols=1, height_ratios=[1, 0.2],
                           figure=fig)
        ax_raster = fig.add_subplot(spec[0, 0])
        ax_fr = fig.add_subplot(spec[1, 0], sharex=ax_raster)

        max_n_clusts = np.max(self.spk_raw.clusters)

        for clust_id in range(max_n_clusts):
            print(f'plotting {clust_id=}...', end='\r')
            _t_spk = self.spk_raw.times[self.spk_raw.clusters == clust_id]
            try:
                _clust_info = self.spk_raw.clust_info[
                    self.spk_raw.clust_info['id'] == clust_id]
                _ch = np.array(_clust_info['ch'])[0]

                _region_id = self.ch.region[_ch]
                _region_key = match_region_partial(
                    _region_id,
                    list(self.plt.colors.keys()))
                _dv = self.ch.dv_plt[_ch]

                if region is None:
                    ax_raster.scatter(_t_spk, np.ones_like(_t_spk)*_dv,
                                      s=scatter_size,
                                      color=self.plt.colors[_region_key])
                else:
                    if region == _region_key:
                        ax_raster.scatter(_t_spk, np.ones_like(_t_spk)*_dv,
                                          s=scatter_size,
                                          color=self.plt.colors[_region_key])
            except IndexError:
                pass


        # firing rate plot
        # ------------
        if plt_areas_separately is True:
            _list_areas = list(self.plt.colors.keys())

            # create struct to store region-specific firing rates
            self.plt.fr = {}
            self.plt.fr_neur_counter = {}
            for _area in _list_areas:
                self.plt.fr[_area] = np.zeros_like(self.spk.kernconv.t,
                                                   dtype='single')
                self.plt.fr_neur_counter[_area] = 0

            for clust_id in range(max_n_clusts):
                try:
                    _clust_info = self.spk_raw.clust_info[
                        self.spk_raw.clust_info['id'] == clust_id]
                    _ch = np.array(_clust_info['ch'])[0]

                    _region_id = self.ch.region[_ch]
                    _region_key = match_region_partial(
                        _region_id,
                        list(self.plt.colors.keys()))

                    self.plt.fr[_region_key] += self.spk.kernconv.fr[kern_sd]\
                        [clust_id].astype(float)
                    self.plt.fr_neur_counter[_region_key] += 1
                except IndexError:
                    pass

            # normalize each region firing rate by number of neurons and plot
            for _area in _list_areas:
                if region is None:
                    if self.plt.fr_neur_counter[_area] != 0:
                        self.plt.fr[_area] /= self.plt.fr_neur_counter[_area]
                    elif self.plt.fr_neur_counter[_area] == 0:
                        pass

                    ax_fr.plot(self.spk.kernconv.t, self.plt.fr[_area],
                               color=self.plt.colors[_area], linewidth=0.6)
                else:
                    if region == _area:
                        self.plt.fr[_area] /= self.plt.fr_neur_counter[_area]
                        ax_fr.plot(self.spk.kernconv.t, self.plt.fr[_area],
                                   color=self.plt.colors[_area], linewidth=0.6)

        ax_fr.set_ylabel('fr (Hz)')
        ax_fr.set_xlabel('time (s)')
        ax_raster.set_ylabel('dv (um)')

        if t_range is not None:
            ax_raster.set_xlim(t_range[0], t_range[1])

        if fig_save is True:
            fig.savefig('figs/raster_fr.pdf')
        plt.show()

        return


