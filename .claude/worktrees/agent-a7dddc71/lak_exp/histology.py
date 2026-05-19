import numpy as np
from types import SimpleNamespace
import os
import pickle


class ClustChMap(object):
    def __init__(self, folder='ephys_g0'):
        """
        Constructs a mapping from cluster number to the closest
        channel (euclidean distance).

        Requires kilosort4 output (incorporating spike_positions.npy).
        """
        print('\tmaking clust->ch map...')
        os.chdir(folder)

        print('\t\tloading cluster and channel info...')
        self.spk_pos = np.load('spike_positions.npy')
        self.spk_clusts = np.load('spike_clusters.npy')
        self.ch_pos = np.load('channel_positions.npy')
        self.ch_map = np.load('channel_map.npy')

        self._map = SimpleNamespace()
        self._map.clusts = np.unique(self.spk_clusts)
        self._map.chs = np.zeros_like(self._map.clusts)

        print('\t\tassociating clusts with chs...')
        for clust in self._map.clusts:
            print(f'\t\t\tclust {clust}...', end='\r')
            _clust_ind = np.argwhere(self._map.clusts == clust)[0]

            _first_ind_in_spk_clust = np.where(self.spk_clusts == clust)[0][0]
            _pos = self.spk_pos[_first_ind_in_spk_clust]
            _ch = self._get_ch_with_min_euclid_dist(_pos)
            self._map.chs[_clust_ind] = _ch

        os.chdir('..')

        return

    def _get_ch_with_min_euclid_dist(self, pos_clust_xy):
        """
        Find the channel with minimum Euclidean distance to a cluster position.

        Parameters
        ----------
        pos_clust_xy : array-like
            2D position coordinates [x, y] of the cluster.

        Returns
        -------
        _ch : int
            Channel number with minimum Euclidean distance to the cluster position.
        """
        _subtr = np.abs(self.ch_pos - pos_clust_xy)
        _euclid_dist = np.sqrt(_subtr[:, 0]**2 + _subtr[:, 1]**2)
        _ind_min_dist = np.argmin(_euclid_dist)
        _ch = self.ch_map[_ind_min_dist]
        return _ch

    def clust_to_ch(self, clust):
        """
        Takes cluster number as input and returns the channel with the
        highest amplitude.
        """
        _clust_ind = np.argwhere(self._map.clusts == clust)[0]

        return self._map.chs[_clust_ind][0]


class ChRegionMap(object):
    def __init__(self, folder='../HERBS', probe_file='probe 0.pkl'):
        """
        Constructs a mapping between channel number and region label,
        using HERBS files for probes.
        """
        print('\tmaking ch->region map...')
        print('\t\tgetting HERBS probe file...')
        with open(os.path.join(folder, probe_file), 'rb') as f:
            self.probe = pickle.load(f)

        # get regions
        _regions = np.flip(self.probe['data']['label_name'])
        _ch_count_in_reg = np.flip(self.probe['data']['region_sites'])
        _n_chs = np.sum(_ch_count_in_reg).astype(int)

        self._map = SimpleNamespace()
        self._map.chs = np.arange(_n_chs)
        self._map.regs = np.empty_like(self._map.chs, dtype=list)

        _count = 0
        for ind, _ch_count in enumerate(_ch_count_in_reg):
            self._map.regs[int(_count):int(_count+_ch_count)] = _regions[ind]
            _count += _ch_count

        return

    def ch_to_reg(self, ch):
        """
        Map channel number to brain region.

        Parameters
        ----------
        ch : int
            Channel number.

        Returns
        -------
        region : str
            Brain region label corresponding to the channel.
        """
        return self._map.regs[ch]


class ChCoordMap(object):
    def __init__(self, folder='../HERBS', probe_file='probe 0.pkl'):
        """
        Constructs a mapping between channel number and region label,
        using HERBS files for probes.
        """
        print('\tmaking ch->coordinate map...')
        print('\t\tgetting HERBS probe file...')
        with open(os.path.join(folder, probe_file), 'rb') as f:
            self.probe = pickle.load(f)

        self._coord_labels = ['ML', 'AP', 'DV']

        _coords_raw = self.probe['data']['sites_loc_b']
        self._coords = np.concatenate((_coords_raw[0], _coords_raw[1],
                                       _coords_raw[2], _coords_raw[3]), axis=0)

        # sort by DV, and adjust to Bregma coordinates
        _sorting_inds = np.argsort(self._coords[:, 2])
        self._coords = self._coords[_sorting_inds, :]
        # self._coords[:, 0] -= 570
        # self._coords[:, 2] += 660

        self._coords = self._coords * 10

    def ch_to_coords(self, ch):
        """
        Map channel number to stereotaxic coordinates.

        Parameters
        ----------
        ch : int
            Channel number.

        Returns
        -------
        coords : np.ndarray
            1D array of [ML, AP, DV] coordinates in micrometers.
        """
        return self._coords[ch, :]


class LoadHist(object):
    """
    Load and manage histological mapping data for Neuropixels recordings.

    Provides a unified interface to map cluster IDs to brain regions and
    stereotaxic coordinates using HERBS probe data and Kilosort outputs.

    Parameters
    ----------
    expref : str
        Path to experiment reference directory.
    ephys_folder : str, optional
        Name of ephys data folder, by default 'ephys_g0'.
    probe_file_name : str, optional
        Name of file containing probe filename reference, by default 'probe_file.txt'.
    n_folders_back_herbs_dir : int, optional
        Number of directory levels up to HERBS folder (1 or 2), by default 1.

    Attributes
    ----------
    map_clust_ch : ClustChMap
        Mapping from cluster IDs to channel numbers.
    map_ch_region : ChRegionMap
        Mapping from channel numbers to brain regions.
    map_ch_coords : ChCoordMap
        Mapping from channel numbers to stereotaxic coordinates.
    """
    def __init__(self, expref, ephys_folder='ephys_g0',
                 probe_file_name='probe_file.txt',
                 n_folders_back_herbs_dir=1):
        print('loading histology...')
        os.chdir(expref)

        # setup HERBS directory location
        if n_folders_back_herbs_dir == 1:
            path_to_herbs_folder = '../HERBS'
        if n_folders_back_herbs_dir == 2:
            path_to_herbs_folder = '../../HERBS'

        # get probe file name
        with open(probe_file_name, 'r') as f:
            probe_file_name = str(f.read())

        self.map_clust_ch = ClustChMap(folder=ephys_folder)
        self.map_ch_region = ChRegionMap(folder=path_to_herbs_folder,
                                         probe_file=probe_file_name)
        self.map_ch_coords = ChCoordMap(folder=path_to_herbs_folder,
                                        probe_file=probe_file_name)

    def get_clust_region(self, clust_id):
        """
        Get the brain region for a given cluster ID.

        Parameters
        ----------
        clust_id : int
            Cluster ID from spike sorting.

        Returns
        -------
        region : str
            Brain region label where the cluster is located.
        """
        _ch = self.map_clust_ch.clust_to_ch(clust_id)
        _reg = self.map_ch_region.ch_to_reg(_ch)
        return _reg

    def get_clust_coords(self, clust_id):
        """
        Get stereotaxic coordinates for a given cluster ID.

        Parameters
        ----------
        clust_id : int
            Cluster ID from spike sorting.

        Returns
        -------
        coord : SimpleNamespace
            Object with attributes:
            - ml : float, mediolateral coordinate (micrometers)
            - ap : float, anteroposterior coordinate (micrometers)
            - dv : float, dorsoventral coordinate (micrometers)
        """
        _ch = self.map_clust_ch.clust_to_ch(clust_id)
        _coords_raw = self.map_ch_coords.ch_to_coords(_ch)

        coord = SimpleNamespace()
        coord.ml = _coords_raw[0]
        coord.ap = _coords_raw[1]
        coord.dv = _coords_raw[2]

        return coord
