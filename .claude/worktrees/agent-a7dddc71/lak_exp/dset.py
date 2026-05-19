import numpy as np
import pandas as pd
import os
import sys


class DSetObj(object):
    def __init__(self,
                 path_reportopto='/Volumes/BRussell/Ephys/'
                 + 'ReportOpto_DataFrame.csv',
                 path_to_server=None):
        """
        Creates a dataset object that stores information about all experiments
        with neuropixels recordings.

        Also contains methods to parse pathnames and navigate to these paths
        quickly, for both behavior and ephys folders of each experiment.

        Parameters
        ------------------
        path_reportopto : str
            Path leading to the location of the .csv holding info about the
            reportopto dataset.
            - Typically called 'ReportOpto_DataFrame.csv' and located on
            Blake Russell's personal QNAP folder.
        path_to_server : None or str
            if None, server path is figured out from the system platform
            (macOS = /Volumes/Data; windows = Z:).

        Attributes
        ----------------
        self._dset_raw : pd.DataFrame
            Stores the raw dataset including experiments with no neuropixels
            recordings.
        self.npix : pd.DataFrame
            Stores filtered dataset including only those with
            neuropixels recordings.
        self.inds_npix : np.array
            Stores indices of all recordings with neuropixels, used as inputs
            to self.npix. (discontinuous indices necessitate the use of this.)
        """

        # figure out platform and server path
        self._platform = sys.platform

        if path_to_server is not None:
            self._path_to_server = path_to_server
        else:
            if 'darwin' in self._platform:
                self._path_to_server = '/Volumes/Data'
            elif 'win32' in self._platform:
                self._path_to_server = "Z:\\"
            elif 'win64' in self._platform:
                self._path_to_server = "Z:\\"

        # load dataset
        print('loading dataset...')
        self.path_reportopto = path_reportopto
        self._dset_raw = pd.read_csv(self.path_reportopto)

        # filter based on presence of npix recordings
        print('filtering neuropixels recordings...')
        self.npix = self._dset_raw[
            (self._dset_raw['Ephys'] == 'Yes')]
        self.inds_npix = np.where(
            self._dset_raw['Ephys'] == 'Yes')[0]

        return

    def _parse_expref(self, ind, ref_frame='npix'):
        """Parses the experimental reference (filepath) for a given ind.
        ref_frame can be 'npix' (neuropixels-indexed) or 'all' (all-indexed)
        """

        if ref_frame == 'npix':
            _animal = self.npix['Animal'].iloc[ind]
            _date = self.npix['Date'].iloc[ind]
            _beh_file = self.npix['Exp_Ref'].iloc[ind].split('\\')[-1]
        elif ref_frame == 'all':
            _animal = self.npix['Animal'][ind]
            _date = self.npix['Date'][ind]
            _beh_file = self.npix['Exp_Ref'][ind].split('\\')[2]

        paths = [_animal, _date, _beh_file]
        return paths

    def get_path_expref(self, ind):
        prefix = self._path_to_server
        paths = self._parse_expref(ind)
        expref_path = os.path.join(prefix, paths[0], paths[1])
        return expref_path

    def get_path_beh(self, ind):
        "Parses the behavior folder (expref/beh_folder) for a given ind"
        paths = self._parse_expref(ind)
        return paths[-1]

    def get_path_ephys(self, ind, ephys_path='ephys_g0'):
        "Parses the ephys folder (expref/ephys_folder) for a given ind"
        if self.npix['Dual_probe'].iloc[ind] == 'No':
            ephys_path = ephys_path
        elif self.npix['Dual_probe'].iloc[ind] == 'Yes':
            ephys_path = os.path.join(
                ephys_path, self.npix['Imec'].iloc[ind])

        return ephys_path

    def goto_expref(self, ind):
        "Navigates to experimental reference folder for a given ind."
        os.chdir(self._path_to_server)

        paths = self._parse_expref(ind)
        os.chdir(paths[0])
        os.chdir(paths[1])
        return

    def goto_beh(self, ind):
        "Navigates to behavior folder (expref/beh_folder) for a given ind."
        self.goto_expref(ind)

        beh_folder = self.get_path_beh(ind)
        os.chdir(beh_folder)
        return

    def goto_ephys(self, ind):
        "Goes to ephys folder (expref/ephys_folder) for a given ind."
        self.goto_expref(ind)

        ephys_folder = self.get_path_ephys(ind)
        os.chdir(ephys_folder)
        return

    def expref_to_ind(self, expref_str):
        """
        Searches for any partial expref (eg animal name or date)
        and returns ind in dset. (Useful for plotting a particular ind
        of interest using ExpObj.plt_exp() method)
        """
        inds = np.where(self.npix['Exp_Ref'].str.contains(expref_str))[0]
        return inds

    def list_exprefs(self, expref_str, list_format='compact'):
        """
        list format can either be 'full' or 'compact'
        """
        inds = self.expref_to_ind(expref_str)
        for ind in inds:
            print(f'{ind=}\n---------------')
            if list_format == 'full':
                print(self.npix.iloc[ind])
                print('\n')
            elif list_format == 'compact':
                expref = self._parse_expref(ind)
                print(f'\tanimal: {expref[0]}')
                print(f'\tdate: {expref[1]}')
                print(f'\tn_rec: {expref[2]}\n')

    def get_region_inds(self, region):
        """
        region can either be a string, or a list of strings
        (treated as logical OR).
        """
        _inds_rec_filt = []

        if type(region) is str:
            for rec in range(self.npix.shape[0]):
                if region.lower() in self.npix.iloc[rec][
                        'Probe_Region_Info'].lower():
                    _inds_rec_filt.append(rec)
        elif type(region) is list:
            for rec in range(self.npix.shape[0]):
                for ind, _region in enumerate(region):
                    if _region.lower() in self.npix.iloc[rec][
                            'Probe_Region_Info'].lower():
                        _inds_rec_filt.append(rec)
            _inds_rec_filt = np.unique(_inds_rec_filt)
        return _inds_rec_filt


class DSetObj_ValuePFC(object):
    def __init__(self,
                 path_pfcdataset='/Users/michaellynn/Documents/MATLAB/'
                 + 'ValuePFC/project_datasets/'
                 + 'session_info/ValuePFC_main_sessionInfo.csv',
                 os_type='mac'):
        """
        Creates a dataset object that stores information about all experiments
        with neuropixels recordings.

        Also contains methods to parse pathnames and navigate to these paths
        quickly, for both behavior and ephys folders of each experiment.

        Parameters
        ------------------
        path_pfcdataset : str
            Path leading to the location of the .csv holding info about the
            value PFC dataset.
            - Typically called 'ValuePFC_main_sessionInfo.csv'
            and found on github page:
                LakLab/ValuePFC/project_datasets/session_info/
                ValuePFC_main_sessionInfo.csv
            (must clone this repo somewhere local and then put path here.)

        os_type : str
            Type of OS. Currently only 'mac' is implemented. Used to
            automatically mount the QNAP volume with an os command.
            (Feasibly both linux and windows could be accomodated with the
            correct os call.)

        Attributes
        ----------------
        self._dset_raw : pd.DataFrame
            Stores the raw dataset including experiments with no neuropixels
            recordings.
        self.npix : pd.DataFrame
            Stores filtered dataset including only those with
            neuropixels recordings.
        self.inds_npix : np.array
            Stores indices of all recordings with neuropixels, used as inputs
            to self.npix. (discontinuous indices necessitate the use of this.)
        """

        # load dataset
        print('loading dataset...')
        self.path_pfcdataset = path_pfcdataset
        self._dset_raw = pd.read_csv(self.path_pfcdataset)

        # # mount the QNAP drive
        # if os_type == 'mac':
        #     os.system("osascript -e 'mount volume "
        #               + "\"smb://QNAP-AL001.dpag.ox.ac.uk/Data\"'")

        # filter based on presence of npix recordings
        print('filtering neuropixels recordings...')
        self.npix = self._dset_raw[
            (self._dset_raw['neuropixels_recording'] == True)
            & (self._dset_raw['aligned_recording'] == True)]
        self.inds_npix = np.where(np.logical_and(
            self._dset_raw['neuropixels_recording'] == True,
            self._dset_raw['aligned_recording'] == True))[0]

        return

    def _parse_expref(self, ind):
        "Parses the experimental reference (filepath) for a given ind"
        _paths_unsorted = np.array(self.npix['exp_ref'])[ind].split('_')
        paths = [_paths_unsorted[2], _paths_unsorted[0], _paths_unsorted[1]]
        return paths

    def get_path_expref(self, ind, prefix='/Volumes/Data'):
        paths = self._parse_expref(ind)
        expref_path = os.path.join(prefix, paths[0], paths[1])
        return expref_path

    def get_path_beh(self, ind):
        "Parses the behavior folder (expref/beh_folder) for a given ind"
        paths = self._parse_expref(ind)
        return paths[-1]

    def get_path_ephys(self, ind):
        "Parses the ephys folder (expref/ephys_folder) for a given ind"
        paths = np.array(self.npix['ephys_data_path'])[ind].split('\\')
        return paths[-1]

    def goto_expref(self, ind):
        "Navigates to experimental reference folder for a given ind."
        os.chdir('/Volumes/Data')

        paths = self._parse_expref(ind)
        os.chdir(paths[0])
        os.chdir(paths[1])
        return

    def goto_beh(self, ind):
        "Navigates to behavior folder (expref/beh_folder) for a given ind."
        self.goto_expref(ind)

        beh_folder = self.get_path_beh(ind)
        os.chdir(beh_folder)
        return

    def goto_ephys(self, ind):
        "Goes to ephys folder (expref/ephys_folder) for a given ind."
        self.goto_expref(ind)

        ephys_folder = self.get_path_ephys(ind)
        os.chdir(ephys_folder)
        return

    def expref_to_ind(self, expref_str):
        """
        Searches for any partial expref (eg animal name or date)
        and returns ind in dset. (Useful for plotting a particular ind
        of interest using ExpObj.plt_exp() method)
        """
        inds = np.where(self.npix['exp_ref'].str.contains(expref_str))[0]
        return inds

    def list_exprefs(self, expref_str, list_format='compact'):
        """
        list format can either be 'full' or 'compact'
        """
        inds = self.expref_to_ind(expref_str)
        for ind in inds:
            print(f'{ind=}\n---------------')
            if list_format == 'full':
                print(self.npix.iloc[ind])
                print('\n')
            elif list_format == 'compact':
                expref = self.npix.iloc[ind]['exp_ref'].split('_')
                print(f'\tanimal: {expref[2]}')
                print(f'\tdate: {expref[0]}')
                print(f'\tn_rec: {expref[1]}\n')



