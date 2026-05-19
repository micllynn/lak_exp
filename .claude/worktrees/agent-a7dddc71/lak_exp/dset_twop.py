import numpy as np
import pandas as pd
import os
import sys
from types import SimpleNamespace


class DSetObj_5HTCtx(object):
    def __init__(self,
                 path_5htctx='/Users/michaellynn/Desktop/'
                 + 'postdoc/projects/5HTCtx/cohorts/visual_pavlov_mbl012-018/'
                 + 'filelist_5HTCtx_3stim.csv',
                 path_to_server=None):
        """
        Creates a dataset object that stores information about all experiments
        with 5HTCtx experiments

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
        self.path_5htctx = path_5htctx
        self._dset_raw = pd.read_csv(self.path_5htctx)

        # # filter based on presence of npix recordings
        # print('filtering recordings...')
        # self.npix = self._dset_raw[
        #     (self._dset_raw['Ephys'] == 'Yes')]
        # self.inds_npix = np.where(
        #     self._dset_raw['Ephys'] == 'Yes')[0]

        return

    def _parse_expref(self, ind):
        """Parses the experimental reference (filepath) for a given ind.
        """
        _expref = self._dset_raw['expref'][ind].split('_')

        paths = SimpleNamespace()
        paths = [_expref[0], _expref[1], _expref[2]]

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

    def get_path_img(self, ind):
        folder_img = self._dset_raw['dual2p_folder'][ind]
        os.chdir(os.path.join(self.get_path_expref(ind), 'TwoP'))
        _folders = os.listdir()
        _img_folder = None
        for _folder in _folders:
            if _folder.endswith(folder_img) and not _folder.startswith('.'):
                _img_folder = _folder

        if _img_folder is not None:
            _img_folder_full = os.path.join('TwoP', _img_folder)
            return _img_folder_full
        elif _img_folder is None:
            raise FileNotFoundError(f'the imaging folder ending in {folder_img}' +
                                    'was not found.')

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

    def goto_img(self, ind, folder_img='t-001'):
        "Goes to ephys folder (expref/ephys_folder) for a given ind."
        self.goto_expref(ind)

        img_folder = self.get_path_img()
        os.chdir(img_folder)
        return

    def expref_to_ind(self, expref_str):
        """
        Searches for any partial expref (eg animal name or date)
        and returns ind in dset. (Useful for plotting a particular ind
        of interest using ExpObj.plt_exp() method)
        """
        inds = np.where(self._dset_raw['expref'].str.contains(expref_str))[0]
        return inds

    def list_exprefs(self, expref_str, list_format='compact'):
        """
        list format can either be 'full' or 'compact'
        """
        inds = self.expref_to_ind(expref_str)
        for ind in inds:
            print(f'{ind=}\n---------------')
            if list_format == 'full':
                print(self._dset_raw.iloc[ind])
                print('\n')
            elif list_format == 'compact':
                expref = self._parse_expref(ind)
                print(f'\tanimal: {expref[0]}')
                print(f'\tdate: {expref[1]}')
                print(f'\tn_rec: {expref[2]}\n')

