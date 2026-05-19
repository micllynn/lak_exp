import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from types import SimpleNamespace
import os

from .utils import find_event_onsets, remove_event_duplicates
from .utils_twop import paq_read
from .beh import TimelineParser


class Aligner_ImgBeh():
    def __init__(self):
        """
        Create an object to align the times of imaging and behavior
        data from an experiment.

        Contains methods to extract imaging (paq) and behavior reward echoes,
        compute an alignment with simple linear regression, and
        apply this alignment to imaging data.
        """
        pass

    def parse_img_rewechoes(self, folder='1', echo_type='paq'):
        """
        Parse reward echoes from imaging data (.paq file).

        Parameters
        ----------
        folder : str, optional
            Folder containing the .paq file, by default '1'.
        echo_type : str, optional
            Type of echo file (currently only 'paq' is supported), by default 'paq'.

        Returns
        -------
        None
            Creates self.rew_echo_img and self.paq attributes.
        """
        # extract paq file

        # open the paq file and parse reward echoes and imaging start time
        fnames = os.listdir(folder)
        for _fname in fnames:
            if _fname.endswith('paq'):
                fname = _fname

        self.paq = paq_read(file_path=os.path.join(folder, fname))
        _ind_rewecho = np.where(np.array(self.paq['chan_names'])
                                == 'reward_echo')[0][0]
        _ind_imgstart = np.where(np.array(self.paq['chan_names'])
                                 == '2p_frame')[0][0]

        _samp_rate = self.paq['rate']
        _n_datapoints = self.paq['data'][_ind_rewecho, :].shape[0]

        _times = np.linspace(0, (1/_samp_rate)*_n_datapoints, _n_datapoints)
        _2p_frames = find_event_onsets(
            self.paq['data'][_ind_imgstart, :], thresh=2)
        _2p_frame_times = _times[_2p_frames]

        # calc rew echo times and subtract frame start times
        _rew_echoes = find_event_onsets(
            self.paq['data'][_ind_rewecho, :], thresh=2)
        _rew_echo_times = _times[_rew_echoes]
        _rew_echo_times = remove_event_duplicates(
            _rew_echo_times, abs_refractory_period=0.1)
        _rew_echo_times -= _2p_frame_times[0]

        self.rew_echo_img = _rew_echo_times

    def parse_beh_rewechoes(self, folder='1', echo_type='mat'):
        """
        Get and parse reward echoes for behavior data.

        Behavior reward echoes are either in the .npy format or in
        the Timeline.mat format.

        Rewriting of getEventTimes.m.

        Parameters
        ----------
        folder : str, optional
            Folder containing the reward echo file, by default '1'.
        echo_type : str, optional
            Either 'npy' or 'mat'.
            Use 'npy' if the filename is 'reward_echo.raw.npy'
            and 'mat' if the filename is 'xxx_Timeline.mat', by default 'mat'.

        Returns
        -------
        None
            Creates self.rew_echo_beh attribute with reward echo times.
        """

        files = os.listdir(folder)

        if echo_type == 'mat':
            for f in files:
                if f.startswith('20') and f.endswith('Timeline.mat'):
                    fname = f

            t = TimelineParser(folder+'/'+fname)
            data = t.get_daq_data()

            _inds_onset = find_event_onsets(data.sig['reward_echo'],
                                            thresh=2.5)
            rew_echo = data.t[_inds_onset]
            self.rew_echo_beh = rew_echo

        if echo_type == 'npy':
            d = np.load(folder+'/reward_echo.raw.npy')
            self.rew_echo_beh = find_event_onsets(d[:, 0], thresh=5)

        return

    def compute_alignment(self):
        """
        Compute alignment between behavior and imaging data reward echoes.

        Performs linear regression and computes coefficients (slope and
        intercept) that can be used to correct either imaging or behavior data.

        Returns
        -------
        None
            Creates self.coeffs attribute with alignment coefficients
            (slope and intercept) and self.reg_results with regression statistics.
        """
        # assert len(self.rew_echo_beh) == len(self.rew_echo_ephys), \
        #     'Size of behavior and ephys reward echos must match.'

        if len(self.rew_echo_beh) > len(self.rew_echo_img):
            print('\ttruncating rew_echo_beh to match rew_echo_img...')
            self.rew_echo_beh = self.rew_echo_beh[0:len(self.rew_echo_img)]
        if len(self.rew_echo_img) > len(self.rew_echo_beh):
            print('\ttruncating rew_echo_img to match rew_echo_beh...')
            self.rew_echo_img = self.rew_echo_img[0:len(self.rew_echo_beh)]

        linreg_corr_img = sp.stats.linregress(
            self.rew_echo_img, self.rew_echo_beh)

        linreg_corr_beh = sp.stats.linregress(
            self.rew_echo_beh, self.rew_echo_img)

        self.regress = {'corr_img': SimpleNamespace(),
                        'corr_beh': SimpleNamespace()}

        self.regress['corr_img'].m = linreg_corr_img.slope
        self.regress['corr_img'].b = linreg_corr_img.intercept

        self.regress['corr_beh'].m = linreg_corr_beh.slope
        self.regress['corr_beh'].b = linreg_corr_beh.intercept

        return

    def correct_img_data(self, t_img):
        """
        Given some t_img (a vector of image frame times),
        corrects these to the behavior reference using the aligner.

        Parameters
        ---------------
        img_data : np.array
            Img data (spktimes) for a given set of
        """
        t_img_corr = self.regress['corr_img'].m*t_img \
            + self.regress['corr_img'].b
        return t_img_corr
