import numpy as np
import scipy as sp
import pandas as pd

from types import SimpleNamespace
import os

from .utils import find_event_onsets
from .beh import TimelineParser


class Aligner_EphysBeh(object):
    def __init__(self):
        """
        Create an object to align the times of ephys and behavior
        data from an experiment.

        Contains methods to extract ephys and behavior reward echoes,
        compute an alignment with simple linear regression, and
        apply this alignment to ephys data.
        """
        pass

    def parse_ephys_rewechoes(self, folder='ephys_g0'):
        """
        Get and parse reward echoes for ephys data.

        ephys reward echoes have the format *XA_0_0.txt
        and are column-separated.

        Rewriting of getRewardEcho.m

        Parameters
        ----------
        folder : str, optional
            Folder where the file is located
            (file should be named *XA_0_0.txt), by default 'ephys_g0'.

        Returns
        -------
        None
            Creates self.ephys.rew_echo attribute with reward echo times.
        """

        files = os.listdir(folder)

        fname = None
        for f in files:
            if f.lower().endswith('xa_0_0.txt'):
                fname = f

        if fname == None:
            raise UnboundLocalError('error in aligning ephys and beh: no rew echo'
                  + ' file for ephys found (xa_0_0.txt)')

        df = pd.read_csv(folder+'/'+fname, header=None)
        self.rew_echo_ephys = np.array(df[0])

        return

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
        Compute alignment between behavior and ephys data reward echoes.

        Performs linear regression and computes coefficients (slope and
        intercept) that can be used to correct either ephys or behavior data.

        Returns
        -------
        None
            Creates self.coeffs attribute with alignment coefficients
            (slope and intercept) and self.reg_results with regression statistics.
        """
        # assert len(self.rew_echo_beh) == len(self.rew_echo_ephys), \
        #     'Size of behavior and ephys reward echos must match.'

        if len(self.rew_echo_beh) > len(self.rew_echo_ephys):
            print('\ttruncating rew_echo_beh to match rew_echo_ephys...')
            self.rew_echo_beh = self.rew_echo_beh[0:len(self.rew_echo_ephys)]
        if len(self.rew_echo_ephys) > len(self.rew_echo_beh):
            print('\ttruncating rew_echo_ephys to match rew_echo_beh...')
            self.rew_echo_ephys = self.rew_echo_ephys[0:len(self.rew_echo_beh)]

        linreg_corr_ephys = sp.stats.linregress(
            self.rew_echo_ephys, self.rew_echo_beh)

        linreg_corr_beh = sp.stats.linregress(
            self.rew_echo_beh, self.rew_echo_ephys)

        self.regress = {'corr_ephys': SimpleNamespace(),
                        'corr_beh': SimpleNamespace()}

        self.regress['corr_ephys'].m = linreg_corr_ephys.slope
        self.regress['corr_ephys'].b = linreg_corr_ephys.intercept

        self.regress['corr_beh'].m = linreg_corr_beh.slope
        self.regress['corr_beh'].b = linreg_corr_beh.intercept

        return

    def correct_ephys_data(self, ephys_data):
        """
        Given some ephys data (spktimes) corrects these to the behavior
        reference using the aligner.

        Parameters
        ---------------
        ephys_data : np.array
            Ephys data (spktimes) for a given set of
        """
        ephys_data_corrected = self.regress['corr_ephys'].m*ephys_data \
            + self.regress['corr_ephys'].b
        return ephys_data_corrected

    def correct_beh_data(self, beh_data):
        """
        Given some behavior data (eventtimes) corrects these to the ephys
        reference using the aligner.

        Parameters
        ---------------
        ephys_data : np.array
            Ephys data (spktimes) for a given set of
        """
        beh_data_corrected = self.regress['corr_beh'].m*beh_data \
            + self.regress['corr_beh'].b
        return beh_data_corrected
