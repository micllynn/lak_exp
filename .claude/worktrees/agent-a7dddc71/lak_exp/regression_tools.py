import numpy as np
import scipy.signal as sp_sig

import statsmodels as sm
from statsmodels.nonparametric.kernel_regression import KernelReg as kr

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns

from types import SimpleNamespace

# test functions (continuous)
# ---------------


def calc_y_sqrt(x, noise_scale=0.1):
    """
    Calculate the square root of x with added Gaussian noise.

    Parameters
    ----------
    x : np.ndarray
        Input array to apply square root transformation.
    noise_scale : float, optional
        Standard deviation of Gaussian noise to add, by default 0.1.

    Returns
    -------
    y : np.ndarray
        Square root of x with added noise.
    """
    y_clean = x ** 0.5
    y = y_clean + np.random.normal(scale=noise_scale,
                                   size=x.shape[0])
    return y


def calc_y_sin(x, noise_scale=0.1):
    """
    Calculate the sine of x with added Gaussian noise.

    Parameters
    ----------
    x : np.ndarray
        Input array to apply sine transformation.
    noise_scale : float, optional
        Standard deviation of Gaussian noise to add, by default 0.1.

    Returns
    -------
    y : np.ndarray
        Sine of x with added noise.
    """
    y_clean = np.sin(x)
    y = y_clean + np.random.normal(scale=noise_scale,
                                   size=x.shape[0])
    return y


def make_data(y_noise=0, x_noise=0.2):
    """
    Generate synthetic test data with square root and sine transformations.

    Creates a SimpleNamespace object containing noisy x values and corresponding
    y values computed using square root and sine transformations.

    Parameters
    ----------
    y_noise : float, optional
        Standard deviation of noise to add to y values, by default 0.
    x_noise : float, optional
        Standard deviation of noise to add to x values, by default 0.2.

    Returns
    -------
    sig : SimpleNamespace
        Object with attributes:
        - x : noisy x values (100 points from 0 to 10)
        - y_sqrt : square root transformation of x with noise
        - y_sqrt_and_sin : sum of square root and sine transformations with noise
    """
    sig = SimpleNamespace()

    x_unispace = np.linspace(0, 10, 100)
    sig.x = x_unispace + np.random.normal(scale=x_noise,
                                          size=x_unispace.shape[0])

    sig.y_sqrt = calc_y_sqrt(sig.x, noise_scale=y_noise)
    sig.y_sqrt_and_sin = calc_y_sqrt(sig.x, noise_scale=y_noise) \
        + calc_y_sin(sig.x, noise_scale=y_noise)

    return sig


def kernel_reg_sample(sig):
    """
    Perform kernel regression on sample data.

    Fits a kernel regression model using the sqrt-transformed data and applies
    it to predict the combined sqrt+sin transformed data.

    Parameters
    ----------
    sig : SimpleNamespace
        Signal object containing attributes:
        - x : independent variable
        - y_sqrt : dependent variable for training
        - y_sqrt_and_sin : dependent variable for prediction

    Returns
    -------
    fit_out_new : SimpleNamespace
        Object with attributes:
        - pred_y : predicted y values
        - marginal_effects : marginal effects from the kernel regression
    """
    kmodel = kr(endog=sig.y_sqrt, exog=sig.x, var_type='c',
                bw='cv_ls', reg_type='ll')

    fit_out = SimpleNamespace()
    fit_out_new = SimpleNamespace()

    fit_out.pred_y, fit_out.marginal_effects = kmodel.fit()
    fit_out_new.pred_y, fit_out_new.marginal_effects = kmodel.fit(
        sig.y_sqrt_and_sin)

    return fit_out_new

# test functions (discrete)
# -------------------

def convolve_kern_matrix(t, ):
    """
    Convolve kernel matrix.

    Parameters
    ----------
    t : TODO
        TODO: Describe parameter.

    Returns
    -------
    TODO
        TODO: Describe return value.

    Notes
    -----
    This function is not yet implemented.
    """


class KernGen(object):
    """
    Generate kernel-convolved point process data for testing.

    Creates synthetic point process data by generating inhomogeneous Poisson
    processes based on square root and sine rate functions, convolving with
    Gaussian kernels, and adding noise.

    Parameters
    ----------
    rate : float, optional
        Base rate parameter (currently unused in implementation), by default 5.
    t_end : float, optional
        End time for signal generation, by default 10.
    t_int : float, optional
        Time interval for signal sampling, by default 0.01.
    rate_scaling_factor : float, optional
        Factor to scale down the rate functions, by default 10.
    kern_std_ind : float, optional
        Standard deviation for Gaussian kernel (in index units), by default 4.
    noise_std : float, optional
        Standard deviation of Gaussian noise to add, by default 0.1.

    Attributes
    ----------
    t : np.ndarray
        Time vector.
    y1_rate : np.ndarray
        Normalized square root rate function.
    y2_rate : np.ndarray
        Normalized sine rate function.
    y1_pointproc : np.ndarray
        Binary point process generated from y1_rate.
    y2_pointproc : np.ndarray
        Binary point process generated from y2_rate.
    kern : np.ndarray
        Gaussian kernel used for convolution.
    y1 : np.ndarray
        Kernel-convolved y1 point process with added noise.
    y2 : np.ndarray
        Kernel-convolved y2 point process with added noise.
    y1_y2_sum : np.ndarray
        Sum of y1 and y2 signals.
    """
    def __init__(self, rate=5, t_end=10, t_int=0.01,
                 rate_scaling_factor=10,
                 kern_std_ind=4, noise_std=0.1):
        # generate target rate signals for inhom poisson
        self.t = np.arange(0, 10, t_int)
        self.y1_rate = self.t ** 0.5
        self.y2_rate = np.sin(self.t)

        # normalize
        self.y1_rate /= (np.max(self.y1_rate)*rate_scaling_factor)
        self.y2_rate /= (np.max(self.y2_rate)*rate_scaling_factor)

        # generate pointprocs
        self.y1_pointproc = self.y1_rate > np.random.rand(self.t.shape[0])
        self.y2_pointproc = self.y2_rate > np.random.rand(self.t.shape[0])

        self.y1_pointproc = self.y1_pointproc.astype(int)
        self.y2_pointproc = self.y2_pointproc.astype(int)

        # convolve with kernel
        self.kern = sp_sig.windows.gaussian(self.t.shape[0],
                                            std=kern_std_ind)
        self.y1 = sp_sig.convolve(self.y1_pointproc,
                                  self.kern, mode='same')
        self.y2 = sp_sig.convolve(self.y2_pointproc,
                                  self.kern, mode='same')

        # add some noise
        self.y1 += np.random.normal(size=self.y1.shape[0], scale=noise_std)
        self.y2 += np.random.normal(size=self.y2.shape[0], scale=noise_std)
        self.y1_y2_sum = self.y1 + self.y2

        return

    def plt(self, markersize=5):
        """
        Plot the generated signals and point processes.

        Creates a two-panel plot showing the kernel-convolved signals (top)
        and the underlying point processes (bottom).

        Parameters
        ----------
        markersize : float, optional
            Size of markers for point process visualization, by default 5.

        Returns
        -------
        None
        """
        fig = plt.figure()
        spec = gs.GridSpec(nrows=2, ncols=1, height_ratios=[1, 0.2])
        ax_sig = fig.add_subplot(spec[0, 0])
        ax_pp = fig.add_subplot(spec[1, 0])

        ax_sig.plot(self.t, self.y1,
                    color=sns.xkcd_rgb['coral'])
        ax_sig.plot(self.t, self.y2,
                    color=sns.xkcd_rgb['steel blue'])

        ax_pp.scatter(self.t[self.y1_pointproc == 1],
                      np.ones_like(self.t[self.y1_pointproc == 1]),
                      color=sns.xkcd_rgb['coral'], s=markersize)
        ax_pp.scatter(self.t[self.y2_pointproc == 1],
                      np.ones_like(self.t[self.y2_pointproc == 1]),
                      color=sns.xkcd_rgb['steel blue'], s=markersize)

        plt.show()

        return


class KernelRegPPTest(object):
    """
    Test class for kernel regression on point process data.

    Creates a test instance with generated kernel point process data.

    Attributes
    ----------
    data_test : KernGen
        Generated test data from KernGen class.
    """
    def __init__(self):
        self.data_test = KernGen()


# real functions for kernel regression on movement and npix data
# ----------------------

class KernelRegMovement(object):
    """
    Kernel regression for modeling movement data based on neural activity.

    Fits a kernel regression model to predict movement from Neuropixels data.

    Parameters
    ----------
    movement_train : array-like
        Training movement data.
    npix_data_train : array-like
        Training Neuropixels spike data.
    kernreg_opts : dict, optional
        Kernel regression options passed to KernelReg, by default
        {'var_type': 'c', 'bw': 'cv_ls', 'reg_type': 'll'}.

    Attributes
    ----------
    kmodel : KernelReg
        Fitted kernel regression model.
    """
    def __init__(self,
                 movement_train,
                 npix_data_train,
                 kernreg_opts={'var_type': 'c',
                               'bw': 'cv_ls',
                               'reg_type': 'll'}):
        self.kmodel = kr(
            endog=preprocess_movement(movement_train),
            exog=preprocess_npix(npix_data_train),
            **kernreg_opts)

    def apply_model(self, movement, npix_data):
        """
        Apply the trained kernel regression model to new data.

        Parameters
        ----------
        movement : array-like
            Movement data to predict.
        npix_data : array-like
            Neuropixels spike data (currently unused due to implementation).

        Returns
        -------
        npix_data : SimpleNamespace
            Object with attributes:
            - pred : predicted values
            - resid : residuals between actual and predicted values

        Notes
        -----
        There appears to be a bug: the input npix_data is overwritten and not used.
        """
        npix_data = SimpleNamespace()
        npix_data.pred, _marg_eff = self.kmodel.fit(movement)
        npix_data.resid = self.calc_resid(npix_data, npix_data.pred)
        return npix_data

    def calc_resid(self, y_real, y_pred):
        """
        Calculate residuals between real and predicted values.

        Parameters
        ----------
        y_real : array-like
            Real/observed values.
        y_pred : array-like
            Predicted values.

        Returns
        -------
        resid : array-like
            Residuals (y_real - y_pred).
        """
        return y_real - y_pred


def preprocess_npix(spk_matrix):
    """
    Preprocess Neuropixels spike matrix data.

    Parameters
    ----------
    spk_matrix : array-like
        Raw spike matrix from Neuropixels recording.

    Returns
    -------
    spk_matrix_preprocessed : array-like
        Preprocessed spike matrix.

    Notes
    -----
    This function is not yet implemented.
    """
    return spk_matrix_preprocessed


def preprocess_movement(mvmt_times):
    """
    Preprocess movement time series data.

    Parameters
    ----------
    mvmt_times : array-like
        Raw movement timing data.

    Returns
    -------
    mvmt_times_preprocessed : array-like
        Preprocessed movement data.

    Notes
    -----
    This function is not yet implemented.
    """
    return mvmt_times_preprocessed
