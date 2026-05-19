import numpy as np
import scipy as sp
import scipy.stats as sp_stats
from sklearn import decomposition
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
import pandas as pd

from types import SimpleNamespace
import os
import copy
import gc

from .load_exp_twop import TwoPRec, TwoPRec_DualColour
from .utils import rescale_to_frac
from .dset_twop import DSetObj_5HTCtx

try:
    plt.style.use('publication_ml')
except:
    pass


def plot_antic_licking(dset_obj, figsize=(2, 2),
                       marker_size=60,
                       plt_xlim=[-0.5, 2.5]):
    n_recs = dset_obj._dset_raw['expref'].shape[0]
    antic_licks = {'0': [], '0.5': [], '1': []}

    colors = sns.cubehelix_palette(
         n_colors=3,
         start=2, rot=0,
         dark=0.2, light=0.8)

    # store anticipatory lickrate for each rec
    for rec in range(n_recs):
        print(f'---------\n{rec=}\n----------')
        try:
            exp = TwoPRec(dset_obj=dset_obj, dset_ind=rec)
            exp.add_lickrates()
            _antic_licks = exp.get_antic_licks_by_trialtype()

            antic_licks['0'].append(_antic_licks['0'])
            antic_licks['0.5'].append(_antic_licks['0.5'])
            antic_licks['1'].append(_antic_licks['1'])
        except:
            pass

    # make plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot([antic_licks['0'],
             antic_licks['0.5'],
             antic_licks['1']],
            color=sns.xkcd_rgb['light grey'])

    ax.scatter(np.ones_like(antic_licks['0'])*0,
               antic_licks['0'], s=marker_size,
               facecolors='none',
               edgecolors=colors[0])
    ax.scatter(np.ones_like(antic_licks['0.5'])*1,
               antic_licks['0.5'], s=marker_size,
               facecolors='none',
               edgecolors=colors[1])
    ax.scatter(np.ones_like(antic_licks['1'])*2,
               antic_licks['1'], s=marker_size,
               facecolors='none',
               edgecolors=colors[2])

    ax.set_xticks([0, 1, 2], ['0%', '50%', '100%'])
    ax.set_ylabel('antic. licking (Hz)')
    ax.set_xlabel('rew. prob.')
    ax.set_xlim(plt_xlim)

    fig.savefig('summary_licks.pdf')

    _res_integ = sp.stats.ttest_rel(
        antic_licks['0'], antic_licks['1'])
    print(f'integ pval={_res_integ.pvalue}')
    print(f'integ stat={_res_integ.statistic}')

    return


# def plt_grab_dset_simple_summary(dset_obj,
#                                  figsize=(6, 3)):
#     fig = plt.figure(figsize=figsize)
#     spec = gs.GridSpec(nrows=1, ncols=4)
#     ax_var = fig.add_subplot(spec[0, 0])

#     return


def plt_fig1_grab_summary(dset_obj,
                          dset_ind_ex=0,
                          ch_img=1,
                          summary_fn=sp.integrate.trapezoid,
                          coding_type='optimism',
                          ind_rec_end=None,
                          rec_qc_rewresp=0.05,
                          optimism_outlier_thresh=10,
                          range_coding_spatial=[-2, 2],
                          n_sectors=10,
                          t_pre=5,
                          t_post=4,
                          colormap=sns.cubehelix_palette(
                              n_colors=3,
                              start=2, rot=0,
                              dark=0.1, light=0.6),
                          marker_size=60,
                          ylim_neur_stim=None,
                          ylim_neur_rew=None,
                          return_data_upon_fail=False):
    """
    Plots all stimulus-evoked GRAB5-HT responses across the whole
    dataset.
    """
    # load and store
    # -----------
    if ind_rec_end is None:
        n_recs = dset_obj._dset_raw['expref'].shape[0]
    else:
        n_recs = ind_rec_end

    tr_conds = ['0', '0.5', '0.5_rew', '0.5_norew', '1']
    plt_params = SimpleNamespace()
    plt_params.colors = {'0': colormap[0],
                         '0.5': colormap[1],
                         '0.5_rew': colormap[1],
                         '0.5_norew': colormap[1],
                         '1': colormap[2]}
    plt_params.linestyles = {'0': 'solid',
                             '0.5': 'solid',
                             '0.5_rew': 'solid',
                             '0.5_norew': 'dashed',
                             '1': 'solid'}

    # load each dset and store summary stats
    _dset_inds_to_delete = []
    for dset_ind in range(n_recs):
        try:
            _exp = TwoPRec(dset_obj=dset_obj,
                           dset_ind=dset_ind,
                           ch_img=ch_img,
                           trial_end=dset_obj._dset_raw['trial_end'][dset_ind])
            _exp.add_frame(t_pre=t_pre, t_post=t_post)
            _exp.add_sectors(n_sectors=n_sectors,
                             t_pre=t_pre, t_post=t_post)
            _exp.add_neurs(t_pre=t_pre, t_post=t_post)
            _exp.add_zetatest_neurs()
            _exp.add_psychometrics(t_pre=t_pre, t_post=t_post)

            _ind_t_stim = np.argmin(np.abs(_exp.frame.t - 0))
            _ind_t_rew = np.argmin(np.abs(_exp.frame.t - 2))
            _ind_t_end = np.argmin(np.abs(_exp.frame.t - (2+t_post)))

            # On first recording, setup data structs
            if dset_ind == 0:
                len_dff = _exp.frame.dff['0'].shape[1]

                data = SimpleNamespace()
                data.dff_stim = {}
                data.dff_neur_stim = {}
                data.neur_zetapval = []
                data.stats = SimpleNamespace(
                    stim_resp={},
                    rew_resp={},
                    neur_stim_resp={},
                    neur_rew_resp={},
                    stim_resp_norm={},
                    rew_resp_norm={},
                    between_sector_variance_stim={},
                    within_sector_variance_stim={},
                    lick_stim_resp={},
                    lick_rew_resp={})
                data.stats.psychometrics = SimpleNamespace()

                for tr_cond in tr_conds:
                    data.dff_stim[tr_cond] = np.zeros((n_recs, len_dff))
                    data.dff_neur_stim[tr_cond] = []
                    data.stats.stim_resp[tr_cond] = np.zeros(n_recs)
                    data.stats.rew_resp[tr_cond] = np.zeros(n_recs)
                    data.stats.neur_stim_resp[tr_cond] = []
                    data.stats.neur_rew_resp[tr_cond] = []

                    data.stats.stim_resp_norm[tr_cond] = np.zeros(n_recs)
                    data.stats.rew_resp_norm[tr_cond] = np.zeros(n_recs)

                    data.stats.lick_stim_resp[tr_cond] = np.zeros(n_recs)
                    data.stats.lick_rew_resp[tr_cond] = np.zeros(n_recs)

                    data.stats.between_sector_variance_stim[tr_cond] \
                        = np.zeros(n_recs)
                    data.stats.within_sector_variance_stim[tr_cond] \
                        = np.zeros(n_recs)

            # if example, store all sector info
            if dset_ind == dset_ind_ex:
                data.exp_ex = _exp

            # store grab data
            print('\tstoring stim responses...')
            data.dff_stim_t = _exp.frame.t
            for tr_cond in tr_conds:
                data.dff_stim[tr_cond][dset_ind, :] = np.mean(
                    _exp.frame.dff[tr_cond], axis=0)
                data.stats.stim_resp[tr_cond][dset_ind] = np.mean(summary_fn(
                    _exp.frame.dff[tr_cond][:, _ind_t_stim:_ind_t_rew],
                    dx=1/_exp.samp_rate,
                    axis=1), axis=0)
                data.stats.rew_resp[tr_cond][dset_ind] = np.mean(summary_fn(
                    _exp.frame.dff[tr_cond][:, _ind_t_rew:_ind_t_end],
                    dx=1/_exp.samp_rate,
                    axis=1), axis=0)

                data.stats.lick_stim_resp[tr_cond][dset_ind] = np.mean(
                    _exp.beh.lick.antic[tr_cond])

            # store neural responses
            print('\tstoring neural responses...')
            data.neur_zetapval.append(_exp.neur.zeta_pval)
            for tr_cond in tr_conds:
                if _exp.neur.dff_aln_mean[tr_cond].shape[0] > 0:
                    # Append neural df/f traces
                    data.dff_neur_stim[tr_cond].append(
                        _exp.neur.dff_aln_mean[tr_cond])

                    # Compute and append stimulus response for each neuron
                    _neur_stim_resp = summary_fn(
                        _exp.neur.dff_aln_mean[tr_cond][:, _ind_t_stim:_ind_t_rew],
                        dx=1/_exp.samp_rate,
                        axis=1)
                    data.stats.neur_stim_resp[tr_cond].append(_neur_stim_resp)

                    # Compute and append reward response for each neuron
                    _neur_rew_resp = summary_fn(
                        _exp.neur.dff_aln_mean[tr_cond][:, _ind_t_rew:_ind_t_end],
                        dx=1/_exp.samp_rate,
                        axis=1)
                    data.stats.neur_rew_resp[tr_cond].append(_neur_rew_resp)

            # store normalized versions of all
            _dff_resp_max = np.max(data.dff_stim[tr_cond][dset_ind, :])
            _stim_resp_max = data.stats.stim_resp['1'][dset_ind]
            _rew_resp_max = data.stats.rew_resp['1'][dset_ind]
            for tr_cond in tr_conds:
                data.stats.stim_resp_norm[tr_cond][dset_ind] \
                    = data.stats.stim_resp[tr_cond][dset_ind] / _stim_resp_max
                data.stats.rew_resp_norm[tr_cond][dset_ind] \
                    = data.stats.rew_resp[tr_cond][dset_ind] / _rew_resp_max

            # psychometric curves
            # ------------------
            print('\tstoring psychometric curves...')
            if dset_ind == 0:
                data.stats.psychometrics.grab = SimpleNamespace(
                    optimism=np.empty(n_recs, dtype=np.ndarray))
                data.stats.psychometrics.grab_frame = SimpleNamespace(
                    optimism=np.empty(n_recs, dtype=np.ndarray))
                data.stats.psychometrics.lick = SimpleNamespace(
                    optimism=np.zeros(n_recs))

            for attr in ['grab', 'grab_frame', 'lick']:
                getattr(data.stats.psychometrics, attr).optimism[dset_ind] \
                    = getattr(_exp.psychometrics, attr).optimism

            # quality control
            # ------------
            if rec_qc_rewresp is not None:
                print('quality control on reward response...')
                _rew_resp_max = data.stats.rew_resp['1'][dset_ind]
                if _rew_resp_max < rec_qc_rewresp:
                    _dset_inds_to_delete.append(dset_ind)
                    print('\trew response is '
                          + f"{_rew_resp_max:.2f} df/f "
                          + f'< {rec_qc_rewresp}')
                    print('\tmarking data for deletion...')
                else:
                    print('\trew response is '
                          + f"{_rew_resp_max:.2f} df/f "
                          + f'> {rec_qc_rewresp}, ok...')

            print('\n')
        except Exception as error:
            print('!! could not load dataset!! ')
            print(f'error: {error}')
            _dset_inds_to_delete.append(dset_ind)
            if return_data_upon_fail is True:
                return data

    # delete recs that didn't load or didn't pass qc
    print(f'\ndeleting recs: {_dset_inds_to_delete}\n----------')
    if len(_dset_inds_to_delete) > 0:
        for tr_cond in tr_conds:
            data.dff_stim[tr_cond] = np.delete(
                data.dff_stim[tr_cond],
                _dset_inds_to_delete, axis=0)
            data.stats.stim_resp[tr_cond] = np.delete(
                data.stats.stim_resp[tr_cond],
                _dset_inds_to_delete)
            data.stats.stim_resp_norm[tr_cond] = np.delete(
                data.stats.stim_resp_norm[tr_cond],
                _dset_inds_to_delete)
            data.stats.rew_resp[tr_cond] = np.delete(
                data.stats.rew_resp[tr_cond],
                _dset_inds_to_delete)
            data.stats.lick_stim_resp[tr_cond] = np.delete(
                data.stats.lick_stim_resp[tr_cond],
                _dset_inds_to_delete)

        data.stats.psychometrics.grab.optimism = np.delete(
            data.stats.psychometrics.grab.optimism,
            _dset_inds_to_delete)
        data.stats.psychometrics.grab_frame.optimism = np.delete(
            data.stats.psychometrics.grab_frame.optimism,
            _dset_inds_to_delete)
        data.stats.psychometrics.lick.optimism = np.delete(
            data.stats.psychometrics.lick.optimism,
            _dset_inds_to_delete)

    # concatenate neural data lists into arrays for plotting
    print('\nconcatenating neural data...')
    for tr_cond in tr_conds:
        if len(data.dff_neur_stim[tr_cond]) > 0:
            data.dff_neur_stim[tr_cond] = np.concatenate(
                data.dff_neur_stim[tr_cond], axis=0)
            data.stats.neur_stim_resp[tr_cond] = np.concatenate(
                data.stats.neur_stim_resp[tr_cond], axis=0)
            data.stats.neur_rew_resp[tr_cond] = np.concatenate(
                data.stats.neur_rew_resp[tr_cond], axis=0)
        else:
            # If no neurons, create empty arrays
            data.dff_neur_stim[tr_cond] = np.array([])
            data.stats.neur_stim_resp[tr_cond] = np.array([])
            data.stats.neur_rew_resp[tr_cond] = np.array([])

    # ~~~~~~~~ setup plots ~~~~~~~~~~~~~
    # -----------
    try:
        print('\nplotting...\n-------------')
        fig = plt.figure(figsize=(6.86, 5))
        spec = gs.GridSpec(nrows=3, ncols=6,
                           figure=fig)
        ax_dff = fig.add_subplot(spec[0, 0:2])
        ax_stats_stim = fig.add_subplot(spec[0, 2])
        ax_stats_rew = fig.add_subplot(spec[0, 3])
        ax_stats_licks = fig.add_subplot(spec[0, 4])
        ax_stats_lick_vs_stim = fig.add_subplot(spec[0, 5])

        ax_optimism_ex = fig.add_subplot(spec[1, 0:2])
        ax_optimism_ex_img = fig.add_subplot(spec[1, 2:4])
        ax_optimism_pop = fig.add_subplot(spec[1, 4])
        ax_optimism_pop_stdev = fig.add_subplot(spec[1, 5])

        ax_dff_neur = fig.add_subplot(spec[2, 0:2])
        ax_stats_neur_stim = fig.add_subplot(spec[2, 2])
        ax_stats_neur_rew = fig.add_subplot(spec[2, 3])

        # ax_optimism = fig.add_subplot(spec[2, 0:2])
        # ax_optimism_stdev = fig.add_subplot(spec[2, 2])

        # plot average df/f trace
        # -----------
        print('\taverage df/f traces...')
        for tr_cond in ['0', '0.5_rew', '0.5_norew', '1']:
            _dff_stim_mean = np.mean(data.dff_stim[tr_cond], axis=0)
            _dff_stim_sem = sp.stats.sem(data.dff_stim[tr_cond], axis=0)
            ax_dff.plot(data.dff_stim_t, _dff_stim_mean,
                        color=plt_params.colors[tr_cond],
                        linestyle=plt_params.linestyles[tr_cond])
            ax_dff.fill_between(data.dff_stim_t,
                                _dff_stim_mean - _dff_stim_sem,
                                _dff_stim_mean + _dff_stim_sem,
                                facecolor=plt_params.colors[tr_cond],
                                alpha=0.2)
        ax_dff.axvline(x=0, color=sns.xkcd_rgb['dark grey'],
                       linewidth=1.5, alpha=0.8,
                       linestyle='dashed')
        ax_dff.axvline(x=2, color=sns.xkcd_rgb['bright blue'],
                       linewidth=1.5, alpha=0.8,
                       linestyle='dashed')
        ax_dff.set_ylabel(r'$GRAB_\mathrm{5-HT}$ ($\frac {\mathrm{d}F}{F}$)')
        ax_dff.set_xlabel('time from stim (s)')

        # plot stats: stimulus-locked dff
        # ------------
        print('\tstimulus-locked df/f...')
        for dset_ind in range(len(data.stats.stim_resp['0'])):
            ax_stats_stim.plot(
                [0, 1, 2],
                [data.stats.stim_resp['0'][dset_ind],
                 data.stats.stim_resp['0.5'][dset_ind],
                 data.stats.stim_resp['1'][dset_ind]],
                color=sns.xkcd_rgb['light grey'],
                linewidth=0.8)

        for tr_ind, tr_cond in enumerate(['0', '0.5', '1']):
            ax_stats_stim.scatter(
                np.ones_like(data.stats.stim_resp[tr_cond])*tr_ind,
                data.stats.stim_resp[tr_cond],
                s=marker_size, facecolors='none',
                edgecolors=plt_params.colors[tr_cond])

        # Add means with filled circles and connecting lines
        _stim_means = [np.mean(data.stats.stim_resp['0']),
                       np.mean(data.stats.stim_resp['0.5']),
                       np.mean(data.stats.stim_resp['1'])]
        ax_stats_stim.plot([0, 1, 2], _stim_means,
                           color='black', linewidth=1.0, alpha=0.8, zorder=10)
        for tr_ind, (tr_cond, mean_val) in enumerate(zip(
                ['0', '0.5', '1'], _stim_means)):
            ax_stats_stim.scatter(tr_ind, mean_val, s=marker_size,
                                  facecolors=plt_params.colors[tr_cond],
                                  edgecolors=plt_params.colors[tr_cond],
                                  alpha=0.8, zorder=11)

        for _ax in [ax_stats_stim]:
            _ax.set_xticks([0, 1, 2])
            _ax.set_xticklabels(['0', '0.5', '1'])
            _ax.set_xlabel('p(rew)')

        # plot stats: rew-locked dff
        # ------------
        print('\treward-locked df/f...')
        for dset_ind in range(len(data.stats.stim_resp['0'])):
            ax_stats_rew.plot(
                [0, 1, 2, 3],
                [data.stats.rew_resp['0'][dset_ind],
                 data.stats.rew_resp['0.5_norew'][dset_ind],
                 data.stats.rew_resp['0.5_rew'][dset_ind],
                 data.stats.rew_resp['1'][dset_ind]],
                color=sns.xkcd_rgb['light grey'],
                linewidth=0.8)

        for tr_ind, tr_cond in enumerate(['0', '0.5_norew', '0.5_rew', '1']):
            ax_stats_rew.scatter(
                np.ones_like(data.stats.rew_resp[tr_cond])*tr_ind,
                data.stats.rew_resp[tr_cond],
                s=marker_size, facecolors='none',
                edgecolors=plt_params.colors[tr_cond])

        # Add means with filled circles and connecting lines
        _rew_means = [np.mean(data.stats.rew_resp['0']),
                      np.mean(data.stats.rew_resp['0.5_norew']),
                      np.mean(data.stats.rew_resp['0.5_rew']),
                      np.mean(data.stats.rew_resp['1'])]
        ax_stats_rew.plot([0, 1, 2, 3], _rew_means,
                         color='black', linewidth=1.0, alpha=0.8, zorder=10)
        for tr_ind, (tr_cond, mean_val) in enumerate(zip(
                ['0', '0.5_norew', '0.5_rew', '1'], _rew_means)):
            ax_stats_rew.scatter(tr_ind, mean_val, s=marker_size,
                                 facecolors=plt_params.colors[tr_cond],
                                 edgecolors=plt_params.colors[tr_cond],
                                 alpha=0.8, zorder=11)

        for _ax in [ax_stats_rew]:
            _ax.set_xticks([0, 1, 2, 3])
            _ax.set_xticklabels(['0', '0.5_norew', '0.5_rew', '1'])
            _ax.set_xlabel('p(rew)')

        # setup y labels for all GRAB summary plots
        # ------------
        _ylabel_base = r'$GRAB_\mathrm{5-HT}$ '
        if summary_fn == sp.integrate.trapezoid:
            _ylabel_suffix = r'($\int \frac{\mathrm{d}F}{F} \mathrm{d}t$)'
        elif summary_fn == np.mean:
            _ylabel_suffix = r'($\frac{\mathrm{d}F}{F}$)'

        ax_stats_stim.set_ylabel(r'{}'.format(
            'stim-evoked ' + _ylabel_base + _ylabel_suffix))
        ax_stats_rew.set_ylabel(r'{}'.format(
            'rew-evoked ' + _ylabel_base + _ylabel_suffix))

        # plot licking
        # --------------
        print('\tlicking stats...')
        for dset_ind in range(len(data.stats.stim_resp['0'])):
            ax_stats_licks.plot(
                [0, 1, 2],
                [data.stats.lick_stim_resp['0'][dset_ind],
                 data.stats.lick_stim_resp['0.5'][dset_ind],
                 data.stats.lick_stim_resp['1'][dset_ind]],
                color=sns.xkcd_rgb['light grey'],
                linewidth=0.8)
            for tr_ind, tr_cond in enumerate(['0', '0.5', '1']):
                ax_stats_licks.scatter(
                    np.ones_like(data.stats.lick_stim_resp[tr_cond])*tr_ind,
                    data.stats.lick_stim_resp[tr_cond],
                    s=marker_size, facecolors='none',
                    edgecolors=plt_params.colors[tr_cond])
        ax_stats_licks.set_xticks([0, 1, 2])
        ax_stats_licks.set_xticklabels(['0', '0.5', '1'])
        ax_stats_licks.set_xlabel('p(rew)')
        ax_stats_licks.set_ylabel('antic. licks (Hz)')

        # plot correlation between licking and GRAB
        # -------------
        _lick_resp_all = []
        _stim_resp_all = []
        for tr_cond in ['0', '0.5', '1']:
            ax_stats_lick_vs_stim.scatter(
                data.stats.lick_stim_resp[tr_cond], data.stats.stim_resp[tr_cond],
                s=marker_size, facecolors='none',
                edgecolors=plt_params.colors[tr_cond])
            _lick_resp_all.append(data.stats.lick_stim_resp[tr_cond])
            _stim_resp_all.append(data.stats.stim_resp[tr_cond])

        ax_stats_lick_vs_stim.set_xlabel('antic. licks (Hz)')
        ax_stats_lick_vs_stim.set_ylabel(r'{}'.format(
            'stim-evoked ' + _ylabel_base + _ylabel_suffix))

        # data.lick_stim_resp_linreg = sp.stats.linregress(
        #     _lick_resp_all, _stim_resp_all)

        # sector heterogeneity
        # ----------------
        print('sector heterogeneity....')
        optim_grab_ex = data.stats.psychometrics.grab.optimism[dset_ind_ex]
        optim_grab = data.stats.psychometrics.grab.optimism
        optim_lick = data.stats.psychometrics.lick.optimism[dset_ind_ex]

        # ---- plot sector heterogeneity example --------
        # cleanup optimism extreme vals
        # _outliers = np.where(np.abs(optim_grab_ex) > optimism_outlier_thresh)
        # optim_grab_ex = np.delete(optim_grab_ex, _outliers)

        ax_optimism_ex.scatter(
            np.arange(len(optim_grab_ex)),
            np.sort(optim_grab_ex),
            facecolors=sns.xkcd_rgb['moss green'])
        ax_optimism_ex.axhline(
            optim_lick,
            linestyle='dashed', color=sns.xkcd_rgb['moss green'],
            linewidth=0.5)

        if range_coding_spatial is None:
            v_abs_max = np.max(np.abs(optim_grab_ex))
            spatial_codinghet = ax_optimism_ex_img.imshow(
                optim_grab_ex.reshape(10, 10),
                vmax=v_abs_max, vmin=-1*v_abs_max,
                cmap='coolwarm')
        else:
            spatial_codinghet = ax_optimism_ex_img.imshow(
                optim_grab_ex.reshape(n_sectors, n_sectors),
                vmin=range_coding_spatial[0],
                vmax=range_coding_spatial[1],
                cmap='coolwarm')

        fig.colorbar(spatial_codinghet, ax=ax_optimism_ex_img,
                     location='right', shrink=0.5)

        # ------- plot sector heterogeneity (all) -------
        _grab_psych_pop_all = np.empty(0)
        for dset_ind in range(n_recs):
            try:
                _grab_psych_pop_all = np.append(
                    _grab_psych_pop_all,
                    optim_grab[dset_ind])
                ax_optimism_pop.scatter(
                    np.arange(len(_grab_psych_pop_all)),
                    np.sort(np.array(_grab_psych_pop_all)),
                    facecolors=sns.xkcd_rgb['moss green'])
            except Exception:
                pass

        for _ax in [ax_optimism_ex, ax_optimism_pop]:
            _ax.set_xlabel('spatial sectors (sorted)')
            if coding_type == 'linearity':
                _ax.axhline(0.5, linestyle='dashed',
                            linewidth=1, color='black')
                _ax.set_ylabel(
                    r'value linearity index'
                    + r' ($\frac {R_{0.5}-R_{0}} {R_{1}-R{0}}$)')
            elif coding_type == 'optimism':
                _ax.axhline(0.5, linestyle='dashed',
                            linewidth=1, color='black')
                _ax.set_ylabel(
                    r'optimism index'
                    + r' ($ \frac {\alpha_{0-0.5}} {(\alpha_{0-0.5} + \alpha_{0.5-1})}$)')

        _stdev_grab_psych_pop = []
        for dset_ind in range(n_recs):
            try:
                _stdev_grab_psych_pop.append(np.std(optim_grab[dset_ind]))
            except:
                pass

        ax_optimism_pop_stdev.scatter(
            np.ones_like(_stdev_grab_psych_pop),
            _stdev_grab_psych_pop, s=marker_size,
            facecolors='none',
            edgecolors=sns.xkcd_rgb['moss green'])
        ax_optimism_pop_stdev.set_ylabel('std. GRAB optimism')

        # plot neural data
        # ----------------
        print('\tneural df/f traces...')
        for tr_cond in ['0', '0.5_rew', '0.5_norew', '1']:
            if len(data.dff_neur_stim[tr_cond]) > 0:
                _dff_neur_mean = np.mean(data.dff_neur_stim[tr_cond], axis=0)
                _dff_neur_sem = sp.stats.sem(data.dff_neur_stim[tr_cond], axis=0)
                ax_dff_neur.plot(data.dff_stim_t, _dff_neur_mean,
                                 color=plt_params.colors[tr_cond],
                                 linestyle=plt_params.linestyles[tr_cond])
                ax_dff_neur.fill_between(data.dff_stim_t,
                                         _dff_neur_mean - _dff_neur_sem,
                                         _dff_neur_mean + _dff_neur_sem,
                                         facecolor=plt_params.colors[tr_cond],
                                         alpha=0.2)
        ax_dff_neur.axvline(x=0, color=sns.xkcd_rgb['dark grey'],
                            linewidth=1.5, alpha=0.8,
                            linestyle='dashed')
        ax_dff_neur.axvline(x=2, color=sns.xkcd_rgb['bright blue'],
                            linewidth=1.5, alpha=0.8,
                            linestyle='dashed')
        ax_dff_neur.set_ylabel(r'Neural ($\frac {\mathrm{d}F}{F}$)')
        ax_dff_neur.set_xlabel('time from stim (s)')

        # plot neural stats: stimulus-locked
        # -----------------------------------
        print('\tstimulus-locked neural responses...')
        # First draw thin lines connecting individual neurons across conditions
        if len(data.stats.neur_stim_resp['0']) > 0:
            n_neurons = len(data.stats.neur_stim_resp['0'])
            for neur_ind in range(n_neurons):
                ax_stats_neur_stim.plot(
                    [0, 1, 2],
                    [data.stats.neur_stim_resp['0'][neur_ind],
                     data.stats.neur_stim_resp['0.5'][neur_ind],
                     data.stats.neur_stim_resp['1'][neur_ind]],
                    color=sns.xkcd_rgb['light grey'],
                    linewidth=0.3, alpha=0.3)

        # Then draw individual neuron scatter points
        for tr_ind, tr_cond in enumerate(['0', '0.5', '1']):
            if len(data.stats.neur_stim_resp[tr_cond]) > 0:
                ax_stats_neur_stim.scatter(
                    np.ones_like(data.stats.neur_stim_resp[tr_cond])*tr_ind,
                    data.stats.neur_stim_resp[tr_cond],
                    s=marker_size/2, facecolors='none',
                    edgecolors=plt_params.colors[tr_cond],
                    alpha=0.3)

        # Add means with filled circles and connecting lines
        # if len(data.stats.neur_stim_resp['0']) > 0:
        _neur_stim_means = [np.mean(data.stats.neur_stim_resp['0']),
                            np.mean(data.stats.neur_stim_resp['0.5']),
                            np.mean(data.stats.neur_stim_resp['1'])]
        ax_stats_neur_stim.plot([0, 1, 2], _neur_stim_means,
                                color='black', linewidth=1.0, alpha=0.8, zorder=10)
        for tr_ind, (tr_cond, mean_val) in enumerate(zip(
                ['0', '0.5', '1'], _neur_stim_means)):
            ax_stats_neur_stim.scatter(tr_ind, mean_val, s=marker_size,
                                       facecolors=plt_params.colors[tr_cond],
                                       edgecolors=plt_params.colors[tr_cond],
                                       alpha=0.8, zorder=11)

        ax_stats_neur_stim.set_xticks([0, 1, 2])
        ax_stats_neur_stim.set_xticklabels(['0', '0.5', '1'])
        ax_stats_neur_stim.set_xlabel('p(rew)')
        _ylabel_neur_base = r'Neural '
        ax_stats_neur_stim.set_ylabel(r'{}'.format(
            'stim-evoked ' + _ylabel_neur_base + _ylabel_suffix))
        if ylim_neur_stim is not None:
            ax_stats_neur_stim.set_ylim(ylim_neur_stim)

        # plot neural stats: reward-locked
        # ---------------------------------
        print('\treward-locked neural responses...')
        # First draw thin lines connecting individual neurons across conditions
        if len(data.stats.neur_rew_resp['0']) > 0:
            n_neurons = len(data.stats.neur_rew_resp['0'])
            for neur_ind in range(n_neurons):
                ax_stats_neur_rew.plot(
                    [0, 1, 2, 3],
                    [data.stats.neur_rew_resp['0'][neur_ind],
                     data.stats.neur_rew_resp['0.5_norew'][neur_ind],
                     data.stats.neur_rew_resp['0.5_rew'][neur_ind],
                     data.stats.neur_rew_resp['1'][neur_ind]],
                    color=sns.xkcd_rgb['light grey'],
                    linewidth=0.3, alpha=0.3)

        # Then draw individual neuron scatter points
        for tr_ind, tr_cond in enumerate(['0', '0.5_norew', '0.5_rew', '1']):
            if len(data.stats.neur_rew_resp[tr_cond]) > 0:
                ax_stats_neur_rew.scatter(
                    np.ones_like(data.stats.neur_rew_resp[tr_cond])*tr_ind,
                    data.stats.neur_rew_resp[tr_cond],
                    s=marker_size/2, facecolors='none',
                    edgecolors=plt_params.colors[tr_cond],
                    alpha=0.3)

        # Add means with filled circles and connecting lines
        # if len(data.stats.neur_rew_resp['0']) > 0:
        _neur_rew_means = [np.mean(data.stats.neur_rew_resp['0']),
                           np.mean(data.stats.neur_rew_resp['0.5_norew']),
                           np.mean(data.stats.neur_rew_resp['0.5_rew']),
                           np.mean(data.stats.neur_rew_resp['1'])]
        ax_stats_neur_rew.plot([0, 1, 2, 3], _neur_rew_means,
                               color='black', linewidth=1.0, alpha=0.8, zorder=10)
        for tr_ind, (tr_cond, mean_val) in enumerate(zip(
                ['0', '0.5_norew', '0.5_rew', '1'], _neur_rew_means)):
            ax_stats_neur_rew.scatter(tr_ind, mean_val, s=marker_size,
                                      facecolors=plt_params.colors[tr_cond],
                                      edgecolors=plt_params.colors[tr_cond],
                                      alpha=0.8, zorder=11)

        ax_stats_neur_rew.set_xticks([0, 1, 2, 3])
        ax_stats_neur_rew.set_xticklabels(['0', '0.5_norew', '0.5_rew', '1'])
        ax_stats_neur_rew.set_xlabel('p(rew)')
        ax_stats_neur_rew.set_ylabel(r'{}'.format(
            'rew-evoked ' + _ylabel_neur_base + _ylabel_suffix))
        if ylim_neur_rew is not None:
            ax_stats_neur_rew.set_ylim(ylim_neur_rew)

        # save and show
        # -------------
        fig.savefig('/Users/michaellynn/Desktop/postdoc/projects/'
                    + '5HTCtx/cohorts/visual_pavlov_mbl012-018/figs/'
                    + f'fig1_summaryfn={summary_fn.__name__}'
                    + f'_{ch_img=}'
                    + f'_{coding_type=}'
                    + '.pdf')
        plt.show()

    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        return data

    return data


def plt_all_optimism(dset_obj,
                     ind_rec_end=None):
    if ind_rec_end is None:
        n_recs = dset_obj._dset_raw['expref'].shape[0]
    else:
        n_recs = dset_obj._dset_raw['expref'].shape[0]

    for dset_ind in range(n_recs):
        try:
            _exp = TwoPRec(dset_obj=dset_obj,
                           dset_ind=dset_ind,
                           ch_img=1,
                           trial_end=dset_obj._dset_raw['trial_end'][dset_ind])
            _exp.plt_optimism(zetatest_mask=False, ylims=(-1, 1), plt_show=False)
            _exp.plt_optimism(zetatest_mask=False, ylims=(-2, 2), plt_show=False)

            _exp.plt_optimism(zetatest_mask=True, ylims=(-1, 1), plt_show=False)
            _exp.plt_optimism(zetatest_mask=True, ylims=(-2, 2), plt_show=False)
        except Exception as e:
            print(e)
            pass
    return


def plt_grab_responses_aligned(dset_obj, dset_ind_start=0,
                               t_pre=1, t_post=6,
                               trial_end=60):
    n_recs = dset_obj._dset_raw['expref'].shape[0]

    plot_conds_chs = SimpleNamespace()
    plot_conds_chs.ch = [1, 2]
    plot_conds_chs.plt_dff_y_gain = [10, 5]

    plot_conds_spec = SimpleNamespace()
    plot_conds_spec.plot_special = ['rew', 'rew_norew']
    for rec in range(dset_ind_start, n_recs):
        for ch in plot_conds_chs.ch:
            print(f'-------\n{ch=}')
            try:
                print('--------------')
                print(f'expref: {dset_obj._dset_raw.iloc[rec]["expref"]}')
                exp = TwoPRec(dset_obj=dset_obj, dset_ind=rec,
                              ch_img=ch,
                              trial_end=trial_end)
            except Exception as e:
                print('failed to load or analyze recording')
                print(e)
                pass

            for _ind_sp, _plot_special in enumerate(plot_conds_spec.plot_special):
                _y_gain = plot_conds_chs.plt_dff_y_gain[_ind_sp]
                exp.plt_stim_aligned_sectors(
                    n_sectors=10, img_ds_factor=50,
                    plt_dff={'x': {'gain': 0.6, 'offset': 0.2},
                             'y': {'gain': _y_gain, 'offset': 0.2}},
                    t_pre=t_pre, t_post=t_post,
                    plot_special=_plot_special,
                    plt_show=False)

    return


def plt_rew_value_from_dset(dset_obj, dset_ind_start=0,
                            ch_img=1, trial_end=60):
    n_recs = dset_obj._dset_raw['expref'].shape[0]
    for rec in range(dset_ind_start, n_recs):
        try:
            print('--------------')
            print(f'expref: {dset_obj._dset_raw.iloc[rec]["expref"]}')
            exp = TwoPRec(dset_obj=dset_obj, dset_ind=rec,
                          ch_img=ch_img,
                          trial_end=trial_end)
            exp.plt_stim_aligned_sectors(
                n_sectors=10, img_ds_factor=50,
                plt_dff={'x': {'gain': 0.6, 'offset': 0.2},
                         'y': {'gain': 10, 'offset': 0.2}},
                t_pre=2, t_post=5, plot_special=None,
                plt_show=False)
            exp.plt_psychometric_stimscaling(plt_show=False)
        except Exception as e:
            print('failed to load or analyze recording')
            print(e)
            pass

    return


def run_correction_method_comparison(dset_obj,
                                     n_trials=100,
                                     methods=None,
                                     n_sectors=10,
                                     t_pre=2, t_post=5,
                                     start_ind_dset=0):
    """
    Iterate through every recording in a DSetObj_5HTCtx, load it as a
    TwoPRec_DualColour (first n_trials trials only), plot sectors on the
    uncorrected signal, then for each correction method run
    correct_signal -> add_sectors -> plt_sectors, closing figures
    immediately after they are saved to disk.

    Parameters
    ----------
    dset_obj : DSetObj_5HTCtx
        Dataset object to iterate over.
    n_trials : int
        Number of trials to load (passed as trial_end). (default: 100)
    methods : list of str or None
        Correction methods to apply in order.
        If None, defaults to ['linear', 'robust', 'pca', 'nmf'].
    n_sectors : int
        Number of spatial sectors in each dimension. (default: 10)
    t_pre : float
        Seconds before stimulus for sector alignment. (default: 2)
    t_post : float
        Seconds after reward for sector alignment. (default: 5)
    start_ind_dset : int
        Skip all recordings with index below this value. (default: 0)
    """
    if methods is None:
        methods = ['linear', 'robust', 'lms']

    n_recs = dset_obj._dset_raw['expref'].shape[0]

    _plt_sectors_kwargs = dict(
        n_sectors=n_sectors,
        t_pre=t_pre,
        t_post=t_post,
        auto_gain_dff=True,
        minimal_output=True,
        plt_show=False)

    _add_sectors_kwargs = dict(
        n_sectors=n_sectors,
        t_pre=t_pre,
        t_post=t_post,
        compute_null=False,
        compute_resid_corr=False)

    def _flush_mem(label=''):
        """Close all figures, delete matplotlib state, collect garbage."""
        plt.close('all')
        plt.clf()
        gc.collect()
        gc.collect()   # second pass catches reference cycles
        if label:
            print(f'  [mem flush: {label}]')

    def _drop_exp_arrays(exp_obj):
        """Delete large in-RAM arrays attached to an exp object."""
        for attr in ('rec_grn_corr', 'sector', 'frame',
                     'rec_grn_original'):
            if hasattr(exp_obj, attr):
                try:
                    delattr(exp_obj, attr)
                except Exception:
                    pass

    for ind in range(start_ind_dset, n_recs):
        print(f'\n{"="*50}')
        print(f'recording {ind}/{n_recs - 1}: '
              f'{dset_obj._dset_raw["expref"][ind]}')
        print('='*50)

        try:
            exp = TwoPRec_DualColour(
                dset_obj=dset_obj,
                dset_ind=ind,
                trial_end=n_trials)
        except Exception as e:
            print(f'  failed to load recording: {e}')
            _flush_mem('load failure')
            continue

        # -- uncorrected baseline --
        print('\n  --- uncorrected ---')
        try:
            for use_zscore in [False]:
                for ch in ['grn', 'red']:
                    exp.add_sectors(**_add_sectors_kwargs,
                                    use_zscore=use_zscore,
                                    channel=ch)
                    exp.plt_sectors(**_plt_sectors_kwargs,
                                    use_zscore=use_zscore,
                                    plt_prefix='uncorrected',
                                    channel=ch)
                    plt.close('all')
        except Exception as e:
            print(f'  failed for uncorrected: {e}')
        finally:
            # Drop sector/frame data before entering the method loop
            _drop_exp_arrays(exp)
            _flush_mem('after uncorrected')

        # -- one pass per correction method --
        for method in methods:
            print(f'\n  --- method: {method} ---')
            try:
                exp.correct_signal(
                    method=method,
                    replace_grn=True,
                    save_to_disk=False)

                for use_zscore in [True, False]:
                    exp.add_sectors(**_add_sectors_kwargs,
                                    use_zscore=use_zscore)
                    exp.plt_sectors(**_plt_sectors_kwargs,
                                    use_zscore=use_zscore,
                                    plt_prefix=f'corr_{method}')
                    plt.close('all')

            except Exception as e:
                print(f'  failed for method={method}: {e}')
            finally:
                # Drop corrected signal and sector arrays after each method
                _drop_exp_arrays(exp)
                _flush_mem(f'after {method}')

        # Drop the full exp object before the next recording
        del exp
        _flush_mem(f'after recording {ind}')

    return
