import numpy as np

import sklearn
import sklearn.linear_model
import sklearn.preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gs
import seaborn as sns

from types import SimpleNamespace


class DecodingObj_2p(object):
    def __init__(self, sector):
        self.sector = sector
        return

    def classify(self,
                 t_start=0, t_end=0.4,
                 t_bl_start=-0.5, t_bl_end=0,
                 thresh_n_miss=6,
                 thresh_n_neurs=80,
                 fold_validation=3,
                 n_resamples=20,
                 hit_vs_miss=True,
                 comp='hit_miss',
                 neur_activ_fn=np.mean,
                 sec_interval=5,
                 neur_sampling_rule='random',
                 n_jobs=6,
                 classifier=sklearn.linear_model.SGDClassifier,
                 classifier_kwargs={'class_weight': 'balanced',
                                    'loss': 'hinge',
                                    'penalty': 'l2',
                                    'alpha': 1,
                                    'max_iter': 100000},
                 scaler=sklearn.preprocessing.StandardScaler,
                 scaler_kwargs={},
                 classifier_scoring='balanced_accuracy'):

        # Setup
        # -----------------

        # self.sector.dff[n_x, n_y][tr_cond][tr, time]




        # n_sess_all, n_sess_all_ind = np.unique(self.ephys.n_sess.opto.corr,
        #                                        return_index=True)
        # n_sess_filt = []
        # sess_firstind = []
        # n_neurs_in_sess = []
        # for ind_sess, sess in enumerate(n_sess_all):
        #     if self.ephys.n_tr.opto.incorr[
        #             n_sess_all_ind[ind_sess]] >= thresh_n_miss:
        #         # include
        #         _n_neurs_in_this_sess = np.where(
        #             self.ephys.n_sess.opto.incorr == sess)[0].shape[0]
        #         if _n_neurs_in_this_sess >= thresh_n_neurs:
        #             n_sess_filt.append(sess)
        #             sess_firstind.append(n_sess_all_ind[ind_sess])
        #             n_neurs_in_sess.append(_n_neurs_in_this_sess)
        # self.n_neurs_in_sess = n_neurs_in_sess

        # n_neurs_min = np.min(np.array(n_neurs_in_sess))
        # n_datasets = len(n_sess_filt)

        # n_neurs_decoder = np.append(np.arange(
        #     neur_interval, n_neurs_min, neur_interval), n_neurs_min)

        n_sectors_decoder = np.arange(0, sec_interval)

        if scaler is None:
            self.clf = make_pipeline(classifier(**classifier_kwargs))
        else:
            self.clf = make_pipeline(scaler(**scaler_kwargs),
                                     classifier(**classifier_kwargs))

        # self.decoded = SimpleNamespace(true=np.empty(n_datasets,
        #                                              dtype=np.ndarray),
        #                                pred=np.empty(n_datasets,
        #                                              dtype=np.ndarray))

        ind_t_start = np.argmin(np.abs(self.ephys.t-t_start))
        ind_t_end = np.argmin(np.abs(self.ephys.t-t_end))
        ind_t_bl_start = np.argmin(np.abs(self.ephys.t-t_bl_start))
        ind_t_bl_end = np.argmin(np.abs(self.ephys.t-t_bl_end))

        # self.cvobj = np.empty(len(n_sess_filt), dtype=np.ndarray)
        self.cvobj = {}
        self.cvobj_full = np.empty(
                len(n_sess_filt), dtype=np.ndarray)
        self.neur_reg_name = np.empty(
            len(n_sess_filt), dtype=np.ndarray)

        for n_neur_decoder in n_neurs_decoder:
            self.cvobj[str(n_neur_decoder)] = np.empty(
                len(n_sess_filt), dtype=np.ndarray)

        # Classify
        # ---------------
        self.perf_means = SimpleNamespace()
        self.perf_means.keys = n_neurs_decoder
        self.perf_means.vals = np.zeros_like(self.perf_means.keys)
        self.perf_means.vals_persess = np.empty(len(n_sess_filt),
                                                dtype=np.ndarray)

        if comp == 'hit_miss':
            n_tr_class1 = self.ephys.n_tr.opto.corr
            n_tr_class2 = self.ephys.n_tr.opto.incorr
            sess_class1 = self.ephys.n_sess.opto.corr
            sess_class2 = self.ephys.n_sess.opto.incorr
            class1 = self.ephys.tr.opto.corr
            class2 = self.ephys.tr.opto.incorr
        elif comp == 'hit_cr':
            n_tr_class1 = self.ephys.n_tr.opto.corr
            n_tr_class2 = self.ephys.n_tr.no_opto.corr
            sess_class1 = self.ephys.n_sess.opto.corr
            sess_class2 = self.ephys.n_sess.no_opto.corr
            class1 = self.ephys.tr.opto.corr
            class2 = self.ephys.tr.no_opto.corr
        elif comp == 'miss_cr':
            n_tr_class1 = self.ephys.n_tr.opto.incorr
            n_tr_class2 = self.ephys.n_tr.no_opto.corr
            sess_class1 = self.ephys.n_sess.opto.incorr
            sess_class2 = self.ephys.n_sess.no_opto.corr
            class1 = self.ephys.tr.opto.incorr
            class2 = self.ephys.tr.no_opto.corr

        for ind_sess, sess in enumerate(n_sess_filt):
            print(f'sess: {sess}')
            # parse trials/neurons in recording
            # -----------
            _n_tr_class1 = n_tr_class1[
                sess_firstind[ind_sess]]
            _n_tr_class2 = n_tr_class2[
                sess_firstind[ind_sess]]
            _neur_inds = np.where(sess_class1 == sess)[0]

            _n_neurs = _neur_inds.shape[0]
            _n_samples = int(_n_tr_class1 + _n_tr_class2)

            # store region names
            self.neur_reg_name[ind_sess] = np.array(
                self.ephys.region_name)[_neur_inds]

            # full decoder
            # ---------
            print('decoder on full dataset\n---------')
            self._data = np.zeros((_n_samples, _n_neurs))
            self._labels = np.zeros(_n_samples)

            _sample_count = 0

            for _tr in range(_n_tr_class1):
                for _ind_neur, neur in enumerate(_neur_inds):
                    _neur_resp = neur_activ_fn(
                        class1[neur][_tr, ind_t_start:ind_t_end]
                        - np.mean(class1[neur][_tr, ind_t_bl_start:ind_t_bl_end]))
                    self._data[_sample_count, _ind_neur] = _neur_resp
                self._labels[_sample_count] = 1
                _sample_count += 1
            for _tr in range(_n_tr_class2):
                for _ind_neur, neur in enumerate(_neur_inds):
                    _neur_resp = neur_activ_fn(
                        class2[neur][_tr, ind_t_start:ind_t_end]
                        - np.mean(class2[neur][_tr, ind_t_bl_start:ind_t_bl_end]))
                    self._data[_sample_count, _ind_neur] = _neur_resp
                self._labels[_sample_count] = 0
                _sample_count += 1

            self.cvobj_full[ind_sess] = cross_validate(
                self.clf, self._data, y=self._labels, cv=fold_validation,
                n_jobs=n_jobs, scoring=classifier_scoring,
                return_estimator=True)
            # self.cvobj_full[0]['estimator'][0]._final_estimator.coef_

            # decoder with changing number of neurons
            # ---------------------
            self.perf_means.vals_persess[ind_sess] \
                = np.zeros_like(self.perf_means.keys, dtype=float)

            print('decoder changing n_neurs\n---------')
            for ind_neur_decoder, n_neur_decoder in enumerate(n_neurs_decoder):
                print(f'\tn_neurs={n_neur_decoder}')
                # neuron sampling is random
                # --------------
                if neur_sampling_rule == 'random':
                    for n_resample in range(n_resamples):
                        # print(f'\t\t{n_resample=}')
                        self._data = np.zeros((_n_samples, n_neur_decoder))
                        self._labels = np.zeros(_n_samples)

                        _neur_inds_subset = np.random.choice(
                            _neur_inds, n_neur_decoder,
                            replace=False)

                        _sample_count = 0

                        for _tr in range(_n_tr_class1):
                            for _ind_neur, neur in enumerate(_neur_inds_subset):
                                _neur_resp = neur_activ_fn(
                                    class1[neur][_tr, ind_t_start:ind_t_end]
                                - np.mean(class1[neur][
                                    _tr, ind_t_bl_start:ind_t_bl_end]))
                                self._data[_sample_count, _ind_neur] = _neur_resp
                            self._labels[_sample_count] = 1
                            _sample_count += 1

                        for _tr in range(_n_tr_class2):
                            for _ind_neur, neur in enumerate(_neur_inds_subset):
                                _neur_resp = neur_activ_fn(
                                    class2[neur][_tr, ind_t_start:ind_t_end]
                                    - np.mean(class2[neur][
                                        _tr, ind_t_bl_start:ind_t_bl_end]))
                                self._data[_sample_count, _ind_neur] = _neur_resp
                            self._labels[_sample_count] = 0
                            _sample_count += 1

                        # Run classification
                        # ------------------
                        self.cvobj[str(n_neur_decoder)][ind_sess] = cross_validate(
                            self.clf, self._data, y=self._labels,
                            cv=fold_validation,
                            n_jobs=n_jobs, scoring=classifier_scoring,
                            return_estimator=True)
                        # store mean value across fold validation
                        _temp_perf = np.mean(self.cvobj[
                            str(n_neur_decoder)][ind_sess]['test_score'])
                        # print(_temp_perf)

                        # store mean value for this iteration
                        self.perf_means.vals_persess[ind_sess][ind_neur_decoder] \
                            += _temp_perf

                    # after all iterations, take mean
                    self.perf_means.vals_persess[ind_sess][ind_neur_decoder] \
                        /= n_resamples
        return

    def plt_weights(self, figsize=(2, 2),
                    plt_abs=False,
                    plt_cdf=False,
                    plt_range=(0, 200),
                    bins=50,
                    plt_lw=1.0,
                    sns_palette_name='rocket',
                    alpha=0.8,
                    return_weights=False):
        # extract weights
        # --------------
        n_sess = self.cvobj_full.shape[0]

        weights = np.empty(n_sess, dtype=np.ndarray)
        reg_names = np.empty(n_sess, dtype=np.ndarray)

        weights_all = np.empty(0)
        reg_names_all = np.empty(0)

        for sess in range(n_sess):
            n_neurs = self.cvobj_full[sess]['estimator'][
                0]._final_estimator.coef_.shape[1]
            n_folds = len(self.cvobj_full[sess]['estimator'])
            weights[sess] = np.zeros(n_neurs)
            reg_names[sess] = self.neur_reg_name[sess]
            for fold in range(n_folds):
                weights[sess] += self.cvobj_full[sess]['estimator'][
                    fold]._final_estimator.coef_[0, :]
            weights[sess] /= n_folds

            weights_all = np.append(weights_all, weights[sess])
            reg_names_all = np.append(reg_names_all, reg_names[sess])

        if plt_abs is True:
            weights_all = np.abs(weights_all)

        # setup figure
        # ----------------
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        reg_unique = np.unique(reg_names_all)
        print(f'{reg_unique=}')

        cmap = sns.color_palette(sns_palette_name,
                                 n_colors=reg_unique.shape[0])
        sns.palplot(cmap)

        for ind_reg, reg in enumerate(reg_unique):
            inds_in_names = np.where(reg_names_all == reg)
            ax.hist(weights_all[inds_in_names],
                    bins=bins, histtype='step',
                    range=plt_range, density=True,
                    color=cmap[ind_reg],
                    cumulative=plt_cdf,
                    lw=plt_lw)
        ax.set_xlabel('decoding weight')
        if plt_cdf is False:
            ax.set_ylabel('pdf')
        elif plt_cdf is True:
            ax.set_ylabel('cdf')
        fig.savefig('decoding_weights')
        plt.show()

        if return_weights is True:
            return weights_all


class DecodingObj_2p_dset(object):
    def __init__(self, dset_sector):
        self.sec = dset_sector
        return

    def classify(self,
                 t_start=0, t_end=0.4,
                 t_bl_start=-0.5, t_bl_end=0,
                 thresh_n_miss=6,
                 thresh_n_neurs=80,
                 fold_validation=3,
                 n_resamples=20,
                 hit_vs_miss=True,
                 comp='hit_miss',
                 neur_activ_fn=np.mean,
                 neur_interval=10,
                 neur_sampling_rule='random',
                 n_jobs=6,
                 classifier=sklearn.linear_model.SGDClassifier,
                 classifier_kwargs={'class_weight': 'balanced',
                                    'loss': 'hinge',
                                    'penalty': 'l2',
                                    'alpha': 1,
                                    'max_iter': 100000},
                 scaler=sklearn.preprocessing.StandardScaler,
                 scaler_kwargs={},
                 classifier_scoring='balanced_accuracy'):

        # Setup
        # -----------------

        # self.exp.sector.dff[n_x, n_y][tr_cond][tr, time]

        n_sess_all, n_sess_all_ind = np.unique(self.ephys.n_sess.opto.corr,
                                               return_index=True)
        n_sess_filt = []
        sess_firstind = []
        n_neurs_in_sess = []
        for ind_sess, sess in enumerate(n_sess_all):
            if self.ephys.n_tr.opto.incorr[
                    n_sess_all_ind[ind_sess]] >= thresh_n_miss:
                # include
                _n_neurs_in_this_sess = np.where(
                    self.ephys.n_sess.opto.incorr == sess)[0].shape[0]
                if _n_neurs_in_this_sess >= thresh_n_neurs:
                    n_sess_filt.append(sess)
                    sess_firstind.append(n_sess_all_ind[ind_sess])
                    n_neurs_in_sess.append(_n_neurs_in_this_sess)
        self.n_neurs_in_sess = n_neurs_in_sess

        n_neurs_min = np.min(np.array(n_neurs_in_sess))
        n_datasets = len(n_sess_filt)

        n_neurs_decoder = np.append(np.arange(
            neur_interval, n_neurs_min, neur_interval), n_neurs_min)

        if scaler is None:
            self.clf = make_pipeline(classifier(**classifier_kwargs))
        else:
            self.clf = make_pipeline(scaler(**scaler_kwargs),
                                     classifier(**classifier_kwargs))

        self.decoded = SimpleNamespace(true=np.empty(n_datasets,
                                                     dtype=np.ndarray),
                                       pred=np.empty(n_datasets,
                                                     dtype=np.ndarray))

        ind_t_start = np.argmin(np.abs(self.ephys.t-t_start))
        ind_t_end = np.argmin(np.abs(self.ephys.t-t_end))
        ind_t_bl_start = np.argmin(np.abs(self.ephys.t-t_bl_start))
        ind_t_bl_end = np.argmin(np.abs(self.ephys.t-t_bl_end))

        # self.cvobj = np.empty(len(n_sess_filt), dtype=np.ndarray)
        self.cvobj = {}
        self.cvobj_full = np.empty(
                len(n_sess_filt), dtype=np.ndarray)
        self.neur_reg_name = np.empty(
            len(n_sess_filt), dtype=np.ndarray)

        for n_neur_decoder in n_neurs_decoder:
            self.cvobj[str(n_neur_decoder)] = np.empty(
                len(n_sess_filt), dtype=np.ndarray)

        # Classify
        # ---------------
        self.perf_means = SimpleNamespace()
        self.perf_means.keys = n_neurs_decoder
        self.perf_means.vals = np.zeros_like(self.perf_means.keys)
        self.perf_means.vals_persess = np.empty(len(n_sess_filt),
                                                dtype=np.ndarray)

        if comp == 'hit_miss':
            n_tr_class1 = self.ephys.n_tr.opto.corr
            n_tr_class2 = self.ephys.n_tr.opto.incorr
            sess_class1 = self.ephys.n_sess.opto.corr
            sess_class2 = self.ephys.n_sess.opto.incorr
            class1 = self.ephys.tr.opto.corr
            class2 = self.ephys.tr.opto.incorr
        elif comp == 'hit_cr':
            n_tr_class1 = self.ephys.n_tr.opto.corr
            n_tr_class2 = self.ephys.n_tr.no_opto.corr
            sess_class1 = self.ephys.n_sess.opto.corr
            sess_class2 = self.ephys.n_sess.no_opto.corr
            class1 = self.ephys.tr.opto.corr
            class2 = self.ephys.tr.no_opto.corr
        elif comp == 'miss_cr':
            n_tr_class1 = self.ephys.n_tr.opto.incorr
            n_tr_class2 = self.ephys.n_tr.no_opto.corr
            sess_class1 = self.ephys.n_sess.opto.incorr
            sess_class2 = self.ephys.n_sess.no_opto.corr
            class1 = self.ephys.tr.opto.incorr
            class2 = self.ephys.tr.no_opto.corr

        for ind_sess, sess in enumerate(n_sess_filt):
            print(f'sess: {sess}')
            # parse trials/neurons in recording
            # -----------
            _n_tr_class1 = n_tr_class1[
                sess_firstind[ind_sess]]
            _n_tr_class2 = n_tr_class2[
                sess_firstind[ind_sess]]
            _neur_inds = np.where(sess_class1 == sess)[0]

            _n_neurs = _neur_inds.shape[0]
            _n_samples = int(_n_tr_class1 + _n_tr_class2)

            # store region names
            self.neur_reg_name[ind_sess] = np.array(
                self.ephys.region_name)[_neur_inds]

            # full decoder
            # ---------
            print('decoder on full dataset\n---------')
            self._data = np.zeros((_n_samples, _n_neurs))
            self._labels = np.zeros(_n_samples)

            _sample_count = 0

            for _tr in range(_n_tr_class1):
                for _ind_neur, neur in enumerate(_neur_inds):
                    _neur_resp = neur_activ_fn(
                        class1[neur][_tr, ind_t_start:ind_t_end]
                        - np.mean(class1[neur][_tr, ind_t_bl_start:ind_t_bl_end]))
                    self._data[_sample_count, _ind_neur] = _neur_resp
                self._labels[_sample_count] = 1
                _sample_count += 1
            for _tr in range(_n_tr_class2):
                for _ind_neur, neur in enumerate(_neur_inds):
                    _neur_resp = neur_activ_fn(
                        class2[neur][_tr, ind_t_start:ind_t_end]
                        - np.mean(class2[neur][_tr, ind_t_bl_start:ind_t_bl_end]))
                    self._data[_sample_count, _ind_neur] = _neur_resp
                self._labels[_sample_count] = 0
                _sample_count += 1

            self.cvobj_full[ind_sess] = cross_validate(
                self.clf, self._data, y=self._labels, cv=fold_validation,
                n_jobs=n_jobs, scoring=classifier_scoring,
                return_estimator=True)
            # self.cvobj_full[0]['estimator'][0]._final_estimator.coef_

            # decoder with changing number of neurons
            # ---------------------
            self.perf_means.vals_persess[ind_sess] \
                = np.zeros_like(self.perf_means.keys, dtype=float)

            print('decoder changing n_neurs\n---------')
            for ind_neur_decoder, n_neur_decoder in enumerate(n_neurs_decoder):
                print(f'\tn_neurs={n_neur_decoder}')
                # neuron sampling is random
                # --------------
                if neur_sampling_rule == 'random':
                    for n_resample in range(n_resamples):
                        # print(f'\t\t{n_resample=}')
                        self._data = np.zeros((_n_samples, n_neur_decoder))
                        self._labels = np.zeros(_n_samples)

                        _neur_inds_subset = np.random.choice(
                            _neur_inds, n_neur_decoder,
                            replace=False)

                        _sample_count = 0

                        for _tr in range(_n_tr_class1):
                            for _ind_neur, neur in enumerate(_neur_inds_subset):
                                _neur_resp = neur_activ_fn(
                                    class1[neur][_tr, ind_t_start:ind_t_end]
                                - np.mean(class1[neur][
                                    _tr, ind_t_bl_start:ind_t_bl_end]))
                                self._data[_sample_count, _ind_neur] = _neur_resp
                            self._labels[_sample_count] = 1
                            _sample_count += 1

                        for _tr in range(_n_tr_class2):
                            for _ind_neur, neur in enumerate(_neur_inds_subset):
                                _neur_resp = neur_activ_fn(
                                    class2[neur][_tr, ind_t_start:ind_t_end]
                                    - np.mean(class2[neur][
                                        _tr, ind_t_bl_start:ind_t_bl_end]))
                                self._data[_sample_count, _ind_neur] = _neur_resp
                            self._labels[_sample_count] = 0
                            _sample_count += 1

                        # Run classification
                        # ------------------
                        self.cvobj[str(n_neur_decoder)][ind_sess] = cross_validate(
                            self.clf, self._data, y=self._labels,
                            cv=fold_validation,
                            n_jobs=n_jobs, scoring=classifier_scoring,
                            return_estimator=True)
                        # store mean value across fold validation
                        _temp_perf = np.mean(self.cvobj[
                            str(n_neur_decoder)][ind_sess]['test_score'])
                        # print(_temp_perf)

                        # store mean value for this iteration
                        self.perf_means.vals_persess[ind_sess][ind_neur_decoder] \
                            += _temp_perf

                    # after all iterations, take mean
                    self.perf_means.vals_persess[ind_sess][ind_neur_decoder] \
                        /= n_resamples
        return

    def plt_weights(self, figsize=(2, 2),
                    plt_abs=False,
                    plt_cdf=False,
                    plt_range=(0, 200),
                    bins=50,
                    plt_lw=1.0,
                    sns_palette_name='rocket',
                    alpha=0.8,
                    return_weights=False):
        # extract weights
        # --------------
        n_sess = self.cvobj_full.shape[0]

        weights = np.empty(n_sess, dtype=np.ndarray)
        reg_names = np.empty(n_sess, dtype=np.ndarray)

        weights_all = np.empty(0)
        reg_names_all = np.empty(0)

        for sess in range(n_sess):
            n_neurs = self.cvobj_full[sess]['estimator'][
                0]._final_estimator.coef_.shape[1]
            n_folds = len(self.cvobj_full[sess]['estimator'])
            weights[sess] = np.zeros(n_neurs)
            reg_names[sess] = self.neur_reg_name[sess]
            for fold in range(n_folds):
                weights[sess] += self.cvobj_full[sess]['estimator'][
                    fold]._final_estimator.coef_[0, :]
            weights[sess] /= n_folds

            weights_all = np.append(weights_all, weights[sess])
            reg_names_all = np.append(reg_names_all, reg_names[sess])

        if plt_abs is True:
            weights_all = np.abs(weights_all)

        # setup figure
        # ----------------
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        reg_unique = np.unique(reg_names_all)
        print(f'{reg_unique=}')

        cmap = sns.color_palette(sns_palette_name,
                                 n_colors=reg_unique.shape[0])
        sns.palplot(cmap)

        for ind_reg, reg in enumerate(reg_unique):
            inds_in_names = np.where(reg_names_all == reg)
            ax.hist(weights_all[inds_in_names],
                    bins=bins, histtype='step',
                    range=plt_range, density=True,
                    color=cmap[ind_reg],
                    cumulative=plt_cdf,
                    lw=plt_lw)
        ax.set_xlabel('decoding weight')
        if plt_cdf is False:
            ax.set_ylabel('pdf')
        elif plt_cdf is True:
            ax.set_ylabel('cdf')
        fig.savefig('decoding_weights')
        plt.show()

        if return_weights is True:
            return weights_all
