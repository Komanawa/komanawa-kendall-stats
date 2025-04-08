"""
created matt_dumont 
on: 21/09/23
"""
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from matplotlib.lines import Line2D
from pathlib import Path
from copy import deepcopy
from scipy.stats import mstats
import itertools
from komanawa.kendall_stats.mann_kendall import _mann_kendall_from_sarray, _seasonal_mann_kendall_from_sarray, \
    _calc_seasonal_senslope, get_colors, _make_s_array


def _generate_startpoints(n, min_size, nparts, check_step=1, check_window=None, test=False):
    if check_window is None:
        if nparts == 2:
            all_start_points_out = np.arange(min_size, n - min_size, check_step)[:, np.newaxis]
            if test:
                assert ((all_start_points_out - 0 >= min_size) & (n - all_start_points_out >= min_size)).all()
        else:
            all_start_points = []
            for part in range(nparts - 1):
                start_points = np.arange(min_size + min_size * part, n - (min_size * (nparts - 1 - part)), check_step)
                all_start_points.append(start_points)

            all_start_points_out = np.array(list(itertools.product(*all_start_points)))
            all_start_points_out = all_start_points_out[np.all(
                [all_start_points_out[:, i] < all_start_points_out[:, i + 1] for i in range(nparts - 2)],
                axis=0)]
            temp = np.concatenate((
                np.array(all_start_points_out),
                np.full((len(all_start_points_out), 1), n)

            ), axis=1)
            sizes = np.diff(temp, axis=1)
            all_start_points_out = all_start_points_out[np.all(sizes >= min_size, axis=1)]
    else:
        check_window = np.atleast_2d(check_window)
        if nparts == 2:
            all_start_points_out = np.arange(check_window[0, 0], check_window[0, 1] + check_step, check_step)[:,
                                   np.newaxis]
        else:
            all_start_points = []
            for part in range(nparts - 1):
                start_points = np.arange(check_window[part, 0], check_window[part, 1] + check_step, check_step)
                all_start_points.append(start_points)
            all_start_points_out = np.array(list(itertools.product(*all_start_points)))
            temp = np.concatenate((
                np.array(all_start_points_out),
                np.full((len(all_start_points_out), 1), n)

            ), axis=1)
            sizes = np.diff(temp, axis=1)
            all_start_points_out = all_start_points_out[np.all(sizes >= min_size, axis=1)]

    if test:
        temp = np.concatenate((
            np.array(all_start_points_out),
            np.full((len(all_start_points_out), 1), n)

        ), axis=1)
        sizes = np.diff(temp, axis=1)
        assert np.all(sizes >= min_size)
    return all_start_points_out


class MultiPartKendall():

    def __init__(self, data, nparts=2, expect_part=(1, -1), min_size=10,
                 alpha=0.05, no_trend_alpha=0.5,
                 data_col=None, rm_na=True,
                 serialise_path=None,
                 check_step=1, check_window=None,
                 recalc=False, initalize=True):
        """
        multi part mann kendall test to indentify a change point(s) in a time series after Frollini et al., 2020, DOI: 10.1007/s11356-020-11998-0 note where the expected trend is zero the lack of a trend is considered significant if p > 1-alpha

        :param data: time series data, if DataFrame or Series, expects the index to be sample order (will sort on index) if np.array or list expects the data to be in sample order
        :param nparts: number of parts to split the time series into
        :param expect_part: expected trend in each part of the time series (1 increasing, -1 decreasing, 0 no trend)
        :param min_size: minimum size for the first and last section of the time series
        :param alpha: significance level
        :param no_trend_alpha: significance level for no trend e.g. will accept if p> no_trend_alpha
        :param data_col: if data is a DataFrame or Series, the column to use
        :param rm_na: remove na values from the data
        :param serialise_path: path to serialised file (as hdf), if None will not serialise
        :param check_step: int, the step to check for breakpoints, e.g. if 1 will check every point, if 2 will check every second point
        :param check_window: the window to check for breakpoints.  if None will use the whole data.  this is used to significantly speed up the mann kendall test. Note that check_step still applies to the check_window (e.g. a check_window of (2, 6) with a check_step of 2 will check the points (2, 4, 6)) One of:

            * None or tuple (start_idx, end_idx) (one breakpoint only)
            * list of tuples of len nparts-1 with a start/end idx for each part,
            * or a 2d array shape (nparts-1, 2) with a start/end idx for each part,

        :param recalc: if True will recalculate the mann kendall even if the serialised file exists
        :param initalize: if True will initalize the class from the data, only set to False used in self.from_file
        :return:
        """
        self.freq_limit = None
        self.trend_dict = {1: 'increasing', -1: 'decreasing', 0: 'no trend'}

        if not initalize:
            assert all([e is None for e in
                        [data, nparts, expect_part, min_size, alpha, no_trend_alpha, data_col, rm_na, serialise_path,
                         recalc]])
        else:
            loaded = False
            if serialise_path is not None:
                serialise_path = Path(serialise_path)
                self.serialise_path = serialise_path
                self.serialise = True
                if Path(serialise_path).exists() and not recalc:
                    loaded = True
                    self._set_from_file(
                        data=data,
                        nparts=nparts,
                        expect_part=expect_part,
                        min_size=min_size,
                        alpha=alpha,
                        no_trend_alpha=no_trend_alpha,
                        data_col=data_col,
                        rm_na=rm_na,
                        check_step=check_step,
                        check_window=check_window,
                        season_col=None)
            else:
                self.serialise = False
                self.serialise_path = None

            if not loaded:
                self._set_from_data(data=data,
                                    nparts=nparts,
                                    expect_part=expect_part,
                                    min_size=min_size,
                                    alpha=alpha,
                                    no_trend_alpha=no_trend_alpha,
                                    data_col=data_col,
                                    rm_na=rm_na,
                                    check_step=check_step,
                                    check_window=check_window,
                                    season_col=None)

            if self.serialise and not loaded:
                self.to_file()

    def __eq__(self, other):
        out = True
        out *= isinstance(other, self.__class__)
        out *= self.check_step == other.check_step
        out *= self.data_col == other.data_col
        out *= self.rm_na == other.rm_na
        out *= self.season_col == other.season_col
        out *= self.nparts == other.nparts
        out *= self.min_size == other.min_size
        out *= self.alpha == other.alpha
        out *= self.no_trend_alpha == other.no_trend_alpha
        out *= all(np.atleast_1d(self.expect_part) == np.atleast_1d(other.expect_part))
        datatype = type(self.data).__name__
        datatype_other = type(other.data).__name__
        out *= datatype == datatype_other

        if self.check_window is None:
            out *= other.check_window is None
        else:
            out *= np.allclose(self.check_window, other.check_window)

        if datatype == datatype_other:
            try:
                # check datasets
                if datatype == 'DataFrame':
                    pd.testing.assert_frame_equal(self.data, other.data, check_dtype=False, check_like=True,
                                                  )
                elif datatype == 'Series':
                    pd.testing.assert_series_equal(self.data, other.data, check_dtype=False, check_like=True,
                                                   )
                elif datatype == 'ndarray':
                    assert np.allclose(self.data, other.data)
                else:
                    raise AssertionError(f'unknown datatype {datatype}')
            except AssertionError:
                out *= False

        out *= np.allclose(self.x, other.x)
        out *= np.allclose(self.idx_values, other.idx_values)
        out *= np.all(self.acceptable_matches.values == other.acceptable_matches.values)
        if self.season_col is not None:
            out *= np.allclose(self.season_data, other.season_data)

        out *= np.allclose(self.s_array, other.s_array)
        out *= np.allclose(self.all_start_points, other.all_start_points)
        try:
            for part in range(self.nparts):
                pd.testing.assert_frame_equal(self.datasets[f'p{part}'], other.datasets[f'p{part}'])
        except AssertionError:
            out *= False
        return bool(out)

    def print_mk_diffs(self, other):
        """
        convenience function to print the differences between two MultiPartKendall classes
        :param other: another MultiPartKendall class
        """
        if not isinstance(other, self.__class__):
            print('problem with class: not same class: got ', type(other))
        if not self.check_step == other.check_step:
            print(f'problem with check step {self.check_step=} {other.check_step=}')
        if not self.data_col == other.data_col:
            print(f'problem with data col {self.data_col=} {other.data_col=}')
        if not self.rm_na == other.rm_na:
            print(f'problem with rm_na {self.rm_na=} other: {other.rm_na=}')
        if not self.season_col == other.season_col:
            print(f'problem with season col {self.season_col=} {other.season_col=}')
        if not self.nparts == other.nparts:
            print(f'problem with nparts: {self.nparts=} {other.nparts=}')
        if not self.min_size == other.min_size:
            print(f'problem with min_size {self.min_size=} {other.min_size=}')
        if not self.alpha == other.alpha:
            print(f'problem with alpha {self.alpha=} {other.alpha=}')
        if not self.no_trend_alpha == other.no_trend_alpha:
            print(f'problem with no_trend_alpha {self.no_trend_alpha=} {other.no_trend_alpha=}')
        if not all(np.atleast_1d(self.expect_part) == np.atleast_1d(other.expect_part)):
            print(f'problem with expect_part {self.expect_part=} {other.expect_part=}')
        datatype = type(self.data).__name__
        datatype_other = type(other.data).__name__
        if not datatype == datatype_other:
            print(f'problem with datatype {datatype=} {datatype_other=}')

        if self.check_window is None:
            if not other.check_window is None:
                print(f'problem with check_window, should both be None {self.check_window=} {other.check_window=}')
        else:
            if not np.allclose(self.check_window, other.check_window):
                print(f'problem with check_window {self.check_window=} {other.check_window=}')

        if datatype == datatype_other:
            try:
                # check datasets
                if datatype == 'DataFrame':
                    pd.testing.assert_frame_equal(self.data, other.data, check_dtype=False, check_like=True,
                                                  )
                elif datatype == 'Series':
                    pd.testing.assert_series_equal(self.data, other.data, check_dtype=False, check_like=True,
                                                   )
                elif datatype == 'ndarray':
                    assert np.allclose(self.data, other.data)
                else:
                    raise AssertionError(f'unknown datatype {datatype}')
            except AssertionError:
                print(f'problem with data')
                print(traceback.format_exc())

        if not np.allclose(self.x, other.x):
            print(f'problem with x')
        if not np.allclose(self.idx_values, other.idx_values):
            print(f'problem with idx_values')
        if not np.all(self.acceptable_matches.values == other.acceptable_matches.values):
            print(f'problem with acceptable_matches')
        if self.season_col is not None:
            if not np.allclose(self.season_data, other.season_data):
                print(f'problem with season_data')

        if not np.allclose(self.s_array, other.s_array):
            print('problem with s_array')
        if not np.allclose(self.all_start_points, other.all_start_points):
            print('problem with all_start_points')
        try:
            for part in range(self.nparts):
                pd.testing.assert_frame_equal(self.datasets[f'p{part}'], other.datasets[f'p{part}'])
        except AssertionError:
            print(f'problem with datasets')
            print(traceback.format_exc())

    def get_acceptable_matches(self):
        """
        get the acceptable matches for the multipart kendall test
        :return: pd.DataFrame
        """
        return self._get_matches(acceptable_only=True)


    def get_all_matches(self):
        """
        get the all matches for the multipart kendall test (including those that are not significant)
        :return: pd.DataFrame
        """
        return self._get_matches(acceptable_only=False)

    def _get_matches(self, acceptable_only):

        if acceptable_only:
            use_idx  = self.acceptable_matches
        else:
            use_idx = np.ones(self.acceptable_matches.shape, dtype=bool)

        outdata = self.datasets['p0'].loc[use_idx]
        outdata = outdata.set_index([f'split_point_{i}' for i in range(1, self.nparts)])
        outdata.rename(columns={f'{e}': f'{e}_p0' for e in ['trend', 'h', 'p', 'z', 's', 'var_s']}, inplace=True)
        for i in range(1, self.nparts):
            next_data = self.datasets[f'p{i}'].loc[use_idx]
            next_data = next_data.set_index([f'split_point_{j}' for j in range(1, self.nparts)])
            next_data.rename(columns={f'{e}': f'{e}_p{i}' for e in ['trend', 'h', 'p', 'z', 's', 'var_s']},
                             inplace=True)
            outdata = pd.merge(outdata, next_data, left_index=True, right_index=True)

        for p in range(self.nparts):
            temp = ((outdata[f'z_p{p}'].abs() - outdata[f'z_p{p}'].abs().min())
                    /
                    (outdata[f'z_p{p}'].abs().max() - outdata[f'z_p{p}'].abs().min()))
            if self.expect_part[p] == 0:
                outdata[f'znorm_p{p}'] = 1 - temp
            else:
                outdata[f'znorm_p{p}'] = temp

        outdata['znorm_joint'] = outdata[[f'znorm_p{p}' for p in range(self.nparts)]].sum(axis=1)
        return deepcopy(outdata)


    def get_maxz_breakpoints(self, raise_on_none=False):
        """
        get the breakpoints for the maximum joint normalised (min-max for each part) z the best match is the maximum znorm_joint value where:

           *  if expected trend == 1 or -1:
              *  znorm = the min-max normalised z value for each part
           *  else: (no trend expected)
              *  znorm = 1 - the min-max normalised z value for each part
           *  and
              *  znorm_joint = the sum of the znorm values for each part

        :param raise_on_none: bool, if True will raise an error if no acceptable matches, otherwise will return None
        :return: array of breakpoint tuples
        """
        acceptable = self.get_acceptable_matches()
        if acceptable.empty:
            if raise_on_none:
                raise ValueError('no acceptable matches')
            else:
                return None

        best = acceptable[acceptable['znorm_joint'] == acceptable['znorm_joint'].max()]

        return best.index.values

    def get_data_from_breakpoints(self, breakpoints):
        """
        get the data from the breakpoints

        :param breakpoints: beakpoints to split the data, e.g. from self.get_acceptable_matches

        :return: outdata: list of dataframes for each part of the time series
        :return: kendal_stats: dataframe of kendal stats for each part of the time series
        """
        breakpoints = np.atleast_1d(breakpoints)
        assert len(breakpoints) == self.nparts - 1
        outdata = []
        kendal_stats = pd.DataFrame(index=[f'p{i}' for i in range(self.nparts)],
                                    columns=['trend', 'h', 'p', 'z', 's', 'var_s', 'senslope',
                                             'senintercept'])
        for p, (pkey, ds) in enumerate(self.datasets.items()):
            assert pkey == f'p{p}'
            temp = ds.set_index([f'split_point_{i}' for i in range(1, self.nparts)])
            outcols = ['trend', 'h', 'p', 'z', 's', 'var_s']
            kendal_stats.loc[f'p{p}', outcols] = temp.loc[tuple(breakpoints), outcols].values

        start = 0
        for i in range(self.nparts):
            if i == self.nparts - 1:
                end = self.n
            else:
                end = breakpoints[i]
            if isinstance(self.data, pd.DataFrame):
                outdata.append(self.data.loc[self.idx_values[start:end]])
            else:
                outdata.append(deepcopy(
                    pd.Series(index=self.idx_values[start:end], data=self.data[self.idx_values[start:end]])))
            start = end

        # calculate the senslope stats
        for i, ds in enumerate(outdata):
            senslope, senintercept = self._calc_senslope(ds)
            kendal_stats.loc[f'p{i}', 'sen_slope'] = senslope
            kendal_stats.loc[f'p{i}', 'sen_intercept'] = senintercept

        return outdata, kendal_stats

    def plot_acceptable_matches(self, key):
        """
        quickly plot the acceptable matches

        :param key: key to plot (one of ['p', 'z', 's', 'var_s','znorm', znorm_joint]) or 'all' a figure for each value note joint stats only have 1 value
        :return:
        """
        poss_keys = ['p', 'z', 's', 'var_s', 'znorm', 'znorm_joint']
        assert key in poss_keys or key == 'all'
        if key == 'all':
            keys = poss_keys
        else:
            keys = np.atleast_1d(key)
        figs, axs = [], []
        for key in keys:
            fig, ax = plt.subplots(figsize=(10, 8))
            acceptable = self.get_acceptable_matches()
            if 'joint' in key:
                use_keys = [key]
            else:
                use_keys = [f'{key}_p{i}' for i in range(self.nparts)]
            acceptable = acceptable[use_keys]
            acceptable.plot(ax=ax, ls='none', marker='o')
            figs.append(fig)
            axs.append(ax)
        if len(figs) == 1:
            return fig, ax
        else:
            return figs, axs

    def plot_data_from_breakpoints(self, breakpoints, ax=None, txt_vloc=-0.05, add_labels=True, **kwargs):
        """
        plot the data from the breakpoints including the senslope fits

        :param breakpoints:
        :param ax: ax to plot on if None then create the ax
        :param txt_vloc: vertical location of the text (in ax.transAxes)
        :param add_labels: boolean, if True add labels (slope, pval) to the plot
        :param kwargs: passed to ax.scatter (all parts)
        :return: fig, ax
        """
        breakpoints = np.atleast_1d(breakpoints)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure

        data, kendal_stats = self.get_data_from_breakpoints(breakpoints)
        trans = blended_transform_factory(ax.transData, ax.transAxes)

        # axhlines at breakpoints
        prev_bp = 0
        for i, bp in enumerate(np.concatenate((breakpoints, [self.n]))):
            if not bp == self.n:
                ax.axvline(self.idx_values[bp], color='k', ls=':')
            sslope = kendal_stats.loc[f"p{i}", "sen_slope"]
            sintercept = kendal_stats.loc[f"p{i}", "sen_intercept"]
            x = self.idx_values[prev_bp:bp]
            if add_labels:
                xval = pd.Series(x).mean()
                ax.text(xval,
                        txt_vloc,
                        f'expected: {self.trend_dict[self.expect_part[i]]}\n'
                        f'got: slope: {sslope:.3e}, '
                        f'pval:{round(kendal_stats.loc[f"p{i}", "p"], 3)}',
                        transform=trans, ha='center', va='top')

            # plot the senslope fit and intercept
            y = (x).astype(float) * sslope + sintercept
            ax.plot(x, y, color='k', ls='--')
            prev_bp = bp

        if self.season_data is None:
            colors = get_colors(data)
            for i, (ds, c) in enumerate(zip(data, colors)):
                if isinstance(self.data, pd.DataFrame):
                    ax.scatter(ds.index, ds[self.data_col], c=c, label=f'part {i}', **kwargs)
                else:
                    ax.scatter(ds.index, ds, color=c, label=f'part {i}', **kwargs)
        else:
            seasons = np.unique(self.season_data)
            colors = get_colors(seasons)
            for i, ds in enumerate(data):
                for s, c in zip(seasons, colors):
                    temp = ds[ds[self.season_col] == s]
                    ax.scatter(temp.index, temp[self.data_col], color=c, label=f'season: {s}', **kwargs)

        legend_handles = [Line2D([0], [0], color='k', ls=':'),
                          Line2D([0], [0], color='k', ls='--')]

        legend_labels = ['breakpoint', 'sen slope fit', ]
        nhandles, nlabels = ax.get_legend_handles_labels()
        temp = dict(zip(nlabels, nhandles))
        legend_handles.extend(temp.values())
        legend_labels.extend(temp.keys())
        ax.legend(legend_handles, legend_labels, loc='best')
        return fig, ax

    def _set_from_file(self, data, nparts, expect_part, min_size, alpha, no_trend_alpha, data_col, rm_na,
                       check_step, check_window, season_col=None, check_inputs=True):
        """
        setup the class data from a serialised file, values are passed to ensure they are consistent

        :param check_inputs: bool, if True will check the inputs match the serialised file
        """
        assert self.serialise_path is not None, 'serialise path not set, should not get here'
        params = pd.read_hdf(self.serialise_path, 'params')
        assert isinstance(params, pd.Series)
        # other parameters
        self.alpha = params['alpha']
        if 'check_step' in params.index:
            self.check_step = int(params['check_step'])
        else:
            self.check_step = 1  # support legacy files
        self.no_trend_alpha = params['no_trend_alpha']
        self.nparts = int(params['nparts'])
        self.min_size = int(params['min_size'])
        self.rm_na = bool(params['rm_na'])
        self.n = int(params['n'])
        self.expect_part = [int(params[f'expect_part{i}']) for i in range(self.nparts)]
        if 'freq_limit' in params.index:
            self.freq_limit = params['freq_limit']
        else:
            self.freq_limit = None

        params_str = pd.read_hdf(self.serialise_path, 'params_str')
        assert isinstance(params_str, pd.Series)
        datatype = params_str['datatype']
        self.season_col = params_str['season_col']
        if self.season_col == 'None':
            self.season_col = None
        self.data_col = params_str['data_col']
        if self.data_col == 'None':
            self.data_col = None

        # d1 data
        d1_data = pd.read_hdf(self.serialise_path, 'd1_data')
        assert isinstance(d1_data, pd.DataFrame)
        self.x = d1_data['x'].values
        self.idx_values = d1_data['idx_values'].values
        self.acceptable_matches = pd.read_hdf(self.serialise_path, 'acceptable_matches')

        # check window
        if 'check_window' in params_str.index:
            temp = params_str['check_window']
            assert temp == 'None'
            self.check_window = None
            self.int_check_window = None
        else:
            with pd.HDFStore(self.serialise_path, 'r') as store:
                keys = [e.replace('/', '') for e in store.keys()]
            if not 'check_window' in keys:  # support legacy files
                self.check_window = None
                self.int_check_window = None
            else:
                temp = pd.read_hdf(self.serialise_path, 'check_window')
                assert isinstance(temp, pd.DataFrame)
                self.check_window = temp.values
                temp = pd.Series(index=self.idx_values, data=np.arange(len(self.idx_values)))
                self.int_check_window = temp[self.check_window.flatten()].values.reshape(self.check_window.shape)

        if datatype == 'pd.DataFrame':
            self.data = pd.read_hdf(self.serialise_path, 'data')
            assert isinstance(self.data, pd.DataFrame)
        elif datatype == 'pd.Series':
            self.data = pd.read_hdf(self.serialise_path, 'data')
            assert isinstance(self.data, pd.Series)
        elif datatype == 'np.array':
            self.data = pd.read_hdf(self.serialise_path, 'data').values
            assert isinstance(self.data, np.ndarray)
            assert self.data.ndim == 1
        else:
            raise ValueError('unknown datatype, thou shall not pass')

        if self.season_col is not None:
            self.season_data = self.data.loc[self.idx_values, self.season_col]
        else:
            self.season_data = None

        # s array
        self.s_array = pd.read_hdf(self.serialise_path, 's_array').values
        assert self.s_array.shape == (self.n, self.n)

        # all start points
        self.all_start_points = pd.read_hdf(self.serialise_path, 'all_start_points').values
        assert self.all_start_points.shape == (len(self.all_start_points), self.nparts - 1)

        # datasets
        dtypes = {'trend': 'float64', 'h': 'bool', 'p': 'float64',
                  'z': 'float64', 's': 'float64', 'var_s': 'float64'}
        for part in range(1, self.nparts):
            dtypes.update({f'split_point_{part}': 'int64'})
        self.datasets = {}
        for part in range(self.nparts):
            self.datasets[f'p{part}'] = pd.read_hdf(self.serialise_path, f'part{part}').astype(dtypes)

        if check_inputs:
            # check parameters have not changed
            assert self.data_col == data_col, 'data_col does not match'
            assert self.rm_na == rm_na, 'rm_na does not match'
            assert self.season_col == season_col, 'season_col does not match'
            assert self.nparts == nparts, 'nparts does not match'
            assert self.min_size == min_size, 'min_size does not match'
            assert self.alpha == alpha, 'alpha does not match'
            assert self.no_trend_alpha == no_trend_alpha, 'no_trend_alpha does not match'
            assert self.check_step == check_step, 'check_step does not match'
            assert all(np.atleast_1d(self.expect_part) == np.atleast_1d(expect_part)), 'expect_part does not match'

            if self.check_window is None:
                assert check_window is None, 'check_window does not match'
            else:
                check_window = np.atleast_2d(check_window)
                assert np.allclose(self.check_window, check_window), 'check_window does not match'

            # check datasets
            if datatype == 'pd.DataFrame':
                pd.testing.assert_frame_equal(self.data, data, check_dtype=False, check_like=True)
            elif datatype == 'pd.Series':
                pd.testing.assert_series_equal(self.data, data, check_dtype=False, check_like=True)
            elif datatype == 'np.array':
                assert np.allclose(self.data, data)

    def _set_from_data(self, data, nparts, expect_part, min_size, alpha, no_trend_alpha, data_col, rm_na,
                       check_step, check_window, season_col=None):
        """
        set up the class data from the input data

        :param data:
        :param nparts:
        :param expect_part:
        :param min_size:
        :param alpha:
        :param data_col:
        :param rm_na:
        :param season_col:
        :return:
        """
        self.data = deepcopy(data)
        self.alpha = alpha
        self.no_trend_alpha = no_trend_alpha
        self.nparts = nparts
        self.min_size = min_size
        self.expect_part = expect_part
        self.data_col = data_col
        self.rm_na = rm_na

        assert len(expect_part) == nparts

        # handle data (including options for season)
        self.season_col = season_col
        if season_col is not None:
            assert isinstance(data, pd.DataFrame) or isinstance(data, dict), ('season_col passed but data is not a '
                                                                              'DataFrame or dictionary')
            assert season_col in data.keys(), 'season_col not in data'
            assert data_col is not None, 'data_col must be passed if season_col is passed'
            assert data_col in data.keys(), 'data_col not in data'
            if rm_na:
                data = data.dropna(subset=[data_col, season_col])
            data = data.sort_index()
            self.season_data = data[season_col]
            self.idx_values = data.index.values
            x = np.array(data[data_col])
            self.x = x
        else:
            self.season_data = None
            if data_col is not None:
                x = pd.Series(data[data_col])
            else:
                x = pd.Series(data)
            if rm_na:
                x = x.dropna(how='any')
            x = x.sort_index()
            self.idx_values = x.index.values
            x = np.array(x)
            self.x = x
        assert x.ndim == 1, 'data must be 1d or multi d but with col_name passed'

        n = len(x)
        self.n = n
        if n / self.nparts < min_size:
            raise ValueError('the time series is too short for the minimum size')
        self.s_array = _make_s_array(x)

        assert isinstance(check_step, int), f'{check_step=} must be an int'
        assert check_step > 0, f'{check_step=} must be > 0'
        self.check_step = check_step

        if check_window is not None:
            check_window = np.atleast_2d(check_window)
            assert check_window.shape == (nparts - 1, 2), f'{check_window=} must have shape (nparts-1, 2)'
            self.check_window = check_window

            if isinstance(self.data, pd.DataFrame) or isinstance(self.data, pd.Series):
                assert np.in1d(check_window.flatten(), self.data.index).all(), (
                    'check_window contains values not in data index')

                if data_col:
                    check_data = self.data[self.data_col]
                else:
                    check_data = self.data

                assert not check_data.loc[check_window.flatten()].isna().any(), (
                    'check_window references nan values')
                if self.season_col is not None:
                    assert not self.data.loc[check_window.flatten(), self.season_col].isna().any(), (
                        'check_window references nan values')

            elif isinstance(self.data, np.ndarray):
                assert set(check_window.flatten()).issubset(np.arange(len(self.data)))
                assert not np.isnan(self.data[check_window.flatten()]).any(), 'check_window references nan values'

            temp = pd.Series(index=self.idx_values, data=np.arange(len(self.idx_values)))
            self.int_check_window = temp[self.check_window.flatten()].values.reshape(self.check_window.shape)
            assert (self.int_check_window.flatten() >= self.min_size).all(), 'check_window contains values < min_size'
            assert (self.n - self.int_check_window.flatten() >= self.min_size).all(), (
                'n - check_window contains values <min_size')
        else:
            self.check_window = None
            self.int_check_window = None

        all_start_points = _generate_startpoints(n, self.min_size, self.nparts, self.check_step, self.int_check_window)
        datasets = {f'p{i}': [] for i in range(nparts)}
        self.all_start_points = all_start_points
        self.datasets = datasets

        self._calc_mann_kendall()

        # find all acceptable matches
        idx = np.ones(len(self.all_start_points), bool)
        for part, expect in enumerate(self.expect_part):
            if expect == 0:
                idx = (idx
                       & (self.datasets[f'p{part}'].trend == expect)
                       & (self.datasets[f'p{part}'].p > self.no_trend_alpha)
                       )
            else:
                idx = (idx
                       & (self.datasets[f'p{part}'].trend == expect)
                       & (self.datasets[f'p{part}'].p < self.alpha)
                       )
        self.acceptable_matches = idx

    def _calc_senslope(self, data):

        if isinstance(self.data, pd.DataFrame):
            senslope, senintercept, lo_slope, up_slope = mstats.theilslopes(data[self.data_col], data.index,
                                                                            alpha=self.alpha)
        else:
            senslope, senintercept, lo_slope, up_slope = mstats.theilslopes(data, data.index, alpha=self.alpha)
        return senslope, senintercept

    def _calc_mann_kendall(self):
        """
        acutually calculate the mann kendall from the sarray, this should be the only thing that needs to be updated for the seasonal kendall

        :return:
        """
        temp_data = {}
        for sp in np.atleast_2d(self.all_start_points):
            start = 0
            for i in range(self.nparts):
                if i == self.nparts - 1:
                    end = self.n
                else:
                    end = sp[i]
                temp_key = (start, end)
                if temp_key in temp_data:
                    datai = temp_data[temp_key]
                else:
                    datai = _mann_kendall_from_sarray(self.x[start:end], alpha=self.alpha,
                                                      sarray=self.s_array[start:end, start:end])
                    temp_data[temp_key] = datai
                data = (*sp, *datai)
                self.datasets[f'p{i}'].append(data)
                start = end
        for part in range(self.nparts):
            self.datasets[f'p{part}'] = pd.DataFrame(self.datasets[f'p{part}'],
                                                     columns=[f'split_point_{i}' for i in range(1, self.nparts)]
                                                             + ['trend', 'h', 'p', 'z', 's', 'var_s'])

    def to_file(self, save_path=None, complevel=9, complib='blosc:lz4'):
        """
        save the data to a hdf file

        :param save_path: None (save to self.serialise_path) or path to save the file
        :param complevel: compression level for hdf
        :param complib: compression library for hdf
        :return:
        """
        if save_path is None:
            assert self.serialise_path is not None, 'serialise path not set, should not get here'
            save_path = self.serialise_path
        with pd.HDFStore(save_path, 'w') as hdf:
            # setup single value parameters
            params = pd.Series()
            params_str = pd.Series()

            # should be 1d+ of same length
            d1_data = pd.DataFrame(index=range(len(self.x)))
            d1_data['x'] = self.x
            d1_data['idx_values'] = self.idx_values
            d1_data.to_hdf(hdf, 'd1_data')

            self.acceptable_matches.to_hdf(hdf, 'acceptable_matches', complevel=complevel, complib=complib)
            # save as own datasets
            if isinstance(self.data, pd.DataFrame):
                self.data.to_hdf(hdf, 'data', complevel=complevel, complib=complib)
                params_str['datatype'] = 'pd.DataFrame'
            elif isinstance(self.data, pd.Series):
                self.data.to_hdf(hdf, 'data', complevel=complevel, complib=complib)
                params_str['datatype'] = 'pd.Series'
            else:
                params_str['datatype'] = 'np.array'
                pd.Series(self.data).to_hdf(hdf, 'data', complevel=complevel, complib=complib)

            assert isinstance(self.s_array, np.ndarray)
            pd.DataFrame(self.s_array).to_hdf(hdf, 's_array', complevel=complevel, complib=complib)
            assert isinstance(self.all_start_points, np.ndarray)
            pd.DataFrame(self.all_start_points).to_hdf(hdf, 'all_start_points', complevel=complevel, complib=complib)

            for part in range(self.nparts):
                self.datasets[f'p{part}'].astype(float).to_hdf(hdf, f'part{part}', complevel=complevel, complib=complib)

            # check_window
            if self.check_window is not None:
                pd.DataFrame(self.check_window).to_hdf(hdf, 'check_window', complevel=complevel, complib=complib)
            else:
                params_str['check_window'] = 'None'

            # other parameters
            params['alpha'] = self.alpha
            params['no_trend_alpha'] = self.no_trend_alpha
            params['nparts'] = float(self.nparts)
            params['min_size'] = float(self.min_size)
            params['rm_na'] = float(self.rm_na)
            params['n'] = float(self.n)
            params['check_step'] = float(self.check_step)
            if self.freq_limit is not None:
                params['freq_limit'] = float(self.freq_limit)
            for i in range(self.nparts):
                params[f'expect_part{i}'] = float(self.expect_part[i])

            params_str['season_col'] = str(self.season_col)
            params_str['data_col'] = str(self.data_col)
            params.to_hdf(hdf, 'params', complevel=complevel, complib=complib)
            params_str.to_hdf(hdf, 'params_str', complevel=complevel, complib=complib)

    @classmethod
    @staticmethod
    def from_file(path):
        """
        load the class from a serialised file

        :param path: path to the serialised file
        :return: MultiPartKendall
        """
        mpk = MultiPartKendall(
            data=None, nparts=None, expect_part=None, min_size=None, alpha=None, no_trend_alpha=None, data_col=None,
            serialise_path=None, recalc=None, rm_na=None, initalize=False)
        mpk.serialise_path = Path(path)
        mpk.serialise = True
        mpk._set_from_file(data=None, nparts=None, expect_part=None, min_size=None, alpha=None, no_trend_alpha=None,
                           data_col=None, rm_na=None,
                           check_step=None, check_window=None,
                           season_col=None, check_inputs=False)
        return mpk


class SeasonalMultiPartKendall(MultiPartKendall):

    def __init__(self, data, data_col, season_col, nparts=2, expect_part=(1, -1), min_size=10,
                 alpha=0.05, no_trend_alpha=0.5,
                 rm_na=True,
                 serialise_path=None, freq_limit=0.05,
                 check_step=1, check_window=None,
                 recalc=False, initalize=True):
        """
        multi part seasonal mann kendall test to indentify a change point(s) in a time series after Frollini et al., 2020, DOI: 10.1007/s11356-020-11998-0

        :param data: time series data, if DataFrame or Series, expects the index to be sample order (will sort on index)if np.array or list expects the data to be in sample order
        :param data_col: if data is a DataFrame or Series, the column to use
        :param season_col: the column to use for the season
        :param nparts: number of parts to split the time series into
        :param expect_part: expected trend in each part of the time series (1 increasing, -1 decreasing, 0 no trend)
        :param min_size: minimum size for the first and last section of the time series
        :param alpha: significance level
        :param no_trend_alpha: significance level for no trend e.g. will accept if p> no_trend_alpha
        :param rm_na: remove na values from the data
        :param serialise_path: path to serialised file (as hdf), if None will not serialise
        :param check_step: int, the step to check for breakpoints, e.g. if 1 will check every point, if 2 will check every second point
        :param check_window: the window to check for breakpoints.  if None will use the whole data.  this is used to significantly speed up the mann kendall test Note that check_step still applies to the check_window (e.g. a check_window of (2, 6) with a check_step of 2 will check the points (2, 4, 6))  one of:

               * None or tuple (start_idx, end_idx) (one breakpoint only)
               * or list of tuples of len nparts-1 with a start/end idx for each part,
               * or a 2d array shape (nparts-1, 2) with a start/end idx for each part

        :param recalc: if True will recalculate the mann kendall even if the serialised file exists
        :param initalize: if True will initalize the class from the data, only set to False used in self.from_file
        :return:
        """
        self.trend_dict = {1: 'increasing', -1: 'decreasing', 0: 'no trend'}
        self.freq_limit = freq_limit

        if not initalize:
            assert all([e is None for e in
                        [data, nparts, expect_part, min_size, alpha, no_trend_alpha, data_col, rm_na, serialise_path,
                         recalc]])
        else:
            loaded = False
            if serialise_path is not None:
                serialise_path = Path(serialise_path)
                self.serialise_path = serialise_path
                self.serialise = True
                if Path(serialise_path).exists() and not recalc:
                    loaded = True
                    self._set_from_file(
                        data=data,
                        nparts=nparts,
                        expect_part=expect_part,
                        min_size=min_size,
                        alpha=alpha,
                        no_trend_alpha=no_trend_alpha,
                        data_col=data_col,
                        rm_na=rm_na,
                        check_step=check_step,
                        check_window=check_window,
                        season_col=season_col)
            else:
                self.serialise = False
                self.serialise_path = None

            if not loaded:
                self._set_from_data(data=data,
                                    nparts=nparts,
                                    expect_part=expect_part,
                                    min_size=min_size,
                                    alpha=alpha,
                                    no_trend_alpha=no_trend_alpha,
                                    data_col=data_col,
                                    rm_na=rm_na,
                                    check_step=check_step,
                                    check_window=check_window,
                                    season_col=season_col)

            if self.serialise and not loaded:
                self.to_file()

    @classmethod
    @staticmethod
    def from_file(path):
        """
        load the class from a serialised file

        :param path:
        :return:
        """
        mpk = SeasonalMultiPartKendall(data=None, data_col=None, season_col=None, nparts=None, expect_part=None,
                                       min_size=None, alpha=None, no_trend_alpha=None, rm_na=None,
                                       serialise_path=None, recalc=None, initalize=False)

        mpk.serialise_path = Path(path)
        mpk.serialise = True
        mpk._set_from_file(data=None, nparts=None, expect_part=None, min_size=None,
                           alpha=None, no_trend_alpha=None, data_col=None,
                           rm_na=None, check_step=None,
                           check_window=None,
                           season_col=None, check_inputs=False)
        return mpk

    def _calc_mann_kendall(self):
        """
        actually calculate the mann kendall from the sarray, this should be the only thing that needs
        to be updated for the seasonal kendall
        :return:
        """
        temp_data = {}
        for sp in self.all_start_points:
            start = 0
            for i in range(self.nparts):
                if i == self.nparts - 1:
                    end = self.n
                else:
                    end = sp[i]
                temp_key = (start, end)
                if temp_key in temp_data:
                    datai = temp_data[temp_key]
                else:
                    datai = _seasonal_mann_kendall_from_sarray(self.x[start:end], alpha=self.alpha,
                                                               season_data=self.season_data.values[start:end],
                                                               sarray=self.s_array[start:end,
                                                                      start:end],
                                                               freq_limit=self.freq_limit)  # and passing the s array
                    temp_data[temp_key] = datai
                data = (*sp, *datai)
                self.datasets[f'p{i}'].append(data)
                start = end
        for part in range(self.nparts):
            self.datasets[f'p{part}'] = pd.DataFrame(self.datasets[f'p{part}'],
                                                     columns=[f'split_point_{i}' for i in range(1, self.nparts)]
                                                             + ['trend', 'h', 'p', 'z', 's', 'var_s'])

    def _calc_senslope(self, data):
        senslope, senintercept, lo_slope, lo_intercept = _calc_seasonal_senslope(data[self.data_col],
                                                                                 data[self.season_col],
                                                                                 x=data.index, alpha=self.alpha)
        return senslope, senintercept
